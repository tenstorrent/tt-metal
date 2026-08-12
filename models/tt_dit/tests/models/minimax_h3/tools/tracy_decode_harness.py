# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tracy capture entry points for the MiniMax-H3 VAE decoders. **Profiling harness, not a gate.**

Nothing here asserts: these exist to put exactly one warmed forward of each decoder inside a
signposted Tracy window for manual profiling. The filename does not start with ``test_``, so pytest's
discovery leaves this file alone; run it by passing the path explicitly (commands below). Separate
from ``test_performance_vae_minimax_h3.py`` so the perf gates stay assert-bearing while these stay
runnable.

A profile also wants the opposite of what the ``_wave``/``_baseline`` timing measurements want.
``_time_it`` runs a warmup plus timed
iterations; three invocations of the 36-layer decoder emit well past Tracy's ~1000-op-per-device
buffer, and the symptom is ``AssertionError: Device data missing: Op <id>``, which reads as a tool bug
rather than as overflow. So these run **one** forward inside the signposted window. Device kernel
durations are warm-independent, so a single warmed forward is the right window; what must stay outside
it is weight upload and activation prep, whose ``TilizeWithValPadding`` / ``Untilize`` run would
otherwise dominate the aggregate and make data movement look like the bottleneck.

    timeout 1800 ./python_env/bin/python -m tracy -p -r -v -m pytest \\
      models/tt_dit/tests/models/minimax_h3/tools/tracy_decode_harness.py -k tracy_visual_decode \\
      -s --timeout 900 &> tracy_vae.log
    tt-perf-report --start-signpost start --end-signpost stop <csv>

Unset TTNN_CONFIG_PATH and do not combine with TT_METAL_WATCHER: profiler, watcher and DPRINT all
consume device SRAM, and all-zero device durations means one of the others is still set.
"""

import os

import pytest
import torch

import ttnn

from models.tt_dit.models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from models.tt_dit.tests.models.minimax_h3.common import (
    DECODE_LATENT_FRAMES,
    LATENT_TILE,
    build_audio_decoder,
    build_visual_decoder,
    load_config,
    random_decoder_state,
    weights_subdir,
)

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

HOP_LENGTH = 800


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_tracy_visual_decode_unit(mesh_device):
    """One video VAE decoder invocation at the shipping work unit, signposted for Tracy.

    The unit is one temporal chunk of one spatial tile -- `DECODE_LATENT_FRAMES` x `LATENT_TILE`^2 --
    which is what `decode_unit_shape` returns and what the pipeline launches 196 of (7 waves x 28 units)
    at the production working point. Profiling the unit and multiplying by the count is both cheaper and
    more honest than profiling the whole stage.
    """
    from loguru import logger
    from tracy import signpost

    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = load_config(weights_dir)
    torch.manual_seed(1)

    decoder = build_visual_decoder(config, mesh_device)
    decoder.load_torch_state_dict(random_decoder_state(config))
    num_patches = DECODE_LATENT_FRAMES * LATENT_TILE * LATENT_TILE
    tokens = ttnn.from_torch(
        torch.randn(1, num_patches, config["latent_channels"]),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
    )

    # Warm the program cache OUTSIDE the window: a cold forward measures compilation.
    _ = decoder(tokens)
    ttnn.synchronize_device(mesh_device)

    logger.info(
        f"tracy: video VAE decoder unit, {DECODE_LATENT_FRAMES}x{LATENT_TILE}x{LATENT_TILE} latents "
        f"-> {num_patches} patches, {config['decoder_num_layers']} layers"
    )
    signpost("start")
    _ = decoder(tokens)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    ttnn.ReadDeviceProfiler(mesh_device)


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_tracy_audio_decode(mesh_device):
    """One audio decode at the shipping duration, signposted for Tracy.

    Not reducible to a smaller unit the way the video decode is: the audio decoder is one pass over the
    whole 207-latent sequence, which measures at ~1680 ops. That is past Tracy's per-device
    buffer, so `--op-support-count` is required:

        timeout 1800 ./python_env/bin/python -m tracy -p -r -v --op-support-count 4000 -m pytest \\
          models/tt_dit/tests/models/minimax_h3/tools/tracy_decode_harness.py -k tracy_audio_decode \\
          -s --timeout 900 &> tracy_audio.log

    The reading caveat that matters here: Tracy **undercounts** this stage's device time by ~6x
    against wall clock -- 224 ms device against 1284 ms wall is the recorded example. Treat the
    per-op ranking as the product, not the absolute total.
    """
    from loguru import logger
    from tracy import signpost

    weights_dir = weights_subdir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = load_config(weights_dir)
    torch.manual_seed(2)

    decoder = build_audio_decoder(config, mesh_device)
    from safetensors.torch import load_file

    # `strict=False` matches the audio baselines: the converted dict carries the encoder's tensors too.
    decoder.load_torch_state_dict(
        convert_minimax_h3_audio_state_dict(
            load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
        ),
        strict=False,
    )
    num_latent_frames = 207
    latents = torch.randn(2, config["latent_channels"], num_latent_frames)

    _ = decoder(latents)
    ttnn.synchronize_device(mesh_device)

    logger.info(
        f"tracy: audio decoder, {num_latent_frames} latents x 2 channels -> {num_latent_frames * HOP_LENGTH} samples"
    )
    signpost("start")
    _ = decoder(latents)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    ttnn.ReadDeviceProfiler(mesh_device)
