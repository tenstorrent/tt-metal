# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tracy capture entry points for the MiniMax-H3 VAE decoders -- profiling harness, not a gate.
Each runs ONE warmed forward inside the signposted window: more overflows Tracy's ~1000-op-per-device
buffer, and the symptom (``AssertionError: Device data missing: Op <id>``) reads as a tool bug.

    timeout 1800 ./python_env/bin/python -m tracy -p -r -v -m pytest \\
      models/tt_dit/tests/models/minimax_h3/tools/tracy_decode_harness.py -k tracy_visual_decode \\
      -s --timeout 900 &> tracy_vae.log
    tt-perf-report --start-signpost start --end-signpost stop <csv>

Unset TTNN_CONFIG_PATH and do not combine with TT_METAL_WATCHER: all-zero device durations means
another device-SRAM consumer is still set.
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
    """One video VAE decoder invocation at the shipping work unit, signposted for Tracy."""
    from loguru import logger
    from tracy import signpost

    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_MODEL_PATH")
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
    """One audio decode at the shipping duration (~1680 ops, so `--op-support-count` is required):

        timeout 1800 ./python_env/bin/python -m tracy -p -r -v --op-support-count 4000 -m pytest \\
          models/tt_dit/tests/models/minimax_h3/tools/tracy_decode_harness.py -k tracy_audio_decode \\
          -s --timeout 900 &> tracy_audio.log

    Tracy undercounts this stage's device time ~6x vs wall clock; trust the per-op ranking, not the total.
    """
    from loguru import logger
    from tracy import signpost

    weights_dir = weights_subdir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_MODEL_PATH")
    config = load_config(weights_dir)
    torch.manual_seed(2)

    decoder = build_audio_decoder(config, mesh_device)
    from safetensors.torch import load_file

    # strict=False: the converted dict carries the encoder's tensors too.
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
