# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy capture of ONE warmed audio decode on a 4x8 Galaxy, T-sharded at factor 8 (the pipeline's default).

Profiling harness, not a gate. The eager decode is ~1700-2000 ops per chip, so `--op-support-count` is required;
one forward inside the signposted window (more overflows Tracy's per-device buffer and reads as
``AssertionError: Device data missing``).

    timeout 2400 ./python_env/bin/python -m tracy -p -r -v --op-support-count 6000 -m pytest \\
      models/tt_dit/tests/models/minimax_h3/tools/tracy_audio_decode_t8_harness.py -k 600lat_b1 \\
      -s --timeout 1800 &> tracy_audio_t8.log

The ops CSV lands under ``generated/profiler/reports/<timestamp>/ops_perf_results_*.csv``; summarize it with
``summarize_audio_ops.py`` (per-op-type device time, op count, wall vs device-busy per chip).

Unset TTNN_CONFIG_PATH and do not combine with TT_METAL_WATCHER: all-zero device durations means another
device-SRAM consumer is still set. The eager decode is dispatch-bound, so the device-time total is far below
the wall clock by design; the per-op durations and ranking are what this harness is for.
"""

import os

import pytest
import torch

import ttnn
from models.tt_dit.models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from models.tt_dit.parallel.config import ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tests.models.minimax_h3.common import build_audio_decoder, load_config, weights_subdir
from models.tt_dit.utils.test import line_params_8k

MESH_4X8 = [
    pytest.param(
        (4, 8),
        {**line_params_8k, "require_exact_physical_num_devices": True, "l1_small_size": 65536},
        id="mesh4x8_8k",
    )
]
HOP_LENGTH = 800


def load_t_sharded_decoder(mesh_device, *, factor: int, axis: int):
    weights_dir = weights_subdir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_MODEL_PATH")
    from safetensors.torch import load_file

    config = load_config(weights_dir)
    pc = None if factor <= 1 else ParallelFactor(factor=factor, mesh_axis=axis)
    ccl = None if pc is None else CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    decoder = build_audio_decoder(config, mesh_device, parallel_config=pc, ccl_manager=ccl)
    decoder.load_torch_state_dict(
        convert_minimax_h3_audio_state_dict(
            load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
        ),
        strict=False,
    )
    return decoder, config


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize(
    ("num_latent_frames", "batch"),
    [(207, 1), (207, 2), (600, 1), (600, 2)],
    ids=["207lat_b1", "207lat_b2", "600lat_b1", "600lat_b2"],  # single tokens: `python -m tracy` re-splits -k args
)
def test_tracy_audio_decode_t8(mesh_device, num_latent_frames, batch):
    """One eager audio decode at T-shard factor 8 (axis 1), signposted for Tracy."""
    from loguru import logger
    from tracy import signpost

    decoder, config = load_t_sharded_decoder(mesh_device, factor=8, axis=1)
    torch.manual_seed(2)
    latents = torch.randn(batch, config["latent_channels"], num_latent_frames)
    _ = decoder(latents)  # warm the program cache outside the window
    ttnn.synchronize_device(mesh_device)
    logger.info(
        f"tracy: audio decoder 4x8 t_factor=8, {num_latent_frames} latents x batch {batch} "
        f"-> {num_latent_frames * HOP_LENGTH} samples"
    )
    signpost("start")
    _ = decoder(latents)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    ttnn.ReadDeviceProfiler(mesh_device)
