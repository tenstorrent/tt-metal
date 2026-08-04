# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy profiling target: the whole MiniMax-H3 encoder on one work unit.

The encoder is ~80 % of VAE wall time (15.7 s of 19.4 s at 768P/5s) and has never been
profiled per-op. The blocking sweep improved it 1.70x while only ever seeing aggregate
timings; the ViT's profile then turned up a 36 %-of-runtime data-movement problem that no
amount of blocking tuning would have found, so the same treatment is owed here.

The whole encoder rather than one level: a unit is ~100 ops, well inside Tracy's 1000-op
per-device buffer, and the levels have very different shapes so a single level would not
generalise.

Run:

    python -m tracy -p -r -m pytest \\
        models/tt_dit/tests/models/minimax_h3/profile_encoder_minimax_h3.py

then read the newest generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv. Two
iterations: the first populates the program cache, so only the second is meaningful.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from ....models.vae.minimax_h3.encoder_minimax_h3 import MiniMaxH3Encoder3d
from .test_performance_vae_minimax_h3 import CLIP_FRAMES, TILE, _config, _random_encoder_state, _weights_dir

SINGLE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
ITERATIONS = 2


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE, indirect=["mesh_device", "device_params"])
def test_profile_encoder(mesh_device):
    weights_dir = _weights_dir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = _config(weights_dir)
    torch.manual_seed(0)

    encoder = MiniMaxH3Encoder3d(
        num_frames=CLIP_FRAMES,
        height=TILE,
        width=TILE,
        in_channels=3,
        out_channels=2 * config["latent_channels"],
        block_out_channels=tuple(config["block_out_channels"]),
        layers_per_block=config["layers_per_block"],
        spatial_downsample_factors=tuple(config["spatial_downsample_factors"]),
        temporal_downsample_factors=tuple(config["temporal_downsample_factors"]),
        temporal_taps=3,
        mesh_device=mesh_device,
    )
    encoder.load_torch_state_dict(_random_encoder_state(config))

    x = torch.randn(1, CLIP_FRAMES, TILE, TILE, encoder.conv_in.in_channels)
    x_device = ttnn.from_torch(x, dtype=ttnn.float32, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)

    for _ in range(ITERATIONS):
        encoder(x_device)
        ttnn.synchronize_device(mesh_device)
