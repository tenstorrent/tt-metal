# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy profiling target: one MiniMax-H3 ViT decoder layer at the shipping shape.

Whole-decoder wall-clock is useless for attribution -- the same code measures 0.34-0.99 s
per 32-unit wave (STATE.md amendment 41), because at ~540 ops per invocation it is
host-dispatch bound. The device profiler sidesteps that entirely: it reports per-op *device*
time, which host jitter does not touch.

One layer, not 36, for two reasons: Tracy drops ops past 1000 per device, and a single layer
is the repeating unit -- whatever dominates here dominates 36x over.

Run:

    python -m tracy -p -r -v -m pytest \\
        models/tt_dit/tests/models/minimax_h3/profile_vit_layer_minimax_h3.py -v

then read the newest generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv.

Two iterations: the first populates the program cache, so only the second is meaningful.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from ....models.vae.minimax_h3.decoder_minimax_h3 import MiniMaxH3ViTDecoder3d
from .test_performance_vae_minimax_h3 import (
    DECODE_LATENT_FRAMES,
    LATENT_TILE,
    _config,
    _random_decoder_state,
    _weights_dir,
)

# Single device: per-op device timing is per-device anyway, and the decoder is
# data-parallel with no CCL, so one device sees exactly the shipping per-unit work.
SINGLE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

PROFILE_LAYERS = 1
ITERATIONS = 2


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE, indirect=["mesh_device", "device_params"])
def test_profile_vit_layer(mesh_device):
    weights_dir = _weights_dir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = _config(weights_dir)
    torch.manual_seed(0)

    decoder = MiniMaxH3ViTDecoder3d(
        num_frames=DECODE_LATENT_FRAMES,
        height=LATENT_TILE,
        width=LATENT_TILE,
        in_channels=config["latent_channels"],
        out_channels=config["out_channels"],
        num_layers=PROFILE_LAYERS,
        mesh_device=mesh_device,
    )
    decoder.load_torch_state_dict(_random_decoder_state(config, num_layers=PROFILE_LAYERS))

    tokens = torch.randn(1, DECODE_LATENT_FRAMES * LATENT_TILE * LATENT_TILE, config["latent_channels"])
    tokens_device = ttnn.from_torch(tokens, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)

    for _ in range(ITERATIONS):
        decoder(tokens_device)
        ttnn.synchronize_device(mesh_device)
