# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy target: one encoder down_block, to settle MINIMAX_H3_USE_STATS_GROUPNORM.

The whole encoder is ~550 ops per iteration and two iterations overrun Tracy's 1000-op
per-device buffer, which is what made the stats-norm profile fail to generate a report
(STATE.md amendment 50). One block is ~90 ops, so two iterations fit comfortably.

Block 0 at (17, 256, 256): the largest spatial extent, four norm sites, and the level where
the tilize/untilize round trip costs most.

    MINIMAX_H3_STATS_GN=0 python -m tracy -p -r -m pytest <this file>
    MINIMAX_H3_STATS_GN=1 python -m tracy -p -r -m pytest <this file>
"""
from __future__ import annotations

import os

import pytest
import torch

import ttnn

from ....models.vae.minimax_h3 import encoder_minimax_h3 as enc_mod

# Read the flag from the environment so one file profiles both configurations.
if "MINIMAX_H3_STATS_GN" in os.environ:
    enc_mod.MINIMAX_H3_USE_STATS_GROUPNORM = os.environ["MINIMAX_H3_STATS_GN"] == "1"

SINGLE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE, indirect=["mesh_device", "device_params"])
def test_profile_downblock(mesh_device):
    torch.manual_seed(0)
    block = enc_mod.MiniMaxH3DownBlock3d(
        128,
        128,
        num_layers=2,
        num_frames=17,
        height=256,
        width=256,
        temporal_downsample_factor=1,
        spatial_downsample_factor=2,
        temporal_taps=3,
        mesh_device=mesh_device,
    )
    state = {}
    for i in range(2):
        for n in ("norm1", "norm2"):
            state[f"resnets.{i}.{n}.weight"] = torch.ones(128)
            state[f"resnets.{i}.{n}.bias"] = torch.zeros(128)
        for n in ("conv1", "conv2"):
            state[f"resnets.{i}.{n}.weight"] = torch.randn(128, 128, 3, 3, 3) * 0.02
            state[f"resnets.{i}.{n}.bias"] = torch.zeros(128)
    state["downsamplers.0.conv.weight"] = torch.randn(128, 128, 3, 3, 3) * 0.02
    state["downsamplers.0.conv.bias"] = torch.zeros(128)
    block.load_torch_state_dict(state)

    x = ttnn.from_torch(
        torch.randn(1, 17, 256, 256, 128), dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    for _ in range(2):
        block(x)
        ttnn.synchronize_device(mesh_device)
