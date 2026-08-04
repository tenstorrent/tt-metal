# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy target: one encoder down_block, selected by level.

The whole encoder is ~550 ops per iteration and two iterations overrun Tracy's 1000-op
per-device buffer, and separately fails to generate a report at all (STATE.md amendment 53).
One block is ~90-140 ops, so two iterations fit comfortably and the report generates.

Profiling every level in turn and summing gives the per-op device-time budget for the whole
encoder, which is what the whole-encoder profile was meant to produce.

    MINIMAX_H3_LEVEL=0 python -m tracy -p -r -m pytest <this file>
    MINIMAX_H3_STATS_GN=0 MINIMAX_H3_LEVEL=0 python -m tracy -p -r -m pytest <this file>
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
if "MINIMAX_H3_FLAT_TILE" in os.environ:
    enc_mod.MINIMAX_H3_FLAT_TILE_RESIDUAL = os.environ["MINIMAX_H3_FLAT_TILE"] == "1"

SINGLE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

# (in_ch, out_ch, T, H, W, temporal_stride, spatial_stride) entering each level, derived the
# same way MiniMaxH3Encoder3d derives them: block_in = (out[0],) + out[:-1], and the shape
# shrinks at the *end* of a level.
LEVELS = [
    (128, 128, 17, 256, 256, 1, 2),
    (128, 256, 17, 128, 128, 2, 2),
    (256, 256, 9, 64, 64, 2, 2),
    (256, 512, 5, 32, 32, 1, 2),
    (512, 512, 5, 16, 16, 1, 1),
    (512, 1024, 5, 16, 16, 1, 1),
]


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE, indirect=["mesh_device", "device_params"])
def test_profile_downblock(mesh_device):
    level = int(os.environ.get("MINIMAX_H3_LEVEL", "0"))
    in_ch, out_ch, t, h, w, t_stride, s_stride = LEVELS[level]
    torch.manual_seed(0)
    block = enc_mod.MiniMaxH3DownBlock3d(
        in_ch,
        out_ch,
        num_layers=2,
        num_frames=t,
        height=h,
        width=w,
        temporal_downsample_factor=t_stride,
        spatial_downsample_factor=s_stride,
        temporal_taps=3,
        mesh_device=mesh_device,
    )
    state = {}
    for i in range(2):
        c_in = in_ch if i == 0 else out_ch
        state["resnets.%d.norm1.weight" % i] = torch.ones(c_in)
        state["resnets.%d.norm1.bias" % i] = torch.zeros(c_in)
        state["resnets.%d.norm2.weight" % i] = torch.ones(out_ch)
        state["resnets.%d.norm2.bias" % i] = torch.zeros(out_ch)
        state["resnets.%d.conv1.weight" % i] = torch.randn(out_ch, c_in, 3, 3, 3) * 0.02
        state["resnets.%d.conv1.bias" % i] = torch.zeros(out_ch)
        state["resnets.%d.conv2.weight" % i] = torch.randn(out_ch, out_ch, 3, 3, 3) * 0.02
        state["resnets.%d.conv2.bias" % i] = torch.zeros(out_ch)
        if c_in != out_ch:
            state["resnets.%d.conv_shortcut.weight" % i] = torch.randn(out_ch, c_in, 1, 1, 1) * 0.02
            state["resnets.%d.conv_shortcut.bias" % i] = torch.zeros(out_ch)
    if t_stride * s_stride > 1:
        state["downsamplers.0.conv.weight"] = torch.randn(out_ch, out_ch, 3, 3, 3) * 0.02
        state["downsamplers.0.conv.bias"] = torch.zeros(out_ch)
    block.load_torch_state_dict(state)

    x = ttnn.from_torch(
        torch.randn(1, t, h, w, in_ch), dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    for _ in range(2):
        block(x)
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
