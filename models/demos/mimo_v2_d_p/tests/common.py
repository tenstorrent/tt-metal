# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for mimo_v2_d_p device tests."""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions


def bc_index(kv_actual, sp, C):
    """Global positions of a chunk in device order (SP row major, then local row)."""
    pos = rotated_chip_positions(kv_actual, sp, C)
    return torch.tensor([pos[c][r] for c in range(sp) for r in range(C)], dtype=torch.long)


def to_mesh_seq(x, mesh_device, idx):
    """host [1, S, H] chunk (natural order) rows ``idx`` -> device [1,1,S_local,H] sharded over SP rows."""
    return ttnn.from_torch(
        x[:, idx][None], device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, None)),
    )


def from_mesh_seq(t, mesh_device):
    """device [1,1,S_local,H] (replicated over TP) -> host [S, H] in device order (TP col 0 of each row)."""
    sp, tp = tuple(mesh_device.shape)
    dts = ttnn.get_device_tensors(t)
    return torch.cat([ttnn.to_torch(dts[s * tp]).float()[0, 0] for s in range(sp)], 0)
