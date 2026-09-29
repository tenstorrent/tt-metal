# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hy4 activation layout on the 2x2 mesh (plan.md): SP=2 over rows (axis 0) x TP=2 over columns (axis 1).

Chip (r, c) holds chunk rows [r*S/2, (r+1)*S/2) and hidden columns [c*3072, (c+1)*3072) of every one of the 4 iHC
streams, packed stream-major along the last dim: the per-chip residual is [1, 1, S/2, 4 x 3072] fp32, local column
j*3072 + k = global flat column j*6144 + c*3072 + k (HF ``flatten(2)`` order is j*6144 + h).

The host helpers here are the harness boundary only (component / swap / hybrid); never call them in a forward.
"""

from __future__ import annotations

import torch

import ttnn

HC = 4  # hc_mult: iHC residual streams
TP = 2  # mesh columns (axis 1): hidden split
SP = 2  # mesh rows (axis 0): sequence split


def streams_cols_to_chip_major(t: torch.Tensor, hidden: int) -> torch.Tensor:
    """[..., HC * hidden] in HF stream order (j, h) -> [..., TP * HC * hidden/TP] in (c, j, k) order, so a plain split
    of the last dim over mesh columns hands chip column c its [HC x hidden/TP] block."""
    lead = t.shape[:-1]
    return t.reshape(*lead, HC, TP, hidden // TP).transpose(-3, -2).reshape(*lead, HC * hidden)


def chip_major_to_streams_cols(t: torch.Tensor, hidden: int) -> torch.Tensor:
    """Inverse of streams_cols_to_chip_major."""
    lead = t.shape[:-1]
    return t.reshape(*lead, TP, HC, hidden // TP).transpose(-3, -2).reshape(*lead, HC * hidden)


def streams_to_device(mesh, x: torch.Tensor, hidden: int, dtype=ttnn.float32) -> ttnn.Tensor:
    """Host streams [S, 4H] (HF flatten(2) order) -> device [1, 1, S/2, 4 x H/2] per chip, TILE, DRAM."""
    s = x.shape[0]
    host = streams_cols_to_chip_major(x.float().reshape(s, HC * hidden), hidden).reshape(1, 1, s, HC * hidden)
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, 3)),
    )


def streams_to_host(mesh, t: ttnn.Tensor, hidden: int) -> torch.Tensor:
    """Device [1, 1, S/2, 4 x H/2] per chip -> host [S, 4H] (HF order)."""
    full = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=tuple(mesh.shape), dims=(2, 3)))
    s = full.shape[-2]
    return chip_major_to_streams_cols(full.reshape(s, HC * hidden), hidden)


def row_split_to_host(mesh, t: ttnn.Tensor, width: int | None = None) -> torch.Tensor:
    """A tensor split by rows over axis 0 and replicated over axis 1 ([1, 1, S/2, W] per chip) -> host [S, W]
    (column 0's copy). ``width`` trims the last dim."""
    full = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=tuple(mesh.shape), dims=(2, 3)))
    w = t.shape[-1]
    out = full[..., :w].reshape(-1, w)
    return out if width is None else out[:, :width]
