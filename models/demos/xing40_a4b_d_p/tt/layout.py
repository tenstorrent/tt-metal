# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Xing4.0 activation layout on the 4x2 mesh (plan.md): SP=4 over rows (axis 0) x TP=2 over columns (axis 1).

Chip (r, c) holds chunk rows [r S/4, (r+1) S/4) and hidden columns [1792 c, 1792 (c+1)) of every one of the 4 mHC
streams, packed stream-major along the last dim: the per-chip residual is [1, 1, S/4, 4 x 1792] fp32; local column
j*1792 + k = global flat column j*3584 + c*1792 + k (the reference's token-major [S * 4, H] viewed as [S, 4H]).

The host helpers here are the harness boundary only (component / swap / hybrid); never call them in a forward.
"""

from __future__ import annotations

import torch

import ttnn

HC = 4  # hc_mult: mHC residual streams
TP = 2  # mesh columns (axis 1): hidden split
SP = 4  # mesh rows (axis 0): sequence split


def streams_cols_to_chip_major(t: torch.Tensor, hidden: int) -> torch.Tensor:
    """[..., HC * hidden] in stream order (j, h) -> [..., TP * HC * hidden/TP] in (c, j, k) order, so a plain split
    of the last dim over mesh columns hands chip column c its [HC x hidden/TP] block."""
    lead = t.shape[:-1]
    return t.reshape(*lead, HC, TP, hidden // TP).transpose(-3, -2).reshape(*lead, HC * hidden)


def chip_major_to_streams_cols(t: torch.Tensor, hidden: int) -> torch.Tensor:
    """Inverse of streams_cols_to_chip_major."""
    lead = t.shape[:-1]
    return t.reshape(*lead, TP, HC, hidden // TP).transpose(-3, -2).reshape(*lead, HC * hidden)


def _mapper(mesh, dims):
    return ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims)


def _composer(mesh):
    return ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=tuple(mesh.shape), dims=(2, 3))


def streams_to_device(mesh, x: torch.Tensor, hidden: int, dtype=ttnn.float32) -> ttnn.Tensor:
    """Host streams [S * 4, H] (token-major) or [S, 4H] -> device [1, 1, S/4, 4 x H/2] per chip, TILE, DRAM."""
    host = x.float().reshape(-1, HC * hidden)
    s = host.shape[0]
    assert s % mesh.shape[0] == 0, f"{s} rows do not split over {mesh.shape[0]} mesh rows"
    host = streams_cols_to_chip_major(host, hidden).reshape(1, 1, s, HC * hidden)
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(mesh, (2, 3)),
    )


def streams_to_host(mesh, t: ttnn.Tensor, hidden: int) -> torch.Tensor:
    """Device [1, 1, S/4, 4 x H/2] per chip -> host [S * 4, H] (token-major, the reference's block boundary)."""
    full = ttnn.to_torch(t, mesh_composer=_composer(mesh))
    s = full.shape[-2]
    return chip_major_to_streams_cols(full.reshape(s, HC * hidden), hidden).reshape(s * HC, hidden)


def row_split_to_host(mesh, t: ttnn.Tensor, width: int | None = None) -> torch.Tensor:
    """A tensor split by rows over axis 0 and replicated over axis 1 ([1, 1, S/4, W] per chip) -> host [S, W]
    (column 0's copy). ``width`` trims the last dim."""
    full = ttnn.to_torch(t, mesh_composer=_composer(mesh))
    w = t.shape[-1]
    out = full[..., :w].reshape(-1, w)
    return out if width is None else out[:, :width]


def row_split_to_device(mesh, x: torch.Tensor, dtype=ttnn.float32) -> ttnn.Tensor:
    """Host [S, W] -> device [1, 1, S/4, W] per chip, split by rows over axis 0, replicated over axis 1, TILE, DRAM."""
    s, w = x.shape
    return ttnn.from_torch(
        x.float().reshape(1, 1, s, w),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(mesh, (2, None)),
    )


def col_split_to_host(mesh, t: ttnn.Tensor) -> torch.Tensor:
    """A tensor split by rows over axis 0 and by hidden columns over axis 1 ([1, 1, S/4, H/2] per chip) -> host
    [S, H]."""
    full = ttnn.to_torch(t, mesh_composer=_composer(mesh))
    return full.reshape(-1, full.shape[-1])


def col_split_to_device(mesh, x: torch.Tensor, dtype=ttnn.float32) -> ttnn.Tensor:
    """Host [S, H] -> device [1, 1, S/4, H/2] per chip, split by rows over axis 0 and by hidden columns over axis 1,
    TILE, DRAM (the layout of TtHcCollapse's output)."""
    s, w = x.shape
    return ttnn.from_torch(
        x.float().reshape(1, 1, s, w),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(mesh, (2, 3)),
    )


def server_order(start: int, chunk: int, sp: int) -> torch.Tensor:
    """The order in which the prefill server lays a chunk's tokens on the SP rows (tt-d-gen ring_sdpa_reshuffle by
    kv_offset = actual_start): out[k] = index into the natural-order chunk of the k-th token of the device-major
    [sp, chunk / sp] payload. Absolute position g goes to row (g // (chunk / sp)) % sp, rising within a row; for a
    start that is a multiple of the chunk this is the identity. Harness boundary only (hooks embed)."""
    w = chunk // sp
    rows = [[] for _ in range(sp)]
    for i in range(chunk):
        rows[((start + i) // w) % sp].append(i)
    assert all(len(r) == w for r in rows), (start, chunk, sp)
    return torch.tensor([i for r in rows for i in r], dtype=torch.long)
