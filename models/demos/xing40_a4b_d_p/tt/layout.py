# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Xing4.0 activation layout on an SP x TP mesh (plan.md): SP over rows (axis 0) x TP over columns (axis 1), both
taken from mesh.shape (4x2 on the LoudBox: S/4 rows and 1792 hidden columns per chip).

Chip (r, c) holds chunk rows [r S/SP, (r+1) S/SP) and hidden columns [w c, w (c+1)), w = H/TP, of every one of the 4
mHC streams, packed stream-major along the last dim: the per-chip residual is [1, 1, S/SP, 4 x w] fp32; local column
j*w + k = global flat column j*H + c*w + k (the reference's token-major [S * 4, H] viewed as [S, 4H]).

The host helpers here are the harness boundary only (component / swap / hybrid); never call them in a forward.
"""

from __future__ import annotations

import torch

import ttnn

HC = 4  # hc_mult: mHC residual streams


def streams_cols_to_chip_major(t: torch.Tensor, hidden: int, tp: int) -> torch.Tensor:
    """[..., HC * hidden] in stream order (j, h) -> [..., tp * HC * hidden/tp] in (c, j, k) order, so a plain split
    of the last dim over the tp mesh columns hands chip column c its [HC x hidden/tp] block."""
    assert hidden % tp == 0, (hidden, tp)
    lead = t.shape[:-1]
    return t.reshape(*lead, HC, tp, hidden // tp).transpose(-3, -2).reshape(*lead, HC * hidden)


def chip_major_to_streams_cols(t: torch.Tensor, hidden: int, tp: int) -> torch.Tensor:
    """Inverse of streams_cols_to_chip_major."""
    assert hidden % tp == 0, (hidden, tp)
    lead = t.shape[:-1]
    return t.reshape(*lead, tp, HC, hidden // tp).transpose(-3, -2).reshape(*lead, HC * hidden)


def _mapper(mesh, dims):
    return ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims)


def _composer(mesh):
    return ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=tuple(mesh.shape), dims=(2, 3))


def streams_to_device(mesh, x: torch.Tensor, hidden: int, dtype=ttnn.float32) -> ttnn.Tensor:
    """Host streams [S * 4, H] (token-major) or [S, 4H] -> device [1, 1, S/SP, 4 x H/TP] per chip, TILE, DRAM."""
    host = x.float().reshape(-1, HC * hidden)
    s = host.shape[0]
    assert s % mesh.shape[0] == 0, f"{s} rows do not split over {mesh.shape[0]} mesh rows"
    host = streams_cols_to_chip_major(host, hidden, mesh.shape[1]).reshape(1, 1, s, HC * hidden)
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(mesh, (2, 3)),
    )


def streams_to_host(mesh, t: ttnn.Tensor, hidden: int) -> torch.Tensor:
    """Device [1, 1, S/SP, 4 x H/TP] per chip -> host [S * 4, H] (token-major, the reference's block boundary)."""
    full = ttnn.to_torch(t, mesh_composer=_composer(mesh))
    s = full.shape[-2]
    return chip_major_to_streams_cols(full.reshape(s, HC * hidden), hidden, mesh.shape[1]).reshape(s * HC, hidden)


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
