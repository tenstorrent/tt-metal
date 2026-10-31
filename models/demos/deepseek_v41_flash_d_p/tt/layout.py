# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 activation layout on an SP x TP mesh (plan.md "Scheme"): SP over mesh rows (axis 0) x TP over mesh
columns (axis 1), both read from mesh.shape (2x2 here, 4x4 / 8x4 / 2x4 on the other boxes).

Residual: chip (r, c) holds chunk rows [r S/SP, (r+1) S/SP) and hidden columns [w c, w (c+1)), w = H/TP, of every one
of the 4 mHC streams, packed stream-major along the last dim: [1, 1, S/SP, 4 x w] (bf16 under BF16_ACTIVATIONS=on,
P.20; else fp32; xing40's layout); local
column j*w + k = global flat column j*H + c*w + k of the reference's [S, 4H] streams. The pending pre [S, 4] (the last
4 columns of the reference's packed block boundary, DS41-3) is row-split and replicated over TP.

The host helpers here are the harness boundary only (component / swap / hybrid); never call them in a forward.
"""

from __future__ import annotations

import torch

import ttnn

from .cores import from_torch

HC = 4  # hc_mult: mHC residual streams


def streams_cols_to_chip_major(t: torch.Tensor, hidden: int, tp: int, n: int = HC) -> torch.Tensor:
    """[..., n * hidden] in stream order (j, h) -> [..., tp * n * hidden/tp] in (c, j, k) order, so a plain split
    of the last dim over the tp mesh columns hands chip column c its [n x hidden/tp] block (n = HC streams, or the
    engram projection's 5 H-wide blocks)."""
    assert hidden % tp == 0, (hidden, tp)
    lead = t.shape[:-1]
    return t.reshape(*lead, n, tp, hidden // tp).transpose(-3, -2).reshape(*lead, n * hidden)


def chip_major_to_streams_cols(t: torch.Tensor, hidden: int, tp: int, n: int = HC) -> torch.Tensor:
    """Inverse of streams_cols_to_chip_major."""
    assert hidden % tp == 0, (hidden, tp)
    lead = t.shape[:-1]
    return t.reshape(*lead, tp, n, hidden // tp).transpose(-3, -2).reshape(*lead, n * hidden)


def _mapper(mesh, dims):
    return ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims)


def _composer(mesh):
    return ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=tuple(mesh.shape), dims=(2, 3))


def streams_to_device(mesh, x: torch.Tensor, hidden: int, dtype=None) -> ttnn.Tensor:
    """Host streams [S, 4H] (or the packed boundary [S, 4H + 4]: the pending pre is dropped) -> device
    [1, 1, S/SP, 4 x H/TP] per chip, TILE, DRAM; ``dtype`` None = the model's stream dtype (BF16_ACTIVATIONS, P.20:
    bf16, the GPU code's storage of the streams)."""
    if dtype is None:
        from .common import stream_dtype

        dtype = stream_dtype()
    host = x.float()[:, : HC * hidden]
    s = host.shape[0]
    assert s % mesh.shape[0] == 0, f"{s} rows do not split over {mesh.shape[0]} mesh rows"
    host = streams_cols_to_chip_major(host, hidden, mesh.shape[1]).reshape(1, 1, s, HC * hidden)
    return from_torch(
        host.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(mesh, (2, 3)),
    )


def streams_to_host(mesh, t: ttnn.Tensor, hidden: int) -> torch.Tensor:
    """Device [1, 1, S/SP, 4 x H/TP] per chip -> host [S, 4H] (the reference's stream columns)."""
    full = ttnn.to_torch(t, mesh_composer=_composer(mesh))
    s = full.shape[-2]
    return chip_major_to_streams_cols(full.reshape(s, HC * hidden), hidden, mesh.shape[1])


def blocks_to_host(mesh, t: ttnn.Tensor, hidden: int, n: int) -> torch.Tensor:
    """Device [1, 1, S/SP, n x H/TP] per chip (chip column c holds its hidden slice of each of n H-wide blocks) ->
    host [S, n H] in block order (the engram projection's [S, 5H] boundary)."""
    full = ttnn.to_torch(t, mesh_composer=_composer(mesh))
    s = full.shape[-2]
    return chip_major_to_streams_cols(full.reshape(s, n * hidden), hidden, mesh.shape[1], n)


def blocks_to_device(mesh, x: torch.Tensor, hidden: int, n: int, dtype=ttnn.float32) -> ttnn.Tensor:
    """Host [S, n H] in block order -> device [1, 1, S/SP, n x H/TP] per chip (chip column c holds its hidden slice of
    each of the n H-wide blocks; the inverse of blocks_to_host, e.g. the engram projection's [S, 5H])."""
    s = x.shape[0]
    assert x.shape[-1] == n * hidden and s % mesh.shape[0] == 0, (x.shape, n, hidden, tuple(mesh.shape))
    host = streams_cols_to_chip_major(x.float(), hidden, mesh.shape[1], n).reshape(1, 1, s, n * hidden)
    return from_torch(
        host.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(mesh, (2, 3)),
    )


def row_split_to_device(mesh, x: torch.Tensor, dtype=ttnn.float32) -> ttnn.Tensor:
    """Host [S, W] -> device [1, 1, S/SP, W] per chip, split by rows over axis 0, replicated over axis 1."""
    s, w = x.shape
    return from_torch(
        x.float().reshape(1, 1, s, w).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(mesh, (2, None)),
    )


def row_split_to_host(mesh, t: ttnn.Tensor, width: int | None = None) -> torch.Tensor:
    """A tensor split by rows over axis 0 and replicated over axis 1 ([1, 1, S/SP, W] per chip) -> host [S, W]
    (mesh column 0's copy). ``width`` trims the last dim."""
    full = ttnn.to_torch(t, mesh_composer=_composer(mesh))
    w = t.shape[-1]
    out = full[..., :w].reshape(-1, w)
    return out if width is None else out[:, :width]


def row_split_copies_to_host(mesh, t: ttnn.Tensor) -> list[torch.Tensor]:
    """Every mesh column's copy of a row-split, TP-replicated tensor, as host [S, W] each (replication check)."""
    full = ttnn.to_torch(t, mesh_composer=_composer(mesh))
    w = t.shape[-1]
    full = full.reshape(-1, full.shape[-1])
    return [full[:, c * w : (c + 1) * w] for c in range(mesh.shape[1])]


def hidden_split_to_device(mesh, x: torch.Tensor, dtype=ttnn.float32) -> ttnn.Tensor:
    """Host [S, H] -> device [1, 1, S/SP, H/TP] per chip: rows split over axis 0, hidden columns over axis 1 (chip
    (r, c) holds columns [c H/TP, (c+1) H/TP))."""
    s, h = x.shape
    assert s % mesh.shape[0] == 0 and h % mesh.shape[1] == 0, (x.shape, tuple(mesh.shape))
    return from_torch(
        x.float().reshape(1, 1, s, h).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(mesh, (2, 3)),
    )


def hidden_split_to_host(mesh, t: ttnn.Tensor) -> torch.Tensor:
    """A tensor split by rows over axis 0 and by hidden columns over axis 1 ([1, 1, S/SP, H/TP] per chip; chip (r, c)
    holds columns [c H/TP, (c+1) H/TP)) -> host [S, H]."""
    full = ttnn.to_torch(t, mesh_composer=_composer(mesh))
    return full.reshape(-1, full.shape[-1])
