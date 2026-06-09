# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""L1 matmul memory helpers for Kokoro decode-shaped ops.

Sweep reference: ``models/experimental/tt_symbiote/tests/test_kokoro_base_matmul_sweep.py``

Memlayout winners (isolated matmul, trace replay):
  * ``L1-width-sharded`` beats ``L1-interleaved`` on device time for all four shapes.
  * BUT width-sharding needs ``InterleavedToSharded`` / ``ShardedToInterleaved`` when
    tensors are not already sharded.  In the host-driven LSTM loop (~64 steps x 2
    directions x reshards/matmul) that host overhead dominates — use ``L1-interleaved``
    there instead.
  * ``L1-width-sharded`` output is still appropriate for **one-shot** matmuls (e.g.
    ``en_nlc`` when ``B==1``): one ``ShardedToInterleaved`` at the end, no per-step tax.
"""

from __future__ import annotations

import math

import ttnn

_TILE = 32

# Width-sharded matmul output was sweep-validated only for ``64x32x192`` (M=64).
# Larger T_aligned should use the 1D-mcast path or default matmul.
_EN_WIDTH_SHARD_MAX_M = 64

# 1x(N/32) output width-shard CoreGrid overflows BH P150 (13-wide) when N >= 416.
_MAX_STYLE_LINEAR_WIDTH_SHARD_CORES = 12


def tile_padded_rows(n: int) -> int:
    return math.ceil(int(n) / _TILE) * _TILE


def matmul_dims_tile_aligned(m: int, k: int, n: int) -> bool:
    return (int(m) % _TILE == 0) and (int(k) % _TILE == 0) and (int(n) % _TILE == 0)


def matmul_m_extent(tensor: ttnn.Tensor, dim: int = -2) -> int:
    """Row extent for matmul output planning (logical vs padded can diverge on long seq)."""
    logical = int(tensor.shape[dim])
    padded = int(tensor.padded_shape[dim])
    return max(logical, padded)


def l1_width_sharded_mc(m: int, k: int, n: int, *, tensor: str) -> ttnn.MemoryConfig:
    """``tensor`` is ``'in0'`` (activation, K-split) or ``'out'`` (N-split)."""
    if tensor == "in0":
        cores = int(k) // _TILE
        shape = (int(m), int(k))
    elif tensor == "out":
        cores = int(n) // _TILE
        shape = (int(m), int(n))
    else:
        raise ValueError(f"tensor must be 'in0' or 'out', got {tensor!r}")
    return ttnn.create_sharded_memory_config(
        shape,
        ttnn.CoreGrid(y=1, x=cores),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
    )


def l1_width_sharded_out_mc(m: int, k: int, n: int) -> ttnn.MemoryConfig:
    """Output-only width shard — preferred for one-shot matmuls (avoids in0 reshard)."""
    return l1_width_sharded_mc(m, k, n, tensor="out")


def feature_dims_width_shardable(rows: int, features: int) -> bool:
    """True when an NLC ``[B, L, C]`` tensor can be L1 width-sharded along ``C``."""
    m = tile_padded_rows(int(rows))
    n = int(features)
    return (m % _TILE == 0) and (n % _TILE == 0)


def l1_width_sharded_feature_mc(rows: int, features: int) -> ttnn.MemoryConfig:
    """Width-shard NLC activations along the channel / feature axis (last dim)."""
    m = tile_padded_rows(int(rows))
    n = int(features)
    if m % _TILE != 0 or n % _TILE != 0:
        raise ValueError(f"rows={rows} features={features} not tile-aligned for width shard")
    return l1_width_sharded_mc(m, n, n, tensor="out")


def is_l1_width_sharded(mc: ttnn.MemoryConfig) -> bool:
    return mc.buffer_type == ttnn.BufferType.L1 and mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED


def pick_l1_activation_mc(
    caller_mc: ttnn.MemoryConfig,
    *,
    rows: int,
    features: int,
    peak_nbytes: int,
    budget_bytes: int = 768 * 1024,
) -> ttnn.MemoryConfig:
    """Prefer L1 width-sharded activations when tile-aligned and within budget."""
    if caller_mc.buffer_type != ttnn.BufferType.DRAM:
        return caller_mc
    if int(peak_nbytes) > int(budget_bytes):
        return caller_mc
    if feature_dims_width_shardable(rows, features):
        return l1_width_sharded_feature_mc(rows, features)
    return ttnn.L1_MEMORY_CONFIG


def en_matmul_plan(
    alignment_TaT: ttnn.Tensor,
    d_nlc: ttnn.Tensor,
    *,
    m_cap_shard: int = 2048,
    m_cap_mcast: int = 4096,
):
    """Plan the ``en_nlc = alignment^T @ d`` matmul (see sweep shape ``64x32x192``)."""
    B = int(d_nlc.shape[0])
    M = matmul_m_extent(alignment_TaT)
    K = int(d_nlc.shape[1])
    N = int(d_nlc.shape[-1])
    if B != 1 or not matmul_dims_tile_aligned(M, K, N):
        return None, None, False
    if M <= _EN_WIDTH_SHARD_MAX_M:
        # Let matmul pick the device-optimal L1 width-sharded grid (2D ND layout on BH).
        # Passing a 1xN CoreGrid here triggers "mem config mismatch" warnings.
        return None, ttnn.L1_MEMORY_CONFIG, True
    # Long T_aligned: default matmul (DRAM/L1 interleaved).  The 1D-mcast grid sized for
    # full N (e.g. 20 cores when N=640) overflows 1x1 mesh devices; width-sharded output
    # also mismatches when logical M != padded physical height.
    return None, None, False


def style_linear_plan(batch: int, style_dim: int, out_features: int):
    """One-shot AdaIN style ``linear`` (e.g. ``32x128x128`` in the generator sweep)."""
    M = tile_padded_rows(int(batch))
    K = int(style_dim)
    N = int(out_features)
    if not matmul_dims_tile_aligned(M, K, N):
        return None, False
    if int(N) // _TILE <= _MAX_STYLE_LINEAR_WIDTH_SHARD_CORES:
        return l1_width_sharded_out_mc(M, K, N), True
    return None, False


def maybe_to_memory_config(x: ttnn.Tensor, mc: ttnn.MemoryConfig) -> tuple[ttnn.Tensor, bool]:
    cur = x.memory_config()
    if cur.buffer_type == mc.buffer_type and cur.memory_layout == mc.memory_layout:
        return x, False
    out = ttnn.to_memory_config(x, mc)
    return out, out is not x


def activation_interleaved_mc(activ_mc: ttnn.MemoryConfig) -> ttnn.MemoryConfig:
    """Interleaved counterpart of an activation memory config (typecast requires matching layout)."""
    if activ_mc.buffer_type == ttnn.BufferType.L1:
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def maybe_reshard_to_caller(x: ttnn.Tensor, caller_mc: ttnn.MemoryConfig) -> ttnn.Tensor:
    """Convert a one-shot width-sharded matmul result back to the caller layout."""
    cur = x.memory_config()
    if cur.buffer_type == caller_mc.buffer_type and cur.memory_layout == caller_mc.memory_layout:
        return x
    if is_l1_width_sharded(cur) and caller_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        out = ttnn.sharded_to_interleaved(x, memory_config=caller_mc)
        if out is not x:
            ttnn.deallocate(x)
        return out
    out, changed = maybe_to_memory_config(x, caller_mc)
    if changed:
        ttnn.deallocate(x)
    return out
