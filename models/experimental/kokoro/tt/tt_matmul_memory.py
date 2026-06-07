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


def tile_padded_rows(n: int) -> int:
    return math.ceil(int(n) / _TILE) * _TILE


def matmul_dims_tile_aligned(m: int, k: int, n: int) -> bool:
    return (int(m) % _TILE == 0) and (int(k) % _TILE == 0) and (int(n) % _TILE == 0)


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


def maybe_to_memory_config(x: ttnn.Tensor, mc: ttnn.MemoryConfig) -> tuple[ttnn.Tensor, bool]:
    cur = x.memory_config()
    if cur.buffer_type == mc.buffer_type and cur.memory_layout == mc.memory_layout:
        return x, False
    out = ttnn.to_memory_config(x, mc)
    return out, out is not x
