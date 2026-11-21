# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

OPS = ("qkv_heads_decode",)

N_HEADS, N_KV_HEADS = 32, 8
ROWS, WIDTH = 32, 6144
SHARD = (32, 192)
GRIDS = (
    (((0, 0), (7, 3)),),
    (((0, 0), (10, 1)), ((0, 2), (9, 2))),
)
_grids = []
_seen = {}


def reset():
    _seen.clear()


def _grid(ranges):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b)) for a, b in ranges])


def _layout_ok(xqkv):
    if tuple(xqkv.padded_shape)[-2:] != (ROWS, WIDTH):
        return False
    mc = xqkv.memory_config()
    if mc.buffer_type != ttnn.BufferType.L1:
        return False
    if not xqkv.is_sharded():
        return True
    if not _grids:
        _grids.extend(_grid(g) for g in GRIDS)
    ss = mc.shard_spec
    return (
        mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        and tuple(ss.shape) == SHARD
        and ss.orientation == ttnn.ShardOrientation.ROW_MAJOR
        and any(ss.grid == g for g in _grids)
    )


def covers(xqkv, num_heads, num_kv_heads):
    if num_heads != N_HEADS or num_kv_heads != N_KV_HEADS or xqkv.dtype != ttnn.bfloat16:
        return False
    key = (tuple(xqkv.padded_shape), xqkv.is_sharded())
    if key not in _seen:
        _seen[key] = _layout_ok(xqkv)
    return _seen[key]


def qkv_heads_decode(xqkv_fused, num_heads, num_kv_heads, memory_config):
    return ttnn.experimental.llama31_8b.qkv_heads_decode(
        xqkv_fused, num_heads, num_kv_heads, memory_config=memory_config
    )
