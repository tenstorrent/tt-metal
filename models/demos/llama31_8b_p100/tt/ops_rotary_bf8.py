# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

OPS = ("rotary_bf8",)

HEADS = (32, 8)
SEQ = 1024
HEAD_DIM = 128


def _dram_bf16_tile(t):
    return (
        t.dtype == ttnn.bfloat16
        and t.layout == ttnn.TILE_LAYOUT
        and not t.is_sharded()
        and t.memory_config().buffer_type == ttnn.BufferType.DRAM
    )


def covers(q, k, cos, sin, trans_mat):
    if not all(_dram_bf16_tile(t) for t in (q, k, cos, sin, trans_mat)):
        return False
    if tuple(cos.padded_shape) != (1, 1, SEQ, HEAD_DIM) or tuple(sin.padded_shape) != (1, 1, SEQ, HEAD_DIM):
        return False
    if tuple(trans_mat.padded_shape)[-2:] != (ttnn.TILE_SIZE, ttnn.TILE_SIZE):
        return False
    return tuple(q.padded_shape) == (1, HEADS[0], SEQ, HEAD_DIM) and tuple(k.padded_shape) == (
        1,
        HEADS[1],
        SEQ,
        HEAD_DIM,
    )


def rotary_bf8(input_tensor, cos, sin, trans_mat, output_dtype):
    return ttnn.experimental.llama31_8b.rotary_bf8(input_tensor, cos, sin, trans_mat, output_dtype=output_dtype)
