# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

OPS = ("concat_heads_decode",)


def covers(attn_output, num_heads, prefetcher):
    return (
        prefetcher is None
        and num_heads == 32
        and attn_output.dtype == ttnn.bfloat16
        and not attn_output.is_sharded()
        and attn_output.memory_config().buffer_type == ttnn.BufferType.DRAM
        and tuple(attn_output.padded_shape) == (1, 1, 32, 128)
    )


def concat_heads_decode(attn_output, num_heads):
    return ttnn.experimental.llama31_8b.concat_heads_decode(attn_output, num_heads)
