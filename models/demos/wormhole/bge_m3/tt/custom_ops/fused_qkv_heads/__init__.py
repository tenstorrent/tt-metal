# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 fused QKV matmul and Q/K/V head split.

    bge_qkv_heads_headsplit(...) -> (q, k, v)
        Splits a fused QKV tensor into Q, K, and V heads. Each core owns a
        head group.

    bge_qkv_heads_scatter(...) -> (q, k, v)
        Writes the QKV matmul output straight into the Q, K, and V head
        buffers, so the split needs no separate program.
"""

from .op import bge_qkv_heads_headsplit, bge_qkv_heads_scatter

__all__ = [
    "bge_qkv_heads_headsplit",
    "bge_qkv_heads_scatter",
]
