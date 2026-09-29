# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 fused QKV matmul and Q/K/V head split.

    bge_qkv_heads_headsplit(...) -> (q, k, v)
        Splits a fused QKV tensor into Q, K, and V heads. Each core owns a
        head group.
"""

from .op import bge_qkv_heads_headsplit

__all__ = [
    "bge_qkv_heads_headsplit",
]
