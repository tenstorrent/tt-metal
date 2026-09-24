# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 fused QKV matmul and Q/K/V head split.

    bge_qkv_heads_headsplit(...) -> (q, k, v)
        Splits a fused QKV tensor into Q, K, and V heads. Each core owns a
        head group.
"""

from .op import (
    bge_qkv_heads_headsplit,
    bge_qkv_heads_scatter,
    bge_qkv_heads_stock,
    bge_qkv_heads_tracka,
)

__all__ = [
    "bge_qkv_heads_headsplit",
    # Sweep baselines. The model path uses head-split; these measure against it.
    "bge_qkv_heads_stock",
    "bge_qkv_heads_tracka",
    "bge_qkv_heads_scatter",
]
