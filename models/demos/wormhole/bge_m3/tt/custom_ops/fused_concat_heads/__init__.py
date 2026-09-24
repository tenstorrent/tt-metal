# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 fused multi-head concat.

    bge_concat_heads_headsplit(context, *, head_groups, out_memcfg)
        Concatenates the attention heads back into one tensor. Each core owns
        a head group, so the reader takes one barrier per block.
"""

from .op import (
    bge_concat_heads_headsplit,
    bge_concat_heads_stock,
    bge_concat_heads_tracka,
)

__all__ = [
    "bge_concat_heads_headsplit",
    # Sweep baselines. The model path uses head-split; these measure against it.
    "bge_concat_heads_stock",
    "bge_concat_heads_tracka",
]
