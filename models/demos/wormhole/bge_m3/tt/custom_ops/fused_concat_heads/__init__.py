# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 fused multi-head concat.

    bge_concat_heads_headsplit(context, *, head_groups, out_memcfg)
        Concatenates the attention heads back into one tensor. Each core owns
        a head group, so the reader takes one barrier per block.
"""

from .op import bge_concat_heads_headsplit

__all__ = ["bge_concat_heads_headsplit"]
