# tt/attention/__init__.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

"""
Public API for attention. Downstream modules import from here, not from
submodules directly.
"""

from .masks import build_causal_mask
from .self_attention import tst_self_attention
from .cross_attention import precompute_cross_attn_kv, tst_cross_attention, tst_cross_attention_with_kv
from .kv_cache import allocate_kv_cache, tst_self_attention_cached

__all__ = [
    "build_causal_mask",
    "tst_self_attention",
    "precompute_cross_attn_kv",
    "tst_cross_attention",
    "tst_cross_attention_with_kv",
    "allocate_kv_cache",
    "tst_self_attention_cached",
]
