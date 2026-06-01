# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
def reset_concat_cache(attn):
    attn.past_key = None
    attn.past_value = None


def attach_paged_kv_cache(attn, k_cache, v_cache):
    attn.paged_kv_cache_key = k_cache
    attn.paged_kv_cache_value = v_cache
    attn.use_paged_attention = True
