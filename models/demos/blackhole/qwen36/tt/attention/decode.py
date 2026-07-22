# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode forward pass for Qwen3.5-9B gated attention.

Branch B: paged decode — uses memory_config=mc and cur_pos_tensor=position_tensor.
"""
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import gated_attention_forward_ttnn


def decode_forward(
    x,
    cos,
    sin,
    weights,
    config,
    device,
    ckc,
    mc,
    position_tensor=None,
    page_table=None,
    paged_kv_cache_key=None,
    paged_kv_cache_value=None,
):
    """Branch B — paged decode: paged_update_cache + paged_sdpa_decode via page_table."""
    output, _, _ = gated_attention_forward_ttnn(
        hidden_states=x,
        q_proj_weight=weights.q_proj,
        k_proj_weight=weights.k_proj,
        v_proj_weight=weights.v_proj,
        o_proj_weight=weights.o_proj,
        q_norm_weight=weights.q_norm,
        k_norm_weight=weights.k_norm,
        cos=cos,
        sin=sin,
        num_attention_heads=config.num_heads,
        num_key_value_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        device=device,
        norm_eps=config.norm_eps,
        compute_kernel_config=ckc,
        use_optimized_concat=True,
        memory_config=mc,
        norm_weights_pre_offset=True,
        cur_pos_tensor=position_tensor,
        page_table=page_table,
        paged_kv_cache_key=paged_kv_cache_key,
        paged_kv_cache_value=paged_kv_cache_value,
    )
    return output
