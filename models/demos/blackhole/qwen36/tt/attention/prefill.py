# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill forward passes for Qwen3.5-9B gated attention.

Branch A: paged prefill (chunk_page_table is not None) — no memory_config, no cur_pos_tensor.
Branch C: concat prefill (else) — uses memory_config, past_key/past_value; returns new_key/new_value.
"""
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import gated_attention_forward_ttnn


def prefill_forward(
    x,
    cos,
    sin,
    weights,
    config,
    device,
    ckc,
    mc=None,
    paged_kv_cache_key=None,
    paged_kv_cache_value=None,
    page_table=None,
    chunk_page_table=None,
    chunk_start_idx=None,
    chunk_start_idx_tensor=None,
    past_key=None,
    past_value=None,
    use_paged_attention=False,
):
    """Dispatch prefill to paged (Branch A) or concat (Branch C) path."""
    if use_paged_attention and chunk_page_table is not None:
        # Branch A — paged prefill: fill K/V into paged cache + chunked SDPA
        # No memory_config, no cur_pos_tensor.
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
            norm_weights_pre_offset=True,
            page_table=page_table,
            paged_kv_cache_key=paged_kv_cache_key,
            paged_kv_cache_value=paged_kv_cache_value,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            chunk_start_idx_tensor=chunk_start_idx_tensor,
        )
        return output
    else:
        # Branch C — concat path: non-paged prefill and short-sequence paged prefill.
        # Has memory_config=mc, past_key/past_value; returns new_key/new_value.
        output, new_key, new_value = gated_attention_forward_ttnn(
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
            past_key=past_key,
            past_value=past_value,
            compute_kernel_config=ckc,
            use_optimized_concat=True,
            memory_config=mc,
            norm_weights_pre_offset=True,
        )
        return output, new_key, new_value
