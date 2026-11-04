# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill forward passes for Qwen3.5-9B gated attention.

Branch A: paged prefill (chunk_page_table is not None) — no memory_config, no cur_pos_tensor.
Branch C: concat prefill (else) — uses memory_config, past_key/past_value; returns new_key/new_value.
"""
import contextlib

import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import gated_attention_forward_ttnn


@contextlib.contextmanager
def _q_proj_program_config(x, q_w):
    """Prefill B=1, T=2048 only: give the Q(+gate) projection (5120x12288) the swept
    1D mcast_in0 program config. The call lives in an off-limits file that passes no
    program_config, so ttnn.linear is wrapped for the duration of this call and only
    the call whose weight is this q_proj gets the config."""
    shape = tuple(x.shape)
    use = len(shape) == 3 and shape[0] == 1 and shape[1] == 2048 and tuple(q_w.shape)[-2:] == (5120, 12288)
    if not use:
        yield
        return
    orig_linear = ttnn.linear

    def _linear(a, b, *args, **kwargs):
        if b is q_w and kwargs.get("program_config") is None:
            kwargs["program_config"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(13, 10),
                in0_block_w=10,
                out_subblock_h=1,
                out_subblock_w=3,
                out_block_h=7,
                out_block_w=15,
                per_core_M=7,
                per_core_N=30,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )
        elif kwargs.get("program_config") is None and list(b.shape)[-2:] == [6144, 5120]:
            m = 1
            for d in list(a.shape)[:-1]:
                m *= d
            if m == 2048:
                kwargs["program_config"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(13, 10),
                    in0_block_w=12,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    out_block_h=7,
                    out_block_w=13,
                    per_core_M=7,
                    per_core_N=13,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=True,
                )
        return orig_linear(a, b, *args, **kwargs)

    ttnn.linear = _linear
    try:
        yield
    finally:
        ttnn.linear = orig_linear


def prefill_forward(x, cos, sin, weights, config, device, ckc, *args, **kwargs):
    with _q_proj_program_config(x, weights.q_proj):
        from models.demos.blackhole.bonsai2.tt.attn_qg import QGSplitTtnn, is_reordered

        if not is_reordered(weights):
            return _prefill_forward_impl(x, cos, sin, weights, config, device, ckc, *args, **kwargs)
        # Bonsai: q_proj stored as [q heads | gate heads] (models/demos/blackhole/bonsai2/tt/attn_qg.py)
        import models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention as _attn_mod

        _orig = _attn_mod.ttnn
        _attn_mod.ttnn = QGSplitTtnn(weights, config.num_heads, config.head_dim)
        try:
            return _prefill_forward_impl(x, cos, sin, weights, config, device, ckc, *args, **kwargs)
        finally:
            _attn_mod.ttnn = _orig


def _prefill_forward_impl(
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
