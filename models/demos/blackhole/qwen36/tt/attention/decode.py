# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode forward pass for Qwen3.5-9B gated attention.

Branch B: paged decode — uses memory_config=mc and cur_pos_tensor=position_tensor.
"""
from models.demos.blackhole.qwen36.tt import tp_common as tpc
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
    """Branch B — paged decode: paged_update_cache + paged_sdpa_decode via page_table.

    I-1 items (single-user decode, B == 1; each flag "0" restores the pre-I-1 path):
      QWEN36_I1_D15: q|k|v from 2 matmuls (weights.qkv_fused + weights.gate_deint, already loaded
                     for prefill) instead of 3 (q_proj q|gate interleaved, k_proj, v_proj).
      QWEN36_I1_D4A: head-major decode RoPE (rotary_embedding_hf on [1,1,H,64]) and no head
                     transposes. Needs the row-replicated [1,32,64] cos/sin that
                     Qwen36Model.prepare_decode_inputs_host packs under the same flag; any other
                     cos/sin shape keeps the pre-I-1 RoPE inside gated_attention_forward_ttnn.
      QWEN36_I1_D6:  one reshape instead of transpose + concatenate_heads for the SDPA output.
    I-3 item (B == 1; "0" restores the pre-I-3 path):
      QWEN36_I3_FA_PROGCFG: explicit 1D program configs for the decode projection matmuls
                            (tp_common._I3_FA_DECODE_PROGCFGS) instead of ttnn auto-config.
    """
    single = x.shape[0] == 1
    i1_kwargs = {}
    if single and tpc.i1_enabled("D15"):
        i1_kwargs.update(qkv_fused_weight=weights.qkv_fused, gate_deint_weight=weights.gate_deint)
    if single and tpc.i1_enabled("D4A"):
        i1_kwargs["decode_head_major_rope"] = True
    if single and tpc.i1_enabled("D6"):
        i1_kwargs["decode_concat_reshape"] = True
    _i3_fa_pc = tpc.i3_fa_decode_progcfg_fn(device) if single else None
    if _i3_fa_pc is not None:
        i1_kwargs["decode_progcfg_fn"] = _i3_fa_pc
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
        **i1_kwargs,
    )
    return output
