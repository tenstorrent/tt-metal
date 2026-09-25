# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wormhole recurrent gated-delta-rule decode. q keeps its dtype; k, v, beta, and g follow high_precision."""
import ttnn
from models.common.utility_functions import is_blackhole
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    _recurrent_read_query_program_config,
    l2_norm_ttnn,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn as _recurrent_gated_delta_rule_decode_upstream,
)


def _write_state_wh(h, k_row, delta, beta_t, outer_bf8=False):
    """h += beta * outer(k, delta). Keep outer_bf8 False unless h is already bf8."""
    B, H, V = h.shape[0], h.shape[1], h.shape[3]
    _L1 = ttnn.L1_MEMORY_CONFIG

    # Unsqueeze to [B,H,1,1]; a reshape would cross tiled dims.
    _beta_bh1 = ttnn.unsqueeze(beta_t, -1)
    beta_expanded = ttnn.unsqueeze(_beta_bh1, -1)
    ttnn.deallocate(_beta_bh1)
    k_col = ttnn.transpose(k_row, 2, 3, memory_config=_L1)
    d_row = ttnn.reshape(delta, [B, H, 1, V], memory_config=_L1)

    d_scaled = ttnn.multiply(d_row, beta_expanded, memory_config=_L1)
    if outer_bf8:
        outer_dtype = None if h.dtype == ttnn.float32 else ttnn.bfloat8_b
    else:
        outer_dtype = ttnn.bfloat8_b if h.dtype == ttnn.bfloat8_b else None
    outer = ttnn.multiply(k_col, d_scaled, memory_config=_L1, dtype=outer_dtype)
    return ttnn.add(h, outer, memory_config=_L1)


def recurrent_gated_delta_rule_decode_wh(
    q,
    k,
    v,
    beta,
    g,
    scale=None,
    initial_state=None,
    device=None,
    high_precision=False,
    tile_opt=False,
):
    """q is not cast to fp32 up front; tile_opt returns [B,H,1,V] instead of [B,1,H,V]."""
    B = q.shape[0]
    H = q.shape[2]
    K = q.shape[3]
    V = v.shape[3]

    if high_precision:
        k = ttnn.typecast(k, ttnn.float32)
        v = ttnn.typecast(v, ttnn.float32)
        beta = ttnn.typecast(beta, ttnn.float32)
        g = ttnn.typecast(g, ttnn.float32)

    # q stays at its incoming dtype through the norm and scale.
    if scale is None:
        scale = K**-0.5
    if tile_opt:
        # Fold L2's 1/sqrt(K) into the attention scale.
        q = ttnn.rms_norm(q, epsilon=1e-6 / K)
        q = ttnn.multiply(q, scale * (K**-0.5), memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        q = l2_norm_ttnn(q, dim=-1)
        q = ttnn.multiply(q, scale, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = l2_norm_ttnn(k, dim=-1)

    # [B,1,H,K] -> [B,H,1,K] by transpose; a reshape that moves the singleton retile.
    q_row = ttnn.transpose(q, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    k_row = ttnn.transpose(k, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    v_t = ttnn.reshape(v, [B, H, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    beta_t = ttnn.reshape(beta, [B, H], memory_config=ttnn.L1_MEMORY_CONFIG)
    g_t = ttnn.reshape(g, [B, H], memory_config=ttnn.L1_MEMORY_CONFIG)

    decay_t = ttnn.exp(g_t, memory_config=ttnn.L1_MEMORY_CONFIG)

    h = initial_state
    if h is None:
        h = ttnn.zeros(
            [B, H, K, V], device=device, dtype=ttnn.float32 if high_precision else ttnn.bfloat16, memory_config=None
        )
    elif high_precision and h.dtype != ttnn.float32:
        h = ttnn.typecast(h, ttnn.float32)

    # Same-config to_memory_config is not a no-op inside a trace.
    if h.memory_config().buffer_type != ttnn.BufferType.L1:
        h = ttnn.to_memory_config(h, ttnn.L1_MEMORY_CONFIG)

    read_query_compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    read_query_prog_cfg = None
    if device is not None:
        try:
            read_query_prog_cfg = _recurrent_read_query_program_config(device, K, V)
        except Exception:
            pass

    # Two unsqueezes, not one reshape: reshape crosses the tiled dims.
    _L1 = ttnn.L1_MEMORY_CONFIG
    _decay_bh1 = ttnn.unsqueeze(decay_t, -1)
    decay_bhkv = ttnn.unsqueeze(_decay_bh1, -1)
    ttnn.deallocate(_decay_bh1)
    h = ttnn.multiply(h, decay_bhkv, memory_config=_L1)

    v_read = ttnn.matmul(
        k_row, h, memory_config=_L1, program_config=read_query_prog_cfg, compute_kernel_config=read_query_compute_cfg
    )
    v_read = ttnn.reshape(v_read, [B, H, V], memory_config=_L1)

    delta = ttnn.subtract(v_t, v_read, memory_config=_L1)
    # Do not pass outer_bf8=tile_opt: bf8 increments regress accuracy while h is bf16.
    h = _write_state_wh(h=h, k_row=k_row, delta=delta, beta_t=beta_t, outer_bf8=False)

    if h.dtype == ttnn.float32 and q_row.dtype != ttnn.float32:
        q_row = ttnn.typecast(q_row, ttnn.float32)

    o_t = ttnn.matmul(
        q_row, h, memory_config=_L1, program_config=read_query_prog_cfg, compute_kernel_config=read_query_compute_cfg
    )

    # tile_opt returns [B,H,1,V]; reshape to [B,1,H,V] crosses tiled dims.
    if tile_opt:
        return o_t, h
    o = ttnn.reshape(o_t, [B, 1, H, V], memory_config=_L1)
    return o, h


def recurrent_gated_delta_rule_decode_dispatch(*args, model_args=None, **kwargs):
    """All Wormhole devices use the local decode; upstream overflows. model_args is ignored."""
    _use_wh = not is_blackhole()
    if not _use_wh:
        return _recurrent_gated_delta_rule_decode_upstream(*args, **kwargs)
    return recurrent_gated_delta_rule_decode_wh(*args, **kwargs)
