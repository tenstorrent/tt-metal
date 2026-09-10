# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-token Kimi Delta Attention recurrence with PER-CHANNEL decay.

Forked from models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py::recurrent_gated_delta_rule_decode_ttnn
(Gated DeltaNet, per-head scalar decay). KDA's log-decay ``g`` has shape [B,1,H,K]; the state decay is therefore a
row-broadcast multiply ``S[b,h,k,:] *= exp(g[b,h,k])`` instead of a scalar per head. Everything else (L2 norm, scale,
delta rule write, read-out) is identical to the Gated DeltaNet step and matches transformers'
``recurrent_kimi_delta_attention`` / fla ``naive_recurrent_kda``:
    q,k <- l2norm; q *= K^-0.5; S <- S*exp(g)[...,None]; delta = beta*(v - k^T S); S += k (x) delta; o = q^T S.
"""

from __future__ import annotations

import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    _recurrent_read_query_program_config,
    fused_decay_and_write_ttnn,
    l2_norm_ttnn,
)

_L1 = ttnn.L1_MEMORY_CONFIG


def recurrent_kda_decode_ttnn(q, k, v, beta, g, state, *, scale=None, device=None):
    """Args: q,k [B,1,H,K]; v [B,1,H,V]; beta [B,1,H]; g [B,1,H,K] (log decay, <= 0); state [B,H,K,V] fp32 TILE.
    Returns (o [B,1,H,V] fp32, new_state [B,H,K,V] fp32). ``state`` is read only."""
    B, _, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5
    q = ttnn.typecast(q, ttnn.float32)
    k = ttnn.typecast(k, ttnn.float32)
    v = ttnn.typecast(v, ttnn.float32)
    beta = ttnn.typecast(beta, ttnn.float32)
    g = ttnn.typecast(g, ttnn.float32)

    q = l2_norm_ttnn(q, dim=-1)
    k = l2_norm_ttnn(k, dim=-1)
    q = ttnn.multiply(q, scale, memory_config=_L1)

    q_row = ttnn.reshape(q, [B, H, 1, K], memory_config=_L1)
    k_row = ttnn.reshape(k, [B, H, 1, K], memory_config=_L1)
    v_t = ttnn.reshape(v, [B, H, V], memory_config=_L1)
    beta_t = ttnn.reshape(beta, [B, H], memory_config=_L1)
    g_bhk1 = ttnn.reshape(g, [B, H, K, 1], memory_config=_L1)

    h = state if state.dtype == ttnn.float32 else ttnn.typecast(state, ttnn.float32)
    h = ttnn.to_memory_config(h, _L1)
    # per-channel decay: rows of the [K,V] state decay independently (exp fused into the multiply)
    h = ttnn.multiply(h, g_bhk1, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], memory_config=_L1)

    compute_cfg = ttnn.init_device_compute_kernel_config(
        device.arch() if device is not None else h.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    prog_cfg = None
    if device is not None:
        try:
            prog_cfg = _recurrent_read_query_program_config(device, K, V)
        except Exception:
            prog_cfg = None

    v_read = ttnn.matmul(k_row, h, memory_config=_L1, program_config=prog_cfg, compute_kernel_config=compute_cfg)
    v_read = ttnn.reshape(v_read, [B, H, V], memory_config=_L1)
    delta = ttnn.subtract(v_t, v_read, memory_config=_L1)
    k_t = ttnn.reshape(k_row, [B, H, K], memory_config=_L1)
    # decay_t is unused with apply_decay=False but is reshaped to [B,H,1,1] inside; hand it a per-head tensor, not the per-channel gate
    h = fused_decay_and_write_ttnn(
        h=h, k_t=k_t, delta=delta, decay_t=beta_t, beta_t=beta_t, device=device, apply_decay=False
    )
    o_t = ttnn.matmul(q_row, h, memory_config=_L1, program_config=prog_cfg, compute_kernel_config=compute_cfg)
    o = ttnn.reshape(o_t, [B, 1, H, V], memory_config=_L1)
    return o, h
