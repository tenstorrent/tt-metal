# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# KDA (Kimi Delta Attention) ttnn ops.
#
# Mirrors models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py
# (the in-tree Gated DeltaNet recurrent op). The ONLY structural change is the forget gate:
#   GDN  decay_t : [B, H]      -> [B, H, 1, 1]   (scalar per head, broadcast over K×V)
#   KDA  decay_t : [B, HV, K]  -> [B, HV, K, 1]  (per-channel/diagonal, broadcast over V)
# See ../API_SPEC.md and ../bringup_log.md (Phase 2 delta analysis).

from __future__ import annotations

import ttnn

# fp32-accumulate matmul config (matches the GDN reference for numerical fidelity).
_MM_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


def l2norm_ttnn(x, eps: float = 1e-6):
    """L2-normalize over the last dim (fused rms_norm path). x: [..., K]."""
    K = x.shape[-1]
    normed = ttnn.rms_norm(x, epsilon=eps / K)
    return ttnn.multiply(normed, K ** -0.5)


def kda_gate_ttnn(g_pre, A_log, dt_bias=None, lower_bound=None):
    """KDA log-space decay gate. g_pre:[...,HV,K], A_log:[HV,1], dt_bias:[HV,K] (or None).

    default : g = -exp(A_log) * softplus(g_pre + dt_bias)
    lb form : g = lower_bound * sigmoid(exp(A_log) * (g_pre + dt_bias))
    All broadcast over the leading (B,T) dims; A_log/dt_bias are per (HV,K).
    """
    g = ttnn.add(g_pre, dt_bias) if dt_bias is not None else g_pre
    expA = ttnn.exp(A_log)  # [HV,1] broadcast over K
    if lower_bound is None:
        g = ttnn.multiply(ttnn.softplus(g), expA)
        return ttnn.neg(g)
    return ttnn.multiply(ttnn.sigmoid(ttnn.multiply(g, expA)), lower_bound)


def _bh_last(x, name):
    return x.shape[-1]


def recurrent_kda_ttnn(q, k, v, g, beta, scale=None, initial_state=None, device=None):
    """Token-by-token KDA recurrence on device — mirrors torch naive_recurrent_kda.

    Inputs (ttnn, TILE, fp32 recommended), already L2-normed (q,k), gated (g log-space),
    and beta already sigmoided:
        q, k   : [B, T, HV, K]   (GVA must be pre-expanded to HV upstream)
        v      : [B, T, HV, V]
        g      : [B, T, HV, K]   log-space decay (<= 0)
        beta   : [B, T, HV]
        initial_state : [B, HV, K, V] or None
    Returns (o [B, T, HV, V], S [B, HV, K, V]).
    """
    B, T, HV, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    # state h: [B, HV, K, V] fp32 in DRAM
    if initial_state is not None:
        h = ttnn.to_layout(ttnn.typecast(initial_state, ttnn.float32), ttnn.TILE_LAYOUT)
    else:
        h = ttnn.zeros([B, HV, K, V], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    h = ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)

    outs = []
    for t in range(T):
        # slice timestep t -> [B, HV, *]
        q_t = ttnn.reshape(q[:, t], [B, HV, 1, K])
        k_t = ttnn.reshape(k[:, t], [B, HV, 1, K])
        v_t = ttnn.reshape(v[:, t], [B, HV, 1, V])
        g_t = g[:, t]                          # [B, HV, K]
        beta_t = beta[:, t]                    # [B, HV]

        # 1) diagonal decay: h *= exp(g_t) with g_t broadcast [K,1] over V  (THE KDA delta)
        decay = ttnn.reshape(ttnn.exp(g_t), [B, HV, K, 1])
        h = ttnn.multiply(h, decay)

        # 2) read: v_read = k_t @ h  -> [B, HV, 1, V]
        v_read = ttnn.matmul(k_t, h, compute_kernel_config=_MM_CFG)

        # 3) delta = v_t - v_read
        delta = ttnn.subtract(v_t, v_read)

        # 4) write: h += beta_t * (k_t^T @ delta)   (outer product [K,1]@[1,V] = [K,V])
        k_col = ttnn.reshape(k_t, [B, HV, K, 1])
        outer = ttnn.matmul(k_col, delta, compute_kernel_config=_MM_CFG)   # [B,HV,K,V]
        outer = ttnn.multiply(outer, ttnn.reshape(beta_t, [B, HV, 1, 1]))
        h = ttnn.add(h, outer)

        # 5) output: o_t = (scale*q_t) @ h  -> [B, HV, 1, V]
        o_t = ttnn.matmul(ttnn.multiply(q_t, scale), h, compute_kernel_config=_MM_CFG)
        outs.append(ttnn.reshape(o_t, [B, 1, HV, V]))

    o = outs[0] if T == 1 else ttnn.concat(outs, dim=1)
    return o, h
