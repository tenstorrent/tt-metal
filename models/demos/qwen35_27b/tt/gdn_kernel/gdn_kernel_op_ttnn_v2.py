# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Optimized GDN prefill: all parallelizable ops (Q/K/V extraction, L2-norm,
head expansion, gate computation) run once for the full N-token chunk before
the per-token loop. Only the causally-sequential DeltaNet state update remains
in the loop.

Dispatch reduction: ~33 ops/token → ~7 ops/token (~4.7× fewer dispatches).

API is identical to gdn_prefill_ttnn — drop-in replacement.
"""

import ttnn
from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op_ttnn import _L1, _l2_norm_ttnn, gdn_recurrence_ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def gdn_prefill_ttnn_opt(
    conv_out,
    a_fused,
    b_fused,
    neg_exp_A,
    dt_bias,
    norm_w,
    scale_tt,
    rms_scale_tt,
    rms_eps_tt,
    state,
    output,
    num_pairs,
    num_tokens,
    Nv_TP,
    Nk_TP,
    repeat_factor,
    key_dim_tp,
):
    """Optimized GDN prefill — same signature and semantics as gdn_prefill_ttnn.

    Moves all token-independent ops outside the per-token recurrence loop so
    they execute as single batched dispatches over all N tokens.

    Args match gdn_prefill_ttnn exactly:
        conv_out  : [1, N, qkv_dim_tp]
        a_fused   : [1, N, Nv_TP]
        b_fused   : [1, N, Nv_TP]
        neg_exp_A : [1, 1, Nv_TP]
        dt_bias   : [1, 1, Nv_TP]
        state     : [num_pairs, Dk, Dv]  — updated in-place
        output    : [num_pairs * N, 1, Dv]  — written on return
    """
    N = num_tokens
    qkv_dim_tp = conv_out.shape[-1]
    value_dim_tp = qkv_dim_tp - 2 * key_dim_tp
    Dk = key_dim_tp // Nk_TP
    Dv = value_dim_tp // Nv_TP

    # =========================================================
    # PRE-LOOP: batch all N tokens — one dispatch per op type
    # =========================================================

    # --- Split Q, K, V from conv_out [1, N, qkv_dim_tp] ---
    q_all = ttnn.slice(conv_out, (0, 0, 0), (1, N, key_dim_tp))
    k_all = ttnn.slice(conv_out, (0, 0, key_dim_tp), (1, N, 2 * key_dim_tp))
    v_all = ttnn.slice(conv_out, (0, 0, 2 * key_dim_tp), (1, N, qkv_dim_tp))

    # --- L2-norm Q per head: reshape [1,N,key_dim_tp] → [1,N*Nk_TP,Dk] ---
    q_rm = ttnn.to_layout(q_all, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(q_all)
    q_h_rm = ttnn.reshape(q_rm, (1, N * Nk_TP, Dk))
    ttnn.deallocate(q_rm)
    q_h = ttnn.to_layout(q_h_rm, ttnn.TILE_LAYOUT, memory_config=_L1)
    ttnn.deallocate(q_h_rm)
    q_n = _l2_norm_ttnn(q_h)  # [1, N*Nk_TP, Dk] in L1
    ttnn.deallocate(q_h)

    # --- Scale Q ---
    q_scaled = ttnn.multiply(q_n, scale_tt, memory_config=_L1)
    ttnn.deallocate(q_n)

    # --- L2-norm K per head ---
    k_rm = ttnn.to_layout(k_all, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(k_all)
    k_h_rm = ttnn.reshape(k_rm, (1, N * Nk_TP, Dk))
    ttnn.deallocate(k_rm)
    k_h = ttnn.to_layout(k_h_rm, ttnn.TILE_LAYOUT, memory_config=_L1)
    ttnn.deallocate(k_h_rm)
    k_n = _l2_norm_ttnn(k_h)  # [1, N*Nk_TP, Dk] in L1
    ttnn.deallocate(k_h)

    # --- Expand Q heads: Nk_TP → Nv_TP via repeat_interleave ---
    # Reshape to [1, N, Nk_TP, Dk] so repeat_interleave acts on head dim only
    q_scaled_rm = ttnn.to_layout(q_scaled, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(q_scaled)
    q_4d_rm = ttnn.reshape(q_scaled_rm, (1, N, Nk_TP, Dk))
    ttnn.deallocate(q_scaled_rm)
    q_4d = ttnn.to_layout(q_4d_rm, ttnn.TILE_LAYOUT, memory_config=_L1)
    ttnn.deallocate(q_4d_rm)
    q_exp = ttnn.repeat_interleave(q_4d, repeat_factor, dim=2)  # [1, N, Nv_TP, Dk]
    ttnn.deallocate(q_4d)

    # Reshape to [N*num_pairs, 1, Dk] for per-token slicing
    q_exp_rm = ttnn.to_layout(q_exp, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(q_exp)
    q_pairs_all_rm = ttnn.reshape(q_exp_rm, (N * num_pairs, 1, Dk))
    ttnn.deallocate(q_exp_rm)
    q_pairs_all = ttnn.to_layout(q_pairs_all_rm, ttnn.TILE_LAYOUT, memory_config=_DRAM)
    ttnn.deallocate(q_pairs_all_rm)

    # --- Expand K heads: Nk_TP → Nv_TP ---
    k_n_rm = ttnn.to_layout(k_n, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(k_n)
    k_4d_rm = ttnn.reshape(k_n_rm, (1, N, Nk_TP, Dk))
    ttnn.deallocate(k_n_rm)
    k_4d = ttnn.to_layout(k_4d_rm, ttnn.TILE_LAYOUT, memory_config=_L1)
    ttnn.deallocate(k_4d_rm)
    k_exp = ttnn.repeat_interleave(k_4d, repeat_factor, dim=2)  # [1, N, Nv_TP, Dk]
    ttnn.deallocate(k_4d)

    k_exp_rm = ttnn.to_layout(k_exp, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(k_exp)
    k_row_all_rm = ttnn.reshape(k_exp_rm, (N * num_pairs, 1, Dk))
    ttnn.deallocate(k_exp_rm)
    k_row_all = ttnn.to_layout(k_row_all_rm, ttnn.TILE_LAYOUT, memory_config=_DRAM)
    ttnn.deallocate(k_row_all_rm)
    # Transpose once for all tokens
    k_col_all = ttnn.transpose(k_row_all, -2, -1, memory_config=_DRAM)  # [N*num_pairs, Dk, 1]

    # --- Reshape V: [1, N, value_dim_tp] → [N*num_pairs, 1, Dv] ---
    v_rm = ttnn.to_layout(v_all, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(v_all)
    v_pairs_all_rm = ttnn.reshape(v_rm, (N * num_pairs, 1, Dv))
    ttnn.deallocate(v_rm)
    v_pairs_all = ttnn.to_layout(v_pairs_all_rm, ttnn.TILE_LAYOUT, memory_config=_DRAM)
    ttnn.deallocate(v_pairs_all_rm)

    # --- Gates: beta = sigmoid(b_fused), g = neg_exp_A * softplus(a_fused + dt_bias) ---
    # Compute for all N tokens at once, reshape to [N*num_pairs, 1, 1]
    beta_flat = ttnn.sigmoid(b_fused, memory_config=_L1)  # [1, N, Nv_TP]
    beta_rm = ttnn.to_layout(beta_flat, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(beta_flat)
    beta_pairs_all_rm = ttnn.reshape(beta_rm, (N * num_pairs, 1, 1))
    ttnn.deallocate(beta_rm)
    beta_pairs_all = ttnn.to_layout(beta_pairs_all_rm, ttnn.TILE_LAYOUT, memory_config=_DRAM)
    ttnn.deallocate(beta_pairs_all_rm)

    a_dt = ttnn.add(a_fused, dt_bias, memory_config=_L1)  # [1, N, Nv_TP]
    sp = ttnn.softplus(a_dt, memory_config=_L1)
    ttnn.deallocate(a_dt)
    g_flat = ttnn.multiply(neg_exp_A, sp, memory_config=_L1)
    ttnn.deallocate(sp)
    g_rm = ttnn.to_layout(g_flat, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(g_flat)
    g_pairs_all_rm = ttnn.reshape(g_rm, (N * num_pairs, 1, 1))
    ttnn.deallocate(g_rm)
    g_pairs_all = ttnn.to_layout(g_pairs_all_rm, ttnn.TILE_LAYOUT, memory_config=_DRAM)
    ttnn.deallocate(g_pairs_all_rm)

    # =========================================================
    # PER-TOKEN LOOP: only the causally-sequential state update
    # =========================================================
    state_l1 = ttnn.to_memory_config(state, _L1)
    token_outs = []

    for t in range(N):
        s, e = t * num_pairs, (t + 1) * num_pairs
        q_t = ttnn.slice(q_pairs_all, (s, 0, 0), (e, 1, Dk))
        k_row_t = ttnn.slice(k_row_all, (s, 0, 0), (e, 1, Dk))
        k_col_t = ttnn.slice(k_col_all, (s, 0, 0), (e, Dk, 1))
        v_t = ttnn.slice(v_pairs_all, (s, 0, 0), (e, 1, Dv))
        g_t = ttnn.slice(g_pairs_all, (s, 0, 0), (e, 1, 1))
        beta_t = ttnn.slice(beta_pairs_all, (s, 0, 0), (e, 1, 1))

        out_t = gdn_recurrence_ttnn(q_t, k_row_t, k_col_t, v_t, g_t, beta_t, state_l1)
        token_outs.append(out_t)

        for tensor in (q_t, k_row_t, k_col_t, v_t, g_t, beta_t):
            ttnn.deallocate(tensor)

    ttnn.copy(state_l1, state)
    ttnn.deallocate(state_l1)

    for precomp in (q_pairs_all, k_row_all, k_col_all, v_pairs_all, g_pairs_all, beta_pairs_all):
        ttnn.deallocate(precomp)

    # =========================================================
    # Rearrange token_outs → output [num_pairs * N, 1, Dv]
    # (identical to gdn_prefill_ttnn)
    # =========================================================
    stacked = ttnn.concat(token_outs, dim=0)
    for o in token_outs:
        ttnn.deallocate(o)

    stacked_rm = ttnn.to_layout(stacked, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(stacked)
    tok_outer = ttnn.reshape(stacked_rm, (N, num_pairs, Dv))
    pair_outer = ttnn.permute(tok_outer, (1, 0, 2))
    ttnn.deallocate(tok_outer)
    ttnn.deallocate(stacked_rm)

    flat_rm = ttnn.reshape(pair_outer, (num_pairs * N, 1, Dv))
    flat = ttnn.to_layout(flat_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(flat_rm)
    ttnn.deallocate(pair_outer)

    ttnn.copy(flat, output)
    ttnn.deallocate(flat)
