# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One GatedDeltaNet layer's chunked prefill (whole 128-token chunks, no padding) with the q/k/v head relayouts
removed: the conv output [1, T, q | k | v] goes straight into the fused chunk preprocessing (chunk_prep.ChunkPrep,
token-major mode), which L2-normalises q and k and maps each value head to its key head itself. The shared path
(gated_deltanet_forward_ttnn + chunk_gated_delta_rule_seq_adapter) untilizes, permutes, repeat-interleaves and
re-tilizes q, k and v first. Everything else follows the shared mega-fused path op for op."""

import os

import ttnn
from models.demos.blackhole.bonsai2.tt.gdn_prefill import chunk_prep

C = 128


def supported(gdn, x, valid_len):
    T = list(x.shape)[1]
    return (
        list(x.shape)[0] == 1
        and T % C == 0
        and (valid_len is None or valid_len >= T)
        and gdn.weights.mega_fused_weight is not None
        and gdn.weights.fused_conv_bias_dev is None
        and gdn.head_k_dim == 128
        and gdn.head_v_dim == 128
        and os.environ.get("QWEN_GDN_INV_DOUBLING", "0") == "0"
    )


def _to_bht(t, B, T, H):
    """[B, T, H] -> [B * H, T] fp32 TILE (as the shared adapter's _to_bht)."""
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    t = ttnn.reshape(t, [B, T, H])
    t = ttnn.permute(t, (0, 2, 1))
    t = ttnn.reshape(t, [B * H, T])
    return ttnn.to_layout(t, ttnn.TILE_LAYOUT, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def gdn_chunk_forward(gdn, x, linear_shim):
    """x [1, T, dim] -> the layer output; updates gdn.recurrent_state and gdn.fused_conv_state in place.
    linear_shim: the qwen36 prefill ttnn shim (its linear() adds the swept in_proj / out_proj program configs)."""
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import (
        _causal_conv1d_fir,
        rms_norm_gated_ttnn,
    )

    w = gdn.weights
    B, T = list(x.shape)[0], list(x.shape)[1]
    Hq, H, K, V = gdn.num_heads, gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim
    dq, da, db = w.mega_qkv_dim, w.mega_a_dim, w.mega_b_dim
    ckc = gdn.compute_kernel_config
    device = gdn.device

    mega = linear_shim.linear(x, w.mega_fused_weight, memory_config=None, compute_kernel_config=ckc)
    qkv = ttnn.to_layout(mega[:, :, :dq], ttnn.TILE_LAYOUT)
    a_raw = ttnn.to_layout(mega[:, :, dq : dq + da], ttnn.TILE_LAYOUT)
    b_raw = ttnn.to_layout(mega[:, :, dq + da : dq + da + db], ttnn.TILE_LAYOUT)
    gate_raw = ttnn.to_layout(mega[:, :, dq + da + db :], ttnn.TILE_LAYOUT)
    ttnn.deallocate(mega)

    qkv, new_conv = _causal_conv1d_fir(
        qkv,
        None,
        None,
        gdn.conv_kernel_size,
        device,
        memory_config=None,
        conv_state=gdn.fused_conv_state,
        weight_taps=w.fused_conv_weight_taps,
        bias_dev=None,
        valid_len=None,
    )

    beta = ttnn.sigmoid(b_raw)
    g = ttnn.multiply(w.A_neg, ttnn.softplus(ttnn.add(a_raw, w.dt_bias)))
    for t in (a_raw, b_raw):
        ttnn.deallocate(t)
    beta_bh = ttnn.reshape(_to_bht(beta, B, T, H), [B * H, T, 1])
    g_bh = _to_bht(g, B, T, H)

    if "device" not in chunk_prep._PREP:
        chunk_prep._PREP["device"] = chunk_prep.ChunkPrep(device)
    tm = dict(
        Wt=dq // 32,
        rf=H // Hq,
        q_off=0,
        k_off=Hq * K // 32,
        v_off=2 * Hq * K // 32,
        eps=1e-6,
        scale=K**-0.5,
    )
    alpha = float(os.environ.get("QWEN_GDN_DIAG_ALPHA", "0.25"))
    prep = chunk_prep._PREP["device"](qkv, qkv, qkv, beta_bh, g_bh, alpha, token_major=tm)
    ttnn.deallocate(qkv)
    s0 = ttnn.typecast(
        ttnn.reshape(gdn.recurrent_state, [B * H, K, V]), ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    o_bh, final_state = ttnn.transformer.gated_delta_attn_seq(
        *prep, initial_state=s0, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for t in prep:
        ttnn.deallocate(t)

    # o [BH, T, V] -> [B, T, H, V], as the shared adapter
    o = ttnn.to_layout(o_bh, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    o = ttnn.reshape(o, [B, H, T, V])
    o = ttnn.permute(o, (0, 2, 1, 3))
    o = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gate = ttnn.reshape(gate_raw, [B, T, H, V])
    o = rms_norm_gated_ttnn(o, gate, w.o_norm_weight, eps=gdn.norm_eps, memory_config=None)
    o = ttnn.clip(o, min=-1e4, max=1e4)
    o = ttnn.reshape(o, [B, T, H * V])
    out = linear_shim.linear(o, w.o_proj_weight, memory_config=None, compute_kernel_config=ckc)

    new_state = ttnn.typecast(ttnn.reshape(final_state, [B, H, K, V]), ttnn.bfloat16)
    ttnn.copy(new_state, gdn.recurrent_state)
    ttnn.deallocate(new_state)
    if new_conv.layout != ttnn.TILE_LAYOUT:
        new_conv = ttnn.to_layout(new_conv, ttnn.TILE_LAYOUT)
    ttnn.copy(new_conv, gdn.fused_conv_state)
    ttnn.deallocate(new_conv)
    return out
