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


def conv_weight_taps(weight_dhwk, device):
    """Pre-slice a depthwise conv weight [D, kernel] into K device taps [1,1,D] (fp32, TILE)."""
    D, kernel = weight_dhwk.shape
    import torch

    return [
        ttnn.from_torch(weight_dhwk[:, k].reshape(1, 1, D).contiguous().to(torch.float32),
                        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        for k in range(kernel)
    ]


def causal_conv1d_silu_ttnn(x, weight_taps, kernel_size, device):
    """Depthwise causal conv1d + SiLU, on device, via K shifted multiply-accumulate slices (FIR).

    Mirrors the in-tree GDN reference (ttnn_gated_deltanet.py::_causal_conv1d_fir) — the path the
    model uses for D>2048 (KDA's real key_dim=4096). Works for any T and D; no L1-scratch or
    native-conv1d shape constraints. x: [B,T,D] (TILE, fp32); weight_taps: K tensors [1,1,D].
    Returns [B,T,D] fp32.
    """
    B, T, D = x.shape
    pad = ttnn.zeros([B, kernel_size - 1, D], device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    x_padded = ttnn.concat([pad, x], dim=1)  # [B, T+K-1, D]

    out = None
    for k in range(kernel_size):
        x_slice = x_padded[:, k : k + T]
        if k != 0:  # only k=0 slice is tile-aligned; re-tilize the rest
            x_slice = ttnn.to_layout(x_slice, ttnn.TILE_LAYOUT)
        out = ttnn.multiply(x_slice, weight_taps[k]) if out is None else ttnn.addcmul(out, x_slice, weight_taps[k])
    return ttnn.silu(out)


def _conv1d_compute_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )


_CONV1D_CFG = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)


def prepare_conv1d_weight(w_bf16_oihw, C, kernel_size, T, device):
    """Prepare a depthwise conv1d weight once (host reprocess) for reuse across forwards. w_bf16_oihw is a
    ttnn ROW_MAJOR bf16 tensor [C,1,1,K] (channel-sharded on a mesh). Lin = (K-1)+T is the padded length."""
    Lin = (kernel_size - 1) + T
    return ttnn.prepare_conv_weights(
        weight_tensor=w_bf16_oihw, input_memory_config=ttnn.DRAM_MEMORY_CONFIG, input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weights_format="OIHW", in_channels=C, out_channels=C, batch_size=1, input_height=1, input_width=Lin,
        kernel_size=(1, kernel_size), stride=(1, 1), padding=(0, 0), dilation=(1, 1), has_bias=False, groups=C,
        device=device, input_dtype=ttnn.bfloat16, conv_config=_CONV1D_CFG, compute_config=_conv1d_compute_config(device),
    )


def causal_conv1d_silu_native(x, w_prep, kernel_size, C, device):
    """Depthwise causal conv1d + SiLU via native ttnn.conv1d (prefill path). Mirrors qwen36 gdn/tp.py.

    x: [1,T,C] (TILE, fp32). w_prep: prepared depthwise weight (see prepare_conv1d_weight). Returns [1,T,C]
    TILE fp32. Zero left-pad (K-1) => causal, single chunk (no cross-chunk carry). SiLU stays SEPARATE
    (folding into conv activation drops PCC on depthwise). ~4x faster / far fewer programs than the FIR.
    """
    _dram = ttnn.DRAM_MEMORY_CONFIG
    T = x.shape[1]
    Lin = (kernel_size - 1) + T
    pad = ttnn.zeros([1, kernel_size - 1, C], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=_dram)
    xin = ttnn.concat([pad, ttnn.typecast(x, ttnn.bfloat16)], dim=1, memory_config=_dram)
    ttnn.deallocate(pad)
    xin = ttnn.to_layout(xin, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
    xin = ttnn.reshape(xin, (1, Lin, 1, C))
    out = ttnn.conv1d(
        input_tensor=xin, weight_tensor=w_prep, device=device, in_channels=C, out_channels=C, batch_size=1,
        input_length=Lin, kernel_size=kernel_size, stride=1, padding=0, dilation=1, groups=C, dtype=ttnn.bfloat16,
        conv_config=_CONV1D_CFG, compute_config=_conv1d_compute_config(device), slice_config=ttnn.Conv2dL1FullSliceConfig,
        return_output_dim=False, return_weights_and_bias=False,
    )
    ttnn.deallocate(xin)
    out = ttnn.sharded_to_interleaved(out, _dram)
    out = ttnn.reshape(out, (1, T, C))
    out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=_dram)
    out = ttnn.silu(out, memory_config=_dram)
    return ttnn.typecast(out, ttnn.float32)  # back to fp32 for the downstream (l2norm / chunk_kda) path


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
