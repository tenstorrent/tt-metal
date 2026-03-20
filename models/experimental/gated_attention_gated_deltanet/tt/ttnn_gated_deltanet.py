# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the Gated DeltaNet layer.
"""

import ttnn

from tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_ttnn,
    chunk_gated_delta_rule_ttnn,
)


def rms_norm_gated_ttnn(x, gate, weight, eps=1e-5, memory_config=ttnn.L1_MEMORY_CONFIG):
    """RMSNorm with SiLU gating using TTNN ops."""
    x_sq = ttnn.multiply(x, x, memory_config=memory_config)
    variance = ttnn.mean(x_sq, dim=-1, keepdim=True, memory_config=memory_config)
    inv_rms = ttnn.rsqrt(ttnn.add(variance, eps, memory_config=memory_config), memory_config=memory_config)
    x_normed = ttnn.multiply(x, inv_rms, memory_config=memory_config)
    x_normed = ttnn.multiply(x_normed, weight, memory_config=memory_config)
    gate_act = ttnn.silu(gate, memory_config=memory_config)
    return ttnn.multiply(x_normed, gate_act, memory_config=memory_config)


def rms_norm_ttnn(x, weight, eps=1e-5, memory_config=ttnn.L1_MEMORY_CONFIG):
    """Standard RMSNorm using TTNN ops."""
    x_sq = ttnn.multiply(x, x, memory_config=memory_config)
    variance = ttnn.mean(x_sq, dim=-1, keepdim=True, memory_config=memory_config)
    inv_rms = ttnn.rsqrt(ttnn.add(variance, eps, memory_config=memory_config), memory_config=memory_config)
    x_normed = ttnn.multiply(x, inv_rms, memory_config=memory_config)
    return ttnn.multiply(x_normed, weight, memory_config=memory_config)


def _causal_conv1d_fir(x, weight, bias, kernel_size, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """
    Manual FIR decomposition of depthwise causal conv1d + SiLU.

    Used for large T where native ttnn.conv1d would OOM in L1.
    Decomposes the convolution into K element-wise multiply+accumulate
    operations on shifted slices. Uses DRAM by default for large-T buffers.
    """
    B, T, D = x.shape[0], x.shape[1], x.shape[2]

    pad = ttnn.zeros(
        [B, kernel_size - 1, D],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )
    x_padded = ttnn.concat([pad, x], dim=1, memory_config=memory_config)  # [B, T+K-1, D]

    weight_torch = ttnn.to_torch(weight)  # [D, 1, K]

    out = None
    for k in range(kernel_size):
        x_slice = x_padded[:, k : k + T]
        x_slice = ttnn.to_layout(x_slice, ttnn.TILE_LAYOUT, memory_config=memory_config)

        w_k = weight_torch[:, 0, k].reshape(1, 1, D).contiguous()
        w_k_dev = ttnn.from_torch(
            w_k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        term = ttnn.multiply(x_slice, w_k_dev, memory_config=memory_config)
        out = term if out is None else ttnn.add(out, term, memory_config=memory_config)

    if bias is not None:
        bias_torch = ttnn.to_torch(bias).reshape(1, 1, D).contiguous()
        bias_dev = ttnn.from_torch(
            bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        out = ttnn.add(out, bias_dev, memory_config=memory_config)

    return ttnn.silu(out, memory_config=memory_config)


def causal_conv1d_ttnn(x, weight, bias, kernel_size, device, max_conv_len=128):
    """
    Depthwise causal conv1d + SiLU using native ttnn.conv1d.

    Falls back to manual FIR decomposition for T > max_conv_len to
    avoid L1 OOM in the conv1d kernel.     Default 128 keeps T<=128 on native path while avoiding L1 OOM at T>=256.

    Args:
        x: [B, T, D] input (TILE_LAYOUT on device)
        weight: conv weight [D, 1, K] (on host, ROW_MAJOR)
        bias: conv bias [D] or None
        kernel_size: int
        device: ttnn device
        max_conv_len: T threshold; above this, use FIR fallback

    Returns:
        output: [B, T, D] (TILE_LAYOUT on device)
    """
    B, T, D = x.shape[0], x.shape[1], x.shape[2]

    if T > max_conv_len:
        return _causal_conv1d_fir(x, weight, bias, kernel_size, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # conv1d requires ROW_MAJOR NLC input
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    # Causal left-padding: prepend (K-1) zeros along the time dim
    pad_zeros = ttnn.zeros(
        [B, kernel_size - 1, D],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x_padded = ttnn.concat([pad_zeros, x_rm], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)  # [B, T+K-1, D]

    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        deallocate_activation=True,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        # Optimization: Store config tensors in DRAM to reduce L1_SMALL pressure
        # This can help with L1 memory management and reduce fragmentation
        config_tensors_in_dram=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
    )

    [out, out_length, _] = ttnn.conv1d(
        input_tensor=x_padded,
        weight_tensor=weight,
        in_channels=D,
        out_channels=D,
        device=device,
        bias_tensor=bias,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        batch_size=B,
        input_length=T + kernel_size - 1,
        groups=D,
        dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    # Convert from HEIGHT_SHARDED to interleaved, reshape, then TILE_LAYOUT
    out = ttnn.sharded_to_interleaved(out, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.reshape(out, [B, T, D])
    out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    return out


def gated_deltanet_forward_ttnn(
    hidden_states,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    a_proj_weight,
    b_proj_weight,
    o_proj_weight,
    q_conv_weight,
    k_conv_weight,
    v_conv_weight,
    q_conv_bias,
    k_conv_bias,
    v_conv_bias,
    A_log,
    dt_bias,
    o_norm_weight,
    g_proj_weight=None,
    num_heads=4,
    num_v_heads=None,
    head_k_dim=256,
    head_v_dim=512,
    conv_kernel_size=4,
    use_gate=True,
    allow_neg_eigval=False,
    norm_eps=1e-5,
    device=None,
    recurrent_state=None,
    mode="recurrent",
    chunk_size=64,
):
    """
    TTNN forward pass for the Gated DeltaNet layer.

    Supports two modes:
      - "recurrent": token-by-token, best for decode (T=1)
      - "chunk": chunked parallel, best for prefill (T>1)

    Args:
        hidden_states: ttnn.Tensor [B, T, hidden_size]
        *_proj_weight: ttnn.Tensor weight matrices in [in_features, out_features] format
                       (transposed from PyTorch convention)
        *_conv_weight: ttnn.Tensor conv1d weights (NOT transposed)
        A_log: ttnn.Tensor [num_v_heads]
        dt_bias: ttnn.Tensor [num_v_heads]
        o_norm_weight: ttnn.Tensor [head_v_dim]
        g_proj_weight: ttnn.Tensor gate projection (if use_gate)
        device: ttnn device
        mode: "recurrent" or "chunk"
        chunk_size: chunk size for chunked mode

    Returns:
        output: ttnn.Tensor [B, T, hidden_size]
        new_state: ttnn.Tensor [B, H, K, V] updated recurrent state
    """
    if num_v_heads is None:
        num_v_heads = num_heads

    B = hidden_states.shape[0]
    T = hidden_states.shape[1]

    # Use DRAM for large-T to avoid L1 OOM (linear/conv outputs scale with T)
    T_DRAM_THRESHOLD = 128
    mem_cfg = ttnn.DRAM_MEMORY_CONFIG if T > T_DRAM_THRESHOLD else ttnn.L1_MEMORY_CONFIG

    # 1. Linear projections
    q = ttnn.linear(hidden_states, q_proj_weight, memory_config=mem_cfg)
    k = ttnn.linear(hidden_states, k_proj_weight, memory_config=mem_cfg)
    v = ttnn.linear(hidden_states, v_proj_weight, memory_config=mem_cfg)

    # 2. Causal conv1d + SiLU
    q = causal_conv1d_ttnn(q, q_conv_weight, q_conv_bias, conv_kernel_size, device)
    k = causal_conv1d_ttnn(k, k_conv_weight, k_conv_bias, conv_kernel_size, device)
    v = causal_conv1d_ttnn(v, v_conv_weight, v_conv_bias, conv_kernel_size, device)

    # 3. Reshape to multi-head
    q = ttnn.reshape(q, [B, T, num_heads, head_k_dim])
    k = ttnn.reshape(k, [B, T, num_heads, head_k_dim])
    v = ttnn.reshape(v, [B, T, num_v_heads, head_v_dim])

    # GVA: repeat q,k
    if num_v_heads > num_heads:
        repeats = num_v_heads // num_heads
        q = ttnn.repeat_interleave(q, repeats, dim=2)
        k = ttnn.repeat_interleave(k, repeats, dim=2)

    # 4. Compute beta and g
    beta = ttnn.sigmoid(
        ttnn.linear(hidden_states, b_proj_weight, memory_config=mem_cfg),
        memory_config=mem_cfg,
    )
    if allow_neg_eigval:
        beta = ttnn.multiply(beta, 2.0, memory_config=mem_cfg)

    a = ttnn.linear(hidden_states, a_proj_weight, memory_config=mem_cfg)
    # g = -A * softplus(a + dt_bias)
    a_biased = ttnn.add(a, dt_bias, memory_config=mem_cfg)
    sp = ttnn.softplus(a_biased, memory_config=mem_cfg)
    A = ttnn.exp(A_log, memory_config=ttnn.L1_MEMORY_CONFIG)
    A_neg = ttnn.neg(A, memory_config=ttnn.L1_MEMORY_CONFIG)
    g = ttnn.multiply(A_neg, sp, memory_config=mem_cfg)

    # 5. Gated delta rule (recurrent or chunked)
    if mode == "chunk":
        o, new_state = chunk_gated_delta_rule_ttnn(
            q=q,
            k=k,
            v=v,
            beta=beta,
            g=g,
            chunk_size=chunk_size,
            initial_state=recurrent_state,
            device=device,
            mem_cfg_large_t=mem_cfg,
        )
    else:
        o, new_state = recurrent_gated_delta_rule_ttnn(
            q=q,
            k=k,
            v=v,
            beta=beta,
            g=g,
            initial_state=recurrent_state,
            device=device,
        )

    # 6. Output normalization (use DRAM for large T to avoid L1 OOM)
    if use_gate and g_proj_weight is not None:
        gate = ttnn.linear(hidden_states, g_proj_weight, memory_config=mem_cfg)
        gate = ttnn.reshape(gate, [B, T, num_v_heads, head_v_dim])
        o = rms_norm_gated_ttnn(o, gate, o_norm_weight, eps=norm_eps, memory_config=mem_cfg)
    else:
        o = rms_norm_ttnn(o, o_norm_weight, eps=norm_eps, memory_config=mem_cfg)

    # 7. Reshape and project output
    o = ttnn.reshape(o, [B, T, num_v_heads * head_v_dim])
    o = ttnn.linear(o, o_proj_weight, memory_config=mem_cfg)

    return o, new_state
