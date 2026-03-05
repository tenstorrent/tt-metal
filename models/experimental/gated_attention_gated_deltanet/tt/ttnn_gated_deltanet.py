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


def rms_norm_gated_ttnn(x, gate, weight, eps=1e-5):
    """RMSNorm with SiLU gating using TTNN ops."""
    x_sq = ttnn.multiply(x, x)
    variance = ttnn.mean(x_sq, dim=-1, keepdim=True)
    inv_rms = ttnn.rsqrt(ttnn.add(variance, eps))
    x_normed = ttnn.multiply(x, inv_rms)
    x_normed = ttnn.multiply(x_normed, weight)
    gate_act = ttnn.silu(gate)
    return ttnn.multiply(x_normed, gate_act)


def rms_norm_ttnn(x, weight, eps=1e-5):
    """Standard RMSNorm using TTNN ops."""
    x_sq = ttnn.multiply(x, x)
    variance = ttnn.mean(x_sq, dim=-1, keepdim=True)
    inv_rms = ttnn.rsqrt(ttnn.add(variance, eps))
    x_normed = ttnn.multiply(x, inv_rms)
    return ttnn.multiply(x_normed, weight)


def _causal_conv1d_fir(x, weight, bias, kernel_size, device):
    """
    Manual FIR decomposition of depthwise causal conv1d + SiLU.

    Used for large T where native ttnn.conv1d would OOM in L1.
    Decomposes the convolution into K element-wise multiply+accumulate
    operations on shifted slices.

    Args / Returns: same as causal_conv1d_ttnn.
    """
    B, T, D = x.shape[0], x.shape[1], x.shape[2]

    pad = ttnn.zeros([B, kernel_size - 1, D], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x_padded = ttnn.concat([pad, x], dim=1)  # [B, T+K-1, D]

    weight_torch = ttnn.to_torch(weight)  # [D, 1, K]

    out = None
    for k in range(kernel_size):
        x_slice = x_padded[:, k : k + T]
        x_slice = ttnn.to_layout(x_slice, ttnn.TILE_LAYOUT)

        w_k = weight_torch[:, 0, k].reshape(1, 1, D).contiguous()
        w_k_dev = ttnn.from_torch(w_k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        term = ttnn.multiply(x_slice, w_k_dev)
        out = term if out is None else ttnn.add(out, term)

    if bias is not None:
        bias_torch = ttnn.to_torch(bias).reshape(1, 1, D).contiguous()
        bias_dev = ttnn.from_torch(bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.add(out, bias_dev)

    return ttnn.silu(out)


def causal_conv1d_ttnn(x, weight, bias, kernel_size, device, max_conv_len=512):
    """
    Depthwise causal conv1d + SiLU using native ttnn.conv1d.

    Falls back to manual FIR decomposition for T > max_conv_len to
    avoid L1 OOM in the conv1d kernel.

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
        return _causal_conv1d_fir(x, weight, bias, kernel_size, device)

    # conv1d requires ROW_MAJOR NLC input
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    # Causal left-padding: prepend (K-1) zeros along the time dim
    pad_zeros = ttnn.zeros([B, kernel_size - 1, D], device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    x_padded = ttnn.concat([pad_zeros, x_rm], dim=1)  # [B, T+K-1, D]

    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        deallocate_activation=True,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
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
    out = ttnn.sharded_to_interleaved(out)
    out = ttnn.reshape(out, [B, T, D])
    out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
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

    # 1. Linear projections
    q = ttnn.linear(hidden_states, q_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.linear(hidden_states, k_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.linear(hidden_states, v_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)

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
    beta = ttnn.sigmoid(ttnn.linear(hidden_states, b_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG))
    if allow_neg_eigval:
        beta = ttnn.multiply(beta, 2.0)

    a = ttnn.linear(hidden_states, a_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    # g = -A * softplus(a + dt_bias)
    a_biased = ttnn.add(a, dt_bias)
    sp = ttnn.softplus(a_biased)
    A = ttnn.exp(A_log)
    A_neg = ttnn.neg(A)
    g = ttnn.multiply(A_neg, sp)

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

    # 6. Output normalization
    if use_gate and g_proj_weight is not None:
        gate = ttnn.linear(hidden_states, g_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
        gate = ttnn.reshape(gate, [B, T, num_v_heads, head_v_dim])
        o = rms_norm_gated_ttnn(o, gate, o_norm_weight, eps=norm_eps)
    else:
        o = rms_norm_ttnn(o, o_norm_weight, eps=norm_eps)

    # 7. Reshape and project output
    o = ttnn.reshape(o, [B, T, num_v_heads * head_v_dim])
    o = ttnn.linear(o, o_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)

    return o, new_state
