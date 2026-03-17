# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Functional torch implementation of the Gated DeltaNet layer.

Extracted from FLA (Flash Linear Attention) library:
  https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py
"""

import torch
import torch.nn.functional as F

from models.experimental.tt_symbiote.core.torch_deltarule_ops import recurrent_gated_delta_rule, chunk_gated_delta_rule


def causal_conv1d_forward(x, weight, bias=None, conv_state=None, kernel_size=4):
    """
    Depthwise causal 1D convolution + SiLU activation.

    Args:
        x: [B, T, D] input
        weight: [D, 1, K] depthwise conv weight
        bias: [D] optional bias
        conv_state: [B, D, K-1] previous state for incremental decoding
        kernel_size: convolution kernel size

    Returns:
        output: [B, T, D], new_conv_state: [B, D, K-1] or None
    """
    B, T, D = x.shape
    x_t = x.transpose(1, 2)  # [B, D, T]

    if conv_state is not None and T == 1:
        x_cat = torch.cat([conv_state, x_t], dim=-1)
        new_state = x_cat[:, :, -(kernel_size - 1) :]
        out = F.conv1d(x_cat, weight, bias, padding=0, groups=D)
        out = F.silu(out)
        return out.transpose(1, 2), new_state

    x_padded = F.pad(x_t, (kernel_size - 1, 0))
    out = F.conv1d(x_padded, weight, bias, padding=0, groups=D)
    out = F.silu(out)

    new_state = None
    if conv_state is not None:
        new_state = x_t[:, :, -(kernel_size - 1) :]

    return out.transpose(1, 2), new_state


def rms_norm_gated(x, gate, weight, eps=1e-5):
    """RMSNorm with SiLU gating: norm(x) * silu(gate)."""
    input_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    x_normed = weight.float() * x_normed
    x_normed = x_normed * F.silu(gate.to(torch.float32))
    return x_normed.to(input_dtype)


def rms_norm(x, weight, eps=1e-5):
    """Standard RMSNorm (without gating)."""
    input_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return (weight.float() * x_normed).to(input_dtype)


def gated_deltanet_forward(
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
    mode="chunk",
    chunk_size=64,
    conv_state_q=None,
    conv_state_k=None,
    conv_state_v=None,
    recurrent_state=None,
    output_final_state=False,
):
    """
    Functional forward pass for the Gated DeltaNet layer.

    Args:
        hidden_states: [B, T, hidden_size]
        *_proj_weight: linear projection weight matrices
        *_conv_weight: depthwise conv1d weights [D, 1, K]
        A_log: [num_v_heads] log of decay base
        dt_bias: [num_v_heads] bias for dt computation
        o_norm_weight: [head_v_dim] output norm weight
        g_proj_weight: gate projection weight (if use_gate)
        num_heads: number of K/Q heads
        num_v_heads: number of V heads (for GVA), defaults to num_heads
        head_k_dim: dimension per K/Q head
        head_v_dim: dimension per V head
        conv_kernel_size: causal conv1d kernel size
        use_gate: whether to use output gating
        allow_neg_eigval: double beta for negative eigenvalues
        norm_eps: RMSNorm epsilon
        mode: "chunk" or "fused_recurrent"
        chunk_size: chunk size for chunked mode
        conv_state_*: conv states for incremental decoding
        recurrent_state: [B, H, K, V] recurrent state
        output_final_state: whether to return updated states

    Returns:
        output: [B, T, hidden_size]
        cache: dict with updated states, or None
    """
    if num_v_heads is None:
        num_v_heads = num_heads

    B, T, _ = hidden_states.shape
    effective_mode = "fused_recurrent" if T <= 64 else mode

    # 1. Linear projections
    q = F.linear(hidden_states, q_proj_weight)
    k = F.linear(hidden_states, k_proj_weight)
    v = F.linear(hidden_states, v_proj_weight)

    # 2. Causal conv1d + SiLU
    q, new_conv_q = causal_conv1d_forward(q, q_conv_weight, q_conv_bias, conv_state_q, conv_kernel_size)
    k, new_conv_k = causal_conv1d_forward(k, k_conv_weight, k_conv_bias, conv_state_k, conv_kernel_size)
    v, new_conv_v = causal_conv1d_forward(v, v_conv_weight, v_conv_bias, conv_state_v, conv_kernel_size)

    # 3. Reshape to multi-head: [B, T, H, D]
    q = q.reshape(B, T, num_heads, head_k_dim)
    k = k.reshape(B, T, num_heads, head_k_dim)
    v = v.reshape(B, T, num_v_heads, head_v_dim)

    # GVA: repeat q,k if num_v_heads > num_heads
    if num_v_heads > num_heads:
        repeats = num_v_heads // num_heads
        q = q.repeat_interleave(repeats, dim=2)
        k = k.repeat_interleave(repeats, dim=2)

    # 4. Compute beta (write strength) and g (decay gate)
    beta = F.linear(hidden_states, b_proj_weight).sigmoid()
    if allow_neg_eigval:
        beta = beta * 2.0

    a = F.linear(hidden_states, a_proj_weight)
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)

    # 5. Gated delta rule
    if effective_mode == "chunk":
        o, new_recurrent = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            chunk_size=chunk_size,
            initial_state=recurrent_state,
            output_final_state=output_final_state,
            use_qk_l2norm=True,
        )
    else:
        o, new_recurrent = recurrent_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            beta=beta,
            g=g,
            initial_state=recurrent_state,
            output_final_state=output_final_state,
            use_qk_l2norm=True,
        )

    # 6. Output normalization
    if use_gate and g_proj_weight is not None:
        gate = F.linear(hidden_states, g_proj_weight)
        gate = gate.reshape(B, T, num_v_heads, head_v_dim)
        o = rms_norm_gated(o, gate, o_norm_weight, eps=norm_eps)
    else:
        o = rms_norm(o, o_norm_weight, eps=norm_eps)

    # 7. Reshape and project output
    o = o.reshape(B, T, num_v_heads * head_v_dim)
    o = F.linear(o, o_proj_weight)

    cache = None
    if output_final_state:
        cache = {
            "conv_state_q": new_conv_q,
            "conv_state_k": new_conv_k,
            "conv_state_v": new_conv_v,
            "recurrent_state": new_recurrent,
        }

    return o, cache
