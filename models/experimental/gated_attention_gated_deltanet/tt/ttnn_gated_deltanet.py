# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the Gated DeltaNet layer.
"""

import ttnn

from tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_ttnn,
    recurrent_gated_delta_rule_decode_ttnn,
    recurrent_gated_delta_rule_decode_inplace_ttnn,
    chunk_gated_delta_rule_ttnn,
)

_L1_SEQ_THRESHOLD = 512


def _seq_memory_config(seq_len):
    """L1 for short sequences (faster), DRAM for long (avoids OOM)."""
    return ttnn.L1_MEMORY_CONFIG if seq_len <= _L1_SEQ_THRESHOLD else None


def rms_norm_gated_ttnn(x, gate, weight, eps=1e-5, memory_config=None):
    """RMSNorm with SiLU gating — fused ttnn.rms_norm (trace-compatible, no try/except)."""
    mc = memory_config
    x_normed = ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=mc)
    gate_act = ttnn.silu(gate, memory_config=mc)
    return ttnn.multiply(x_normed, gate_act, memory_config=mc)


def rms_norm_ttnn(x, weight, eps=1e-5, memory_config=None):
    """Standard RMSNorm — fused ttnn.rms_norm (trace-compatible, no try/except)."""
    return ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=memory_config)


def _causal_conv1d_decode_t1_split(
    x, conv_state_list, kernel_size, device, memory_config=None, weight_taps=None, bias_dev=None
):
    """
    Optimized conv1d + SiLU for T=1 decode with split conv state.

    Like _causal_conv1d_decode_t1 but takes conv_state as a list of [B, 1, D] tensors
    instead of a single [B, kernel_size-1, D] tensor, eliminating slice operations.

    Returns:
        output: [B, 1, D] (TILE_LAYOUT on device)
        new_state_list: list of [B, 1, D] ttnn tensors on device
    """
    mc = memory_config

    # Compute output: weighted sum of state positions and current input
    out = ttnn.multiply(x, weight_taps[kernel_size - 1], memory_config=mc)
    for k in range(kernel_size - 1):
        term = ttnn.multiply(conv_state_list[k], weight_taps[k], memory_config=mc)
        out = ttnn.add(out, term, memory_config=mc)

    if bias_dev is not None:
        out = ttnn.add(out, bias_dev, memory_config=mc)

    # New state: drop oldest (index 0), shift others left, append current input
    new_state_list = conv_state_list[1:] + [x]

    return ttnn.silu(out, memory_config=mc), new_state_list


def _causal_conv1d_decode_t1(x, conv_state, kernel_size, device, memory_config=None, weight_taps=None, bias_dev=None):
    """
    Optimized conv1d + SiLU for T=1 decode with conv state.

    For T=1 with conv_state [B, kernel_size-1, D], we know exactly which
    positions map to which kernel taps, avoiding concat + slice overhead.

    tap 0 → state[:, 0:1, :] (oldest)
    tap 1 → state[:, 1:2, :]
    tap 2 → state[:, 2:3, :] (newest in state)
    tap 3 → x               (current input)

    Returns:
        output: [B, 1, D] (TILE_LAYOUT on device)
        new_state: [B, kernel_size-1, D] ttnn tensor on device
    """
    mc = memory_config

    # Compute output: weighted sum of state positions and current input
    # Taps 0..kernel_size-2 come from state, tap kernel_size-1 comes from x
    out = ttnn.multiply(x, weight_taps[kernel_size - 1], memory_config=mc)
    for k in range(kernel_size - 1):
        s_k = conv_state[:, k : k + 1, :]
        s_k = ttnn.to_layout(s_k, ttnn.TILE_LAYOUT)
        term = ttnn.multiply(s_k, weight_taps[k], memory_config=mc)
        out = ttnn.add(out, term, memory_config=mc)

    if bias_dev is not None:
        out = ttnn.add(out, bias_dev, memory_config=mc)

    # New state: drop oldest, append current input
    new_state = ttnn.concat([conv_state[:, 1:, :], x], dim=1, memory_config=mc)
    new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT)

    return ttnn.silu(out, memory_config=mc), new_state


def _causal_conv1d_decode_t1_inplace(
    x, conv_buffer, kernel_size, device, memory_config=None, weight_taps=None, bias_dev=None
):
    """Conv1d decode with copy-back to pre-allocated buffer for trace capture.

    conv_buffer: pre-allocated [B, kernel_size-1, D] tensor (fixed address).
    Computes output, then updates conv_buffer in-place via ttnn.copy().
    """
    mc = memory_config

    # Compute output: weighted sum of state positions and current input
    out = ttnn.multiply(x, weight_taps[kernel_size - 1], memory_config=mc)
    for k in range(kernel_size - 1):
        s_k = conv_buffer[:, k : k + 1, :]
        s_k = ttnn.to_layout(s_k, ttnn.TILE_LAYOUT)
        term = ttnn.multiply(s_k, weight_taps[k], memory_config=mc)
        out = ttnn.add(out, term, memory_config=mc)

    if bias_dev is not None:
        out = ttnn.add(out, bias_dev, memory_config=mc)

    # Update conv buffer in-place: shift left and append x
    new_state = ttnn.concat([conv_buffer[:, 1:, :], x], dim=1, memory_config=mc)
    new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT)
    ttnn.copy(new_state, conv_buffer)
    ttnn.deallocate(new_state)

    return ttnn.silu(out, memory_config=mc), conv_buffer


def _causal_conv1d_fir(
    x, weight, bias, kernel_size, device, memory_config=None, conv_state=None, weight_taps=None, bias_dev=None
):
    """
    Manual FIR decomposition of depthwise causal conv1d + SiLU.

    Decomposes the convolution into K element-wise multiply+accumulate
    operations on shifted slices.

    Args:
        x: [B, T, D] input (TILE_LAYOUT on device)
        weight: conv weight [D, 1, K] (used only if weight_taps is None)
        bias: conv bias [D] or None (used only if bias_dev is None)
        kernel_size: int
        device: ttnn device
        memory_config: optional memory config
        conv_state: [B, kernel_size-1, D] previous state — ttnn tensor on device,
                    or list of [B, 1, D] tensors (split state), or None
        weight_taps: optional list of K pre-sliced [1, 1, D] ttnn tensors on device
        bias_dev: optional pre-converted [1, 1, D] ttnn tensor on device

    Returns:
        output: [B, T, D] (TILE_LAYOUT on device)
        new_state: [B, kernel_size-1, D] ttnn tensor on device
    """
    mc = memory_config
    B, T, D = x.shape[0], x.shape[1], x.shape[2]

    # Fast path for T=1 decode with existing state
    if T == 1 and conv_state is not None and weight_taps is not None:
        return _causal_conv1d_decode_t1(
            x, conv_state, kernel_size, device, memory_config=mc, weight_taps=weight_taps, bias_dev=bias_dev
        )

    if conv_state is not None:
        x_padded = ttnn.concat([conv_state, x], dim=1, memory_config=mc)
    else:
        pad = ttnn.zeros(
            [B, kernel_size - 1, D],
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mc,
        )
        x_padded = ttnn.concat([pad, x], dim=1, memory_config=mc)

    # New conv state: last kernel_size-1 tokens from x_padded, kept on device
    total_len = (kernel_size - 1) + T
    start = total_len - (kernel_size - 1)
    new_state = x_padded[:, start:, :]
    new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT)

    # Precompute weight taps if not provided
    if weight_taps is None:
        pass

        weight_torch = ttnn.to_torch(weight)
        weight_taps = []
        for k in range(kernel_size):
            w_k = weight_torch[:, 0, k].reshape(1, 1, D).contiguous()
            weight_taps.append(
                ttnn.from_torch(w_k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            )

    total_len = (kernel_size - 1) + T
    out = None
    for k in range(kernel_size):
        x_slice = x_padded[:, k : k + T]
        x_slice = ttnn.to_layout(x_slice, ttnn.TILE_LAYOUT)
        term = ttnn.multiply(x_slice, weight_taps[k], memory_config=mc)
        out = term if out is None else ttnn.add(out, term, memory_config=mc)

    # Apply bias
    if bias_dev is not None:
        out = ttnn.add(out, bias_dev, memory_config=mc)
    elif bias is not None:
        pass

        bias_torch = ttnn.to_torch(bias).reshape(1, 1, D).contiguous()
        bias_dev_tmp = ttnn.from_torch(
            bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        )
        out = ttnn.add(out, bias_dev_tmp, memory_config=mc)

    return ttnn.silu(out, memory_config=mc), new_state


def causal_conv1d_ttnn(
    x,
    weight,
    bias,
    kernel_size,
    device,
    max_conv_len=512,
    memory_config=None,
    conv_state=None,
    weight_taps=None,
    bias_dev=None,
):
    """
    Depthwise causal conv1d + SiLU with conv state support.

    Args:
        x: [B, T, D] input (TILE_LAYOUT on device)
        weight: conv weight [D, 1, K] (on host, ROW_MAJOR)
        bias: conv bias [D] or None
        kernel_size: int
        device: ttnn device
        max_conv_len: T threshold; above this, use FIR fallback
        conv_state: [B, kernel_size-1, D] previous state (ttnn tensor on device) or None
        weight_taps: optional list of K pre-sliced [1, 1, D] ttnn tensors on device
        bias_dev: optional pre-converted [1, 1, D] ttnn tensor on device

    Returns:
        output: [B, T, D] (TILE_LAYOUT on device)
        new_state: [B, kernel_size-1, D] ttnn tensor on device
    """
    B, T, D = x.shape[0], x.shape[1], x.shape[2]
    mc = memory_config

    # Use FIR path when we have conv state (simpler to handle) or when T is large
    if conv_state is not None or T > max_conv_len:
        return _causal_conv1d_fir(
            x,
            weight,
            bias,
            kernel_size,
            device,
            memory_config=mc,
            conv_state=conv_state,
            weight_taps=weight_taps,
            bias_dev=bias_dev,
        )

    # No conv state — use native ttnn.conv1d with zero padding (prefill from scratch)
    if mc is not None:
        x = ttnn.to_memory_config(x, mc)
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    pad_zeros = ttnn.zeros(
        [B, kernel_size - 1, D],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=mc,
    )
    x_padded = ttnn.concat([pad_zeros, x_rm], dim=1, memory_config=mc)

    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        deallocate_activation=True,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
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

    out = ttnn.sharded_to_interleaved(out, memory_config=mc)
    out = ttnn.reshape(out, [B, T, D])
    out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=mc)

    # Save conv state: last kernel_size-1 tokens from input, kept on device
    # For prefill we need to get the state — use x directly since it's still on device
    x_tile = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    if T >= kernel_size - 1:
        new_state = x_tile[:, -(kernel_size - 1) :, :]
        new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT)
    else:
        pad_needed = kernel_size - 1 - T
        pad_state = ttnn.zeros(
            [B, pad_needed, D],
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mc,
        )
        new_state = ttnn.concat([pad_state, x_tile], dim=1, memory_config=mc)

    return out, new_state


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
    conv_state_q=None,
    conv_state_k=None,
    conv_state_v=None,
    mode="recurrent",
    chunk_size=64,
    q_weight_taps=None,
    k_weight_taps=None,
    v_weight_taps=None,
    q_bias_dev=None,
    k_bias_dev=None,
    v_bias_dev=None,
    qkv_proj_weight=None,
    q_dim=None,
    k_dim=None,
    compute_kernel_config=None,
    A_neg_precomputed=None,
    # Fused conv parameters (optimization for T=1 decode)
    fused_conv_weight_taps=None,
    fused_conv_bias_dev=None,
    fused_conv_state=None,
    # Fused a+b projection weight (optimization: 1 matmul instead of 2)
    ab_proj_weight=None,
    # Mega-fused weight: QKV + a + b + g in one matmul
    mega_fused_weight=None,
    mega_qkv_dim=None,
    mega_a_dim=None,
    mega_b_dim=None,
    mega_g_dim=None,
    # Pre-cached mask matrices for chunk mode (avoids recreating per call)
    cached_masks=None,
    # Trace capture support: inplace state updates via ttnn.copy()
    use_inplace_state=False,
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
    mc = _seq_memory_config(T)

    # 1. Linear projections — fused QKV when available (1 matmul instead of 3)
    ckc = compute_kernel_config

    # Mega-fused decode path: one matmul for QKV + a + b + g, fused conv on QKV
    use_mega_fused = (
        T == 1
        and mega_fused_weight is not None
        and fused_conv_weight_taps is not None
        and fused_conv_state is not None
        and mega_qkv_dim is not None
    )

    # Fused conv decode path: keep QKV concatenated through conv, split after
    use_fused_conv = (
        not use_mega_fused
        and T == 1
        and fused_conv_weight_taps is not None
        and fused_conv_state is not None
        and qkv_proj_weight is not None
        and q_dim is not None
    )

    if use_mega_fused:
        # Single matmul for everything: [B, 1, 4096] × [4096, 12352]
        mega_out = ttnn.linear(hidden_states, mega_fused_weight, memory_config=mc, compute_kernel_config=ckc)
        # Split: QKV | a | b | g
        qkv = mega_out[:, :, :mega_qkv_dim]
        qkv = ttnn.to_layout(qkv, ttnn.TILE_LAYOUT)
        a_raw = mega_out[:, :, mega_qkv_dim : mega_qkv_dim + mega_a_dim]
        a_raw = ttnn.to_layout(a_raw, ttnn.TILE_LAYOUT)
        b_raw = mega_out[:, :, mega_qkv_dim + mega_a_dim : mega_qkv_dim + mega_a_dim + mega_b_dim]
        b_raw = ttnn.to_layout(b_raw, ttnn.TILE_LAYOUT)
        gate_raw = mega_out[:, :, mega_qkv_dim + mega_a_dim + mega_b_dim :]
        gate_raw = ttnn.to_layout(gate_raw, ttnn.TILE_LAYOUT)
        ttnn.deallocate(mega_out)

        # Fused conv1d on QKV
        conv_fn = _causal_conv1d_decode_t1_inplace if use_inplace_state else _causal_conv1d_decode_t1
        qkv, new_fused_conv_state = conv_fn(
            qkv,
            fused_conv_state,
            conv_kernel_size,
            device,
            memory_config=mc,
            weight_taps=fused_conv_weight_taps,
            bias_dev=fused_conv_bias_dev,
        )
        # Split QKV after conv
        q = qkv[:, :, :q_dim]
        k = qkv[:, :, q_dim : q_dim + k_dim]
        v = qkv[:, :, q_dim + k_dim :]
        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
        ttnn.deallocate(qkv)
        new_conv_q = None
        new_conv_k = None
        new_conv_v = None

        # Pre-extracted a, b, g from mega projection
        _mega_extracted = True
    elif use_fused_conv:
        # Fused path: QKV projection -> fused conv -> split
        qkv = ttnn.linear(hidden_states, qkv_proj_weight, memory_config=mc, compute_kernel_config=ckc)
        # Fused conv1d on concatenated QKV [B, 1, D_total]
        conv_fn = _causal_conv1d_decode_t1_inplace if use_inplace_state else _causal_conv1d_decode_t1
        qkv, new_fused_conv_state = conv_fn(
            qkv,
            fused_conv_state,
            conv_kernel_size,
            device,
            memory_config=mc,
            weight_taps=fused_conv_weight_taps,
            bias_dev=fused_conv_bias_dev,
        )
        # Split after conv
        q = qkv[:, :, :q_dim]
        k = qkv[:, :, q_dim : q_dim + k_dim]
        v = qkv[:, :, q_dim + k_dim :]
        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
        ttnn.deallocate(qkv)
        new_conv_q = None
        new_conv_k = None
        new_conv_v = None
        _mega_extracted = False
    elif qkv_proj_weight is not None and q_dim is not None:
        qkv = ttnn.linear(hidden_states, qkv_proj_weight, memory_config=mc, compute_kernel_config=ckc)
        # Fused conv prefill path: run one conv on concatenated QKV, then split
        if T > 1 and fused_conv_weight_taps is not None:
            qkv = ttnn.to_layout(qkv, ttnn.TILE_LAYOUT)
            qkv, new_fused_conv_state_raw = _causal_conv1d_fir(
                qkv,
                None,
                None,
                conv_kernel_size,
                device,
                memory_config=mc,
                conv_state=fused_conv_state,
                weight_taps=fused_conv_weight_taps,
                bias_dev=fused_conv_bias_dev,
            )
            # Split after conv
            q = qkv[:, :, :q_dim]
            k = qkv[:, :, q_dim : q_dim + k_dim]
            v = qkv[:, :, q_dim + k_dim :]
            q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
            k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
            v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
            ttnn.deallocate(qkv)
            # Extract per-stream conv states from the fused state
            D_total = (
                q_dim + k_dim + (qkv_proj_weight.shape[-1] - q_dim - k_dim)
                if hasattr(qkv_proj_weight, "shape")
                else None
            )
            new_conv_q = new_fused_conv_state_raw[:, :, :q_dim]
            new_conv_q = ttnn.to_layout(new_conv_q, ttnn.TILE_LAYOUT)
            new_conv_k = new_fused_conv_state_raw[:, :, q_dim : q_dim + k_dim]
            new_conv_k = ttnn.to_layout(new_conv_k, ttnn.TILE_LAYOUT)
            new_conv_v = new_fused_conv_state_raw[:, :, q_dim + k_dim :]
            new_conv_v = ttnn.to_layout(new_conv_v, ttnn.TILE_LAYOUT)
            new_fused_conv_state = new_fused_conv_state_raw
            _mega_extracted = False
        else:
            q = qkv[:, :, :q_dim]
            k = qkv[:, :, q_dim : q_dim + k_dim]
            v = qkv[:, :, q_dim + k_dim :]
            q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
            k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
            v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
            ttnn.deallocate(qkv)
            new_fused_conv_state = None
            _mega_extracted = False
            q, new_conv_q = causal_conv1d_ttnn(
                q,
                q_conv_weight,
                q_conv_bias,
                conv_kernel_size,
                device,
                memory_config=mc,
                conv_state=conv_state_q,
                weight_taps=q_weight_taps,
                bias_dev=q_bias_dev,
            )
            k, new_conv_k = causal_conv1d_ttnn(
                k,
                k_conv_weight,
                k_conv_bias,
                conv_kernel_size,
                device,
                memory_config=mc,
                conv_state=conv_state_k,
                weight_taps=k_weight_taps,
                bias_dev=k_bias_dev,
            )
            v, new_conv_v = causal_conv1d_ttnn(
                v,
                v_conv_weight,
                v_conv_bias,
                conv_kernel_size,
                device,
                memory_config=mc,
                conv_state=conv_state_v,
                weight_taps=v_weight_taps,
                bias_dev=v_bias_dev,
            )
    else:
        q = ttnn.linear(hidden_states, q_proj_weight, memory_config=mc, compute_kernel_config=ckc)
        k = ttnn.linear(hidden_states, k_proj_weight, memory_config=mc, compute_kernel_config=ckc)
        v = ttnn.linear(hidden_states, v_proj_weight, memory_config=mc, compute_kernel_config=ckc)
        new_fused_conv_state = None
        _mega_extracted = False
        q, new_conv_q = causal_conv1d_ttnn(
            q,
            q_conv_weight,
            q_conv_bias,
            conv_kernel_size,
            device,
            memory_config=mc,
            conv_state=conv_state_q,
            weight_taps=q_weight_taps,
            bias_dev=q_bias_dev,
        )
        k, new_conv_k = causal_conv1d_ttnn(
            k,
            k_conv_weight,
            k_conv_bias,
            conv_kernel_size,
            device,
            memory_config=mc,
            conv_state=conv_state_k,
            weight_taps=k_weight_taps,
            bias_dev=k_bias_dev,
        )
        v, new_conv_v = causal_conv1d_ttnn(
            v,
            v_conv_weight,
            v_conv_bias,
            conv_kernel_size,
            device,
            memory_config=mc,
            conv_state=conv_state_v,
            weight_taps=v_weight_taps,
            bias_dev=v_bias_dev,
        )

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
    if _mega_extracted:
        # a_raw and b_raw already extracted from mega projection
        a = a_raw
        beta = ttnn.sigmoid(b_raw, memory_config=mc)
    elif ab_proj_weight is not None:
        ab = ttnn.linear(hidden_states, ab_proj_weight, memory_config=mc, compute_kernel_config=ckc)
        num_v = num_v_heads if num_v_heads is not None else num_heads
        a = ab[:, :, :num_v]
        a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
        b_raw = ab[:, :, num_v:]
        b_raw = ttnn.to_layout(b_raw, ttnn.TILE_LAYOUT)
        ttnn.deallocate(ab)
        beta = ttnn.sigmoid(b_raw, memory_config=mc)
    else:
        beta = ttnn.sigmoid(
            ttnn.linear(hidden_states, b_proj_weight, memory_config=mc, compute_kernel_config=ckc),
            memory_config=mc,
        )
        a = ttnn.linear(hidden_states, a_proj_weight, memory_config=mc, compute_kernel_config=ckc)
    if allow_neg_eigval:
        beta = ttnn.multiply(beta, 2.0, memory_config=mc)
    a_biased = ttnn.add(a, dt_bias, memory_config=mc)
    sp = ttnn.softplus(a_biased, memory_config=mc)
    if A_neg_precomputed is not None:
        g = ttnn.multiply(A_neg_precomputed, sp, memory_config=mc)
    else:
        A = ttnn.exp(A_log, memory_config=mc)
        A_neg = ttnn.neg(A, memory_config=mc)
        g = ttnn.multiply(A_neg, sp, memory_config=mc)

    # 5. Gated delta rule (recurrent or chunked)
    # Chunk mode uses float32 for state computation, recurrent uses bfloat16.
    # For prefill (T>1), always prefer chunk mode for precision — the chunk
    # implementation already handles T < chunk_size via padding.
    # For decode (T=1), use optimized decode path (fewer ops, no loop).
    if mode == "chunk" and T > 1:
        o, new_state = chunk_gated_delta_rule_ttnn(
            q=q,
            k=k,
            v=v,
            beta=beta,
            g=g,
            chunk_size=chunk_size,
            initial_state=recurrent_state,
            device=device,
            cached_masks=cached_masks,
        )
    elif T == 1:
        if use_inplace_state and recurrent_state is not None:
            o, new_state = recurrent_gated_delta_rule_decode_inplace_ttnn(
                q=q,
                k=k,
                v=v,
                beta=beta,
                g=g,
                state_buffer=recurrent_state,
                device=device,
            )
        else:
            o, new_state = recurrent_gated_delta_rule_decode_ttnn(
                q=q,
                k=k,
                v=v,
                beta=beta,
                g=g,
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
        if _mega_extracted:
            gate = ttnn.reshape(gate_raw, [B, T, num_v_heads, head_v_dim])
        else:
            gate = ttnn.linear(hidden_states, g_proj_weight, memory_config=mc, compute_kernel_config=ckc)
            gate = ttnn.reshape(gate, [B, T, num_v_heads, head_v_dim])
        o = rms_norm_gated_ttnn(o, gate, o_norm_weight, eps=norm_eps, memory_config=mc)
    else:
        o = rms_norm_ttnn(o, o_norm_weight, eps=norm_eps, memory_config=mc)

    # 7. Reshape and project output
    o = ttnn.reshape(o, [B, T, num_v_heads * head_v_dim])
    if mc is not None:
        o = ttnn.to_memory_config(o, mc)
    o = ttnn.linear(o, o_proj_weight, memory_config=mc, compute_kernel_config=ckc)

    return o, new_state, new_conv_q, new_conv_k, new_conv_v, new_fused_conv_state
