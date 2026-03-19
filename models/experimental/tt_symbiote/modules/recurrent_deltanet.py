# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Optional

from ..core.module import TTNNModule

# Tile size used by TTNN matmul (wormhole)
_TILE_H = 32
_TILE_W = 32


def _recurrent_outer_product_program_config(device, K, V):
    """Build MatmulMultiCoreReuseProgramConfig for the outer-product matmul."""
    grid = device.compute_with_storage_grid_size()
    if hasattr(grid, "x"):
        grid_x, grid_y = int(grid.x), int(grid.y)
    else:
        grid_x, grid_y = int(grid[0]), int(grid[1])

    M_tiles = (K + _TILE_H - 1) // _TILE_H
    N_tiles = (V + _TILE_W - 1) // _TILE_W
    K_tiles_inner = max(1, (1 + _TILE_W - 1) // _TILE_W)  # 1

    # N == per_core_N (mandatory)
    per_core_N = N_tiles
    # in0_block_w: inner dim in tiles
    in0_block_w = K_tiles_inner

    # per_core_M must divide M_tiles
    per_core_M = M_tiles
    while per_core_M > 1 and (M_tiles % per_core_M != 0 or (grid_x * grid_y) < (M_tiles // per_core_M)):
        per_core_M -= 1
    if per_core_M < 1:
        per_core_M = 1

    # out_subblock must divide per_core; profiler suggests out_subblock_h * out_subblock_w >= 2
    out_subblock_h = min(2, per_core_M) if per_core_M >= 2 else 1
    out_subblock_w = min(2, per_core_N) if per_core_N >= 2 else 1

    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
    )


def _recurrent_read_query_program_config(device, K, V):
    """Build MatmulMultiCoreReuseProgramConfig for read/query matmuls."""
    grid = device.compute_with_storage_grid_size()
    N_tiles = (V + _TILE_W - 1) // _TILE_W
    K_tiles = (K + _TILE_W - 1) // _TILE_W

    per_core_N = N_tiles
    per_core_M = 1
    in0_block_w = K_tiles
    out_subblock_h = 1
    out_subblock_w = min(2, per_core_N) if per_core_N >= 2 else 1

    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
    )


def l2_norm_ttnn(x, dim=-1, eps=1e-6):
    """L2 normalization along a given dimension."""
    x_sq = ttnn.multiply(x, x, memory_config=ttnn.L1_MEMORY_CONFIG)
    norm_sq = ttnn.sum(x_sq, dim=dim, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
    inv_norm = ttnn.rsqrt(
        ttnn.add(norm_sq, eps, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.L1_MEMORY_CONFIG
    )
    return ttnn.multiply(x, inv_norm, memory_config=ttnn.L1_MEMORY_CONFIG)


def fused_decay_and_write_ttnn(
    h,
    k_t,
    delta,
    decay_t,
    beta_t,
    device=None,
    outer_product_prog_cfg=None,
    step_idx=0,
):
    """
    Logical fusion for the recurrent delta rule state update:
    h = decay * h + beta_t * (k_t ⊗ delta)
    """
    B = h.shape[0]
    H = h.shape[1]
    K = h.shape[2]
    V = h.shape[3]

    decay = ttnn.reshape(
        ttnn.typecast(decay_t, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG),
        [B, H, 1, 1],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    beta_expanded = ttnn.reshape(beta_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

    k_col = ttnn.to_layout(
        ttnn.reshape(k_t, [B, H, K, 1], memory_config=ttnn.L1_MEMORY_CONFIG),
        ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    d_row = ttnn.to_layout(
        ttnn.reshape(delta, [B, H, 1, V], memory_config=ttnn.L1_MEMORY_CONFIG),
        ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    if outer_product_prog_cfg is None and device is not None:
        try:
            outer_product_prog_cfg = _recurrent_outer_product_program_config(device, K, V)
        except Exception:
            pass

    matmul_compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    outer = ttnn.matmul(
        k_col,
        d_row,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=matmul_compute_cfg,
        program_config=outer_product_prog_cfg,
    )

    outer = ttnn.multiply(
        outer,
        beta_expanded,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    h = ttnn.multiply(h, decay, memory_config=ttnn.L1_MEMORY_CONFIG)
    h = ttnn.add(h, outer, memory_config=ttnn.L1_MEMORY_CONFIG)

    return h


def recurrent_delta_rule_step_ttnn(
    q_t,
    k_t,
    v_t,
    beta_t,
    decay_t,
    h,
    seq_len=None,
    device=None,
    step_idx=0,
    read_query_prog_cfg=None,
    read_query_compute_cfg=None,
    outer_product_prog_cfg=None,
):
    """Recurrent delta rule step using TTNN ops."""
    B = q_t.shape[0]
    H = q_t.shape[1]
    K = q_t.shape[2]
    V = v_t.shape[2]

    h = ttnn.to_layout(h, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Cache compute config if not provided
    if read_query_compute_cfg is None:
        read_query_compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # Cache program config if not provided
    if read_query_prog_cfg is None and device is not None:
        try:
            read_query_prog_cfg = _recurrent_read_query_program_config(device, K, V)
        except Exception:
            pass

    k_row = ttnn.to_layout(
        ttnn.reshape(k_t, [B, H, 1, K], memory_config=ttnn.L1_MEMORY_CONFIG),
        ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    v_prime = ttnn.matmul(
        k_row,
        h,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        program_config=read_query_prog_cfg,
        compute_kernel_config=read_query_compute_cfg,
    )
    v_prime = ttnn.reshape(v_prime, [B, H, V], memory_config=ttnn.L1_MEMORY_CONFIG)

    delta = ttnn.subtract(v_t, v_prime, memory_config=ttnn.L1_MEMORY_CONFIG)
    h = fused_decay_and_write_ttnn(
        h=h,
        k_t=k_t,
        delta=delta,
        decay_t=decay_t,
        beta_t=beta_t,
        device=device,
        outer_product_prog_cfg=outer_product_prog_cfg,
        step_idx=step_idx,
    )
    q_row = ttnn.to_layout(
        ttnn.reshape(q_t, [B, H, 1, K], memory_config=ttnn.L1_MEMORY_CONFIG),
        ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    o_t = ttnn.matmul(
        q_row,
        h,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        program_config=read_query_prog_cfg,
        compute_kernel_config=read_query_compute_cfg,
    )
    use_l1 = seq_len is not None and seq_len <= 64
    o_t = ttnn.reshape(o_t, [B, H, V], memory_config=ttnn.L1_MEMORY_CONFIG if use_l1 else None)

    return o_t, h


def recurrent_gated_delta_rule_ttnn(
    q,
    k,
    v,
    beta,
    g,
    initial_state=None,
    device=None,
    scale=None,
):
    """
    Token-by-token recurrent gated delta rule using TTNN ops.
    Used for decode (T=1).

    For each timestep t:
      1. Decay the state:  h = h * exp(g_t)
      2. Read from state:  v_read = sum_k(h * k_t)
      3. Compute delta:    delta = (v_t - v_read) * beta_t
      4. Write to state:   h = h + outer(k_t, delta)
      5. Query state:      o_t = h @ q_t

    Args:
        q: [B, T, H, K] (already L2 normalized)
        k: [B, T, H, K] (already L2 normalized)
        v: [B, T, H, V]
        beta: [B, T, H]
        g: [B, T, H]
        scale: float
        initial_state: [B, H, K, V]
        device: ttnn device

    Returns:
        output: [B, T, H, V]
        final_state: [B, H, K, V]
    """
    B = q.shape[0]
    T = q.shape[1]
    H = q.shape[2]
    K = q.shape[3]
    V = v.shape[3]

    if scale is None:
        scale = K**-0.5

    q = ttnn.transpose(q, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.transpose(k, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.transpose(v, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    beta = ttnn.transpose(beta, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)  # [B, H, T]
    g = ttnn.transpose(g, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)  # [B, H, T]

    q = ttnn.multiply(q, scale, memory_config=ttnn.L1_MEMORY_CONFIG)

    q = ttnn.typecast(q, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.typecast(k, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.typecast(v, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    beta = ttnn.typecast(beta, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    g = ttnn.typecast(g, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

    g_exp = ttnn.exp(g, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Initialize state
    h = ttnn.zeros(
        [B, H, K, V], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    if initial_state is not None:
        h = ttnn.typecast(
            ttnn.reshape(initial_state, [B, H, K, V], memory_config=ttnn.L1_MEMORY_CONFIG),
            ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    read_query_prog_cfg = None
    read_query_compute_cfg = None
    outer_product_prog_cfg = None
    if device is not None:
        try:
            read_query_prog_cfg = _recurrent_read_query_program_config(device, K, V)
            outer_product_prog_cfg = _recurrent_outer_product_program_config(device, K, V)
        except Exception:
            pass
        read_query_compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    outputs = []
    for t in range(T):
        q_t = q[:, :, t]
        k_t = k[:, :, t]  # [B, H, K]
        v_t = v[:, :, t]  # [B, H, V]
        beta_t = beta[:, :, t]  # [B, H]
        decay_t = g_exp[:, :, t]  # [B, H] - already exp(g_t)

        o_t, h = recurrent_delta_rule_step_ttnn(
            q_t,
            k_t,
            v_t,
            beta_t,
            decay_t,
            h,
            seq_len=T,
            device=device,
            step_idx=t,
            read_query_prog_cfg=read_query_prog_cfg,
            read_query_compute_cfg=read_query_compute_cfg,
            outer_product_prog_cfg=outer_product_prog_cfg,
        )
        o_4d = ttnn.reshape(o_t, [B, H, 1, V], memory_config=ttnn.L1_MEMORY_CONFIG)
        outputs.append(o_4d)

    o = ttnn.concat(outputs, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
    o = ttnn.transpose(o, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    o = ttnn.typecast(o, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    return o, h


def chunk_gated_delta_rule_ttnn(
    q,
    k,
    v,
    beta,
    g,
    chunk_size=64,
    initial_state=None,
    device=None,
):
    """Chunked gated delta rule implementation."""
    return recurrent_gated_delta_rule_ttnn(q, k, v, beta, g, initial_state, device)


def rms_norm_ttnn(x, weight, eps=1e-6):
    """RMS normalization using TTNN."""
    x_sq = ttnn.multiply(x, x, memory_config=ttnn.L1_MEMORY_CONFIG)
    mean_sq = ttnn.mean(x_sq, dim=-1, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
    rsqrt = ttnn.rsqrt(ttnn.add(mean_sq, eps, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.L1_MEMORY_CONFIG)
    normed = ttnn.multiply(x, rsqrt, memory_config=ttnn.L1_MEMORY_CONFIG)
    return ttnn.multiply(normed, weight, memory_config=ttnn.L1_MEMORY_CONFIG)


def rms_norm_gated_ttnn(x, gate, weight, eps=1e-6):
    """RMSNorm with SiLU gating using TTNN ops."""
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    gate = ttnn.to_layout(gate, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    x_sq = ttnn.multiply(x, x, memory_config=ttnn.L1_MEMORY_CONFIG)
    variance = ttnn.mean(x_sq, dim=-1, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)

    inv_rms = ttnn.rsqrt(
        ttnn.add(variance, eps, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.L1_MEMORY_CONFIG
    )

    x_normed = ttnn.multiply(x, inv_rms, memory_config=ttnn.L1_MEMORY_CONFIG)

    weight = ttnn.to_layout(weight, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    x_normed = ttnn.multiply(x_normed, weight, memory_config=ttnn.L1_MEMORY_CONFIG)

    gate = ttnn.to_layout(gate, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    gate_act = ttnn.silu(gate, memory_config=ttnn.L1_MEMORY_CONFIG)

    x_normed = ttnn.to_layout(x_normed, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    gate_act = ttnn.to_layout(gate_act, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    result = ttnn.multiply(x_normed, gate_act, memory_config=ttnn.L1_MEMORY_CONFIG)
    return result


def _causal_conv1d_fir(x, weight, bias, kernel_size, device):
    """
    Manual FIR decomposition of depthwise causal conv1d + SiLU.

    Used for large T where native ttnn.conv1d would OOM in L1.
    Decomposes the convolution into K element-wise multiply+accumulate
    operations on shifted slices.

    Args / Returns: same as causal_conv1d_ttnn.
    """
    B, T, D = x.shape[0], x.shape[1], x.shape[2]

    pad = ttnn.zeros(
        [B, kernel_size - 1, D],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x_padded = ttnn.concat([pad, x], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)  # [B, T+K-1, D]

    weight_torch = ttnn.to_torch(weight)  # [D, 1, K]

    out = None
    for k in range(kernel_size):
        x_slice = x_padded[:, k : k + T]
        x_slice = ttnn.to_layout(x_slice, ttnn.TILE_LAYOUT)

        w_k = weight_torch[:, 0, k].reshape(1, 1, D).contiguous()
        w_k_dev = ttnn.from_torch(
            w_k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        term = ttnn.multiply(x_slice, w_k_dev, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = term if out is None else ttnn.add(out, term, memory_config=ttnn.L1_MEMORY_CONFIG)

    if bias is not None:
        bias_torch = ttnn.to_torch(bias).reshape(1, 1, D).contiguous()
        bias_dev = ttnn.from_torch(
            bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        out = ttnn.add(out, bias_dev, memory_config=ttnn.L1_MEMORY_CONFIG)

    return ttnn.silu(out, memory_config=ttnn.L1_MEMORY_CONFIG)


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
    """TTNN forward pass for the Gated DeltaNet layer."""
    if num_v_heads is None:
        num_v_heads = num_heads

    B = hidden_states.shape[0]
    T = hidden_states.shape[1]

    q = ttnn.linear(hidden_states, q_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.linear(hidden_states, k_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.linear(hidden_states, v_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)

    q = causal_conv1d_ttnn(q, q_conv_weight, q_conv_bias, conv_kernel_size, device)
    k = causal_conv1d_ttnn(k, k_conv_weight, k_conv_bias, conv_kernel_size, device)
    v = causal_conv1d_ttnn(v, v_conv_weight, v_conv_bias, conv_kernel_size, device)

    q = ttnn.reshape(q, [B, T, num_heads, head_k_dim])
    k = ttnn.reshape(k, [B, T, num_heads, head_k_dim])
    v = ttnn.reshape(v, [B, T, num_v_heads, head_v_dim])

    if num_v_heads > num_heads:
        repeats = num_v_heads // num_heads
        q = ttnn.repeat_interleave(q, repeats, dim=2)
        k = ttnn.repeat_interleave(k, repeats, dim=2)

    q = l2_norm_ttnn(q, dim=-1)
    k = l2_norm_ttnn(k, dim=-1)

    beta = ttnn.sigmoid(
        ttnn.linear(hidden_states, b_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    if allow_neg_eigval:
        beta = ttnn.multiply(beta, 2.0, memory_config=ttnn.L1_MEMORY_CONFIG)

    a = ttnn.linear(hidden_states, a_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    a_biased = ttnn.add(a, dt_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    sp = ttnn.softplus(a_biased, memory_config=ttnn.L1_MEMORY_CONFIG)
    A = ttnn.exp(A_log, memory_config=ttnn.L1_MEMORY_CONFIG)
    A_neg = ttnn.neg(A, memory_config=ttnn.L1_MEMORY_CONFIG)
    g = ttnn.multiply(A_neg, sp, memory_config=ttnn.L1_MEMORY_CONFIG)

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

    if use_gate and g_proj_weight is not None:
        gate = ttnn.linear(hidden_states, g_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
        gate = ttnn.reshape(gate, [B, T, num_v_heads, head_v_dim])
        o = rms_norm_gated_ttnn(o, gate, o_norm_weight, eps=norm_eps)
    else:
        o = rms_norm_ttnn(o, o_norm_weight, eps=norm_eps)

    o = ttnn.reshape(o, [B, T, num_v_heads * head_v_dim])
    o = ttnn.linear(o, o_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)

    return o


class TTNNRecurrentDeltaNet(TTNNModule):
    """TTNN-accelerated Recurrent DeltaNet layer."""

    def __init__(
        self,
        num_heads: int = 4,
        num_v_heads: Optional[int] = None,
        head_k_dim: int = 256,
        head_v_dim: int = 512,
        conv_kernel_size: int = 4,
        use_gate: bool = True,
        allow_neg_eigval: bool = False,
        norm_eps: float = 1e-5,
        mode: str = "recurrent",
        chunk_size: int = 64,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads or num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_kernel_size = conv_kernel_size
        self.use_gate = use_gate
        self.allow_neg_eigval = allow_neg_eigval
        self.norm_eps = norm_eps
        self.mode = mode
        self.chunk_size = chunk_size

    @classmethod
    def from_torch(cls, torch_layer, *args, **kwargs):
        """Create TTNNRecurrentDeltaNet from PyTorch module."""
        new_layer = cls(*args, **kwargs)
        new_layer._fallback_torch_layer = torch_layer
        return new_layer

    def preprocess_weights_impl(self):
        """Preprocess weights for TTNN operations."""
        if self.torch_layer is None:
            return

        # Extract weights from PyTorch layer
        state_dict = self.torch_layer.state_dict()

        # Preprocess linear projection weights
        self.q_proj_weight = self._preprocess_linear_weight(state_dict.get("q_proj.weight"))
        self.k_proj_weight = self._preprocess_linear_weight(state_dict.get("k_proj.weight"))
        self.v_proj_weight = self._preprocess_linear_weight(state_dict.get("v_proj.weight"))
        self.a_proj_weight = self._preprocess_linear_weight(state_dict.get("a_proj.weight"))
        self.b_proj_weight = self._preprocess_linear_weight(state_dict.get("b_proj.weight"))
        self.o_proj_weight = self._preprocess_linear_weight(state_dict.get("o_proj.weight"))

        if self.use_gate:
            self.g_proj_weight = self._preprocess_linear_weight(state_dict.get("g_proj.weight"))

        # Preprocess convolution weights
        self.q_conv_weight = self._preprocess_conv_weight(state_dict.get("q_conv.weight"))
        self.k_conv_weight = self._preprocess_conv_weight(state_dict.get("k_conv.weight"))
        self.v_conv_weight = self._preprocess_conv_weight(state_dict.get("v_conv.weight"))

        # Preprocess biases
        self.q_conv_bias = self._preprocess_bias(state_dict.get("q_conv.bias"))
        self.k_conv_bias = self._preprocess_bias(state_dict.get("k_conv.bias"))
        self.v_conv_bias = self._preprocess_bias(state_dict.get("v_conv.bias"))

        # Preprocess other parameters
        self.A_log = self._preprocess_param(state_dict.get("A_log"))
        self.dt_bias = self._preprocess_param(state_dict.get("dt_bias"))
        self.o_norm_weight = self._preprocess_param(state_dict.get("o_norm.weight"))

    def _preprocess_linear_weight(self, weight):
        """Preprocess linear weight for TTNN."""
        if weight is None:
            return None
        # Transpose for TTNN convention [in_features, out_features]
        weight_t = weight.transpose(0, 1)
        return ttnn.from_torch(weight_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def _preprocess_conv_weight(self, weight):
        """Preprocess convolution weight for TTNN."""
        if weight is None:
            return None
        return ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _preprocess_bias(self, bias):
        """Preprocess bias for TTNN."""
        if bias is None:
            return None
        return ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def _preprocess_param(self, param):
        """Preprocess parameter for TTNN."""
        if param is None:
            return None
        return ttnn.from_torch(param, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""
        if self.q_proj_weight is not None:
            self.q_proj_weight = ttnn.to_device(self.q_proj_weight, self.device)
        if self.k_proj_weight is not None:
            self.k_proj_weight = ttnn.to_device(self.k_proj_weight, self.device)
        if self.v_proj_weight is not None:
            self.v_proj_weight = ttnn.to_device(self.v_proj_weight, self.device)
        if self.a_proj_weight is not None:
            self.a_proj_weight = ttnn.to_device(self.a_proj_weight, self.device)
        if self.b_proj_weight is not None:
            self.b_proj_weight = ttnn.to_device(self.b_proj_weight, self.device)
        if self.o_proj_weight is not None:
            self.o_proj_weight = ttnn.to_device(self.o_proj_weight, self.device)
        if self.use_gate and self.g_proj_weight is not None:
            self.g_proj_weight = ttnn.to_device(self.g_proj_weight, self.device)
        if self.q_conv_weight is not None:
            self.q_conv_weight = ttnn.to_device(self.q_conv_weight, self.device)
        if self.k_conv_weight is not None:
            self.k_conv_weight = ttnn.to_device(self.k_conv_weight, self.device)
        if self.v_conv_weight is not None:
            self.v_conv_weight = ttnn.to_device(self.v_conv_weight, self.device)
        if self.q_conv_bias is not None:
            self.q_conv_bias = ttnn.to_device(self.q_conv_bias, self.device)
        if self.k_conv_bias is not None:
            self.k_conv_bias = ttnn.to_device(self.k_conv_bias, self.device)
        if self.v_conv_bias is not None:
            self.v_conv_bias = ttnn.to_device(self.v_conv_bias, self.device)
        if self.A_log is not None:
            self.A_log = ttnn.to_device(self.A_log, self.device)
        if self.dt_bias is not None:
            self.dt_bias = ttnn.to_device(self.dt_bias, self.device)
        if self.o_norm_weight is not None:
            self.o_norm_weight = ttnn.to_device(self.o_norm_weight, self.device)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        recurrent_state: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Forward pass through Recurrent DeltaNet.

        Args:
            hidden_states: [B, T, hidden_size] input tensor
            recurrent_state: [B, H, K, V] optional initial state

        Returns:
            output: [B, T, hidden_size] output tensor
        """
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        output = gated_deltanet_forward_ttnn(
            hidden_states=hidden_states,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
            a_proj_weight=self.a_proj_weight,
            b_proj_weight=self.b_proj_weight,
            o_proj_weight=self.o_proj_weight,
            q_conv_weight=self.q_conv_weight,
            k_conv_weight=self.k_conv_weight,
            v_conv_weight=self.v_conv_weight,
            q_conv_bias=self.q_conv_bias,
            k_conv_bias=self.k_conv_bias,
            v_conv_bias=self.v_conv_bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            o_norm_weight=self.o_norm_weight,
            g_proj_weight=getattr(self, "g_proj_weight", None),
            num_heads=self.num_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_kernel_size=self.conv_kernel_size,
            use_gate=self.use_gate,
            allow_neg_eigval=self.allow_neg_eigval,
            norm_eps=self.norm_eps,
            device=self.device,
            recurrent_state=recurrent_state,
            mode=self.mode,
            chunk_size=self.chunk_size,
        )

        return output
