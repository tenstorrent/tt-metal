# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementations of the gated delta rule.
Contains both the recurrent (token-by-token) and chunked (parallel prefill)
implementations of the gated delta rule attention mechanism.

Tensor layout convention (FLA style):
  q, k: [B, T, H, K]   (batch, time, heads, key_dim)
  v:    [B, T, H, V]   (batch, time, heads, value_dim)
  beta: [B, T, H]      (batch, time, heads)
  g:    [B, T, H]      (batch, time, heads) -- log-space decay
  state:[B, H, K, V]   (batch, heads, key_dim, value_dim)
"""

import math
import torch
import ttnn

# Tile size used by TTNN matmul (wormhole)
_TILE_H = 32
_TILE_W = 32


def _recurrent_outer_product_program_config(device, K, V):
    """
    Build MatmulMultiCoreReuseProgramConfig for the outer-product matmul:

        k_col [B*H, K, 1] @ d_row [B*H, 1, V]  ->  [B*H, K, V]

    TTNN constraints (from matmul_device_operation.cpp MatmulMultiCoreReuseProgramConfig):
      - N == per_core_N  (N is total N in tiles; no split along N)
      - M % per_core_M == 0 when per_core_M <= M; else total_M % per_core_M == 0
      - in0_block_w divides K (inner dim in tiles)
    Tensor dims (per batch): M_tiles = ceil(K/32), K_inner = 1, N_tiles = ceil(V/32).
    """
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
    """
    Build MatmulMultiCoreReuseProgramConfig for read/query matmuls:

        row [B*H, 1, K] @ h [B*H, K, V]  ->  [B*H, 1, V]

    M_tiles=1, K_tiles=ceil(K/32), N_tiles=ceil(V/32). Constraint N == per_core_N.
    """
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
    x_sq = ttnn.multiply(x, x)
    norm_sq = ttnn.sum(x_sq, dim=dim, keepdim=True)
    inv_norm = ttnn.rsqrt(ttnn.add(norm_sq, eps))
    return ttnn.multiply(x, inv_norm)


def fused_decay_and_write_ttnn(
    h,
    k_t,
    delta,
    decay_t,
    beta_t,
    device=None,
):
    """
    Logical fusion for the recurrent delta rule state update:

        h = decay * h + beta_t * (k_t ⊗ delta)

    Implemented using existing TTNN ops so call sites are stable.
    Can be replaced by a true fused kernel later.
    """
    B = h.shape[0]
    H = h.shape[1]
    K = h.shape[2]
    V = h.shape[3]

    # decay: [B, H] -> [B, H, 1, 1]
    # decay_t is already exp(g_t); keep recurrent path in BF16.
    decay = ttnn.typecast(decay_t, ttnn.bfloat16)
    decay = ttnn.reshape(decay, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

    # beta: [B, H] -> [B, H, 1, 1]
    beta_expanded = ttnn.reshape(beta_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

    # k_t: [B, H, K] -> [B, H, K, 1]
    k_col = ttnn.reshape(k_t, [B, H, K, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

    # delta: [B, H, V] -> [B, H, 1, V]
    d_row = ttnn.reshape(delta, [B, H, 1, V], memory_config=ttnn.L1_MEMORY_CONFIG)

    k_col = ttnn.to_layout(k_col, ttnn.TILE_LAYOUT)
    d_row = ttnn.to_layout(d_row, ttnn.TILE_LAYOUT)
    k_col = ttnn.to_memory_config(k_col, ttnn.L1_MEMORY_CONFIG)
    d_row = ttnn.to_memory_config(d_row, ttnn.L1_MEMORY_CONFIG)

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
        program_config=None,
    )

    # apply beta
    outer = ttnn.multiply(
        outer,
        beta_expanded,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # fused-style update: decay * h + outer
    h = ttnn.multiply(h, decay)
    h = ttnn.add(h, outer)

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
):
    """
    Recurrent delta rule step using TTNN ops, with a logically fused
    state update implemented via `fused_decay_and_write_ttnn`.

    This keeps the call site ready for a future single-kernel
    implementation without changing model code.
    """
    B = q_t.shape[0]
    H = q_t.shape[1]
    K = q_t.shape[2]
    V = v_t.shape[2]

    h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)
    h = ttnn.to_memory_config(h, ttnn.L1_MEMORY_CONFIG)

    read_query_compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    read_query_prog_cfg = None
    if device is not None:
        try:
            read_query_prog_cfg = _recurrent_read_query_program_config(device, K, V)
        except Exception:
            pass

    k_row = ttnn.reshape(k_t, [B, H, 1, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    k_row = ttnn.to_layout(k_row, ttnn.TILE_LAYOUT)
    k_row = ttnn.to_memory_config(k_row, ttnn.L1_MEMORY_CONFIG)
    v_read = ttnn.matmul(
        k_row,
        h,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        program_config=read_query_prog_cfg,
        compute_kernel_config=read_query_compute_cfg,
    )
    v_read = ttnn.reshape(v_read, [B, H, V], memory_config=ttnn.L1_MEMORY_CONFIG)

    # 2. Compute delta (pre-beta): delta = v_t - v_read
    delta = ttnn.subtract(v_t, v_read, memory_config=ttnn.L1_MEMORY_CONFIG)

    # 3. Fused-style decay + write to state (decay_t already = exp(g_t))
    h = fused_decay_and_write_ttnn(
        h=h,
        k_t=k_t,
        delta=delta,
        decay_t=decay_t,
        beta_t=beta_t,
        device=device,
    )

    q_row = ttnn.reshape(q_t, [B, H, 1, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    q_row = ttnn.to_layout(q_row, ttnn.TILE_LAYOUT)
    q_row = ttnn.to_memory_config(q_row, ttnn.L1_MEMORY_CONFIG)
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
    scale=None,
    initial_state=None,
    device=None,
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
        q: [B, T, H, K]
        k: [B, T, H, K]
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
    q = l2_norm_ttnn(q, dim=-1)
    k = l2_norm_ttnn(k, dim=-1)

    B = q.shape[0]
    T = q.shape[1]
    H = q.shape[2]
    K = q.shape[3]
    V = v.shape[3]

    if scale is None:
        scale = K**-0.5

    q = ttnn.multiply(q, scale)

    # Transpose to [B, H, T, D] for head-first processing
    q = ttnn.transpose(q, 1, 2)
    k = ttnn.transpose(k, 1, 2)
    v = ttnn.transpose(v, 1, 2)
    beta = ttnn.transpose(beta, 1, 2)  # [B, H, T]
    g = ttnn.transpose(g, 1, 2)  # [B, H, T]

    q = ttnn.typecast(q, ttnn.bfloat16)
    k = ttnn.typecast(k, ttnn.bfloat16)
    v = ttnn.typecast(v, ttnn.bfloat16)
    beta = ttnn.typecast(beta, ttnn.bfloat16)
    g = ttnn.typecast(g, ttnn.bfloat16)

    # Precompute exp(g) once and slice per timestep in the loop.
    g_exp = ttnn.exp(g)

    if initial_state is not None:
        h = ttnn.typecast(initial_state, ttnn.bfloat16)
    else:
        h = ttnn.zeros([B, H, K, V], device=device, dtype=ttnn.bfloat16)

    outputs = []
    for i in range(T):
        q_t = q[:, :, i]  # [B, H, K]
        k_t = k[:, :, i]  # [B, H, K]
        v_t = v[:, :, i]  # [B, H, V]
        beta_t = beta[:, :, i]  # [B, H]
        decay_t = g_exp[:, :, i]  # [B, H]

        o_t, h = recurrent_delta_rule_step_ttnn(q_t, k_t, v_t, beta_t, decay_t, h, seq_len=T, device=device)
        outputs.append(o_t)

    outputs_4d = [ttnn.reshape(o, [B, H, 1, V], memory_config=ttnn.L1_MEMORY_CONFIG) for o in outputs]
    o = ttnn.concat(outputs_4d, dim=2)
    o = ttnn.transpose(o, 1, 2)
    o = ttnn.typecast(o, ttnn.bfloat16)

    return o, h


def chunk_gated_delta_rule_ttnn(
    q,
    k,
    v,
    beta,
    g,
    chunk_size=64,
    scale=None,
    initial_state=None,
    device=None,
):
    """
    Chunked gated delta rule using TTNN ops. Used for prefill.

    Processes the sequence in chunks of `chunk_size` tokens.
    Within each chunk: batched matmuls (parallel over tokens).
    Across chunks: sequential state propagation (T/chunk_size steps).

    Uses repeated-squaring Neumann series to resolve intra-chunk
    dependencies in O(log(chunk_size)) matmuls instead of O(chunk_size).
    Cumsum is computed via matmul with upper-triangular ones matrix.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g: [B, T, H]
        chunk_size: int
        scale: float
        initial_state: [B, H, K, V]
        device: ttnn device

    Returns:
        output: [B, T, H, V]
        final_state: [B, H, K, V]
    """
    q = l2_norm_ttnn(q, dim=-1)
    k = l2_norm_ttnn(k, dim=-1)

    B = q.shape[0]
    T = q.shape[1]
    H = q.shape[2]
    K = q.shape[3]
    V = v.shape[3]
    BH = B * H

    if scale is None:
        scale = K**-0.5

    # Transpose to [B, H, T, D], cast to float32
    q = ttnn.typecast(ttnn.transpose(q, 1, 2), ttnn.float32)
    k = ttnn.typecast(ttnn.transpose(k, 1, 2), ttnn.float32)
    v = ttnn.typecast(ttnn.transpose(v, 1, 2), ttnn.float32)
    beta = ttnn.typecast(ttnn.transpose(beta, 1, 2), ttnn.float32)
    g = ttnn.typecast(ttnn.transpose(g, 1, 2), ttnn.float32)

    q = ttnn.multiply(q, scale)

    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    L = T + pad_len
    num_chunks = L // chunk_size
    batch = BH * num_chunks

    # Flatten to [BH, T, D]
    q = ttnn.reshape(q, [BH, T, K])
    k = ttnn.reshape(k, [BH, T, K])
    v = ttnn.reshape(v, [BH, T, V])
    beta_flat = ttnn.reshape(beta, [BH, T, 1])
    g = ttnn.reshape(g, [BH, T])

    if pad_len > 0:
        q = ttnn.concat(
            [q, ttnn.zeros([BH, pad_len, K], device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)], dim=1
        )
        k = ttnn.concat(
            [k, ttnn.zeros([BH, pad_len, K], device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)], dim=1
        )
        v = ttnn.concat(
            [v, ttnn.zeros([BH, pad_len, V], device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)], dim=1
        )
        beta_flat = ttnn.concat(
            [beta_flat, ttnn.zeros([BH, pad_len, 1], device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)], dim=1
        )
        g_3d = ttnn.reshape(g, [BH, T, 1])
        g_3d = ttnn.concat(
            [g_3d, ttnn.zeros([BH, pad_len, 1], device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)], dim=1
        )
        g = ttnn.reshape(g_3d, [BH, L])
        beta_flat = ttnn.reshape(beta_flat, [BH, L, 1])
    else:
        beta_flat = ttnn.reshape(beta_flat, [BH, L, 1])

    # v_beta = v * beta, k_beta = k * beta
    v_beta = ttnn.multiply(v, beta_flat)
    k_beta = ttnn.multiply(k, beta_flat)

    # Reshape into chunks: [BH*nc, cs, D]
    q_c = ttnn.reshape(q, [batch, chunk_size, K])
    k_c = ttnn.reshape(k, [batch, chunk_size, K])
    v_c = ttnn.reshape(v, [batch, chunk_size, V])
    k_beta_c = ttnn.reshape(k_beta, [batch, chunk_size, K])
    v_beta_c = ttnn.reshape(v_beta, [batch, chunk_size, V])
    g_c = ttnn.reshape(g, [batch, chunk_size])

    # --- Cumsum via matmul with upper-triangular ones ---
    triu_torch = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.float32))
    triu_ones = ttnn.from_torch(triu_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    triu_ones = ttnn.reshape(triu_ones, [1, chunk_size, chunk_size])

    g_c_3d = ttnn.reshape(g_c, [batch, 1, chunk_size])
    decay = ttnn.reshape(ttnn.matmul(g_c_3d, triu_ones), [batch, chunk_size])

    # decay_exp for weighting k_beta: [batch, cs, 1]
    decay_exp = ttnn.reshape(ttnn.exp(decay), [batch, chunk_size, 1])

    # --- L_mask: exp(decay_i - decay_j) for j <= i ---
    decay_col = ttnn.reshape(decay, [batch, chunk_size, 1])
    decay_row = ttnn.reshape(decay, [batch, 1, chunk_size])
    L_diff = ttnn.subtract(decay_col, decay_row)

    tril_torch = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32))
    tril_mask = ttnn.from_torch(tril_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tril_mask = ttnn.reshape(tril_mask, [1, chunk_size, chunk_size])

    L_diff_masked = ttnn.multiply(L_diff, tril_mask)
    L_mask = ttnn.multiply(ttnn.exp(L_diff_masked), tril_mask)

    # --- Intra-chunk interaction matrix M ---
    k_c_t = ttnn.transpose(k_c, 1, 2)
    kk = ttnn.matmul(k_beta_c, k_c_t)

    M = ttnn.neg(ttnn.multiply(kk, L_mask))
    strict_lower_torch = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32), diagonal=-1)
    strict_lower = ttnn.from_torch(strict_lower_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    strict_lower = ttnn.reshape(strict_lower, [1, chunk_size, chunk_size])
    M = ttnn.multiply(M, strict_lower)

    # --- Woodbury via repeated-squaring Neumann series ---
    # Compute (I - M)^{-1} = I + M + M^2 + ... for nilpotent M
    eye_torch = torch.eye(chunk_size, dtype=torch.float32)
    eye = ttnn.from_torch(eye_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    eye = ttnn.reshape(eye, [1, chunk_size, chunk_size])

    R = ttnn.add(M, eye)
    P = ttnn.matmul(M, M)
    num_steps = max(int(math.ceil(math.log2(max(chunk_size, 2)))) - 1, 0)
    for _ in range(num_steps):
        R = ttnn.add(R, ttnn.matmul(R, P))
        P = ttnn.matmul(P, P)

    attn = R

    # --- Corrected values and keys ---
    v_corrected = ttnn.matmul(attn, v_beta_c)
    k_cumdecay = ttnn.matmul(attn, ttnn.multiply(k_beta_c, decay_exp))

    # --- Cross-chunk recurrence ---
    q_c_4d = ttnn.reshape(q_c, [BH, num_chunks, chunk_size, K])
    k_c_4d = ttnn.reshape(k_c, [BH, num_chunks, chunk_size, K])
    v_cor_4d = ttnn.reshape(v_corrected, [BH, num_chunks, chunk_size, V])
    k_cum_4d = ttnn.reshape(k_cumdecay, [BH, num_chunks, chunk_size, K])
    L_mask_4d = ttnn.reshape(L_mask, [BH, num_chunks, chunk_size, chunk_size])
    decay_3d = ttnn.reshape(decay, [BH, num_chunks, chunk_size])

    # Precompute total decay per chunk (= cumsum at last position)
    decay_last = ttnn.reshape(ttnn.sum(g_c, dim=-1), [BH, num_chunks, 1])

    lower_causal_torch = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32))
    lower_causal = ttnn.from_torch(lower_causal_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    S = ttnn.zeros([BH, K, V], device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    if initial_state is not None:
        S = ttnn.typecast(ttnn.reshape(initial_state, [BH, K, V]), ttnn.float32)

    outputs = []
    for i in range(num_chunks):
        # Slice and re-tilize (slicing from 4D may lose TILE_LAYOUT)
        q_i = ttnn.to_layout(q_c_4d[:, i], ttnn.TILE_LAYOUT)
        k_i = ttnn.to_layout(k_c_4d[:, i], ttnn.TILE_LAYOUT)
        v_i = ttnn.to_layout(v_cor_4d[:, i], ttnn.TILE_LAYOUT)
        k_cum_i = ttnn.to_layout(k_cum_4d[:, i], ttnn.TILE_LAYOUT)
        L_mask_i = ttnn.to_layout(L_mask_4d[:, i], ttnn.TILE_LAYOUT)
        decay_i = decay_3d[:, i]

        # Intra-chunk attention: (q @ k^T) * L_mask, lower-triangular
        k_i_t = ttnn.transpose(k_i, 1, 2)
        intra_attn = ttnn.multiply(ttnn.matmul(q_i, k_i_t), L_mask_i)
        intra_attn = ttnn.multiply(intra_attn, lower_causal)

        # Cross-chunk: read from state
        v_prime = ttnn.matmul(k_cum_i, S)
        v_new = ttnn.subtract(v_i, v_prime)

        decay_i_exp = ttnn.reshape(ttnn.exp(decay_i), [BH, chunk_size, 1])
        o_inter = ttnn.matmul(ttnn.multiply(q_i, decay_i_exp), S)

        o_i = ttnn.add(o_inter, ttnn.matmul(intra_attn, v_new))
        outputs.append(ttnn.reshape(o_i, [BH, 1, chunk_size, V]))

        # Update state
        dl_i = decay_last[:, i]
        dl_i_exp = ttnn.reshape(ttnn.exp(dl_i), [BH, 1, 1])
        S = ttnn.multiply(S, dl_i_exp)

        dl_i_2d = ttnn.reshape(dl_i, [BH, 1])
        decay_diff = ttnn.subtract(dl_i_2d, decay_i)
        k_decay = ttnn.multiply(k_i, ttnn.reshape(ttnn.exp(decay_diff), [BH, chunk_size, 1]))
        k_decay_t = ttnn.transpose(k_decay, 1, 2)
        S = ttnn.add(S, ttnn.matmul(k_decay_t, v_new))

    o = ttnn.concat(outputs, dim=1)
    o = ttnn.reshape(o, [BH, L, V])

    if pad_len > 0:
        o = o[:, :T]

    o = ttnn.reshape(o, [B, H, T, V])
    o = ttnn.transpose(o, 1, 2)
    o = ttnn.typecast(o, ttnn.bfloat16)

    final_state = ttnn.reshape(S, [B, H, K, V])
    return o, final_state
