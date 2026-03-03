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

TILE_SIZE = 32  # TTNN tile size


def create_matmul_program_config(device, M, K, N, grid_size=None):
    """
    Create a multi-core matmul program config to parallelize across available cores.

    Args:
        device: TTNN device
        M: First dimension of output (M x N)
        K: Inner dimension (M x K @ K x N)
        N: Second dimension of output
        grid_size: Optional (cores_x, cores_y) tuple. If None, uses device's compute grid.

    Returns:
        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
    """
    if grid_size is None:
        device_grid = device.compute_with_storage_grid_size()
        cores_x = device_grid.x
        cores_y = device_grid.y
    else:
        cores_x, cores_y = grid_size

    # Calculate tiles
    M_tiles = math.ceil(M / TILE_SIZE)
    N_tiles = math.ceil(N / TILE_SIZE)
    K_tiles = math.ceil(K / TILE_SIZE)

    # Calculate per-core work
    per_core_M = max(1, math.ceil(M_tiles / cores_y))
    per_core_N = max(1, math.ceil(N_tiles / cores_x))

    # Calculate in0_block_w (must divide K_tiles evenly across cores_y)
    # Higher is better for performance
    K_tiles_per_core = math.ceil(K_tiles / cores_y)
    in0_block_w = 1
    if K_tiles_per_core > 0:
        # Find largest divisor that fits
        for divisor in range(K_tiles_per_core, 0, -1):
            if K_tiles_per_core % divisor == 0:
                in0_block_w = divisor
                break

    # Subblock sizes (must divide per_core dimensions)
    # CRITICAL: out_subblock_h * out_subblock_w <= 4 for FP32 accumulation
    # Find the best combination that maximizes product while staying <= 4
    out_subblock_h = 1
    out_subblock_w = 1
    max_product = 0

    # Find all valid divisors for per_core_M and per_core_N
    valid_h = [h for h in range(1, per_core_M + 1) if per_core_M % h == 0]
    valid_w = [w for w in range(1, per_core_N + 1) if per_core_N % w == 0]

    # Find combination with maximum product <= 4
    for h in valid_h:
        for w in valid_w:
            product = h * w
            if product <= 4 and product > max_product:
                max_product = product
                out_subblock_h = h
                out_subblock_w = w

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(cores_x, cores_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def l2_norm_ttnn(x, dim=-1, eps=1e-6):
    """L2 normalization along a given dimension."""
    x_sq = ttnn.multiply(x, x)
    norm_sq = ttnn.sum(x_sq, dim=dim, keepdim=True)
    inv_norm = ttnn.rsqrt(ttnn.add(norm_sq, eps))
    return ttnn.multiply(x, inv_norm)


def recurrent_delta_rule_step_ttnn(
    q_t,
    k_t,
    v_t,
    beta_t,
    g_t,
    h,
):
    """
    Single recurrent step of the gated delta rule using TTNN ops.

    Uses ttnn.matmul for K-dimension reductions to leverage hardware
    float32 accumulation, improving numerical precision.

    Args:
        q_t: [B, H, K] query for this timestep
        k_t: [B, H, K] key for this timestep
        v_t: [B, H, V] value for this timestep
        beta_t: [B, H] write strength
        g_t: [B, H] log-space decay
        h: [B, H, K, V] recurrent state

    Returns:
        o_t: [B, H, V] output
        h: [B, H, K, V] updated state
    """
    B = q_t.shape[0]
    H = q_t.shape[1]
    K = q_t.shape[2]
    V = v_t.shape[2]

    # 1. Decay the state: h = h * exp(g_t)
    decay = ttnn.exp(g_t)  # [B, H]
    decay = ttnn.reshape(decay, [B, H, 1, 1])  # [B, H, 1, 1]
    h = ttnn.multiply(h, decay)

    # 2. Read from state via matmul: v_read = k^T @ h
    k_row = ttnn.reshape(k_t, [B, H, 1, K])  # [B, H, 1, K]
    v_read = ttnn.matmul(k_row, h)  # [B, H, 1, V]
    v_read = ttnn.reshape(v_read, [B, H, V])  # [B, H, V]

    # 3. Compute delta: delta = (v_t - v_read) * beta_t
    delta = ttnn.subtract(v_t, v_read)
    beta_expanded = ttnn.reshape(beta_t, [B, H, 1])  # [B, H, 1]
    delta = ttnn.multiply(delta, beta_expanded)

    # 4. Write to state via matmul outer product: h += k @ delta^T
    k_col = ttnn.reshape(k_t, [B, H, K, 1])  # [B, H, K, 1]
    d_row = ttnn.reshape(delta, [B, H, 1, V])  # [B, H, 1, V]
    outer = ttnn.matmul(k_col, d_row)  # [B, H, K, V]
    h = ttnn.add(h, outer)

    # 5. Query state via matmul: o_t = q^T @ h
    q_row = ttnn.reshape(q_t, [B, H, 1, K])  # [B, H, 1, K]
    o_t = ttnn.matmul(q_row, h)  # [B, H, 1, V]
    o_t = ttnn.reshape(o_t, [B, H, V])  # [B, H, V]

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

    # Cast to float32 for recurrent precision (matches torch reference)
    q = ttnn.typecast(q, ttnn.float32)
    k = ttnn.typecast(k, ttnn.float32)
    v = ttnn.typecast(v, ttnn.float32)
    beta = ttnn.typecast(beta, ttnn.float32)
    g = ttnn.typecast(g, ttnn.float32)

    # Initialize state in float32
    if initial_state is not None:
        h = ttnn.typecast(initial_state, ttnn.float32)
    else:
        h = ttnn.zeros([B, H, K, V], device=device, dtype=ttnn.float32)

    outputs = []
    for i in range(T):
        q_t = q[:, :, i]  # [B, H, K]
        k_t = k[:, :, i]  # [B, H, K]
        v_t = v[:, :, i]  # [B, H, V]
        beta_t = beta[:, :, i]  # [B, H]
        g_t = g[:, :, i]  # [B, H]

        o_t, h = recurrent_delta_rule_step_ttnn(q_t, k_t, v_t, beta_t, g_t, h)
        outputs.append(o_t)

    # Concat outputs: reshape each [B, H, V] -> [B, H, 1, V] then concat
    outputs_4d = [ttnn.reshape(o, [B, H, 1, V]) for o in outputs]
    o = ttnn.concat(outputs_4d, dim=2)  # [B, H, T, V]

    # Transpose back to [B, T, H, V]
    o = ttnn.transpose(o, 1, 2)

    # Cast back to bfloat16
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

    # OPTIMIZATION: Create program configs for pre-loop matmuls
    # k_beta_c @ k_c_t: [batch, chunk_size, K] @ [batch, K, chunk_size] -> [batch, chunk_size, chunk_size]
    mm_kk_config = create_matmul_program_config(device, batch * chunk_size, K, chunk_size)
    # Neumann series matmuls: [batch, chunk_size, chunk_size] @ [batch, chunk_size, chunk_size]
    mm_neumann_config = create_matmul_program_config(device, batch * chunk_size, chunk_size, chunk_size)
    # attn @ v_beta_c: [batch, chunk_size, chunk_size] @ [batch, chunk_size, V] -> [batch, chunk_size, V]
    mm_attn_v_beta_config = create_matmul_program_config(device, batch * chunk_size, chunk_size, V)
    # attn @ k_beta_c: [batch, chunk_size, chunk_size] @ [batch, chunk_size, K] -> [batch, chunk_size, K]
    mm_attn_k_beta_config = create_matmul_program_config(device, batch * chunk_size, chunk_size, K)

    # --- Cumsum via matmul with upper-triangular ones ---
    triu_torch = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.float32))
    triu_ones = ttnn.from_torch(triu_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    triu_ones = ttnn.reshape(triu_ones, [1, chunk_size, chunk_size])

    g_c_3d = ttnn.reshape(g_c, [batch, 1, chunk_size])
    # Small matmul [batch, 1, chunk_size] @ [1, chunk_size, chunk_size] - may not benefit from multi-core
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
    kk = ttnn.matmul(k_beta_c, k_c_t, program_config=mm_kk_config)

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
    P = ttnn.matmul(M, M, program_config=mm_neumann_config)
    num_steps = max(int(math.ceil(math.log2(max(chunk_size, 2)))) - 1, 0)
    for _ in range(num_steps):
        R = ttnn.add(R, ttnn.matmul(R, P, program_config=mm_neumann_config))
        P = ttnn.matmul(P, P, program_config=mm_neumann_config)

    attn = R

    # --- Corrected values and keys ---
    v_corrected = ttnn.matmul(attn, v_beta_c, program_config=mm_attn_v_beta_config)
    k_cumdecay = ttnn.matmul(attn, ttnn.multiply(k_beta_c, decay_exp), program_config=mm_attn_k_beta_config)

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

    S = ttnn.zeros(
        [BH, K, V], device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    if initial_state is not None:
        S = ttnn.typecast(ttnn.reshape(initial_state, [BH, K, V]), ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)

    # OPTIMIZATION: Create matmul program configs to use more cores
    # These configs parallelize matmuls across available device cores
    # q_i @ k_i_t: [BH, chunk_size, K] @ [BH, K, chunk_size] -> [BH, chunk_size, chunk_size]
    mm_qk_config = create_matmul_program_config(device, BH * chunk_size, K, chunk_size)
    # k_cum_i @ S and q_i @ S: [BH, chunk_size, K] @ [BH, K, V] -> [BH, chunk_size, V]
    mm_kv_config = create_matmul_program_config(device, BH * chunk_size, K, V)
    # intra_attn @ v_new: [BH, chunk_size, chunk_size] @ [BH, chunk_size, V] -> [BH, chunk_size, V]
    mm_attn_v_config = create_matmul_program_config(device, BH * chunk_size, chunk_size, V)
    # k_decay_t @ v_new: [BH, K, chunk_size] @ [BH, chunk_size, V] -> [BH, K, V]
    mm_k_decay_v_config = create_matmul_program_config(device, BH * K, chunk_size, V)

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
        intra_attn = ttnn.multiply(
            ttnn.matmul(q_i, k_i_t, memory_config=ttnn.L1_MEMORY_CONFIG, program_config=mm_qk_config),
            L_mask_i,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        intra_attn = ttnn.multiply(intra_attn, lower_causal, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Cross-chunk: read from state
        v_prime = ttnn.matmul(k_cum_i, S, memory_config=ttnn.L1_MEMORY_CONFIG, program_config=mm_kv_config)
        v_new = ttnn.subtract(v_i, v_prime, memory_config=ttnn.L1_MEMORY_CONFIG)

        decay_i_exp = ttnn.reshape(ttnn.exp(decay_i, memory_config=ttnn.L1_MEMORY_CONFIG), [BH, chunk_size, 1])
        o_inter = ttnn.matmul(
            ttnn.multiply(q_i, decay_i_exp, memory_config=ttnn.L1_MEMORY_CONFIG),
            S,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=mm_kv_config,
        )

        o_i = ttnn.add(
            o_inter,
            ttnn.matmul(intra_attn, v_new, memory_config=ttnn.L1_MEMORY_CONFIG, program_config=mm_attn_v_config),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        outputs.append(ttnn.reshape(o_i, [BH, 1, chunk_size, V]))

        # Update state
        dl_i = decay_last[:, i]
        dl_i_exp = ttnn.reshape(ttnn.exp(dl_i, memory_config=ttnn.L1_MEMORY_CONFIG), [BH, 1, 1])
        S = ttnn.multiply(S, dl_i_exp, memory_config=ttnn.L1_MEMORY_CONFIG)

        dl_i_2d = ttnn.reshape(dl_i, [BH, 1])
        decay_diff = ttnn.subtract(dl_i_2d, decay_i, memory_config=ttnn.L1_MEMORY_CONFIG)
        k_decay = ttnn.multiply(
            k_i,
            ttnn.reshape(ttnn.exp(decay_diff, memory_config=ttnn.L1_MEMORY_CONFIG), [BH, chunk_size, 1]),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        k_decay_t = ttnn.transpose(k_decay, 1, 2)
        S = ttnn.add(
            S,
            ttnn.matmul(k_decay_t, v_new, memory_config=ttnn.L1_MEMORY_CONFIG, program_config=mm_k_decay_v_config),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    o = ttnn.concat(outputs, dim=1)
    o = ttnn.reshape(o, [BH, L, V])

    if pad_len > 0:
        o = o[:, :T]

    o = ttnn.reshape(o, [B, H, T, V])
    o = ttnn.transpose(o, 1, 2)
    o = ttnn.typecast(o, ttnn.bfloat16)

    final_state = ttnn.reshape(S, [B, H, K, V])
    return o, final_state
