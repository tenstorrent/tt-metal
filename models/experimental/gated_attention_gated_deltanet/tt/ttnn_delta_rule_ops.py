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


def _get_binary_core_grid(grid_size=None):
    """Create CoreRangeSet for binary operations (multiply, add, subtract, etc.).

    Args:
        grid_size: Optional (cores_x, cores_y) tuple. If None, returns None (auto-config).

    Returns:
        CoreRangeSet or None if auto-config is better
    """
    if grid_size is None:
        return None

    cores_x, cores_y = grid_size
    # Create a CoreRangeSet from (0,0) to (cores_x-1, cores_y-1)
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores_x - 1, cores_y - 1))})


def _get_matmul_program_config(m, k, n, grid_size=None, in0_block_w=None):
    """Create optimized program config for matmul operations.

    Args:
        m: M dimension (rows of first matrix)
        k: K dimension (shared dimension)
        n: N dimension (cols of second matrix)
        grid_size: Optional (cores_x, cores_y) tuple. If None, auto-selects based on shape.
        in0_block_w: Optional block width. If None, auto-selects.

    Returns:
        MatmulProgramConfig or None if auto-config is better
    """
    TILE_SIZE = 32

    # For very small matmuls, let ttnn auto-select
    if m < 32 or n < 32 or k < 32:
        return None

    m_tiles = math.ceil(m / TILE_SIZE)
    n_tiles = math.ceil(n / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)

    # Calculate per-core dimensions
    if grid_size is None:
        # Smart grid size selection based on shape
        # Consider total work (m * n * k) to determine optimal core count
        total_work = m * n * k

        # For very small matmuls (< 32K elements), let auto-config handle it
        if total_work < 32768:
            return None

        # For smaller matmuls, let ttnn auto-select - it often performs better
        # Only use manual configs for larger matmuls where we're confident they help
        if m_tiles == 2 and n_tiles == 2:
            # 64x64 matmuls - let auto-config handle it (performs better)
            return None
        elif m_tiles == 2 and n_tiles == 4:
            # 64x128 matmuls - let auto-config handle it (may perform better)
            return None
        elif m_tiles == 2 and n_tiles == 8:
            # 64x256 matmuls - try auto-config first, manual configs may not help
            # If needed, can use: grid_size = (8, 2)  # 16 cores
            return None
        elif m_tiles == 4 and n_tiles == 2:
            # 128x64 matmuls - let auto-config handle it
            return None
        elif m_tiles == 4 and n_tiles == 8:
            # 128x256 matmuls - use 32 cores (4x8 grid) for maximum parallelism
            # This is large enough that manual config likely helps
            grid_size = (8, 4)  # 32 cores
        elif m_tiles <= 4 and n_tiles <= 4:
            # Small-medium: use 2-4 cores
            if n_tiles >= 4:
                grid_size = (2, 1)  # 1D grid along N
            elif m_tiles >= 4:
                grid_size = (1, 2)  # 1D grid along M
            else:
                # 2x2 to 4x4 tiles, use 2-4 cores
                if total_work >= 262144:  # >= 64K elements
                    grid_size = (2, 2)  # 4 cores
                else:
                    grid_size = (2, 1)  # 2 cores
        elif m_tiles >= 4 and n_tiles >= 8:
            # Large: use 2D grid
            cores_y = min(4, m_tiles)
            cores_x = min(8, n_tiles)
            grid_size = (cores_x, cores_y)
        else:
            # Medium: use 1D or 2D grid based on aspect ratio
            if n_tiles >= 4:
                # Wider than tall, use cores along N
                total_cores = min(8, max(2, n_tiles))
                grid_size = (total_cores, 1)
            elif m_tiles >= 4:
                # Taller than wide, use cores along M
                total_cores = min(8, max(2, m_tiles))
                grid_size = (1, total_cores)
            else:
                # Let auto-config handle it
                return None

    cores_x, cores_y = grid_size

    # Calculate per-core M and N
    per_core_M = math.ceil(m_tiles / cores_y)
    per_core_N = math.ceil(n_tiles / cores_x)

    # Ensure minimum values
    per_core_M = max(1, per_core_M)
    per_core_N = max(1, per_core_N)

    # Auto-select in0_block_w if not provided
    if in0_block_w is None:
        # Try to maximize in0_block_w (better for performance)
        # in0_block_w divides K dimension across cores
        k_per_core = math.ceil(k_tiles / cores_x) if cores_x > 1 else k_tiles
        # Start with a reasonable value
        in0_block_w = min(4, max(1, k_per_core))
        # Ensure it divides evenly into k_tiles
        while k_tiles % (in0_block_w * cores_x) != 0 and in0_block_w > 1:
            in0_block_w -= 1
        if in0_block_w < 1:
            in0_block_w = 1

    # Calculate output subblock sizes - optimize for performance
    # For FP32 accumulation, DST has 4 tiles; for BF16, 8 tiles
    # Try to maximize subblock size for better performance
    max_subblock_size = 4  # Conservative for FP32

    # Try larger subblocks first (better performance)
    out_subblock_h = min(per_core_M, max_subblock_size)
    out_subblock_w = min(per_core_N, max_subblock_size)

    # Ensure subblock product fits in DST (4 tiles for FP32)
    if out_subblock_h * out_subblock_w > max_subblock_size:
        # Reduce to fit
        if out_subblock_h > out_subblock_w:
            out_subblock_h = max_subblock_size // out_subblock_w
        else:
            out_subblock_w = max_subblock_size // out_subblock_h

    # Ensure subblock divides evenly
    while per_core_M % out_subblock_h != 0 and out_subblock_h > 1:
        out_subblock_h -= 1
    while per_core_N % out_subblock_w != 0 and out_subblock_w > 1:
        out_subblock_w -= 1

    # Ensure we have valid subblock sizes
    if out_subblock_h < 1 or out_subblock_w < 1 or out_subblock_h * out_subblock_w > max_subblock_size:
        # Fallback to smaller subblocks
        out_subblock_h = min(per_core_M, 2)
        out_subblock_w = min(per_core_N, 2)
        while per_core_M % out_subblock_h != 0 and out_subblock_h > 1:
            out_subblock_h -= 1
        while per_core_N % out_subblock_w != 0 and out_subblock_w > 1:
            out_subblock_w -= 1

    if out_subblock_h < 1 or out_subblock_w < 1:
        return None

    try:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    except Exception:
        # If config creation fails, return None to use auto-config
        return None


def l2_norm_ttnn(x, dim=-1, eps=1e-6):
    """L2 normalization along a given dimension."""
    x_sq = ttnn.multiply(x, x, memory_config=ttnn.L1_MEMORY_CONFIG)
    norm_sq = ttnn.sum(x_sq, dim=dim, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
    inv_norm = ttnn.rsqrt(
        ttnn.add(norm_sq, eps, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.L1_MEMORY_CONFIG
    )
    return ttnn.multiply(x, inv_norm, memory_config=ttnn.L1_MEMORY_CONFIG)


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
    decay = ttnn.exp(g_t, memory_config=ttnn.L1_MEMORY_CONFIG)  # [B, H]
    decay = ttnn.reshape(decay, [B, H, 1, 1])  # [B, H, 1, 1]
    h = ttnn.multiply(h, decay, memory_config=ttnn.L1_MEMORY_CONFIG)

    # 2. Read from state via matmul: v_read = k^T @ h
    k_row = ttnn.reshape(k_t, [B, H, 1, K])  # [B, H, 1, K]
    v_read = ttnn.matmul(k_row, h, memory_config=ttnn.L1_MEMORY_CONFIG)  # [B, H, 1, V]
    v_read = ttnn.reshape(v_read, [B, H, V])  # [B, H, V]

    # 3. Compute delta: delta = (v_t - v_read) * beta_t
    delta = ttnn.subtract(v_t, v_read, memory_config=ttnn.L1_MEMORY_CONFIG)
    beta_expanded = ttnn.reshape(beta_t, [B, H, 1])  # [B, H, 1]
    delta = ttnn.multiply(delta, beta_expanded, memory_config=ttnn.L1_MEMORY_CONFIG)

    # 4. Write to state via matmul outer product: h += k @ delta^T
    k_col = ttnn.reshape(k_t, [B, H, K, 1])  # [B, H, K, 1]
    d_row = ttnn.reshape(delta, [B, H, 1, V])  # [B, H, 1, V]
    outer = ttnn.matmul(k_col, d_row, memory_config=ttnn.L1_MEMORY_CONFIG)  # [B, H, K, V]
    h = ttnn.add(h, outer, memory_config=ttnn.L1_MEMORY_CONFIG)

    # 5. Query state via matmul: o_t = q^T @ h
    q_row = ttnn.reshape(q_t, [B, H, 1, K])  # [B, H, 1, K]
    o_t = ttnn.matmul(q_row, h, memory_config=ttnn.L1_MEMORY_CONFIG)  # [B, H, 1, V]
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

    q = ttnn.multiply(q, scale, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Transpose to [B, H, T, D] for head-first processing
    q = ttnn.transpose(q, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.transpose(k, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.transpose(v, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    beta = ttnn.transpose(beta, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)  # [B, H, T]
    g = ttnn.transpose(g, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)  # [B, H, T]

    # Cast to float32 for recurrent precision (matches torch reference)
    q = ttnn.typecast(q, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.typecast(k, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.typecast(v, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
    beta = ttnn.typecast(beta, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
    g = ttnn.typecast(g, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Initialize state in float32
    if initial_state is not None:
        h = ttnn.typecast(initial_state, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        h = ttnn.zeros([B, H, K, V], device=device, dtype=ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)

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
    o = ttnn.concat(outputs_4d, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)  # [B, H, T, V]

    # Transpose back to [B, T, H, V]
    o = ttnn.transpose(o, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Cast back to bfloat16
    o = ttnn.typecast(o, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

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
    q = ttnn.typecast(
        ttnn.transpose(q, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG), ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    k = ttnn.typecast(
        ttnn.transpose(k, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG), ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    v = ttnn.typecast(
        ttnn.transpose(v, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG), ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    beta = ttnn.typecast(
        ttnn.transpose(beta, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG),
        ttnn.float32,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    g = ttnn.typecast(
        ttnn.transpose(g, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG), ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    q = ttnn.multiply(q, scale, memory_config=ttnn.L1_MEMORY_CONFIG)

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
            [
                q,
                ttnn.zeros(
                    [BH, pad_len, K],
                    device=device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                ),
            ],
            dim=1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        k = ttnn.concat(
            [
                k,
                ttnn.zeros(
                    [BH, pad_len, K],
                    device=device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                ),
            ],
            dim=1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        v = ttnn.concat(
            [
                v,
                ttnn.zeros(
                    [BH, pad_len, V],
                    device=device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                ),
            ],
            dim=1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        beta_flat = ttnn.concat(
            [
                beta_flat,
                ttnn.zeros(
                    [BH, pad_len, 1],
                    device=device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                ),
            ],
            dim=1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        g_3d = ttnn.reshape(g, [BH, T, 1])
        g_3d = ttnn.concat(
            [
                g_3d,
                ttnn.zeros(
                    [BH, pad_len, 1],
                    device=device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                ),
            ],
            dim=1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        g = ttnn.reshape(g_3d, [BH, L])
        beta_flat = ttnn.reshape(beta_flat, [BH, L, 1])
    else:
        beta_flat = ttnn.reshape(beta_flat, [BH, L, 1])

    # v_beta = v * beta, k_beta = k * beta
    v_beta = ttnn.multiply(v, beta_flat, memory_config=ttnn.L1_MEMORY_CONFIG)
    k_beta = ttnn.multiply(k, beta_flat, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Reshape into chunks: [BH*nc, cs, D]
    # Use L1 memory config for reshapes to minimize transfer overhead
    q_c = ttnn.reshape(q, [batch, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    k_c = ttnn.reshape(k, [batch, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    v_c = ttnn.reshape(v, [batch, chunk_size, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    k_beta_c = ttnn.reshape(k_beta, [batch, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    v_beta_c = ttnn.reshape(v_beta, [batch, chunk_size, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    g_c = ttnn.reshape(g, [batch, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)

    # --- Cumsum via matmul with upper-triangular ones ---
    triu_torch = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.float32))
    triu_ones = ttnn.from_torch(
        triu_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    triu_ones = ttnn.reshape(triu_ones, [1, chunk_size, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)

    g_c_3d = ttnn.reshape(g_c, [batch, 1, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)
    decay = ttnn.reshape(
        ttnn.matmul(g_c_3d, triu_ones, memory_config=ttnn.L1_MEMORY_CONFIG),
        [batch, chunk_size],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # decay_exp for weighting k_beta: [batch, cs, 1]
    decay_exp = ttnn.reshape(
        ttnn.exp(decay, memory_config=ttnn.L1_MEMORY_CONFIG),
        [batch, chunk_size, 1],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # --- L_mask: exp(decay_i - decay_j) for j <= i ---
    decay_col = ttnn.reshape(decay, [batch, chunk_size, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    decay_row = ttnn.reshape(decay, [batch, 1, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)
    L_diff = ttnn.subtract(decay_col, decay_row, memory_config=ttnn.L1_MEMORY_CONFIG)

    tril_torch = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32))
    tril_mask = ttnn.from_torch(
        tril_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tril_mask = ttnn.reshape(tril_mask, [1, chunk_size, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)

    L_diff_masked = ttnn.multiply(L_diff, tril_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
    L_mask = ttnn.multiply(
        ttnn.exp(L_diff_masked, memory_config=ttnn.L1_MEMORY_CONFIG), tril_mask, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # --- Intra-chunk interaction matrix M ---
    # Optimize: (batch, chunk_size, K) @ (batch, K, chunk_size) -> (batch, chunk_size, chunk_size)
    # Shape: chunk_size x K x chunk_size (e.g., 64 x 64 x 64)
    k_c_t = ttnn.transpose(k_c, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    prog_config_kk = _get_matmul_program_config(chunk_size, K, chunk_size, grid_size=None)
    if prog_config_kk:
        kk = ttnn.matmul(k_beta_c, k_c_t, program_config=prog_config_kk, memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        kk = ttnn.matmul(k_beta_c, k_c_t, memory_config=ttnn.L1_MEMORY_CONFIG)

    M = ttnn.neg(ttnn.multiply(kk, L_mask, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.L1_MEMORY_CONFIG)
    strict_lower_torch = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32), diagonal=-1)
    strict_lower = ttnn.from_torch(
        strict_lower_torch,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    strict_lower = ttnn.reshape(strict_lower, [1, chunk_size, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)
    M = ttnn.multiply(M, strict_lower, memory_config=ttnn.L1_MEMORY_CONFIG)

    # --- Woodbury via repeated-squaring Neumann series ---
    # Compute (I - M)^{-1} = I + M + M^2 + ... for nilpotent M
    eye_torch = torch.eye(chunk_size, dtype=torch.float32)
    eye = ttnn.from_torch(
        eye_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    eye = ttnn.reshape(eye, [1, chunk_size, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)

    R = ttnn.add(M, eye, memory_config=ttnn.L1_MEMORY_CONFIG)
    P = ttnn.matmul(M, M, memory_config=ttnn.L1_MEMORY_CONFIG)
    num_steps = max(int(math.ceil(math.log2(max(chunk_size, 2)))) - 1, 0)
    for _ in range(num_steps):
        R = ttnn.add(R, ttnn.matmul(R, P, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.L1_MEMORY_CONFIG)
        P = ttnn.matmul(P, P, memory_config=ttnn.L1_MEMORY_CONFIG)

    attn = R

    # --- Corrected values and keys ---
    # Optimize: (batch, chunk_size, chunk_size) @ (batch, chunk_size, V) -> (batch, chunk_size, V)
    # Shape: chunk_size x chunk_size x V (e.g., 64 x 64 x 256)
    prog_config_vcorr = _get_matmul_program_config(chunk_size, chunk_size, V, grid_size=None)
    if prog_config_vcorr:
        v_corrected = ttnn.matmul(attn, v_beta_c, program_config=prog_config_vcorr, memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        v_corrected = ttnn.matmul(attn, v_beta_c, memory_config=ttnn.L1_MEMORY_CONFIG)

    k_beta_decay = ttnn.multiply(k_beta_c, decay_exp, memory_config=ttnn.L1_MEMORY_CONFIG)
    if prog_config_vcorr:
        k_cumdecay = ttnn.matmul(
            attn, k_beta_decay, program_config=prog_config_vcorr, memory_config=ttnn.L1_MEMORY_CONFIG
        )
    else:
        k_cumdecay = ttnn.matmul(attn, k_beta_decay, memory_config=ttnn.L1_MEMORY_CONFIG)

    # --- Cross-chunk recurrence ---
    # Use L1 memory config for reshapes to minimize transfer overhead
    q_c_4d = ttnn.reshape(q_c, [BH, num_chunks, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    k_c_4d = ttnn.reshape(k_c, [BH, num_chunks, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    v_cor_4d = ttnn.reshape(v_corrected, [BH, num_chunks, chunk_size, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    k_cum_4d = ttnn.reshape(k_cumdecay, [BH, num_chunks, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    L_mask_4d = ttnn.reshape(L_mask, [BH, num_chunks, chunk_size, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)
    decay_3d = ttnn.reshape(decay, [BH, num_chunks, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)

    # Precompute total decay per chunk (= cumsum at last position)
    decay_last = ttnn.reshape(
        ttnn.sum(g_c, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG),
        [BH, num_chunks, 1],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    lower_causal_torch = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32))
    lower_causal = ttnn.from_torch(
        lower_causal_torch,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    S = ttnn.zeros(
        [BH, K, V], device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    if initial_state is not None:
        S = ttnn.typecast(
            ttnn.reshape(initial_state, [BH, K, V], memory_config=ttnn.L1_MEMORY_CONFIG),
            ttnn.float32,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    # Pre-compute program_configs outside the loop to avoid recreation overhead
    # This reduces op-to-op gaps caused by program_config creation
    prog_config_qk = _get_matmul_program_config(chunk_size, K, chunk_size, grid_size=None)
    prog_config_vprime = _get_matmul_program_config(chunk_size, K, V, grid_size=None)
    prog_config_o_inter = _get_matmul_program_config(chunk_size, K, V, grid_size=None)
    prog_config_intra = _get_matmul_program_config(chunk_size, chunk_size, V, grid_size=None)
    prog_config_state = _get_matmul_program_config(K, chunk_size, V, grid_size=None)

    # Optional: Pre-compute core grids for binary operations
    # Note: Binary operations typically work well with auto-config, but manual grids
    # can be specified for consistency or experimentation. For element-wise ops,
    # auto-config usually performs well, so this is optional.
    # To enable manual core grids, uncomment and set a grid size, e.g.:
    # binary_core_grid = _get_binary_core_grid((8, 4))  # 32 cores (8x4 grid)
    # For most cases, auto-config (None) performs well for binary operations.
    binary_core_grid = None

    outputs = []
    for i in range(num_chunks):
        # Slice and re-tilize (slicing from 4D may lose TILE_LAYOUT)
        # Using L1_MEMORY_CONFIG to minimize transfer overhead
        q_i = ttnn.to_layout(q_c_4d[:, i], ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        k_i = ttnn.to_layout(k_c_4d[:, i], ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        v_i = ttnn.to_layout(v_cor_4d[:, i], ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        k_cum_i = ttnn.to_layout(k_cum_4d[:, i], ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        L_mask_i = ttnn.to_layout(L_mask_4d[:, i], ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        decay_i = decay_3d[:, i]

        # Intra-chunk attention: (q @ k^T) * L_mask, lower-triangular
        # Optimize: (BH, chunk_size, K) @ (BH, K, chunk_size) -> (BH, chunk_size, chunk_size)
        # Shape: chunk_size x K x chunk_size (e.g., 64 x 64 x 64)
        k_i_t = ttnn.transpose(k_i, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
        # Use pre-computed program_config to avoid recreation overhead
        if prog_config_qk:
            qk = ttnn.matmul(q_i, k_i_t, program_config=prog_config_qk, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            qk = ttnn.matmul(q_i, k_i_t, memory_config=ttnn.L1_MEMORY_CONFIG)
        # Combine two multiplies into one by pre-multiplying masks
        # This reduces one BinaryNgDeviceOperation (reduces from 46 to fewer ops)
        # Optional: Use manual core grid for binary operations (typically auto-config works well)
        multiply_kwargs = {"memory_config": ttnn.L1_MEMORY_CONFIG}
        if binary_core_grid is not None:
            multiply_kwargs["sub_core_grids"] = binary_core_grid
        combined_mask = ttnn.multiply(L_mask_i, lower_causal, **multiply_kwargs)
        intra_attn = ttnn.multiply(qk, combined_mask, **multiply_kwargs)

        # Cross-chunk: read from state
        # Optimize: (BH, chunk_size, K) @ (BH, K, V) -> (BH, chunk_size, V)
        # Shape: chunk_size x K x V (e.g., 64 x 64 x 256)
        # Use pre-computed program_config to avoid recreation overhead
        if prog_config_vprime:
            v_prime = ttnn.matmul(k_cum_i, S, program_config=prog_config_vprime, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            v_prime = ttnn.matmul(k_cum_i, S, memory_config=ttnn.L1_MEMORY_CONFIG)
        # Optional: Use manual core grid for binary operations
        subtract_kwargs = {"memory_config": ttnn.L1_MEMORY_CONFIG}
        if binary_core_grid is not None:
            subtract_kwargs["sub_core_grids"] = binary_core_grid
        v_new = ttnn.subtract(v_i, v_prime, **subtract_kwargs)

        decay_i_exp = ttnn.reshape(
            ttnn.exp(decay_i, memory_config=ttnn.L1_MEMORY_CONFIG),
            [BH, chunk_size, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # Optimize: (BH, chunk_size, K) @ (BH, K, V) -> (BH, chunk_size, V)
        # Shape: chunk_size x K x V (e.g., 64 x 64 x 256)
        q_decay = ttnn.multiply(q_i, decay_i_exp, **multiply_kwargs)
        # Use pre-computed program_config to avoid recreation overhead
        if prog_config_o_inter:
            o_inter = ttnn.matmul(q_decay, S, program_config=prog_config_o_inter, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            o_inter = ttnn.matmul(q_decay, S, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Optimize: (BH, chunk_size, chunk_size) @ (BH, chunk_size, V) -> (BH, chunk_size, V)
        # Shape: chunk_size x chunk_size x V (e.g., 64 x 64 x 256)
        # Use pre-computed program_config to avoid recreation overhead
        if prog_config_intra:
            intra_v = ttnn.matmul(
                intra_attn, v_new, program_config=prog_config_intra, memory_config=ttnn.L1_MEMORY_CONFIG
            )
        else:
            intra_v = ttnn.matmul(intra_attn, v_new, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Optional: Use manual core grid for binary operations
        add_kwargs = {"memory_config": ttnn.L1_MEMORY_CONFIG}
        if binary_core_grid is not None:
            add_kwargs["sub_core_grids"] = binary_core_grid
        o_i = ttnn.add(o_inter, intra_v, **add_kwargs)
        outputs.append(ttnn.reshape(o_i, [BH, 1, chunk_size, V], memory_config=ttnn.L1_MEMORY_CONFIG))

        # Update state
        dl_i = decay_last[:, i]
        dl_i_exp = ttnn.reshape(
            ttnn.exp(dl_i, memory_config=ttnn.L1_MEMORY_CONFIG), [BH, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        S = ttnn.multiply(S, dl_i_exp, **multiply_kwargs)

        dl_i_2d = ttnn.reshape(dl_i, [BH, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        decay_diff = ttnn.subtract(dl_i_2d, decay_i, **subtract_kwargs)
        k_decay = ttnn.multiply(
            k_i,
            ttnn.reshape(
                ttnn.exp(decay_diff, memory_config=ttnn.L1_MEMORY_CONFIG),
                [BH, chunk_size, 1],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            ),
            **multiply_kwargs,
        )
        k_decay_t = ttnn.transpose(k_decay, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
        # Optimize: (BH, K, chunk_size) @ (BH, chunk_size, V) -> (BH, K, V)
        # Shape: K x chunk_size x V (e.g., 64 x 64 x 256)
        # Use pre-computed program_config to avoid recreation overhead
        if prog_config_state:
            state_update = ttnn.matmul(
                k_decay_t, v_new, program_config=prog_config_state, memory_config=ttnn.L1_MEMORY_CONFIG
            )
        else:
            state_update = ttnn.matmul(k_decay_t, v_new, memory_config=ttnn.L1_MEMORY_CONFIG)
        S = ttnn.add(S, state_update, **add_kwargs)

    o = ttnn.concat(outputs, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
    o = ttnn.reshape(o, [BH, L, V], memory_config=ttnn.L1_MEMORY_CONFIG)

    if pad_len > 0:
        o = o[:, :T]

    o = ttnn.reshape(o, [B, H, T, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    o = ttnn.transpose(o, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    o = ttnn.typecast(o, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

    final_state = ttnn.reshape(S, [B, H, K, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    return o, final_state
