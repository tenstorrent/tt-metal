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


def _create_eye_matrix_ttnn(size, device, dtype=ttnn.float32, memory_config=None):
    """Create identity matrix directly on device using TTNN operations.

    Args:
        size: Size of the square identity matrix
        device: TTNN device
        dtype: Data type (default: ttnn.float32)
        memory_config: Memory configuration (default: L1_MEMORY_CONFIG)

    Returns:
        TTNN tensor of shape [size, size] with identity matrix
    """
    ones = ttnn.ones(
        shape=(size, size),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    eye = ttnn.tril(ttnn.triu(ones, diagonal=0), diagonal=0, memory_config=memory_config)
    return eye


def _create_triu_ones_ttnn(size, device, dtype=ttnn.float32, memory_config=None):
    """Create upper triangular ones matrix directly on device using TTNN operations.

    Args:
        size: Size of the square matrix
        device: TTNN device
        dtype: Data type (default: ttnn.float32)
        memory_config: Memory configuration (default: L1_MEMORY_CONFIG)

    Returns:
        TTNN tensor of shape [size, size] with upper triangular ones
    """
    ones = ttnn.ones(
        shape=(size, size),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    triu_ones = ttnn.triu(ones, diagonal=0, memory_config=memory_config)
    return triu_ones


def _create_tril_ones_ttnn(size, device, dtype=ttnn.float32, memory_config=None):
    """Create lower triangular ones matrix directly on device using TTNN operations.

    Args:
        size: Size of the square matrix
        device: TTNN device
        dtype: Data type (default: ttnn.float32)
        memory_config: Memory configuration (default: L1_MEMORY_CONFIG)

    Returns:
        TTNN tensor of shape [size, size] with lower triangular ones
    """
    ones = ttnn.ones(
        shape=(size, size),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    tril_ones = ttnn.tril(ones, diagonal=0, memory_config=memory_config)
    return tril_ones


def _create_strict_lower_tril_ttnn(size, device, dtype=ttnn.float32, memory_config=None):
    """Create strict lower triangular ones matrix (diagonal=-1) directly on device.

    Args:
        size: Size of the square matrix
        device: TTNN device
        dtype: Data type (default: ttnn.float32)
        memory_config: Memory configuration (default: L1_MEMORY_CONFIG)

    Returns:
        TTNN tensor of shape [size, size] with strict lower triangular ones (diagonal excluded)
    """
    ones = ttnn.ones(
        shape=(size, size),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    strict_lower = ttnn.tril(ones, diagonal=-1, memory_config=memory_config)
    return strict_lower


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

    if m < 32 or n < 32 or k < 32:
        return None

    m_tiles = math.ceil(m / TILE_SIZE)
    n_tiles = math.ceil(n / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)

    if grid_size is None:
        total_work = m * n * k

        if total_work < 32768:
            return None

        if m_tiles == 2 and n_tiles == 2:
            return None
        elif m_tiles == 2 and n_tiles == 4:
            return None
        elif m_tiles == 2 and n_tiles == 8:
            return None
        elif m_tiles == 4 and n_tiles == 2:
            return None
        elif m_tiles == 4 and n_tiles == 8:
            grid_size = (8, 4)
        elif m_tiles <= 4 and n_tiles <= 4:
            if n_tiles >= 4:
                grid_size = (2, 1)
            elif m_tiles >= 4:
                grid_size = (1, 2)
            else:
                if total_work >= 262144:
                    grid_size = (2, 2)
                else:
                    grid_size = (2, 1)
        elif m_tiles >= 4 and n_tiles >= 8:
            cores_y = min(4, m_tiles)
            cores_x = min(8, n_tiles)
            grid_size = (cores_x, cores_y)
        else:
            if n_tiles >= 4:
                total_cores = min(8, max(2, n_tiles))
                grid_size = (total_cores, 1)
            elif m_tiles >= 4:
                total_cores = min(8, max(2, m_tiles))
                grid_size = (1, total_cores)
            else:
                return None

    cores_x, cores_y = grid_size

    per_core_M = math.ceil(m_tiles / cores_y)
    per_core_N = math.ceil(n_tiles / cores_x)
    per_core_M = max(1, per_core_M)
    per_core_N = max(1, per_core_N)

    if in0_block_w is None:
        k_per_core = math.ceil(k_tiles / cores_x) if cores_x > 1 else k_tiles
        in0_block_w = min(4, max(1, k_per_core))
        while k_tiles % (in0_block_w * cores_x) != 0 and in0_block_w > 1:
            in0_block_w -= 1
        if in0_block_w < 1:
            in0_block_w = 1

    max_subblock_size = 4
    out_subblock_h = min(per_core_M, max_subblock_size)
    out_subblock_w = min(per_core_N, max_subblock_size)

    if out_subblock_h * out_subblock_w > max_subblock_size:
        if out_subblock_h > out_subblock_w:
            out_subblock_h = max_subblock_size // out_subblock_w
        else:
            out_subblock_w = max_subblock_size // out_subblock_h

    while per_core_M % out_subblock_h != 0 and out_subblock_h > 1:
        out_subblock_h -= 1
    while per_core_N % out_subblock_w != 0 and out_subblock_w > 1:
        out_subblock_w -= 1

    if out_subblock_h < 1 or out_subblock_w < 1 or out_subblock_h * out_subblock_w > max_subblock_size:
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
        return None


def l2_norm_ttnn(x, dim=-1, eps=1e-6):
    """L2 normalization along a given dimension."""
    x_sq = ttnn.multiply(x, x, memory_config=None)
    norm_sq = ttnn.sum(x_sq, dim=dim, keepdim=True, memory_config=None)
    inv_norm = ttnn.rsqrt(ttnn.add(norm_sq, eps, memory_config=None), memory_config=None)
    return ttnn.multiply(x, inv_norm, memory_config=None)


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
    decay = ttnn.reshape(decay, [B, H, 1, 1], memory_config=None)

    # beta: [B, H] -> [B, H, 1, 1]
    beta_expanded = ttnn.reshape(beta_t, [B, H, 1, 1], memory_config=None)

    # k_t: [B, H, K] -> [B, H, K, 1]
    k_col = ttnn.reshape(k_t, [B, H, K, 1], memory_config=None)

    # delta: [B, H, V] -> [B, H, 1, V]
    d_row = ttnn.reshape(delta, [B, H, 1, V], memory_config=None)

    k_col = ttnn.to_layout(k_col, ttnn.TILE_LAYOUT)
    d_row = ttnn.to_layout(d_row, ttnn.TILE_LAYOUT)
    k_col = ttnn.to_memory_config(k_col, ttnn.DRAM_MEMORY_CONFIG)
    d_row = ttnn.to_memory_config(d_row, ttnn.DRAM_MEMORY_CONFIG)

    matmul_compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    outer = ttnn.matmul(
        k_col,
        d_row,
        memory_config=None,
        compute_kernel_config=matmul_compute_cfg,
        program_config=None,
    )

    # apply beta
    outer = ttnn.multiply(
        outer,
        beta_expanded,
        memory_config=None,
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

    q_t = ttnn.to_layout(q_t, ttnn.TILE_LAYOUT)
    k_t = ttnn.to_layout(k_t, ttnn.TILE_LAYOUT)
    v_t = ttnn.to_layout(v_t, ttnn.TILE_LAYOUT)
    h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)
    h = ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)

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

    k_row = ttnn.reshape(k_t, [B, H, 1, K], memory_config=None)
    k_row = ttnn.to_layout(k_row, ttnn.TILE_LAYOUT)
    k_row = ttnn.to_memory_config(k_row, ttnn.DRAM_MEMORY_CONFIG)
    v_read = ttnn.matmul(
        k_row,
        h,
        memory_config=None,
        program_config=read_query_prog_cfg,
        compute_kernel_config=read_query_compute_cfg,
    )
    v_read = ttnn.reshape(v_read, [B, H, V], memory_config=None)

    delta = ttnn.subtract(v_t, v_read, memory_config=None)

    h = fused_decay_and_write_ttnn(
        h=h,
        k_t=k_t,
        delta=delta,
        decay_t=decay_t,
        beta_t=beta_t,
        device=device,
    )

    q_row = ttnn.reshape(q_t, [B, H, 1, K], memory_config=None)
    q_row = ttnn.to_layout(q_row, ttnn.TILE_LAYOUT)
    q_row = ttnn.to_memory_config(q_row, ttnn.DRAM_MEMORY_CONFIG)
    o_t = ttnn.matmul(
        q_row,
        h,
        memory_config=None,
        program_config=read_query_prog_cfg,
        compute_kernel_config=read_query_compute_cfg,
    )
    use_l1 = seq_len is not None and seq_len <= 64
    o_t = ttnn.reshape(o_t, [B, H, V], memory_config=None if use_l1 else None)

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

    q = ttnn.multiply(q, scale, memory_config=None)

    q = ttnn.transpose(q, 1, 2, memory_config=None)
    k = ttnn.transpose(k, 1, 2, memory_config=None)
    v = ttnn.transpose(v, 1, 2, memory_config=None)
    beta = ttnn.transpose(beta, 1, 2, memory_config=None)
    g = ttnn.transpose(g, 1, 2, memory_config=None)

    q = ttnn.typecast(q, ttnn.bfloat16, memory_config=None)
    k = ttnn.typecast(k, ttnn.bfloat16, memory_config=None)
    v = ttnn.typecast(v, ttnn.bfloat16, memory_config=None)
    beta = ttnn.typecast(beta, ttnn.bfloat16, memory_config=None)
    g = ttnn.typecast(g, ttnn.bfloat16, memory_config=None)

    # Precompute exp(g) once and slice per timestep in the loop.
    g_exp = ttnn.exp(g)

    if initial_state is not None:
        h = ttnn.typecast(initial_state, ttnn.bfloat16, memory_config=None)
    else:
        h = ttnn.zeros([B, H, K, V], device=device, dtype=ttnn.bfloat16, memory_config=None)

    outputs = []
    for i in range(T):
        q_t = q[:, :, i]  # [B, H, K]
        k_t = k[:, :, i]  # [B, H, K]
        v_t = v[:, :, i]  # [B, H, V]
        beta_t = beta[:, :, i]  # [B, H]
        decay_t = g_exp[:, :, i]  # [B, H]

        o_t, h = recurrent_delta_rule_step_ttnn(q_t, k_t, v_t, beta_t, decay_t, h, seq_len=T, device=device)
        outputs.append(o_t)

    outputs_4d = [ttnn.reshape(o, [B, H, 1, V], memory_config=None) for o in outputs]
    o = ttnn.concat(outputs_4d, dim=2, memory_config=None)
    o = ttnn.transpose(o, 1, 2, memory_config=None)
    o = ttnn.typecast(o, ttnn.bfloat16, memory_config=None)

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

    q = ttnn.typecast(ttnn.transpose(q, 1, 2, memory_config=None), ttnn.float32, memory_config=None)
    k = ttnn.typecast(ttnn.transpose(k, 1, 2, memory_config=None), ttnn.float32, memory_config=None)
    v = ttnn.typecast(ttnn.transpose(v, 1, 2, memory_config=None), ttnn.float32, memory_config=None)
    beta = ttnn.typecast(
        ttnn.transpose(beta, 1, 2, memory_config=None),
        ttnn.float32,
        memory_config=None,
    )
    g = ttnn.typecast(ttnn.transpose(g, 1, 2, memory_config=None), ttnn.float32, memory_config=None)

    q = ttnn.multiply(q, scale, memory_config=None)

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
                    memory_config=None,
                ),
            ],
            dim=1,
            memory_config=None,
        )
        k = ttnn.concat(
            [
                k,
                ttnn.zeros(
                    [BH, pad_len, K],
                    device=device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=None,
                ),
            ],
            dim=1,
            memory_config=None,
        )
        v = ttnn.concat(
            [
                v,
                ttnn.zeros(
                    [BH, pad_len, V],
                    device=device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=None,
                ),
            ],
            dim=1,
            memory_config=None,
        )
        beta_flat = ttnn.concat(
            [
                beta_flat,
                ttnn.zeros(
                    [BH, pad_len, 1],
                    device=device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=None,
                ),
            ],
            dim=1,
            memory_config=None,
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
                    memory_config=None,
                ),
            ],
            dim=1,
            memory_config=None,
        )
        g = ttnn.reshape(g_3d, [BH, L])
        beta_flat = ttnn.reshape(beta_flat, [BH, L, 1])
    else:
        beta_flat = ttnn.reshape(beta_flat, [BH, L, 1])

    v_beta = ttnn.multiply(v, beta_flat, memory_config=None)
    k_beta = ttnn.multiply(k, beta_flat, memory_config=None)

    q_c = ttnn.reshape(q, [batch, chunk_size, K], memory_config=None)
    k_c = ttnn.reshape(k, [batch, chunk_size, K], memory_config=None)
    v_c = ttnn.reshape(v, [batch, chunk_size, V], memory_config=None)
    k_beta_c = ttnn.reshape(k_beta, [batch, chunk_size, K], memory_config=None)
    v_beta_c = ttnn.reshape(v_beta, [batch, chunk_size, V], memory_config=None)
    g_c = ttnn.reshape(g, [batch, chunk_size], memory_config=None)

    triu_ones = _create_triu_ones_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=None)
    triu_ones = ttnn.reshape(triu_ones, [1, chunk_size, chunk_size], memory_config=None)

    g_c_3d = ttnn.reshape(g_c, [batch, 1, chunk_size], memory_config=None)
    decay = ttnn.reshape(
        ttnn.matmul(g_c_3d, triu_ones, memory_config=None),
        [batch, chunk_size],
        memory_config=None,
    )

    decay_exp = ttnn.reshape(
        ttnn.exp(decay, memory_config=None),
        [batch, chunk_size, 1],
        memory_config=None,
    )

    decay_col = ttnn.reshape(decay, [batch, chunk_size, 1], memory_config=None)
    decay_row = ttnn.reshape(decay, [batch, 1, chunk_size], memory_config=None)
    L_diff = ttnn.subtract(decay_col, decay_row, memory_config=None)

    tril_mask = _create_tril_ones_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=None)
    tril_mask = ttnn.reshape(tril_mask, [1, chunk_size, chunk_size], memory_config=None)

    L_diff_masked = ttnn.multiply(L_diff, tril_mask, memory_config=None)
    L_mask = ttnn.multiply(ttnn.exp(L_diff_masked, memory_config=None), tril_mask, memory_config=None)

    k_c_t = ttnn.transpose(k_c, 1, 2, memory_config=None)
    prog_config_kk = _get_matmul_program_config(chunk_size, K, chunk_size, grid_size=None)
    if prog_config_kk:
        kk = ttnn.matmul(k_beta_c, k_c_t, program_config=prog_config_kk, memory_config=None)
    else:
        kk = ttnn.matmul(k_beta_c, k_c_t, memory_config=None)

    M = ttnn.neg(ttnn.multiply(kk, L_mask, memory_config=None), memory_config=None)
    strict_lower = _create_strict_lower_tril_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=None)
    strict_lower = ttnn.reshape(strict_lower, [1, chunk_size, chunk_size], memory_config=None)
    M = ttnn.multiply(M, strict_lower, memory_config=None)

    eye = _create_eye_matrix_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=None)
    eye = ttnn.reshape(eye, [1, chunk_size, chunk_size], memory_config=None)

    R = ttnn.add(M, eye, memory_config=None)
    P = ttnn.matmul(M, M, memory_config=None)
    num_steps = max(int(math.ceil(math.log2(max(chunk_size, 2)))) - 1, 0)
    prog_config_woodbury = _get_matmul_program_config(chunk_size, chunk_size, chunk_size, grid_size=None)
    for _ in range(num_steps):
        if prog_config_woodbury:
            R = ttnn.add(
                R,
                ttnn.matmul(R, P, program_config=prog_config_woodbury, memory_config=None),
                memory_config=None,
            )
            P = ttnn.matmul(P, P, program_config=prog_config_woodbury, memory_config=None)
        else:
            R = ttnn.add(R, ttnn.matmul(R, P, memory_config=None), memory_config=None)
            P = ttnn.matmul(P, P, memory_config=None)

    attn = R

    prog_config_vcorr = _get_matmul_program_config(chunk_size, chunk_size, V, grid_size=None)
    if prog_config_vcorr:
        v_corrected = ttnn.matmul(attn, v_beta_c, program_config=prog_config_vcorr, memory_config=None)
    else:
        v_corrected = ttnn.matmul(attn, v_beta_c, memory_config=None)

    k_beta_decay = ttnn.multiply(k_beta_c, decay_exp, memory_config=None)
    if prog_config_vcorr:
        k_cumdecay = ttnn.matmul(attn, k_beta_decay, program_config=prog_config_vcorr, memory_config=None)
    else:
        k_cumdecay = ttnn.matmul(attn, k_beta_decay, memory_config=None)

    q_c_4d = ttnn.reshape(q_c, [BH, num_chunks, chunk_size, K], memory_config=None)
    q_c_4d = ttnn.to_layout(q_c_4d, ttnn.TILE_LAYOUT, memory_config=None)
    k_c_4d = ttnn.reshape(k_c, [BH, num_chunks, chunk_size, K], memory_config=None)
    k_c_4d = ttnn.to_layout(k_c_4d, ttnn.TILE_LAYOUT, memory_config=None)
    v_cor_4d = ttnn.reshape(v_corrected, [BH, num_chunks, chunk_size, V], memory_config=None)
    v_cor_4d = ttnn.to_layout(v_cor_4d, ttnn.TILE_LAYOUT, memory_config=None)
    k_cum_4d = ttnn.reshape(k_cumdecay, [BH, num_chunks, chunk_size, K], memory_config=None)
    k_cum_4d = ttnn.to_layout(k_cum_4d, ttnn.TILE_LAYOUT, memory_config=None)
    L_mask_4d = ttnn.reshape(L_mask, [BH, num_chunks, chunk_size, chunk_size], memory_config=None)
    L_mask_4d = ttnn.to_layout(L_mask_4d, ttnn.TILE_LAYOUT, memory_config=None)
    decay_3d = ttnn.reshape(decay, [BH, num_chunks, chunk_size], memory_config=None)

    decay_last = ttnn.reshape(
        ttnn.sum(g_c, dim=-1, memory_config=None),
        [BH, num_chunks, 1],
        memory_config=None,
    )

    lower_causal = _create_tril_ones_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=None)

    S = ttnn.zeros([BH, K, V], device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=None)
    if initial_state is not None:
        S = ttnn.typecast(
            ttnn.reshape(initial_state, [BH, K, V], memory_config=None),
            ttnn.float32,
            memory_config=None,
        )

    prog_config_qk = _get_matmul_program_config(chunk_size, K, chunk_size, grid_size=None)
    prog_config_vprime = _get_matmul_program_config(chunk_size, K, V, grid_size=None)
    prog_config_o_inter = _get_matmul_program_config(chunk_size, K, V, grid_size=None)
    prog_config_intra = _get_matmul_program_config(chunk_size, chunk_size, V, grid_size=None)
    prog_config_state = _get_matmul_program_config(K, chunk_size, V, grid_size=None)

    outputs = []
    for i in range(num_chunks):
        q_i = q_c_4d[:, i]
        k_i = k_c_4d[:, i]
        v_i = v_cor_4d[:, i]
        k_cum_i = k_cum_4d[:, i]
        L_mask_i = L_mask_4d[:, i]
        decay_i = decay_3d[:, i]

        k_i_t = ttnn.transpose(k_i, 1, 2, memory_config=None)
        if prog_config_qk:
            qk = ttnn.matmul(q_i, k_i_t, program_config=prog_config_qk, memory_config=None)
        else:
            qk = ttnn.matmul(q_i, k_i_t, memory_config=None)
        combined_mask = ttnn.multiply(L_mask_i, lower_causal, memory_config=None)
        intra_attn = ttnn.multiply(qk, combined_mask, memory_config=None)

        if prog_config_vprime:
            v_prime = ttnn.matmul(k_cum_i, S, program_config=prog_config_vprime, memory_config=None)
        else:
            v_prime = ttnn.matmul(k_cum_i, S, memory_config=None)
        v_new = ttnn.subtract(v_i, v_prime, memory_config=None)

        decay_i_exp = ttnn.reshape(
            ttnn.exp(decay_i, memory_config=None),
            [BH, chunk_size, 1],
            memory_config=None,
        )
        q_decay = ttnn.multiply(q_i, decay_i_exp, memory_config=None)
        if prog_config_o_inter:
            o_inter = ttnn.matmul(q_decay, S, program_config=prog_config_o_inter, memory_config=None)
        else:
            o_inter = ttnn.matmul(q_decay, S, memory_config=None)

        if prog_config_intra:
            intra_v = ttnn.matmul(intra_attn, v_new, program_config=prog_config_intra, memory_config=None)
        else:
            intra_v = ttnn.matmul(intra_attn, v_new, memory_config=None)

        o_i = ttnn.add(o_inter, intra_v, memory_config=None)
        outputs.append(ttnn.reshape(o_i, [BH, 1, chunk_size, V], memory_config=None))

        dl_i = decay_last[:, i]
        dl_i_exp = ttnn.exp(dl_i, memory_config=None)
        S = ttnn.multiply(
            S,
            ttnn.reshape(dl_i_exp, [BH, 1, 1], memory_config=None),
            memory_config=None,
        )

        decay_diff = ttnn.subtract(
            ttnn.reshape(dl_i, [BH, 1], memory_config=None),
            decay_i,
            memory_config=None,
        )
        decay_diff_exp = ttnn.exp(decay_diff, memory_config=None)
        k_decay = ttnn.multiply(
            k_i,
            ttnn.reshape(decay_diff_exp, [BH, chunk_size, 1], memory_config=None),
            memory_config=None,
        )
        k_decay_t = ttnn.transpose(k_decay, 1, 2, memory_config=None)
        if prog_config_state:
            state_update = ttnn.matmul(k_decay_t, v_new, program_config=prog_config_state, memory_config=None)
        else:
            state_update = ttnn.matmul(k_decay_t, v_new, memory_config=None)
        S = ttnn.add(S, state_update, memory_config=None)

    o = ttnn.concat(outputs, dim=1, memory_config=None)

    if pad_len > 0:
        o = o[:, :T]
        o = ttnn.reshape(o, [BH, T, V], memory_config=None)
    else:
        o = ttnn.reshape(o, [BH, L, V], memory_config=None)

    o = ttnn.reshape(o, [B, H, T, V], memory_config=None)
    o = ttnn.transpose(o, 1, 2, memory_config=None)
    o = ttnn.typecast(o, ttnn.bfloat16, memory_config=None)

    final_state = ttnn.reshape(S, [B, H, K, V], memory_config=None)
    return o, final_state
