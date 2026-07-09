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

# Cached identity matrices for forward substitution (avoids per-call allocation)
_EYE_CACHE = {}


def _chunk_eye(size):
    """Return a cached [size, size] float32 identity matrix on CPU."""
    if size not in _EYE_CACHE:
        _EYE_CACHE[size] = torch.eye(size, dtype=torch.float32)
    return _EYE_CACHE[size]


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
    # Cap per-core M tiles to avoid L1 circular buffer overflow on Blackhole.
    # Float32 tiles are 4KB each. Double-buffered in0 + in1 + out + partials ≈ 18KB per M-tile.
    # At 2 M-tiles per core: ~40KB total, fits safely in per-core L1.
    # At 4 M-tiles: ~72KB, overflows (observed clash at 155648 vs 168448).
    MAX_PER_CORE_M = 2

    if m < 32 or n < 32 or k < 32:
        return None

    m_tiles = math.ceil(m / TILE_SIZE)
    n_tiles = math.ceil(n / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)

    # Large inner dimension (k > 256) overflows L1 circular buffers during
    # accumulation even with in0_block_w=1. Let TTNN auto-select.
    if k_tiles > 8:
        return None

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

    # If per_core_M exceeds the L1 safety limit, increase cores_y to reduce it
    # and force in0_block_w=1 to keep double-buffered circular buffers within L1.
    if per_core_M > MAX_PER_CORE_M:
        cores_y = math.ceil(m_tiles / MAX_PER_CORE_M)
        cores_y = min(cores_y, 8)  # max 8 cores along Y
        per_core_M = math.ceil(m_tiles / cores_y)
        if per_core_M > MAX_PER_CORE_M:
            return None  # Can't reduce below limit with 8 cores; let TTNN auto-select
        grid_size = (cores_x, cores_y)
        in0_block_w = 1  # prevent in0 buffer from scaling with k_tiles

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
    # Use DRAM for large tensors (T>512 produces tensors that don't fit in L1)
    T = x.shape[1] if len(x.shape) >= 3 else x.shape[0]
    mc = ttnn.L1_MEMORY_CONFIG if T <= 512 else ttnn.DRAM_MEMORY_CONFIG
    x_sq = ttnn.multiply(x, x, memory_config=mc)
    norm_sq = ttnn.sum(x_sq, dim=dim, keepdim=True, memory_config=mc)
    inv_norm = ttnn.rsqrt(ttnn.add(norm_sq, eps, memory_config=mc), memory_config=mc)
    return ttnn.multiply(x, inv_norm, memory_config=mc)


def fused_decay_and_write_ttnn(
    h,
    k_t,
    delta,
    decay_t,
    beta_t,
    device=None,
    apply_decay=True,
):
    """
    Logical fusion for the recurrent delta rule state update:

        h = decay * h + beta_t * (k_t ⊗ delta)

    Implemented using existing TTNN ops so call sites are stable.
    Can be replaced by a true fused kernel later.

    apply_decay=False: the caller has ALREADY decayed h (canonical gated-delta-rule
    order is decay -> read -> write, so the decay must happen before the read and must
    NOT be re-applied here). In that case this only adds the outer product: h = h + outer.
    """
    B = h.shape[0]
    H = h.shape[1]
    K = h.shape[2]
    V = h.shape[3]

    # decay: [B, H] -> [B, H, 1, 1]
    # decay_t is already exp(g_t) in BF16; no typecast needed.
    decay = ttnn.reshape(decay_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

    # beta: [B, H] -> [B, H, 1, 1]
    beta_expanded = ttnn.reshape(beta_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

    # k_t: [B, H, K] -> [B, H, K, 1]
    k_col = ttnn.reshape(k_t, [B, H, K, 1], memory_config=None)

    # delta: [B, H, V] -> [B, H, 1, V]
    d_row = ttnn.reshape(delta, [B, H, 1, V], memory_config=None)

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

    # fused-style update: decay * h + outer.
    # When apply_decay is False, h has already been decayed by the caller (canonical
    # decay -> read -> write order), so only the outer product is added here.
    if apply_decay:
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

    # Canonical gated-delta-rule order: DECAY the state BEFORE reading from it (matches the
    # FLA reference + chunked prefill). Reading the un-decayed state uses the wrong state in
    # the delta correction (v - k·h).
    decay_bhkv = ttnn.reshape(decay_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    h = ttnn.multiply(h, decay_bhkv)

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

    # Write the outer product WITHOUT re-decaying (h already decayed above).
    h = fused_decay_and_write_ttnn(
        h=h,
        k_t=k_t,
        delta=delta,
        decay_t=decay_t,
        beta_t=beta_t,
        device=device,
        apply_decay=False,
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


def recurrent_gated_delta_rule_decode_ttnn(
    q,
    k,
    v,
    beta,
    g,
    scale=None,
    initial_state=None,
    device=None,
    high_precision=False,
):
    """
    Optimized single-token (T=1) decode path for gated delta rule.

    Eliminates overhead from the general recurrent path:
    - No loop (T is always 1)
    - No slicing/concat for timesteps
    - Reduced transpose/typecast operations
    - Keeps state in current memory config

    Args:
        q: [B, 1, H, K]
        k: [B, 1, H, K]
        v: [B, 1, H, V]
        beta: [B, 1, H]
        g: [B, 1, H]
        scale: float
        initial_state: [B, H, K, V]
        device: ttnn device

    Returns:
        output: [B, 1, H, V]
        final_state: [B, H, K, V]
    """
    B = q.shape[0]
    H = q.shape[2]
    K = q.shape[3]
    V = v.shape[3]

    # high_precision=True: run the entire single-token recurrent step in fp32. The state update
    # h = decay * h + beta * (k ⊗ delta) compounds every decode step, and `decay = exp(g)` sits
    # near 1.0 where bf16 resolution is ~0.008 — so a bf16 decay quantizes the per-step forgetting
    # coarsely and the error accumulates over hundreds of steps (the long-decode "state saturation"
    # / repetition collapse). Casting q/k/v/beta/g (and the state below) to fp32 here keeps decay
    # and the accumulation exact; the read/write matmuls already accumulate in fp32
    # (fp32_dest_acc_en). Tensors are tiny at decode (T=1) so the cost is negligible. Default off
    # → all other callers/tests of this shared op are byte-unchanged.
    if high_precision:
        q = ttnn.typecast(q, ttnn.float32)
        k = ttnn.typecast(k, ttnn.float32)
        v = ttnn.typecast(v, ttnn.float32)
        beta = ttnn.typecast(beta, ttnn.float32)
        g = ttnn.typecast(g, ttnn.float32)

    # L2 norm
    q = l2_norm_ttnn(q, dim=-1)
    k = l2_norm_ttnn(k, dim=-1)

    if scale is None:
        scale = K**-0.5
    q = ttnn.multiply(q, scale, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Reshape directly to shapes needed by matmul (skip intermediate 3D form)
    # q and k need [B, H, 1, K] for matmul against h [B, H, K, V]
    # v needs [B, H, V] for subtract with v_read
    # Inputs are already TILE_LAYOUT from the caller, so skip redundant to_layout.
    q_row = ttnn.reshape(q, [B, H, 1, K], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k_row = ttnn.reshape(k, [B, H, 1, K], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_t = ttnn.reshape(v, [B, H, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    beta_t = ttnn.reshape(beta, [B, H], memory_config=ttnn.L1_MEMORY_CONFIG)
    g_t = ttnn.reshape(g, [B, H], memory_config=ttnn.L1_MEMORY_CONFIG)

    # Compute decay
    decay_t = ttnn.exp(g_t, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Ensure state is ready
    h = initial_state
    if h is None:
        h = ttnn.zeros(
            [B, H, K, V], device=device, dtype=ttnn.float32 if high_precision else ttnn.bfloat16, memory_config=None
        )
    elif high_precision and h.dtype != ttnn.float32:
        # fp32-exact step needs an fp32 state; cast if the caller kept a bf16 rec_state.
        h = ttnn.typecast(h, ttnn.float32)

    # Run single recurrent step
    # h is always TILE_LAYOUT: either from fused_decay_and_write_ttnn, ttnn.zeros with TILE_LAYOUT,
    # or _init_recurrent_state which uses ttnn.from_torch(..., layout=ttnn.TILE_LAYOUT)
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

    # Canonical gated-delta-rule order: DECAY the state BEFORE reading from it. The FLA
    # reference and the chunked prefill path both decay-then-read; reading the un-decayed
    # state makes the delta correction (v - k·h) use the wrong state.
    decay_bhkv = ttnn.reshape(decay_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    h = ttnn.multiply(h, decay_bhkv)

    # Read from the DECAYED state: v_read = k @ h  (k_row already [B,H,1,K] TILE_LAYOUT)
    v_read = ttnn.matmul(
        k_row, h, memory_config=None, program_config=read_query_prog_cfg, compute_kernel_config=read_query_compute_cfg
    )
    v_read = ttnn.reshape(v_read, [B, H, V], memory_config=None)

    # Delta and state update — write the outer product WITHOUT re-decaying (h already decayed).
    delta = ttnn.subtract(v_t, v_read, memory_config=None)
    k_t = ttnn.reshape(k_row, [B, H, K], memory_config=None)
    h = fused_decay_and_write_ttnn(
        h=h, k_t=k_t, delta=delta, decay_t=decay_t, beta_t=beta_t, device=device, apply_decay=False
    )

    # Query state: o = q @ h  (q_row already [B,H,1,K] TILE_LAYOUT)
    o_t = ttnn.matmul(
        q_row, h, memory_config=None, program_config=read_query_prog_cfg, compute_kernel_config=read_query_compute_cfg
    )

    # Reshape output to [B, 1, H, V]
    o = ttnn.reshape(o_t, [B, 1, H, V], memory_config=None)

    return o, h


def recurrent_gated_delta_rule_decode_inplace_ttnn(
    q,
    k,
    v,
    beta,
    g,
    state_buffer,
    scale=None,
    device=None,
    high_precision=False,
):
    """Like recurrent_gated_delta_rule_decode_ttnn but writes state back to pre-allocated buffer.

    For trace capture: state_buffer is a pre-allocated [B, H, K, V] tensor.
    After computing the new state, we copy it back to state_buffer so the
    trace replay reads the correct address on the next iteration.
    """
    o, new_h = recurrent_gated_delta_rule_decode_ttnn(
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g,
        scale=scale,
        initial_state=state_buffer,
        device=device,
        high_precision=high_precision,
    )
    # Copy new state back to pre-allocated buffer for trace consistency
    ttnn.copy(new_h, state_buffer)
    ttnn.deallocate(new_h)
    return o, state_buffer


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
