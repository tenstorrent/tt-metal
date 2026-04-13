# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Chunk-parallel DeltaNet recurrence ops for GDN prefill.

Ported from models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py
(qwen9b-p150 branch) for use with Qwen3.5-27B TP=4 mesh.
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


def create_chunk_masks(chunk_size, device):
    """Pre-create all mask matrices needed by chunk_gated_delta_rule_ttnn.

    Call once during model init and pass as `cached_masks` to avoid
    recreating these constant matrices on every forward call (24 layers × every prefill).
    """
    triu_ones = _create_triu_ones_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=None)
    triu_ones = ttnn.reshape(triu_ones, [1, chunk_size, chunk_size], memory_config=None)
    tril_mask = _create_tril_ones_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=None)
    tril_mask = ttnn.reshape(tril_mask, [1, chunk_size, chunk_size], memory_config=None)
    strict_lower = _create_strict_lower_tril_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=None)
    strict_lower = ttnn.reshape(strict_lower, [1, chunk_size, chunk_size], memory_config=None)
    eye = _create_eye_matrix_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=None)
    eye = ttnn.reshape(eye, [1, chunk_size, chunk_size], memory_config=None)
    lower_causal = _create_tril_ones_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=None)
    return {
        "triu_ones": triu_ones,
        "tril_mask": tril_mask,
        "strict_lower": strict_lower,
        "eye": eye,
        "lower_causal": lower_causal,
    }


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
    cached_masks=None,
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

    _dram = ttnn.DRAM_MEMORY_CONFIG if T > 512 else ttnn.L1_MEMORY_CONFIG
    q = ttnn.typecast(ttnn.transpose(q, 1, 2, memory_config=_dram), ttnn.float32, memory_config=_dram)
    k = ttnn.typecast(ttnn.transpose(k, 1, 2, memory_config=_dram), ttnn.float32, memory_config=_dram)
    v = ttnn.typecast(ttnn.transpose(v, 1, 2, memory_config=_dram), ttnn.float32, memory_config=_dram)
    beta = ttnn.typecast(
        ttnn.transpose(beta, 1, 2, memory_config=_dram),
        ttnn.float32,
        memory_config=_dram,
    )
    g = ttnn.typecast(ttnn.transpose(g, 1, 2, memory_config=_dram), ttnn.float32, memory_config=_dram)

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
    ttnn.deallocate(beta)
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
        ttnn.deallocate(g)
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
    del beta_flat

    q_c = ttnn.reshape(q, [batch, chunk_size, K], memory_config=None)
    k_c = ttnn.reshape(k, [batch, chunk_size, K], memory_config=None)
    del v
    k_beta_c = ttnn.reshape(k_beta, [batch, chunk_size, K], memory_config=None)
    v_beta_c = ttnn.reshape(v_beta, [batch, chunk_size, V], memory_config=None)
    g_c = ttnn.reshape(g, [batch, chunk_size], memory_config=None)
    del q, v_beta, k_beta

    # Use cached masks if available, otherwise create them
    if cached_masks is not None:
        triu_ones = cached_masks["triu_ones"]
        tril_mask = cached_masks["tril_mask"]
    else:
        triu_ones = _create_triu_ones_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=None)
        triu_ones = ttnn.reshape(triu_ones, [1, chunk_size, chunk_size], memory_config=None)
        tril_mask = _create_tril_ones_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=None)
        tril_mask = ttnn.reshape(tril_mask, [1, chunk_size, chunk_size], memory_config=None)

    g_c_3d = ttnn.reshape(g_c, [batch, 1, chunk_size], memory_config=None)
    decay = ttnn.reshape(
        ttnn.matmul(g_c_3d, triu_ones, memory_config=None),
        [batch, chunk_size],
        memory_config=None,
    )

    # Per-chunk normalization: subtract first element so cumsum starts at 0.
    # This keeps values in [-decay_range, 0] instead of starting at a large negative.
    # Used for Site 1 (L_diff mask) and Site 5 (decay differences) where offset cancels.
    # Sites 2 and 3 use decay_raw (un-normalized) since they need absolute decay
    # values for correct state interaction (exp(raw_decay) scales against inter-chunk state S).
    decay_offset = decay[:, 0:1]  # [batch, 1]
    decay_raw = decay  # save raw cumsum before normalization
    decay = ttnn.subtract(decay_raw, decay_offset, memory_config=None)  # normalized: starts at 0
    ttnn.deallocate(decay_offset)

    # Site 2: key scaling needs raw (absolute) decay for state correction term.
    # k_cumdecay = R @ (k_beta * decay_exp) feeds into v_prime = k_cumdecay @ S,
    # and S carries absolute inter-chunk decay. Using normalized here would be off by exp(offset).
    decay_exp = ttnn.reshape(
        ttnn.exp(ttnn.clip(decay_raw, min=-20.0, max=0.0), memory_config=None),
        [batch, chunk_size, 1],
        memory_config=None,
    )

    # For large chunk sizes, tensors shaped [batch, chunk_size, chunk_size] (e.g. 512MB at
    # chunk_size=2048) overflow L1 when auto-placed alongside matmul circular buffers.
    # Force DRAM placement for large batched 4D tensors in the pre-loop section.
    _batch_mc = ttnn.DRAM_MEMORY_CONFIG if chunk_size > 64 else None
    # Inner loop per-iteration tensors are small ([BH, chunk_size, K/V] ≈ 2MB = 512 tiles,
    # ~4 tiles per L1 bank across 130 banks). L1 gives ~10-20x lower latency than DRAM
    # for these memory-bound small matmuls that run on only 2-8 cores.
    _loop_mc = ttnn.L1_MEMORY_CONFIG if chunk_size > 64 else None

    # HiFi2 + fp32 accumulation for all chunk matmuls. The default compute kernel uses
    # lower fidelity, which introduces per-matmul rounding errors that compound across
    # the Neumann series iterations and inter-chunk state propagation steps. The recurrent
    # step already uses HiFi2 + fp32_dest_acc_en — match that precision here.
    _hifi_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    decay_col = ttnn.reshape(decay, [batch, chunk_size, 1], memory_config=None)
    decay_row = ttnn.reshape(decay, [batch, 1, chunk_size], memory_config=None)
    L_diff = ttnn.subtract(decay_col, decay_row, memory_config=_batch_mc)
    del decay_col, decay_row

    # Clamp before exp to prevent overflow/underflow
    L_diff_masked = ttnn.multiply(L_diff, tril_mask, memory_config=_batch_mc)
    ttnn.deallocate(L_diff)
    L_diff_clamped = ttnn.clip(L_diff_masked, min=-20.0, max=0.0)
    ttnn.deallocate(L_diff_masked)
    L_mask = ttnn.multiply(ttnn.exp(L_diff_clamped, memory_config=_batch_mc), tril_mask, memory_config=_batch_mc)
    ttnn.deallocate(L_diff_clamped)

    del k
    k_c = ttnn.move(k_c)
    k_c_t = ttnn.transpose(k_c, 1, 2, memory_config=_batch_mc)
    prog_config_kk = _get_matmul_program_config(chunk_size, K, chunk_size, grid_size=None)
    if prog_config_kk:
        kk = ttnn.matmul(
            k_beta_c, k_c_t, program_config=prog_config_kk, memory_config=_batch_mc, compute_kernel_config=_hifi_cfg
        )
    else:
        kk = ttnn.matmul(k_beta_c, k_c_t, memory_config=_batch_mc, compute_kernel_config=_hifi_cfg)
    ttnn.deallocate(k_c_t)

    # Compute -(kk * L_mask) = lower-triangular correction matrix including diagonal.
    attn_raw = ttnn.neg(ttnn.multiply(kk, L_mask, memory_config=_batch_mc), memory_config=_batch_mc)
    ttnn.deallocate(kk)

    # Forward substitution: compute R = (I - A)^{-1} where A is lower triangular.
    # Uses LAPACK batched triangular solve instead of a Python loop — same math,
    # but ~10-50x faster on CPU for large chunk_size (e.g. 256 iterations → 1 BLAS call).

    A = ttnn.to_torch(attn_raw).float()  # [batch, chunk_size, chunk_size]
    ttnn.deallocate(attn_raw)
    eye = _chunk_eye(chunk_size)
    I_minus_A = eye - A
    del A
    attn_cpu = torch.linalg.solve_triangular(I_minus_A, eye.expand_as(I_minus_A), upper=False)
    del I_minus_A
    attn = ttnn.from_torch(
        attn_cpu, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_batch_mc
    )
    del attn_cpu

    prog_config_vcorr = _get_matmul_program_config(chunk_size, chunk_size, V, grid_size=None)
    if prog_config_vcorr:
        v_corrected = ttnn.matmul(
            attn, v_beta_c, program_config=prog_config_vcorr, memory_config=_batch_mc, compute_kernel_config=_hifi_cfg
        )
    else:
        v_corrected = ttnn.matmul(attn, v_beta_c, memory_config=_batch_mc, compute_kernel_config=_hifi_cfg)
    del v_beta_c

    k_beta_decay = ttnn.multiply(k_beta_c, decay_exp, memory_config=_batch_mc)
    if prog_config_vcorr:
        k_cumdecay = ttnn.matmul(
            attn,
            k_beta_decay,
            program_config=prog_config_vcorr,
            memory_config=_batch_mc,
            compute_kernel_config=_hifi_cfg,
        )
    else:
        k_cumdecay = ttnn.matmul(attn, k_beta_decay, memory_config=_batch_mc, compute_kernel_config=_hifi_cfg)

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
    decay_raw_3d = ttnn.reshape(decay_raw, [BH, num_chunks, chunk_size], memory_config=None)

    decay_last_raw = ttnn.reshape(
        ttnn.sum(g_c, dim=-1, memory_config=None),
        [BH, num_chunks, 1],
        memory_config=None,
    )
    # decay_last in normalized coordinates = last value of normalized decay per chunk
    # = sum(g) - offset = decay_last_raw - g[0] per chunk
    # We compute this from the normalized decay's last element
    decay_last_normalized = decay_3d[:, :, -1:]  # [BH, num_chunks, 1] - last element per chunk
    decay_last_normalized = ttnn.reshape(decay_last_normalized, [BH, num_chunks, 1], memory_config=None)

    if cached_masks is not None:
        lower_causal = cached_masks["lower_causal"]
    else:
        lower_causal = _create_tril_ones_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=None)

    S = ttnn.zeros([BH, K, V], device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=_loop_mc)
    if initial_state is not None:
        S = ttnn.typecast(
            ttnn.reshape(initial_state, [BH, K, V], memory_config=None),
            ttnn.float32,
            memory_config=_loop_mc,
        )

    prog_config_vprime = _get_matmul_program_config(chunk_size, K, V, grid_size=None)
    prog_config_o_inter = _get_matmul_program_config(chunk_size, K, V, grid_size=None)
    prog_config_intra = _get_matmul_program_config(chunk_size, chunk_size, V, grid_size=None)
    prog_config_state = _get_matmul_program_config(K, chunk_size, V, grid_size=None)

    # Pre-compute state-independent ops as batched 4D tensors (hoisted from the loop).
    # This converts num_chunks × 3D ops into single batched 4D ops, improving hardware
    # utilization and reducing Python dispatch overhead in the sequential loop.

    # 1. Batched qk: [BH, num_chunks, cs, K] @ [BH, num_chunks, K, cs] → [BH, num_chunks, cs, cs]
    k_c_4d_t = ttnn.transpose(k_c_4d, 2, 3, memory_config=_batch_mc)
    qk_4d = ttnn.matmul(q_c_4d, k_c_4d_t, memory_config=_batch_mc, compute_kernel_config=_hifi_cfg)
    ttnn.deallocate(k_c_4d_t)

    # 2. Batched intra_attn: qk * L_mask * lower_causal
    lower_causal_4d = ttnn.reshape(lower_causal, [1, 1, chunk_size, chunk_size], memory_config=None)
    combined_mask_4d = ttnn.multiply(L_mask_4d, lower_causal_4d, memory_config=_batch_mc)
    intra_attn_4d = ttnn.multiply(qk_4d, combined_mask_4d, memory_config=_batch_mc)
    ttnn.deallocate(qk_4d)
    ttnn.deallocate(combined_mask_4d)

    # 3. Batched q_decay: q * exp(clip(decay_raw)) for inter-chunk state query
    # decay_raw_3d: [BH, num_chunks, cs] → [BH, num_chunks, cs, 1] for broadcast
    decay_raw_exp_4d = ttnn.reshape(
        ttnn.exp(ttnn.clip(decay_raw_3d, min=-20.0, max=0.0), memory_config=_batch_mc),
        [BH, num_chunks, chunk_size, 1],
        memory_config=_batch_mc,
    )
    q_decay_4d = ttnn.multiply(q_c_4d, decay_raw_exp_4d, memory_config=_batch_mc)
    ttnn.deallocate(decay_raw_exp_4d)

    # 4. Batched k_decay_t: k * exp(clip(decay_last - decay)) then transpose
    # decay_diff = decay_last_normalized - decay (per-chunk normalized coordinates)
    decay_last_norm_4d = ttnn.reshape(decay_last_normalized, [BH, num_chunks, 1], memory_config=_batch_mc)
    decay_diff_3d = ttnn.subtract(decay_last_norm_4d, decay_3d, memory_config=_batch_mc)
    decay_diff_exp_4d = ttnn.reshape(
        ttnn.exp(ttnn.clip(decay_diff_3d, min=-20.0, max=0.0), memory_config=_batch_mc),
        [BH, num_chunks, chunk_size, 1],
        memory_config=_batch_mc,
    )
    k_decay_4d = ttnn.multiply(k_c_4d, decay_diff_exp_4d, memory_config=_batch_mc)
    ttnn.deallocate(decay_diff_exp_4d)
    k_decay_t_4d = ttnn.transpose(k_decay_4d, 2, 3, memory_config=_batch_mc)
    ttnn.deallocate(k_decay_4d)

    # 5. Batched state decay factors: exp(clip(decay_last_raw))
    # decay_last_raw: [BH, num_chunks, 1] → dl_exp_3d: [BH, num_chunks, 1]
    dl_exp_3d = ttnn.exp(ttnn.clip(decay_last_raw, min=-20.0, max=0.0), memory_config=_batch_mc)

    outputs = []
    for i in range(num_chunks):
        v_i = v_cor_4d[:, i]
        k_cum_i = k_cum_4d[:, i]
        intra_attn_i = intra_attn_4d[:, i]
        q_decay_i = q_decay_4d[:, i]
        k_decay_t_i = k_decay_t_4d[:, i]

        # v_prime = k_cumdecay @ S (state-dependent)
        if prog_config_vprime:
            v_prime = ttnn.matmul(
                k_cum_i, S, program_config=prog_config_vprime, memory_config=_loop_mc, compute_kernel_config=_hifi_cfg
            )
        else:
            v_prime = ttnn.matmul(k_cum_i, S, memory_config=_loop_mc, compute_kernel_config=_hifi_cfg)
        v_new = ttnn.subtract(v_i, v_prime, memory_config=_loop_mc)

        # o_inter = q_decay @ S (state-dependent)
        if prog_config_o_inter:
            o_inter = ttnn.matmul(
                q_decay_i,
                S,
                program_config=prog_config_o_inter,
                memory_config=_loop_mc,
                compute_kernel_config=_hifi_cfg,
            )
        else:
            o_inter = ttnn.matmul(q_decay_i, S, memory_config=_loop_mc, compute_kernel_config=_hifi_cfg)

        # intra_v = intra_attn @ v_new (depends on v_new which depends on S)
        if prog_config_intra:
            intra_v = ttnn.matmul(
                intra_attn_i,
                v_new,
                program_config=prog_config_intra,
                memory_config=_loop_mc,
                compute_kernel_config=_hifi_cfg,
            )
        else:
            intra_v = ttnn.matmul(intra_attn_i, v_new, memory_config=_loop_mc, compute_kernel_config=_hifi_cfg)

        o_i = ttnn.add(o_inter, intra_v, memory_config=_loop_mc)
        outputs.append(ttnn.reshape(o_i, [BH, 1, chunk_size, V], memory_config=_loop_mc))

        # State update: S = S * decay_factor + k_decay_t @ v_new
        dl_i_exp = dl_exp_3d[:, i]
        S = ttnn.multiply(
            S,
            ttnn.reshape(dl_i_exp, [BH, 1, 1], memory_config=_loop_mc),
            memory_config=_loop_mc,
        )
        if prog_config_state:
            state_update = ttnn.matmul(
                k_decay_t_i,
                v_new,
                program_config=prog_config_state,
                memory_config=_loop_mc,
                compute_kernel_config=_hifi_cfg,
            )
        else:
            state_update = ttnn.matmul(k_decay_t_i, v_new, memory_config=_loop_mc, compute_kernel_config=_hifi_cfg)
        S = ttnn.add(S, state_update, memory_config=_loop_mc)

    o = ttnn.concat(outputs, dim=1, memory_config=None)
    # o shape: [BH, num_chunks, chunk_size, V]
    # First merge chunk dims to get [BH, L, V], then trim padding if needed
    o = ttnn.reshape(o, [BH, L, V], memory_config=None)

    if pad_len > 0:
        o = o[:, :T, :]
        o = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=None)

    o = ttnn.reshape(o, [B, H, T, V], memory_config=None)
    o = ttnn.transpose(o, 1, 2, memory_config=None)
    o = ttnn.typecast(o, ttnn.bfloat16, memory_config=None)

    final_state = ttnn.reshape(S, [B, H, K, V], memory_config=None)
    return o, final_state


def rms_norm_gated_ttnn(x, gate, weight, eps=1e-6, memory_config=None):
    """RMSNorm with SiLU gating for GDN output."""
    mc = memory_config
    x_normed = ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=mc)
    gate_act = ttnn.silu(gate, memory_config=mc)
    gate_act = ttnn.clip(gate_act, min=-1e4, max=1e4)
    return ttnn.multiply(x_normed, gate_act, memory_config=mc)
