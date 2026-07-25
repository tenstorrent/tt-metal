# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN gated delta rule: recurrent (decode) and chunked (prefill) paths.

FLA layout: q,k [B,T,H,K]; v [B,T,H,V]; beta,g [B,T,H]; state [B,H,K,V].
"""

import math

import torch
import ttnn

# Wormhole tile size
_TILE_H = 32
_TILE_W = 32

# Cached identity matrices for forward substitution
_EYE_CACHE = {}


def _chunk_eye(size):
    """Return a cached [size, size] float32 identity matrix on CPU."""
    if size not in _EYE_CACHE:
        _EYE_CACHE[size] = torch.eye(size, dtype=torch.float32)
    return _EYE_CACHE[size]


def _recurrent_read_query_program_config(device, K, V):
    """Progcfg for read/query matmul: [B*H,1,K] @ [B*H,K,V] -> [B*H,1,V]. N == per_core_N."""
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
    """Create [size,size] identity on device via triu/tril."""
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
    """Create [size,size] upper-triangular ones on device."""
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
    """Create [size,size] lower-triangular ones on device."""
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
    """Create [size,size] strict lower-tri ones (diagonal=-1) on device."""
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
    """Matmul progcfg for (m,k,n). Returns None when auto-config is better."""
    TILE_SIZE = 32
    # Cap per_core_M to avoid L1 CB overflow on BH (~40KB safe at 2 M-tiles/core).
    MAX_PER_CORE_M = 2

    if m < 32 or n < 32 or k < 32:
        return None

    m_tiles = math.ceil(m / TILE_SIZE)
    n_tiles = math.ceil(n / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)

    # Large k (k_tiles>8) overflows L1 CB; let TTNN auto-select.
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

    # Increase cores_y if per_core_M too large; force in0_block_w=1 for L1 safety.
    if per_core_M > MAX_PER_CORE_M:
        cores_y = math.ceil(m_tiles / MAX_PER_CORE_M)
        cores_y = min(cores_y, 8)  # max 8 cores along Y
        per_core_M = math.ceil(m_tiles / cores_y)
        if per_core_M > MAX_PER_CORE_M:
            return None  # Can't fit with 8 cores; auto-select
        grid_size = (cores_x, cores_y)
        in0_block_w = 1  # keep in0 buffer from scaling with k_tiles

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
    """L2 norm along dim. Last dim: fused rms_norm path (~3 fewer kernels)."""
    # L1 for T<=512; DRAM otherwise
    T = x.shape[1] if len(x.shape) >= 3 else x.shape[0]
    mc = ttnn.L1_MEMORY_CONFIG if T <= 512 else ttnn.DRAM_MEMORY_CONFIG
    if dim in (-1, len(x.shape) - 1):
        K = x.shape[-1]
        normed = ttnn.rms_norm(x, epsilon=eps / K)
        return ttnn.multiply(normed, K**-0.5, memory_config=mc)
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
    """State update: h = decay*h + beta*(k⊗delta). apply_decay=False if caller already decayed h."""
    B = h.shape[0]
    H = h.shape[1]
    K = h.shape[2]
    V = h.shape[3]

    # decay_t: [B,H] -> [B,H,1,1] (already exp(g) in BF16)
    decay = ttnn.reshape(decay_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

    # beta: [B,H] -> [B,H,1,1]
    beta_expanded = ttnn.reshape(beta_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

    # Decode opt: keep state-write operands in L1 (tiny at B=1).
    _L1 = ttnn.L1_MEMORY_CONFIG
    k_col = ttnn.reshape(k_t, [B, H, K, 1], memory_config=_L1)
    d_row = ttnn.reshape(delta, [B, H, 1, V], memory_config=_L1)

    k_col = ttnn.to_memory_config(k_col, _L1)
    d_row = ttnn.to_memory_config(d_row, _L1)

    matmul_compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    outer = ttnn.matmul(
        k_col,
        d_row,
        memory_config=_L1,
        compute_kernel_config=matmul_compute_cfg,
        program_config=None,
    )

    # apply beta
    outer = ttnn.multiply(
        outer,
        beta_expanded,
        memory_config=_L1,
    )

    # apply_decay=False: h already decayed (decay->read->write order).
    if apply_decay:
        h = ttnn.multiply(h, decay, memory_config=_L1)
    h = ttnn.add(h, outer, memory_config=_L1)

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
    """Single recurrent delta-rule step via fused_decay_and_write_ttnn."""
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

    # Decay state before read (matches FLA + chunked prefill).
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

    # Outer product only (h already decayed).
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
    """T=1 decode path: no loop/slice overhead. Returns o [B,1,H,V], state [B,H,K,V]."""
    B = q.shape[0]
    H = q.shape[2]
    K = q.shape[3]
    V = v.shape[3]

    # high_precision: fp32 step avoids bf16 decay quantization error over long decode.
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

    # Reshape to matmul shapes; caller already TILE_LAYOUT.
    # Decode opt: q_row/k_row in L1 with L1-resident state h.
    q_row = ttnn.reshape(q, [B, H, 1, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    k_row = ttnn.reshape(k, [B, H, 1, K], memory_config=ttnn.L1_MEMORY_CONFIG)
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
        h = ttnn.typecast(h, ttnn.float32)

    # Decode opt: keep [B,H,K,V] state in L1 (~0.8MB fp32 at B=1).
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

    # Decay before read; keep recurrence step L1-resident.
    _L1 = ttnn.L1_MEMORY_CONFIG
    decay_bhkv = ttnn.reshape(decay_t, [B, H, 1, 1], memory_config=_L1)
    h = ttnn.multiply(h, decay_bhkv, memory_config=_L1)

    # v_read = k @ h (decayed state)
    v_read = ttnn.matmul(
        k_row, h, memory_config=_L1, program_config=read_query_prog_cfg, compute_kernel_config=read_query_compute_cfg
    )
    v_read = ttnn.reshape(v_read, [B, H, V], memory_config=_L1)

    # Delta + state write (no re-decay).
    delta = ttnn.subtract(v_t, v_read, memory_config=_L1)
    k_t = ttnn.reshape(k_row, [B, H, K], memory_config=_L1)
    h = fused_decay_and_write_ttnn(
        h=h, k_t=k_t, delta=delta, decay_t=decay_t, beta_t=beta_t, device=device, apply_decay=False
    )

    # o = q @ h
    o_t = ttnn.matmul(
        q_row, h, memory_config=_L1, program_config=read_query_prog_cfg, compute_kernel_config=read_query_compute_cfg
    )

    # Reshape output to [B, 1, H, V]
    o = ttnn.reshape(o_t, [B, 1, H, V], memory_config=_L1)

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
    """Decode with in-place state copy to pre-allocated buffer (trace-safe addresses)."""
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
    # Copy state back to pre-allocated buffer for trace replay.
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
    """Token-by-token recurrent gated delta rule (decode, general T).

    Per step: decay h; v_read=k·h; delta=v-v_read; write outer; o=q·h.
    Returns output [B,T,H,V], final_state [B,H,K,V].
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

    # Precompute exp(g); slice per timestep in loop.
    g_exp = ttnn.exp(g)

    if initial_state is not None:
        h = ttnn.typecast(initial_state, ttnn.bfloat16, memory_config=None)
    else:
        h = ttnn.zeros([B, H, K, V], device=device, dtype=ttnn.bfloat16, memory_config=None)

    outputs = []
    for i in range(T):
        q_t = q[:, :, i]  # [B,H,K]
        k_t = k[:, :, i]
        v_t = v[:, :, i]
        beta_t = beta[:, :, i]
        decay_t = g_exp[:, :, i]

        o_t, h = recurrent_delta_rule_step_ttnn(q_t, k_t, v_t, beta_t, decay_t, h, seq_len=T, device=device)
        outputs.append(o_t)

    outputs_4d = [ttnn.reshape(o, [B, H, 1, V], memory_config=None) for o in outputs]
    o = ttnn.concat(outputs_4d, dim=2, memory_config=None)
    o = ttnn.transpose(o, 1, 2, memory_config=None)
    o = ttnn.typecast(o, ttnn.bfloat16, memory_config=None)

    return o, h
