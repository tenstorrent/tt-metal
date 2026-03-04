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


def _get_core_grid(device):
    """Get CoreGrid from device (y, x order for ttnn)."""
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=grid.x)


def create_state_shard_config(h_shape, device):
    """
    Create L1 width-sharded memory config for state tensor h [B, H, K, V].
    Shards the V dimension across cores. Uses ttnn 2D view: (B*H*K, V).
    """
    B, H, K, V = h_shape
    core_grid = _get_core_grid(device)
    total_cores = core_grid.x * core_grid.y
    if V % total_cores != 0:
        return None
    # shape as 2D for create_sharded_memory_config: (height, width)
    height, width = B * H * K, V
    return ttnn.create_sharded_memory_config(
        (height, width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def create_output_shard_config(B, H, V, device):
    """
    Create L1 width-sharded memory config for matmul outputs [B, H, 1, V].
    Same sharding as state but for height B*H.
    """
    core_grid = _get_core_grid(device)
    total_cores = core_grid.x * core_grid.y
    if V % total_cores != 0:
        return None
    height, width = B * H, V
    return ttnn.create_sharded_memory_config(
        (height, width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def create_matmul_1d_config(K, V, device):
    """
    Create 1D multicast matmul program config for (1, K) @ (K, V) per batch,
    with mcast_in0=True (in0 height-sharded, in1 width-sharded).
    """
    grid = device.compute_with_storage_grid_size()
    core_grid = _get_core_grid(device)
    total_cores = core_grid.x * core_grid.y
    if total_cores == 0:
        return None
    in0_block_w = max(1, K // 32 // total_cores)
    per_core_N = max(1, V // 32 // total_cores)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=min(per_core_N, 8),
        out_block_h=1,
        out_block_w=per_core_N,
        per_core_M=1,
        per_core_N=per_core_N,
        fuse_batch=True,
        mcast_in0=True,
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
    state_shard_config=None,
    output_shard_config=None,
    matmul_1d_config=None,
):
    """
    Single recurrent step of the gated delta rule using TTNN ops.

    Uses ttnn.matmul for K-dimension reductions to leverage hardware
    float32 accumulation, improving numerical precision.
    When state_shard_config, output_shard_config, and matmul_1d_config are
    provided, matmuls that read from h use L1 width-sharded 1D config (auto-sharded).

    Args:
        q_t: [B, H, K] query for this timestep
        k_t: [B, H, K] key for this timestep
        v_t: [B, H, V] value for this timestep
        beta_t: [B, H] write strength
        g_t: [B, H] log-space decay
        h: [B, H, K, V] recurrent state
        state_shard_config: optional L1 width-sharded config for h
        output_shard_config: optional L1 width-sharded config for v_read/o_t output
        matmul_1d_config: optional MatmulMultiCoreReuseMultiCast1DProgramConfig

    Returns:
        o_t: [B, H, V] output
        h: [B, H, K, V] updated state
    """
    B = q_t.shape[0]
    H = q_t.shape[1]
    K = q_t.shape[2]
    V = v_t.shape[2]

    use_sharded = state_shard_config is not None and output_shard_config is not None and matmul_1d_config is not None
    if use_sharded and not h.memory_config().is_sharded():
        h = ttnn.to_memory_config(h, state_shard_config)

    # 1. Decay the state: h = h * exp(g_t) — always L1 (sharded when configs provided)
    decay = ttnn.exp(g_t, memory_config=ttnn.L1_MEMORY_CONFIG)
    decay = ttnn.reshape(decay, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    h = ttnn.multiply(h, decay)

    # 2. Read from state via matmul: v_read = k^T @ h (L1, auto-sharded when configs provided)
    k_row = ttnn.reshape(k_t, [B, H, 1, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    if use_sharded:
        v_read = ttnn.matmul(
            k_row,
            h,
            program_config=matmul_1d_config,
            memory_config=output_shard_config,
        )
    else:
        v_read = ttnn.matmul(k_row, h, memory_config=ttnn.L1_MEMORY_CONFIG)
    v_read = ttnn.reshape(v_read, [B, H, V], memory_config=ttnn.L1_MEMORY_CONFIG)

    # 3. Compute delta: delta = (v_t - v_read) * beta_t
    delta = ttnn.subtract(v_t, v_read, memory_config=ttnn.L1_MEMORY_CONFIG)
    beta_expanded = ttnn.reshape(beta_t, [B, H, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    delta = ttnn.multiply(delta, beta_expanded, memory_config=ttnn.L1_MEMORY_CONFIG)

    # 4. Write to state via matmul outer product: h += k @ delta^T — always L1 (sharded when configs provided)
    k_col = ttnn.reshape(k_t, [B, H, K, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    d_row = ttnn.reshape(delta, [B, H, 1, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    outer = ttnn.matmul(
        k_col,
        d_row,
        memory_config=state_shard_config if use_sharded else ttnn.L1_MEMORY_CONFIG,
    )
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

    # L1 width-sharded configs for matmuls (when V divisible by core count)
    state_shard_config = create_state_shard_config([B, H, K, V], device)
    output_shard_config = create_output_shard_config(B, H, V, device)
    matmul_1d_config = create_matmul_1d_config(K, V, device)

    # Initialize state in float32; use L1 sharded when config available
    if initial_state is not None:
        h = ttnn.typecast(initial_state, ttnn.float32)
        if state_shard_config is not None:
            h = ttnn.to_memory_config(h, state_shard_config)
    else:
        if state_shard_config is not None:
            h = ttnn.zeros(
                [B, H, K, V],
                device=device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=state_shard_config,
            )
        else:
            h = ttnn.zeros([B, H, K, V], device=device, dtype=ttnn.float32)

    outputs = []
    for i in range(T):
        q_t = q[:, :, i]  # [B, H, K]
        k_t = k[:, :, i]  # [B, H, K]
        v_t = v[:, :, i]  # [B, H, V]
        beta_t = beta[:, :, i]  # [B, H]
        g_t = g[:, :, i]  # [B, H]

        o_t, h = recurrent_delta_rule_step_ttnn(
            q_t,
            k_t,
            v_t,
            beta_t,
            g_t,
            h,
            state_shard_config=state_shard_config,
            output_shard_config=output_shard_config,
            matmul_1d_config=matmul_1d_config,
        )
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
