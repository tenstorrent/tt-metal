# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
GDN (Gated DeltaNet) recurrence using ttnn ops (FALLBACK version).

This is the safe fallback that uses standard ttnn ops. It is ~12x slower than
the fused kernel but works on all platforms including Blackhole P150.

Implements the recurrence:
  1. state *= exp(g)                    -- decay
  2. kv_mem = k_row @ state             -- [1,K] x [K,V] -> [1,V]
  3. delta = beta * (v - kv_mem)        -- element-wise
  4. state += outer(k_col, delta)       -- [K,1] x [1,V] -> [K,V]
  5. output = q @ state                 -- [1,K] x [K,V] -> [1,V]

All tensors are [num_pairs, ...] where num_pairs = batch * num_heads.
"""


import ttnn

_L1 = ttnn.L1_MEMORY_CONFIG


def _hs_l1_config(num_pairs, shard_h, shard_w):
    """HEIGHT_SHARDED L1 config: exactly num_pairs shards of [shard_h × shard_w].

    Finds an (ncols, nrows) factorization of num_pairs that fits in the
    hardware grid (≤8 cols, ≤10 rows).  Falls back to a two-range CoreRangeSet
    if no exact rectangle exists.
    """
    # Try to find ncols such that num_pairs % ncols == 0 and nrows <= 10
    for nc in range(min(num_pairs, 8), 0, -1):
        if num_pairs % nc == 0:
            nr = num_pairs // nc
            if nr <= 10:
                core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nc - 1, nr - 1))])
                return ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(core_grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
                )
    # Fallback: fill rows with max_cols cores, remainder in the next row
    nc = 8
    full_rows = num_pairs // nc
    remainder = num_pairs % nc
    ranges = []
    if full_rows > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nc - 1, full_rows - 1)))
    if remainder > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(remainder - 1, full_rows)))
    core_grid = ttnn.CoreRangeSet(ranges)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _to_4d(t, num_pairs, d1, d2):
    """[num_pairs, d1, d2] → ROW_MAJOR → [1, num_pairs, d1, d2] → TILE L1.

    Converts to TILE before freeing RM sources so reshape views are consumed first.
    """
    t_rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    t4_rm = ttnn.reshape(t_rm, (1, num_pairs, d1, d2))
    result = ttnn.to_layout(t4_rm, ttnn.TILE_LAYOUT, memory_config=_L1)
    ttnn.deallocate(t4_rm)
    ttnn.deallocate(t_rm)
    return result


def _from_4d(t4, num_pairs, d1, d2, mem=None):
    """[1, num_pairs, d1, d2] → ROW_MAJOR → [num_pairs, d1, d2] → TILE.

    mem: memory config for the output tensor; defaults to L1_MEMORY_CONFIG.
    Converts to TILE before freeing RM sources so reshape views are consumed first.
    """
    mc = mem if mem is not None else _L1
    t4_rm = ttnn.to_layout(t4, ttnn.ROW_MAJOR_LAYOUT)
    t3_rm = ttnn.reshape(t4_rm, (num_pairs, d1, d2))
    result = ttnn.to_layout(t3_rm, ttnn.TILE_LAYOUT, memory_config=mc)
    ttnn.deallocate(t3_rm)
    ttnn.deallocate(t4_rm)
    return result


def gdn_recurrence_ttnn(q, k_row, k_col, v, g, beta, state):
    """
    GDN recurrence step using standard ttnn ops with L1 sharding.

    Tensors are reshaped from [num_pairs, 1/Dk, D] → [1, num_pairs, 1/Dk, D]
    before matmuls so TTNN sees [1, 12, 128] as M-dimension, enabling
    12-core parallelism via HEIGHT_SHARDED L1.

    Args:
        q: [num_pairs, 1, Dk] query (already L2-normed and scaled)
        k_row: [num_pairs, 1, Dk] key row vector (already L2-normed)
        k_col: [num_pairs, Dk, 1] key column vector (k transposed)
        v: [num_pairs, 1, Dv] value
        g: [num_pairs, 1, 1] log-space decay (negative values)
        beta: [num_pairs, 1, 1] beta scalar
        state: [num_pairs, Dk, Dv] recurrence state (modified in-place)

    Returns:
        output: [num_pairs, 1, Dv]
        (state is updated in-place via ttnn.copy)
    """
    num_pairs = q.shape[0]
    Dk = q.shape[-1]
    Dv = v.shape[-1]

    # Step 1: decay — keep in current layout, output to L1
    g_exp = ttnn.exp(g, memory_config=_L1)
    state_b = ttnn.multiply(state, g_exp, memory_config=_L1)
    ttnn.deallocate(g_exp)

    # Reshape all operands to 4D: [1, num_pairs, d1, d2]
    # This lets TTNN treat the num_pairs dimension as M=num_pairs (heads as sequence),
    # matching the user's [1, 12, 128] target shape and enabling 12-core parallelism.
    q4 = _to_4d(q, num_pairs, 1, Dk)
    k_row4 = _to_4d(k_row, num_pairs, 1, Dk)
    k_col4 = _to_4d(k_col, num_pairs, Dk, 1)
    v4 = _to_4d(v, num_pairs, 1, Dv)
    beta4 = _to_4d(beta, num_pairs, 1, 1)
    state_b4 = _to_4d(state_b, num_pairs, Dk, Dv)
    ttnn.deallocate(state_b)

    # Step 2: kv_mem = k_row4 @ state_b4  →  [1, num_pairs, 1, Dv]
    kv_mem4 = ttnn.matmul(k_row4, state_b4, memory_config=_L1)
    ttnn.deallocate(k_row4)

    # Step 3: delta = beta * (v - kv_mem)
    diff4 = ttnn.subtract(v4, kv_mem4, memory_config=_L1)
    ttnn.deallocate(kv_mem4)
    ttnn.deallocate(v4)
    delta4 = ttnn.multiply(beta4, diff4, memory_config=_L1)
    ttnn.deallocate(beta4)
    ttnn.deallocate(diff4)

    # Step 4: new_state4 = state_b4 + outer(k_col4, delta4)
    outer4 = ttnn.matmul(k_col4, delta4, memory_config=_L1)
    ttnn.deallocate(k_col4)
    ttnn.deallocate(delta4)
    new_state4 = ttnn.add(state_b4, outer4, memory_config=_L1)
    ttnn.deallocate(state_b4)
    ttnn.deallocate(outer4)

    # Step 5: output4 = q4 @ new_state4  →  [1, num_pairs, 1, Dv]
    output4 = ttnn.matmul(q4, new_state4, memory_config=_L1)
    ttnn.deallocate(q4)

    # Reshape output back to [num_pairs, 1, Dv] — use DRAM so accumulated token_outs
    # don't fill L1 when this function is called N times inside gdn_prefill_ttnn.
    output = _from_4d(output4, num_pairs, 1, Dv, mem=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(output4)

    # Reshape new_state4 back to [num_pairs, Dk, Dv] and update state in-place
    # (state_l1 is short-lived here — immediately overwritten — so L1 is fine)
    new_state = _from_4d(new_state4, num_pairs, Dk, Dv)
    ttnn.deallocate(new_state4)
    ttnn.copy(new_state, state)
    ttnn.deallocate(new_state)

    return output


def gdn_recurrence_fused_inplace(q, k_row, k_col, v, g, beta, state, output, num_cores=10):
    """Drop-in replacement: computes recurrence and writes result to output tensor.

    The num_cores parameter is accepted for API compatibility but ignored
    (ttnn ops handle parallelism automatically).
    """
    result = gdn_recurrence_ttnn(q, k_row, k_col, v, g, beta, state)
    ttnn.copy(result, output)
    ttnn.deallocate(result)


def _l2_norm_ttnn(x):
    """L2-normalize x along last dim using ttnn ops."""
    x_sq = ttnn.multiply(x, x, memory_config=_L1)
    ssq = ttnn.sum(x_sq, dim=-1, keepdim=True, memory_config=_L1)
    ttnn.deallocate(x_sq)
    inv = ttnn.rsqrt(ttnn.add(ssq, 1e-6, memory_config=_L1), memory_config=_L1)
    ttnn.deallocate(ssq)
    normed = ttnn.multiply(x, inv, memory_config=_L1)
    ttnn.deallocate(inv)
    return normed


def gdn_prefill_ttnn(
    conv_out,
    a_fused,
    b_fused,
    neg_exp_A,
    dt_bias,
    norm_w,
    scale_tt,
    rms_scale_tt,
    rms_eps_tt,
    state,
    output,
    num_pairs,
    num_tokens,
    Nv_TP,
    Nk_TP,
    repeat_factor,
    key_dim_tp,
):
    """GDN prefill using pure ttnn ops — drop-in for gdn_prefill_fused.

    Processes N tokens sequentially with the same math as the fused kernel.
    Per token:
      1. Slice Q/K/V from conv_out
      2. L2-norm Q, K
      3. Scale Q; repeat-interleave Q/K from Nk_TP to Nv_TP heads
      4. Compute gates: beta=sigmoid(b), g=neg_exp_A*softplus(a+dt_bias)
      5. DeltaNet recurrence (state updated in-place)
    After loop: rearrange per-token outputs into output buffer.

    Args match gdn_prefill_fused exactly:
        conv_out  : [1, N, qkv_dim_tp]
        a_fused   : [1, N, Nv_TP]
        b_fused   : [1, N, Nv_TP]
        neg_exp_A : [1, 1, Nv_TP]
        dt_bias   : [1, 1, Nv_TP]
        state     : [num_pairs, Dk, Dv]  — updated in-place
        output    : [num_pairs * N, 1, Dv]  — written on return
    """
    N = num_tokens
    qkv_dim_tp = conv_out.shape[-1]
    value_dim_tp = qkv_dim_tp - 2 * key_dim_tp
    Dk = key_dim_tp // Nk_TP
    Dv = value_dim_tp // Nv_TP

    # Move state to L1 INTERLEAVED for the loop duration: eliminates DRAM round-trips
    # on every token's matmul without requiring HEIGHT_SHARDED broadcast compatibility.
    state_l1 = ttnn.to_memory_config(state, _L1)

    token_outs = []

    for t in range(N):
        # ---- Extract token t ----
        conv_t = ttnn.slice(conv_out, (0, t, 0), (1, t + 1, qkv_dim_tp))  # [1,1,qkv]

        q_raw = ttnn.slice(conv_t, (0, 0, 0), (1, 1, key_dim_tp))
        k_raw = ttnn.slice(conv_t, (0, 0, key_dim_tp), (1, 1, 2 * key_dim_tp))
        v_raw = ttnn.slice(conv_t, (0, 0, 2 * key_dim_tp), (1, 1, qkv_dim_tp))
        ttnn.deallocate(conv_t)

        # ---- L2 norm Q and K per head ----
        q_h = ttnn.reshape(q_raw, (1, Nk_TP, Dk))
        ttnn.deallocate(q_raw)
        q_n = _l2_norm_ttnn(q_h)
        ttnn.deallocate(q_h)

        k_h = ttnn.reshape(k_raw, (1, Nk_TP, Dk))
        ttnn.deallocate(k_raw)
        k_n = _l2_norm_ttnn(k_h)
        ttnn.deallocate(k_h)

        # ---- Scale Q ----
        q_scaled = ttnn.multiply(q_n, scale_tt, memory_config=_L1)
        ttnn.deallocate(q_n)

        # ---- Expand Nk_TP heads → Nv_TP via repeat_interleave ----
        q_exp = ttnn.repeat_interleave(q_scaled, repeat_factor, dim=1)
        ttnn.deallocate(q_scaled)
        q_exp_rm = ttnn.to_layout(q_exp, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(q_exp)
        q_pairs_rm = ttnn.reshape(q_exp_rm, (num_pairs, 1, Dk))
        ttnn.deallocate(q_exp_rm)
        q_pairs = ttnn.to_layout(q_pairs_rm, ttnn.TILE_LAYOUT, memory_config=_L1)
        ttnn.deallocate(q_pairs_rm)

        k_exp = ttnn.repeat_interleave(k_n, repeat_factor, dim=1)
        ttnn.deallocate(k_n)
        k_exp_rm = ttnn.to_layout(k_exp, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(k_exp)
        k_row_rm = ttnn.reshape(k_exp_rm, (num_pairs, 1, Dk))
        ttnn.deallocate(k_exp_rm)
        k_row = ttnn.to_layout(k_row_rm, ttnn.TILE_LAYOUT, memory_config=_L1)
        ttnn.deallocate(k_row_rm)
        k_col = ttnn.transpose(k_row, -2, -1, memory_config=_L1)

        v_rm = ttnn.to_layout(v_raw, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(v_raw)
        v_pairs_rm = ttnn.reshape(v_rm, (num_pairs, 1, Dv))
        ttnn.deallocate(v_rm)
        v_pairs = ttnn.to_layout(v_pairs_rm, ttnn.TILE_LAYOUT, memory_config=_L1)
        ttnn.deallocate(v_pairs_rm)

        # ---- Gates ----
        a_t = ttnn.slice(a_fused, (0, t, 0), (1, t + 1, Nv_TP))
        b_t = ttnn.slice(b_fused, (0, t, 0), (1, t + 1, Nv_TP))

        beta_flat = ttnn.sigmoid(b_t, memory_config=_L1)
        ttnn.deallocate(b_t)
        beta_rm = ttnn.to_layout(beta_flat, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(beta_flat)
        beta_pairs_rm = ttnn.reshape(beta_rm, (num_pairs, 1, 1))
        ttnn.deallocate(beta_rm)
        beta_pairs = ttnn.to_layout(beta_pairs_rm, ttnn.TILE_LAYOUT, memory_config=_L1)
        ttnn.deallocate(beta_pairs_rm)

        a_dt = ttnn.add(a_t, dt_bias, memory_config=_L1)
        ttnn.deallocate(a_t)
        sp = ttnn.softplus(a_dt, memory_config=_L1)
        ttnn.deallocate(a_dt)
        g_flat = ttnn.multiply(neg_exp_A, sp, memory_config=_L1)
        ttnn.deallocate(sp)
        g_rm = ttnn.to_layout(g_flat, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(g_flat)
        g_pairs_rm = ttnn.reshape(g_rm, (num_pairs, 1, 1))
        ttnn.deallocate(g_rm)
        g_pairs = ttnn.to_layout(g_pairs_rm, ttnn.TILE_LAYOUT, memory_config=_L1)
        ttnn.deallocate(g_pairs_rm)

        # ---- DeltaNet recurrence (state_l1 updated in-place) ----
        out_t = gdn_recurrence_ttnn(q_pairs, k_row, k_col, v_pairs, g_pairs, beta_pairs, state_l1)

        token_outs.append(out_t)

        ttnn.deallocate(q_pairs)
        ttnn.deallocate(k_row)
        ttnn.deallocate(k_col)
        ttnn.deallocate(v_pairs)
        ttnn.deallocate(g_pairs)
        ttnn.deallocate(beta_pairs)

    # Write final L1 state back to the caller's DRAM state tensor
    ttnn.copy(state_l1, state)
    ttnn.deallocate(state_l1)

    # ---- Rearrange token_outs to output layout [num_pairs * N, 1, Dv] ----
    stacked = ttnn.concat(token_outs, dim=0)
    for o in token_outs:
        ttnn.deallocate(o)

    stacked_rm = ttnn.to_layout(stacked, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(stacked)

    tok_outer = ttnn.reshape(stacked_rm, (N, num_pairs, Dv))
    pair_outer = ttnn.permute(tok_outer, (1, 0, 2))
    ttnn.deallocate(tok_outer)
    ttnn.deallocate(stacked_rm)

    flat_rm = ttnn.reshape(pair_outer, (num_pairs * N, 1, Dv))
    flat = ttnn.to_layout(flat_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(flat_rm)
    ttnn.deallocate(pair_outer)

    ttnn.copy(flat, output)
    ttnn.deallocate(flat)
