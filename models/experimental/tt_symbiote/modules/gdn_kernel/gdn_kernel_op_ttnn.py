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


def gdn_recurrence_ttnn(q, k_row, k_col, v, g, beta, state):
    """
    GDN recurrence step using standard ttnn ops.

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
    # Step 1: decay
    g_exp = ttnn.exp(g)
    state_b = ttnn.multiply(state, g_exp)
    ttnn.deallocate(g_exp)

    # Step 2: kv_mem = k_row @ state
    kv_mem = ttnn.matmul(k_row, state_b)

    # Step 3: delta = beta * (v - kv_mem)
    diff = ttnn.subtract(v, kv_mem)
    ttnn.deallocate(kv_mem)
    delta = ttnn.multiply(beta, diff)
    ttnn.deallocate(diff)

    # Step 4: state += outer(k_col, delta)
    outer = ttnn.matmul(k_col, delta)
    ttnn.deallocate(delta)
    new_state = ttnn.add(state_b, outer)
    ttnn.deallocate(state_b)
    ttnn.deallocate(outer)

    # Step 5: output = q @ new_state
    output = ttnn.matmul(q, new_state)

    # Update state in-place (preserves tensor ID for tracing)
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
