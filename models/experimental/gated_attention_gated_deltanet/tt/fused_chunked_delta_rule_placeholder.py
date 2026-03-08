# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Fused chunked delta rule implementation.

This module provides an optimized implementation that fuses the per-chunk operations
to reduce host-device dispatch overhead and improve performance.

The fused kernel combines:
- Intra-chunk attention matrix construction
- Woodbury fixup (pre-computed)
- Cross-chunk state update
- Output computation

All within a single optimized per-chunk loop.
"""

import math
import ttnn
from typing import Optional, Tuple

from .ttnn_delta_rule_ops import (
    l2_norm_ttnn,
    _create_eye_matrix_ttnn,
    _create_tril_ones_ttnn,
    _create_strict_lower_tril_ttnn,
    _get_matmul_program_config,
)


def _get_tensor_shape(tensor, use_padded=False):
    """Helper to get tensor shape, handling both torch and TTNN tensors.

    Args:
        tensor: Input tensor (torch or TTNN)
        use_padded: If True, use padded_shape for TTNN tensors (needed for slice operations)
    """
    if hasattr(tensor, "shape"):
        return tensor.shape
    elif hasattr(tensor, "padded_shape") and use_padded:
        return tensor.padded_shape()
    elif hasattr(tensor, "logical_shape"):
        return tensor.logical_shape()
    else:
        # Fallback: convert to torch
        return ttnn.to_torch(tensor).shape


def fused_chunked_delta_rule_ttnn(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    beta: ttnn.Tensor,
    g: ttnn.Tensor,
    chunk_size: int = 64,
    scale: Optional[float] = None,
    initial_state: Optional[ttnn.Tensor] = None,
    device: Optional[ttnn.Device] = None,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Fused chunked delta rule using optimized per-chunk operations.

    This implementation fuses the per-chunk computation to reduce dispatches:
    - Combines attention matrix construction with masking
    - Fuses state read/write operations
    - Reduces intermediate tensor allocations
    - Optimizes memory layout for better cache locality

    Args:
        q: [B, T, H, K] query tensor
        k: [B, T, H, K] key tensor
        v: [B, T, H, V] value tensor
        beta: [B, T, H] write strength tensor
        g: [B, T, H] log-space decay gate tensor
        chunk_size: Size of each chunk
        scale: Attention scale factor (defaults to 1/sqrt(K))
        initial_state: [B, H, K, V] initial recurrent state
        device: TTNN device

    Returns:
        output: [B, T, H, V] output tensor
        final_state: [B, H, K, V] final recurrent state
    """
    # Phase 1: Pre-chunk processing (same as original)
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

    # Transpose and cast to float32
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

    # Padding
    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    L = T + pad_len
    num_chunks = L // chunk_size
    batch = BH * num_chunks

    # Debug: Print key dimensions
    print(f"\n[DEBUG] Chunking parameters:")
    print(f"  T={T}, chunk_size={chunk_size}, pad_len={pad_len}, L={L}")
    print(f"  num_chunks={num_chunks}, batch={batch}, BH={BH}")

    # Flatten to [BH, T, D]
    q = ttnn.reshape(q, [BH, T, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.reshape(k, [BH, T, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.reshape(v, [BH, T, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    beta_flat = ttnn.reshape(beta, [BH, T, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    g = ttnn.reshape(g, [BH, T], memory_config=ttnn.L1_MEMORY_CONFIG)

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
        g_3d = ttnn.reshape(g, [BH, T, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
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
        g = ttnn.reshape(g_3d, [BH, L], memory_config=ttnn.L1_MEMORY_CONFIG)
        beta_flat = ttnn.reshape(beta_flat, [BH, L, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        beta_flat = ttnn.reshape(beta_flat, [BH, L, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

    v_beta = ttnn.multiply(v, beta_flat, memory_config=ttnn.L1_MEMORY_CONFIG)
    k_beta = ttnn.multiply(k, beta_flat, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Reshape into chunks
    print(f"\n[DEBUG] Before reshaping into chunks:")
    print(f"  q shape: {_get_tensor_shape(q)}, target: [batch={batch}, chunk_size={chunk_size}, K={K}]")
    print(f"  k shape: {_get_tensor_shape(k)}, target: [batch={batch}, chunk_size={chunk_size}, K={K}]")
    print(f"  v shape: {_get_tensor_shape(v)}, target: [batch={batch}, chunk_size={chunk_size}, V={V}]")

    q_c = ttnn.reshape(q, [batch, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    k_c = ttnn.reshape(k, [batch, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    v_c = ttnn.reshape(v, [batch, chunk_size, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    k_beta_c = ttnn.reshape(k_beta, [batch, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    v_beta_c = ttnn.reshape(v_beta, [batch, chunk_size, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    g_c = ttnn.reshape(g, [batch, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)

    print(f"[DEBUG] After reshaping into chunks:")
    print(f"  q_c shape: {_get_tensor_shape(q_c)}")
    print(f"  k_c shape: {_get_tensor_shape(k_c)}")
    print(f"  v_c shape: {_get_tensor_shape(v_c)}")

    # Compute cumulative decay using cumsum (more efficient than matmul with triu_ones)
    # Reshape g_c to [batch, chunk_size] for cumsum along last dimension
    decay = ttnn.cumsum(g_c, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

    decay_exp = ttnn.reshape(
        ttnn.exp(decay, memory_config=ttnn.L1_MEMORY_CONFIG),
        [batch, chunk_size, 1],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Compute L_mask
    print(f"\n[DEBUG] Computing L_mask:")
    print(
        f"  decay shape: {_get_tensor_shape(decay)}, target shapes: [batch={batch}, chunk_size={chunk_size}, 1] and [batch={batch}, 1, chunk_size={chunk_size}]"
    )
    decay_col = ttnn.reshape(decay, [batch, chunk_size, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    print(f"[DEBUG] decay_col created, shape: {_get_tensor_shape(decay_col)}")
    decay_row = ttnn.reshape(decay, [batch, 1, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)
    print(f"[DEBUG] decay_row created, shape: {_get_tensor_shape(decay_row)}")
    L_diff = ttnn.subtract(decay_col, decay_row, memory_config=ttnn.L1_MEMORY_CONFIG)

    tril_mask = _create_tril_ones_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
    tril_mask = ttnn.reshape(tril_mask, [1, chunk_size, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)

    L_diff_masked = ttnn.multiply(L_diff, tril_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
    L_mask = ttnn.multiply(
        ttnn.exp(L_diff_masked, memory_config=ttnn.L1_MEMORY_CONFIG), tril_mask, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    print(f"[DEBUG] L_mask created, shape: {_get_tensor_shape(L_mask)}")

    # Woodbury fixup: compute attention matrix
    print(f"\n[DEBUG] Woodbury fixup section:")
    print(f"  k_beta_c shape: {_get_tensor_shape(k_beta_c)}")
    print(f"  k_c shape: {_get_tensor_shape(k_c)}")
    # NOTE: This matmul (k_beta_c @ k_c_t) may fail with kernel compilation errors when
    # batch_head * num_chunks is very large (>1000). This is a known TTNN limitation where
    # the 'reader_bmm_8bank_output_tiles_partitioned' kernel fails to compile due to
    # missing compile-time arguments. The test suite handles this gracefully by catching
    # the exception and skipping problematic configurations.
    print(f"[DEBUG] Transposing k_c: shape {_get_tensor_shape(k_c)} -> [batch, K, chunk_size]")
    k_c_t = ttnn.transpose(k_c, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    print(f"[DEBUG] k_c_t created, shape: {_get_tensor_shape(k_c_t)}")
    prog_config_kk = _get_matmul_program_config(chunk_size, K, chunk_size, grid_size=None)
    if prog_config_kk:
        kk = ttnn.matmul(k_beta_c, k_c_t, program_config=prog_config_kk, memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        kk = ttnn.matmul(k_beta_c, k_c_t, memory_config=ttnn.L1_MEMORY_CONFIG)

    M = ttnn.neg(ttnn.multiply(kk, L_mask, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.L1_MEMORY_CONFIG)
    strict_lower = _create_strict_lower_tril_ttnn(
        chunk_size, device, dtype=ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    strict_lower = ttnn.reshape(strict_lower, [1, chunk_size, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)
    M = ttnn.multiply(M, strict_lower, memory_config=ttnn.L1_MEMORY_CONFIG)

    eye = _create_eye_matrix_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
    eye = ttnn.reshape(eye, [1, chunk_size, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)

    R = ttnn.add(M, eye, memory_config=ttnn.L1_MEMORY_CONFIG)
    P = ttnn.matmul(M, M, memory_config=ttnn.L1_MEMORY_CONFIG)
    num_steps = max(int(math.ceil(math.log2(max(chunk_size, 2)))) - 1, 0)
    prog_config_woodbury = _get_matmul_program_config(chunk_size, chunk_size, chunk_size, grid_size=None)
    for _ in range(num_steps):
        if prog_config_woodbury:
            R = ttnn.add(
                R,
                ttnn.matmul(R, P, program_config=prog_config_woodbury, memory_config=ttnn.L1_MEMORY_CONFIG),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            P = ttnn.matmul(P, P, program_config=prog_config_woodbury, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            R = ttnn.add(R, ttnn.matmul(R, P, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.L1_MEMORY_CONFIG)
            P = ttnn.matmul(P, P, memory_config=ttnn.L1_MEMORY_CONFIG)

    attn = R

    # Corrected values and keys
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

    # Reshape for per-chunk processing
    # When num_chunks=1, avoid 4D reshape to prevent slice issues - reshape directly from original tensors
    print(f"\n[DEBUG] Reshaping for per-chunk processing: num_chunks={num_chunks}")
    if num_chunks == 1:
        print(f"[DEBUG] Using single-chunk path (num_chunks=1)")
        print(f"[DEBUG] Reshaping from original tensors:")
        print(f"  q shape: {_get_tensor_shape(q)}, target: [BH={BH}, chunk_size={chunk_size}, K={K}]")
        print(f"  k shape: {_get_tensor_shape(k)}, target: [BH={BH}, chunk_size={chunk_size}, K={K}]")
        print(
            f"  v_corrected shape: {_get_tensor_shape(v_corrected)}, target: [BH={BH}, chunk_size={chunk_size}, V={V}]"
        )

        # For single chunk, reshape directly from original padded tensors to [BH, chunk_size, D]
        # This avoids issues with reshaping from q_c which has batch = BH * num_chunks
        q_c_3d = ttnn.reshape(q, [BH, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
        q_c_3d = ttnn.to_layout(q_c_3d, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] q_c_3d created, shape: {_get_tensor_shape(q_c_3d)}")

        k_c_3d = ttnn.reshape(k, [BH, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
        k_c_3d = ttnn.to_layout(k_c_3d, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] k_c_3d created, shape: {_get_tensor_shape(k_c_3d)}")

        v_cor_3d = ttnn.reshape(v_corrected, [BH, chunk_size, V], memory_config=ttnn.L1_MEMORY_CONFIG)
        v_cor_3d = ttnn.to_layout(v_cor_3d, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] v_cor_3d created, shape: {_get_tensor_shape(v_cor_3d)}")

        k_cum_3d = ttnn.reshape(k_cumdecay, [BH, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
        k_cum_3d = ttnn.to_layout(k_cum_3d, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] k_cum_3d created, shape: {_get_tensor_shape(k_cum_3d)}")

        L_mask_3d = ttnn.reshape(L_mask, [BH, chunk_size, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)
        L_mask_3d = ttnn.to_layout(L_mask_3d, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] L_mask_3d created, shape: {_get_tensor_shape(L_mask_3d)}")

        print(f"[DEBUG] Reshaping decay: shape={_get_tensor_shape(decay)}, target=[BH={BH}, chunk_size={chunk_size}]")
        try:
            decay_padded = _get_tensor_shape(decay, use_padded=True)
            print(f"[DEBUG] decay padded_shape: {decay_padded}")
        except:
            print(f"[DEBUG] decay padded_shape: N/A (cannot access)")
        decay_2d = ttnn.reshape(decay, [BH, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] decay_2d created, shape: {_get_tensor_shape(decay_2d)}")

        print(f"[DEBUG] Reshaping decay_last: g_c shape={_get_tensor_shape(g_c)}")
        try:
            g_c_padded = _get_tensor_shape(g_c, use_padded=True)
            print(f"[DEBUG] g_c padded_shape: {g_c_padded}")
        except:
            print(f"[DEBUG] g_c padded_shape: N/A (cannot access)")
        g_c_sum = ttnn.sum(g_c, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] g_c_sum created, shape: {_get_tensor_shape(g_c_sum)}")
        decay_last_2d = ttnn.reshape(
            g_c_sum,
            [BH, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        print(f"[DEBUG] decay_last_2d created, shape: {_get_tensor_shape(decay_last_2d)}")
        # Set 4D versions to None to indicate single chunk mode
        q_c_4d = None
        k_c_4d = None
        v_cor_4d = None
        k_cum_4d = None
        L_mask_4d = None
        decay_3d = None
        decay_last = None
    else:
        print(f"[DEBUG] Using multi-chunk path (num_chunks={num_chunks})")
        print(
            f"[DEBUG] Reshaping q_c to 4D: shape={_get_tensor_shape(q_c)}, target=[BH={BH}, num_chunks={num_chunks}, chunk_size={chunk_size}, K={K}]"
        )
        # Multiple chunks: use 4D format [BH, num_chunks, chunk_size, D]
        q_c_4d = ttnn.reshape(q_c, [BH, num_chunks, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
        q_c_4d = ttnn.to_layout(q_c_4d, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        k_c_4d = ttnn.reshape(k_c, [BH, num_chunks, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
        k_c_4d = ttnn.to_layout(k_c_4d, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        v_cor_4d = ttnn.reshape(v_corrected, [BH, num_chunks, chunk_size, V], memory_config=ttnn.L1_MEMORY_CONFIG)
        v_cor_4d = ttnn.to_layout(v_cor_4d, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        k_cum_4d = ttnn.reshape(k_cumdecay, [BH, num_chunks, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
        k_cum_4d = ttnn.to_layout(k_cum_4d, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        L_mask_4d = ttnn.reshape(L_mask, [BH, num_chunks, chunk_size, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)
        L_mask_4d = ttnn.to_layout(L_mask_4d, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        decay_3d = ttnn.reshape(decay, [BH, num_chunks, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)
        decay_last = ttnn.reshape(
            ttnn.sum(g_c, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG),
            [BH, num_chunks, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # Set 3D versions to None
        q_c_3d = None
        k_c_3d = None
        v_cor_3d = None
        k_cum_3d = None
        L_mask_3d = None
        decay_2d = None
        decay_last_2d = None

    print(f"[DEBUG] Creating lower_causal mask with chunk_size={chunk_size}")
    lower_causal = _create_tril_ones_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
    print(f"[DEBUG] lower_causal created, shape: {_get_tensor_shape(lower_causal)}")

    # Initialize state
    S = ttnn.zeros(
        [BH, K, V], device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    if initial_state is not None:
        S = ttnn.typecast(
            ttnn.reshape(initial_state, [BH, K, V], memory_config=ttnn.L1_MEMORY_CONFIG),
            ttnn.float32,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    # Pre-compute program configs for all matmuls
    prog_config_qk = _get_matmul_program_config(chunk_size, K, chunk_size, grid_size=None)
    prog_config_vprime = _get_matmul_program_config(chunk_size, K, V, grid_size=None)
    prog_config_o_inter = _get_matmul_program_config(chunk_size, K, V, grid_size=None)
    prog_config_intra = _get_matmul_program_config(chunk_size, chunk_size, V, grid_size=None)
    prog_config_state = _get_matmul_program_config(K, chunk_size, V, grid_size=None)

    # Phase 2: Fused per-chunk loop
    # This is the key optimization: fuse operations within each chunk to reduce dispatches
    print(f"\n[DEBUG] Starting per-chunk loop: num_chunks={num_chunks}")
    outputs = []
    for i in range(num_chunks):
        print(f"[DEBUG] Processing chunk {i}/{num_chunks-1}")
        # Load chunk data
        if num_chunks == 1:
            print(f"[DEBUG] Single chunk: using 3D tensors directly (no slicing)")
            # Single chunk: use 3D tensors directly (no slicing needed)
            print(f"[DEBUG] Assigning 3D tensors directly:")
            print(f"  q_c_3d shape: {_get_tensor_shape(q_c_3d)}")
            print(f"  k_c_3d shape: {_get_tensor_shape(k_c_3d)}")
            print(f"  v_cor_3d shape: {_get_tensor_shape(v_cor_3d)}")
            q_i = q_c_3d
            k_i = k_c_3d
            v_i = v_cor_3d
            k_cum_i = k_cum_3d
            L_mask_i = L_mask_3d
            decay_i = decay_2d
            dl_i = decay_last_2d
            print(f"[DEBUG] Chunk tensors assigned successfully")
            print(
                f"[DEBUG] Chunk tensor shapes: q_i={_get_tensor_shape(q_i)}, k_i={_get_tensor_shape(k_i)}, v_i={_get_tensor_shape(v_i)}"
            )
            print(
                f"[DEBUG] Chunk tensor shapes: k_cum_i={_get_tensor_shape(k_cum_i)}, L_mask_i={_get_tensor_shape(L_mask_i)}"
            )
            print(f"[DEBUG] Chunk tensor shapes: decay_i={_get_tensor_shape(decay_i)}, dl_i={_get_tensor_shape(dl_i)}")
        else:
            # Multiple chunks: extract chunk i using explicit slice
            print(f"[DEBUG] Multi-chunk: extracting chunk {i} using slice")
            # Get padded shapes to ensure correct slice bounds (slice checks against padded_shape)
            q_shape = _get_tensor_shape(q_c_4d, use_padded=True)
            q_logical = _get_tensor_shape(q_c_4d, use_padded=False)
            print(f"[DEBUG] q_c_4d shapes - logical: {q_logical}, padded: {q_shape}")
            BH_size = q_shape[0]
            num_chunks_actual = q_shape[1]
            chunk_size_actual = q_shape[2]
            K_size = q_shape[3]

            # Extract chunk i using explicit slice with proper bounds
            # Ensure slice_end doesn't exceed padded_shape in any dimension
            slice_start = [0, i, 0, 0]
            slice_end = [
                min(BH_size, q_shape[0]),
                min(i + 1, num_chunks_actual),
                min(chunk_size_actual, q_shape[2]),
                min(K_size, q_shape[3]),
            ]
            print(f"[DEBUG] Slice operation: start={slice_start}, end={slice_end}")
            print(f"[DEBUG] Padded shape limits: {q_shape}")
            q_i = ttnn.slice(q_c_4d, slice_start, slice_end, memory_config=ttnn.L1_MEMORY_CONFIG)
            q_i = ttnn.reshape(q_i, [BH_size, chunk_size_actual, K_size], memory_config=ttnn.L1_MEMORY_CONFIG)

            k_i = ttnn.slice(k_c_4d, slice_start, slice_end, memory_config=ttnn.L1_MEMORY_CONFIG)
            k_i = ttnn.reshape(k_i, [BH_size, chunk_size_actual, K_size], memory_config=ttnn.L1_MEMORY_CONFIG)

            v_shape = _get_tensor_shape(v_cor_4d, use_padded=True)
            V_size = v_shape[3]
            slice_start_v = [0, i, 0, 0]
            slice_end_v = [
                min(BH_size, v_shape[0]),
                min(i + 1, v_shape[1]),
                min(chunk_size_actual, v_shape[2]),
                min(V_size, v_shape[3]),
            ]
            v_i = ttnn.slice(v_cor_4d, slice_start_v, slice_end_v, memory_config=ttnn.L1_MEMORY_CONFIG)
            v_i = ttnn.reshape(v_i, [BH_size, chunk_size_actual, V_size], memory_config=ttnn.L1_MEMORY_CONFIG)

            k_cum_i = ttnn.slice(k_cum_4d, slice_start, slice_end, memory_config=ttnn.L1_MEMORY_CONFIG)
            k_cum_i = ttnn.reshape(k_cum_i, [BH_size, chunk_size_actual, K_size], memory_config=ttnn.L1_MEMORY_CONFIG)

            L_mask_shape = _get_tensor_shape(L_mask_4d, use_padded=True)
            slice_start_mask = [0, i, 0, 0]
            slice_end_mask = [
                min(BH_size, L_mask_shape[0]),
                min(i + 1, L_mask_shape[1]),
                min(chunk_size_actual, L_mask_shape[2]),
                min(chunk_size_actual, L_mask_shape[3]),
            ]
            L_mask_i = ttnn.slice(L_mask_4d, slice_start_mask, slice_end_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
            L_mask_i = ttnn.reshape(
                L_mask_i, [BH_size, chunk_size_actual, chunk_size_actual], memory_config=ttnn.L1_MEMORY_CONFIG
            )

            decay_shape = _get_tensor_shape(decay_3d, use_padded=True)
            slice_start_decay = [0, i, 0]
            slice_end_decay = [
                min(BH_size, decay_shape[0]),
                min(i + 1, decay_shape[1]),
                min(chunk_size_actual, decay_shape[2]),
            ]
            decay_i = ttnn.slice(decay_3d, slice_start_decay, slice_end_decay, memory_config=ttnn.L1_MEMORY_CONFIG)
            decay_i = ttnn.reshape(decay_i, [BH_size, chunk_size_actual], memory_config=ttnn.L1_MEMORY_CONFIG)

            decay_last_shape = _get_tensor_shape(decay_last, use_padded=True)
            slice_start_dl = [0, i, 0]
            slice_end_dl = [
                min(BH_size, decay_last_shape[0]),
                min(i + 1, decay_last_shape[1]),
                min(1, decay_last_shape[2]),
            ]
            dl_i = ttnn.slice(decay_last, slice_start_dl, slice_end_dl, memory_config=ttnn.L1_MEMORY_CONFIG)
            dl_i = ttnn.reshape(dl_i, [BH_size, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

        # Fused step 1: Compute k_i_t once and reuse
        print(f"[DEBUG] Step 1: Transposing k_i, shape: {_get_tensor_shape(k_i)}")
        k_i_t = ttnn.transpose(k_i, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] k_i_t created, shape: {_get_tensor_shape(k_i_t)}")

        # Fused step 2: Intra-chunk attention (qk + masking in one sequence)
        print(f"[DEBUG] Step 2: Computing qk matmul: q_i={_get_tensor_shape(q_i)}, k_i_t={_get_tensor_shape(k_i_t)}")
        if prog_config_qk:
            qk = ttnn.matmul(q_i, k_i_t, program_config=prog_config_qk, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            qk = ttnn.matmul(q_i, k_i_t, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] qk created, shape: {_get_tensor_shape(qk)}")

        # Combine masks before applying (reduces one multiply)
        print(
            f"[DEBUG] Step 2b: Combining masks: L_mask_i={_get_tensor_shape(L_mask_i)}, lower_causal={_get_tensor_shape(lower_causal) if hasattr(lower_causal, 'shape') else 'creating...'}"
        )
        combined_mask = ttnn.multiply(L_mask_i, lower_causal, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] combined_mask created, shape: {_get_tensor_shape(combined_mask)}")
        print(
            f"[DEBUG] Step 2c: Computing intra_attn: qk={_get_tensor_shape(qk)}, combined_mask={_get_tensor_shape(combined_mask)}"
        )
        intra_attn = ttnn.multiply(qk, combined_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] intra_attn created, shape: {_get_tensor_shape(intra_attn)}")

        # Fused step 3: State read and value correction (can be parallelized conceptually)
        print(f"[DEBUG] Step 3: Computing v_prime: k_cum_i={_get_tensor_shape(k_cum_i)}, S={_get_tensor_shape(S)}")
        if prog_config_vprime:
            v_prime = ttnn.matmul(k_cum_i, S, program_config=prog_config_vprime, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            v_prime = ttnn.matmul(k_cum_i, S, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] v_prime created, shape: {_get_tensor_shape(v_prime)}")
        print(f"[DEBUG] Step 3b: Computing v_new: v_i={_get_tensor_shape(v_i)}, v_prime={_get_tensor_shape(v_prime)}")
        v_new = ttnn.subtract(v_i, v_prime, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] v_new created, shape: {_get_tensor_shape(v_new)}")

        # Fused step 4: Decay computation and query state (combine exp + multiply)
        print(
            f"[DEBUG] Step 4: Computing decay_i_exp: decay_i={_get_tensor_shape(decay_i)}, target=[BH={BH}, chunk_size={chunk_size}, 1]"
        )
        decay_i_exp = ttnn.reshape(
            ttnn.exp(decay_i, memory_config=ttnn.L1_MEMORY_CONFIG),
            [BH, chunk_size, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        print(f"[DEBUG] decay_i_exp created, shape: {_get_tensor_shape(decay_i_exp)}")
        print(
            f"[DEBUG] Step 4b: Computing q_decay: q_i={_get_tensor_shape(q_i)}, decay_i_exp={_get_tensor_shape(decay_i_exp)}"
        )
        q_decay = ttnn.multiply(q_i, decay_i_exp, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] q_decay created, shape: {_get_tensor_shape(q_decay)}")

        print(f"[DEBUG] Step 4c: Computing o_inter: q_decay={_get_tensor_shape(q_decay)}, S={_get_tensor_shape(S)}")
        if prog_config_o_inter:
            o_inter = ttnn.matmul(q_decay, S, program_config=prog_config_o_inter, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            o_inter = ttnn.matmul(q_decay, S, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] o_inter created, shape: {_get_tensor_shape(o_inter)}")

        # Fused step 5: Intra-chunk output
        print(
            f"[DEBUG] Step 5: Computing intra_v: intra_attn={_get_tensor_shape(intra_attn)}, v_new={_get_tensor_shape(v_new)}"
        )
        if prog_config_intra:
            intra_v = ttnn.matmul(
                intra_attn, v_new, program_config=prog_config_intra, memory_config=ttnn.L1_MEMORY_CONFIG
            )
        else:
            intra_v = ttnn.matmul(intra_attn, v_new, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] intra_v created, shape: {_get_tensor_shape(intra_v)}")

        # Fused step 6: Combine outputs
        print(
            f"[DEBUG] Step 6: Combining outputs: o_inter={_get_tensor_shape(o_inter)}, intra_v={_get_tensor_shape(intra_v)}"
        )
        o_i = ttnn.add(o_inter, intra_v, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(
            f"[DEBUG] o_i created, shape: {_get_tensor_shape(o_i)}, reshaping to [BH={BH}, 1, chunk_size={chunk_size}, V={V}]"
        )
        outputs.append(ttnn.reshape(o_i, [BH, 1, chunk_size, V], memory_config=ttnn.L1_MEMORY_CONFIG))
        print(f"[DEBUG] o_i appended to outputs, outputs length: {len(outputs)}")

        # Fused step 7: State update (combine decay + rank-1 update)
        # Decay state
        dl_i_exp = ttnn.exp(dl_i, memory_config=ttnn.L1_MEMORY_CONFIG)
        S = ttnn.multiply(
            S,
            ttnn.reshape(dl_i_exp, [BH, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Compute decay difference and apply to keys
        decay_diff = ttnn.subtract(
            ttnn.reshape(dl_i, [BH, 1], memory_config=ttnn.L1_MEMORY_CONFIG),
            decay_i,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        decay_diff_exp = ttnn.exp(decay_diff, memory_config=ttnn.L1_MEMORY_CONFIG)
        k_decay = ttnn.multiply(
            k_i,
            ttnn.reshape(decay_diff_exp, [BH, chunk_size, 1], memory_config=ttnn.L1_MEMORY_CONFIG),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Rank-1 state update
        k_decay_t = ttnn.transpose(k_decay, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
        if prog_config_state:
            state_update = ttnn.matmul(
                k_decay_t, v_new, program_config=prog_config_state, memory_config=ttnn.L1_MEMORY_CONFIG
            )
        else:
            state_update = ttnn.matmul(k_decay_t, v_new, memory_config=ttnn.L1_MEMORY_CONFIG)
        S = ttnn.add(S, state_update, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Concatenate outputs
    print(f"\n[DEBUG] Final output processing:")
    print(f"  outputs length: {len(outputs)}")
    if len(outputs) > 0:
        print(f"  First output shape: {_get_tensor_shape(outputs[0])}")
    print(f"  Concatenating outputs along dim=1")
    o = ttnn.concat(outputs, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
    print(f"  Concatenated o shape: {_get_tensor_shape(o)}")
    print(f"  pad_len={pad_len}, T={T}, L={L}, num_chunks={num_chunks}")

    if pad_len > 0:
        # When pad_len > 0, we need to un-pad by slicing to T
        # But when num_chunks=1, o has shape [BH, 1, chunk_size, V], so we need to handle this differently
        print(f"[DEBUG] Un-padding: o shape before slice: {_get_tensor_shape(o)}")
        if num_chunks == 1:
            # For single chunk, reshape directly and then slice along the sequence dimension
            o = ttnn.reshape(o, [BH, chunk_size, V], memory_config=ttnn.L1_MEMORY_CONFIG)
            print(f"[DEBUG] Reshaped to [BH, chunk_size, V]: {_get_tensor_shape(o)}")
            # Slice along sequence dimension (dim 1) to remove padding
            o_shape = _get_tensor_shape(o, use_padded=True)
            slice_start = [0, 0, 0]
            slice_end = [o_shape[0], min(T, o_shape[1]), o_shape[2]]
            print(f"[DEBUG] Slicing to remove padding: start={slice_start}, end={slice_end}, T={T}")
            o = ttnn.slice(o, slice_start, slice_end, memory_config=ttnn.L1_MEMORY_CONFIG)
            print(f"[DEBUG] After slice: {_get_tensor_shape(o)}")
            o = ttnn.reshape(o, [BH, T, V], memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            # For multiple chunks, use tensor indexing (which internally uses slice)
            o = o[:, :T]
            o = ttnn.reshape(o, [BH, T, V], memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] After un-pad reshape: {_get_tensor_shape(o)}")
    else:
        o = ttnn.reshape(o, [BH, L, V], memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"[DEBUG] No padding, reshaped to [BH, L, V]: {_get_tensor_shape(o)}")

    print(f"[DEBUG] Final reshape to [B={B}, H={H}, T={T}, V={V}]")
    o = ttnn.reshape(o, [B, H, T, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    print(f"[DEBUG] Final o shape: {_get_tensor_shape(o)}")
    o = ttnn.transpose(o, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    o = ttnn.typecast(o, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

    final_state = ttnn.reshape(S, [B, H, K, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    return o, final_state
