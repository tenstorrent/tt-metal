# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


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

    q_c = ttnn.reshape(q, [batch, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    k_c = ttnn.reshape(k, [batch, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    v_c = ttnn.reshape(v, [batch, chunk_size, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    k_beta_c = ttnn.reshape(k_beta, [batch, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    v_beta_c = ttnn.reshape(v_beta, [batch, chunk_size, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    g_c = ttnn.reshape(g, [batch, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)

    decay = ttnn.cumsum(g_c, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

    decay_exp = ttnn.reshape(
        ttnn.exp(decay, memory_config=ttnn.L1_MEMORY_CONFIG),
        [batch, chunk_size, 1],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    decay_col = ttnn.reshape(decay, [batch, chunk_size, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    decay_row = ttnn.reshape(decay, [batch, 1, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)
    L_diff = ttnn.subtract(decay_col, decay_row, memory_config=ttnn.L1_MEMORY_CONFIG)

    tril_mask = _create_tril_ones_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
    tril_mask = ttnn.reshape(tril_mask, [1, chunk_size, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)

    L_diff_masked = ttnn.multiply(L_diff, tril_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
    L_mask = ttnn.multiply(
        ttnn.exp(L_diff_masked, memory_config=ttnn.L1_MEMORY_CONFIG), tril_mask, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # NOTE: This matmul (k_beta_c @ k_c_t) may fail with kernel compilation errors when
    # batch_head * num_chunks is very large (>1000). This is a known TTNN limitation where
    # the 'reader_bmm_8bank_output_tiles_partitioned' kernel fails to compile due to
    # missing compile-time arguments. The test suite handles this gracefully by catching
    # the exception and skipping problematic configurations.
    k_c_t = ttnn.transpose(k_c, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
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

    if num_chunks == 1:
        q_c_3d = ttnn.reshape(q, [BH, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
        q_c_3d = ttnn.to_layout(q_c_3d, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        k_c_3d = ttnn.reshape(k, [BH, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
        k_c_3d = ttnn.to_layout(k_c_3d, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        v_cor_3d = ttnn.reshape(v_corrected, [BH, chunk_size, V], memory_config=ttnn.L1_MEMORY_CONFIG)
        v_cor_3d = ttnn.to_layout(v_cor_3d, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        k_cum_3d = ttnn.reshape(k_cumdecay, [BH, chunk_size, K], memory_config=ttnn.L1_MEMORY_CONFIG)
        k_cum_3d = ttnn.to_layout(k_cum_3d, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        L_mask_3d = ttnn.reshape(L_mask, [BH, chunk_size, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)
        L_mask_3d = ttnn.to_layout(L_mask_3d, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        decay_2d = ttnn.reshape(decay, [BH, chunk_size], memory_config=ttnn.L1_MEMORY_CONFIG)

        g_c_sum = ttnn.sum(g_c, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        decay_last_2d = ttnn.reshape(
            g_c_sum,
            [BH, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        q_c_4d = None
        k_c_4d = None
        v_cor_4d = None
        k_cum_4d = None
        L_mask_4d = None
        decay_3d = None
        decay_last = None
    else:
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
        q_c_3d = None
        k_c_3d = None
        v_cor_3d = None
        k_cum_3d = None
        L_mask_3d = None
        decay_2d = None
        decay_last_2d = None

    lower_causal = _create_tril_ones_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)

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

    outputs = []
    for i in range(num_chunks):
        if num_chunks == 1:
            q_i = q_c_3d
            k_i = k_c_3d
            v_i = v_cor_3d
            k_cum_i = k_cum_3d
            L_mask_i = L_mask_3d
            decay_i = decay_2d
            dl_i = decay_last_2d
        else:
            q_shape = _get_tensor_shape(q_c_4d, use_padded=True)
            BH_size = q_shape[0]
            num_chunks_actual = q_shape[1]
            chunk_size_actual = q_shape[2]
            K_size = q_shape[3]

            slice_start = [0, i, 0, 0]
            slice_end = [
                min(BH_size, q_shape[0]),
                min(i + 1, num_chunks_actual),
                min(chunk_size_actual, q_shape[2]),
                min(K_size, q_shape[3]),
            ]
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

        k_i_t = ttnn.transpose(k_i, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)

        if prog_config_qk:
            qk = ttnn.matmul(q_i, k_i_t, program_config=prog_config_qk, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            qk = ttnn.matmul(q_i, k_i_t, memory_config=ttnn.L1_MEMORY_CONFIG)

        combined_mask = ttnn.multiply(L_mask_i, lower_causal, memory_config=ttnn.L1_MEMORY_CONFIG)
        intra_attn = ttnn.multiply(qk, combined_mask, memory_config=ttnn.L1_MEMORY_CONFIG)

        if prog_config_vprime:
            v_prime = ttnn.matmul(k_cum_i, S, program_config=prog_config_vprime, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            v_prime = ttnn.matmul(k_cum_i, S, memory_config=ttnn.L1_MEMORY_CONFIG)
        v_new = ttnn.subtract(v_i, v_prime, memory_config=ttnn.L1_MEMORY_CONFIG)

        decay_i_exp = ttnn.reshape(
            ttnn.exp(decay_i, memory_config=ttnn.L1_MEMORY_CONFIG),
            [BH, chunk_size, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        q_decay = ttnn.multiply(q_i, decay_i_exp, memory_config=ttnn.L1_MEMORY_CONFIG)

        if prog_config_o_inter:
            o_inter = ttnn.matmul(q_decay, S, program_config=prog_config_o_inter, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            o_inter = ttnn.matmul(q_decay, S, memory_config=ttnn.L1_MEMORY_CONFIG)

        if prog_config_intra:
            intra_v = ttnn.matmul(
                intra_attn, v_new, program_config=prog_config_intra, memory_config=ttnn.L1_MEMORY_CONFIG
            )
        else:
            intra_v = ttnn.matmul(intra_attn, v_new, memory_config=ttnn.L1_MEMORY_CONFIG)

        o_i = ttnn.add(o_inter, intra_v, memory_config=ttnn.L1_MEMORY_CONFIG)
        outputs.append(ttnn.reshape(o_i, [BH, 1, chunk_size, V], memory_config=ttnn.L1_MEMORY_CONFIG))

        dl_i_exp = ttnn.exp(dl_i, memory_config=ttnn.L1_MEMORY_CONFIG)
        dl_i_exp = ttnn.exp(dl_i, memory_config=ttnn.L1_MEMORY_CONFIG)
        S = ttnn.multiply(
            S,
            ttnn.reshape(dl_i_exp, [BH, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

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

        k_decay_t = ttnn.transpose(k_decay, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
        if prog_config_state:
            state_update = ttnn.matmul(
                k_decay_t, v_new, program_config=prog_config_state, memory_config=ttnn.L1_MEMORY_CONFIG
            )
        else:
            state_update = ttnn.matmul(k_decay_t, v_new, memory_config=ttnn.L1_MEMORY_CONFIG)
        S = ttnn.add(S, state_update, memory_config=ttnn.L1_MEMORY_CONFIG)

    o = ttnn.concat(outputs, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

    if pad_len > 0:
        if num_chunks == 1:
            o = ttnn.reshape(o, [BH, chunk_size, V], memory_config=ttnn.L1_MEMORY_CONFIG)
            o_shape = _get_tensor_shape(o, use_padded=True)
            slice_start = [0, 0, 0]
            slice_end = [o_shape[0], min(T, o_shape[1]), o_shape[2]]
            o = ttnn.slice(o, slice_start, slice_end, memory_config=ttnn.L1_MEMORY_CONFIG)
            o = ttnn.reshape(o, [BH, T, V], memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            o = o[:, :T]
            o = ttnn.reshape(o, [BH, T, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        o = ttnn.reshape(o, [BH, L, V], memory_config=ttnn.L1_MEMORY_CONFIG)

    o = ttnn.reshape(o, [B, H, T, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    o = ttnn.transpose(o, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    o = ttnn.typecast(o, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

    final_state = ttnn.reshape(S, [B, H, K, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    return o, final_state
