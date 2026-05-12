# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import math
import ttnn
from typing import Optional, Tuple, Dict, Any

from .ttnn_delta_rule_ops import (
    l2_norm_ttnn,
    _create_eye_matrix_ttnn,
    _create_tril_ones_ttnn,
    _create_strict_lower_tril_ttnn,
    _get_matmul_program_config,
)


_L1 = ttnn.L1_MEMORY_CONFIG


def build_fused_chunked_delta_rule_constants(
    device,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_k_dim: int,
    head_v_dim: int,
    chunk_size: int,
) -> Dict[str, Any]:
    """Eagerly build host-to-device constants used by ``fused_chunked_delta_rule_ttnn``.

    ``ttnn.ones`` / ``ttnn.zeros`` create a host buffer and then copy it to the device, which is
    not permitted during trace capture (``fd_mesh_command_queue.cpp`` asserts "Writes are not
    supported during trace capture"). Call this once, outside of any trace capture region, and
    pass the returned dict into ``fused_chunked_delta_rule_ttnn`` via ``precomputed_constants``.
    """
    B = batch_size
    H = num_heads
    T = seq_len
    K = head_k_dim
    V = head_v_dim
    BH = B * H

    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    L = T + pad_len

    tril_mask = _create_tril_ones_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=_L1)
    tril_mask = ttnn.reshape(tril_mask, [1, chunk_size, chunk_size], memory_config=_L1)

    strict_lower = _create_strict_lower_tril_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=_L1)
    strict_lower = ttnn.reshape(strict_lower, [1, chunk_size, chunk_size], memory_config=_L1)

    eye = _create_eye_matrix_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=_L1)
    eye = ttnn.reshape(eye, [1, chunk_size, chunk_size], memory_config=_L1)

    initial_state_zeros = ttnn.zeros(
        [BH, K, V], device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=_L1
    )

    pad_zeros = None
    if pad_len > 0:
        pad_zeros = {
            "q": ttnn.zeros(
                [BH, pad_len, K], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=_L1
            ),
            "k": ttnn.zeros(
                [BH, pad_len, K], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=_L1
            ),
            "v": ttnn.zeros(
                [BH, pad_len, V], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=_L1
            ),
            "beta": ttnn.zeros(
                [BH, pad_len, 1], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=_L1
            ),
            "g": ttnn.zeros(
                [BH, pad_len, 1], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=_L1
            ),
        }

    return {
        "tril_mask": tril_mask,
        "strict_lower": strict_lower,
        "eye": eye,
        "initial_state_zeros": initial_state_zeros,
        "pad_zeros": pad_zeros,
        "shape_key": (B, H, T, K, V, chunk_size, L, pad_len),
    }


def _get_tensor_shape(tensor, use_padded=False):
    """Helper to get tensor shape, handling both torch and TTNN tensors."""
    if hasattr(tensor, "shape"):
        return tensor.shape
    elif hasattr(tensor, "padded_shape") and use_padded:
        return tensor.padded_shape()
    elif hasattr(tensor, "logical_shape"):
        return tensor.logical_shape()
    else:
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
    precomputed_constants: Optional[Dict[str, Any]] = None,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Fused chunked delta rule.

    Args:
        q, k: [B, T, H, K]
        v:    [B, T, H, V]
        beta, g: [B, T, H]
    Returns:
        o: [B, T, H, V]
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

    # [B, T, H, D] -> [B, H, T, D]. Only typecast if not already bfloat16, since
    # typecast is a full-tensor dispatch (~70us host) even when it's a no-op.
    def _tr4(x):
        t = ttnn.transpose(x, 1, 2, memory_config=_L1)
        if t.dtype != ttnn.bfloat16:
            t = ttnn.typecast(t, ttnn.bfloat16, memory_config=_L1)
        return t

    q = _tr4(q)
    k = _tr4(k)
    v = _tr4(v)
    beta = _tr4(beta)  # [B, H, T]
    g = _tr4(g)  # [B, H, T]

    q = ttnn.multiply(q, scale, memory_config=_L1)

    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    L = T + pad_len
    num_chunks = L // chunk_size

    # Flatten batch/heads into leading dim.
    q = ttnn.reshape(q, [BH, T, K], memory_config=_L1)
    k = ttnn.reshape(k, [BH, T, K], memory_config=_L1)
    v = ttnn.reshape(v, [BH, T, V], memory_config=_L1)
    beta_flat = ttnn.reshape(beta, [BH, T, 1], memory_config=_L1)
    g = ttnn.reshape(g, [BH, T], memory_config=_L1)

    if pad_len > 0:
        pre_pad = (precomputed_constants or {}).get("pad_zeros")

        def _pad_zero(shape, key):
            if pre_pad is not None and key in pre_pad:
                return pre_pad[key]
            return ttnn.zeros(shape, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=_L1)

        q = ttnn.concat([q, _pad_zero([BH, pad_len, K], "q")], dim=1, memory_config=_L1)
        k = ttnn.concat([k, _pad_zero([BH, pad_len, K], "k")], dim=1, memory_config=_L1)
        v = ttnn.concat([v, _pad_zero([BH, pad_len, V], "v")], dim=1, memory_config=_L1)
        beta_flat = ttnn.concat([beta_flat, _pad_zero([BH, pad_len, 1], "beta")], dim=1, memory_config=_L1)
        g_3d = ttnn.reshape(g, [BH, T, 1], memory_config=_L1)
        g_3d = ttnn.concat([g_3d, _pad_zero([BH, pad_len, 1], "g")], dim=1, memory_config=_L1)
        g = ttnn.reshape(g_3d, [BH, L], memory_config=_L1)

    v_beta = ttnn.multiply(v, beta_flat, memory_config=_L1)
    k_beta = ttnn.multiply(k, beta_flat, memory_config=_L1)

    # Chunked views: [BH*num_chunks, chunk_size, D]. For num_chunks==1 these are shape-identical
    # to q/k/v/..., so keep the aliases cheap (a reshape is a dispatch even when no-op, so only
    # materialize if num_chunks>1).
    if num_chunks == 1:
        q_c = q
        k_c = k
        v_c = v
        k_beta_c = k_beta
        v_beta_c = v_beta
        g_c = g
        batch = BH
    else:
        batch = BH * num_chunks
        q_c = ttnn.reshape(q, [batch, chunk_size, K], memory_config=_L1)
        k_c = ttnn.reshape(k, [batch, chunk_size, K], memory_config=_L1)
        v_c = ttnn.reshape(v, [batch, chunk_size, V], memory_config=_L1)
        k_beta_c = ttnn.reshape(k_beta, [batch, chunk_size, K], memory_config=_L1)
        v_beta_c = ttnn.reshape(v_beta, [batch, chunk_size, V], memory_config=_L1)
        g_c = ttnn.reshape(g, [batch, chunk_size], memory_config=_L1)

    # Per-chunk cumulative decay.
    decay = ttnn.cumsum(g_c, dim=-1, memory_config=_L1)
    decay_col = ttnn.reshape(decay, [batch, chunk_size, 1], memory_config=_L1)
    decay_row = ttnn.reshape(decay, [batch, 1, chunk_size], memory_config=_L1)
    L_diff = ttnn.subtract(decay_col, decay_row, memory_config=_L1)

    if precomputed_constants is not None and "tril_mask" in precomputed_constants:
        tril_mask = precomputed_constants["tril_mask"]
    else:
        tril_mask = _create_tril_ones_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=_L1)
        tril_mask = ttnn.reshape(tril_mask, [1, chunk_size, chunk_size], memory_config=_L1)

    # L_mask = tril(exp(tril(L_diff))) -- pre-mask input to keep exp from blowing up on
    # the upper triangle (L_diff there is positive for decreasing cumsum).
    L_diff_masked = ttnn.multiply(L_diff, tril_mask, memory_config=_L1)
    L_mask = ttnn.multiply(ttnn.exp(L_diff_masked, memory_config=_L1), tril_mask, memory_config=_L1)

    # Woodbury-style inversion of (I - M) where M is strictly lower-triangular.
    k_c_t = ttnn.transpose(k_c, 1, 2, memory_config=_L1)
    prog_config_kk = _get_matmul_program_config(chunk_size, K, chunk_size, grid_size=None)
    kk = (
        ttnn.matmul(k_beta_c, k_c_t, program_config=prog_config_kk, memory_config=_L1)
        if prog_config_kk
        else ttnn.matmul(k_beta_c, k_c_t, memory_config=_L1)
    )

    if precomputed_constants is not None and "strict_lower" in precomputed_constants:
        strict_lower = precomputed_constants["strict_lower"]
    else:
        strict_lower = _create_strict_lower_tril_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=_L1)
        strict_lower = ttnn.reshape(strict_lower, [1, chunk_size, chunk_size], memory_config=_L1)

    # M = -(kk * L_mask) * strict_lower  ->  fold via neg(strict_lower * L_mask): same op count
    # but we can cut one op by doing kk * L_mask then multiplying by a pre-negated strict_lower
    # mask -- however this still needs two mults + a neg (3 ops). Keep it simple and correct.
    M = ttnn.neg(ttnn.multiply(kk, L_mask, memory_config=_L1), memory_config=_L1)
    M = ttnn.multiply(M, strict_lower, memory_config=_L1)

    if precomputed_constants is not None and "eye" in precomputed_constants:
        eye = precomputed_constants["eye"]
    else:
        eye = _create_eye_matrix_ttnn(chunk_size, device, dtype=ttnn.float32, memory_config=_L1)
        eye = ttnn.reshape(eye, [1, chunk_size, chunk_size], memory_config=_L1)

    R = ttnn.add(M, eye, memory_config=_L1)
    P = ttnn.matmul(M, M, memory_config=_L1)
    num_steps = max(int(math.ceil(math.log2(max(chunk_size, 2)))) - 1, 0)
    prog_config_ww = _get_matmul_program_config(chunk_size, chunk_size, chunk_size, grid_size=None)
    for _ in range(num_steps):
        if prog_config_ww:
            R = ttnn.add(R, ttnn.matmul(R, P, program_config=prog_config_ww, memory_config=_L1), memory_config=_L1)
            P = ttnn.matmul(P, P, program_config=prog_config_ww, memory_config=_L1)
        else:
            R = ttnn.add(R, ttnn.matmul(R, P, memory_config=_L1), memory_config=_L1)
            P = ttnn.matmul(P, P, memory_config=_L1)

    attn = R  # [batch, chunk_size, chunk_size]

    decay_exp = ttnn.exp(decay_col, memory_config=_L1)  # already [batch, chunk_size, 1]

    prog_config_vcorr = _get_matmul_program_config(chunk_size, chunk_size, V, grid_size=None)
    v_corrected = (
        ttnn.matmul(attn, v_beta_c, program_config=prog_config_vcorr, memory_config=_L1)
        if prog_config_vcorr
        else ttnn.matmul(attn, v_beta_c, memory_config=_L1)
    )

    k_beta_decay = ttnn.multiply(k_beta_c, decay_exp, memory_config=_L1)
    prog_config_kcum = _get_matmul_program_config(chunk_size, chunk_size, K, grid_size=None)
    k_cumdecay = (
        ttnn.matmul(attn, k_beta_decay, program_config=prog_config_kcum, memory_config=_L1)
        if prog_config_kcum
        else ttnn.matmul(attn, k_beta_decay, memory_config=_L1)
    )

    # Initial recurrent state.
    if initial_state is not None:
        S = ttnn.typecast(ttnn.reshape(initial_state, [BH, K, V], memory_config=_L1), ttnn.float32, memory_config=_L1)
    elif precomputed_constants is not None and "initial_state_zeros" in precomputed_constants:
        S = precomputed_constants["initial_state_zeros"]
    else:
        S = ttnn.zeros([BH, K, V], device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=_L1)

    prog_config_qk = _get_matmul_program_config(chunk_size, K, chunk_size, grid_size=None)
    prog_config_vprime = _get_matmul_program_config(chunk_size, K, V, grid_size=None)
    prog_config_o_inter = prog_config_vprime
    prog_config_intra = _get_matmul_program_config(chunk_size, chunk_size, V, grid_size=None)
    prog_config_state = _get_matmul_program_config(K, chunk_size, V, grid_size=None)

    outputs = []
    # decay_last per chunk: last cumsum column (== sum(g_c)).  Instead of a separate sum op,
    # we materialize it as a 2D slice of `decay` later when needed.
    # decay: [batch, chunk_size] -> last element per row is the sum.
    if num_chunks == 1:
        # Single chunk fast path: no slicing, no 4D reshapes.
        qk = (
            ttnn.matmul(q_c, k_c_t, program_config=prog_config_qk, memory_config=_L1)
            if prog_config_qk
            else ttnn.matmul(q_c, k_c_t, memory_config=_L1)
        )
        # L_mask is already lower-triangular, no need to re-apply a causal mask.
        intra_attn = ttnn.multiply(qk, L_mask, memory_config=_L1)

        v_prime = (
            ttnn.matmul(k_cumdecay, S, program_config=prog_config_vprime, memory_config=_L1)
            if prog_config_vprime
            else ttnn.matmul(k_cumdecay, S, memory_config=_L1)
        )
        v_new = ttnn.subtract(v_corrected, v_prime, memory_config=_L1)

        q_decay = ttnn.multiply(q_c, decay_exp, memory_config=_L1)
        o_inter = (
            ttnn.matmul(q_decay, S, program_config=prog_config_o_inter, memory_config=_L1)
            if prog_config_o_inter
            else ttnn.matmul(q_decay, S, memory_config=_L1)
        )

        intra_v = (
            ttnn.matmul(intra_attn, v_new, program_config=prog_config_intra, memory_config=_L1)
            if prog_config_intra
            else ttnn.matmul(intra_attn, v_new, memory_config=_L1)
        )

        o = ttnn.add(o_inter, intra_v, memory_config=_L1)  # [BH, chunk_size, V]

        # decay_last = sum(g, dim=-1) keepdim: [BH, 1].  A reduce is ~5x cheaper in host
        # dispatch than slicing one column out of cumsum(decay) (which triggers
        # untilize+slice+tilize).
        decay_last_2d = ttnn.sum(g_c, dim=-1, keepdim=True, memory_config=_L1)  # [BH, 1]
        dl_exp = ttnn.exp(decay_last_2d, memory_config=_L1)
        S = ttnn.multiply(S, ttnn.reshape(dl_exp, [BH, 1, 1], memory_config=_L1), memory_config=_L1)

        decay_diff = ttnn.subtract(decay_last_2d, decay, memory_config=_L1)  # [BH, chunk_size]
        decay_diff_exp = ttnn.reshape(
            ttnn.exp(decay_diff, memory_config=_L1),
            [BH, chunk_size, 1],
            memory_config=_L1,
        )
        k_decay = ttnn.multiply(k_c, decay_diff_exp, memory_config=_L1)
        k_decay_t = ttnn.transpose(k_decay, 1, 2, memory_config=_L1)
        state_update = (
            ttnn.matmul(k_decay_t, v_new, program_config=prog_config_state, memory_config=_L1)
            if prog_config_state
            else ttnn.matmul(k_decay_t, v_new, memory_config=_L1)
        )
        S = ttnn.add(S, state_update, memory_config=_L1)

        o_full = ttnn.reshape(o, [BH, 1, chunk_size, V], memory_config=_L1)
    else:
        # Multi-chunk path: index along the chunk axis via reshape-to-4D + slice.
        q_c_4d = ttnn.reshape(q_c, [BH, num_chunks, chunk_size, K], memory_config=_L1)
        k_c_4d = ttnn.reshape(k_c, [BH, num_chunks, chunk_size, K], memory_config=_L1)
        v_cor_4d = ttnn.reshape(v_corrected, [BH, num_chunks, chunk_size, V], memory_config=_L1)
        k_cum_4d = ttnn.reshape(k_cumdecay, [BH, num_chunks, chunk_size, K], memory_config=_L1)
        L_mask_4d = ttnn.reshape(L_mask, [BH, num_chunks, chunk_size, chunk_size], memory_config=_L1)
        decay_3d = ttnn.reshape(decay, [BH, num_chunks, chunk_size], memory_config=_L1)

        # Pre-compute per-chunk decay-last via sum (cheaper than slicing cumsum).
        g_c_3d = ttnn.reshape(g_c, [BH, num_chunks, chunk_size], memory_config=_L1)
        decay_last_all = ttnn.sum(g_c_3d, dim=-1, keepdim=True, memory_config=_L1)  # [BH, num_chunks, 1]

        for i in range(num_chunks):
            q_i = ttnn.reshape(
                ttnn.slice(q_c_4d, [0, i, 0, 0], [BH, i + 1, chunk_size, K], memory_config=_L1),
                [BH, chunk_size, K],
                memory_config=_L1,
            )
            k_i = ttnn.reshape(
                ttnn.slice(k_c_4d, [0, i, 0, 0], [BH, i + 1, chunk_size, K], memory_config=_L1),
                [BH, chunk_size, K],
                memory_config=_L1,
            )
            v_i = ttnn.reshape(
                ttnn.slice(v_cor_4d, [0, i, 0, 0], [BH, i + 1, chunk_size, V], memory_config=_L1),
                [BH, chunk_size, V],
                memory_config=_L1,
            )
            k_cum_i = ttnn.reshape(
                ttnn.slice(k_cum_4d, [0, i, 0, 0], [BH, i + 1, chunk_size, K], memory_config=_L1),
                [BH, chunk_size, K],
                memory_config=_L1,
            )
            L_mask_i = ttnn.reshape(
                ttnn.slice(L_mask_4d, [0, i, 0, 0], [BH, i + 1, chunk_size, chunk_size], memory_config=_L1),
                [BH, chunk_size, chunk_size],
                memory_config=_L1,
            )
            decay_i = ttnn.reshape(
                ttnn.slice(decay_3d, [0, i, 0], [BH, i + 1, chunk_size], memory_config=_L1),
                [BH, chunk_size],
                memory_config=_L1,
            )

            k_i_t = ttnn.transpose(k_i, 1, 2, memory_config=_L1)
            qk = (
                ttnn.matmul(q_i, k_i_t, program_config=prog_config_qk, memory_config=_L1)
                if prog_config_qk
                else ttnn.matmul(q_i, k_i_t, memory_config=_L1)
            )
            intra_attn = ttnn.multiply(qk, L_mask_i, memory_config=_L1)

            v_prime = (
                ttnn.matmul(k_cum_i, S, program_config=prog_config_vprime, memory_config=_L1)
                if prog_config_vprime
                else ttnn.matmul(k_cum_i, S, memory_config=_L1)
            )
            v_new = ttnn.subtract(v_i, v_prime, memory_config=_L1)

            decay_i_exp = ttnn.reshape(
                ttnn.exp(decay_i, memory_config=_L1),
                [BH, chunk_size, 1],
                memory_config=_L1,
            )
            q_decay = ttnn.multiply(q_i, decay_i_exp, memory_config=_L1)
            o_inter = (
                ttnn.matmul(q_decay, S, program_config=prog_config_o_inter, memory_config=_L1)
                if prog_config_o_inter
                else ttnn.matmul(q_decay, S, memory_config=_L1)
            )
            intra_v = (
                ttnn.matmul(intra_attn, v_new, program_config=prog_config_intra, memory_config=_L1)
                if prog_config_intra
                else ttnn.matmul(intra_attn, v_new, memory_config=_L1)
            )
            o_i = ttnn.add(o_inter, intra_v, memory_config=_L1)
            outputs.append(ttnn.reshape(o_i, [BH, 1, chunk_size, V], memory_config=_L1))

            # dl_i = sum(g_c[:, i, :], dim=-1) -> [BH, 1], via pre-computed decay_last_all.
            dl_i = ttnn.reshape(
                ttnn.slice(decay_last_all, [0, i, 0], [BH, i + 1, 1], memory_config=_L1),
                [BH, 1],
                memory_config=_L1,
            )
            dl_exp = ttnn.exp(dl_i, memory_config=_L1)
            S = ttnn.multiply(S, ttnn.reshape(dl_exp, [BH, 1, 1], memory_config=_L1), memory_config=_L1)

            decay_diff = ttnn.subtract(dl_i, decay_i, memory_config=_L1)  # broadcast over chunk_size
            decay_diff_exp = ttnn.reshape(
                ttnn.exp(decay_diff, memory_config=_L1),
                [BH, chunk_size, 1],
                memory_config=_L1,
            )
            k_decay = ttnn.multiply(k_i, decay_diff_exp, memory_config=_L1)
            k_decay_t = ttnn.transpose(k_decay, 1, 2, memory_config=_L1)
            state_update = (
                ttnn.matmul(k_decay_t, v_new, program_config=prog_config_state, memory_config=_L1)
                if prog_config_state
                else ttnn.matmul(k_decay_t, v_new, memory_config=_L1)
            )
            S = ttnn.add(S, state_update, memory_config=_L1)

        o_full = ttnn.concat(outputs, dim=1, memory_config=_L1)  # [BH, num_chunks, chunk_size, V]

    # Un-pad / reshape back to [B, T, H, V].
    if num_chunks == 1:
        # o_full is [BH, 1, chunk_size, V]; drop the singleton chunk dim.
        o = ttnn.reshape(o_full, [BH, chunk_size, V], memory_config=_L1)
        if pad_len > 0:
            o_shape = _get_tensor_shape(o, use_padded=True)
            o = ttnn.slice(o, [0, 0, 0], [o_shape[0], min(T, o_shape[1]), o_shape[2]], memory_config=_L1)
            o = ttnn.reshape(o, [BH, T, V], memory_config=_L1)
    else:
        # [BH, num_chunks, chunk_size, V] -> [BH, L, V]
        o = ttnn.reshape(o_full, [BH, L, V], memory_config=_L1)
        if pad_len > 0:
            o = o[:, :T]
            o = ttnn.reshape(o, [BH, T, V], memory_config=_L1)

    o = ttnn.reshape(o, [B, H, T, V], memory_config=_L1)
    o = ttnn.transpose(o, 1, 2, memory_config=_L1)
    o = ttnn.typecast(o, ttnn.bfloat16, memory_config=_L1)

    final_state = ttnn.reshape(S, [B, H, K, V], memory_config=_L1)
    return o, final_state
