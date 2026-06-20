# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Chunked gated delta rule using the C++ `gated_delta_attn_seq` kernel (Path A).

Python preprocessing computes:
  - All cheap elementwise ops and two matmuls (kk, intra_attn).
  - L_inv: 4 diagonal block inverses of L_mat via _solve_lower_triangular_ttnn
    (D^{-1} Neumann + 2 Newton-Schulz steps — same path as parallel scan).
The C++ kernel performs blocked forward substitution + inter-chunk state scan.
"""

import torch

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.gdn_chunk_ops import (
    _create_tril_ones,
    _create_triu_ones,
    _solve_lower_triangular_ttnn,
)

_TILE = 32


def _compute_L_inv_ttnn(L_mat_4d, BH, NC, C, mesh_device, _cmc=None, eye_32=None):
    """Compute diagonal block inverses of L_mat using _solve_lower_triangular_ttnn.

    L_mat_4d: [BH, NC, C, C] float32 lower-triangular, positive diagonal (~2)
    eye_32:   [1, 32, 32] float32 identity pre-allocated on device (required for trace compat)
    Returns:  [BH, NC, C, 32] float32 — 4 diagonal block inverses stacked as [C, 32]

    Each 32x32 diagonal block B is inverted via the same D^{-1}-Neumann-NS path
    used by _solve_lower_triangular_ttnn in the parallel scan.
    """
    if eye_32 is None:
        # Fallback for tests that don't pass cached_masks — not trace-compatible.
        eye_32 = ttnn.from_torch(
            torch.eye(32, dtype=torch.float32).unsqueeze(0),  # [1, 32, 32]
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    Ct = C // 32  # = 4
    batch = BH * NC
    L_flat = ttnn.reshape(L_mat_4d, [batch, C, C], memory_config=_cmc)

    inv_blocks = []
    for b in range(Ct):
        row_start = b * 32
        col_start = b * 32
        block = ttnn.slice(
            L_flat, [0, row_start, col_start], [batch, row_start + 32, col_start + 32], memory_config=_cmc
        )
        # _solve_lower_triangular_ttnn: D^{-1} normalization + Neumann + 2 NS steps
        block_inv = _solve_lower_triangular_ttnn(block, eye_32, mesh_device)
        ttnn.deallocate(block)
        inv_blocks.append(block_inv)  # [batch, 32, 32]

    # Do NOT deallocate L_flat: it is a reshape (view) of L_mat_4d which is the
    # same L_unit_4d tensor passed as a kernel input.  Freeing L_flat frees L_unit_4d's
    # buffer while the C++ kernel still needs to read from it.

    L_inv_flat = ttnn.concat(inv_blocks, dim=1, memory_config=_cmc)
    for blk in inv_blocks:
        ttnn.deallocate(blk)

    # Do NOT deallocate L_inv_flat here — ttnn.reshape returns a view that shares
    # the same DRAM buffer. Freeing L_inv_flat while L_inv_4d (the view) is still
    # in use causes the kernel to read from freed memory on later runs when the
    # allocator reuses that address. The caller's ttnn.deallocate(L_inv_4d) will
    # release the buffer when it is no longer needed.
    L_inv_4d = ttnn.reshape(L_inv_flat, [BH, NC, C, 32], memory_config=_cmc)
    return L_inv_4d


def chunk_gated_delta_rule_seq(
    q,  # [BH, T, K] float32 on mesh
    k,  # [BH, T, K] float32 on mesh
    v,  # [BH, T, V] float32 on mesh
    beta,  # [BH, T, 1] float32 on mesh
    g,  # [BH, T]    float32 on mesh
    chunk_size=128,
    scale=None,
    initial_state=None,  # [BH, K, V] float32 or None
    mesh_device=None,
    cached_masks=None,
):
    """Chunked gated delta rule using C++ sequential scan kernel (Path A).

    Python preprocessing: ~9ms (vs ~40ms in the previous Path A-lite version).
    C++ kernel: triangular solve + inter-chunk state scan (~1.5ms).
    Total per GDN layer: ~28ms (vs 57ms before).
    """
    _hifi_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    BH = q.shape[0]
    T = q.shape[1]
    K = q.shape[2]
    V = v.shape[2]

    if scale is None:
        scale = K**-0.5

    q = ttnn.multiply(q, scale, memory_config=None)

    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    L = T + pad_len
    num_chunks = L // chunk_size
    batch = BH * num_chunks

    beta_flat = beta
    if pad_len > 0:
        zeros_q = ttnn.zeros(
            [BH, pad_len, K], device=mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=None
        )
        zeros_v = ttnn.zeros(
            [BH, pad_len, V], device=mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=None
        )
        zeros_beta = ttnn.zeros(
            [BH, pad_len, 1], device=mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=None
        )
        q = ttnn.concat([q, zeros_q], dim=1, memory_config=None)
        k = ttnn.concat([k, zeros_q], dim=1, memory_config=None)
        v = ttnn.concat([v, zeros_v], dim=1, memory_config=None)
        beta_flat = ttnn.concat([beta_flat, zeros_beta], dim=1, memory_config=None)
        g_3d = ttnn.reshape(g, [BH, T, 1])
        ttnn.deallocate(g)
        zeros_g = ttnn.zeros(
            [BH, pad_len, 1], device=mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=None
        )
        g_3d = ttnn.concat([g_3d, zeros_g], dim=1, memory_config=None)
        g = ttnn.reshape(g_3d, [BH, L])
        beta_flat = ttnn.reshape(beta_flat, [BH, L, 1])
    else:
        beta_flat = ttnn.reshape(beta_flat, [BH, L, 1])

    v_beta = ttnn.multiply(v, beta_flat, memory_config=None)
    k_beta = ttnn.multiply(k, beta_flat, memory_config=None)
    del beta_flat

    q_c = ttnn.reshape(q, [batch, chunk_size, K], memory_config=None)
    k_c = ttnn.reshape(k, [batch, chunk_size, K], memory_config=None)
    k_beta_c = ttnn.reshape(k_beta, [batch, chunk_size, K], memory_config=None)
    v_beta_c = ttnn.reshape(v_beta, [batch, chunk_size, V], memory_config=None)
    g_c = ttnn.reshape(g, [batch, chunk_size], memory_config=None)
    del q, v, k_beta, v_beta

    _eye_32 = None
    if cached_masks is not None:
        triu_ones = cached_masks["triu_ones"]
        tril_mask = cached_masks["tril_mask"]
        _eye_1cc = cached_masks["eye"]
        lower_causal = cached_masks["lower_causal"]
        _eye_32 = cached_masks.get("eye_32")
    else:
        triu_ones = _create_triu_ones(chunk_size, mesh_device, dtype=ttnn.float32)
        triu_ones = ttnn.reshape(triu_ones, [1, chunk_size, chunk_size])
        tril_mask = _create_tril_ones(chunk_size, mesh_device, dtype=ttnn.float32)
        tril_mask = ttnn.reshape(tril_mask, [1, chunk_size, chunk_size])
        _eye_1cc = ttnn.from_torch(
            torch.eye(chunk_size, dtype=torch.float32).unsqueeze(0),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        lower_causal = _create_tril_ones(chunk_size, mesh_device, dtype=ttnn.float32)

    _cmc = ttnn.DRAM_MEMORY_CONFIG if chunk_size > 64 else None

    # ----------------------------------------------------------------
    # Decay preprocessing (unchanged from prior version)
    # ----------------------------------------------------------------
    g_c_3d = ttnn.reshape(g_c, [batch, 1, chunk_size], memory_config=None)
    decay = ttnn.reshape(
        ttnn.matmul(g_c_3d, triu_ones, memory_config=None, compute_kernel_config=_hifi_cfg),
        [batch, chunk_size],
        memory_config=None,
    )
    decay_offset = decay[:, 0:1]
    decay_raw = decay
    decay = ttnn.subtract(decay_raw, decay_offset, memory_config=None)
    ttnn.deallocate(decay_offset)

    decay_exp = ttnn.reshape(
        ttnn.exp(ttnn.clip(decay_raw, min=-20.0, max=0.0), memory_config=None),
        [batch, chunk_size, 1],
        memory_config=None,
    )

    decay_col = ttnn.reshape(decay, [batch, chunk_size, 1], memory_config=None)
    decay_row = ttnn.reshape(decay, [batch, 1, chunk_size], memory_config=None)
    L_diff = ttnn.subtract(decay_col, decay_row, memory_config=_cmc)
    del decay_col, decay_row

    L_diff_masked = ttnn.multiply(L_diff, tril_mask, memory_config=_cmc)
    ttnn.deallocate(L_diff)
    L_diff_clamped = ttnn.clip(L_diff_masked, min=-20.0, max=0.0)
    ttnn.deallocate(L_diff_masked)
    L_mask = ttnn.multiply(ttnn.exp(L_diff_clamped, memory_config=_cmc), tril_mask, memory_config=_cmc)
    ttnn.deallocate(L_diff_clamped)

    # ----------------------------------------------------------------
    # kk = k_beta @ k.T  [batch, C, C]  (1 matmul dispatch)
    # ----------------------------------------------------------------
    del k
    k_c = ttnn.move(k_c)
    k_c_t = ttnn.transpose(k_c, 1, 2, memory_config=_cmc)
    kk = ttnn.matmul(k_beta_c, k_c_t, memory_config=_cmc, compute_kernel_config=_hifi_cfg)
    ttnn.deallocate(k_c_t)

    # ----------------------------------------------------------------
    # Build L_mat = I + kk * L_mask  (2 cheap elementwise dispatches)
    # ----------------------------------------------------------------
    L_mat = ttnn.add(
        _eye_1cc,
        ttnn.multiply(kk, L_mask, memory_config=_cmc),
        memory_config=_cmc,
    )
    ttnn.deallocate(kk)

    # ----------------------------------------------------------------
    # Normalize L_mat to unit-diagonal form: L_unit = D^{-1} L_mat
    # Keeps off-diagonal correction values smaller → better float32 precision
    # in blocked forward substitution.
    # ----------------------------------------------------------------
    D_mat = ttnn.multiply(L_mat, _eye_1cc, memory_config=_cmc)
    D_diag = ttnn.sum(D_mat, dim=-1, memory_config=_cmc)
    D_inv = ttnn.reciprocal(D_diag, memory_config=_cmc)
    ttnn.deallocate(D_diag)
    D_inv_row = ttnn.reshape(D_inv, [batch, chunk_size, 1], memory_config=_cmc)

    L_strict = ttnn.subtract(L_mat, D_mat, memory_config=_cmc)
    ttnn.deallocate(D_mat)
    ttnn.deallocate(L_mat)
    N = ttnn.multiply(D_inv_row, L_strict, memory_config=_cmc)
    ttnn.deallocate(L_strict)
    L_unit = ttnn.add(_eye_1cc, N, memory_config=_cmc)
    ttnn.deallocate(N)

    v_beta_sc = ttnn.multiply(D_inv_row, v_beta_c, memory_config=_cmc)
    del v_beta_c
    k_beta_decay = ttnn.multiply(k_beta_c, decay_exp, memory_config=_cmc)
    k_bd_sc = ttnn.multiply(D_inv_row, k_beta_decay, memory_config=_cmc)
    ttnn.deallocate(k_beta_decay)
    ttnn.deallocate(D_inv_row)

    # ----------------------------------------------------------------
    # Precompute intra_attn: q_decay @ k.T * L_mask * lower_causal
    # (1 matmul dispatch — still cheaper than full solve)
    # ----------------------------------------------------------------
    decay_3d = ttnn.reshape(decay, [BH, num_chunks, chunk_size], memory_config=None)
    decay_raw_3d = ttnn.reshape(decay_raw, [BH, num_chunks, chunk_size], memory_config=None)

    decay_last_raw = ttnn.reshape(
        ttnn.sum(g_c, dim=-1, memory_config=None),
        [BH, num_chunks, 1],
        memory_config=None,
    )
    decay_last_normalized = ttnn.reshape(decay_3d[:, :, -1:], [BH, num_chunks, 1], memory_config=None)

    decay_raw_exp_4d = ttnn.reshape(
        ttnn.exp(ttnn.clip(decay_raw_3d, min=-20.0, max=0.0), memory_config=_cmc),
        [BH, num_chunks, chunk_size, 1],
        memory_config=_cmc,
    )
    q_c_4d = ttnn.to_layout(
        ttnn.reshape(q_c, [BH, num_chunks, chunk_size, K], memory_config=None),
        ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    q_decay_4d = ttnn.multiply(q_c_4d, decay_raw_exp_4d, memory_config=_cmc)
    ttnn.deallocate(decay_raw_exp_4d)

    decay_last_norm_4d = ttnn.reshape(decay_last_normalized, [BH, num_chunks, 1], memory_config=_cmc)
    decay_diff_3d = ttnn.subtract(decay_last_norm_4d, decay_3d, memory_config=_cmc)
    decay_diff_exp_4d = ttnn.reshape(
        ttnn.exp(ttnn.clip(decay_diff_3d, min=-20.0, max=0.0), memory_config=_cmc),
        [BH, num_chunks, chunk_size, 1],
        memory_config=_cmc,
    )
    k_c_4d = ttnn.to_layout(
        ttnn.reshape(k_c, [BH, num_chunks, chunk_size, K], memory_config=None),
        ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    k_decay_4d = ttnn.multiply(k_c_4d, decay_diff_exp_4d, memory_config=_cmc)
    ttnn.deallocate(decay_diff_exp_4d)
    k_decay_t_4d = ttnn.transpose(k_decay_4d, 2, 3, memory_config=_cmc)
    ttnn.deallocate(k_decay_4d)

    dl_exp_3d = ttnn.exp(ttnn.clip(decay_last_raw, min=-20.0, max=0.0), memory_config=_cmc)
    dl_exp_4d = ttnn.reshape(
        ttnn.to_layout(
            ttnn.typecast(dl_exp_3d, ttnn.float32, memory_config=_cmc)
            if dl_exp_3d.dtype != ttnn.float32
            else dl_exp_3d,
            ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        [BH, num_chunks, 1, 1],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    L_mask_4d = ttnn.reshape(L_mask, [BH, num_chunks, chunk_size, chunk_size], memory_config=None)
    L_mask_4d = ttnn.to_layout(L_mask_4d, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    lower_causal_4d = ttnn.reshape(lower_causal, [1, 1, chunk_size, chunk_size], memory_config=None)
    combined_mask_4d = ttnn.multiply(L_mask_4d, lower_causal_4d, memory_config=_cmc)
    ttnn.deallocate(L_mask_4d)
    k_c_4d_t = ttnn.transpose(k_c_4d, 2, 3, memory_config=_cmc)
    qk_4d = ttnn.matmul(q_c_4d, k_c_4d_t, memory_config=_cmc, compute_kernel_config=_hifi_cfg)
    ttnn.deallocate(k_c_4d_t)
    intra_attn_4d = ttnn.multiply(qk_4d, combined_mask_4d, memory_config=_cmc)
    ttnn.deallocate(qk_4d)
    ttnn.deallocate(combined_mask_4d)

    # ----------------------------------------------------------------
    # Reshape preprocessing outputs to 4D for C++ kernel
    # ----------------------------------------------------------------
    def _to4d_f32(t, d1, d2):
        t4 = ttnn.reshape(t, [BH, num_chunks, d1, d2], memory_config=None)
        return ttnn.to_layout(t4, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    L_unit_4d = _to4d_f32(L_unit, chunk_size, chunk_size)
    # Do NOT deallocate L_unit/v_beta_sc/k_bd_sc here: _to4d_f32 returns a reshape
    # (view) followed by to_layout (no-op when already TILE+DRAM), so the 4D tensor
    # aliases the original buffer.  Calling ttnn.deallocate on the original while the
    # view is still in use as a kernel input causes use-after-free on the 3rd+ call.
    v_beta_sc_4d = _to4d_f32(v_beta_sc, chunk_size, V)
    k_bd_sc_4d = _to4d_f32(k_bd_sc, chunk_size, K)

    def _ensure_f32_dram(t):
        if t.dtype != ttnn.float32:
            t = ttnn.typecast(t, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        elif t.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            t = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        return t

    L_unit_4d = _ensure_f32_dram(L_unit_4d)
    v_beta_sc_4d = _ensure_f32_dram(v_beta_sc_4d)
    k_bd_sc_4d = _ensure_f32_dram(k_bd_sc_4d)
    intra_attn_4d = _ensure_f32_dram(intra_attn_4d)
    q_decay_4d = _ensure_f32_dram(q_decay_4d)
    k_decay_t_4d = _ensure_f32_dram(k_decay_t_4d)

    # Compute diagonal block inverses of L_unit via Neumann+NS (_solve_lower_triangular_ttnn).
    # L_unit has unit diagonal so D^{-1} inside _solve_lower_triangular_ttnn is trivial (D=I),
    # and the 2 NS steps correct any float32 error from Neumann on the nilpotent N.
    L_inv_4d = _compute_L_inv_ttnn(L_unit_4d, BH, num_chunks, chunk_size, mesh_device, _cmc, eye_32=_eye_32)

    # Initial state
    S0_tt = None
    if initial_state is not None:
        S0_tt = ttnn.typecast(
            ttnn.reshape(initial_state, [BH, K, V], memory_config=None),
            ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ----------------------------------------------------------------
    # C++ sequential scan kernel (Path A)
    # ----------------------------------------------------------------
    out_4d, final_state = ttnn.transformer.gated_delta_attn_seq(
        L_unit_4d,
        v_beta_sc_4d,
        k_bd_sc_4d,
        intra_attn_4d,
        q_decay_4d,
        k_decay_t_4d,
        dl_exp_4d,
        L_inv_4d,
        initial_state=S0_tt,
    )
    ttnn.deallocate(L_inv_4d)

    # Reshape output to [BH, L, V]
    out_4d = ttnn.to_layout(
        ttnn.typecast(out_4d, ttnn.float32, memory_config=None) if out_4d.dtype != ttnn.float32 else out_4d,
        ttnn.TILE_LAYOUT,
        memory_config=None,
    )
    o = ttnn.reshape(out_4d, [BH, L, V], memory_config=None)

    if pad_len > 0:
        o = o[:, :T, :]
        o = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=None)

    return o, final_state
