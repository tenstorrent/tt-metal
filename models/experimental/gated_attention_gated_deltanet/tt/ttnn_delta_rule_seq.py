# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Chunk-parallel gated delta rule using the C++ `ttnn.transformer.gated_delta_attn_seq`
kernel (Path A) — imported from the Qwen3.5-27B branch (gdn_chunk_ops_seq.py /
gdn_chunk_ops.py) and adapted for the single-device Qwen3.5-9B model.

Python preprocessing computes (all float32):
  - cheap elementwise ops + two matmuls (kk, intra_attn),
  - L_inv: 4 diagonal block inverses of L_unit via `_solve_lower_triangular_ttnn`
    (D^{-1} Neumann doubling + 2 Newton-Schulz steps — accurate intra-chunk inverse).
The C++ kernel performs blocked forward substitution + the sequential inter-chunk
state scan.

Tensor layout convention (FLA style):
  q, k: [BH, T, K]   beta: [BH, T, 1]   g: [BH, T]   v: [BH, T, V]   state: [BH, K, V]
"""

import math as _math
import os as _os

import torch

import ttnn

_DBG = _os.environ.get("QWEN9B_GDN_DBG")


def _ck(name, t):
    if not _DBG:
        return
    try:
        x = ttnn.to_torch(t).float()
        print(
            f"  [dbg] {name}: max|.|={x.abs().max().item():.4g} min={x.min().item():.4g} inf={bool(torch.isinf(x).any())} nan={bool(torch.isnan(x).any())}",
            flush=True,
        )
    except Exception as e:
        print(f"  [dbg] {name}: <err {e}>", flush=True)


# Mask helpers are shared with the existing (bf16) chunk path. Relative import resolves
# whether this package is reached via its fully-qualified path or as a top-level `tt`.
from .ttnn_delta_rule_ops import (
    _create_triu_ones_ttnn as _create_triu_ones,
    _create_tril_ones_ttnn as _create_tril_ones,
    l2_norm_ttnn,
)

_TILE = 32

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def chunk_gated_delta_rule_seq_adapter(
    q,  # [B, T, H, K]
    k,  # [B, T, H, K]
    v,  # [B, T, H, V]
    beta,  # [B, T, H]
    g,  # [B, T, H]
    chunk_size=128,
    scale=None,
    initial_state=None,  # [B, H, K, V] (any dtype) or None
    device=None,
    cached_masks=None,
    valid_len=None,
):
    """Drop-in replacement for chunk_gated_delta_rule_ttnn that runs the C++
    chunk-parallel `gated_delta_attn_seq` kernel.

    Same interface ([B,T,H,*] inputs, returns (o [B,T,H,V], new_state [B,H,K,V])).
    Internally L2-norms q/k (matching the bf16 chunk path), converts to the seq
    kernel's [BH,T,*] float32 layout, runs the kernel, and converts the output
    and final state back. final_state is returned as bfloat16 to match the
    decode recurrent_state dtype.
    """
    B = q.shape[0]
    T = q.shape[1]
    H = q.shape[2]
    K = q.shape[3]
    V = v.shape[3]
    BH = B * H

    def _tilize_f32(t):
        # Tilize + fp32 cast in one op, which avoids a separate typecast pass.
        return ttnn.to_layout(t, ttnn.TILE_LAYOUT, dtype=ttnn.float32, memory_config=_DRAM)

    def _to_bhtd(t, D):  # [B,T,H,D] -> [BH,T,D] float32 TILE/DRAM (ROW_MAJOR-correct)
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=_DRAM)
        t = ttnn.reshape(t, [B, T, H, D])
        t = ttnn.permute(t, (0, 2, 1, 3))  # [B,H,T,D]
        t = ttnn.reshape(t, [BH, T, D])
        return _tilize_f32(t)

    def _to_bht(t):  # [B,T,H] -> [BH,T] float32 TILE/DRAM
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=_DRAM)
        t = ttnn.reshape(t, [B, T, H])
        t = ttnn.permute(t, (0, 2, 1))  # [B,H,T]
        t = ttnn.reshape(t, [BH, T])
        return _tilize_f32(t)

    q_bh = _to_bhtd(q, K)
    k_bh = _to_bhtd(k, K)
    v_bh = _to_bhtd(v, V)

    q_bh = l2_norm_ttnn(q_bh, dim=-1)
    k_bh = l2_norm_ttnn(k_bh, dim=-1)
    g_bh = _to_bht(g)
    beta_bh = ttnn.reshape(_to_bht(beta), [BH, T, 1])

    # initial_state [B,H,K,V] -> [BH,K,V] (leading-dim merge; seq typecasts to f32).
    s0 = None
    if initial_state is not None:
        s0 = ttnn.reshape(initial_state, [BH, K, V])

    o_bh, final_state = chunk_gated_delta_rule_seq(
        q_bh,
        k_bh,
        v_bh,
        beta_bh,
        g_bh,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=s0,
        mesh_device=device,
        cached_masks=cached_masks,
        valid_len=valid_len,
    )

    # o [BH,T,V] -> [B,T,H,V]
    o = ttnn.to_layout(o_bh, ttnn.ROW_MAJOR_LAYOUT, memory_config=_DRAM)
    o = ttnn.reshape(o, [B, H, T, V])
    o = ttnn.permute(o, (0, 2, 1, 3))  # [B,T,H,V]
    o = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=_DRAM)

    # final_state [BH,K,V] -> [B,H,K,V] bf16 (matches recurrent_state)
    new_state = ttnn.reshape(final_state, [B, H, K, V])
    if new_state.dtype != ttnn.bfloat16:
        new_state = ttnn.typecast(new_state, ttnn.bfloat16, memory_config=_DRAM)

    return o, new_state


def create_chunk_masks_seq(chunk_size, device):
    """Pre-create the float32 masks the seq chunk kernel reads from `cached_masks`.

    Keys: triu_ones, tril_mask, eye ([1,C,C]), lower_causal, eye_32 ([1,32,32]).

    IMPORTANT: built via from_torch (NOT ttnn.tril/triu). On this ttnn build,
    ttnn.tril/ttnn.triu produce INCORRECT results at size 128 (wrong diagonal +
    spurious off-diagonal entries), which corrupts the chunk computation
    (D_diag gets zeros -> reciprocal -> inf). from_torch guarantees exact masks.
    Pre-allocate once at model init (constant across layers) for trace safety.
    """

    # Replicate across a multi-device mesh (TP). Single device leaves the mapper unset
    # (the validated single-device behavior). Mirrors the kernel's uncached fallback, which
    # builds these same masks with ReplicateTensorToMesh on a mesh, so the cached masks match.
    _mesh_mapper = ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None

    def _from(m):
        return ttnn.from_torch(
            m.to(torch.float32).reshape(1, m.shape[-2], m.shape[-1]),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=_mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    ones = torch.ones(chunk_size, chunk_size)
    return {
        "triu_ones": _from(torch.triu(ones, diagonal=0)),
        "tril_mask": _from(torch.tril(ones, diagonal=0)),
        "lower_causal": _from(torch.tril(ones, diagonal=0)),
        "eye": _from(torch.eye(chunk_size)),
        "eye_32": _from(torch.eye(32)),
    }


def _solve_lower_triangular_ttnn(L, eye_1cc, mesh_device):
    """Compute L^{-1} for a batch of lower triangular matrices using Neumann doubling.

    Decomposes L = D (I + N) where D = diag(L), N = D^{-1}(L - D) strictly lower triangular.
    Since N is nilpotent (N^C = 0), the Neumann series is exact:
      (I + N)^{-1} = sum_{k=0}^{C-1} (-N)^k
    Computed in ceil(log2(C)) doubling steps, then refined with 2 Newton-Schulz steps.

    Args:
        L: [batch, C, C] float32 lower triangular, positive diagonal
        eye_1cc: [1, C, C] float32 identity (pre-allocated, broadcast to batch)
    Returns:
        L_inv: [batch, C, C] float32
    """
    _hifi_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    mc = ttnn.L1_MEMORY_CONFIG

    C = L.shape[1]
    batch = L.shape[0]

    D_mat = ttnn.multiply(L, eye_1cc, memory_config=mc)  # [batch, C, C]
    D_diag = ttnn.sum(D_mat, dim=-1, memory_config=mc)  # [batch, C]
    D_inv = ttnn.reciprocal(D_diag, memory_config=mc)  # [batch, C] all in (0, 1]
    ttnn.deallocate(D_diag)
    # Clone after reshape: ttnn.reshape returns a view sharing D_inv's buffer.
    D_inv_row = ttnn.clone(ttnn.reshape(D_inv, [batch, C, 1], memory_config=mc), memory_config=mc)
    D_inv_col = ttnn.clone(ttnn.reshape(D_inv, [batch, 1, C], memory_config=mc), memory_config=mc)
    ttnn.deallocate(D_inv)

    # N = D^{-1} (L - D) via row scaling
    L_strict = ttnn.subtract(L, D_mat, memory_config=mc)
    ttnn.deallocate(D_mat)
    N = ttnn.multiply(D_inv_row, L_strict, memory_config=mc)
    ttnn.deallocate(L_strict)

    # Neumann doubling: f(2n) = f(n) @ (I + P), P = (-N)^n
    P = ttnn.neg(N, memory_config=mc)  # P = -N
    ttnn.deallocate(N)
    R = ttnn.add(eye_1cc, P, memory_config=mc)  # R = I - N = f(2)
    P_new = ttnn.matmul(P, P, memory_config=mc, compute_kernel_config=_hifi_cfg)  # P = N^2
    ttnn.deallocate(P)
    P = P_new

    n_steps = _math.ceil(_math.log2(C)) if C > 1 else 0
    for _ in range(n_steps - 1):
        I_plus_P = ttnn.add(eye_1cc, P, memory_config=mc)
        R_new = ttnn.matmul(R, I_plus_P, memory_config=mc, compute_kernel_config=_hifi_cfg)
        ttnn.deallocate(I_plus_P)
        ttnn.deallocate(R)
        R = R_new
        P_new = ttnn.matmul(P, P, memory_config=mc, compute_kernel_config=_hifi_cfg)
        ttnn.deallocate(P)
        P = P_new

    ttnn.deallocate(P)

    # L_inv = (I+N)^{-1} @ D^{-1} via column scaling
    L_inv = ttnn.multiply(R, D_inv_col, memory_config=mc)
    ttnn.deallocate(R)
    ttnn.deallocate(D_inv_row)
    ttnn.deallocate(D_inv_col)

    # Newton-Schulz refinement: X <- X(2I - LX). One step squares the residual.
    for _ in range(2):
        LX = ttnn.matmul(L, L_inv, memory_config=mc, compute_kernel_config=_hifi_cfg)
        two_I_minus_LX = ttnn.subtract(ttnn.add(eye_1cc, eye_1cc, memory_config=mc), LX, memory_config=mc)
        ttnn.deallocate(LX)
        L_inv_new = ttnn.matmul(L_inv, two_I_minus_LX, memory_config=mc, compute_kernel_config=_hifi_cfg)
        ttnn.deallocate(two_I_minus_LX)
        ttnn.deallocate(L_inv)
        L_inv = L_inv_new

    return L_inv


def _compute_L_inv_ttnn(L_mat_4d, BH, NC, C, mesh_device, _cmc=None, eye_32=None):
    """Compute diagonal block inverses of L_mat using _solve_lower_triangular_ttnn.

    L_mat_4d: [BH, NC, C, C] float32 unit-diagonal lower-triangular
    eye_32:   [1, 32, 32] float32 identity (pre-allocated; required for trace compat)
    Returns:  [BH, NC, C, 32] float32 — Ct=C/32 diagonal block inverses stacked as [C, 32]
    """
    if eye_32 is None:
        # Fallback for tests that don't pass cached_masks — not trace-compatible.
        eye_32 = ttnn.from_torch(
            torch.eye(32, dtype=torch.float32).unsqueeze(0),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    Ct = C // 32
    batch = BH * NC
    L_flat = ttnn.reshape(L_mat_4d, [batch, C, C], memory_config=_cmc)

    inv_blocks = []
    for b in range(Ct):
        row_start = b * 32
        col_start = b * 32
        block = ttnn.slice(
            L_flat, [0, row_start, col_start], [batch, row_start + 32, col_start + 32], memory_config=_cmc
        )
        block_inv = _solve_lower_triangular_ttnn(block, eye_32, mesh_device)
        ttnn.deallocate(block)
        inv_blocks.append(block_inv)  # [batch, 32, 32]

    # Do NOT deallocate L_flat: it is a reshape (view) of L_mat_4d (== L_unit_4d kernel input).
    L_inv_flat = ttnn.concat(inv_blocks, dim=1, memory_config=_cmc)
    for blk in inv_blocks:
        ttnn.deallocate(blk)

    # Do NOT deallocate L_inv_flat — ttnn.reshape returns a view sharing the buffer.
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
    valid_len=None,
):
    """Chunked gated delta rule using the C++ sequential scan kernel (Path A).

    Returns (output [BH, T, V], final_state [BH, K, V]), both float32.

    valid_len: when set (< T), positions [valid_len, T) are treated as right-padding
    and zeroed in q/k/v/beta/g BEFORE the scan — exactly mirroring the function's own
    internal zero-pad (pad_len). Those positions then produce identity state updates
    (beta=0 -> no write, g=0 -> exp(0)=1 -> no decay), so final_state reflects only the
    first valid_len tokens. This lets a fixed bucket length T serve any real length
    valid_len<=T (one compiled program per bucket) without corrupting the recurrent state.
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

    # Right-padding mask: zero every state-affecting input past valid_len. The mask
    # SHAPE is fixed by the bucket length T (only its values depend on valid_len), so a
    # single program serves all real lengths. Mirrors the zeros concatenated below for
    # pad_len; here it covers the [valid_len, T) region the caller padded.
    if valid_len is not None and valid_len < T:
        _m = torch.zeros(BH, T, 1, dtype=torch.float32)
        _m[:, :valid_len, :] = 1.0
        _m = ttnn.from_torch(_m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        q = ttnn.multiply(q, _m, memory_config=None)
        k = ttnn.multiply(k, _m, memory_config=None)
        v = ttnn.multiply(v, _m, memory_config=None)
        beta = ttnn.multiply(beta, _m, memory_config=None)
        g = ttnn.reshape(g, [BH, T, 1], memory_config=None)
        g = ttnn.multiply(g, _m, memory_config=None)
        g = ttnn.reshape(g, [BH, T], memory_config=None)
        ttnn.deallocate(_m)

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

    # ---- Decay preprocessing ----
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

    # ---- kk = k_beta @ k.T ----
    del k
    k_c = ttnn.move(k_c)
    k_c_t = ttnn.transpose(k_c, 1, 2, memory_config=_cmc)
    kk = ttnn.matmul(k_beta_c, k_c_t, memory_config=_cmc, compute_kernel_config=_hifi_cfg)
    ttnn.deallocate(k_c_t)

    _ck("kk", kk)
    # ---- L_mat = I + kk * L_mask ----
    L_mat = ttnn.add(_eye_1cc, ttnn.multiply(kk, L_mask, memory_config=_cmc), memory_config=_cmc)
    ttnn.deallocate(kk)
    _ck("L_mat", L_mat)

    # ---- Normalize to unit-diagonal: L_unit = D^{-1} L_mat ----
    D_mat = ttnn.multiply(L_mat, _eye_1cc, memory_config=_cmc)
    D_diag = ttnn.sum(D_mat, dim=-1, memory_config=_cmc)
    _ck("D_diag", D_diag)
    D_inv = ttnn.reciprocal(D_diag, memory_config=_cmc)
    _ck("D_inv", D_inv)
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

    # ---- intra_attn = (q_decay @ k.T) * L_mask * lower_causal ----
    decay_3d = ttnn.reshape(decay, [BH, num_chunks, chunk_size], memory_config=None)
    decay_raw_3d = ttnn.reshape(decay_raw, [BH, num_chunks, chunk_size], memory_config=None)

    decay_last_raw = ttnn.reshape(ttnn.sum(g_c, dim=-1, memory_config=None), [BH, num_chunks, 1], memory_config=None)
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

    # ---- Reshape preprocessing outputs to 4D for the C++ kernel ----
    def _to4d_f32(t, d1, d2):
        t4 = ttnn.reshape(t, [BH, num_chunks, d1, d2], memory_config=None)
        return ttnn.to_layout(t4, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    L_unit_4d = _to4d_f32(L_unit, chunk_size, chunk_size)
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

    _ck("L_unit", L_unit_4d)
    _ck("v_beta_sc", v_beta_sc_4d)
    _ck("k_bd_sc", k_bd_sc_4d)
    _ck("intra_attn", intra_attn_4d)
    _ck("q_decay", q_decay_4d)
    _ck("k_decay_t", k_decay_t_4d)
    _ck("dl_exp", dl_exp_4d)

    # Diagonal block inverses of L_unit (Neumann + Newton-Schulz).
    L_inv_4d = _compute_L_inv_ttnn(L_unit_4d, BH, num_chunks, chunk_size, mesh_device, _cmc, eye_32=_eye_32)
    _ck("L_inv", L_inv_4d)

    # Initial state
    S0_tt = None
    if initial_state is not None:
        S0_tt = ttnn.typecast(
            ttnn.reshape(initial_state, [BH, K, V], memory_config=None),
            ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ---- C++ sequential scan kernel (Path A) ----
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
