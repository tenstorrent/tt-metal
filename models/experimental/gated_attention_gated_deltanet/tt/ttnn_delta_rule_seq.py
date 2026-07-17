# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Chunk-parallel gated delta rule via C++ `ttnn.transformer.gated_delta_attn_seq` (Path A).

Python (float32): elementwise ops + kk/intra_attn matmuls; L_inv from 4 diagonal blocks
via `_solve_lower_triangular_ttnn` (default: Horner Neumann; legacy doubling behind
QWEN_GDN_INV_DOUBLING=1). C++ kernel: blocked forward substitution + inter-chunk scan.

Layout (FLA): q,k [BH,T,K]; beta [BH,T,1]; g [BH,T]; v [BH,T,V]; state [BH,K,V].
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


# Mask helpers shared with bf16 chunk path; import works via fq path or top-level `tt`.
from .ttnn_delta_rule_ops import (
    _create_triu_ones_ttnn as _create_triu_ones,
    _create_tril_ones_ttnn as _create_tril_ones,
    l2_norm_ttnn,
)

_TILE = 32

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def _bmm_progcfg(device, mt, nt, kt):
    """Batched-matmul program config: one full [mt,nt] output block per core, with the batch
    spread across the whole device grid. TTNN's auto-config pins the recurrence's [128,128]@[128,128]
    per-chunk bmms (kk, qk) to ~16 cores (one 4x4-tile output tiled across cores, batch looped
    serially); here num_output_blocks = batch so the ~192 chunk-batch elements fan out over ~130
    cores instead. Pure parallelization — matmul math is config-independent.

    Returns None (-> TTNN auto-config, prior behaviour) when device is None or the shapes don't
    tile cleanly.
    """
    if device is None or mt < 1 or nt < 1 or kt < 1:
        return None
    try:
        grid = device.compute_with_storage_grid_size()
        per_core_M, per_core_N = mt, nt
        # out_subblock: largest h*w <= 4 tiles (fp32 DST limit) dividing per_core_M/N
        osb_h, osb_w, best = 1, 1, 0
        for h in range(1, per_core_M + 1):
            if per_core_M % h:
                continue
            for w in range(1, per_core_N + 1):
                if per_core_N % w:
                    continue
                if h * w <= 4 and h * w > best:
                    best, osb_h, osb_w = h * w, h, w
        # in0_block_w: largest divisor of kt, capped at 4 for L1 safety in fp32
        in0_bw = 1
        for c in (4, 3, 2, 1):
            if c <= kt and kt % c == 0:
                in0_bw = c
                break
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=in0_bw,
            out_subblock_h=osb_h,
            out_subblock_w=osb_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
        )
    except Exception:
        return None


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
    qkv_head_dims=None,  # (Hq,K,H,V): flat q/k/v [B,T,Hq*K]/[B,T,H*V]
    return_o_bh=False,  # True: return o as [BH,T,V], skip token-major relayout
):
    """Drop-in for chunk_gated_delta_rule_ttnn using `gated_delta_attn_seq`.

    [B,T,H,*] in -> (o [B,T,H,V], new_state [B,H,K,V]). L2-norms q/k, converts to
    [BH,T,*] float32, runs kernel, converts back. new_state is bf16 (decode dtype) unless
    QWEN_GDN_FP32_STATE=1.
    """
    B = q.shape[0]
    T = q.shape[1]
    if qkv_head_dims is not None:
        # FLAT q/k [B,T,Hq*K], v [B,T,H*V]: head-split in _to_bhtd; L2 norm deferred (bit-identical).
        Hq, K, H, V = qkv_head_dims
        _defer_l2 = True
    else:
        H = v.shape[2]  # Nv: beta/g/v/output/state
        Hq = q.shape[2]  # q/k heads; may be < H (GQA) — norm at Hq, expand to H after via repeat_interleave
        K = q.shape[3]
        V = v.shape[3]
        _defer_l2 = False
    BH = B * H

    # L2-norm q/k (kernel doesn't); deferred for flat inputs via _defer_l2.
    if not _defer_l2:
        q = l2_norm_ttnn(q, dim=-1)
        k = l2_norm_ttnn(k, dim=-1)

    # Relayout untilize/permute in L1; land kernel inputs in DRAM (CBs ~1.36MB/core clash with L1).
    _L1 = ttnn.L1_MEMORY_CONFIG

    def _tilize_f32(t):
        # Tilize + fp32 cast in one op (main #49565), avoiding a separate typecast pass.
        return ttnn.to_layout(t, ttnn.TILE_LAYOUT, dtype=ttnn.float32, memory_config=_DRAM)

    def _to_bhtd(t, D, Hh, l2=False):  # [B,T,Hh,D] (or flat [B,T,Hh*D]) -> [B*Hh,T,D] float32 TILE
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=_L1)  # untilize -> L1
        t = ttnn.reshape(t, [B, T, Hh, D])  # head-split (no-op if already 4D)
        t = ttnn.permute(t, (0, 2, 1, 3))
        t = ttnn.reshape(t, [B * Hh, T, D])
        if l2:
            # Per-head L2 on [BH,T,D] bf16 (before the fp32 cast) — bit-identical to pre-split norm,
            # so keep tilize/typecast separate here (fused f32 tilize would norm in fp32 instead).
            t = ttnn.to_layout(t, ttnn.TILE_LAYOUT, memory_config=_DRAM)  # kernel input in DRAM
            t = l2_norm_ttnn(t, dim=-1)
            if t.dtype != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32, memory_config=_DRAM)
            return t
        return _tilize_f32(t)  # non-normed (v): fused tilize+cast

    def _to_bht(t):  # [B,T,H] -> [BH,T] float32 TILE
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=_L1)  # untilize -> L1
        t = ttnn.reshape(t, [B, T, H])
        t = ttnn.permute(t, (0, 2, 1))
        t = ttnn.reshape(t, [BH, T])
        return _tilize_f32(t)  # no norm on g/beta -> fused tilize+cast is safe

    q_bh = _to_bhtd(q, K, Hq, l2=_defer_l2)
    k_bh = _to_bhtd(k, K, Hq, l2=_defer_l2)
    v_bh = _to_bhtd(v, V, H)
    if Hq != H:
        # GQA: repeat_interleave q/k along BH dim (Hq -> H heads); norm+permute ran on Hq only.
        rf = H // Hq
        q_bh = ttnn.repeat_interleave(q_bh, rf, dim=0)
        k_bh = ttnn.repeat_interleave(k_bh, rf, dim=0)
    g_bh = _to_bht(g)
    beta_bh = ttnn.reshape(_to_bht(beta), [BH, T, 1])

    # initial_state [B,H,K,V] -> [BH,K,V]
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

    # o [BH,T,V] -> [B,T,H,V] (L1 shuffle, DRAM output). return_o_bh skips for caller-side fusion.
    if return_o_bh:
        o = o_bh
    else:
        o = ttnn.to_layout(o_bh, ttnn.ROW_MAJOR_LAYOUT, memory_config=_L1)
        o = ttnn.reshape(o, [B, H, T, V])
        o = ttnn.permute(o, (0, 2, 1, 3))  # [B,T,H,V]
        o = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=_DRAM)

    # final_state [BH,K,V] -> [B,H,K,V]. Default bf16; QWEN_GDN_FP32_STATE=1 avoids ~128x requant in long prefill.
    new_state = ttnn.reshape(final_state, [B, H, K, V])
    _state_dtype = ttnn.float32 if _os.environ.get("QWEN_GDN_FP32_STATE", "0") != "0" else ttnn.bfloat16
    if new_state.dtype != _state_dtype:
        new_state = ttnn.typecast(new_state, _state_dtype, memory_config=_DRAM)

    return o, new_state


def create_chunk_masks_seq(chunk_size, device):
    """Pre-create float32 masks for seq kernel `cached_masks`.

    Keys: triu_ones, tril_mask, eye, lower_causal, eye_32.
    Use from_torch (not ttnn.tril/triu): at C=128 those ops produce wrong diagonals
    (zeros on D_diag -> inf). Pre-allocate at init for trace safety.
    """

    # Replicate on TP mesh; single device leaves mapper unset (matches kernel fallback).
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
    """Batch L^{-1} for lower triangular L = D(I+N), N strictly lower, N^C=0.

    DEFAULT: Horner Neumann R = I + (-N)R (forward-substitution; stable for large ||N||).
    LEGACY (QWEN_GDN_INV_DOUBLING=1): Neumann doubling + Newton-Schulz — stable only with
    caller's damped L_mat (small ||N||); unstable on undamped form (||N||~19 -> fp32 overflow).

    Args: L [batch,C,C], eye_1cc [1,C,C]. Returns L_inv [batch,C,C].
    """
    _hifi_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    mc = ttnn.L1_MEMORY_CONFIG

    C = L.shape[1]
    batch = L.shape[0]

    D_mat = ttnn.multiply(L, eye_1cc, memory_config=mc)  # [batch, C, C]
    # keepdim -> D_inv_row is [batch, C, 1] straight from the reduce (no reshape+clone; the
    # [batch,C]->[batch,C,1] reshape is a TILE relayout). Col form still needs one reshape.
    D_diag = ttnn.sum(D_mat, dim=-1, keepdim=True, memory_config=mc)  # [batch, C, 1]
    D_inv_row = ttnn.reciprocal(D_diag, memory_config=mc)  # [batch, C, 1] all in (0, 1]
    ttnn.deallocate(D_diag)
    D_inv_col = ttnn.clone(ttnn.reshape(D_inv_row, [batch, 1, C], memory_config=mc), memory_config=mc)

    # N = D^{-1} (L - D) via row scaling
    L_strict = ttnn.subtract(L, D_mat, memory_config=mc)
    ttnn.deallocate(D_mat)
    N = ttnn.multiply(D_inv_row, L_strict, memory_config=mc)
    ttnn.deallocate(L_strict)

    # Block inverse (I+N)^{-1}. DEFAULT=Horner; legacy doubling behind QWEN_GDN_INV_DOUBLING=1.
    # Doubling overflows fp32 when ||N||~19 (N^16 ~1e9); Horner matches FLA forward-substitution.
    if _os.environ.get("QWEN_GDN_INV_DOUBLING", "0") != "0":
        # LEGACY: doubling + Newton-Schulz; paired with damped L_mat under same flag.
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

        # Newton-Schulz: X <- X(2I - LX)
        for _ in range(2):
            LX = ttnn.matmul(L, L_inv, memory_config=mc, compute_kernel_config=_hifi_cfg)
            two_I_minus_LX = ttnn.subtract(ttnn.add(eye_1cc, eye_1cc, memory_config=mc), LX, memory_config=mc)
            ttnn.deallocate(LX)
            L_inv_new = ttnn.matmul(L_inv, two_I_minus_LX, memory_config=mc, compute_kernel_config=_hifi_cfg)
            ttnn.deallocate(two_I_minus_LX)
            ttnn.deallocate(L_inv)
            L_inv = L_inv_new

        return L_inv

    # DEFAULT: Horner R = I + (-N)@R
    neg_N = ttnn.neg(N, memory_config=mc)  # -N (strictly lower)
    ttnn.deallocate(N)
    # Pre-broadcast eye to [batch,C,C] when batch>1 (~-4% prefill; batch==1 unchanged).
    _eye = eye_1cc
    if batch > 1:
        _eye = ttnn.repeat(eye_1cc, ttnn.Shape([batch, 1, 1]), memory_config=mc)
    R = ttnn.add(_eye, neg_N, memory_config=mc)  # R_1 = I - N  ([batch,C,C])
    for _ in range(C - 2):  # R_1 -> R_{C-1} = sum (-N)^j (exact: N^C=0)
        NR = ttnn.matmul(neg_N, R, memory_config=mc, compute_kernel_config=_hifi_cfg)  # (-N) @ R
        R_new = ttnn.add(_eye, NR, memory_config=mc)  # I + (-N) @ R
        ttnn.deallocate(NR)
        ttnn.deallocate(R)
        R = R_new
    ttnn.deallocate(neg_N)
    if _eye is not eye_1cc:
        ttnn.deallocate(_eye)

    # L_inv = (I+N)^{-1} @ D^{-1} via column scaling
    L_inv = ttnn.multiply(R, D_inv_col, memory_config=mc)
    ttnn.deallocate(R)
    ttnn.deallocate(D_inv_row)
    ttnn.deallocate(D_inv_col)
    return L_inv


def _compute_L_inv_ttnn(L_mat_4d, BH, NC, C, mesh_device, _cmc=None, eye_32=None):
    """Diagonal block inverses of L_mat via _solve_lower_triangular_ttnn.

    L_mat_4d [BH,NC,C,C]; eye_32 [1,32,32]. Returns [BH,NC,C,32] (Ct=C/32 blocks as [C,32]).
    """
    if eye_32 is None:
        # Test fallback without cached_masks — not trace-compatible.
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

    diagonal_blocks = []
    for b in range(Ct):
        row_start = b * 32
        col_start = b * 32
        block = ttnn.slice(
            L_flat, [0, row_start, col_start], [batch, row_start + 32, col_start + 32], memory_config=_cmc
        )
        diagonal_blocks.append(block)  # [batch, 32, 32]

    # The diagonal 32x32 blocks are independent. Batch them together so the Horner
    # inverse runs once over [Ct*batch, 32, 32] instead of Ct separate 30-step solves.
    stacked_blocks = ttnn.concat(diagonal_blocks, dim=0, memory_config=_cmc)
    for block in diagonal_blocks:
        ttnn.deallocate(block)

    stacked_inv = _solve_lower_triangular_ttnn(stacked_blocks, eye_32, mesh_device)
    ttnn.deallocate(stacked_blocks)

    inv_blocks = []
    for b in range(Ct):
        block_start = b * batch
        block_inv = ttnn.slice(stacked_inv, [block_start, 0, 0], [block_start + batch, 32, 32], memory_config=_cmc)
        inv_blocks.append(block_inv)  # [batch, 32, 32]

    # L_flat is a view of L_mat_4d — do not deallocate.
    L_inv_flat = ttnn.concat(inv_blocks, dim=1, memory_config=_cmc)
    for blk in inv_blocks:
        ttnn.deallocate(blk)
    ttnn.deallocate(stacked_inv)

    # L_inv_flat is a view — do not deallocate.
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
    """Chunked gated delta rule via C++ sequential scan (Path A).

    Returns (output [BH,T,V], final_state [BH,K,V]) float32.
    valid_len: zero q/k/v/beta/g past valid_len (padding); identity state updates preserve recurrent state.
    """
    # Preprocessing matmuls: HiFi4 (matches block-inverse fidelity).
    _hifi_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
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

    # Batched-bmm progcfg for the per-chunk [C,C]@[C,C] matmuls (kk, qk): fan the chunk-batch
    # across the full device grid instead of the ~16-core auto-config. Same math (see _bmm_progcfg).
    _bmm_cfg = _bmm_progcfg(mesh_device, chunk_size // _TILE, chunk_size // _TILE, K // _TILE)

    # Zero inputs past valid_len (fixed T bucket, variable real length). Per-row valid_len: BH rows b*H..(b+1)*H.
    _is_per_row = isinstance(valid_len, (list, tuple))
    if _is_per_row or (valid_len is not None and valid_len < T):
        _m = torch.zeros(BH, T, 1, dtype=torch.float32)
        if _is_per_row:
            _Bv = len(valid_len)
            _H = BH // _Bv
            for _b in range(_Bv):
                _m[_b * _H : (_b + 1) * _H, : int(valid_len[_b]), :] = 1.0
        else:
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
    # decay = g_c @ triu_ones (prefix-sum of g along the chunk). triu_ones is broadcast across
    # the batch, so the per-batch [1,C]@[C,C] bmm (M=1 -> pinned to ~4 cores) is bit-identical to
    # a single 2D [batch,C]@[C,C] matmul, which spreads the batch rows across ~24 cores. Same
    # math, same HiFi4/fp32 fidelity — pure parallelization win.
    triu_ones_2d = ttnn.reshape(triu_ones, [chunk_size, chunk_size], memory_config=None)
    decay = ttnn.matmul(g_c, triu_ones_2d, memory_config=None, compute_kernel_config=_hifi_cfg)
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
    kk = ttnn.matmul(k_beta_c, k_c_t, memory_config=_cmc, compute_kernel_config=_hifi_cfg, program_config=_bmm_cfg)
    ttnn.deallocate(k_c_t)

    _ck("kk", kk)
    # L_mat diagonal regularization (QWEN_GDN_DIAG_ALPHA): diag = 1 + alpha*diag(kk*L_mask).
    # alpha=0: exact HF/FLA (unit diag, undamped ||N||~19). alpha=1: full 1/(1+beta) damping.
    # Default 0.25: partial damping prevents GDN state saturation at 256k (alpha=0 rides doc narrative).
    # Horner inverse stable at any alpha; QWEN_GDN_INV_DOUBLING=1 forces alpha=1 + doubling (A/B pair).
    if _os.environ.get("QWEN_GDN_INV_DOUBLING", "0") != "0":
        # A/B: full diagonal-included L_mat + doubling inverse.
        L_mat = ttnn.add(_eye_1cc, ttnn.multiply(kk, L_mask, memory_config=_cmc), memory_config=_cmc)
        ttnn.deallocate(kk)
    else:
        # L_mat = I + kk*L_mask - (1-alpha)*diag(kk*L_mask)
        alpha = float(_os.environ.get("QWEN_GDN_DIAG_ALPHA", "0.25"))
        kk_lmask = ttnn.multiply(kk, L_mask, memory_config=_cmc)
        ttnn.deallocate(kk)
        kk_diag = ttnn.multiply(kk_lmask, _eye_1cc, memory_config=_cmc)  # diag(kk*L_mask)
        if alpha == 0.0:
            # alpha=0: strip diagonal -> unit diag (torch/FLA-equivalent)
            kk_reg = ttnn.subtract(kk_lmask, kk_diag, memory_config=_cmc)
        else:
            # keep alpha*diag; drop (1-alpha)*diag
            kk_drop = ttnn.multiply(kk_diag, 1.0 - alpha, memory_config=_cmc)
            kk_reg = ttnn.subtract(kk_lmask, kk_drop, memory_config=_cmc)
            ttnn.deallocate(kk_drop)
        ttnn.deallocate(kk_lmask)
        ttnn.deallocate(kk_diag)
        L_mat = ttnn.add(_eye_1cc, kk_reg, memory_config=_cmc)
        ttnn.deallocate(kk_reg)
    _ck("L_mat", L_mat)

    # ---- Normalize to unit-diagonal: L_unit = D^{-1} L_mat ----
    D_mat = ttnn.multiply(L_mat, _eye_1cc, memory_config=_cmc)
    # keepdim -> reduce writes [batch, C, 1] directly; skips the [batch,C]->[batch,C,1] reshape,
    # which on TILE is a physical relayout (~60us). Bit-identical to sum+reshape.
    D_diag = ttnn.sum(D_mat, dim=-1, keepdim=True, memory_config=_cmc)
    _ck("D_diag", D_diag)
    D_inv_row = ttnn.reciprocal(D_diag, memory_config=_cmc)  # [batch, C, 1] row-broadcast scale
    _ck("D_inv", D_inv_row)
    ttnn.deallocate(D_diag)

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

    decay_last_raw = ttnn.reshape(ttnn.sum(g_c, dim=-1, memory_config=None), [BH, num_chunks, 1], memory_config=None)
    decay_last_normalized = ttnn.reshape(decay_3d[:, :, -1:], [BH, num_chunks, 1], memory_config=None)

    # decay_raw_exp_4d == exp(clip(decay_raw)) again, just rank-4: identical values to decay_exp
    # ([batch,C,1]). Reuse via a cheap leading-dim split instead of recomputing exp+clip and
    # relaying out decay_raw_3d. Bit-identical.
    decay_raw_exp_4d = ttnn.reshape(decay_exp, [BH, num_chunks, chunk_size, 1], memory_config=_cmc)
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
    qk_4d = ttnn.matmul(q_c_4d, k_c_4d_t, memory_config=_cmc, compute_kernel_config=_hifi_cfg, program_config=_bmm_cfg)
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

    # L_inv via Horner solve (default); legacy doubling behind QWEN_GDN_INV_DOUBLING.
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
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(L_inv_4d)

    _out_l1 = ttnn.L1_MEMORY_CONFIG
    out_4d = ttnn.to_layout(
        ttnn.typecast(out_4d, ttnn.float32, memory_config=_out_l1) if out_4d.dtype != ttnn.float32 else out_4d,
        ttnn.TILE_LAYOUT,
        memory_config=_out_l1,
    )
    o = ttnn.reshape(out_4d, [BH, L, V], memory_config=_out_l1)

    if pad_len > 0:
        o = o[:, :T, :]
        o = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=_out_l1)

    return o, final_state
