# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Chunk-parallel gated delta rule using the C++ `ttnn.transformer.gated_delta_attn_seq`
kernel (Path A) — imported from the Qwen3.5-27B branch (gdn_chunk_ops_seq.py /
gdn_chunk_ops.py) and adapted for the single-device Qwen3.5-9B model.

Python preprocessing computes (all float32):
  - cheap elementwise ops + two matmuls (kk, intra_attn),
  - L_inv: 4 diagonal block inverses of L_unit via `_solve_lower_triangular_ttnn`
    (default: stable D^{-1} Horner-form Neumann series; legacy Neumann doubling +
    Newton-Schulz behind QWEN_GDN_INV_DOUBLING=1 — see that function for why).
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
    H = v.shape[2]  # value-head count (Nv): beta/g/v/output/state all use this
    Hq = q.shape[2]  # q/k head count — may be < H (GQA). When < H, q/k are L2-normed + transformed
    # at Hq heads (3x less work for a 4:1 ratio) and expanded to H AFTER, via a cheap dim-0 block
    # repeat (no untilize). No-op when Hq == H (the pre-expanded path / 9B) → fully backward-compatible.
    K = q.shape[3]
    V = v.shape[3]
    BH = B * H

    # L2-norm q/k (the seq kernel does NOT normalize; it only scales q internally).
    q = l2_norm_ttnn(q, dim=-1)
    k = l2_norm_ttnn(k, dim=-1)

    # OPT (prefill): run the seq-major<->head-major relayout DATA MOVEMENT (untilize->permute) in L1
    # so the actual head-shuffle is L1<->L1 and the untilize/tilize each get one side in L1. But LAND
    # the final kernel-input tensor back in DRAM: gated_delta_attn_seq allocates ~1.36MB/core of static
    # circular buffers, and all 5 relayout outputs (q/k/v/g/beta) are alive as its inputs — keeping them
    # in L1 clashes with those CBs (OOM). The transient ROW_MAJOR intermediate is only alive DURING the
    # relayout (no kernel running then), so L1 there is safe. Intermediate=_L1, kernel-input=_DRAM.
    _L1 = ttnn.L1_MEMORY_CONFIG

    def _to_bhtd(t, D, Hh):  # [B,T,Hh,D] -> [B*Hh,T,D] float32 TILE (ROW_MAJOR-correct)
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=_L1)  # untilize -> L1
        t = ttnn.reshape(t, [B, T, Hh, D])
        t = ttnn.permute(t, (0, 2, 1, 3))  # [B,Hh,T,D] shuffle in L1
        t = ttnn.reshape(t, [B * Hh, T, D])
        t = ttnn.to_layout(t, ttnn.TILE_LAYOUT, memory_config=_DRAM)  # land kernel input in DRAM (CB room)
        if t.dtype != ttnn.float32:
            t = ttnn.typecast(t, ttnn.float32, memory_config=_DRAM)
        return t

    def _to_bht(t):  # [B,T,H] -> [BH,T] float32 TILE
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=_L1)  # untilize -> L1
        t = ttnn.reshape(t, [B, T, H])
        t = ttnn.permute(t, (0, 2, 1))  # [B,H,T] shuffle in L1
        t = ttnn.reshape(t, [BH, T])
        t = ttnn.to_layout(t, ttnn.TILE_LAYOUT, memory_config=_DRAM)  # land kernel input in DRAM (CB room)
        if t.dtype != ttnn.float32:
            t = ttnn.typecast(t, ttnn.float32, memory_config=_DRAM)
        return t

    q_bh = _to_bhtd(q, K, Hq)
    k_bh = _to_bhtd(k, K, Hq)
    v_bh = _to_bhtd(v, V, H)
    if Hq != H:
        # GQA late expand: replicate each q/k head rf times along the BH (outer, non-tile) axis —
        # a cheap tile-block copy. q_bh rows are b*Hq+h, so dim-0 repeat_interleave maps output row
        # b*H + (h*rf+j) <- key-head h, i.e. value-head (h*rf+j) uses key-head h. Identical to a
        # token-major pre-expand, but the L2-norm + permute above ran on Hq (not H) heads.
        rf = H // Hq
        q_bh = ttnn.repeat_interleave(q_bh, rf, dim=0)
        k_bh = ttnn.repeat_interleave(k_bh, rf, dim=0)
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

    # o [BH,T,V] -> [B,T,H,V]  (shuffle in L1, land in DRAM — feeds the out-proj matmul's CBs downstream)
    o = ttnn.to_layout(o_bh, ttnn.ROW_MAJOR_LAYOUT, memory_config=_L1)
    o = ttnn.reshape(o, [B, H, T, V])
    o = ttnn.permute(o, (0, 2, 1, 3))  # [B,T,H,V]
    o = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=_DRAM)

    # final_state [BH,K,V] -> [B,H,K,V]. DEFAULT casts to bf16 (the decode recurrent_state dtype).
    # QWEN_GDN_FP32_STATE=1: keep the inter-chunk state at full fp32 precision. In chunk-outer prefill
    # the kernel computes final_state in fp32 but this cast rounds it to bf16 EACH of the ~128 outer
    # 2048-tok chunks of a 256k prompt, so the carried recurrent state is requantized ~128x. The HF/FLA
    # reference instead keeps the state in fp32 across the WHOLE sequence in one chunked pass (it upcasts
    # q/k/v/beta/g to float32 and never round-trips), which is why the reference runs the exact alpha=0
    # delta rule on-task at 256k. rec_state is already an fp32 buffer by default (tp.py reset_state) and
    # decode consumes fp32, so keeping fp32 here makes the device carry match the reference end-to-end.
    new_state = ttnn.reshape(final_state, [B, H, K, V])
    _state_dtype = ttnn.float32 if _os.environ.get("QWEN_GDN_FP32_STATE", "0") != "0" else ttnn.bfloat16
    if new_state.dtype != _state_dtype:
        new_state = ttnn.typecast(new_state, _state_dtype, memory_config=_DRAM)

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
    """Compute L^{-1} for a batch of lower triangular matrices.

    Decomposes L = D (I + N) where D = diag(L), N = D^{-1}(L - D) strictly lower triangular.
    Since N is nilpotent (N^C = 0), the Neumann series is exact: (I + N)^{-1} = sum_{k=0}^{C-1} (-N)^k.

    DEFAULT: evaluate that series in stable HORNER form, R = I + (-N) R (C-1 iterations) — i.e.
    forward-substitution: each step forms only (-N)@R (intermediates stay O(||N||)), so it is accurate
    even for large ||N||. This is what the torch/FLA reference effectively does.

    LEGACY (QWEN_GDN_INV_DOUBLING=1): Neumann DOUBLING (square N->N^2->...->N^16 in ceil(log2 C) steps)
    + 2 Newton-Schulz. Faster (fewer matmuls). This is HALF of the original path: the caller pairs it with
    the original diagonal-INCLUDED L_mat (damped, D=1+beta), which keeps ||N|| small enough (the 1/(1+beta)
    damping) that the N^16 intermediate stays within fp32 -> stable, reproducing the ORIGINAL working
    (coherent, non-torch-equivalent) behavior. Doubling is UNSTABLE only if fed the DEFAULT undamped
    strictly-lower form (||N|| ~ 19 -> N^16 ~ 1e9-1e10 overflows fp32 -> garbage inverse -> long-context
    '!!!!') — which is why both halves move together under the one flag, never mixed. A/B only.

    Args:
        L: [batch, C, C] float32 lower triangular, positive diagonal
        eye_1cc: [1, C, C] float32 identity (pre-allocated, broadcast to batch)
    Returns:
        L_inv: [batch, C, C] float32
    """
    # HiFi4 (full fp32 cross-terms) for the block-inverse matmuls — accurate and validated.
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

    # ====================================================================================
    # Block inverse (I+N)^{-1}, N strictly-lower (nilpotent: N^C = 0).
    # DEFAULT = stable Horner-form Neumann series; legacy doubling behind QWEN_GDN_INV_DOUBLING=1.
    #
    # WHY (root cause): the legacy Neumann DOUBLING (N->N^2->...->N^16) is the source of the alpha=0
    # long-context "!!!!" collapse. For real GDN blocks with large strictly-lower norm (||N|| ~ 19 at
    # e.g. layer 44), the intermediate power N^16 has entries ~1e9-1e10; summing those (alternating
    # signs) down to the O(1) inverse loses everything in fp32 -> garbage inverse (measured residual
    # ||L*Linv-I|| ~ 2-6 on 222/256 real blocks) -> garbage v_cor/k_cum -> the scan overflows fp32 ->
    # degenerate logits. It is NOT a precision tier (HiFi4 doesn't help) and NOT the math: the exact
    # reference is stable; the torch/FLA reference inverts by direct forward-substitution (no matrix
    # powers). The legacy D=1+beta / alpha-damping only "worked" by shrinking ||N|| below the overflow
    # threshold.
    #
    # FIX: Horner series  R_k = I + (-N) R_{k-1},  R_0 = I  =>  R_{C-1} = sum_{j=0}^{C-1} (-N)^j = (I+N)^{-1}
    # (exact since N^C = 0). Each step forms only (-N)@R (bounded ~O(||N||)), never the huge N^16, so it is
    # accurate even for large ||N|| (unit-test: 4.2e-3 @ ~bf16 on the ||N||=19 blocks where doubling = 4.3).
    # This is forward-substitution in matmul form, matching the reference, and lets exact alpha=0 run stably.
    # ====================================================================================
    if _os.environ.get("QWEN_GDN_INV_DOUBLING", "0") != "0":
        # ---- LEGACY: Neumann doubling + Newton-Schulz. Stable HERE because the caller also restores the
        # original damped (diagonal-included) L_mat under this same flag, so ||N|| is small. A/B only. ----
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

    # ---- DEFAULT: stable Horner-form Neumann series  R = I + (-N) @ R  (forward-substitution) ----
    neg_N = ttnn.neg(N, memory_config=mc)  # -N (strictly lower)
    ttnn.deallocate(N)
    R = ttnn.add(eye_1cc, neg_N, memory_config=mc)  # R_1 = I - N  ([batch,C,C])
    for _ in range(C - 2):  # R_1 -> R_{C-1} = sum_{j=0}^{C-1} (-N)^j  (exact: N^C = 0)
        NR = ttnn.matmul(neg_N, R, memory_config=mc, compute_kernel_config=_hifi_cfg)  # (-N) @ R
        R_new = ttnn.add(eye_1cc, NR, memory_config=mc)  # I + (-N) @ R
        ttnn.deallocate(NR)
        ttnn.deallocate(R)
        R = R_new
    ttnn.deallocate(neg_N)

    # L_inv = (I+N)^{-1} @ D^{-1} via column scaling
    L_inv = ttnn.multiply(R, D_inv_col, memory_config=mc)
    ttnn.deallocate(R)
    ttnn.deallocate(D_inv_row)
    ttnn.deallocate(D_inv_col)
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
    # kk / decay / intra_attn preprocessing matmuls (feed L_unit and the block inverse): HiFi4,
    # matching the block-inverse fidelity in _solve_lower_triangular_ttnn.
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

    # Right-padding mask: zero every state-affecting input past valid_len. The mask
    # SHAPE is fixed by the bucket length T (only its values depend on valid_len), so a
    # single program serves all real lengths. Mirrors the zeros concatenated below for
    # pad_len; here it covers the [valid_len, T) region the caller padded.
    # valid_len may be a scalar (one length for all BH rows) or a per-row list/tuple of length B
    # (batched prefill): BH rows are ordered b*H + h, so user b owns rows [b*H, (b+1)*H).
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
    # ---- L_mat: intra-chunk matrix with a tunable diagonal regularization (QWEN_GDN_DIAG_ALPHA) ----
    #
    # The intra-chunk system is (I + tril(kk*L_mask))^{-1}. How much of (kk*L_mask)'s DIAGONAL we keep
    # controls a damping that is load-bearing at long context. With alpha in [0,1]:
    #       diag(L_mat) = 1 + alpha * diag(kk*L_mask)   (diag(kk*L_mask) = beta_i*|k_i|^2 ~= beta_i)
    # and the downstream unit-diagonal normalization (L_unit = D^{-1} L_mat, v_beta_sc = D^{-1} v_beta)
    # folds D^{-1} = 1/(1+alpha*beta) onto BOTH the off-diagonals (-> ||N||) and the value term.
    #
    #   alpha = 0  -> EXACT strictly-lower unit-diagonal form == HF/FLA reference (token i reads the
    #                 recurrent state BEFORE its own write, so the diagonal is masked out). Torch-equivalent
    #                 / correct <think>+tool dialect, but UNDAMPED (||N|| ~ 19 on real blocks).
    #   alpha = 1  -> full 1/(1+beta) damping == the ORIGINAL kernel's diagonal-included form.
    #   0<alpha<1  -> partial damping (regularized).
    #
    # WHY a nonzero DEFAULT (0.25): with alpha=0 the undamped per-chunk corrections accumulate into the
    # finite-capacity GDN recurrent state and saturate it over the ~128 chunks of a full 262144-token
    # prompt, diluting the most recent tokens (the seeded <think> + instruction) so the model rides the
    # document's narrative momentum instead of reasoning. CONFIRMED on hw at ISL=256k: alpha=0 continues
    # the source novel; the damped form enters the <think> reasoning process and identifies the task. The
    # damping shrinks the corrections enough to preserve the recent-suffix signal. alpha=0.25 keeps the
    # dialect (validated argmax 'The'->'Thinking', coherent 4k..128k) while restoring long-context behavior.
    #
    # The Horner / forward-substitution block inverse (_solve_lower_triangular_ttnn, default) is stable for
    # ANY alpha, so this is a pure Python lever — no rebuild. (QWEN_GDN_INV_DOUBLING=1 is a separate A/B
    # that forces the exact original {full-damping diagonal form + Neumann-doubling inverse}; the doubling
    # inverse only stays within fp32 BECAUSE of that full damping, so it ignores alpha and uses the form
    # below's alpha=1 limit directly.)
    if _os.environ.get("QWEN_GDN_INV_DOUBLING", "0") != "0":
        # ORIGINAL exact-reproduction A/B: full diagonal-included form (alpha=1) + doubling inverse.
        L_mat = ttnn.add(_eye_1cc, ttnn.multiply(kk, L_mask, memory_config=_cmc), memory_config=_cmc)
        ttnn.deallocate(kk)
    else:
        # DEFAULT (Horner inverse): regularized diagonal  L_mat = I + kk*L_mask - (1-alpha)*diag(kk*L_mask).
        alpha = float(_os.environ.get("QWEN_GDN_DIAG_ALPHA", "0.25"))
        kk_lmask = ttnn.multiply(kk, L_mask, memory_config=_cmc)
        ttnn.deallocate(kk)
        kk_diag = ttnn.multiply(kk_lmask, _eye_1cc, memory_config=_cmc)  # diag(kk*L_mask)
        if alpha == 0.0:
            # exact: strip the whole diagonal -> unit diagonal (torch/FLA-equivalent)
            kk_reg = ttnn.subtract(kk_lmask, kk_diag, memory_config=_cmc)
        else:
            # keep alpha*diagonal: drop (1-alpha)*diag so diag(L_mat) = 1 + alpha*beta
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

    # Diagonal block inverses of L_unit via the stable Horner / forward-substitution solve
    # (default; legacy Neumann doubling behind QWEN_GDN_INV_DOUBLING — see _solve_lower_triangular_ttnn).
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
