# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wormhole variant of the shared GDN chunk-seq kernel wrapper.

WHY THIS FILE EXISTS
--------------------
``chunk_gated_delta_rule_seq`` in
``models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_seq.py`` relayouts the
kernel output as an L1-resident ``[BH, L, V]`` fp32 tensor. At L=2048 that is
32*2048*128*4 = 33,554,432 B. Blackhole absorbs it (140 cores, ~210MB L1, 80 banks); Wormhole
cannot (80 cores, ~114MB L1, 64 banks -> ~1.25x more per bank), so every long-context and
trace-capturing prefill test dies with "Out of Memory: Not enough space to allocate 33554432 B".

bf16 halves it to 16,777,216 B, which fits, and costs nothing measurable in accuracy
(logit PCC 0.9998-1.0000, identical to fp32). The dtype is a *local* inside that function, so
there is no narrow seam to override -- hence this copy, kept in this model's folder so the
shared module is not edited.

BLACKHOLE IS NEVER AFFECTED
---------------------------
``chunk_gated_delta_rule_seq_dispatch`` delegates to the original upstream function whenever
``is_blackhole()``. Blackhole therefore always executes upstream code, not this copy, so this
file cannot change Blackhole behaviour even as upstream evolves.

MAINTENANCE
-----------
The body below is a verbatim copy of upstream at the commit this was written against, with
exactly one change (marked "THE ONE CHANGE vs upstream"). If upstream changes that function,
re-copy it and re-apply that single edit. ``wh_compat.apply()`` asserts the upstream anchor
still exists so drift fails loudly instead of silently running stale code.
"""
import torch  # noqa: F401  (used by the copied body)

import ttnn
from models.common.utility_functions import is_blackhole
from models.experimental.gated_attention_gated_deltanet.tt import ttnn_delta_rule_seq as _seq
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import (  # noqa: F401
    _TILE,
    _bmm_progcfg,
    _ck,
    _compute_L_inv_ttnn,
    _create_tril_ones,
    _create_triu_ones,
    _os,
)

_UPSTREAM = _seq.chunk_gated_delta_rule_seq


def chunk_gated_delta_rule_seq_dispatch(*args, **kwargs):
    """Blackhole -> upstream (verbatim); Wormhole (N300, N150) -> the bf16 variant below;
    T3K -> upstream too (see DELIBERATE narrowing note).

    DELIBERATE narrowing (Wormhole gating audit, item 1, chunk-seq piece): this used to be
    unconditional on Wormhole (is_blackhole()-gated only), so T3K got the bf16 fork along with
    N300 and N150. Narrowed away for T3K specifically, but NOT for N150 -- the two are not
    symmetric:
      * N150 (9B, TP=1, no head split) holds this kernel's [BH,L,V] output at the model's FULL
        head count on one chip -- 2x N300's per-chip size (9B: 32 value heads / TP=2 = 16/chip
        vs N150's 32/chip, same head_dim). N300 already needs bf16 to avoid the fp32 "Out of
        Memory" this file's module docstring documents, at HALF N150's per-chip size -- so N150
        must keep the fix. Narrowing this to wh_9b_n300 (which excludes N150 too) would very
        likely reproduce that exact OOM on N150; left alone on purpose.
      * T3K (27B, TP=8) has 48 value heads / TP=8 = 6/chip -- ~0.375x N300's per-chip size, i.e.
        LESS L1 pressure from this specific tensor than the config that first needed the fix, not
        more. This is a head-count/TP-division ESTIMATE, not a T3K hardware measurement -- accepted
        per explicit instruction to narrow T3K wherever defensible. If a real OOM ever shows up on
        T3K here, the fix is reverting this one exclusion (T3K back to the bf16 fork), not
        reverting N150's protection.
    Detects T3K via the mesh_device kwarg the shared module always passes at its one call site
    (ttnn_delta_rule_seq.py: `chunk_gated_delta_rule_seq(..., mesh_device=device, ...)`) -- not via
    model_args, which this generically-invoked monkeypatch target has no way to receive."""
    if is_blackhole():
        return _UPSTREAM(*args, **kwargs)
    _mesh_device = kwargs.get("mesh_device")
    if _mesh_device is not None and _mesh_device.get_num_devices() == 8:
        return _UPSTREAM(*args, **kwargs)
    return _chunk_gated_delta_rule_seq_wh(*args, **kwargs)


def _chunk_gated_delta_rule_seq_wh(
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
    # THE ONE CHANGE vs upstream: bf16 instead of fp32 for this L1-resident relayout.
    # fp32 costs BH*L*V*4 = 32*2048*128*4 = 33,554,432 B, which does not fit Wormhole's L1
    # (80 cores / ~114MB / 64 banks) beside the live prefill working set; bf16 halves it to
    # 16,777,216 B, which does. Measured: fp32 -> 14x "Out of Memory ... 33554432 B" and 7
    # test_prefill failures; bf16 -> 0 OOM, 17/17 pass, logit PCC 0.9998-1.0000 (unchanged).
    # L1 (not DRAM) is kept deliberately: the DRAM path makes the `o[:, :T, :]` slice below do
    # host reads, and this runs inside begin_trace_capture.
    _out_dtype = ttnn.bfloat16
    # No memory_config: kernel output is already TILE, so it'd be a no-op that warns; the reshape below places it in L1.
    out_4d = ttnn.to_layout(
        ttnn.typecast(out_4d, _out_dtype, memory_config=_out_l1) if out_4d.dtype != _out_dtype else out_4d,
        ttnn.TILE_LAYOUT,
    )
    o = ttnn.reshape(out_4d, [BH, L, V], memory_config=_out_l1)

    if pad_len > 0:
        o = o[:, :T, :]
        o = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=_out_l1)

    return o, final_state
