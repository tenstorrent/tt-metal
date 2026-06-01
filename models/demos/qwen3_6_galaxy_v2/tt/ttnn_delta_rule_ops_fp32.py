# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-decode-debug fp32 fork of the recurrent gated delta rule.

Forked from
``models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py``
(commit context: ``2b32350d33e``).  Adapted so the recurrent state and the
per-step accumulator stay at ``ttnn.float32`` throughout the loop instead of
being typecast to ``ttnn.bfloat16`` (upstream lines 616–649).

Rationale
---------
The 64L decode logits PCC was decaying super-linearly across DeltaNet layers
(4L 0.9996 → 16L 0.998 → 32L 0.94 → 64L 0.30).  The compounding source is the
bf16 round-trip of the recurrent state in each of the 48 DeltaNet layers — the
state buffer in v2 was allocated as bf16, and the kernel additionally typecast
the state to bf16 on entry (upstream line 641).  Every layer's DeltaNet
output is therefore pinched through bf16 precision at the recurrent matmul
boundary, and that quantization compounds via the residual stream.

This file keeps the state at fp32 from start to finish.  Matmul compute kernels
already run with ``fp32_dest_acc_en=True`` — we now also use HiFi4 fidelity and
let the matmul output remain fp32 (instead of typecasting back down to bf16).
The final output is cast to bf16 right before returning so the downstream
GroupRMSNormGated / out-proj matmuls in
``qwen36_delta_attention.TtQwen36DeltaAttention._apply_norm_gated`` consume the
same dtype they were tuned for.

The chunked (prefill) kernel is NOT forked — the upstream ``chunk_gated_delta
_rule_ttnn`` already runs the entire state path at fp32 (upstream lines
941–1310), so prefill is already pristine.  Only the recurrent decode kernel
needs the fp32-state fix.
"""
from __future__ import annotations

import os

import ttnn


def _recur_tiled_enabled():
    """Fuse B: keep the recurrent decode step in its matmul-ready tiled 4D
    layout end-to-end (eliminate the per-op reshape + to_layout(TILE) churn).
    Default-on; flag off restores the old reshape-per-op path."""
    return os.environ.get("QWEN36_DN_RECUR_TILED", "1").strip() not in ("0", "false", "False", "")


def _recur_bf16_in_enabled():
    """Fuse E: drop the up-front fp32 typecasts on the recurrent-decode inputs
    q,k,v,beta (they arrive already bf16-quantized from the conv/projection, so
    casting them UP to fp32 buys zero precision).  The recurrent STATE ``h`` and
    every matmul accumulation / state update stay fp32 (matmuls run with
    ``dtype=fp32`` overrides + ``fp32_dest_acc_en`` so the bf16 inputs feed
    fp32-accumulating ops).  ``g``/decay is kept fp32 because it multiplies the
    fp32 state directly.  Default-OFF — only flipped on after the full-model
    coherence gate (token 248068 + coherent text) passes, since precision
    compounds over 48 DeltaNet layers.

    Flipped DEFAULT-ON after all Fuse-E gates passed: unit PCC bf16-in o_t
    0.999997 / h 0.999999 (delta vs fp32-in only -2e-6); ISL-128 next token
    248068 + coherent text + 20.59 tok/s (vs 20.36 off); 1L block median 6.94 ms
    (vs 7.13 off).  Set "0" to restore the all-fp32-input path."""
    return os.environ.get("QWEN36_DN_RECUR_BF16_IN", "1").strip() not in ("0", "false", "False", "")


# Re-use the program-config helpers from the upstream module (they do not touch
# dtype; they only construct MatmulMultiCoreReuseProgramConfig from shapes).
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    _recurrent_outer_product_program_config,
    _recurrent_read_query_program_config,
    l2_norm_ttnn,
)


def _fp32_compute_cfg_hifi4():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _fused_decay_and_write_fp32(
    h,
    k_t,
    delta,
    decay_t,
    beta_t,
    device=None,
    outer_product_prog_cfg=None,
    matmul_compute_cfg=None,
):
    """fp32-state version of ``fused_decay_and_write_ttnn``.

    h: fp32 [B, H, K, V]
    k_t: fp32 [B, H, K]
    delta: fp32 [B, H, V]
    decay_t: fp32 [B, H]
    beta_t: fp32 [B, H]
    """
    B = h.shape[0]
    H = h.shape[1]
    K = h.shape[2]
    V = h.shape[3]

    decay = ttnn.reshape(decay_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    beta_expanded = ttnn.reshape(beta_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

    k_col = ttnn.to_layout(
        ttnn.reshape(k_t, [B, H, K, 1], memory_config=ttnn.L1_MEMORY_CONFIG),
        ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    d_row = ttnn.to_layout(
        ttnn.reshape(delta, [B, H, 1, V], memory_config=ttnn.L1_MEMORY_CONFIG),
        ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    if outer_product_prog_cfg is None and device is not None:
        try:
            outer_product_prog_cfg = _recurrent_outer_product_program_config(device, K, V)
        except Exception:
            pass

    if matmul_compute_cfg is None:
        matmul_compute_cfg = _fp32_compute_cfg_hifi4()

    outer = ttnn.matmul(
        k_col,
        d_row,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=matmul_compute_cfg,
        program_config=outer_product_prog_cfg,
    )
    k_col.deallocate(True)
    d_row.deallocate(True)

    # V2-11 (lever G): fuse h*decay + outer*beta into 2 ops via addcmul.
    # Original (3 ops):
    #   outer_beta = outer * beta
    #   h_decayed = h * decay
    #   h_new = h_decayed + outer_beta
    # Fused (2 ops):
    #   h_decayed = h * decay
    #   h_new = addcmul(h_decayed, outer, beta_expanded, value=1.0)
    #          = h_decayed + 1.0 * outer * beta_expanded
    # 1 fewer op × 48 DeltaNet layers = 48 fewer ops per decode step.
    h_decayed = ttnn.multiply(h, decay, memory_config=ttnn.L1_MEMORY_CONFIG)
    decay.deallocate(True)
    h_new = ttnn.addcmul(
        h_decayed,
        outer,
        beta_expanded,
        value=1.0,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    h_decayed.deallocate(True)
    outer.deallocate(True)
    beta_expanded.deallocate(True)

    return h_new


def _fused_decay_and_write_fp32_tiled(
    h,
    k_row,
    delta_row,
    decay_t,
    beta_t,
    device=None,
    outer_product_prog_cfg=None,
    matmul_compute_cfg=None,
):
    """Fuse B tiled variant of ``_fused_decay_and_write_fp32``.

    Math-identical to the original, but consumes tensors already in the
    recurrent step's tiled 4D layout so the per-op reshape+to_layout(TILE)
    pairs vanish:

      h:         fp32 [B, H, K, V] tiled (the recurrent state)
      k_row:     fp32 [B, H, 1, K] tiled  (the step's k, NOT reshaped to [B,H,K,1])
      delta_row: fp32 [B, H, 1, V] tiled  (the step's delta, NOT reshaped)
      decay_t:   fp32 [B, H, 1]            (reshaped to [B,H,1,1] here — tiny)
      beta_t:    fp32 [B, H, 1]

    The ONLY layout op kept is the genuine ``k_col`` transpose of k's last two
    dims ([B,H,1,K] -> [B,H,K,1]) for the outer product — that is a real data
    movement, not churn.  Eliminated vs the old path: k reshape+to_layout,
    delta reshape+to_layout.
    """
    B = h.shape[0]
    H = h.shape[1]
    K = h.shape[2]
    V = h.shape[3]

    decay = ttnn.reshape(decay_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    beta_expanded = ttnn.reshape(beta_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

    # k_col: genuine K-major transpose of the tiled [B,H,1,K] k_row -> [B,H,K,1].
    # This is the one layout op that legitimately stays (the outer product needs
    # k as a column). No reshape / no to_layout — transpose preserves TILE.
    k_col = ttnn.transpose(k_row, -1, -2, memory_config=ttnn.L1_MEMORY_CONFIG)
    # d_row is delta itself, already [B,H,1,V] tiled — no reshape, no to_layout.
    d_row = delta_row

    if outer_product_prog_cfg is None and device is not None:
        try:
            outer_product_prog_cfg = _recurrent_outer_product_program_config(device, K, V)
        except Exception:
            pass

    if matmul_compute_cfg is None:
        matmul_compute_cfg = _fp32_compute_cfg_hifi4()

    # Fuse E: outer = k_col @ d_row.  Under bf16-in, k_col is bf16 (transpose of
    # the bf16 k) and d_row=delta is fp32 — force fp32 output so the outer-product
    # increment to the fp32 state keeps full precision.  No-op when fp32.
    outer = ttnn.matmul(
        k_col,
        d_row,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=matmul_compute_cfg,
        program_config=outer_product_prog_cfg,
        dtype=ttnn.float32,
    )
    k_col.deallocate(True)

    h_decayed = ttnn.multiply(h, decay, memory_config=ttnn.L1_MEMORY_CONFIG)
    decay.deallocate(True)
    h_new = ttnn.addcmul(
        h_decayed,
        outer,
        beta_expanded,
        value=1.0,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    h_decayed.deallocate(True)
    outer.deallocate(True)
    beta_expanded.deallocate(True)

    return h_new


def _recurrent_delta_rule_step_fp32(
    q_t,
    k_t,
    v_t,
    beta_t,
    decay_t,
    h,
    seq_len=None,
    device=None,
    read_query_prog_cfg=None,
    matmul_compute_cfg=None,
    outer_product_prog_cfg=None,
):
    """fp32-state recurrent step (all tensors fp32).

    Two layout regimes (math-identical):
      - OLD (3D inputs q_t/k_t/v_t = [B,H,K], beta/decay = [B,H]): each of
        k_row/q_row/d_row/k_col is built per-op via reshape + to_layout(TILE),
        then the matmul output is reshaped back — the layout churn Fuse B
        targets.
      - TILED (Fuse B, QWEN36_DN_RECUR_TILED=1; 4D inputs q_t/k_t/v_t =
        [B,H,1,K] already tiled, beta/decay = [B,H,1]): tensors stay in their
        matmul-ready tiled 4D layout end-to-end; only the genuine k_col
        transpose remains.  Selected automatically when the inputs arrive 4D
        and the flag is on (the pre_transposed decode path).
    """
    tiled = _recur_tiled_enabled() and len(q_t.shape) == 4

    if tiled:
        return _recurrent_delta_rule_step_fp32_tiled(
            q_t,
            k_t,
            v_t,
            beta_t,
            decay_t,
            h,
            seq_len=seq_len,
            device=device,
            read_query_prog_cfg=read_query_prog_cfg,
            matmul_compute_cfg=matmul_compute_cfg,
            outer_product_prog_cfg=outer_product_prog_cfg,
        )

    B = q_t.shape[0]
    H = q_t.shape[1]
    K = q_t.shape[2]
    V = v_t.shape[2]

    # V2-11 (lever G2): skip the explicit to_layout(TILE) at the top.
    # h comes from one of (a) the persistent dn_state_buffer (allocated
    # at __init__ with layout=TILE_LAYOUT — see _build_dn_state_buffer),
    # (b) the previous iteration of this same loop (where it was the
    # output of a matmul — also TILE_LAYOUT), or (c) ttnn.zeros with
    # layout=TILE_LAYOUT. In all paths h is already in TILE_LAYOUT, so
    # the unconditional to_layout is a no-op metadata-write that wastes
    # ~0.05 ms / call × 48 layers = ~2.4 ms / decode step.
    if h.layout != ttnn.TILE_LAYOUT:
        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    if matmul_compute_cfg is None:
        matmul_compute_cfg = _fp32_compute_cfg_hifi4()
    if read_query_prog_cfg is None and device is not None:
        try:
            read_query_prog_cfg = _recurrent_read_query_program_config(device, K, V)
        except Exception:
            pass

    k_row = ttnn.to_layout(
        ttnn.reshape(k_t, [B, H, 1, K], memory_config=ttnn.L1_MEMORY_CONFIG),
        ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    v_read_4d = ttnn.matmul(
        k_row,
        h,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        program_config=read_query_prog_cfg,
        compute_kernel_config=matmul_compute_cfg,
    )
    k_row.deallocate(True)
    v_read = ttnn.reshape(v_read_4d, [B, H, V], memory_config=ttnn.L1_MEMORY_CONFIG)

    delta = ttnn.subtract(v_t, v_read, memory_config=ttnn.L1_MEMORY_CONFIG)
    v_read_4d.deallocate(True)

    h = _fused_decay_and_write_fp32(
        h=h,
        k_t=k_t,
        delta=delta,
        decay_t=decay_t,
        beta_t=beta_t,
        device=device,
        outer_product_prog_cfg=outer_product_prog_cfg,
        matmul_compute_cfg=matmul_compute_cfg,
    )
    delta.deallocate(True)

    q_row = ttnn.to_layout(
        ttnn.reshape(q_t, [B, H, 1, K], memory_config=ttnn.L1_MEMORY_CONFIG),
        ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    o_t = ttnn.matmul(
        q_row,
        h,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        program_config=read_query_prog_cfg,
        compute_kernel_config=matmul_compute_cfg,
    )
    q_row.deallocate(True)
    use_l1 = seq_len is not None and seq_len <= 64
    o_t = ttnn.reshape(o_t, [B, H, V], memory_config=ttnn.L1_MEMORY_CONFIG if use_l1 else None)
    return o_t, h


def _recurrent_delta_rule_step_fp32_tiled(
    q_t,
    k_t,
    v_t,
    beta_t,
    decay_t,
    h,
    seq_len=None,
    device=None,
    read_query_prog_cfg=None,
    matmul_compute_cfg=None,
    outer_product_prog_cfg=None,
):
    """Fuse B: de-churned recurrent step (decode, pre_transposed path).

    Inputs are ALREADY in the matmul-ready tiled 4D layout:
      q_t, k_t, v_t: fp32 [B, H, 1, K] tiled  (no per-token slice / reshape)
      beta_t, decay_t: fp32 [B, H, 1]
      h: fp32 [B, H, K, V] tiled (recurrent state)

    Math is bit-for-bit the same delta-rule step as
    ``_recurrent_delta_rule_step_fp32`` — only the layout plumbing differs.
    Eliminated reshape+to_layout(TILE) pairs vs the old path:
      - k_row build (reshape [B,H,K]->[B,H,1,K] + to_layout)  -> k_t used as-is
      - v_read reshape-back ([B,H,1,V]->[B,H,V])               -> kept 4D
      - delta / d_row reshape+to_layout                        -> delta kept 4D
      - q_row build (reshape + to_layout)                      -> q_t used as-is
      - o_t reshape-back                                       -> kept 4D
    The only layout op kept is the genuine k_col transpose (inside the tiled
    decay-and-write helper).
    """
    B = q_t.shape[0]
    H = q_t.shape[1]
    K = q_t.shape[3]
    V = v_t.shape[3]

    if h.layout != ttnn.TILE_LAYOUT:
        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    if matmul_compute_cfg is None:
        matmul_compute_cfg = _fp32_compute_cfg_hifi4()
    if read_query_prog_cfg is None and device is not None:
        try:
            read_query_prog_cfg = _recurrent_read_query_program_config(device, K, V)
        except Exception:
            pass

    # Fuse E: when k_t/v_t arrive bf16 (QWEN36_DN_RECUR_BF16_IN), the read matmul
    # is mixed bf16(k) x fp32(h).  Force dtype=fp32 on the output so the state
    # read-back keeps full precision (a mixed matmul otherwise packs to bf16,
    # which would pinch the fp32 state at the read boundary).  No-op when inputs
    # are fp32.
    # READ: v_read = k @ h.  k_t is already [B,H,1,K] tiled — the read_query
    # matmul's exact in0 contract.  Output stays [B,H,1,V] (no reshape-back).
    v_read = ttnn.matmul(
        k_t,
        h,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        program_config=read_query_prog_cfg,
        compute_kernel_config=matmul_compute_cfg,
        dtype=ttnn.float32,
    )

    # delta stays [B,H,1,V] tiled — directly usable as d_row.  Force fp32 so the
    # subtract(v_t[bf16], v_read[fp32]) does not pack down to bf16.
    delta = ttnn.subtract(v_t, v_read, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.float32)
    v_read.deallocate(True)

    h = _fused_decay_and_write_fp32_tiled(
        h=h,
        k_row=k_t,
        delta_row=delta,
        decay_t=decay_t,
        beta_t=beta_t,
        device=device,
        outer_product_prog_cfg=outer_product_prog_cfg,
        matmul_compute_cfg=matmul_compute_cfg,
    )
    delta.deallocate(True)

    # READOUT: o = q @ h_new.  q_t is already [B,H,1,K] tiled.  Output [B,H,1,V]
    # — returned 4D directly (the caller's reshape to [B,H,1,V] is then a no-op).
    # Force fp32 so the q[bf16] x h[fp32] readout does not pack to bf16 before
    # the final boundary cast (matches the all-fp32 path's output precision).
    o_t = ttnn.matmul(
        q_t,
        h,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        program_config=read_query_prog_cfg,
        compute_kernel_config=matmul_compute_cfg,
        dtype=ttnn.float32,
    )
    return o_t, h


def recurrent_gated_delta_rule_ttnn_fp32(
    q,
    k,
    v,
    beta,
    g,
    scale=None,
    initial_state=None,
    device=None,
    pre_transposed=False,
):
    """Token-by-token recurrent gated delta rule, fp32-state.

    Same math/semantics as upstream ``recurrent_gated_delta_rule_ttnn`` but
    keeps the state and all per-step intermediates at ``ttnn.float32`` instead
    of casting down to ``ttnn.bfloat16``.  Used by the qwen3.6 DeltaNet decode
    path to remove the per-layer bf16 state quantization (see module docstring).

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g: [B, T, H]
        scale: float
        initial_state: [B, H, K, V] — accepted at fp32 OR bf16; will be cast
                       up to fp32 internally.
        device: ttnn device

    Returns:
        output: [B, T, H, V] (bf16; matches downstream norm/proj contract)
        final_state: [B, H, K, V] (fp32)
    """
    q = l2_norm_ttnn(q, dim=-1)
    k = l2_norm_ttnn(k, dim=-1)

    if pre_transposed:
        # Inputs already in [B, H, T, D] (q/k/v) / [B, H, T] (beta/g) layout —
        # the caller produced the per-head tensors directly so the 5 input
        # transposes below can be skipped. At decode T=1 this is bit-identical.
        B = q.shape[0]
        H = q.shape[1]
        T = q.shape[2]
        K = q.shape[3]
        V = v.shape[3]
    else:
        B = q.shape[0]
        T = q.shape[1]
        H = q.shape[2]
        K = q.shape[3]
        V = v.shape[3]

    if scale is None:
        scale = K**-0.5

    q_prev = q
    q = ttnn.multiply(q_prev, scale, memory_config=ttnn.L1_MEMORY_CONFIG)
    q_prev.deallocate(True)

    if not pre_transposed:
        # Transpose [B, T, H, D] -> [B, H, T, D] / [B, H, T]
        q_prev = q
        q = ttnn.transpose(q_prev, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
        q_prev.deallocate(True)
        k_prev = k
        k = ttnn.transpose(k_prev, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
        k_prev.deallocate(True)
        v = ttnn.transpose(v, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
        beta = ttnn.transpose(beta, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
        g = ttnn.transpose(g, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Fuse E: by default promote q,k,v,beta,g to fp32 (upstream was bf16).  With
    # QWEN36_DN_RECUR_BF16_IN=1 we SKIP the q,k,v promotions — these three feed
    # the recurrent matmuls (read v_read=k@h, outer k_col@delta, readout q@h_new)
    # and arrive already bf16-quantized, so the up-cast buys zero precision; the
    # state + matmul accumulation stay fp32 via the dtype=fp32 output overrides
    # inside the tiled step (mixed bf16xfp32 matmul, fp32_dest_acc).
    #
    # ``beta`` and ``g`` are ALWAYS promoted to fp32: beta_expanded feeds the
    # ``addcmul`` that increments the fp32 state (and addcmul with a bf16 scalar
    # operand produces NaN on this build — verified), and exp(g)=decay multiplies
    # the fp32 state in ``multiply(h, decay)``.  Keeping both fp32 avoids a
    # state-precision pinch (and the NaN), and they are tiny per-head scalars so
    # the two casts cost ~nothing — the win is the q/k/v casts.
    bf16_in = _recur_bf16_in_enabled()
    if not bf16_in:
        q_prev = q
        q = ttnn.typecast(q_prev, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        if q is not q_prev:
            q_prev.deallocate(True)
        k_prev = k
        k = ttnn.typecast(k_prev, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        if k is not k_prev:
            k_prev.deallocate(True)
        v_prev = v
        v = ttnn.typecast(v_prev, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        if v is not v_prev:
            v_prev.deallocate(True)
    beta_prev = beta
    beta = ttnn.typecast(beta_prev, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
    if beta is not beta_prev:
        beta_prev.deallocate(True)
    g_prev = g
    g = ttnn.typecast(g_prev, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
    if g is not g_prev:
        g_prev.deallocate(True)

    g_exp = ttnn.exp(g, memory_config=ttnn.L1_MEMORY_CONFIG)
    g.deallocate(True)

    if initial_state is not None:
        if initial_state.dtype != ttnn.float32:
            h = ttnn.typecast(initial_state, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            # copy out of DRAM into L1 (also gives us a writable buffer).
            h = ttnn.to_memory_config(initial_state, ttnn.L1_MEMORY_CONFIG)
    else:
        h = ttnn.zeros(
            [B, H, K, V],
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    matmul_compute_cfg = _fp32_compute_cfg_hifi4()
    read_query_prog_cfg = None
    outer_product_prog_cfg = None
    if device is not None:
        try:
            read_query_prog_cfg = _recurrent_read_query_program_config(device, K, V)
            outer_product_prog_cfg = _recurrent_outer_product_program_config(device, K, V)
        except Exception:
            pass

    # Fuse B: in the pre_transposed decode path keep the per-token tensors in
    # their tiled 4D [B,H,1,K] layout (q[:, :, i:i+1] keeps the T axis) so the
    # step never has to reshape+to_layout per op.  The old path slices to 3D
    # [B,H,K] and rebuilds the layout inside the step (its original behaviour).
    step_tiled = pre_transposed and _recur_tiled_enabled()

    outputs_4d_list = []
    for i in range(T):
        if step_tiled:
            q_t = q[:, :, i : i + 1]  # [B,H,1,K] tiled
            k_t = k[:, :, i : i + 1]
            v_t = v[:, :, i : i + 1]
            beta_t = beta[:, :, i : i + 1]  # [B,H,1]
            decay_t = g_exp[:, :, i : i + 1]
        else:
            q_t = q[:, :, i]
            k_t = k[:, :, i]
            v_t = v[:, :, i]
            beta_t = beta[:, :, i]
            decay_t = g_exp[:, :, i]

        o_t, h = _recurrent_delta_rule_step_fp32(
            q_t,
            k_t,
            v_t,
            beta_t,
            decay_t,
            h,
            seq_len=T,
            device=device,
            read_query_prog_cfg=read_query_prog_cfg,
            matmul_compute_cfg=matmul_compute_cfg,
            outer_product_prog_cfg=outer_product_prog_cfg,
        )
        o_4d = ttnn.reshape(o_t, [B, H, 1, V], memory_config=ttnn.L1_MEMORY_CONFIG)
        outputs_4d_list.append(o_4d)

    o = ttnn.concat(outputs_4d_list, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
    outputs_4d_list.clear()
    o_prev = o
    o = ttnn.transpose(o_prev, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    o_prev.deallocate(True)
    # Downstream norm/out_proj expects bf16 — cast back at the boundary only.
    o_prev = o
    o = ttnn.typecast(o_prev, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    if o is not o_prev:
        o_prev.deallocate(True)

    return o, h
