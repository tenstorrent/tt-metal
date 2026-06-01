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

import ttnn

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
    """fp32-state recurrent step (all tensors fp32)."""
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

    # fp32 promote — was bf16 in upstream
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

    outputs_4d_list = []
    for i in range(T):
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
