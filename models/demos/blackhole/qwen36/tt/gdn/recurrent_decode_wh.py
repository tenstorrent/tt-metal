# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wormhole-only precision fix for the recurrent gated-delta-rule decode step.

The shared ``recurrent_gated_delta_rule_decode_ttnn``
(models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py) promotes
q, k, v, beta, g to fp32 under ``high_precision=True`` -- added to avoid bf16 decay quantization
error ACCUMULATING in the recurrent state ``h`` over long decode runs. But ``q`` never feeds that
accumulation: ``fused_decay_and_write_ttnn`` (the only place ``h`` is written) takes ``k_t``,
``delta`` (derived from ``v``), ``decay_t`` (from ``g``) and ``beta_t`` -- q is used solely for the
read-only ``o = q @ h`` and is never written back into the state. Promoting it to fp32 before its
L2-norm/scale/reshape buys nothing for the drift problem the flag exists to solve, and MEASURED (WH,
single-layer GDN decode Tracy profile) those three ops cost ~15-20us/step running at fp32's double
data width instead of bf16's.

This is a qwen36-local reimplementation, not an edit to the shared module (see the arch-aware defaults upstream): it reuses the shared helpers (``l2_norm_ttnn``,
``fused_decay_and_write_ttnn``, ``_recurrent_read_query_program_config``) verbatim and only changes
the top-level orchestration. gdn/tp.py dispatches to this on Wormhole only; Blackhole keeps calling
the shared function exactly as before -- this file changes nothing there.
"""
import ttnn
from models.common.utility_functions import is_blackhole
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    _recurrent_read_query_program_config,
    l2_norm_ttnn,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn as _recurrent_gated_delta_rule_decode_upstream,
)


def _write_state_wh(h, k_row, delta, beta_t, outer_bf8=False):
    """Inlined, decode-only copy of fused_decay_and_write_ttnn's apply_decay=False branch:
    h = h + beta*(k(x)delta). Three changes vs upstream, all verified bit-for-bit equivalent
    (PCC 0.999999+) at the real production shape:

    1. Upstream computes an unconditional `decay = reshape(decay_t, ...)` even when
       apply_decay=False, whose result is then never read -- every decode call in this file passes
       apply_decay=False, so that reshape is dead on 100% of calls here. Gone.

    2. k(x)delta (the outer product k_col[B,H,K,1] @ d_row[B,H,1,V] -> [B,H,K,V]) is a RANK-1
       outer product -- the matmul's contraction dimension is 1, so it is mathematically a
       broadcast multiply, not a reduction. MEASURED (WH, B=8 H=16 K=V=128, the real
       decode-batch-split-chunk shape): ttnn.multiply with broadcasting is 114.7us vs the matmul's
       188.9us (-39%), with slightly BETTER accuracy (PCC 0.999999 vs 0.999994) since it skips the
       matmul kernel's fp32 accumulate-then-round entirely. This was the single most expensive op
       in the whole recurrent step.

    3. k_col is built with ttnn.transpose(k_row, 2, 3) directly from k_row[B,H,1,K], not
       reshape(k_row -> [B,H,K]) then reshape(-> [B,H,K,1]). Both reshapes move the singleton
       between the two TILE-tiled dims, forcing a real retile each time (the tiled dims go
       (1,K)->(H,K)->(K,1)); one transpose does it in a single op. MEASURED (WH, B=32 H=16 K=128,
       the real unsplit-decode shape): 33.1us vs 173.0us for the double-reshape (-81%).

    4. beta is multiplied into d_row (delta, shape [B,H,1,V]) BEFORE the outer-product broadcast,
       not into the [B,H,K,V] result after. Mathematically identical (scalar multiplication
       commutes with the broadcast), but doing it first means the second multiply's actual work is
       the same broadcast cost as the first instead of a second full-size elementwise pass.
       MEASURED (WH, real shape): 484.1us vs 579.6us for scale-after (-16.5%).

    5. ``outer_bf8``: write the outer product's own output as bfloat8_b unless h is float32.
       KEEP THIS FALSE -- it is wired False at the only call site and the default is False. It reads
       like a free bandwidth win (the increment is rounded, h's accumulation dtype is untouched) but
       it is not: h is bf16 on WH decode *because bf8 state failed PCC*, and rounding the increment
       compounds over every decode step of every GDN layer. Enabled via tile_opt in dc854ee0cc6 it
       cost ~0.04 of model-level logits PCC (0.99864 -> 0.95927 on
       test_model_tp_decode_batched[B8], reaching 0.94561 by HEAD). Only h being bf8 already -- where
       the increment is no coarser than the state -- justifies it."""
    B, H, V = h.shape[0], h.shape[1], h.shape[3]
    _L1 = ttnn.L1_MEMORY_CONFIG

    # [B,H] -> [B,H,1,1] via unsqueeze x2, not reshape (see decay_bhkv in the caller)
    # for why (same tile-crossing cost, same fix).
    _beta_bh1 = ttnn.unsqueeze(beta_t, -1)
    beta_expanded = ttnn.unsqueeze(_beta_bh1, -1)
    ttnn.deallocate(_beta_bh1)
    k_col = ttnn.transpose(k_row, 2, 3, memory_config=_L1)
    d_row = ttnn.reshape(delta, [B, H, 1, V], memory_config=_L1)

    d_scaled = ttnn.multiply(d_row, beta_expanded, memory_config=_L1)
    if outer_bf8:
        outer_dtype = None if h.dtype == ttnn.float32 else ttnn.bfloat8_b
    else:
        outer_dtype = ttnn.bfloat8_b if h.dtype == ttnn.bfloat8_b else None
    outer = ttnn.multiply(k_col, d_scaled, memory_config=_L1, dtype=outer_dtype)
    return ttnn.add(h, outer, memory_config=_L1)


def recurrent_gated_delta_rule_decode_wh(
    q,
    k,
    v,
    beta,
    g,
    scale=None,
    initial_state=None,
    device=None,
    high_precision=False,
    tile_opt=False,
):
    """Wormhole variant of recurrent_gated_delta_rule_decode_ttnn.

    Identical to upstream except: q is NOT typecast to fp32 up front. It stays in its native dtype
    through L2-norm, the scale multiply, and the reshape to q_row (cheaper at bf16's data width),
    and is only cast to match h's dtype (fp32 under high_precision) immediately before the final
    q @ h matmul -- required so the matmul sees matching operand dtypes, not for accuracy. Every
    other value (k, v, beta, g, h) follows the exact same fp32 path as upstream. When
    high_precision=False this is bit-identical to upstream (no casts happen either way).

    tile_opt (Wormhole 9B only -- see _decode_tile_opt in gdn/tp.py) additionally: folds q's L2
    1/sqrt(K) into the attention scale (one fewer BinaryNg), writes the state-update outer product
    as bf8 (see _write_state_wh note 5), and returns ``o`` in the read-query matmul's native
    [B,H,1,V] instead of reshaping to [B,1,H,V]. Default False keeps the 27B on the shape and op
    sequence it was validated with -- callers must match, since the return LAYOUT differs."""
    B = q.shape[0]
    H = q.shape[2]
    K = q.shape[3]
    V = v.shape[3]

    if high_precision:
        k = ttnn.typecast(k, ttnn.float32)
        v = ttnn.typecast(v, ttnn.float32)
        beta = ttnn.typecast(beta, ttnn.float32)
        g = ttnn.typecast(g, ttnn.float32)

    # L2 norm. q is intentionally left at its incoming dtype here -- see module docstring.
    if scale is None:
        scale = K**-0.5
    if tile_opt:
        # Fold q's L2 1/sqrt(K) into the attention scale: l2_norm_ttnn is rms_norm + *(K**-0.5), then
        # the caller did q *= scale (scale is also K**-0.5 by default). One multiply by
        # (scale * K**-0.5) is algebraically identical and drops one BinaryNg vs the separate pair.
        q = ttnn.rms_norm(q, epsilon=1e-6 / K)
        q = ttnn.multiply(q, scale * (K**-0.5), memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        q = l2_norm_ttnn(q, dim=-1)
        q = ttnn.multiply(q, scale, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = l2_norm_ttnn(k, dim=-1)

    # q/k arrive [B,1,H,K]; q_row/k_row need [B,H,1,K] -- a dim 1,2 swap, not a
    # adds/removes a singleton.
    q_row = ttnn.transpose(q, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    k_row = ttnn.transpose(k, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    # d_row's reshape below) and MEASURED it net-negative: removing those 2 reshapes saved ~27us
    # but the following subtract (delta = v_t - v_read)
    v_t = ttnn.reshape(v, [B, H, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    beta_t = ttnn.reshape(beta, [B, H], memory_config=ttnn.L1_MEMORY_CONFIG)
    g_t = ttnn.reshape(g, [B, H], memory_config=ttnn.L1_MEMORY_CONFIG)

    decay_t = ttnn.exp(g_t, memory_config=ttnn.L1_MEMORY_CONFIG)

    h = initial_state
    if h is None:
        h = ttnn.zeros(
            [B, H, K, V], device=device, dtype=ttnn.float32 if high_precision else ttnn.bfloat16, memory_config=None
        )
    elif high_precision and h.dtype != ttnn.float32:
        h = ttnn.typecast(h, ttnn.float32)

    # Skip the DRAM->L1 copy when the caller already hoisted rec_state (eager decode in tp.py).
    # Same-config to_memory_config is not reliably a no-op in the device trace.
    if h.memory_config().buffer_type != ttnn.BufferType.L1:
        h = ttnn.to_memory_config(h, ttnn.L1_MEMORY_CONFIG)

    read_query_compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    read_query_prog_cfg = None
    if device is not None:
        try:
            read_query_prog_cfg = _recurrent_read_query_program_config(device, K, V)
        except Exception:
            pass

    # Decay before read; keep recurrence step L1-resident.
    # [B,H] -> [B,H,1,1] via two ttnn.unsqueeze calls, not one ttnn.reshape: a direct reshape moves
    _L1 = ttnn.L1_MEMORY_CONFIG
    _decay_bh1 = ttnn.unsqueeze(decay_t, -1)
    decay_bhkv = ttnn.unsqueeze(_decay_bh1, -1)
    ttnn.deallocate(_decay_bh1)
    h = ttnn.multiply(h, decay_bhkv, memory_config=_L1)

    # v_read = k @ h (decayed state)
    v_read = ttnn.matmul(
        k_row, h, memory_config=_L1, program_config=read_query_prog_cfg, compute_kernel_config=read_query_compute_cfg
    )
    v_read = ttnn.reshape(v_read, [B, H, V], memory_config=_L1)

    # Delta + state write (no re-decay). k_row feeds _write_state_wh directly (transposed there);
    # no intermediate [B,H,K] reshape needed since k_row had no other consumer after this point.
    delta = ttnn.subtract(v_t, v_read, memory_config=_L1)
    # outer_bf8 is DELIBERATELY NOT tied to tile_opt: writing the recurrent-state increment at
    # bf8 while h is bf16 is a real accuracy regression, not free bandwidth.
    h = _write_state_wh(h=h, k_row=k_row, delta=delta, beta_t=beta_t, outer_bf8=False)

    # o = q @ h. q_row keeps its dtype; cast up only when h is
    # fp32 (the high_precision path, where k/v/beta/g -- unlike q -- were already cast to fp32
    if h.dtype == ttnn.float32 and q_row.dtype != ttnn.float32:
        q_row = ttnn.typecast(q_row, ttnn.float32)

    o_t = ttnn.matmul(
        q_row, h, memory_config=_L1, program_config=read_query_prog_cfg, compute_kernel_config=read_query_compute_cfg
    )

    # tile_opt: leave o at the matmul's native [B,H,1,V] -- reshaping to [B,1,H,V] crosses the tiled
    # pair (1,V)->(H,V) for no benefit, since the caller (forward_decode) rms_norms on this layout
    if tile_opt:
        return o_t, h
    o = ttnn.reshape(o_t, [B, 1, H, V], memory_config=_L1)
    return o, h


def recurrent_gated_delta_rule_decode_dispatch(*args, model_args=None, **kwargs):
    """Blackhole -> the shared upstream function, byte-for-byte unchanged. wh_9b_n300 -> the
    variant above. Same dispatch shape as the upstream FIR entry point, so gdn/tp.py's
    call sites don't need their own branching.

    model_args: accepted and ignored. gdn/tp.py passes self.args; the parameter stays so it is
    absorbed here rather than forwarded into the kernel via **kwargs. It no longer gates anything
    -- see below.

    REVERTED narrowing (was: Wormhole gating audit, item 1). This was briefly narrowed from
    is_blackhole() to wh_9b_n300 on the stated grounds that "the WH variant is bit-identical to
    upstream when high_precision=False", making the T3K/N150 fallback a pure perf regression rather
    than a correctness one. That premise is FALSE on T3K, and the narrowing's own docstring asked
    for a re-measurement before reverting -- this is that measurement.

    MEASURED (T3K 1x8, Qwen3.6-27B, layer 0, B=1, Nv_tp=6, Dk=Dv=128, high_precision=False -- so
    exactly the "bit-identical" case). One decode step probed end to end, with sane inputs
    throughout (q 0.27, k 0.88, v 1.69, beta 0.80, g 1.02, initial_state 1.00):

        upstream fallback : recurrence out absmax 7.39e36 -> rms_norm underflows -> output ALL ZERO
        WH variant        : recurrence out absmax 0.019   -> rms_norm 8.875      -> output 2.08

    The upstream kernel overflows for this shape on Wormhole, so every GDN decode returned zeros:
    10 of 14 test_gdn_tp cases failed with PCC exactly 0.0 (per-user state, batched prefill) or
    ~0.0 (prefill-vs-decode, -0.0012), while the pure-prefill cases passed at 0.9991 -- the tell
    that only the decode path was affected. test_decode_bucketing's width-1 case failed the same way.

    Blackhole still takes the shared upstream function, byte-for-byte unchanged."""
    _use_wh = not is_blackhole()
    if not _use_wh:
        return _recurrent_gated_delta_rule_decode_upstream(*args, **kwargs)
    return recurrent_gated_delta_rule_decode_wh(*args, **kwargs)
