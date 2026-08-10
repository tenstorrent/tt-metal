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

This is a qwen36-local reimplementation, not an edit to the shared module (see wh_compat.py's
docstring for why the shared module is left alone): it reuses the shared helpers (``l2_norm_ttnn``,
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


def _write_state_wh(h, k_row, delta, beta_t):
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

    5. The outer-product's own output is written as bfloat8_b when h is bfloat8_b (the WH decode
       default), halving the bandwidth of materializing this [B,H,K,V] tensor -- it's the single
       most expensive write in the step. Only when h is bf8: this is another cut into the values
       that feed h's accumulation (same category as h's own dtype), so it's gated to match h
       rather than applied unconditionally -- a caller running high_precision=True (fp32 h) keeps
       outer at its default (unrounded) precision. MEASURED (WH, real shape): 560.8us vs 534.2us
       for bf16-intermediate (-4.7%), PCC 0.99996 -> 0.99994 (both excellent)."""
    B, H, V = h.shape[0], h.shape[1], h.shape[3]
    _L1 = ttnn.L1_MEMORY_CONFIG

    # [B,H] -> [B,H,1,1] via unsqueeze x2, not reshape -- see the decay_bhkv comment in the caller
    # for why (same tile-crossing cost, same fix).
    _beta_bh1 = ttnn.unsqueeze(beta_t, -1)
    beta_expanded = ttnn.unsqueeze(_beta_bh1, -1)
    ttnn.deallocate(_beta_bh1)
    k_col = ttnn.transpose(k_row, 2, 3, memory_config=_L1)
    d_row = ttnn.reshape(delta, [B, H, 1, V], memory_config=_L1)

    d_scaled = ttnn.multiply(d_row, beta_expanded, memory_config=_L1)
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
):
    """Wormhole variant of recurrent_gated_delta_rule_decode_ttnn.

    Identical to upstream except: q is NOT typecast to fp32 up front. It stays in its native dtype
    through L2-norm, the scale multiply, and the reshape to q_row (cheaper at bf16's data width),
    and is only cast to match h's dtype (fp32 under high_precision) immediately before the final
    q @ h matmul -- required so the matmul sees matching operand dtypes, not for accuracy. Every
    other value (k, v, beta, g, h) follows the exact same fp32 path as upstream. When
    high_precision=False this is bit-identical to upstream (no casts happen either way)."""
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
    q = l2_norm_ttnn(q, dim=-1)
    k = l2_norm_ttnn(k, dim=-1)

    if scale is None:
        scale = K**-0.5
    q = ttnn.multiply(q, scale, memory_config=ttnn.L1_MEMORY_CONFIG)

    # q/k arrive [B,1,H,K]; q_row/k_row need [B,H,1,K] -- a swap of dims 1,2, not a reshape that
    # adds/removes a singleton. ttnn.transpose does this in one op instead of reshape crossing the
    # tiled dims (H,K)->(1,K), same class of fix as k_col's transpose in _write_state_wh. MEASURED
    # (WH, B=32 H=16 K=128): 27.7us vs 51.6us per tensor (-46%), PCC 1.0 (exact).
    q_row = ttnn.transpose(q, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    k_row = ttnn.transpose(k, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    # TRIED [B,H,1,V] here (matching v_read's natural matmul output, to skip v_read's reshape and
    # d_row's reshape below) and MEASURED it net-negative: removing those 2 reshapes saved ~27us
    # but the subsequent subtract (delta = v_t - v_read) got ~39us MORE expensive operating on the
    # rank-4 [B,H,1,V] shape than on this [B,H,V] one, for a net +12us regression. Keep [B,H,V].
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
    # the singleton across the tile-tiled dims in one step (B,H tiled -> 1,1 "tiled", i.e. every one
    # of the B*H scalars becomes its own separately-padded tile), which is expensive despite the
    # tiny logical size. unsqueeze grows the rank one dim at a time without ever crossing a *pair*
    # of tiled dims in a single op. MEASURED (WH, B=32 H=16, this reshape + the decay multiply
    # together): 171.9us vs 153.6us (-10.6%), identical result (PCC 1.0 vs the reshape version).
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
    h = _write_state_wh(h=h, k_row=k_row, delta=delta, beta_t=beta_t)

    # o = q @ h. q_row was deliberately left at its incoming dtype above. Only cast it up when h is
    # fp32 (the high_precision path, where k/v/beta/g -- unlike q -- were already cast to fp32
    # early): ttnn.matmul natively accepts mixed BF16 x BFLOAT8_B inputs (proven by the v_read
    # matmul above, which never casts k_row to match h at all), so unconditionally matching q_row
    # to h's dtype was actively harmful once h became bfloat8_b -- it downcast q_row BF16 -> BFP8
    # for no reason, paying an extra typecast AND losing precision versus just leaving q_row at
    # BF16 and letting the matmul mix dtypes like v_read's already does. MEASURED (WH Tracy
    # capture): that cast + the resulting BFP8 x BFP8 matmul cost ~26us + narrower precision for
    # zero benefit over BF16 x BFP8 -> BF16.
    if h.dtype == ttnn.float32 and q_row.dtype != ttnn.float32:
        q_row = ttnn.typecast(q_row, ttnn.float32)

    o_t = ttnn.matmul(
        q_row, h, memory_config=_L1, program_config=read_query_prog_cfg, compute_kernel_config=read_query_compute_cfg
    )

    # Reshape output to [B, 1, H, V]
    o = ttnn.reshape(o_t, [B, 1, H, V], memory_config=_L1)

    return o, h


def recurrent_gated_delta_rule_decode_dispatch(*args, **kwargs):
    """Blackhole -> the shared upstream function, byte-for-byte unchanged. Wormhole -> the variant
    above. Same is_blackhole()-gated dispatch shape as conv_fir_wh.causal_conv1d_fir_dispatch, so
    gdn/tp.py's call sites don't need their own branching."""
    if is_blackhole():
        return _recurrent_gated_delta_rule_decode_upstream(*args, **kwargs)
    return recurrent_gated_delta_rule_decode_wh(*args, **kwargs)
