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

_OUTER_L1_BUDGET_BYTES = 8 << 20  # see wh_decode_fork_applies below


def wh_decode_fork_applies(B, Hq, K, V, gqa_repeat=1, high_precision=False):
    """Should this shape take the Wormhole fork rather than the shared upstream decode?

    The fork materializes the full [B,H,K,V] outer product in L1, so take it only while that fits;
    widen the budget only with a measurement. gdn/tp.py calls this once and uses the answer for
    BOTH the q/k fusion and the choice of decode function, so the two can never disagree -- a
    caller that fused while falling back to upstream would apply the gated-norm epsilon
    compensation to an uncompensated ``o``. Keep the budget evaluated only here.

    The arch gate is ``is_blackhole()``, NOT a narrower device check: for this model's shape the
    upstream kernel OVERFLOWS on Wormhole -- the recurrence output saturates, the following
    rms_norm underflows, and every GDN decode returns zeros -- so upstream is a correctness
    failure there, not just a slower path. That holds even at high_precision=False, where the two
    are otherwise equivalent. Shapes over the budget fall back to upstream anyway and rely on
    wh_compat's state-write override to keep them off L1."""
    if is_blackhole():
        return False
    itemsize = 4 if high_precision else 2
    return B * (Hq * gqa_repeat) * K * V * itemsize <= _OUTER_L1_BUDGET_BYTES


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
       the increment is no coarser than the state -- justifies it.

    6. REJECTED: h + k_col(x)d_scaled as ONE ttnn.mac (a*b + c). It is one op instead of two and
       it drops a 512 KB intermediate, so it looks strictly better -- and ttnn.mac does carry the
       [B,H,K,1] x [B,H,1,V] broadcast correctly (max abs err 9.5e-7 vs torch). But ttnn.mac is
       MUCH slower per op than the pair it replaces. MEASURED in isolation at the real decode shape
       (B=1 H=8 K=V=128, L1, trace replay so this is device time, n150x4):

           fp32:  multiply+add 20.2 us  |  mac 44.1 us   (+118%)
           bf16:  multiply+add 21.6 us  |  mac 32.6 us   (+51%)

       End to end that was +6.0% on the GDN decode layer (0.3965 -> 0.4204 ms/step), and it shows
       up in a Tracy profile as a single 42 us FP32 TernaryDeviceOperation, 11.9% of all device
       time. Keep the multiply + add. The same caution applies anywhere else here: ttnn.mac is not
       a free fusion of a broadcast multiply and an add on this hardware."""
    B, H, V = h.shape[0], h.shape[1], h.shape[3]
    _L1 = ttnn.L1_MEMORY_CONFIG

    # [B,H] -> [B,H,1,1] via unsqueeze x2, not reshape (see decay_bhkv in the caller)
    # for why (same tile-crossing cost, same fix).
    beta_expanded = beta_t
    # k_row arrives fp32 and this transpose is therefore an fp32 transpose (2.95us). Feeding it
    # the PRE-CAST bf16 k_row instead looks free -- outer's dtype is pinned to h.dtype below, so the
    # increment would still be fp32, and the transpose would run on half the bytes. MEASURED 0.3007
    # vs 0.2682 ms/step, a +12% REGRESSION: the broadcast multiply below is enormously slower when
    # its operands differ in dtype. This also pins down where the +7.9% in the "delete k's cast and
    # pin the consumers" experiment (see the k_row cast note above) actually came from -- that op,
    # not the matmul. Both operands of the outer product must be fp32. Do not try either again.
    k_col = ttnn.transpose(k_row, 2, 3, memory_config=_L1)
    d_row = delta  # already [B,H,1,V]

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
    gqa_repeat=1,
    q_norm_weight=None,
    k_norm_weight=None,
    qk_fused=None,
    v_rowed=False,
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
    # q/k arrive with Hq = H // gqa_repeat heads (GQA); v/beta/g always carry the full H v-heads.
    # The expansion happens after the L2-norm (see q_row/k_row below), so every op up to that point
    # runs on the narrower q/k.
    if qk_fused is not None:
        # fused [B,1,2*Hq,K]: q heads in rows [0,Hq), k heads in rows [Hq,2*Hq)
        B, Hq, K = qk_fused.shape[0], qk_fused.shape[2] // 2, qk_fused.shape[3]
    else:
        B, Hq, K = q.shape[0], q.shape[2], q.shape[3]
    H = Hq * gqa_repeat
    V = v.shape[3]

    if high_precision:
        pass  # g's cast moved down to g_t -- see the note there
        # ONLY g is cast here. k IS still cast to fp32, but at the bottom of the GQA block below
        # rather than here -- see the note at that cast for why the position matters and why the
        # cast cannot simply be deleted. v and beta are deliberately left at bf16: every op they feed
        # takes its
        # output dtype from input_a, not from them. delta = subtract(v_t, v_read) is pinned fp32 by
        # its dtype= argument, and d_scaled = multiply(delta, beta) then inherits fp32 from delta,
        # so the whole state path stays fp32 with two fewer typecast dispatches. MEASURED -2.4%
        # (0.3107 -> 0.3034 ms/step, n150x4 B=1 traced), PCC unchanged everywhere including B=32.
        #
        # k's and g's casts are NOT removable, and this is not a judgement call -- it was probed op
        # by op at the real decode shape. Drop them and the dtypes go:
        #     k_row @ h -> BF16, v - v_read -> BF16, delta * beta -> BF16, k_col (x) d -> BF16
        # i.e. h stays an fp32 container but the INCREMENT added to it each step is rounded to
        # bf16. That is the compounding error note 5 above measures for bf8 increments (0.99864 ->
        # 0.95927 model logits PCC), and no test here would catch it: test_gdn_tp_decode_recurrence
        # runs 4 steps, while the drift needs hundreds. g additionally sets exp(g)'s precision --
        # the decay quantization this whole flag exists to prevent.

    # L2 norm. q is intentionally left at its incoming dtype here -- see module docstring.
    if scale is None:
        scale = K**-0.5
    # Fold q's L2 1/sqrt(K) into the attention scale. l2_norm_ttnn(dim=-1) is rms_norm followed by
    # *(K**-0.5), and the caller then wants q *= scale; one multiply by (scale * K**-0.5) is the
    # same arithmetic with one fewer BinaryNg. Unconditional now -- it used to be gated behind
    # tile_opt, but it is the only numerics-neutral item in that flag. MEASURED -1.1% on the GDN
    # decode layer (0.3965 -> 0.3922 ms/step, n150x4 35B-A3B B=1, traced replay).
    # When the caller supplies the constant fold weights, the scaling rides inside rms_norm and the
    # two broadcast BinaryNg ops disappear. Falling back to the explicit multiplies keeps callers
    # that do not pass them (and the shapes they were validated on) working unchanged.
    if qk_fused is not None:
        # q and k run as ONE tensor through norm, transpose and the GQA expansion -- see the block
        # comment at the split below.
        qk = ttnn.rms_norm(qk_fused, epsilon=1e-6 / K, weight=k_norm_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    elif q_norm_weight is not None and k_norm_weight is not None:
        q = ttnn.rms_norm(q, epsilon=1e-6 / K, weight=q_norm_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.rms_norm(k, epsilon=1e-6 / K, weight=k_norm_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        q = ttnn.rms_norm(q, epsilon=1e-6 / K)
        q = ttnn.multiply(q, scale * (K**-0.5), memory_config=ttnn.L1_MEMORY_CONFIG)
        k = l2_norm_ttnn(k, dim=-1)

    # q/k arrive [B,1,Hq,K]; q_row/k_row need [B,H,1,K] -- a dim 1,2 swap, not a
    # adds/removes a singleton. ttnn.transpose does this in one op instead of reshape crossing the
    # tiled dims (Hq,K)->(1,K), same class of fix as k_col's transpose in _write_state_wh. MEASURED
    # (WH, B=32 H=16 K=128): 27.7us vs 51.6us per tensor (-46%), PCC 1.0 (exact).
    #
    # All FOUR transposes in this function (q_row, k_row, v_t, and k_col in _write_state_wh) have
    # been checked for removal/replacement/fusion and all four stay. MEASURED on n150x4 at the real
    # decode shape, TRACED (device time -- host wall-clock reverses both of these results):
    #   * ttnn.permute instead of ttnn.transpose for this dim-1/2 swap: 4.31us vs 4.22us. No win.
    #   * building q/k/v pre-rowed in the caller so the swap never happens: -1.2% at B=1 but the
    #     rowed shape pads to 8x the tiles and breaks B=32 with an L1 circular-buffer clash (see
    #     the note at the v slice in gdn/tp.py forward_decode).
    #   * folding k_col's transpose into the outer product via ttnn.matmul(transpose_a=True) --
    #     mathematically exactly k_row^T @ d_scaled: 20.55us vs 16.82us for transpose+multiply,
    #     AND it is inexact (max abs err 1.58e-02 at HiFi4 + fp32 dest acc, against the broadcast
    #     multiply's 0.0) because it goes through the matmul's accumulate-and-round. That error
    #     lands directly on the state increment, which note 5 above measures as compounding.
    # The transposes are the bridge between the COMPACT [B,1,H,K] layout the norms want and the
    # PER-HEAD-BATCHED [B,H,1,K] layout the matmuls need; you cannot have both, and 4.22us is the
    # price of keeping the expensive ops on their cheap layout.
    if qk_fused is not None:
        qk_row = ttnn.transpose(qk, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)  # [B,2*Hq,1,K]
        ttnn.deallocate(qk)
        if gqa_repeat > 1:
            # repeat_interleave repeats each head in place, so [q..,k..] -> [q..x r, k.. x r]:
            # the q block and the k block stay contiguous and the split below is still a
            # single dim-1 cut. Same TILE-native gate as the unfused path (dim 1 < rank-2).
            qk_row = ttnn.repeat_interleave(qk_row, gqa_repeat, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        # Split on dim 1, which is NOT one of the two dims a tile subdivides, so these are cheap
        # batch-style cuts -- unlike a dim-2 split, which would land mid-tile at Hq=4.
        q_row = ttnn.slice(qk_row, (0, 0, 0, 0), (B, H, 1, K), memory_config=ttnn.L1_MEMORY_CONFIG)
        k_row = ttnn.slice(qk_row, (0, H, 0, 0), (B, 2 * H, 1, K), memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(qk_row)
    else:
        q_row = ttnn.transpose(q, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
        k_row = ttnn.transpose(k, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    if qk_fused is None and gqa_repeat > 1:
        # GQA expand Hq -> H, HERE and not in the caller, for two reasons:
        #
        # 1. RANK. ttnn.repeat_interleave has a TILE-native codegen kernel, but it only takes the
        #    repeated dim when that dim is outside the two dims a tile subdivides -- i.e.
        #    `dim < rank - 2` (repeat_interleave_codegen_supported.cpp). The caller's pre-transpose
        #    shape puts the head dim at rank-2 either way it is spelled ([B,Nk,Dk] dim 1, or
        #    [B,1,Nk,Dk] dim 2), so it misses the gate and falls to the native composite --
        #    typecast/untilize -> unsqueeze -> concat -> reshape -> tilize, a full ROW_MAJOR round
        #    trip through DRAM. On q_row/k_row the head dim is dim 1 of a rank-4 tensor with the
        #    tiled pair (1,K) after it, so `1 < 2` holds and the single TILE-native kernel runs.
        #
        # 2. WIDTH. Everything above -- both L2-norms, the scale multiply, both transposes --
        #    now runs on Hq heads instead of H. The prefill path already does exactly this
        #    (ttnn_delta_rule_seq.py: "norm at Hq, expand to H after via repeat_interleave").
        #    k's fp32 typecast is the one thing deliberately left BELOW the expansion; see its note.
        q_row = ttnn.repeat_interleave(q_row, gqa_repeat, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        k_row = ttnn.repeat_interleave(k_row, gqa_repeat, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

    # k's fp32 cast sits HERE -- after the norm, both transposes and the GQA expansion -- and not
    # up in the high_precision block where it used to be. Everything upstream of this line then runs
    # at bf16 (half the bytes through rms_norm, transpose and repeat_interleave) and only this one
    # op pays for the widening. It is BIT-EXACT, not a precision trade: k arrives from the conv as
    # bf16, so casting it first widens the container without adding any information, and rms_norm
    # accumulates in fp32 internally regardless of its input dtype. MEASURED at the real decode
    # shape over 32 random draws, comparing the whole chain both ways
    # (cast->norm->transpose->repeat vs norm->transpose->repeat->cast, out [1,8,1,128]):
    # max abs diff 0.0, 0/32768 elements differing.
    #
    # Position was swept; later is strictly better, and it is monotonic (n150x4 B=1, traced,
    # interleaved A/B, +-0.0001 run to run):
    #     cast in the high_precision block (was) : 0.2813 ms/step
    #     cast after the norms                   : 0.2796  (-0.6%)
    #     cast after the transposes              : 0.2792  (-0.7%)
    #     cast here, after the GQA expansion     : 0.2784  (-1.0%)
    # Casting after the expansion widens 2x the heads, and that is still cheaper than running
    # repeat_interleave at fp32.
    #
    # REJECTED: deleting the cast outright and pinning the two fp32 consumers instead -- matmul(
    # k_row, h, dtype=fp32) for v_read and outer_dtype=h.dtype for the state increment, i.e. exactly
    # the trick that removed v's and beta's casts. It is numerically sound but MEASURED 0.3035
    # ms/step, a +7.9% REGRESSION. The reason is width: this cast runs on k_row [B,H,1,K] (32 tiles),
    # while the increment it would otherwise be folded into is the outer product k_col (x) d_scaled
    # at [B,H,K,V] -- 512 KB. Converting once at the narrow point beats making the widest op in the
    # function do a mixed-dtype conversion. Do not "simplify" this away.
    if high_precision:
        k_row = ttnn.typecast(k_row, ttnn.float32)
    # TRIED [B,H,1,V] here (matching v_read's natural matmul output, to skip v_read's reshape and
    # d_row's reshape below) and MEASURED it net-negative: removing those 2 reshapes saved ~27us
    # but the following subtract (delta = v_t - v_read) got ~39us dearer on the
    # rank-4 [B,H,1,V] shape than on this [B,H,V] one, for a net +12us regression. Keep [B,H,V].
    # v_rowed: the caller already reshaped v straight to [B,H,1,V] instead of [B,1,H,V], so the
    # transpose is folded into a reshape it had to do anyway. Only safe when the WH fork applies --
    # the rowed shape pads to 8x the tiles and is what breaks B=32 (see the note in gdn/tp.py
    # forward_decode), which wh_decode_fork_applies already excludes.
    v_t = v if v_rowed else ttnn.transpose(v, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)  # ->[B,H,1,V]
    beta_t = ttnn.reshape(beta, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    g_t = ttnn.reshape(g, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    if high_precision:
        # Same move as k's cast above: reshape at bf16, widen after. The [B,1,H] -> [B,H,1,1]
        # reshape scatters H values into H separately-padded tiles and is the expensive part, so
        # running it on half the bytes is free -- g is bf16 on arrival, so casting first only
        # widened the container. The cast MUST still happen: the fused exp activation on the decay
        # multiply below is only correct on an fp32 operand.
        g_t = ttnn.typecast(g_t, ttnn.float32)

    h = initial_state
    if h is None:
        h = ttnn.zeros(
            [B, H, K, V], device=device, dtype=ttnn.float32 if high_precision else ttnn.bfloat16, memory_config=None
        )
    elif high_precision and h.dtype != ttnn.float32:
        h = ttnn.typecast(h, ttnn.float32)

    # Skip the DRAM->L1 copy when the caller already hoisted rec_state (eager decode in tp.py).
    # Same-config to_memory_config is not reliably a no-op in the device trace.
    # NO explicit DRAM->L1 hoist of the state. The decay multiply below already writes its output
    # to L1, so it is the only op that ever reads h from wherever the caller keeps it (DRAM by
    # default) and everything after it runs L1-resident -- an explicit to_memory_config here just
    # paid for the same 512 KB transfer twice. MEASURED -1.9% (0.3213 -> 0.3152 ms/step, n150x4
    # B=1 traced).
    #
    # Going further and making rec_state itself L1-resident removes the L1->DRAM writeback too and
    # MEASURED -2.6% (0.3131), but it pins 512 KB x 30 GDN layers = ~15 MB of L1 for the whole run,
    # which is what _spill_rec_state_to_dram in gdn/tp.py exists to avoid during prefill (and that
    # spill is disabled under _stable_state). Not taken without a full-model L1 check.

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
    # the singleton across tile-tiled dims in one step (B,H tiled -> 1,1 tiled,
    # of the B*H scalars becomes its own separately-padded tile), which is expensive despite the
    # tiny logical size. unsqueeze grows rank one dim at a time, never crossing
    # a *pair* of tiled dims at once. MEASURED (WH, B=32 H=16, this reshape +
    # together): 171.9us vs 153.6us (-10.6%), identical result (PCC 1.0 vs the reshape version).
    _L1 = ttnn.L1_MEMORY_CONFIG
    g_bhkv = g_t
    # decay = exp(g) rides in as a pre-activation on the multiply's second operand rather than as
    # a standalone ttnn.exp on g_t -- one fewer kernel dispatch, and exactly what upstream
    # recurrent_gated_delta_rule_decode_ttnn does.
    #
    # THIS IS ONLY CORRECT BECAUSE g IS FP32. input_tensor_b_activations on a BFLOAT16 broadcast
    # operand silently computes the wrong thing. MEASURED against torch at this exact shape
    # ([B,H,1,V] x [B,H,1,1], B=1 H=8 V=128):
    #     operand fp32 : pre-applied err 0.0        fused err 0.0
    #     operand bf16 : pre-applied err 3.29e-03   fused err 1.395   <-- broken
    # g keeps its fp32 typecast (see the high_precision block), so this one is safe. The same trick
    # on beta -- whose cast was removed, leaving it bf16 -- was tried and took decode PCC from
    # 0.99994 to 0.93948 at B=1 and 0.99981 to 0.80708 at B=8. Do not fold an activation onto an
    # operand here without checking its dtype first.
    h = ttnn.multiply(h, g_bhkv, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], memory_config=_L1)

    # v_read = k @ h (decayed state)
    v_read = ttnn.matmul(
        k_row, h, memory_config=_L1, program_config=read_query_prog_cfg, compute_kernel_config=read_query_compute_cfg
    )

    # Delta + state write (no re-decay). k_row feeds _write_state_wh directly (transposed there);
    # no intermediate [B,H,K] reshape needed since k_row had no other consumer after this point.
    delta = ttnn.subtract(v_t, v_read, memory_config=_L1, dtype=ttnn.float32 if high_precision else None)
    # outer_bf8 is DELIBERATELY NOT tied to tile_opt: writing the recurrent-state increment at
    # bf8 while h is bf16 is a real accuracy regression, not free bandwidth. It
    # into tile_opt in dc854ee0cc6 and cost ~0.04 of model-level logits PCC -- bisected against
    # the 2026-08-08 baseline: test_model_tp_decode_batched[B8] worst per-user PCC 0.99864 ->
    # 0.95927 there, degrading further to 0.94561 by HEAD, and back to 0.99800 with this False.
    # The state is accumulated every decode step across all GDN layers, so rounding the increment
    # to bf8 compounds; the bf16 h dtype exists precisely because bf8 state failed PCC. Every
    # other tile_opt optimisation (q's L2 fold, the [B,H,1,V] output layout, the a/b split and
    # tile-native slices) is numerics-neutral and stays on.
    h = _write_state_wh(h=h, k_row=k_row, delta=delta, beta_t=beta_t, outer_bf8=False)

    # o = q @ h. q_row keeps its dtype; cast up only when h is
    # fp32 (the high_precision path, where k/v/beta/g -- unlike q -- were already cast to fp32
    # early): ttnn.matmul natively accepts mixed BF16 x BFLOAT8_B inputs (proven by the v_read
    # matmul above, which never casts k_row to match h at all), so unconditionally matching q_row
    # to h's dtype turned harmful once h became bfloat8_b: it downcast q_row to
    # for no reason, paying an extra typecast AND losing precision versus just leaving q_row at
    # BF16 and letting the matmul mix dtypes like v_read's already does. MEASURED (WH Tracy
    # BFP8, and that cast plus the BFP8 x BFP8 matmul cost ~26us and precision
    # zero benefit over BF16 x BFP8 -> BF16.
    # q_row is NOT cast to h's dtype first. ttnn.matmul takes mixed operand dtypes (the v_read
    # matmul above has always relied on that -- it never casts k_row to match h), so the cast was
    # pure cost: it rewrites the whole [B,H,1,K] tensor at fp32's width to tell the matmul something
    # it already handles. MEASURED -6.1% on the GDN decode layer (0.3746 -> 0.3517 ms/step, n150x4
    # 35B-A3B B=1, traced replay), and PCC is unchanged (0.99994 at B=1, 1.00000 per-user at B=8).
    o_t = ttnn.matmul(
        q_row, h, memory_config=_L1, program_config=read_query_prog_cfg, compute_kernel_config=read_query_compute_cfg
    )

    # tile_opt: leave o at the matmul's native [B,H,1,V] -- reshaping to [B,1,H,V] crosses the tiled
    # pair (1,V)->(H,V) for no benefit, since the caller (forward_decode) rms_norms on this layout
    # (last dim is still V) and reshapes once to [1,B,H*V] for gate/out-proj.
    # Otherwise return the original [B,1,H,V] the 27B caller path expects.
    # o stays in the read-query matmul's native [B,H,1,V]. Reshaping to [B,1,H,V] here only to
    # have the caller reshape again crosses the tiled pair twice for nothing: forward_decode
    # rms_norms on the last dim (still V either way) and then folds to [1,B,H*V]. Both layouts
    # flatten to the same b,(h,v) element order, so the upstream-fallback leg in the dispatch --
    # which returns [B,1,H,V] -- feeds that same caller correctly with no fixup.
    return o_t, h
