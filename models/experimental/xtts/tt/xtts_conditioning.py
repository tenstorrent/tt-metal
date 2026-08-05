# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the XTTS-v2 audio conditioning path.

Mirrors ``reference/xtts_conditioning.py``: ``ConditioningEncoder`` (init conv +
6 attention blocks) followed by ``PerceiverResampler`` (32 latents, depth 2),
producing the GPT conditioning latents ``[b, 1024, 32]`` from a mel ``[b, 80, s]``.

Everything runs in ``[batch, seq, channels]`` (tokens x channels) layout so the
per-timestep ``Conv1d(k=1)`` layers become plain ``ttnn.linear`` and attention is
``ttnn.transformer.scaled_dot_product_attention``. Key equivalences used:

  * ConditioningEncoder QKV scale ``1/sqrt(sqrt(ch))`` on both q and k == the
    standard ``1/sqrt(head_dim)`` SDPA scale (default, non-causal).
  * The perceiver's ``F.normalize(x, dim=-1) * sqrt(dim) * gamma`` == ``ttnn.rms_norm``.

GroupNorm(32, 1024) is computed manually (see ``_group_norm``): free of any reshape OR
transpose, using a block-diagonal group-averaging matmul.

PERF NOTES (device time per pass, mel s=269, blackhole). The head-split/merge plumbing used to
dominate: a ``reshape`` splitting the LAST dim (``[1,s,3072] -> [1,s,16,192]``) is not a view — it
untilizes + retilizes the whole activation (189 us/block) — and the ``permute`` + last-dim
``reshape`` merge cost another 101 us/block. Both are gone:

  * The per-head-interleaved checkpoint qkv layout ``[h0:q,k,v | h1:q,k,v | ...]`` is repermuted ONCE
    on host (``_perm_qkv_out``) into the standard ``[q_all | k_all | v_all]`` head-major blocks, so
    the fused ``ttnn.experimental.nlp_create_qkv_heads`` (one op: split + per-head reshape) and
    ``ttnn.transformer.concatenate_heads`` (one op: permute + merge) replace 1 reshape + 3 slices +
    4 permutes + 1 reshape per block. Only a leading-singleton ``reshape`` remains, which IS a view.
  * The perceiver cross-attention fuses ``to_q``/``to_kv`` into ONE ``[1024, 1536]`` weight applied
    to ``[latents ; context]``, so it too goes through ``nlp_create_qkv_heads``; the latents' Q is
    then a tile-aligned ROW slice (rows 0..32). The fused matmul reads the same weight bytes as the
    two it replaces, so it is not slower.
  * The perceiver GEGLU no longer slices ``[1, 32, 5460]`` in half — an offset-2730 slice is not
    tile-aligned, so it forced an untilize/retilize. The single ``ff.0`` weight is split on host into
    the value/gate halves (two matmuls, same total weight bytes), and the gate's GELU is fused into
    its matmul epilogue via ``self._ff_gate_mm``'s ``fused_activation`` (exact erf GELU, matching
    ``F.gelu``). Passing ``activation="gelu"`` to ``ttnn.linear`` does NOT fuse it — see below.
  * The group norm no longer permutes into a channels-first ``[1, 1024, s]`` layout to reduce over
    seq. ``ttnn.mean`` over ``dim=-2`` is a native H reduction (``reduce_impl``'s ``single_reduce_op``
    covers ``rank-1`` and ``rank-2``, so only OTHER dims get a transpose injected under the hood), so
    the reduction runs in place on ``[1, s, 1024]`` and the two ``permute``s per block are gone —
    12 of the pass's 14 ``TransposeDeviceOperation``s. Traced device time (mel s=269, blackhole):
    Transpose 32.5 -> 3.0 us, and FillPad 94.7 -> 59.1 us as a side effect, because the reduce's
    ``fill_implicit_tile_padding`` now writes 19 WHOLE rows of 1024 instead of a 19-column stripe
    down 1024 rows. Those two are worth -65 us but the row layout makes the group norm's own work
    slightly dearer (its expand matmuls +17 us, the H-broadcast eltwise +21 us), so the pass nets
    2228 -> 2201 us, -1.2%; traced wall 2.31 -> 2.26 ms. Eager is unchanged (12.6 ms either way) —
    12 fewer launches out of 183 does not move it. It also HELPS accuracy: over a mel-length sweep
    vs the fp32 reference, min PCC 0.9922 -> 0.9938, mean 0.9963 -> 0.9973.
  * The INIT matmul ([1,s,80] @ [80,1024]) has a PINNED program config, ``_mm_2d``: ttnn's auto
    choice was a 32-core 1D config at 13 us, the 2D 8x9 grid does it on 72 cores in 5 us, and the
    bias folds into the matmul epilogue instead of a separate ``BinaryNg`` (-3 us more): 2201 ->
    2190 us. See ``_mm_2d`` for the config sweep and ``INIT_KERNEL_CONFIG`` for why this one op
    wants fp32 dest accumulation once its program config is pinned.
  * The two group-norm EXPAND matmuls ([1,1024] x [1024,1024], 12 per pass) have a pinned config too,
    ``self._gn_mm``. This one is a gemv whose cost is purely the 2 MB DRAM read of E, and the lever is
    ``in0_block_w``, not cores: 16 -> 10 us each, 217 GB/s instead of 130 (42% of DRAM peak, up from
    25%). Biggest single win in the pass: 2190 -> 2103 us, -4.0%, and bit-identical numerically.
  * The AttentionBlock's qkv and proj matmuls are pinned too (``_attn_pcs`` / ``ATTN_*``): qkv 49 -> 47
    us on 108 cores, proj 49 -> 21 us on 99 (it was on 32), and the bias folds into both epilogues
    so 12 full-size ``BinaryNg`` ops per pass disappear. 2103 -> 1832 us, -12.9%, and the worst-case
    latents PCC IMPROVES 0.9939 -> 0.9954.
  * Those same two matmuls then went to HiFi2 (``ATTN_KERNEL_CONFIG``), which is where the rest of the
    win is: they are MATH-bound, not bandwidth-bound, so halving the math passes takes qkv 46 -> 30 us
    (39 -> 61 TFLOPs) and proj 20 -> 14 us. 1832 -> 1691 us, -7.7%. This one is a genuine TRADE, not a
    free win — worst-case latents PCC 0.9954 -> 0.9926 and ECAPA2 speaker similarity 0.757 -> 0.748.
    To revert it, set ATTN_KERNEL_CONFIG back to HiFi4 and the two IBWs to 2.
  * The perceiver's ``concat([latents ; context])`` was silently a ROW-MAJOR round trip, and it was the
    single most expensive thing left in the pass. ``ttnn.concat``'s tiled kernel copies whole tiles, so
    a mid-tile concat dim (context is 288[259]) makes it untilize EVERY input, concat in row major and
    retilize — four ops, 162 us/pass, three of them Mt-bound onto 1, 9 and 10 cores. ``_tile_concat``
    relabels the concat dim to its own padded height with two metadata-only ``reshape``s, so the tiled
    kernel applies: 1658 -> 1496 us/pass, -9.8%, and ``TilizeWithValPadding`` (10 cores, 76 us) plus
    both ``UntilizeWithUnpadding``s (1 core 44 us, 9 cores 42 us) leave the profile entirely. Every
    other op in the pass moves by <= 1.2 us. See ``_tile_concat`` for why it is numerically a no-op.
  * The PERCEIVER's fused qkv matmul is pinned to a 2D config (``PERC_QKV_GX``): 48 cores -> 120, and
    54.0 -> 23.6 us per call, so 108 -> 47 us/pass. 1496 -> 1437 us/pass, -4.0%, PCC unchanged. It was
    the only remaining matmul in the pass with an M axis worth spreading over (Mt=10 vs Mt=1 for every
    other perceiver matmul), and ttnn's auto choice was a 1D config that can use at most Nt=48 cores.
  * Three EPILOGUE folds, all aimed at ops that were nearly pure launch overhead: 1436 -> 1384 us/pass,
    -3.7%, and 152 -> 140 device ops. Two of them are ``ttnn.linear`` traps worth knowing:
      - ``activation=`` DOES NOT FUSE unless the grid is also pinned. ``matmul.cpp`` applies it as a
        separate ``unary_chain`` op when neither ``core_grid`` nor a program config is given, so the
        GEGLU gate's "fused" GELU was really a standalone 3.57 us ``UnaryDeviceOperation`` on
        [1, 32, 2752], twice a pass. Moving GELU into the program config's ``fused_activation``
        (``self._ff_gate_mm``) removes it. Do NOT pass both.
      - ``bias=`` only folds into the epilogue when the program config is pinned, exactly as it did for
        the attn matmuls. Pinning the gate AND value GEGLU matmuls therefore removed ~4 more
        ``BinaryNg`` ops per pass, and as a bonus took all four of those matmuls 22.44 -> 17.79 us/call
        (-18.6 us/pass) at ``FF_GATE_IBW`` = 8. (The profiler labels the removed broadcast rows with the
        encoder's [1, 288, 1024] shape rather than the perceiver's, which cannot be literally right;
        the op COUNTS and the mechanism match, the shape column on those rows does not. Totals are
        reproducible: 140 ops and 1383.6 / 1383.9 us across two runs.)
      - GROUP_NORM_EPS rides the variance matmul's bias slot (``self._gn_eps_row``), which removes the
        standalone 2.36 us add, 6x per pass, and is numerically EXACT: PCC unchanged to all 16 digits.
    The ``rsqrt`` after it cannot be folded the same way even though ``fused_activation`` accepts any
    ``UnaryOpType`` at build time — matmul's runtime whitelist is only GELU, GELU_TANH, TANH, SILU,
    RELU6, SIGMOID, HARDSIGMOID, HARDTANH, SELU and SOFTPLUS, so RSQRT throws "Unsupported UnaryOpType
    for fused activation". It stays a standalone op, 13 us/pass on 32 tiles.
  * THE BIG ONE ON THE TABLE, MEASURED AND NOT TAKEN: ``ttnn.group_norm`` replaces this whole manual
    chain and is worth -122 us/pass (-8.8%), but it lands the latents at PCC 0.988, BELOW the 0.99 gate.
    It is a real option if that gate is ever renegotiated, so here are the numbers. The layout objection
    at the top of this file does NOT apply to it: it forms groups along the LAST dim and wants
    ``[N, 1, H*W, C]``, which is our native ``[1, s, 1024]`` plus a leading singleton, so both reshapes
    are free views and the ``[1, 32, 32s]`` reshape never enters. In-model, 1384 -> 1261 us/pass: one
    ``GroupNorm`` op at 38.7 us/call on 72 cores (vs ~59 us for the 10-op chain) and ``FillPad`` (-54),
    ``Reduce`` (-52), ``rsqrt`` (-13), the expand matmuls (-104) and 131 us of ``BinaryNg`` all vanish.
    The accuracy cost is NOT recoverable. The whole knob space was measured, and the reason is simply
    that the op is ~15x less accurate than this chain on this shape. Per-op against fp32 torch, same
    input, both bf16 in and out — and note PCC SATURATES here and hides the gap entirely, so judge these
    by relative L2, not PCC:
                                    PCC          relL2      max abs err
        manual chain (this file)  0.9999949    4.24e-03      0.060
        ttnn.group_norm           0.9999744    6.21e-02      0.361
    Six of those in series is what takes the pass from 0.9956 to 0.988. Every lever tried:
      - fp32 INPUT: unsupported, hard-faults. fp32 output ``dtype``: no change at all (6.208e-02), so
        the error is internal to the kernel, not output rounding.
      - ``use_welford=True`` (there are dedicated welford_groupnorm kernels): 6.21e-02 -> 5.83e-02, a 6%
        improvement, and end to end it was WORSE (0.985/0.974). So this is NOT a one-pass-variance
        stability problem, which was the obvious hypothesis and is wrong.
      - ``num_out_blocks``: 2 and 4 hard-fault; 1 is bit-identical.
      - Applying gamma/beta elementwise afterwards rather than via the op's ``weight``/``bias``: no
        change (0.9884 vs 0.9882). So it is also NOT the coherent-gamma-fold effect described below,
        despite landing on nearly the same number as that experiment.
      - Hand-rolled mask/affine vs the official ``dram_group_norm_params_from_torch`` +
        ``determine_expected_group_norm_dram_grid_size`` (which picks an 8x9 grid): IDENTICAL, so the
        setup was never the problem.
      - It already runs with ``STATS_KERNEL_CONFIG`` (HiFi4 + fp32 dest).
    The conclusion worth keeping: this hand-rolled chain is not merely a layout workaround for a reshape
    ttnn.group_norm would not even need — it is an ACCURACY asset, worth 15x on the norm's own error,
    which is what buys the headroom the HiFi2 attn matmuls then spend.
  * MEASURED AND NOT TAKEN: fusing the group norm's ``rsqrt`` (13 us/pass, 6 ops on a [1, 1, 1024]
    tensor, i.e. almost pure launch overhead). Every fusion route is closed:
      - Matmul ``fused_activation`` rejects RSQRT at runtime (whitelist above).
      - ``ttnn.multiply``'s ``activations=`` is a POST activation, verified numerically: it computes
        ``rsqrt(a*b)``, not ``a*rsqrt(b)`` (err 0.003 vs 1.27 against the two candidates).
      - ``lhs_activations``/``rhs_activations`` exist in ``binary.hpp`` but are NOT bound in Python.
      - ``ttnn.mac``/``ttnn.addcmul`` are COMPOSITES — ``ternary_composite_op.cpp`` implements mac as
        ``add(multiply(a, b), c)``, two device ops, so they cannot merge the gamma/beta pair either.
  * WHY THE REST OF THE LOW-CORE OPS ARE LEFT ALONE. 41% of the remaining pass runs on <= 32 cores, but
    for these the core count is an OUTPUT of the shape, not a knob: ttnn's interleaved factories call
    ``split_work_to_cores`` with one work unit per output TILE ROW, and this path has Mt=9 (s=259 -> 288)
    in the encoder and Mt=1 (32 latents) in the perceiver. Verified per op rather than assumed:
      - ``FillPad`` (54 us, 32 cores) is ALREADY fully parallel — its ``total_work`` counts only BORDER
        tiles, and 1024/32 = 32 of them exist, so it runs 1 tile/core. It is launch-bound, not starved.
      - ``Reduce`` over H (52 us, 32 cores): the output is 1x1024 = 32 tiles and the split is over
        output width, so 32 IS the maximum.
      - The 32-core matmuls (203 us) all have Mt=1, which collapses any 2D grid to a single row, so
        cores = Nt = 32 by construction. They sit at 0.7% FPU util and are pure DRAM reads; the lever
        is ``in0_block_w`` (see ``self._gn_mm``, already at 42% of DRAM peak), never the core count.
      - The ops that DO get all 130 cores are at 0-2.1% FPU util (SDPA 28.6 us, BinaryNg 3.7 us), i.e.
        bandwidth-bound and already at the ceiling. More cores is not the available lever anywhere here;
        fewer ops and fewer bytes is.
  * MEASURED AND NOT TAKEN: pinning the GEGLU output matmul (``ff2``, [1,32,2752] x [2752,1024], 32
    cores, 40 us/call). An eager sweep said 63 -> 46 us, but in-model it measured 40.2 -> 40.3 us — the
    sweep was a strawman, because it omitted the bias and so provoked a worse "auto" choice than the
    model actually gets. Beware this whole class of result: the eager harness has a ~30-40 us dispatch
    floor, which also makes it useless for ``to_out`` (8.8 us/call) and ``ff_val`` (22.4 us/call) —
    its "auto" for ff_val was 92 us against the model's real 22.4. Confirm every candidate in-model
    with the profiler before pinning it. (``in0_block_w`` must also DIVIDE Kt, or the op hard-faults:
    Kt=86 for ff2 admits only 1, 2, 43, 86.)
  * MEASURED AND NOT TAKEN: removing the perceiver's Q ``ttnn.slice``. It is the pass's ONLY slice and
    costs 1.18 us/call, 2.36 us/pass = 0.16% — 16 tiles on 16 cores, i.e. already 1 tile/core, so the
    op itself has nothing left to give. What it hints at is real but unreachable: the fused qkv matmul
    computes Q for all 320 rows when only the 32 latent rows are used, and ``nlp_create_qkv_heads``
    then moves all 320 before 90% is discarded (~10 us/pass of wasted data movement). Both escape
    routes are closed:
      - Feeding the two-tensor form ``nlp_create_qkv_heads(q=[1,1,32,512], kv=[1,1,320,1024])`` FATALs.
        ``nlp_create_qkv_heads_device_operation.cpp`` asserts "KV tensor seq_len dim must be same as Q
        tensor seq_len", so Q cannot be narrower than KV. Verified by running it, not just by reading.
      - Splitting the heads for a 32-row Q with ``reshape`` + ``permute`` instead measured 2x the
        slice's cost: the ``[1,32,512] -> [1,32,8,64]`` reshape splits the LAST dim, so it untilizes.
    Same wall on the perceiver's ``NLPConcatHeads`` (5.24 us/call on ONE core): that factory's
    ``num_blocks = ashape[0] * ashape[2] / TILE_HEIGHT`` = 1, because it splits over the SEQ axis only
    and iterates heads per core. Nothing in this family is config-tunable at Mt=1.
  * NOT taken, both measured: SDPA's chunking (see ``_attn_block``) and sharding the head ops.
    ``NlpCreateHeads`` (31 us) and ``NLPConcatHeads`` (11 us) run on only 9 cores because they
    parallelize over M tile-rows and Mt=9, but their sharded factories need the qkv tensor
    width-sharded with num_cores dividing the 16 heads, which the qkv matmul cannot emit without
    collapsing to a 1-row grid; going through an explicit reshard of the 1.77 MB tensor costs more
    than the 42 us it targets. The op also self-documents "1 Head Per Core Max for now".
  * ACTIVATIONS in L1, WEIGHTS in DRAM — no exceptions. Activations stay on-chip end to end so the
    matmuls read input-0 from L1 instead of round-tripping to DRAM; every constant operand, trained
    or generated, stays in DRAM in the input-1 slot, where the matmul's own prefetch covers the
    read. Both L1 weight pins this file used to have were measured and dropped: see ``_mm_2d``
    (init weight, ~1 us) and ``_gn_expand`` (2 MB, no measurable difference at all).
"""

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_gpt_block import HIDDEN_SIZE
from models.experimental.xtts.reference.xtts_conditioning import (
    NUM_ATTN_HEADS,
    NUM_LATENTS,
)

GROUP_NORM_GROUPS = 32
GROUP_NORM_EPS = 1e-5
ENC_HEAD_DIM = HIDDEN_SIZE // NUM_ATTN_HEADS  # 64
PERCEIVER_HEADS = 8
PERCEIVER_HEAD_DIM = 64
PERCEIVER_DEPTH = 2
PERCEIVER_INNER = PERCEIVER_HEADS * PERCEIVER_HEAD_DIM  # 512

L1 = ttnn.L1_MEMORY_CONFIG

# HiFi4 for every matmul in this path. All of them are BANDWIDTH-bound, not FLOP-bound (the perf
# report puts them at 22-74% of DRAM peak but only 3-14% of FLOP peak), which is exactly the case
# where the extra math passes are almost free: measured +0.3 ms/pass eager. It buys real accuracy —
# over a 10-point sweep of mel lengths, mean PCC vs the fp32 reference 0.9954 -> 0.9978 and the
# WORST case 0.9898 -> 0.9953. At HiFi2 the worst case sits below the 0.99 test gate, and a
# conditioning prompt that drifts changes the voice the GPT then generates, so this is not a
# free-accuracy-for-nothing tweak — it is load-bearing.
# NOTE: fp32_dest_acc_en is deliberately OFF here. It halves the tiles per math pass on these large
# matmuls (+0.4 ms) and measured WORSE than plain HiFi4 (mean 0.9973 vs 0.9978).
COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# The group-norm STATISTICS additionally accumulate in fp32. The reference GroupNorm32 deliberately
# computes in fp32 (``super().forward(x.float())``), and these are long reductions — over seq for the
# per-channel stats, then over the 32 channels of a group — where bf16 accumulation was the accuracy
# floor of the whole path (it cost ~0.01 PCC on one sample). The tensors are [1, 1024, 1], so fp32
# dest accumulation is free here, unlike on the big matmuls above.
STATS_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# The INIT matmul gets its own config, and unlike the big matmuls above it DOES want fp32 dest
# accumulation. Pinning its program config (see ``_mm_2d``) changes the K-accumulation/subblock
# rounding, and with plain bf16 dest that cost real end-to-end accuracy: over an 8-point mel sweep
# the worst-case latents PCC fell 0.9938 -> 0.9929. fp32 dest acc on this one op buys it back and
# then some (worst case 0.9939, i.e. better than before the program config was pinned) for ~1 us,
# because K here is only 3 tiles so there is very little to re-run at half the tiles per pass.
# NOTE: fp32 dest acc is NOT a free win in general here — with the AUTO program config it measured
# worst case 0.9884, BELOW the 0.99 gate. It is specifically the pinned config that wants it.
# The two AttentionBlock matmuls (grid width + in0_block_w) and the kernel config they run under.
#
# Chosen on END-TO-END latents PCC, not on the ops' own PCC — the two are different questions. Every
# config below measures 0.99997+ on the matmul alone, but its output feeds 6 blocks and the
# perturbation amplifies unevenly across mel lengths. Over 6 mel lengths (worst / mean), with the
# per-op medians from the speed sweep beside them:
#
#   qkv+proj config              worst    mean    qkv+proj us   verdict
#   auto (both)                  0.9939  0.9969      122.6      the pre-pin baseline
#   ibw4, bf16 dest              0.9893  0.9957       67.1      BELOW the 0.99 gate
#   ibw4, fp32 dest              0.9856  0.9956       67.7      BELOW the 0.99 gate
#   ibw2, bf16 dest              0.9911  0.9944       71.3      passes by only 0.001
#   ibw2, fp32 dest  <-- USED    0.9954  0.9976       74.8      better worst case than baseline
#   ibw1, bf16 dest              0.9960  0.9976       87.1      as accurate, 12 us slower
#   proj pinned only (ibw2)      0.9967  0.9976       84.5      most accurate, 10 us slower
#
# Two cautions before touching these. Pinning EITHER matmul alone passes comfortably (proj-only
# 0.9967, qkv-only 0.9941) while pinning BOTH at ibw4 fails: the two roundings compound, so they
# cannot be tuned independently. And fp32 dest accumulation is what makes the fast setting safe
# (0.9911 -> 0.9954 at ibw2) while costing almost nothing at this shape (+4.9 us combined) — the
# opposite of its effect on COMPUTE_KERNEL_CONFIG's matmuls.
ATTN_QKV_GX, ATTN_QKV_IBW = 12, 4
ATTN_PROJ_GX, ATTN_PROJ_IBW = 13, 4
# The PERCEIVER's fused qkv matmul ([1, 32+s, 1024] x [1024, 1536]) also needs pinning, for the same
# reason the init matmul did: ttnn's auto choice is a 1D width-multicast config, which can only ever
# use Nt = 48 cores, and it lands on 53.7 us/call in-model (59 GB/s, 12% of DRAM peak, ~19 TFLOPs —
# bound by NEITHER, just under-parallelized). Unlike the Mt=1 gemvs elsewhere in the perceiver this
# one has Mt=10, so a 2D grid has a real M axis to spread over. Swept eagerly (30-launch back-to-back
# batches; absolute numbers run high vs the in-model profiler but the ordering held):
#   2D 12x10 =120c ibw=4   37.2   <-- USED
#   2D  8x10 = 80c ibw=2   38.4
#   2D 12x10 =120c ibw=1   40.0
#   2D 13x10 =130c ibw=1   40.4
#   1D  64c      ibw=2     56.2   best 1D, i.e. EVERY 2D config beat every 1D one
#   1D  32c      ibw=4    102.7
# ibw is a weak lever here compared to the init matmul (37-44 us across ibw=1,2,4 at 80-120 cores)
# but it turns sharply bad above it: ibw=8 costs 79 us at 120 cores and ibw=4 costs 97 at 130. gx=12
# beats 13 because 12 divides Nt=48 exactly, the same reason gx=8 won for the init matmul.
PERC_QKV_GX, PERC_QKV_IBW = 12, 4
# in0_block_w for the GEGLU gate matmul (Kt=32, so it must divide 32). See self._ff_gate_mm.
FF_GATE_IBW = 8
# HiFi2, NOT the HiFi4 the rest of this file uses — these two matmuls are MATH-bound, and the
# file-level claim above that "all of them are BANDWIDTH-bound, so the extra math passes are almost
# free" is simply false for these two. Two independent measurements say so:
#   * Weight dtype does nothing. bfp8 halves the weight bytes and bfp4 quarters them, yet qkv only
#     moves 47.9 -> 45.9 -> 44.6 us while achieved bandwidth collapses 131 -> 69 -> 35 GB/s. A
#     bandwidth-bound op would have scaled with the bytes. (The profiler's GB/s is DERIVED from the
#     weight size, so "implied GB/s matches reported GB/s" is circular and proves nothing.)
#   * Fidelity does everything. HiFi4 achieves 30-32 TFLOPs, which is ~100% of the effective HiFi4
#     ceiling (~150 TFLOPs bf16 peak / 4 math passes). HiFi2 halves the passes and nearly doubles
#     throughput to 52 TFLOPs: qkv 52.8 -> 32.4 us, proj 23.1 -> 17.5 us, i.e. -26 us per block and
#     -156 us on the pass.
# fp32 dest accumulation is kept and matters MORE than the multiplier fidelity here: at HiFi2 it is
# worth 0.999992 vs 0.999971 op PCC, and it is free at this shape.
# The cost is real and was measured end to end over 6 mel lengths: worst-case latents PCC
# 0.9954 -> 0.9926, mean 0.9976 -> 0.9966. That still clears the 0.99 gate by 2.6x, and unlike the
# SDPA chunk sizes there is no cliff anywhere nearby (every fidelity/dest/ibw combination tried stayed
# above 0.9918, LoFi included). LoFi was 1.7 us faster still but is not used: its mean is worse
# (0.9957) and a 1-pass multiplier is a bigger step than this path should take for ~3% of a block.
ATTN_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

INIT_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


def _lin(torch_tensor, device):
    """torch [out, in] (or conv [out, in, 1]) -> ttnn linear weight [in, out] on device (DRAM)."""
    w = torch_tensor
    if w.dim() == 3:  # conv1d kernel-1 -> [out, in]
        w = w.squeeze(-1)
    return ttnn.from_torch(
        w.t().contiguous().to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )


def _vec(torch_tensor, device):
    """torch [n] -> ttnn tile [n] on device (bias / affine params)."""
    return ttnn.from_torch(torch_tensor.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)


def _row(torch_tensor, device):
    """torch [n] -> ttnn ``[1, 1, n]`` on device: a per-CHANNEL parameter for the group-norm's
    ``[1, s, 1024]`` (channels-last) layout, broadcast over seq on dim 1."""
    return ttnn.from_torch(
        torch_tensor.reshape(1, 1, -1).to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )


def _perm_qkv_out(t):
    """Reorder the output channels of a ConditioningEncoder qkv weight/bias from the checkpoint's
    per-head-interleaved layout into the ``[q_all | k_all | v_all]`` head-major layout that
    ``nlp_create_qkv_heads`` expects.

    The reference ``QKVAttention`` reads channel ``h*192 + t*64 + d`` as (head h, t in {q,k,v},
    dim d); the fused op wants ``t*1024 + h*64 + d``. ``t`` is the leading axis after the permute,
    so indexing the first (output-channel) dim with it does the whole relabel on host, once."""
    idx = torch.arange(3 * HIDDEN_SIZE).reshape(NUM_ATTN_HEADS, 3, ENC_HEAD_DIM).permute(1, 0, 2).reshape(-1)
    return t[idx]


def _clamp_gx(preferred_gx, grid_x, nt):
    """Fit preferred 2D grid width onto the device (P150 is 11x10; larger BH may use gx=12/13)."""
    max_gx = max(1, min(int(preferred_gx), int(grid_x)))
    for gx in range(max_gx, 0, -1):
        if nt % gx == 0:
            return gx
    return max_gx


def _1d_grid_covering(n_tiles, grid):
    """Smallest (gx, gy) with gx<=grid.x, gy<=grid.y and gx*gy >= n_tiles."""
    max_x, max_y = int(grid.x), int(grid.y)
    best = None
    for gy in range(1, max_y + 1):
        for gx in range(1, max_x + 1):
            cores = gx * gy
            if cores < n_tiles:
                continue
            key = (cores, abs(gx - gy), gx * gy)
            if best is None or key < best[0]:
                best = (key, gx, gy)
    if best is None:
        return max_x, max_y
    return best[1], best[2]


def _mm_2d(grid, mt, kt, nt, gx=8, ibw=None, fp32_acc=False):
    """2D (block) multicast matmul program config for an ``[Mt, Kt] x [Kt, Nt]`` tile shape.

    Why pin one at all: for the tiny INIT matmul (Mt=9, Kt=3, Nt=32) ttnn's auto-selection picks a
    1D width-multicast config on 32 cores; in the model that is 13 us, and this 2D 8x9 grid (72
    cores, one 32-row stripe of M per core row, 4 N-tiles per core column) is 4 us. K is only 3
    tiles, so per-core cost is dominated by moving the [288, 1024] OUTPUT rather than by math —
    spreading the output over more cores is what buys the time, not extra FLOPs.

    Swept in isolation (traced repeat harness, so the absolute numbers run high vs the in-model
    profiler, but the ordering held): auto 19.5, 1D configs 17-56 (best at its own 32 cores), 2D
    4x9 9.6, 2D 11x9 8.1, 2D 13x9 7.8, 2D 8x9 7.6. ``gx=8`` wins because it DIVIDES Nt=32; 11 and
    13 leave their last core column partly idle. Also measured and NOT used: HiFi2 saves ~0.3 us
    (not worth any accuracy on a path where PCC is load-bearing), a DRAM output costs +3.6, in0
    L1-height-sharded reaches 6.1 but only via a single-COLUMN shard grid the upstream ``permute``
    does not produce, so it needs an extra reshard that costs more than the 0.7 us it saves, and
    in0 block-sharded (13.1) / width-sharded weight (7.6) both lost to plain L1 interleaved.
    Holding the WEIGHT in L1 measured 6.8 vs 7.6 and is bit-identical numerically, but weights
    belong in DRAM, so it is not used.

    ``gy`` is chosen per call from the actual mel length: the conditioning module is built once but
    sees several sequence lengths, and per_core_M must cover Mt.

    ``gx`` is clamped to ``grid.x`` (prefer a divisor of ``nt``) so pins work on P150 (11x10)."""
    gx = _clamp_gx(gx, grid.x, nt)
    gy = max(1, min(mt, grid.y))
    per_core_m, per_core_n = -(-mt // gy), -(-nt // gx)
    # out_subblock_h * out_subblock_w must fit the DEST register budget, each dividing its per_core
    # dim. The budget is 8 tiles but HALVES to 4 under fp32_dest_acc_en — pass fp32_acc=True for
    # those or the op fatals ("out_subblock_w 8 times out_subblock_h 1 needs to be at most 4").
    cap = 4 if fp32_acc else 8
    sub_w = max(w for w in range(1, per_core_n + 1) if per_core_n % w == 0 and w <= cap)
    sub_h = max(h for h in range(1, per_core_m + 1) if per_core_m % h == 0 and h * sub_w <= cap)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=min(ibw or kt, kt),
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )


class TtXttsConditioning(LightweightModule):
    def __init__(self, state_dict, device):
        super().__init__()
        self.device = device
        e = "gpt.conditioning_encoder."
        p = "gpt.conditioning_perceiver."

        # Block-diagonal group-averaging matrix E [1024, 1024] (E[c,c'] = 1/cpg iff channels c,c'
        # share a group) used by _group_norm to reduce per-group WITHOUT a reshape to [1,32,32s]
        # (that reshape needed ROW_MAJOR<->TILE conversions = Tilize/Untilize ops every block).
        # It lives in DRAM. It used to be pinned in L1, because back when the group norm ran in the
        # channels-first layout the product was ``E @ cmean``, i.e. E was input-0 (the activation
        # slot), and from DRAM those matmuls only reached ~25% of peak bandwidth. The transpose-free
        # group norm reversed the product to ``cmean @ E``, so E is now input-1 — the weight slot,
        # where the matmul's own prefetch covers the read. Measured either way, the 12 expand
        # matmuls are 16 us each and the pass is 2191 (L1) vs 2190 us (DRAM), so the 2 MB of L1 was
        # buying nothing and is given back to the activations.
        cpg = HIDDEN_SIZE // GROUP_NORM_GROUPS
        e_mat = torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE)
        for gi in range(GROUP_NORM_GROUPS):
            e_mat[gi * cpg : (gi + 1) * cpg, gi * cpg : (gi + 1) * cpg] = 1.0 / cpg
        self._gn_expand = ttnn.from_torch(
            e_mat.reshape(1, HIDDEN_SIZE, HIDDEN_SIZE).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.bfloat16,
        )
        # Program config for the two expand matmuls. Unlike the init matmul this shape never varies
        # with the mel length (always [1, 1024] x [1024, 1024]), so it is built once here.
        #
        # It is a gemv whose cost IS the 2 MB DRAM read of E, and the lever is ``in0_block_w`` — how
        # many K tiles are staged per pass — NOT the core count. Swept with E in DRAM: at ibw=1 every
        # core count lands on ~29 us, at ibw=4 ~10.7 us, at ibw=8 ~9.1 us, then it degrades again
        # (ibw=16 ~10.4, ibw=32 ~12-16). Core count barely registers by comparison: at ibw=8, 32 /
        # 64 / 130 cores are 9.6 / 9.5 / 9.0 us. ttnn's auto choice is ibw=1-equivalent at 16.6 us.
        # 32 cores is chosen because N is 32 tiles, so per_core_N=1 gives every core exactly one
        # output tile with nothing left over.
        self._gn_mm = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),  # 32 cores = 32 output tiles
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        # The GEGLU GATE matmul ([1, 32, 1024] x [1024, 2752]). Its config is pinned ONLY so the GELU
        # actually fuses. Passing ``activation="gelu"`` to ``ttnn.linear`` does NOT fuse it on its own:
        # ``matmul.cpp`` applies the activation as a SEPARATE ``unary_chain`` op whenever no
        # ``core_grid``/program config pins the grid, which showed up as a 3.57 us ``UnaryDeviceOperation``
        # on [1, 32, 2752], twice a pass. Put GELU in the program config's ``fused_activation`` instead
        # and the standalone op disappears. Nt=86 with Mt=1 — was hard-coded 13x7=91 (large BH only);
        # on P150 (11x10) that TT_FATALs, so pick the smallest device-fitting grid covering Nt.
        _ff_nt = 2752 // 32  # 86 output tiles
        _ff_gx, _ff_gy = _1d_grid_covering(_ff_nt, self.device.compute_with_storage_grid_size())
        self._ff_gate_mm = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(_ff_gx, _ff_gy),
            in0_block_w=FF_GATE_IBW,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, False),  # False = exact erf GELU
            mcast_in0=True,
        )
        # The GEGLU VALUE matmul is the same [1024, 2752] shape with no activation, and pinning the gate
        # turned out to be worth ~3 us/call beyond just removing the GELU op, so it gets the same config.
        self._ff_val_mm = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(_ff_gx, _ff_gy),
            in0_block_w=FF_GATE_IBW,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        # GROUP_NORM_EPS as a [1, 1, 1024] bias row, so the VARIANCE expand matmul emits ``var + eps``
        # from its own epilogue instead of paying a standalone ``BinaryNg`` for it. That add cost 2.36 us
        # x 6 = 14 us/pass to add a constant to a [1, 1, 1024] tensor (32 tiles) — essentially all
        # per-op launch overhead, on tiles the matmul is already writing. The bias slot is free because
        # the compute kernel is ``bmm_large_block_zm_fused_bias_activation`` and the config is pinned.
        #
        # The ``rsqrt`` that follows CANNOT join it, though it looks like it should: RSQRT is a valid
        # ``UnaryOpType`` and ``fused_activation`` accepts one, so this builds and then throws at
        # runtime — ``matmul_utilities.hpp``'s ``TT_THROW("Unsupported UnaryOpType for fused
        # activation")``. Matmul's whitelist is only GELU, GELU_TANH, TANH, SILU, RELU6, SIGMOID,
        # HARDSIGMOID, HARDTANH, SELU, SOFTPLUS. So the rsqrt stays a standalone op (13 us/pass).
        # NOTE this row goes in the bias/input-1 slot, so unlike self.latents it stays in DRAM.
        self._gn_eps_row = _vec(torch.full((HIDDEN_SIZE,), GROUP_NORM_EPS), device)

        # --- ConditioningEncoder ---
        self.init_w = _lin(state_dict[e + "init.weight"], device)  # [80 -> 1024]
        self.init_b = _vec(state_dict[e + "init.bias"], device)
        self._grid = device.compute_with_storage_grid_size()
        self._pc_cache = {}  # seq len -> (qkv, proj) program configs; see _attn_pcs
        self._perc_qkv_pc = {}  # 32+s -> perceiver fused-qkv program config; see PERC_QKV_GX

        self.blocks = []
        i = 0
        while (e + f"attn.{i}.qkv.weight") in state_dict:
            self.blocks.append(
                {
                    # group-norm affine as [1, 1, 1024] rows, broadcast over seq on dim 1.
                    "gn_w": _row(state_dict[e + f"attn.{i}.norm.weight"], device),
                    "gn_b": _row(state_dict[e + f"attn.{i}.norm.bias"], device),
                    # qkv output channels relabelled to [q|k|v] head-major for nlp_create_qkv_heads.
                    "qkv_w": _lin(_perm_qkv_out(state_dict[e + f"attn.{i}.qkv.weight"]), device),  # [1024 -> 3072]
                    "qkv_b": _vec(_perm_qkv_out(state_dict[e + f"attn.{i}.qkv.bias"]), device),
                    "proj_w": _lin(state_dict[e + f"attn.{i}.proj_out.weight"], device),  # [1024 -> 1024]
                    "proj_b": _vec(state_dict[e + f"attn.{i}.proj_out.bias"], device),
                }
            )
            i += 1

        # --- PerceiverResampler ---
        # Stored pre-shaped [1, 32, 1024] (host reshape) so the forward never reshapes a weight.
        #
        # DRAM, per the weights-in-DRAM rule, and DELIBERATELY not L1. It was moved to L1 once, because
        # it is input-0 of ``_tile_concat`` where no matmul prefetch hides the read, and the profile
        # showed the pass's two concats as DIFFERENT ops for that reason (perceiver layer 0 reading the
        # latents from DRAM at 2.45 us vs layer 1 reading layer 0's L1 residual at 2.36). But that is
        # worth only 0.09 us/pass, and this tensor is PERSISTENT: 64 KB pinned in L1 for the whole run,
        # still held while the GPT prefills. In the demo that was enough to make the GPT's statically
        # allocated circular buffers clash with it ("Statically allocated circular buffers in program
        # ... clash with L1 buffers on core range [0-0 - 12-9]", thrown from _embed_dev). 0.09 us is not
        # worth a demo that does not run. Keep module-lifetime constants in DRAM; L1 is for activations
        # whose lifetime ends inside the pass.
        self.latents = ttnn.from_torch(
            state_dict[p + "latents"].reshape(1, NUM_LATENTS, HIDDEN_SIZE).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.bfloat16,
        )
        self.layers = []
        for j in range(PERCEIVER_DEPTH):
            # to_q [512, 1024] and to_kv [1024, 1024] fused into ONE [1024 -> 1536] weight whose
            # output blocks are [q | k | v] head-major (to_kv's own halves are already [k | v]) —
            # one matmul over [latents ; context] + one nlp_create_qkv_heads for all three.
            qkv = torch.cat([state_dict[p + f"layers.{j}.0.to_q.weight"], state_dict[p + f"layers.{j}.0.to_kv.weight"]])
            # GEGLU value/gate halves split on host: the fused [1, 32, 5460] output could only be
            # halved with an offset-2730 (non-tile-aligned) slice, which untilizes the tensor.
            ff0_w, ff0_b = state_dict[p + f"layers.{j}.1.0.weight"], state_dict[p + f"layers.{j}.1.0.bias"]
            inner = ff0_w.shape[0] // 2  # 2730
            self.layers.append(
                {
                    "qkv_w": _lin(qkv, device),  # [1024 -> 1536]
                    "to_out": _lin(state_dict[p + f"layers.{j}.0.to_out.weight"], device),  # [512 -> 1024]
                    "ff_val_w": _lin(ff0_w[:inner], device),  # [1024 -> 2730] (GEGLU value half)
                    "ff_val_b": _vec(ff0_b[:inner], device),
                    "ff_gate_w": _lin(ff0_w[inner:], device),  # [1024 -> 2730] (GEGLU gate half)
                    "ff_gate_b": _vec(ff0_b[inner:], device),
                    "ff2_w": _lin(state_dict[p + f"layers.{j}.1.2.weight"], device),  # [2730 -> 1024]
                    "ff2_b": _vec(state_dict[p + f"layers.{j}.1.2.bias"], device),
                }
            )
        self.perc_norm_gamma = _vec(state_dict[p + "norm.gamma"], device)

    # ------------------------------------------------------------------ #
    def _group_norm(self, x, gamma_row, beta_row):
        """GroupNorm(32, 1024) over (channels-in-group, seq). x: [1, s, 1024] -> [1, s, 1024]. Consumes x.

        Reshape-FREE: a full group mean/var is order-independent, so per-group stats == the group
        average of per-channel stats. Compute the per-channel mean over seq, expand to per-group via
        a matmul with the block-diagonal averaging matrix ``self._gn_expand``, and likewise for the
        (centered) variance. Everything stays TILE, so this avoids the old reshape-to-[1,32,32s]
        round trip and its four Tilize/Untilize ops.

        TRANSPOSE-FREE too: it runs in the module's native ``[1, s, 1024]`` layout. Reducing over seq
        there is ``mean(dim=-2)``, which ttnn lowers to a native ``ReduceOpDim::H`` kernel — H and W
        are the two dims ``reduce_impl`` handles without injecting a ``transpose`` — so there is no
        reason to permute to channels-first (``[1, 1024, s]``) just to make seq the last dim. Two
        ``permute``s per block, 12 per pass, dropped for free. The group-averaging matmul flips
        accordingly (``cmean @ E`` instead of ``E @ cmean``), which is exact: E is symmetric.

        gamma/beta are ``[1, 1, 1024]`` rows broadcast over seq on dim 1."""
        cmean = ttnn.mean(
            x, dim=-2, keepdim=True, compute_kernel_config=STATS_KERNEL_CONFIG
        )  # [1, 1, 1024] per-channel mean over seq
        mu = ttnn.matmul(
            cmean,
            self._gn_expand,
            memory_config=L1,
            compute_kernel_config=STATS_KERNEL_CONFIG,
            program_config=self._gn_mm,
        )  # group mean, expanded per channel
        ttnn.deallocate(cmean)
        xc = ttnn.subtract(x, mu, memory_config=L1)  # center by group mean (stable variance)
        ttnn.deallocate(mu)
        sq = ttnn.multiply(xc, xc, memory_config=L1)
        cvar = ttnn.mean(
            sq, dim=-2, keepdim=True, compute_kernel_config=STATS_KERNEL_CONFIG
        )  # [1, 1, 1024] per-channel var
        ttnn.deallocate(sq)
        # Group variance PLUS eps: the expand is the matmul and the +eps rides its bias epilogue, so
        # there is no standalone add. See ``self._gn_eps_row``.
        var = ttnn.linear(
            cvar,
            self._gn_expand,
            bias=self._gn_eps_row,
            memory_config=L1,
            compute_kernel_config=STATS_KERNEL_CONFIG,
            program_config=self._gn_mm,
        )  # [1, 1, 1024] group variance + eps
        ttnn.deallocate(cvar)
        # NOTE: gamma is applied to the ACTIVATION, not folded into this [1, 1, 1024] scale. Folding it
        # (scale = gamma * rsqrt(var+eps)) saves one full-size eltwise op (~48 us/pass) but rounds the
        # per-channel product to bf16 ONCE, so every position of that channel gets the SAME wrong
        # scale — a coherent distortion, where applying gamma elementwise leaves incoherent noise that
        # partly cancels. Measured over 7 real mels: folded min PCC 0.9886 (one input BELOW the 0.99
        # gate) vs 0.9906 unfolded. Not worth 2% of the pass.
        rs = ttnn.rsqrt(var, memory_config=L1)
        ttnn.deallocate(var)
        y = ttnn.multiply(xc, rs, memory_config=L1)
        ttnn.deallocate(xc)
        ttnn.deallocate(rs)
        ttnn.multiply(y, gamma_row, memory_config=L1, output_tensor=y)  # gamma/beta bcast over seq
        ttnn.add(y, beta_row, memory_config=L1, output_tensor=y)
        return y

    def _attn_pcs(self, s):
        """The (qkv, proj) program configs for a sequence length, built once per length.

        Both depend on Mt = ceil(s/32) because per_core_M must cover it, so they cannot be built in
        ``__init__`` (one module instance serves several mel lengths) — but they also must not be
        rebuilt 12x per pass, hence the cache."""
        pcs = self._pc_cache.get(s)
        if pcs is None:
            mt, kt = -(-s // 32), HIDDEN_SIZE // 32
            pcs = (
                _mm_2d(self._grid, mt, kt, 3 * HIDDEN_SIZE // 32, ATTN_QKV_GX, ATTN_QKV_IBW, fp32_acc=True),
                _mm_2d(self._grid, mt, kt, HIDDEN_SIZE // 32, ATTN_PROJ_GX, ATTN_PROJ_IBW, fp32_acc=True),
            )
            self._pc_cache[s] = pcs
        return pcs

    def _attn_block(self, x, blk):
        """One ConditioningEncoder AttentionBlock: y = gn(x); y + proj(attn(qkv(y))). Consumes x."""
        y = self._group_norm(x, blk["gn_w"], blk["gn_b"])  # consumes x
        qkv_pc, proj_pc = self._attn_pcs(y.shape[1])
        # Pinning the program config also folds the bias into the matmul EPILOGUE; under ttnn's auto
        # choice each of these two linears emitted a separate full-size BinaryNg for it (9 us + 5 us
        # per block), so this drops 12 device ops per pass on top of being faster.
        qkv = ttnn.linear(
            y,
            blk["qkv_w"],
            bias=blk["qkv_b"],
            memory_config=L1,
            compute_kernel_config=ATTN_KERNEL_CONFIG,
            program_config=qkv_pc,
        )  # [1, s, 3072] = [q|k|v]
        b, s, _ = qkv.shape
        # Leading-singleton reshape only (a metadata view — nlp_create_qkv_heads wants rank 4);
        # the heads split itself is the fused op, no last-dim reshape / slices / permutes.
        qkv = ttnn.reshape(qkv, (b, 1, s, 3 * HIDDEN_SIZE))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=NUM_ATTN_HEADS, transpose_k_heads=False, memory_config=L1
        )  # each [1, heads, s, head_dim]
        ttnn.deallocate(qkv)
        # SDPA stays on ttnn's DEFAULT chunking on purpose. Pinning q_chunk=64/k_chunk=128 is worth a
        # real -64 us on the pass (29 -> 18 us per block; the create->SDPA->concat chain measured 65 vs
        # 141-183 us), but EVERY pinned chunking measured worse end-to-end than the default: worst-case
        # latents PCC 0.9954 (auto) vs 0.9919 (q128/k288), 0.9915 (q64/k288), 0.9906 (q64/k128), 0.9900
        # (q64/k64, below the gate) — and q288/k288 collapses to 0.79.
        #
        # That collapse is the tell. s is padded to a tile multiple (269 -> 288) and this attention is
        # NON-causal with no mask, so the ~19 garbage key rows in the padding get attended to, and the
        # chunk size decides how that pollution lands. Mel length is whatever the caller's reference
        # audio produces, so a chunking validated on a few lengths is not safe on the rest. Masking the
        # pad (or making s tile-aligned) is the prerequisite for collecting this win.
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, memory_config=L1)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        out = ttnn.transformer.concatenate_heads(attn, memory_config=L1)  # [1, s, 1024] (fused permute + merge)
        ttnn.deallocate(attn)
        h = ttnn.linear(
            out,
            blk["proj_w"],
            bias=blk["proj_b"],
            memory_config=L1,
            compute_kernel_config=ATTN_KERNEL_CONFIG,
            program_config=proj_pc,
        )
        ttnn.deallocate(out)
        res = ttnn.add(y, h, memory_config=L1)  # residual is on the NORMED y (matches the reference)
        ttnn.deallocate(y)
        ttnn.deallocate(h)
        return res

    @staticmethod
    def _tile_concat(latents, context):
        """``concat([latents, context], dim=1)`` without the row-major round trip.

        ``ttnn.concat``'s tiled kernel copies WHOLE TILES, so it refuses any input whose concat dim is
        mid-tile and falls back to ``untilize_with_unpadding`` on EVERY input -> row-major concat ->
        ``tilize_with_val_padding`` (see ``build_untilize_rm_retilize_concat``'s predicate,
        ``logical_shape[dim] != padded_shape[dim]``). ``context`` is 288[259], so the fallback fires,
        and the four ops it expands into were 87 us -- the biggest single item in the pass -- because
        three of them are Mt-bound: the latents' untilize gets ONE core (Mt=1, and it is untilized
        despite already being tile aligned, because the predicate is per-LIST not per-tensor), the
        context's untilize 9, the retilize 10.

        The concat-dim padding is the only obstacle, and it is a LABEL, not data: rows 259..287
        physically exist inside the 9th tile either way. So relabel ``context`` up to logical 288
        (its own padded height), concat on whole tiles, then relabel the 320-row result back down to
        the true logical 291. Both relabels keep the last dim and the padded shape's tile count, so
        they are metadata-only -- no kernel, no copy. Everything downstream sees exactly the shape it
        saw before, so this is numerically a NO-OP: measured max abs err 0.0 over all 291 valid rows.

        The one real behaviour change is the CONTENT of rows 291..319, which now hold the encoder
        output's tile padding instead of the retilize's zeros. That padding is NOT zero (the group
        norm's ``x - mu`` writes ``-mu`` into it), so this was checked end to end rather than argued:
        over 10 mel lengths (2-6 s, two reference samples) both paths give min PCC 0.995580 and mean
        0.997799 vs the fp32 reference — equal to 6 decimal places. SDPA's DEFAULT chunking masks past
        the logical 291, so those rows never reach the softmax. Note this is exactly the property
        ``_attn_block`` warns is lost if you pin the chunk sizes (q288/k288 collapses to 0.79), so if
        SDPA chunking is ever pinned here, re-check this too.
        """
        n_lat, n_ctx = latents.shape[1], context.shape[1]
        ctx_pad = context.padded_shape[1]  # 288 for s=259
        c = context.shape[-1]
        aligned = ttnn.reshape(context, ttnn.Shape([1, ctx_pad, c]), ttnn.Shape([1, ctx_pad, c]))
        cat = ttnn.concat([latents, aligned], dim=1, memory_config=L1)  # [1, 32+288, 1024], whole tiles
        return ttnn.reshape(cat, ttnn.Shape([1, n_lat + n_ctx, c]), ttnn.Shape([1, n_lat + ctx_pad, c]))

    def _perceiver_attn(self, latents, context, layer):
        """Cross-attention: latents attend to [latents ; context].

        One fused matmul over the concatenated sequence gives [q|k|v] for every row, so the heads
        split is a single ``nlp_create_qkv_heads``; the latents' Q is rows 0..NUM_LATENTS of that
        result — a tile-aligned row slice (NUM_LATENTS is 32), not a data shuffle.

        The concat is done on TILE-ALIGNED shapes via two free metadata relabels (see ``_tile_concat``);
        done naively it is the single most expensive thing in the pass."""
        ctx = self._tile_concat(latents, context)  # [1, 32+s, 1024]
        n = ctx.shape[1]
        pc = self._perc_qkv_pc.get(n)
        if pc is None:
            pc = _mm_2d(
                self._grid, -(-n // 32), HIDDEN_SIZE // 32, 3 * PERCEIVER_INNER // 32, PERC_QKV_GX, PERC_QKV_IBW
            )
            self._perc_qkv_pc[n] = pc
        qkv = ttnn.linear(
            ctx, layer["qkv_w"], memory_config=L1, compute_kernel_config=COMPUTE_KERNEL_CONFIG, program_config=pc
        )  # [1, 32+s, 1536]
        ttnn.deallocate(ctx)
        n = qkv.shape[1]
        qkv = ttnn.reshape(qkv, (1, 1, n, 3 * PERCEIVER_INNER))  # leading-singleton view
        q_all, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=PERCEIVER_HEADS, transpose_k_heads=False, memory_config=L1
        )  # each [1, 8, 32+s, 64]
        ttnn.deallocate(qkv)
        q = ttnn.slice(
            q_all, [0, 0, 0, 0], [1, PERCEIVER_HEADS, NUM_LATENTS, PERCEIVER_HEAD_DIM], memory_config=L1
        )  # the latents' rows
        ttnn.deallocate(q_all)
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, memory_config=L1)  # [1,8,32,64]
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        out = ttnn.transformer.concatenate_heads(attn, memory_config=L1)  # [1, 32, 512]
        ttnn.deallocate(attn)
        proj = ttnn.linear(
            out, layer["to_out"], memory_config=L1, compute_kernel_config=COMPUTE_KERNEL_CONFIG
        )  # [1, 32, 1024]
        ttnn.deallocate(out)
        return proj

    def _perceiver_ff(self, x, layer):
        """GEGLU feed-forward. The value/gate halves are separate weights (see __init__), so there is
        no non-tile-aligned half-slice, and the gate's GELU rides the matmul epilogue."""
        val = ttnn.linear(
            x,
            layer["ff_val_w"],
            bias=layer["ff_val_b"],
            memory_config=L1,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG,
            program_config=self._ff_val_mm,
        )  # [1, 32, 2730]
        gate = ttnn.linear(
            x,
            layer["ff_gate_w"],
            bias=layer["ff_gate_b"],
            memory_config=L1,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG,
            program_config=self._ff_gate_mm,
        )  # exact erf GELU, fused in the epilogue via the program config (NOT via activation=)
        h = ttnn.multiply(gate, val, memory_config=L1)
        ttnn.deallocate(gate)
        ttnn.deallocate(val)
        out = ttnn.linear(
            h, layer["ff2_w"], bias=layer["ff2_b"], memory_config=L1, compute_kernel_config=COMPUTE_KERNEL_CONFIG
        )  # [1, 32, 1024]
        ttnn.deallocate(h)
        return out

    # ------------------------------------------------------------------ #
    def mel_to_device(self, mel):
        """Host log-mel ``[1, 80, s]`` -> device bf16 TILE tensor (the ``from_torch`` host->device
        write, kept OUTSIDE any trace capture — writes are fatal inside a trace)."""
        return ttnn.from_torch(
            mel.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16, memory_config=L1
        )

    def forward(self, mel):
        """mel: torch tensor ``[1, 80, s]`` -> conditioning latents ttnn ``[1, 1024, 32]``."""
        return self.forward_dev(self.mel_to_device(mel))

    def forward_dev(self, mel_tt):
        """Trace-compatible: ``mel_tt`` is an already-on-device ``[1, 80, s]`` bf16 tensor (no
        host->device write here), so this can run inside a captured trace. -> ttnn ``[1, 1024, 32]``.
        ``mel_tt`` is the caller's tensor and is left allocated."""
        x = ttnn.permute(mel_tt, (0, 2, 1), memory_config=L1)  # [1, s, 80]
        s = x.shape[1]
        h = ttnn.linear(
            x,
            self.init_w,
            bias=self.init_b,
            memory_config=L1,
            compute_kernel_config=INIT_KERNEL_CONFIG,
            program_config=_mm_2d(self._grid, -(-s // 32), -(-x.shape[2] // 32), HIDDEN_SIZE // 32, fp32_acc=True),
        )  # [1, s, 1024]
        ttnn.deallocate(x)
        x = h

        for blk in self.blocks:
            x = self._attn_block(x, blk)  # consumes x; ConditioningEncoder output [1, s, 1024]

        # PerceiverResampler (self.latents is stored pre-shaped [1, 32, 1024] — never freed here)
        latents = self.latents
        for layer in self.layers:
            attn = self._perceiver_attn(latents, x, layer)
            latents = ttnn.add(attn, latents, memory_config=L1)
            ttnn.deallocate(attn)
            ff = self._perceiver_ff(latents, layer)
            nxt = ttnn.add(ff, latents, memory_config=L1)
            ttnn.deallocate(ff)
            ttnn.deallocate(latents)
            latents = nxt
        ttnn.deallocate(x)
        normed = ttnn.rms_norm(latents, weight=self.perc_norm_gamma, epsilon=1e-12, memory_config=L1)  # [1, 32, 1024]
        ttnn.deallocate(latents)
        out = ttnn.permute(normed, (0, 2, 1), memory_config=L1)  # [1, 1024, 32]
        ttnn.deallocate(normed)
        return out
