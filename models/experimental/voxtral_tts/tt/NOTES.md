# TTNN module notes

Everything the `tt/*.py` modules used to carry inline: the reasoning, the measurements, and
the ideas that were tried and rejected. The code keeps each docstring's summary paragraph
plus `# NOTES.md [id]` pointers.

**Why this file exists.** The four modules were 61% prose -- 2087 lines, only 809 of them
code -- which made them hard to read for the thing they exist for. **Nothing was deleted.**
Every block below moved verbatim, and a check confirms the modules' AST is identical apart
from docstrings.

**How to use it.** Grep an id (`gpt-07`) to jump either direction. Read a whole file's
section to understand a block before changing it -- most entries exist because something was
measured and lost, and the note is the only thing stopping it being retried.

**Scope.** This is per-module detail. `../STATUS.md` is the project record: what ships, what
was rejected, the numbers behind both, and the traps that cost real time.

## Contents

- [`ttnn_voxtral_gpt.py`](#ttnn-voxtral-gptpy) — Block 1 -- 3.4B autoregressive backbone
- [`ttnn_voxtral_flow.py`](#ttnn-voxtral-flowpy) — Block 2 -- 390M flow-matching acoustic transformer
- [`ttnn_voxtral_codec.py`](#ttnn-voxtral-codecpy) — Block 3 -- codec decoder, codes to waveform
- [`ttnn_voxtral_pipeline.py`](#ttnn-voxtral-pipelinepy) — the three blocks wired together

---

## Where the remaining headroom is

Read this before hunting for optimizations, because most of the obvious surface is already closed
and the notes below say why. **RTF is ~0.64-0.69, ~51 ms/frame**, split roughly:

| | ms/frame | state |
|---|---|---|
| Block 1, decode | ~26 | six linears **at the 194 GB/s DRAM ceiling**; dtype is the only lever and it is spent |
| Block 2, 7 Euler steps | ~20 | five weight matmuls **at their 13.4 ms weight-read floor** |
| Block 2, rest | ~2 | semantic head 1.25, host FSQ tail 0.7 |
| Block 3, codec | ~0.2 | 0.4% of wall warm; effectively closed |

**Three levers are genuinely open, all structural:**

1. **Euler steps 7 → 5** ([flow-01]) removes 28% of Block 2's solve outright — the largest single
   win left anywhere, ~5.7 ms/frame. It changes what the model *produces*, so it needs a listening
   pass, not a metric. This is a product call more than an engineering one.
2. **Concurrent requests.** Block 2's 3-token sequence uses 6 of 32 tile rows and nothing inside one
   utterance can fill them (steps are sequential, frames autoregressive). Throughput only — it will
   not move single-utterance RTF.
3. **Two upstream ttnn bugs** worth reporting, neither fixable here: `nlp_create_qkv_heads` has a
   ~97-122 µs floor at 16 GB/s on pure data movement ([flow-10]), worth ~2.5 ms/frame if fixed; and
   the `halo_gather` out-of-range NOC write ([codec-22], [pipe-02]) is still live in ttnn — we only
   stopped calling it.

**Closed by measurement — do not retry without reading the note.** Device tracing (both blocks,
+0.16-6 ms), sdpa in Block 2 (1.147x for 3x the code errors, rejected twice), BFP4 weights (1.139x
for 8.4x the errors *and* only 12% of the time for 47% fewer bytes), lower math fidelity, w1+w3
fusion, DRAM-sharded matmul, L1-resident weights, per-stage codec slab, `nlp_create_qkv_heads`
alternatives, and the CFG-combine / inplace-norm micro-fusions.

**Two rules of thumb this port learned the hard way, both of which paid three times each:**

- **Cost is per-OP, not per-byte, at these sizes.** One single-row slice of a 16 MB tensor costs the
  same 0.38 ms as a six-row slice. A 4 KB slice costs 170 µs. Count launches, not bytes — except in
  Block 1 decode, where bytes really are the speed.
- **Shift the NARROW side.** When you must both shuffle and shrink, shrink first: the codec's output
  projection ([codec-22]), its reflected prefix ([codec-08]), and Block 2's `_trunk` ([flow-14]) all
  won this way, 2-85x less data moved for the same answer.

And a measurement rule, because it has bitten repeatedly: **the decode gate's prompt-to-prompt spread
is 0.45 pp, larger than most changes measured with it.** Gate on all 15 prompts, both arms in one
session, and never compare against a number recorded in another session. See `../STATUS.md` §6.15.

---

## `ttnn_voxtral_gpt.py`

*Block 1 -- 3.4B autoregressive backbone*

### [gpt-01] `module docstring`

```text
"""Voxtral-TTS Block 1: the 3.4B autoregressive backbone, on TTNN.

Ours. It replaced a `models/tt_transformers` wrapper, which is gone -- see git history if you
need it (`tt/ttnn_voxtral_backbone.py`, removed once this beat it on every metric). The rationale
for owning this is in STATUS.md's Block 1 section.

Measured against the fp32 CPU reference on REAL prompts (never random ones -- STATUS.md trap #12),
with `tt_transformers` on the same metric for comparison:

                          ours, SHIPPED   ours all-BFP8   ours all-bf16   tt_transformers
    decode min PCC              0.998969        0.998045        0.999357        0.981
    decode mean worst-sample      0.93%           1.17%           0.86%          -
    decode p90  worst-sample      1.35%           1.75%           1.18%          -
    decode ms/step                31.4            28.9            45.8           48
    prefill PCC, last position  0.999883              -               -        0.999564

Decode rows are 15 prompts x 22 teacher-forced frames, all in one session (STATUS.md 6.16). An
earlier version of this table quoted two-prompt numbers, which this gate cannot support -- its
prompt-to-prompt spread is 0.45 pp. Always run all 15: `tt_gates.py --gate decode`.

There is deliberately no end-to-end WER row: the same code at three seeds spans 0.88-2.06% on that
metric, so it cannot separate two builds. The gate is long-form WER (0.00% over 298 words) plus the
teacher-forced numbers above. STATUS.md 6.7 has the seed table; read it before quoting any WER.

W2 IS THE ONE WEIGHT IN bf16, and as of STATUS.md 6.16 that is an ACCURACY choice, not the hang --
the hang was fixed in 6.13 and BFP8 there is now safe. It costs 0.24 pp of mean worst-sample for
2.5 ms/step, which is 8x the worst trade of the other two, so it stays bf16. wqkv, wo, FF1 and FF3
are all BFP8. See WEIGHT_DTYPE below for the full five-arm table and how to flip it.

The decode gap against tt_transformers (0.9990 vs 0.981) is real and reproduces at every weight
dtype we tried, so it is not ours to explain -- but note it is NOT caused by sdpa_decode, which is
nearly free at these shapes.

Mirrors `voxtral_backbone_ref._layer` op-for-op. Structurally this is Block 2's `_block` plus
three things -- RoPE, a causal mask in PREFILL, and a KV cache -- so what Block 2 already proved
carries over unchanged: the row fold, q/k/v fused into one weight, HiFi4 with fp32_dest_acc_en,
bf16 activations. And so does its central lesson, which decode needed twice: a
BATCHED matmul costs per batch element, so fold whatever you can into ROWS (see `_layer_step`).

DECODE IS BOUND BY WEIGHT BYTES, so the weight dtype is the speed. At the shipped precision the
six linears stream ~3.9 GB per step and measure ~20.3 ms, i.e. 194 GB/s -- the ceiling a plain
interleaved matmul reaches, and hand-tuned matmul program configs measured SLOWER (169 vs 193 GB/s
on wq). There is no layout trick left in them: only fewer BYTES help, and the only weight still in
bf16 is w2, which is held there for accuracy (see WEIGHT_DTYPE). So this line is finished unless
0.24 pp of worst-sample becomes acceptable, and 6.16 argues it should not be.

WHERE THE TIME GOES, per decode frame, steady state on one N150 (measure with
the scratch probe_perf.py harness; the numbers below are case 2, 448 frames):

    six linears, weight streaming     ~20.7 ms   AT THE CEILING -- 194 GB/s, and a plain
                                                 interleaved matmul cannot do better here.
                                                 Tuned matmul program configs are SLOWER
                                                 (169 vs 193 GB/s on wq). Only fewer BYTES help.
    rms_norm x2                         1.1 ms   WIDTH-SHARDED and shipped -- was 6.0 ms, worth
                                                 4.4 ms/frame (STATUS.md 6.9). It was rejected once
                                                 on a max-worst-sample reading of 1.06% -> 1.95%,
                                                 measured against weights we no longer ship; on the
                                                 CURRENT config it is 0.86% -> 0.84% mean. Dropping
                                                 fp32 accumulation is a SEPARATE idea and is still
                                                 closed: 2.4x, model PCC 0.992. See _norm.
    sdpa_decode                         1.8 ms   68 us/layer at pos~200; grows with cache length
    everything else                    ~2.9 ms   qkv heads, rope, 2x cache write, reshapes,
                                                 residual adds -- all of it under 4% now, which
                                                 is the decode-native layout's payoff
    ------------------------------------------
    Block 1 total                      25.7 ms   (Block 2 is now ~23, so Block 1 is still
                                                 the larger half -- see ttnn_voxtral_flow)

    CAVEAT on that total: the breakdown came from probe_perf.py, the gate ladder above reads
    ms/STEP on a host-contended harness, and the pipeline reports ~50 ms/frame for BOTH blocks.
    The three do not reconcile and only the PIPELINE number is safe to quote end-to-end. The
    breakdown's SHARES are what it is for; its absolute total predates w2 returning to bf16
    (+2.5 ms) and has not been re-derived.

So the only remaining lever on Block 1 is WEIGHT BYTES, and it is now capped by ACCURACY rather than
by the hang: every weight that can be halved cheaply has been, and the last one (w2) costs 0.24 pp of
mean worst-sample for 2.5 ms/step -- see WEIGHT_DTYPE. Everything else is small change, and the one
thing that looked like an exception, a width-sharded RMSNorm worth ~5 ms, shipped after all once
re-measured against the config it would actually run in (STATUS.md 6.9); read _norm.

That idea is now spent: wqkv and wo ARE in BFP8, worth 3.32 ms/frame for 0.04 pp of mean
worst-sample. w2 is the only weight left in bf16 -- BFP8 there is safe since 6.13 but costs 0.24 pp
for 2.5 ms (6.16), so the byte lever is finished at any precision worth having.

TWO THINGS THAT ARE NOT LIKE BLOCK 2:

1. ROPE, AND WE PERMUTE THE WEIGHTS TO AVOID THE HARD VERSION. The checkpoint is Mistral-native, so
   q/k are stored for INTERLEAVED-pair rotation (r1,i1,r2,i2,...). Applying that on device means
   shuffling even/odd lanes inside a tile, which is awkward. Instead we permute wq/wk ONCE at load
   time into the half-split layout (r1,r2,...,i1,i2,...) and then apply the easy `rotate_half` form.
   The two are equivalent, and the permute is the same one `scripts/export_backbone_hf` uses and
   asserts bit-exact. Getting this wrong does NOT raise -- it produces fluent nonsense.

2. `n_heads * head_dim` (4096) != `dim` (3072), so wq and wo are NOT square. Anything assuming
   dim throughout will be silently wrong on shapes that happen to broadcast.
"""
```

### [gpt-02] `module level` — Prefill pads its sequence to this. Correctness needs only...

```text
# Prefill pads its sequence to this. Correctness needs only TILE -- nothing here shards the
# sequence across the core grid, which is what forced the tt_transformers path to 256.
#
# WHY PAD AT ALL, given that ttnn handles a sub-tile remainder itself: our prefill builds an
# EXPLICIT [1,1,Sp,Sp] causal mask, and a tile-aligned length keeps mask, scores and softmax from
# disagreeing at a ragged edge. Implementations that use sdpa(is_causal=True) materialise no mask
# and so need no padding at all -- that is what the XTTS-v2 GPT and the ign/voxtral_p150_qb2
# branch both do. Measured here, sdpa prefill costs 0.99988 -> 0.99977 PCC at the LAST position,
# the one value that seeds every decode step, and saves ~30 ms of a ~100 ms prefill. Not worth it;
# see below for why 30 ms is noise.
#
# WHY 128 SPECIFICALLY. Every op in prefill carries the sequence dimension -- the norms, all five
# linears, the head split, RoPE, fill_cache, and the [1,32,Sp,Sp] score tensor -- so each distinct
# Sp is its own set of compiled kernels. The 15 fixture prompts span P=74..357: unpadded that is
# 15 shape-sets, at 128 it is three (128/256/384). Padding also caps the QUADRATIC term at three
# known sizes; the worst case is P=357 -> 384, costing (384/357)^2 = 1.16x on attention.
#
# The repo's shared helper, tt_transformers' get_padded_prefill_len, uses a coarser ladder --
# 128, then 1024, then powers of two. Do not adopt it here: it would send our 357 to 1024, i.e.
# 7x the quadratic work, because it is tuned for LLM serving where prompts vary by orders of
# magnitude. Ours are bounded by a voice preset plus a sentence.
#
# PREFILL COST HAS THREE TIERS, which is why "prefill is slow" and "prefill is free" are both
# quoted in this repo's history:
#     first ever at a shape, cold disk kernel cache      ~6 s
#     first in a process, warm disk / empty program cache ~1.5 s
#     subsequent in the same process                     ~100-146 ms
# A long-lived server pays tier 2 once per shape then tier 3 forever, so prefill settles at ~0.4%
# of a 36 s utterance -- less than Block 3's codec pass. Fewer shapes means fewer tier-1 and
# tier-2 hits, which is the second reason to keep the padding.
```

### [gpt-03] `module level` — DECODE'S INTERMEDIATES LIVE IN L1, not DRAM -- the same...

```text
# DECODE'S INTERMEDIATES LIVE IN L1, not DRAM -- the same finding as Block 2's `_L1`
# (ttnn_voxtral_flow), and it transfers: 26.43 -> 25.53 ms/step for NO accuracy change at all (min
# PCC 0.999850, mean worst-sample 0.85%, p90 1.09% -- byte-identical before and after over 44
# teacher-forced frames). Decode's values are 6-24 KB and each is consumed within an op or two, so
# a DRAM round trip per intermediate is pure latency.
#
#     shipped                          26.43 ms
#     + wo output and residual L1       26.19    1.009x
#     + MLP intermediates (g, u) L1     25.53    1.035x   <- shipped
#
# TWO THINGS THAT DO NOT PAY, so they are not done -- and the MECHANISM, since "L1 is faster than
# DRAM" makes the opposite prediction and is the obvious thing to believe.
#
# A MATMUL DOES NOT CARE WHERE ITS ACTIVATION LIVES, because the weight dominates the bytes. `a` is
# [1,1,4096] bf16 = 8 KB against a 4096x3072 BFP8 wo weight = 12.8 MB, so the activation is 1/1632 of
# the read traffic -- 0.061%. Measured, the wo matmul with `a`:
#     DRAM interleaved (ships)   82.8 us
#     L1 interleaved             81.7 us    1.013x, i.e. inside the noise
#     sharded, straight from sdpa 82.2 us   1.006x
# You cannot win a meaningful fraction of a matmul by relocating 0.06% of its input. Same for w2's
# `u` (18 KB against 54 MB, 0.033%): 1.003x.
#
# AND MOVING A VALUE INTO L1 COSTS MORE THAN THE CONSUMER SAVES. sdpa_decode emits `o` as
# INTERLEAVED DRAM already, so the choice is not "keep it in L1" but "move it to L1":
#     to_memory_config(o, DRAM) + reshape   21.0 us
#     to_memory_config(o, L1)   + reshape   27.3 us    +6.3 us
# Pay 6.3 to save 1.1. That is the 0.999x, and it is a conversion cost, not a property of L1.
#
# SO THE RULE IS: L1 PAYS WHEN THE VALUE IS BORN THERE, never when you move it. `memory_config=_L1`
# on a producing op costs nothing extra -- it writes somewhere else -- and saves a real write+read
# round trip. That is what the 1.009x and 1.035x rows above are. `o` is born in DRAM by an op whose
# output placement we do not set, so there is nothing to win. (sdpa_decode was asked for an L1 output
# directly; see the probe. Even if it obliged, the matmul is indifferent, so the ceiling is ~1 us a
# layer.) In Block 2 the norm output was neutral for the same reason.
#
# THE REDUNDANT `to_memory_config(o, DRAM)` IS NOW GONE. It asked to convert a tensor to the layout
# it already had, and ttnn does not short-circuit that -- it returned a fresh tensor, so the line read
# as a no-op and was not one. Removed: output bit-identical (max abs diff 0.0 over 26 layers) and the
# step measures 23.999 vs 24.000 ms against a 0.025 ms spread, i.e. free.
#
# An ISOLATED pass had said the opposite -- reshape alone 19.67 us against 18.37 for the pair, i.e.
# doing less work costing more -- and that was noise at 8 KB. Same lesson as 6.18 and 6.19: for a
# memory-config change the smallest valid unit of measurement is the whole step.
```

### [gpt-04] `module level` — the decode RMSNorm is NOT sharded on Blackhole (p150 fork)

**THIS ENTRY REVERSED ON BLACKHOLE. STATUS.md §6.39 is the current answer; everything below it
is the N150 record, kept because the contrast is the finding.**

On this p150 the decode norm is the plain interleaved `_norm`, the same op prefill uses.
`_NORM_SHARD`, `_NORM_PRG`, the three `_NORM_GRID_*` constants and `_norm_dec` are deleted.
Width-sharding is worth **−4.381 ms/step** here, against **+4.4 ms/frame** on the N150 — same
code, opposite sign. The isolated op says why, and §6.9's own words survive intact ("the
reshard is the tax, not the reduction"): only the tax made the trip.

```text
                    N150 (6.9/6.18)   p150 (6.39)
    interleaved         115.5 us          63.7 us
    sharded 8x4          54.6 us          93.5 us
```

Two further N150 claims below do not hold here. The core-count curve has **no interior
minimum** on Blackhole — it is monotone, fewer cores is uniformly better, and 96 cores (newly
reachable on a 13x10 grid, impossible on 8x8) is the worst config measured. The divisor rule
IS unchanged and is a property of the tensor, not the chip: `block_w` is tiles-per-core and a
32x3072 tensor is 1 x 96 tiles, so the count must divide 96.

Gated before shipping: decode gate better on every stable column (mean 0.94→0.91%, p90
1.35→1.30%, min PCC 0.999260→0.999302), long-form WER 1 wrong of 894 over three seeds against
0 of 894 — the same `"I am"→"I'm"` contraction §6.9 already accepted at 1/1/0.

---

**N150 RECORD BELOW — historical on this fork.**

```text
# RMSNORM, WIDTH-SHARDED, for the DECODE shape. The interleaved op costs 115 us on a [1,1,3072] row
# -- latency, not arithmetic: one core reduces the whole row with a DRAM round trip either side.
# Sharded it is 44 us including BOTH memory_config moves, i.e. ~4.9 ms/frame over 52 calls.
#
# fp32 accumulation is UNCHANGED, so this is NOT the rejected "drop the compute config" trade.
#
# THE CORE COUNT DOES MATTER, AND THIS PARAGRAPH USED TO SAY OTHERWISE -- see STATUS.md 6.18.
# The old claim was "2/4/8 cores measure 42.4/40.5/44.1 us, flat, so it barely matters", drawn from
# the ISOLATED norm. That measurement is ANTI-CORRELATED with end-to-end: re-swept properly,
#     grid  cores  block_w   isolated   ms/step
#     2x1     2      48       43.5 us    25.53
#     4x1     4      24       43.9 us    24.84
#     8x1     8      12       45.6 us    24.57   <- what used to ship
#     8x2    16       6       48.2 us    24.45
#     8x3    24       4          -        24.42
#     8x4    32       3       54.6 us    24.41   <- ships now, SLOWEST in isolation
#     8x6    48       2          -        24.44   legal, and SLOWER -- the curve turns around
# THE COUNT MUST DIVIDE THE TILE COUNT: 32x3072 is 1 x 96 tiles, a tile is indivisible, and block_w
# IS tiles-per-core. Only divisors of 96 are legal, so 40 (8x5), 56 (8x7) and 64 (8x8) cannot build
# at all -- 96/64 = 1.5. 32 is the measured OPTIMUM, not the largest legal grid: 48 divides evenly
# and loses. `subblock_w` is inert (1/2/3/4 within 0.02 ms; >=6 will not build). And a norm cannot
# be benchmarked alone -- it is ~16 us of reduction inside ~28 us of resharding, and the reshard's
# cost depends on what consumes it next.
#
# THE SECOND RESHARD CANNOT BE DODGED, and both ways of trying are now measured.
#
# Feeding the sharded result straight to the DEFAULT matmul is slower -- 8.94 vs 5.32 ms per 26
# norm+linear pairs -- and the reason is the AXIS. Width-sharding splits the matmul's CONTRACTION
# dimension, so each core can only form a partial sum and the cross-core reduce is full-output-sized
# ([32,6144] x 8). Interleaved, ttnn splits by OUTPUT COLUMNS instead: each core owns its columns
# outright, reads the whole 6 KB activation, and there is nothing to reduce. The same axis that makes
# the NORM fast (it reduces over width, so the cross-core step is 8 scalars) makes the matmul slow.
#
# AND YOU CANNOT SHARD ONLY THE WEIGHT INSTEAD. Width-sharding the WEIGHT splits the OUTPUT columns,
# which is the axis that needs no reduction, so it ought to be the free version -- but ttnn couples
# the two. A width-sharded in1 is accepted by ONLY the DRAM-sharded config, which also requires a
# width-sharded in0; every other config asserts `in1.memory_config().memory_layout() == INTERLEAVED`
# (matmul_device_operation.cpp:1188ff, and the comment at :1199 states the pairing outright). So
# "sharded weight, interleaved activation" is not an expressible combination, and the paired form is
# the one measured below. An L1-RESIDENT weight fails the same assertion, and would be capped anyway:
# Block 1 streams ~3.9 GB/frame against 96 MB of total L1, i.e. 2.4% of the model. The model not
# fitting in L1 is exactly WHY 194 GB/s is the wall.
#
# RE-MEASURED WITH TUNED BLOCKING (STATUS.md 6.28) and the rejection holds: 180.7 us against the
# shipped route's 108.6, i.e. 1.66x SLOWER, even though tuning in0_block_w was worth 1.68x on its own
# (303.7 -> 180.7). It also needs the WEIGHT width-sharded, which is Out of Memory in L1 at 8 cores and
# silently returns 1.4e+14 at 32. Better blocking makes the cross-core reduce cheaper; it cannot remove
# it. Below is the original measurement, which reached the same verdict.
#
# `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` REQUIRES a width-sharded activation, so it
# is the one config that wants what the norm already produces. It BUILDS here -- Block 1's wqkv is
# per_core_N=24 tiles where Block 2's N=9216 overflowed L1 at 36 (STATUS.md) -- and it is still
# slower: 125.4 us against 100.9, or 128.5 charging the output unshard, i.e. 0.72 ms/frame WORSE,
# and not bit-exact (4.9e-04). No mystery: the default path already runs this matmul at ~198 GB/s,
# so there is no bandwidth left for the DRAM-sharded machinery to win back. Closed both ways.
#
# THIS WAS REVERTED ONCE AND THE REVERT WAS WRONG. It was measured while wqkv and wo were still
# bf16, where it read mean worst-sample 0.86% -> 0.92%. On the current weights it reads 0.86% ->
# 0.84% mean and 1.10% -> 1.06% p90, i.e. no cost, reproduced twice. The lesson is that a precision
# change here is not separable from the others -- re-measure against the CURRENT config, never
# against a recorded number.
```

**Grid and blocking: the sweep, and why the core count has a MINIMUM at 32**

```text
 Grid and blocking for the sharded norm. NOTES.md [gpt-04] has the sweep; the short version is
 that `block_w` is NOT a free knob (it is DIM // cores // TILE, so it only moves when the core
 count does), `subblock_w` IS free but inert (1/2/3/4 within 0.02 ms, and >=6 will not build), and
 and the core count has a MINIMUM AT 32, not a monotone trend: 2/4/8/16/24/32/48 cores measure
 25.53/24.84/24.56/24.45/24.42/24.41/24.44 ms/step. The count must DIVIDE the 96 width-tiles of a
 32x3072 tensor, since block_w is tiles-per-core and a tile is indivisible -- so 40, 56 and 64 do
 not build at all (96/64 = 1.5), while 48 is legal and simply loses.
```

### [gpt-05] `module level` — Decode runs in ttnn's DECODE-NATIVE head layout, [1,...

```text
# Decode runs in ttnn's DECODE-NATIVE head layout, [1, batch, heads, head_dim], and these are the
# memory configs those ops demand. At batch 1 the layout lines everything up for free:
# nlp_create_qkv_heads_decode emits q exactly as sdpa_decode wants it and k/v exactly as
# paged_update_cache wants them, already sharded -- so there is no permute, no hand-built shard,
# no cache slice and no attention mask (sdpa_decode bounds the cache with cur_pos instead).
# Worth 6.6 ms/frame over a hand-rolled interior, at the same decode PCC.
#
# THE 8 IS NOT THE CORE COUNT, and above ~6 cores it does not matter. The device has 64 Tensix cores
# (8x8); this uses one row. THE UNIT HERE IS THE HEAD, NOT THE TILE: 6144 = 48 heads x 128 and the
# consumers want whole heads per core, so the count must divide 48 -- which admits
# 1,2,3,4,6,8,12,16,24,48 and rules out 32 at 1.5 heads/core. (The norm's grid must divide 96 TILES
# instead, so 32 is legal there and is in fact its optimum. Different op, different divisor set.)
#
# Re-swept on the shipped config, 5 interleaved rounds, mean ms/step:
#     1c 24.683   2c 24.520   3c 24.470   4c 24.436   6c 24.416
#     8c 24.410 <- ships       12c 24.412  16c 24.419  24c 24.409  48c 24.424
# Everything from 6 up sits inside the 0.020 ms within-config spread, and the output is BIT-IDENTICAL
# at every count -- this shard is pure data placement, with no reduction to reorder. Below 6 it does
# cost: one core is +0.273 ms. The earlier sweep only went down to 8 and recorded a flat 31.36-31.46
# ms, which was true but on a build with w2 in BFP8 and the norm on 8x1.
#
# THE GRID DOES NOT REACH THE CONSUMERS, which is the real reason the count is inert and a stronger
# one than "the total did not move". nlp_create_qkv_heads_decode imposes its OWN output layout:
# feed it 1 / 6 / 8 / 24 / 48 cores and qh, kh and vh all come out as 1 core, shard (32, 128), every
# time. So paged_update_cache and sdpa_decode never see this config. Only the shard fill and the
# split op can be affected by it.
#
# Do NOT try to confirm that by timing the consumer chain in isolation. Feeding it a PRE-COMPUTED
# qkv tensor (instead of the wqkv matmul's fresh output) inverts the answer -- the isolated chain
# says 1 core is 0.157 ms CHEAPER while the 26-layer step says it is 0.273 ms DEARER -- and the
# prefix split produced a NEGATIVE marginal cost for paged_update_cache V, which is proof the method
# has broken down at these sizes. Same trap as the norm: isolating an op changes what you measure.
#
# THIS NOTE USED TO ARGUE THE OPPOSITE ABOUT THE NORM -- that its count "DOES matter: 16.2 us on 8
# cores against 21.2 on 32". Those are ISOLATED norm timings, and STATUS.md 6.18 showed that metric is
# ANTI-CORRELATED with end-to-end: 32 cores is the slowest norm in isolation and the fastest step
# overall. The real contrast is narrower than it looked -- both grids are nearly free; the norm's
# spread end to end is 0.16 ms and this one's is 0.02.
```

**`_QKV_GRID_X` — one number used twice, and the core count is inert**

```text
The core count is inert for speed -- 6 to 48 all land inside a 0.020 ms spread.

One number, used twice: the literal 8 used to appear in BOTH the shard width and the grid, and
changing one without the other yields a silently wrong shard rather than an error. Hence the
single named constant rather than two literals.
```

### [gpt-06] `module level` — WEIGHT PRECISION -- load-bearing for CORRECTNESS, not...

```text
# WEIGHT PRECISION -- load-bearing for CORRECTNESS, not just speed.
#
# BFP8 EVERYWHERE EXCEPT w2. Decode is bandwidth-limited on weight bytes and the six linears run at
# the 194 GB/s DRAM ceiling, so the weight dtype IS the speed; every matrix that can be halved has
# been. Each was measured on its own:
#
#     weights                        min PCC   ms/step   mean WS   p90 WS   hang?
#     bf16 (all)                     0.999357    45.8      0.86%    1.18%    no
#     + BFP8 on FF1, FF3               ~        34.7        -        -       no
#     + BFP8 on wqkv, wo   <- SHIPS  0.998969    31.4      0.93%    1.35%    no
#     + BFP8 on w2 (i.e. all)        0.998045    28.9      1.17%    1.75%    no (fixed)
#
# RE-MEASURED on all 15 prompts x 22 frames in ONE session (STATUS.md 6.16). The older version of
# this table read 0.86% -> 0.86% -> 0.84% and concluded each step was FREE. It was not: the total is
# +0.31 pp mean / +0.57 pp p90, and it was measured on two unrecorded prompts where the gate's own
# prompt spread (0.44 pp) is larger than each increment (~0.1 pp). Priced per ms from all-BFP8:
#
#     revert        mean recovered   ms/step back   pp per ms
#     w2               -0.24 pp          2.5          0.096     <- 77% of the cost, 15% of the win
#     wqkv + wo        -0.04 pp          3.3          0.012
#     FF1 + FF3        -0.04 pp         11.1          0.004     <- 24x better value than w2
#
# The three are additive (0.32 vs a measured 0.31), so w2 IS the accuracy story here: 77% of the
# cost for 15% of the win. It is back in bf16 on that basis -- 2.5 ms/step for 0.24 pp of mean and
# 0.40 of p90 is the worst trade of the three by 8x. If 0.04 pp is ever wanted back on top, revert
# ATTN (3.3 ms), never FF (11.1 ms): they buy identical quality at a third of the price.
#
# W2 IS IN bf16 FOR ACCURACY NOW, NOT BECAUSE IT HANGS. Those were two different eras and conflating
# them will waste someone's week:
#   - It USED to be pinned to bf16 because BFP8 there wedged the card. That is FIXED, and it was
#     never a Block 1 bug: an out-of-range NOC write in ttnn's conv `halo_gather` kernel, on the
#     second execution of the CODEC's output-projection conv. That conv was OUR call, so we stopped
#     making it -- see ttnn_voxtral_codec._graph. 45 utterances clean in BFP8.
#   - So BFP8 here is AVAILABLE and SAFE. It is simply not worth 0.24 pp. Flip it and you get
#     28.9 ms/step and mean/p90 1.17%/1.75%; leave it and you get 31.4 and 0.93%/1.35%.
# STATUS.md 6.12 (diagnosis), 6.13 (fix), 6.16 (the accuracy measurement that put it back).
#
# THE CODEC'S MATMUL PROJECTION MUST STAY even though w2 no longer needs it. It is now FASTER than
# the conv it replaced (3.45 vs 4.29 ms, STATUS.md 6.14) and it dodges a live ttnn bug. It is also
# still the thing standing between w2-in-BFP8 and a wedged card, for anyone who flips the line above.
#
# If a hang ever appears here again: TT_METAL_WATCHER=10 turns it into a clean abort naming the stuck
# kernel instead of a wedged card, and recovery is a board reset -- `tt-smi` is not on PATH, the
# Wormhole build is /home/software/syseng/wh/tt-smi and the command is `-wr 0` (this vintage has no
# plain `-r`). open_device just hangs until you do it. The cheapest repro of the OLD hang, still the
# best test of any change meant to affect it: short gen + 128-bucket codec decode, then two long gens
# that both land in the 512 bucket, the second a pure cache HIT.
#
# Judge any change here on MEAN and P90 worst-sample over ALL 15 PROMPTS, never on max and never on
# two prompts. max is an unstable order statistic (moved 1.28-4.28% non-monotonically), and the
# gate's own prompt-to-prompt spread is 0.45 pp -- larger than most changes measured with it, which
# is exactly how the increments in the table above were once recorded as free. STATUS.md 6.15/6.16.
```

### [gpt-07] `interleaved_to_halfsplit`

```text
    """Mistral-native (interleaved-pair) q/k weight -> half-split layout, so `rotate_half` applies.

    Identical to `scripts/export_backbone_hf.meta_to_hf_permute`, which asserts the round trip
    bit-exact. `t` is torch [out, in]; rows are grouped per head.
    """
```

### [gpt-08] `rope_tables`

```text
    """-> (cos, sin) torch [seq_len, head_dim], each half duplicated for the half-split form.

    `rope_cis` in the reference returns complex [S, d/2] for interleaved pairs; the half-split form
    wants the same angles laid out as [cos(theta), cos(theta)] so one elementwise multiply covers
    both halves.
    """
```

### [gpt-09] `__init__`

```text
        """`state` takes an already-loaded `load_backbone_state` dict. Pass it when the caller
        also needs the fp32 reference weights: that dict is ~13 GB, and loading it twice is the
        difference between comfortable and swapping.

        `max_seq_len` sizes the KV cache: 2 x n_layers x 8 x max_seq x 128 bf16, i.e. 218 MB at
        2048. Pass 0 to skip it (prefill-only harnesses; `step` then raises).
        """
```

### [gpt-10] `__init__` — q, k and v fused into ONE weight: one matmul and one...

```text
                # q, k and v fused into ONE weight: one matmul and one weight stream instead of
                # two, and it is what the decode-native head op expects. Same bytes as the old
                # wq + wkv pair, so this costs no memory.
```

### [gpt-11] `_rope`

```text
        """Half-split RoPE on [1, heads, S, head_dim]: x*cos + rotate_half(x)*sin.

        `rotary_embedding_hf` is the HALF-SPLIT (HF) form, which is the right one HERE only
        because wq/wk were permuted at load; on the unpermuted Mistral-native weights it would be
        silently wrong. `is_decode_mode=False` for both prefill and decode: our decode q is
        [1, heads, 1, d], i.e. the prefill layout at S=1, not the op's batch-major decode layout.

        Measured against an fp32 host rotation this scores the same as the hand-rolled
        slice/neg/concat/mul/mul/add it replaces (0.999996 both, differing by bf16 rounding), and
        it is 7 dispatches fewer per call -- which is what decode is actually bound by.
        """
```

### [gpt-12] `_norm`

```text
        """RMSNorm.

        The compute config is NOT optional. Dropping it makes this op 2.4x faster (48 us against
        115 on a 6 KB tensor, ~3.5 ms/frame over 26 layers) and takes the MODEL from decode PCC
        0.99991 to 0.992, worst sample 1.7% -> 18.9%. Per-op PCC barely moves (0.999993 vs
        0.999996) -- a 3e-6 difference amplified ~100x through 26 layers, because every norm feeds
        the next residual. The fp32 accumulation in the mean-of-squares is load-bearing.

        NOR IS THE SHARDED FORM FREE, which is the less obvious version of the same trap. This op
        costs ~115 us on a 6 KB tensor -- latency, not arithmetic, one core reducing the row with a
        DRAM round trip either side. Width-sharding it over 8 cores WITH fp32 accumulation intact
        makes the norm+linear pair 1.46x (5.32 vs 7.78 ms per 26), which is ~5 ms/frame over 52
        calls, and it looked free: 0.9999973 against this op, and the decode gate barely moved.
        It is not free. Over 24 REAL teacher-forced frames the WORST SAMPLE went 1.06% -> 1.95%
        while PCC stayed flat at 0.99991. Same amplification as above, same reason, and per-op PCC
        hid it -- gate norm changes here on worst-sample against the fp32 reference, never on PCC.

        Teacher-forced is the load-bearing word: both builds see IDENTICAL inputs at every step, so
        no trajectory is involved and the comparison is deterministic. An earlier version of this
        note also cited natural WER moving 0.88% -> 2.06%. THAT PART WAS WORTHLESS -- the same code
        at seeds 0/1/2 spans 0.88-2.06% all by itself (score_quality_set.py, LONGFORM_MIN_WORDS).
        The worst-sample number is the evidence; the WER number was a coin flip.

        Block 2 DOES use the sharded form (ttnn_voxtral_flow._norm) and is fine: 3 layers instead
        of 26, so there is no 100x to amplify into.
        """
```

### [gpt-13] `_attend`

```text
        """PREFILL attention: [1,32,S,128] x [1,8,S,128] -> merged [1,S,4096], `mask` additive.

        Decode does NOT come through here -- it has its own interior in `_layer_step`, because the
        two want opposite things: prefill has S rows and a triangular mask, decode has one row and
        wins by folding the head batch down (see there).

        The scale/mask/softmax is deliberately three ops. `ttnn.scale_mask_softmax_in_place` fuses
        them but READS ONLY ROW 0 OF THE MASK unless `is_causal_mask=True` -- undocumented, and it
        does not raise. On our [1,1,S,S] triangular mask it silently applied row 0 to every row and
        the 1-layer gate fell to 0.517, which reads exactly like a RoPE convention error. Decode
        uses it safely because its mask genuinely has one row. Here it would save ~1.4 ms of a
        prefill that happens once per utterance, which is not worth depending on `is_causal_mask`
        continuing to mean what our mask assumes.
        """
```

### [gpt-14] `_mlp`

```text
        """Residual + SwiGLU over an ALREADY-NORMED `h`. Shared by prefill and decode.

        `h` and `mc` are both passed in rather than decided here, because they are exactly what the
        two paths do differently, and both are load-bearing:
          * `h` -- decode norms width-sharded (`_norm_dec`), prefill interleaved (`_norm`). See
            _norm for why that distinction has bitten twice.
          * `mc` -- decode keeps intermediates in L1 (see _L1, worth 0.9 ms). Prefill cannot: its
            `g` is [1,S,9216] and S reaches 384, i.e. 6.8 MB, so it passes DRAM.

        `activation="silu"` on the FF1 matmul is bit-identical to a separate `ttnn.silu` and one
        dispatch cheaper.
        """
```

### [gpt-15] `_layer`

```text
        """x [1,S,3072] -> same. Pre-norm GQA with RoPE + causal mask, then SwiGLU.

        Rows are already folded (batch 1 here), so every linear reads its weights once. Attention is
        the ONLY row-mixing op: it runs on the unfolded [1, heads, S, d] view. Any future row-mixing
        op must go inside that same window -- see ttnn_voxtral_flow._block.

        `cache` is (k, v) [1,8,MAX,128] and is FILLED here, not read: prefill attends to the k/v it
        just computed, so the cache is purely an output. It receives all S PADDED rows; the garbage
        past the real length is never read back because `step` masks everything above `self.pos`.
        """
```

### [gpt-16] `_layer_step`

```text
        """One decode position. x [1,1,3072] -> same, against `cache` written up to `pos_t`.

        Head layout is [1, batch, heads, head_dim] here, not prefill's [1, heads, seq, head_dim] --
        see _QKV_SHARD for why that makes the whole interior glue-free. `pos_t` is a DEVICE tensor
        because paged_update_cache and sdpa_decode both take the position that way.
        """
```

### [gpt-17] `prefill`

```text
        """embeds torch [1,S,3072] -> hidden torch [1,S,3072], or [1,1,3072] if `last_only`.

        No cache yet (increment 4). `last_only` is what the pipeline actually wants -- Block 2
        only ever sees the final position -- and it keeps the [1,S,3072] readback off the host.

        PADDING IS OURS TO CHOOSE, unlike the `tt_transformers` path which needs a 256-multiple as
        a hard constraint. Correctness needs only TILE; PREFILL_MULTIPLE is set above it to keep
        the kernel-shape count down. Zeros are safe: the causal mask keeps real positions from
        attending to the pad, and the padded QUERY rows are computed-then-discarded (each still
        attends to real keys, so no all-masked row and no NaN).
        """
```

### [gpt-18] `step`

```text
        """embed torch [1,1,3072] (one frame) -> hidden torch [1,1,3072]. Advances self.pos.

        Mirrors `IncrementalBackbone.step`. No causal mask and no cache slice: every cached
        position is in the past, and sdpa_decode reads the whole cache bounded by `pos_t`.
        """
```


---

## `ttnn_voxtral_flow.py`

*Block 2 -- 390M flow-matching acoustic transformer*

### [flow-01] `module docstring`

```text
"""TTNN port of the Voxtral-TTS flow-matching acoustic transformer (BLOCK 2, 390M).

Mirrors reference/voxtral_flow_ref.py op-for-op. Per generated frame:

    h [B,3072] --semantic head--> mask --> argmax                  = semantic code   [B,1]
             \\--7 x Euler step of a 3-layer bidirectional TF -->  = acoustic floats [B,36]
                                        clamp/scale/round (FSQ)    = acoustic codes  [B,36]
    = audio_codes [B,37], offset by N_AUDIO_SPECIAL, ready for Block 1's embed_frame

WHAT MAKES THIS BLOCK UNUSUAL:
  * The sequence is exactly THREE tokens -- [input_proj(x_t), time_proj(t_emb), llm_proj(h)] --
    and the velocity is read off POSITION 0. The other two exist only so attention can mix time
    and LLM conditioning into position 0.
  * Attention is BIDIRECTIONAL and unmasked: no RoPE, no causal mask, despite the GQA 32/8 head
    layout. `rope_theta` in params.json is inert here. Adding RoPE would be silently wrong.
  * CFG is batched, not doubled: cond and uncond (zeroed h) go through as one 2B forward, so a
    step is a batch-2 graph rather than two graphs.
  * Every one of the 7 steps is THE SAME SHAPE. That made the whole solver capturable as one
    device trace; it was correct and bit-identical but ~6 ms/frame SLOWER on this N150, so the
    capture machinery was removed rather than left as a dormant second path.

IT IS NOT WEIGHT-READ BOUND, AND BYTES ARE NOT THE LEVER -- both halves measured. It WAS
weight-bound once, which is why the CFG batch fold in `_trunk` was worth 2.23x (a batched matmul
re-reads the whole weight per batch element, so batch-2 doubled every read; 6 rows still fit one
32-row tile, so folding is free). Where it stands now: the velocity net is 349M params (3 layers x
116.4M) at BFP8, so 7 steps stream ~2.6 GB, a 13.4 ms floor at 194 GB/s.

Two measurements say do NOT plan around that floor. First, the five weight matmuls already sum to
13.28 ms of `_block`'s 19.24 -- they ARE at the floor, 168-205 GB/s each, so no op work touches them.
Second, BFP4 weights cut the bytes 47% (2.60 -> 1.37 GB) and returned only 12% of the time (25.70 ->
22.56 ms) where the model predicts ~6.3 ms. Halving bytes here buys about a quarter of what the
arithmetic promises, unlike Block 1 where dtype really is the speed. STATUS.md 6.17.

THE GAP IS PER-KERNEL COST -- and note the precise wording, because the obvious reading of it is
wrong and was tested. A step is ~88 ops and only 18 carry weights; the rest are reshapes,
transposes, softmaxes, slices and typecasts on tiny tensors. It is tempting to conclude "delete
small ops", and that DOES NOT WORK: fusing the CFG combine from 5 ops to 3 measured 1.001x, and
`inplace=True` on the norm measured 0.997x. Tracing, which removes host dispatch entirely, is
+0.16 ms. The cost is device-side and per KERNEL, and these kernels are already at the floor.

WHAT ACTUALLY WORKS IS FEWER, BIGGER KERNELS. Every win here has that shape: the CFG row fold
(2.23x) and the GQA row fold (1.40x) both made matmuls bigger by folding work into unused tile
rows, and the qkv fusion (0.96 ms) merged a 2048-wide matmul that was costing the same as a
4096-wide one into a single larger call. Judge a proposal on whether it makes kernels BIGGER, not
on how many ops it deletes.

WHERE THE TIME GOES, per frame, steady state on one N150 (~22 ms of a ~51 ms frame; Block 1's
is the larger half):

    _solve -- 7 Euler steps          ~20 ms    7 SEQUENTIAL velocity evaluations, each a 3-layer
                                               transformer over 3 tokens, CFG batch-2 folded to 6
                                               rows. Nominally 1.6x a 13.4 ms weight-read floor,
                                               but BFP4 showed that floor governs less of this than
                                               it looks -- see above.
                                               The biggest single non-matmul line is
                                               nlp_create_qkv_heads at ~97 us x21 -- see _block,
                                               where its floor is measured and shown to be fixed.
    semantic_code                      1.25 ms [B,8320] masked argmax, now ON DEVICE in fp32.
                                               Was 2.74 ms of real host CPU. fp32 is mandatory --
                                               it produces an INDEX; see semantic_dev.
    host tail (FSQ quantise etc)      0.08 ms  MEASURED 81.8 us on the p150 for all
                                               THREE host steps (6.50); the 0.7 ms
                                               here was an N150-era estimate
    ------------------------------------------
    Block 2 total                     ~22 ms   (42.5 before the row fold, sharded norm, qkv
                                               fusion, device semantic head, L1 interior and the
                                               three op-level wins in STATUS.md 6.17)

The structural problem is the SEQUENCE: 7 steps that each depend on the previous, so none of the
usual batching tricks apply within a frame.

TRIED AND REJECTED, so they do not get retried: BFP4 weights (1.139x for 8.4x the differing
codes, 19/576 -> 159/576 over 8 draws -- and only 12% of the time for 47% fewer bytes, see above;
per-weight numbers in STATUS.md 6.17, and w2 alone is 0.016 ms for 2.6x the errors); lower math
fidelity (HiFi2/LoFi save ~4 ms for 10-20x the integer-code errors -- see COMPUTE_CONFIG); sdpa for
the attention interior (1.147x, but codes go 7/288 -> 21/288 over 8 draws -- rejected once before
BFP8 and again after, same answer); the CFG-combine and inplace-norm micro-fusions above; w1+w3
merged into one 18432-wide matmul (0.998x, 0.951x once the output split is charged); and a device
trace, re-measured after all of the above at +0.16 ms (1.006x) with bit-identical codes.
BFP8 weights ARE on and were worth 1.23x.

WHAT LANDED INSTEAD, once the matmuls turned out to be at their floor, was op count: silu fused into
the w1 matmul, SCALE folded into wqkv's q rows, and `_trunk` projecting BEFORE it narrows. Together
+1.19 ms/frame in isolation with accuracy slightly BETTER (velocity PCC 0.9999816 -> 0.9999851,
differing codes 19/576 -> 16/576). End to end the pipeline shows only ~-0.4 ms/frame of that and the
factor of ~3 is unexplained -- quote the pipeline number, see STATUS.md 6.17.

STILL OPEN, and both are structural rather than op-level:
  * FEWER EULER STEPS. 7 -> 5 removes 28% of the solve outright. This changes what the model
    produces, so it needs a listening pass, not a metric.
  * CONCURRENT REQUESTS. The 3-token sequence wastes 26 of 32 tile rows and nothing within one
    utterance can fill them, since the steps are sequential and the frames autoregressive. This is
    throughput, not latency -- it will not move RTF for a single utterance.

That trace result is the load-bearing evidence for the per-kernel framing at the top: tracing
removes host command submission almost entirely, and removing it changes nothing. STATUS.md 6.6
has the table.

HOST vs DEVICE. The whole Euler solve -- the 3-layer transformer, the CFG combine and the state
update -- runs on device, with nothing left in the loop that a trace could not capture. Two things
stay on host, deliberately, and both are once per frame rather than once per step:
  * the semantic argmax -- a [B,8320] masked argmax whose result is an INDEX used to look up an
    embedding on host anyway (same reasoning as the codec's semantic gather).
  * the FSQ quantise -- clamp/scale/round on [B,36]; 36 values per frame is not worth a dispatch.
`time_embedding` is also host code, but it is no longer per step or even per frame: the solver
schedule is fixed, so `_schedule()` builds its projections once and caches them on device.
"""
```

### [flow-02] `module level` — EVERY INTERMEDIATE INSIDE `_block` LIVES IN L1, not DRAM....

```text
# EVERY INTERMEDIATE INSIDE `_block` LIVES IN L1, not DRAM. This is the same finding as the
# L1-resident q/k/v (see _block): at this block's shapes, WHERE a tensor lives matters as much as
# how big the kernel is. Nothing here exceeds ~110 KB ([1,6,9216] bf16), and each value is consumed
# within a few ops of being produced, so a DRAM round trip per intermediate is pure latency.
#
# Measured cumulatively, 8 draws, all at IDENTICAL accuracy (9/288 differing codes throughout):
#
#     q/k/v L1 only (was)                      24.18 ms
#     + attention interior (scores, scaled, av) 23.85    1.014x
#     + MLP intermediates (g, w3_out, u)        23.22    1.041x
#     + residual stream                         23.04    1.049x   <- shipped
#
# The one candidate that does NOT pay is the `_norm` output: 0.999x alone, so it stays DRAM. That
# is worth knowing -- it is not "L1 everywhere is better", it is specifically the values with a
# consumer close behind. And note the LIMIT: a width-SHARDED activation into a DRAM-weight matmul
# is SLOWER (8.94 vs 5.32 ms per 26 norm+linear pairs). Interleaved-L1 is the useful middle.
```

### [flow-03] `module level` — Math fidelity for the VELOCITY NETWORK. HiFi4 + fp32...

```text
# Math fidelity for the VELOCITY NETWORK. HiFi4 + fp32 destination accumulation is the most
# expensive setting available and it STAYS: lowering it is a bad trade at this block's shapes.
#
#     config                    velocity PCC   codes differ   ms/frame
#     HiFi4 fp32acc BFP8 W       0.9999845        4/222         42.57   <- shipped
#     HiFi2 no-fp32acc bf16 W    0.9998382       35/222         48.72
#     LoFi  no-fp32acc bf16 W    0.9992816       62/222         48.59
#
# ~4 ms for 10-20x the integer-code errors. Codes are what reach the audio, which is why the gate
# is "codes differing from the reference frame" and not PCC alone.
#
# This does NOT touch the solver arithmetic. `x` and the CFG combine stay fp32 in decode_frame,
# deliberately: `x` accumulates n_steps increments so its error compounds, and the CFG combine is
# a difference of two nearly-equal vectors where the small difference IS the signal (fp32 2.4e-7
# vs bf16 7.0e-3, ~29,000x).
```

### [flow-04] `module level` — Activation dtype. Every ttnn op here inherits its input's...

```text
# Activation dtype. Every ttnn op here inherits its input's dtype, so this one constant sets all
# activations; matmul accumulation stays fp32 via fp32_dest_acc_en above. Measured at 1.42x
# end-to-end for no quality cost (STATUS.md 3.2).
```

### [flow-05] `module level` — MATMUL weight storage, independent of the activation...

```text
# MATMUL weight storage, independent of the activation dtype. BFP8 is worth 1.23x here for ONE
# extra differing code in 222 (see the table above). That reverses an earlier conclusion in
# STATUS.md, which compared bfp8 WITHOUT the batch fold against bf16 WITH it; measured on top of
# the fold, bfp8 wins.
```

### [flow-06] `module level` — The SEMANTIC head is the one thing here that is not...

```text
# The SEMANTIC head is the one thing here that is not bf16/BFP8, and it is not a free
# choice -- see semantic_dev in __init__. It emits an INDEX, so a rounding difference is
# not a small error, it is a different code.
```

### [flow-07] `module level` — the norm is NOT sharded on Blackhole (p150 fork)

**REVERSED ON BLACKHOLE, exactly as [gpt-04] did. STATUS.md §6.40 is the current answer; the
N150 record is kept below because the contrast is the finding.**

Width-sharding this norm is worth **−4.5 ms/frame** here over its 49 calls, against +1.46x on
the N150. And it is **closer to fp32 truth, not further** — 8 real prompts, vs the fp32 CPU
reference: identical acoustic codes (10/288 both arms), identical semantic (0/8 both), velocity
max-abs 3.233e-02 sharded against **2.569e-02** interleaved.

Two things §6.40 records that are worth carrying: at THREE seeds the WER read 1 vs 4 and looked
like a regression; at six it is 6 vs 4 the other way. And the `codes != 8x4` column that prompted
the scare measures divergence from the shipped config, not error — §6.25's trap.

---

**N150 RECORD BELOW — historical on this fork.**

```text
# RMSNORM, WIDTH-SHARDED. Same finding as Block 1's _NORM_SHARD, and it matters more here: 7 norms
# per Euler step x 7 steps = 49 calls per frame. Interleaved, each costs ~115 us on a [1,6,3072]
# tensor -- latency, not arithmetic, since one core reduces the row with a DRAM round trip either
# side. Spread over 8 cores the norm+linear pair measures 1.46x (5.32 vs 7.78 ms per 26).
#
# fp32 accumulation is UNCHANGED, so this is not the rejected "lower the fidelity" trade above.
# The row count is pinned by the program config, which is fine: every norm in the solve sees the
# same [1, B*3, 3072], tile-padded to 32 rows.
```

**Grid and blocking for Block 2's norm**

```text
 Grid and blocking: swept, see NOTES.md [gpt-04] for the Block 1 version of the same experiment.
 More cores is monotonically faster end to end here too -- 8x1/8x2/8x4 measure 24.628/24.572/24.465
 ms/frame -- and `subblock_w` is INERT: 4 and 1 measure byte-identical at 8x1. 32 cores is the most
 that divides evenly (3072/32 = 96 tiles; 96/64 is not an integer).
```

### [flow-08] `__init__` — SEMANTIC HEAD, ON DEVICE AND IN FP32. This used to be a...

```text
        # SEMANTIC HEAD, ON DEVICE AND IN FP32. This used to be a host matmul -- [1,3072] @
        # [3072,8320] plus a mask and an argmax -- and it measured 2.74 ms per frame of real CPU
        # time, ~4% of the frame. On device it is 1.25 ms, so 1.49 ms/frame.
        #
        # FP32 IS NOT NEGOTIABLE HERE, unlike everywhere else in this module. The result is an
        # INDEX, not a value: two close logits ranked the other way round change the semantic code
        # outright. Measured over 64 hidden states, bf16 weights pick a DIFFERENT index on 4 of
        # them; fp32 matches the host answer on all 64. bf16 would be 1.04 ms -- 0.2 ms more, for
        # a wrong primary code on ~6% of frames.
        #
        # The mask is additive and prebuilt rather than the host version's two -inf assignments:
        # -1e9 underflows exp() to zero the same way, and one add is cheaper than two writes.
        # [EMPTY_AUDIO] is forbidden; [END_AUDIO] is ALLOWED, since that is how generation stops.
```

### [flow-08a] `semantic_code` — the mask and the argmax belong on the HOST, worth 0.353 ms/frame

```text
        SPLIT FIRST (STATUS.md 6.31). One semantic_code call is 1229 us, the second-largest line in
        Block 2's whole map, and it decomposes as:

            linear fp32 against a (3072, 8320) head   562.0 us   45.7%
            argmax over 8320 values                   490.1 us   39.9%
            from_torch H->D                            53.0 us    4.3%
            add semantic_mask                          36.3 us    3.0%
            to_torch D->H                              21.4 us    1.7%

        The matmul is NOT the problem: the head is 3072x8320 fp32 = 102 MB, so 562 us is 182 GB/s,
        which is at the roofline. The ARGMAX is the problem -- 490 us to reduce 33 KB is 0.07 GB/s,
        so it is all launch overhead and none of it is data movement.

        AND IT COSTS NOTHING TO MOVE, because this call ALREADY ended in a device->host copy. Doing
        the reduce on the host does not add a round trip; it only makes that copy 8320 fp32 values
        instead of 1. 33 KB is 0.17 us of PCIe. The mask add rides along for free once the logits
        are on the host.

            A shipped: linear fp32 + device mask + argmax   1213.5 us   1.000x   0 of 8 ids changed
            B host argmax, device mask                       893.5 us   1.386x   0 of 8
            C host argmax AND host mask  <- SHIPPED          860.3 us   1.439x   0 of 8
            D bf16 head, device argmax                       946.9 us   1.308x   0 of 8
            E bf16 head + host argmax + host mask            595.7 us   2.079x   0 of 8

        C is EXACTLY the same fp32 arithmetic -- same values, same order, the add and the reduce
        just happen on a different processor -- so it carries no numerical risk at all.

        E IS FASTER STILL (2.079x, another 0.265 ms/frame) AND IS NOT SHIPPED. Halving the head to
        bf16 changes the logits, and this is an ARGMAX over a vocabulary: what decides whether a
        token flips is the top-2 GAP, not a norm. 0 of 8 real prompts moved, but 8 single frames is
        a thin sample for a discrete decision, and the semantic token is the highest-stakes integer
        in the model -- it feeds Block 1's next input embedding, so ONE flip redirects the entire
        remaining generation. Needs a broad real-prompt gate before it can ship.

        A NOTE ON HOW THIS WAS NEARLY GOT WRONG. Round 1 of this probe scored the bf16 candidate on
        64 RANDOM gaussian draws and reported 0 changed ids. That is worthless -- STATUS trap #12
        records random embeddings reading PCC 0.892 where real prompts gave 0.9994, and an argmax is
        precisely where that bites. Round 2 pulls real hidden states out of Block 1 on the fixture
        prompts. ALWAYS GATE ON REAL PROMPTS.
```

### [flow-09] `__init__` — q, k and v fused into ONE weight -> one matmul instead of...

```text
                # q, k and v fused into ONE weight -> one matmul instead of three. torch stores
                # Linear as [out, in], so concatenate along dim 0 before the transpose in `lin`.
                #
                # Fusing q in as well (it used to be its own matmul alongside a fused kv) is worth
                # 1.449x on this pair -- 2.13 against 3.09 ms per frame -- and it is the same
                # arithmetic: a linear computes each output column independently, and 4096 is a
                # multiple of the 32-wide tile, so the BFP8 blocks are unchanged too.
                #
                # WHY IT WINS, because it says where the next one is: `wkv` alone is only 2048
                # wide and measured 73 us, the SAME as the 4096-wide `wq`. There is a fixed cost
                # of roughly 40-50 us per matmul launch at these shapes, and width past ~4096 is
                # nearly free. Fusing pays exactly when one of the pair is too narrow to earn its
                # launch. It does NOT pay otherwise: w1+w3 are 9216 each, and merging them into
                # 18432 measured 0.998x -- no gain -- and 0.951x once the output split is charged.
                #
                # THE 1/sqrt(head_dim) SCALE IS FOLDED IN HERE, into the q rows only. It used to be
                # a `multiply(s, SCALE)` on the scores, once per block call -- 21 op launches a
                # frame for one constant. Folding it is worth 0.28 ms/frame AND slightly more
                # accurate: the scores are rounded once instead of twice, and it took differing
                # acoustic codes from 19/576 to 16/576 with velocity PCC 0.9999816 -> 0.9999851.
                # Not bit-identical (SCALE is not a power of two, so the BFP8 mantissas in those
                # columns change), which is exactly why the code counts were measured.
```

### [flow-10] `_block` — the FUSED head split ships on Blackhole; §6.31 reverses

**REVERSED. STATUS.md §6.45.** `nlp_create_qkv_heads` replaces the 9-op hand-rolled split, worth
> **READ THE DENOMINATOR (STATUS.md §6.54).** Every `n/288` in this file is **8 real prompts ×
> 36 codes**. `--gate codes` prints an identically-shaped `n/288` from **synthetic** embeddings
> and reads ~6× worse (85/288 vs ~11/288) — different measurement, same units. Do not compare one
> to the other; that mismatch cost a session's worth of doubt. Real prompts are 100% off-by-one on
> a 21-level FSQ axis; only synthetic input ever produces |delta| > 1.

**+3.836 ms/frame** at **identical accuracy** (10/288 acoustic codes vs the fp32 reference, 0/8
semantic, velocity maxabs 2.569e-02 and PCC 0.99998504 — the same numbers, not merely close).

§6.33 already established the hand-rolled win was a SYSTEM EFFECT and not the op being faster.
A system effect is exactly what does not travel between chips, and it did not: a small op costs
3.4x more on the p150 (67.7 us against ~20), so 9 ops for 1 is now the wrong trade.

---

**N150 RECORD BELOW — historical on this fork.**

### [flow-10-n150] HAND-ROLLED HEAD SPLIT, worth 1.233 ms/frame over `nlp_create_qkv_heads`

```text
        # HAND-ROLLED HEAD SPLIT -- 3 slice + 3 reshape + 3 permute, replacing one
        # nlp_create_qkv_heads. Worth 1.233 ms/frame (22.375 -> 21.142), BYTE-IDENTICAL codes.
        #
        # The block of prose below is the ORIGINAL finding, which rejected exactly this
        # restructuring at 158 us. It was re-measured (STATUS.md 6.30/6.31) and is WRONG for the
        # spelling here, though the L1 lesson in it is not:
        #
        #   * every output must be forced to _L1. That is the whole difference. Left at the default
        #     the three tensors land in DRAM, measure the same 122 us as the fused op, and cost the
        #     downstream consumers the 2.5 ms/frame the original finding correctly identified.
        #   * ISOLATED, THE FUSED OP IS NOT SLOWER. Settled at 500 reps x 6 alternating rounds
        #     (STATUS.md 6.33): fused 122.3 us with a 0.2 us spread, manual 125.7 us with a 5.8 us
        #     spread. An earlier reading of 112.4 was a low sample from that wide distribution. The
        #     effect (~10 us) is smaller than the noise of the thing measuring it, so isolation
        #     CANNOT decide this in either direction.
        #   * ON THE WHOLE BLOCK it wins 1.233 ms/frame -- 22.375 -> 21.142, 1.0586/1.0585/1.0578
        #     over three interleaved 200-frame rounds, spread 0.0008 on a 0.058 effect. So the win
        #     is ENTIRELY a system effect around the op, not the op. Candidate mechanisms, none
        #     verified: the fused path's 4D reshape builds a 768 KB intermediate ([1,6,6144] pads to
        #     ONE 32-row slab, [2,1,3,6144] to TWO) that my column slices never create; different L1
        #     layout for the four downstream consumers; different overlap with neighbouring ops.
        #   * THE FUSED OP IS FLAT IN S, NOT IN B. The ~97 us below was measured growing S. Growing
        #     the batch: B=2/4/8/16/32/64 -> 122.9/166.0/200.0/284.1/442.1/758.2 us, and it TT_FATALs
        #     at B=128. Manual wins every size from B=4 to B=64, but by less and less (0.788x ->
        #     0.942x) because manual makes THREE passes over memory where fused makes ONE. Block 2 is
        #     structurally 3 tokens, so B is the only growth axis and B=64 is 32x anything planned.
        #   * why 158 us before and today's numbers differ is UNRECONCILED. Recorded, not papered over.
        #
        # k is permuted (0,2,3,1) rather than (0,2,1,3), which emits it already transposed for the
        # scores matmul -- what transpose_k_heads=True used to do, at no extra cost.
        #
        # ORIGINAL FINDING FOLLOWS, kept for the L1 measurement and the sibling-op survey.
```

```text
        # THIS OP HAS A ~97 us FLOOR AND IT IS THE MOST EXPENSIVE NON-MATMUL LINE IN THE BLOCK --
        # 2.7 ms/frame over 21 calls, more than the wqkv matmul that feeds it. The floor is fixed
        # cost, not data movement: the same call on S=32 (10.7x the data) also measures 97 us. So
        # there is nothing to win by feeding it less or by laying the input out differently, and
        # both restructurings that avoid it are WORSE -- hand-rolled slice+reshape+permute is
        # 158 us, and riding the CFG pair on the sequence dim is 259 us. The two sibling ttnn ops
        # (create_qkv_heads, transformer.split_query_key_value_and_split_heads) reject GQA shapes.
        #
        # WHAT DOES WORK IS THE OUTPUT MEMORY CONFIG, and for a reason the op-level numbers hide.
        # Isolated, an L1 output saves only ~7 us on the op itself. In the real block it is worth
        # 2.5 ms/frame, because q/k/v then stay in L1 for the four ops that consume them:
        #
        #     output   transpose_k   ms/frame   codes != fp32 ref (8 draws)
        #     DRAM     False           26.75         7/288          <- was
        #     DRAM     True            26.72         7/288
        #     L1       False           24.28         9/288
        #     L1       True            24.17         9/288          <- shipped, 1.106x
        #
        # So L1 carries all of the speed AND all of the cost: 2 extra differing codes in 288.
        # `transpose_k_heads=True` is free -- it emits k already transposed for the scores matmul,
        # deleting our own transpose op -- but worth almost nothing on its own (1.001x); it is on
        # because one fewer op is one fewer thing to read.
        #
        # NOT bit-exact, and the chain is: the three tensors are bit-identical either way (verified
        # with torch.equal), but an L1-resident operand makes the downstream matmul pick a different
        # program config, hence a different accumulation order. Velocity PCC 0.99998522 ->
        # 0.99998164. Gated on codes and the 15-case run, not on PCC.
```

**The slice offsets, and why they are named constants**

```text
_Q_WIDTH  = 32 q  heads x 128 = 4096   -> q is columns    0 .. 4095  = tiles   0..127
_KV_WIDTH =  8 kv heads x 128 = 1024   -> k is columns 4096 .. 5119  = tiles 128..159
                                          v is columns 5120 .. 6143  = tiles 160..191

Every boundary is a multiple of 32, so all three slices land EXACTLY on tile boundaries and copy
whole tiles -- no row ever moves inside a tile. That is why these are named constants rather than
inlined literals: they are the head split's cut points, and a width that is not a multiple of 32
would silently make the slices expensive instead of raising.
```

**Why k gets a different permute**

```text
q and v are permuted (0,2,1,3) -> [B, heads, tokens, HD].
k is permuted (0,2,3,1)        -> [B, heads, HD, tokens].

That emits k ALREADY TRANSPOSED for the scores matmul, which is what the fused op's
transpose_k_heads=True did. It costs nothing extra -- it is the same op with a different
permutation -- and it saves the separate ttnn.transpose the scores matmul would otherwise need.
```

### [flow-11] `_block` — sdpa replaces the interior on Blackhole; the row fold is gone

**REVERSED. STATUS.md §6.45.** `scaled_dot_product_attention` handles GQA natively, so the row
fold and `REP` are both unnecessary and deleted. 4 ops become 1, worth **+2.555 ms/frame**.

§6.37 rejected this on the N150 at **6.48x** the error vs fp64, and it cost a WER word. Here it
is **1.57x** (velocity maxabs 4.043e-02 against 2.569e-02) and the discrete output does not move:
10/288 acoustic codes vs the fp32 reference and 0/8 semantic on 8 real prompts, both identical to
the hand-rolled path, and `--gate flow` reads 2 of 74 codes against 3. [flow-03] is explicit that
codes, not PCC, are the gate here — "codes are what reach the audio".

Long-form WER over 3 seeds: **1 wrong of 894 with sdpa against 4 without**, 15/15 `[END_AUDIO]`.

`scale=1.0` IS MANDATORY: SCALE is folded into wqkv's q rows ([flow-09]), so the default applies
1/sqrt(d) twice — §6.37 measured 3.8e-01 relative error for forgetting it.

---

**N150 RECORD BELOW — historical on this fork.**

### [flow-11-n150] GQA BY ROW FOLD, NOT BY REPEAT -- the same lesson as the...

```text
        # GQA BY ROW FOLD, NOT BY REPEAT -- the same lesson as the CFG fold above, and worth 1.40x
        # on this block. Mathematically the same attention: on device it gives the same velocity
        # PCC and the same 3-of-74 code diff, and in fp32 on host the two agree to 6e-07, which is
        # reduction order, not a different computation. Query head j reads kv head j//4, so reshaping
        # q from [B,32,3,d] to [B,8,12,d] stacks those 4 heads' 3 tokens as 12 ROWS against a
        # single kv head. Heads are contiguous in dim 1, so the grouping lands exactly on the GQA
        # mapping and the inverse reshape puts them back. Equivalent because a row of the score
        # matrix only ever interacts with itself: softmax is over the last dim (the 3 keys).
        #
        # The win is the same one the CFG fold gets: `repeat_interleave` made the two attention
        # matmuls BATCH-32, and a batched matmul costs per batch element. This makes them batch-8
        # with 12 rows each, and rows inside a 32-row tile are free. It also deletes the two
        # repeat_interleave ops, which were materialising k/v 4x.
        #
        # NO mask and NO RoPE here -- bidirectional by design. sdpa would fuse the interior into
        # ONE op and measures faster still (1.147x), but it triples the differing codes (7/288 ->
        # 21/288 over 8 draws); see STATUS.md 6.8.
        # No `multiply(s, SCALE)` here: 1/sqrt(head_dim) is folded into wqkv's q rows at load time.
```

### [flow-23] `_block` — shares Block 1's decode matmul program configs

Block 2 imports `DECODE_PRG` from `ttnn_voxtral_gpt` rather than defining its own. This is exact,
not approximate: the two blocks have **identical** dims (`DIM == FM_DIM == 3072`,
`HIDDEN_DIM == FM_HIDDEN_DIM == 9216`, 32 heads × 128, GQA 32/8), and Block 2's `B*3` is 3 or 6
rows — one tile, the same as Block 1's single row. So all five configs apply unchanged.

`ttnn_voxtral_gpt` does not import `ttnn_voxtral_flow`, so the direction cannot cycle.

Of the −4.24 ms/frame in [gpt-26], Block 2 contributes **−1.87** (21.10 → 19.12 ms) against Block
1's −2.36 (20.23 → 17.87). Block 2 sees 21 of the 47 w1 calls per frame despite having 3 layers to
Block 1's 26, because the 7 Euler steps re-enter the whole trunk each time.

### [flow-22] `_block` / `_trunk` — in-place elementwise, and the L1 trap in it

The two residual adds and the SwiGLU multiply run in place: **+0.790 ms/frame, codes unmoved**
(0 of 36 against the previous build, 1 of 36 against the fp32 reference either way). Frame counts
reproduce 68/452/493 on cases 0/2/3.

**THE TRAP, AND IT IS THE WHOLE REASON `_trunk`'s concat CARRIES A `memory_config`.** `add_`
writes wherever `x` ALREADY lives. The shipped `add(x, r, memory_config=_L1)` placed the residual
in L1 deliberately ([flow-02], worth 1.049x); `add_` inherits whatever it is given instead. With
`x` arriving from `_trunk` in DRAM, in-place silently reverts that decision — and per [flow-10]
an L1-vs-DRAM operand makes the downstream matmul pick a different program config, so it also
**moved a code**: 1 of 36 against the previous build and 2 of 36 against fp32, where shipped is 1.

Starting the stream in L1 instead — one kwarg on `_trunk`'s concat, no extra op — fixes both:

```text
    mul + resid, concat -> L1   21.104 ms/frame   +0.790   0/36 vs ship   1/36 vs fp32
    mul + resid, x in DRAM      21.262            +0.631   1/36           2/36
    mul only                    21.711            +0.183   0/36           1/36
    shipped                     21.894             --      --             1/36
```

So the L1 version is both FASTER and accuracy-neutral. **If that `memory_config=_L1` is ever
removed from the concat, the in-place adds silently change the model** — there is no error, only
a moved code.

**Measured and NOT taken: in-place in `_solve`.** The CFG combine and the Euler update are only
7 calls a frame against `_block`'s 21, and all three arms landed inside the noise floor
(+0.055 / −0.013 / −0.073 against 0.015 ms). Left alone. STATUS.md §6.48.

### [flow-12] `_block` — SiLU rides along on the w1 matmul instead of being its...

> **CORRECTED on p150 — the premise was wrong on both chips.** `activation="silu"` is **not
> fused**: it measures 98.8 µs against a plain matmul's 85.5, the same +14.9 as a separate
> `ttnn.silu()`. SiLU was its own op the entire time; the 0.16 ms/frame this note claims came from
> somewhere else. Real fusion needs a program config's `fused_activation` (88.1 µs) and is now
> how both blocks do it — worth **2.42 ms/frame** across the 47 w1 calls. See **[gpt-26]** and
> STATUS.md §6.52. The N150 record below is preserved as written.

```text
        # SiLU rides along on the w1 matmul instead of being its own op -- bit-identical (verified
        # here at 19/576 acoustic codes and velocity PCC unchanged), 0.16 ms/frame, and the same
        # thing Block 1 has always done. NOT the idea rejected in STATUS.md 6.8, which was fusing
        # silu into the MULTIPLY below via input_tensor_a_activations; that one really is worthless.
```

The second sentence of that comment still stands and is worth keeping straight: §6.8 rejected
fusing silu into the **multiply**, which is a different op and really is worthless. What was
missed is that the fusion into the **matmul** never happened either.

### [flow-13] `_trunk` — FOLD THE CFG BATCH INTO ROWS -- worth 2.23x,...

```text
        # FOLD THE CFG BATCH INTO ROWS -- worth 2.23x, bit-identical. A batched matmul re-reads the
        # whole weight per batch element, so batch-2 doubled every weight read; 6 rows still fit one
        # 32-row tile, so folding is free. A linear applies per row independently. Attention is the
        # only thing that needs the batch separated again. Numbers in STATUS.md.
```

### [flow-14] `_trunk` — PROJECT FIRST, THEN NARROW -- worth 1.09 ms/frame,...

```text
        # PROJECT FIRST, THEN NARROW -- worth 1.09 ms/frame, bit-identical, the single biggest
        # op-level win left in this block. It used to reshape to [B,3,3072], slice position 0 out at
        # 3072 wide, and project that. Both of those moves are now 36 wide instead: 85x less data,
        # and the linear runs batch-1 over 6 rows rather than batch-2 over 3, so it reads the weight
        # once instead of twice (a batched matmul re-reads the whole weight per batch element).
        # Computing all 6 rows costs nothing -- 3 and 6 are both one 32-row tile.
        # Same lesson as the codec's output projection, STATUS.md 6.13/6.14: shift the NARROW side.
        # Row r of seq is (batch r//3, token r%3) and a linear is per-row, so rows 0 and 3 are the
        # two position-0 vectors -- which is exactly what the reshape and slice then pick out.
```

### [flow-15] `_cfg_input`

```text
        """-> [2B, 3072] = llm_hidden (cond) over zeros (uncond), in a buffer reused per batch.

        CFG's unconditional half is a ZEROED hidden state, so the bottom half is zeros on every
        frame forever -- only the top half is ever written, so the zeros cannot drift. Rebuilding
        it with `cat([h, zeros_like(h)])` allocated twice per frame, ~12x per second of audio.
        """
```

### [flow-16] `_schedule`

```text
        """-> (time-conditioning tokens on device, per-step dt). Built once per (batch, n_steps).

        The solver schedule `linspace(0, 1, n_steps+1)` never changes, so BOTH halves of it are
        constants: `time_projection(time_embedding(t))` does not depend on the prompt, the frame or
        x (it used to be recomputed every step -- a host sin/cos plus a 3072x3072 matmul each), and
        neither do the step widths. They are derived and cached together here so a change to the
        schedule cannot move the tokens without moving the dt values with them."""
```

### [flow-17] `_solve`

```text
        """(x0 fp32 [B,1,36], cond++uncond [2B,3072]) -> x fp32 [B,1,36]. PURE DEVICE GRAPH.

        No host arithmetic, no allocation from torch, nothing shape-dependent on the data -- which
        is what makes it capturable as a trace. Keep it that way: one host op in here silently
        makes the whole solve untraceable again."""
```

### [flow-18] `decode_frame`

```text
        """[B,1], [B,3072] -> acoustic codes [B,36] int64, offset applied.

        THE WHOLE SOLVE STAYS ON DEVICE. It used to round-trip per step -- upload x/t/h, download
        the velocity, then do the CFG combine and Euler update in torch -- which cost n_steps host
        round-trips per frame and, more importantly, made the loop untraceable, since a device trace
        cannot contain host arithmetic.

        PRECISION IS NOT UNIFORM HERE, deliberately. The velocity network runs at self.dtype (bf16
        by default) but the solver state does NOT:
          * `x` is fp32 and stays fp32 for its whole life. It accumulates n_steps increments, so any
            error in it COMPOUNDS -- unlike a per-step rounding error in the velocity, which does
            not.
          * the CFG combine is fp32. `cfg_alpha*v_cond + (1-cfg_alpha)*v_uncond` with alpha=1.2 is a
            difference of two nearly-equal vectors, and the small difference is the entire point of
            CFG. Measured on device: the combine is accurate to 2.4e-7 in fp32 but only 7.0e-3 from
            bf16 inputs -- ~29,000x worse. Doing this in bf16 would be a real quality bug that PCC
            on the velocity would not reveal.
        Hence the explicit typecasts: down to self.dtype only to enter the network, straight back up
        to fp32 on the way out. The cast entering `input_projection` is also load-bearing for SPEED:
        ttnn allows an fp32 activation against a bf16 weight and returns fp32, which would silently
        promote the entire trunk to fp32 and forfeit the 1.55x.
        """
```


---

### [flow-19] `_trunk` — the caller supplies [B,1,3072]; hoisting the reshapes is worth 0.107 ms/frame

```text
        CALLER SUPPLIES [B,1,3072], not [B,3072]. The three reshapes used to live here, inside the
        per-step loop, and cost 10.1 us a step for nothing. What each operand actually does:

            p0  = linear(x_t, input_projection)   changes every STEP
                  -- and arrives ALREADY [B,1,3072], so its reshape was a pure no-op
            p1s = the time schedule               constant for the model's life
                  -- hoisted into _schedule's cache, paid once ever
            p2  = linear(h, llm_projection)       changes once per FRAME
                  -- hoisted to once per frame in _solve, alongside the matmul that builds it

            shipped:  3 reshape + concat(3) + reshape        115.9 us
            hoisted:      0 or 1 + concat(3) + reshape       105.8 us   1.107x, bit-exact

        On the whole Block 2 frame: 22.583 -> 22.476 ms, and the entire gain survives, which is rare
        for a small-op change (see STATUS.md 6.30). Bit-exact: same ops in the same order, just
        without the redundant ones.

        A NEARBY IDEA THAT DOES NOT WORK, recorded so it is not retried. Pre-concatenating p1++p2
        into a [B,2,3072] and making this a concat(2) measures 81.2 us, 1.442x -- but p1s[i] varies
        per step, so that concat does not vanish, it MOVES. Seven new concats a frame to save seven
        cheaper ones; the op count goes UP. Only operands that change LESS OFTEN than this runs are
        hoistable, which is the whole content of the list above.
```

## `ttnn_voxtral_codec.py`

*Block 3 -- codec decoder, codes to waveform*

### [codec-01] `module docstring`

```text
"""TTNN port of the Voxtral Codec DECODER (Block 3): audio codes -> 24 kHz waveform.

Mirrors reference/voxtral_codec_ref.py op-for-op. All compute runs on device; the only host-side
step is the semantic codebook gather (see _quantizer_host):

    codes [1,37,T] --quantizer--> [1,292,T] --conv(k3,replicate)--> [1,1024,T]
      --4x { Transformer(2 layers, ALiBi + causal + sliding window) [+ ConvTranspose(k4,s2)] }
      --output_proj(k7,reflect)--> [1,240,T'] --unpatch--> [1,1,T'*240] @ 24 kHz

Channel width is a constant 1024 through every upsample; only the final projection narrows it, to
240, and those 240 channels then BECOME time (each frame carries a 240-sample waveform patch).
The signal stays a channels-last [1,1,L,C] device tensor across conv stages and [1,L,C] across
transformer stages, so there is no host round-trip between ops -- codes in, waveform out.

=== HOW THE REFERENCE'S TORCH OPS MAP HERE ===
  * `ttnn.conv1d` is ZERO-pad only, while every conv here is a CAUSAL left-pad with
    reflect/replicate. `_pad_causal` builds those from slice+concat (the pad is k-1, i.e. 2 or 6
    columns, so it is a handful of slices -- ttnn has no flip).
  * conv_transpose1d does not exist; done via `ttnn.conv_transpose2d` with a singleton HEIGHT and
    length on the WIDTH axis, which the XTTS-v2 vocoder work showed is ~10x faster than mapping
    length to height (height slicing hits a circular-buffer/L1 clash).
  * ALiBi + causal + sliding window collapse into ONE additive pre-softmax bias, built on host and
    cached per (S, window, dtype); it does not depend on the weights. Being a function of (j - i)
    only, it is constant along diagonals -- which is exactly what lets one slab-sized bias serve
    every chunk of every utterance.
  * QK-norm is an RMSNorm over the FULL 1024-wide projection, BEFORE the head split.
  * LayerScale multiplies each residual branch by a learned [1024] vector.
  * `norm_eps` is 1e-2 here (not 1e-5) -- from params.json, and load-bearing.

=== PRECISION: MEASURED, NOT INHERITED ===
fp32 accumulation (HiFi3 + fp32_dest_acc_en) is on throughout; activations outside attention are
always fp32. The two knobs were swept on real weights, and scored on BOTH metrics -- PCC, and the
worst single sample as a % of peak, which is what the ear notices and what PCC can hide (see
"trap: PCC hides outliers" below).

    weights  attn   | synthetic worst-sample %peak  | real speech        | warm ms
                    |  T=64    T=256   T=469        |  PCC       worst%  | 64/256/469
    fp32     fp32   |  8.95    8.32    29.08        |  0.999988  0.81%   | 46/85/163
    bf16     fp32   | 13.66   14.90    49.78        |  0.999986  1.38%   | 44/83/162
    fp32     bf16   | 10.45    8.66    11.56        |  0.999984  1.16%   | 44/81/156  <- DEFAULT
    bf16     bf16   | 13.87   10.88    25.16        |  0.999983  1.93%   | 42/79/154

bf16 ATTENTION is the default and wins on both metrics: best PCC of the four (0.999800/0.999865/
0.999795 on synthetic), best synthetic worst-sample at T=469 by a wide margin, faster, and it
halves the largest tensor (the attention bias). fp32 attention is slightly better on real speech
(0.81% vs 1.16%) but is 2.5x worse on synthetic at T=469 and ~5% slower, and synthetic is
deliberately the conservative gate.

bf16 WEIGHTS are BAD and there is no longer a knob for them. On their own they fall below the
0.999 PCC gate at T=469, and they do not buy speed either (44 vs 46 ms) -- an earlier sweep showed
~20%, but that was conv weight PREPARATION cost, since hoisted out of the per-call path. And
bf16+bf16 measures 1.93% worst-sample against a 2.00% gate, i.e. 3.5% of margin, for a ~1% gain.
If you want to re-open this, re-run the real-speech fixture, not just the synthetic one.

Made no difference: keeping the small per-channel tensors (RMSNorm / QK-norm weights, LayerScale)
in fp32 while matmul weights are bf16 -- PCC 0.999512 either way, so the bf16 loss is in the matmul
weights, not the norms.

The fp32 -> bf16 conversion of q/k/v is itself worth 3.70e-03 of worst-case error, about half of
the hand-rolled path's total 5.85e-03 (internal math alone: 4.22e-03). Real but not the dominant
term, and the sweep above already prices it in.

Not carried over: the XTTS-v2 HiFi-GAN result that bf16 costs 0.91-0.96 PCC. That was a 34-conv
chain with bf16 ACTIVATIONS throughout; here attention output enters the residual scaled by
LayerScale (~0.01) and there are only 8 attention ops, so it does not transfer.

=== PERFORMANCE (warm, N150, defaults) ===
27.3 ms for 5.1 s of audio (RTF 0.0053, 188x real-time), 97.0 ms for 37.5 s (387x), 345 ms for
120 s (348x). Upstream report RTF 0.103 for their WHOLE pipeline on an H200, so this block is
~3-5% of the end-to-end budget. It is NOT where the end-to-end answer gets decided -- Block 1 is
(87% of the parameters, 12.5 sequential steps per second of audio).

Per-block profile after the head fusions (L=4096, one of 8 blocks), which is what says the block is
now arithmetic-bound rather than movement-bound:

    norms  qkv   qkn   split  attn   merge  wo    mlp   resid | TOTAL
    0.46   1.71  0.45  0.95   9.68   0.21   0.58  7.87  1.07  | 22.97 ms

attn (42%) + mlp (34%) = 76% is real matmul work. split and merge were 11.67 and 4.93 ms before
optimization #7; they are now 0.95 and 0.21. Nothing movement-shaped is left to remove.

=== OPTIMIZATIONS APPLIED, AND WHAT EACH WAS WORTH ===
  1. bf16 attention (above): best accuracy of the four configs AND faster.
  2. Chunked windowed attention (_attention), slab 512: O(S^2) -> O(S*slab). At S=12000, warm
     892 -> 497 ms, cold 10580 -> 1178 ms, mask 2304 MB -> 4.2 MB. EXACT, not approximate.
  3. Uniform slab-sized chunks: one cached bias per window, and one attention shape for the
     process lifetime. Bias cache 23 tensors/53 MB -> 6/18.1 MB, and it stops growing with
     utterance length (measured keys: (128,2) (256,4) (512,2) (512,4) (512,8) (512,16)).
  4. Conv length bucketing (BUCKET): on a stream of 12 distinct lengths, 120.9 s -> 1.66 s (73x).
  5. Hoisted conv weight preparation (_prepared): 2.6x at short lengths (112.8 -> 43.7 ms at
     T=128); host share of wall time 88% -> 24%.
  6. Content-deduplicated prepared weights (_prepared): 730 MB -> 98 MB, bit-identical.
  7. FUSED head split/merge in _block: 1.6x on the WHOLE block (155 -> 97 ms at T=469). The
     reshape+permute pair was 41% of Block 3 -- larger than attention -- and pure data movement.
     nlp_create_qkv_heads 11.57 -> 0.95 ms and concatenate_heads 4.88 -> 0.20 ms at L=4096.
     Accuracy is unchanged (real speech PCC 0.999984, worst 1.16% of peak, both identical).
     Found late: the six rejected items below all targeted attention or the convs, and the
     reshapes were never profiled. On this hardware the wins are in op count and data movement,
     not arithmetic -- every FLOP-reducing idea here lost, and this one won.

=== MEASURED AND REJECTED -- do not retry without new information ===
  * sdpa instead of the hand-rolled attention interior: 1.44-2.27x FASTER but 3.3x worse
    worst-case error, failing 11 tests. Eight levers exhausted including a tt-metal source patch.
    Full detail and the exhaustion list in _attention_slab's docstring.
  * Smaller slab, toward the compute optimum 2*window: slab=32 computes 9x FEWER scores and runs
    9x SLOWER (334 vs 36 ms per decoder pass). Per-kernel cost dominates arithmetic here.
  * Device trace capture: 1.00x at every slab size, including 3570 ops. The async command queue
    already hides host dispatch behind device execution, so there is nothing to recover. (Trap:
    "time is in the TTNN wrapper" != "time is dispatch".)
  * Batching the chunks into one matmul: the batched attention is 2.4x faster, but building the
    stacked tensor costs more than that saves (11.30 vs 9.67 ms). The chunks overlap by `window`,
    so the gather is an unavoidable copy -- ttnn has no strided view.
  * Unchunked attention with a full [S,S] mask: identical accuracy (so chunking really is exact),
    but 3x slower at S=4096 and the mask grows quadratically to 268 MB. Only S<=1024 prefers it,
    which is a `chunk_min` question, not a chunking one -- see CHUNK_MIN.
  * Fused SwiGLU MLP (one [1024, 2*4096] matmul + ttnn.swiglu, weight ordered w3|w1): 0.77-0.86x,
    i.e. SLOWER, and maxabs 3.4e-02 vs the current path at short L so it is not even equivalent.
    Note swiglu needs 4D input. The MLP is ~116 GMAC at L=4096 -- arithmetic-bound, nothing to fuse.
  * Fused QKV projection (one [1024, 3072] matmul + 3 slices instead of three [1024, 1024]
    matmuls): 0.74-0.82x. Same FLOPs, and it BUYS data movement -- the exact mirror of why
    optimization #7 won. maxabs also drifts to 2.4e-03 at long L, so not a free swap either.
  * Fused residual+norm (`rms_norm(x, residual_input_tensor=r)` instead of add-then-norm): this one
    is genuinely FASTER, 1.54-1.73x, but the base is small -- 0.17 ms per site, 2 sites x 8 blocks
    = ~2.7 ms of 97 ms (2.8%) -- and maxabs is 4.4e-03 on the RESIDUAL path, which every later layer
    inherits. Not taken: a 2.8% gain does not justify perturbing the residual stream. Revisit only
    with the real-speech worst-sample fixture as the gate.

=== TRAPS ===
  * PCC HIDES OUTLIERS. It is a correlation: it can sit at 0.9998 while individual samples are
    badly wrong, and for audio the outliers are what you hear. Every accuracy claim here is
    therefore paired with a worst-sample bound, and the real-speech fixture asserts both.
  * Prepared conv weights are NOT length-independent -- same shape, different bytes. See _prepared.
  * `prepare_conv_*`'s `input_dtype` is the ACTIVATION dtype. See _prep_weight.

=== MEMORY ===
Prepared conv weights are cached and deduplicated by content: 8 distinct layouts across all 5
convs and all 12 buckets, so 98 MB rather than the 730 MB that keying by length alone produced
(0.8% of an N150's DRAM instead of 6.5%). Plus 60.8 MB of host copies, kept so a new length can
still be prepared, and 18.1 MB of attention bias (6 tensors).

Validate against the reference (per-stage PCC bisect + the default bucketed path):
    TT_METAL_HOME=<repo> PYTHONPATH=<repo> python models/experimental/voxtral_tts/tt/ttnn_voxtral_codec.py
"""
```

### [codec-02] `module level` — The reference masks with -inf. On device a...

```text
# The reference masks with -inf. On device a max-subtracting softmax would compute
# inf-inf -> NaN, so use a large finite negative instead: exp() underflows to exactly 0,
# and it is unambiguous against real ALiBi values (which reach only about -16).
```

### [codec-03] `module level` — Chunked attention -- see OPTIMIZATIONS in [codec-01] #2/#3. `slab` MUST...

```text
# Chunked attention -- see OPTIMIZATIONS in [codec-01] #2/#3. `slab` MUST be tile-aligned: TILE_LAYOUT pads every
# dim to 32, so a slab of 272 would silently become 288 and waste a row and column of tiles.
# 512 is measured-optimal; both smaller and larger lose (see MEASURED AND REJECTED).
```

### [codec-04] `module level` — Chunk only above this length. Below it the full mask is...

```text
# Chunk only above this length. Below it the full mask is already cheap and chunking loses a few
# percent to per-op cost (T=64: 89 -> 97 ms). Measured crossover is S ~ 2000, so 1024 would be
# slightly better (1.06x on attention, ~0.1% end-to-end) at the cost of a 16.8 MB bias.
```

### [codec-05] `module level` — Conv length bucketing -- see OPTIMIZATIONS in [codec-01] #4. Every...

```text
# Conv length bucketing -- see OPTIMIZATIONS in [codec-01] #4. Every conv's input length scales with T, so each
# distinct utterance length compiles 5 new conv programs at 1-5 s each, and T is whatever the model
# generates -- a ONE-frame difference cost 5.5 s vs 181 ms warm. Rounding T up caps the shape count:
# costs 7-25% on warm steady-state (padded compute), which production never sees, and 128 gives 12
# buckets for the model's ~1500-frame ceiling. Set None if you truly decode one fixed length.
# NOTE: 128 is wrong for STREAMING -- a 1-second chunk then costs the same as a 10-second one.
```

### [codec-06] `module level` — Rows in the output projection's reflected prefix. It only...

```text
# Rows in the output projection's reflected prefix. It only NEEDS PATCH_PROJ_KERNEL-1 = 6, but a
# 6-row prefix makes the following concat land off a tile boundary, and the ragged version costs
# 1.815 ms against 0.281 for the aligned one. So pad the prefix to a full 32-row tile and start the
# output slices at OUT_PREFIX-6 instead of 0. The extra 26 rows are copies of x[0]; they are finite,
# they feed the matmul, and every output row they touch is sliced away again. See _graph.
```

### [codec-07] `__init__` — With PRE-PREPARED weights the op can no longer infer...

```text
        # With PRE-PREPARED weights the op can no longer infer weights_dtype from a host tensor,
        # so it must be stated explicitly -- and the SAME config must go to prepare_* and to the
        # conv call, or the prepared layout will not match what the kernel expects.
```

### [codec-08] `__init__` — ...and the reflected prefix those taps slide over, as a...

```text
        # ...and the reflected prefix those taps slide over, as a GATHER INDEX rather than six
        # single-row slices. Row OUT_PREFIX-6+m of the prefix takes x[6-m], giving x6,x5,x4,x3,x2,x1;
        # rows 0..OUT_PREFIX-7 take x[0] and are discarded. Length-independent, so it is built once.
```

### [codec-09] `__init__` — --- convs ---

```text
        # --- convs ---
        # Weights stay on HOST here and are prepared on first use by `_prepared`, which also
        # deduplicates them -- see that method for the layout table and the memory numbers.
        # WHY at all: ttnn.conv1d transforms and re-uploads its weights INSIDE the op, so without
        # this it redid that work for all 5 convs on EVERY call -- see OPTIMIZATIONS in [codec-01] #5 for what
        # hoisting it was worth. 60.8 MB of host copies, kept so a new length can still be prepared.
```

### [codec-10] `_prepared`

```text
        """Prepared weight for this conv AT THIS INPUT LENGTH, DEDUPLICATED BY CONTENT.

        Prepared layouts are length-specific, but they change at only ONE length threshold per
        conv -- and for `up6` and `out` they never change at all. Measured across all 12 buckets:

            conv   distinct layouts   lengths sharing one
            in            2           {128} {256..1536}
            up2           2           {128,256} {384..1536}
            up4           2           {128} {256..1536}
            up6           1           {128..1536}   all identical
            out           1           {128..1536}   all identical

        So keying only by length stored up to 12 BYTE-IDENTICAL copies: 8 distinct layouts held as
        60 tensors, 730 MB instead of 98 MB. Hashing the prepared bytes and sharing the tensor is
        pure deduplication -- the tensors are bit-identical, so there is no accuracy question.

        Cost is one host readback per newly-seen (conv, length), on top of the 5-24 ms preparation.
        Both are first-touch only, and the hot path is a plain dict hit."""
```

### [codec-11] `_prep_weight`

```text
        """One wrapper for both prepare_conv_weights and prepare_conv_transpose2d_weights: the two
        take an identical kwarg set and differ only in the weight layout they expect
        (ConvTranspose1d's [in,out,k] is IOHW; Conv1d's [out,in,k] is OIHW).

        `input_dtype` is the ACTIVATION dtype (always DTYPE here), NOT the weight dtype. Back when
        weights were switchable, passing the WEIGHT dtype here prepared a layout for bf16
        activations while the real activations were fp32, and silently produced PCC 0.008."""
```

### [codec-12] `_pad_causal`

```text
        """x [1,1,L,C] -> [1,1,L+pad,C]. `mode` mirrors torch F.pad on the length axis:
        replicate repeats column 0; reflect mirrors about column 0, excluding it.

        Production only reaches `replicate` now -- it needs one slice, so it is cheap. `reflect`
        needs `pad` of them and cost 3.07 ms in the output projection, which builds its prefix with
        ttnn.gather instead (see _graph). The branch stays because it is the faithful mirror of
        F.pad and test_codec_ttnn_pcc checks both modes against torch."""
```

### [codec-13] `_attention_slab`

```text
        """Attention with an additive pre-softmax bias, in ATTN_DTYPE. [1,H,S,d] -> [1,H,S,d].

Hand-rolled rather than ttnn.transformer.scaled_dot_product_attention, which was
        measured and rejected. sdpa DOES accept an arbitrary additive mask (`attn_mask` with
        `is_causal=False`), and dropped into this chunk loop with the same slab bias it is
        1.44-2.27x FASTER (25.1 -> 17.0 ms per decoder pass) because the fused kernel never
        materialises the [slab,slab] scores in DRAM, where this path writes them four times.

        It loses on ACCURACY. Against an exact fp64 answer at S=512:

            path                 PCC vs exact   max abs err   mean abs err
            hand-rolled (this)   0.99999559     5.85e-03      3.98e-04
            sdpa                 0.99985772     1.95e-02      2.35e-03   <- 3.3x worse worst-case

        Adopting it failed 11 tests, including the real-speech fixture's worst-sample bound -- the
        gate that guards what is audible -- and per-stage PCC 0.916 after one 2-layer stage. The
        per-slab PCC was a healthy 0.9998, so this is the PCC-hides-outliers trap firing exactly.

        DO NOT RE-LITIGATE without new information. Every lever below leaves the worst-case error at
        exactly 1.951e-02, which is why the cause is believed to be the compute kernel's arithmetic:
          * chunk geometry -- 1 vs 4 k-blocks; chunked vs unchunked full mask
          * `exp_approx_mode=False`; `fp32_dest_acc_en=True`; HiFi4 (`use_high_precision_compute`)
          * fp32 q/k/v -- rejected outright, sdpa is bf16/bfp8/bfp4 only
          * patching sdpa_program_factory.cpp so im_df/stats_df are Float32, and REBUILDING.
            Confirmed live via a marker (im_df=Float32, fp32_dest_acc_en=true) -- error unchanged.
            So closed issue #13364 ("Enable FP32 Accumulate in Flash Attention") is a red herring
            here. Isolating the inputs confirms it too: with a bf16-rounded reference (input
            conversion costing zero) sdpa's internal error is 2.07e-02 against our 4.22e-03.
        Reaching it would mean editing kernels/compute/sdpa.cpp. Re-test only if that changes.

        Separately, sdpa forbids `attn_mask` together with `is_causal` or `sliding_window_size`
        (explicit TT_FATALs), so its native block-skipping is unreachable. That costs almost nothing:
        its windowed path measured only 1.22x faster than this chunking and cannot express ALiBi at
        all (PCC 0.64 without it).

        Hand-rolling also keeps `numeric_stable=True` on the softmax. Runs in ATTN_DTYPE (bf16) --
        see the PRECISION table in the module docstring."""
```

### [codec-14] `_attention`

```text
        """[1,H,S,d] -> [1,H,S,d]. Chunks when S > chunk_min, else one full-S pass.

        EXACT, not an approximation: attention is causal AND windowed, so output[i] depends only
        on input[i-window .. i]. A slab starting `window` positions early therefore has all the
        context its kept rows need; the leading `window` rows are dropped because THEIR context is
        missing. Verified against full-S attention (max abs diff ~1e-7) and against the unchunked
        device path.

        EVERY CHUNK IS EXACTLY `slab` LONG, so there is ONE cached bias per window and attention
        sees one shape for the process lifetime. Two details buy that:
          * chunk 0 starts at lo=0, so it needs NO left context -- all `slab` rows are valid
            outputs (cut=0) and local index == absolute index, so the ordinary slab bias is
            already correct. No left padding, no special-case bias.
          * the final chunk is padded on the RIGHT to `slab`. Safe with the same bias because
            causal masking already forbids any real row from looking forward into the padding;
            the padding rows are computed and discarded.
        Without this, first/last chunk lengths varied (the last with S mod C), which meant a new
        bias AND a new kernel compilation for every distinct utterance length."""
```

### [codec-15] `_block` — Head split/merge via the FUSED ops, not reshape+permute....

```text
        # Head split/merge via the FUSED ops, not reshape+permute. Measured at L=4096: the split
        # went 11.57 -> 0.95 ms (12x) and the merge 4.88 -> 0.20 ms (24x). Reshape+permute was the
        # single largest cost in this block -- larger than attention itself -- and it is pure data
        # movement. `nlp_create_qkv_heads` wants q separate and k|v fused, and 4D inputs (the
        # reshape to [1,1,L,C] is metadata only); it takes q/k ALREADY QK-normed, so it does not
        # conflict with normalising over the full 1024 width before the split.
```

### [codec-16] `_quantizer_host`

```text
        """codes torch [1,37,T] -> HOST torch [1,1,T,292] channels-last.

        Semantic is a table lookup, acoustic is pure FSQ arithmetic. Kept on host: ttnn.embedding
        needs a BFLOAT16 table and the semantic entries are large (|x| ~ 10), so a bf16 table
        would inject ~0.4% before a deep conv stack that does not cancel error. Split out from the
        upload so the upload target is explicit."""
```

### [codec-17] `_graph` — OUTPUT PROJECTION AS MATMULS, NOT ttnn.conv1d -- and the...

```text
        # OUTPUT PROJECTION AS MATMULS, NOT ttnn.conv1d -- and the reason is a ttnn BUG, not speed.
        #
        # `ttnn.conv1d` here was the exact op that made Block 1's w2 unusable in BFP8. Its
        # sliding_window `halo_gather` kernel issues an out-of-range NOC write (13,897,728 bytes to
        # a nonexistent core) on the SECOND execution of this shape -- a program-cache hit -- and
        # hangs the card. Full dump in ttnn_voxtral_pipeline; STATUS.md 6.12 has the investigation.
        #
        # A k=7 stride-1 conv over an ALREADY-PADDED tensor is just a sliding-window matmul:
        #     out[t] = sum_j  xpad[t+j] @ W[j]
        # so 7 slices, 7 matmuls and 6 adds compute it exactly, touching no halo kernel at all.
        #
        # SHIFT THE OUTPUT, NOT THE INPUT. Both orders compute the same sum, but the shift has to
        # be a slice, and slicing the 1024-wide INPUT costs 0.624 ms a time against 0.145 for the
        # 240-wide OUTPUT. Multiply the full padded input by each tap, THEN slice the narrow result:
        #     conv1d (broken)                              4.29 ms
        #     7 matmuls, shift the INPUT  (slice xp)        9.16 ms   +4.87
        #     7 matmuls, shift the OUTPUT (slice y)         6.26 ms   +1.98
        #     + GATHERED prefix instead of _pad_causal      3.45 ms   -0.84   <- this, BIT-IDENTICAL
        # The input slices were 4.37 of the 9.16 ms, more than the seven matmuls together (1.93).
        #
        # THE PAD WAS THE EXPENSE, NOT THE MATMULS -- 3.07 of the 6.26 ms. `_pad_causal` builds the
        # reflection with SIX single-row slices, and against a 16 MiB TILE_LAYOUT tensor one
        # single-row slice costs 0.381 ms whether it returns 1 row or 6:
        #     one single-row slice of x        0.381 ms      six of them   2.282 ms
        #     one SIX-row slice of x           0.358 ms      ragged concat 1.815 ms
        # Cost is per op, not per byte. So take the prefix in ONE aligned 32-row slice and do the
        # reversal with ttnn.gather (see _out_prefix_idx): 3 ops and 0.071 ms instead of 8 and 3.07.
        # Bit-identical output, verified max-abs-diff exactly 0 -- gather moves data, it does not
        # arithmetic. A permutation MATMUL also works and is equally fast, but loses 2.4e-04: fp32
        # matmul multiplies at bf16 precision here, and HiFi4 + fp32_dest_acc does not change it.
        #
        # FUSING THE TAPS was measured and NOT taken. Concatenating g taps into one [1024, g*256]
        # weight cuts the 7 matmuls to 7/g and reads xp once per group instead of once per tap:
        #     g=1 (this)  3.45 ms  bit-exact      g=3   2.94 ms  err 1.1e-04
        #     g=2         3.15 ms  err 3.6e-04    g=7   worse -- its 28 MiB output goes to DRAM
        # 0.51 ms more, at the price of exactness (a [1024,768] matmul decomposes differently from
        # three [1024,240] ones) on an op that is 0.01% of wall. Not a trade worth making here.
        # Note also that blocks must be 256-aligned: at pitch 240 the half-tile column offset comes
        # back SILENTLY WRONG out of L1 (rel err 5e-01, no exception raised).
        #
        # Not parallelisable, before anyone tries: the seven passes are independent, but every ttnn
        # op already uses the whole 64-core grid, so running them at once would just give each pass
        # 9 cores. Nothing is idle. And holding xp in L1 to avoid re-reading it does not fit --
        # 16.8 MB overflows the allocator.
```

### [codec-18] `__call__` — return_stages BYPASSES bucketing on purpose: it exists to...

```text
        # return_stages BYPASSES bucketing on purpose: it exists to bisect against the
        # reference's per-stage goldens, and a bucketed run's stages are at the padded length
        # (T, 2T, 4T, 8T of the BUCKET), so they would not correspond. Trimming each stage
        # separately would work but is error-prone for a debug-only path.
```

### [codec-19] `__call__` — repeat the LAST frame rather than zero-pad: the tail then...

```text
                # repeat the LAST frame rather than zero-pad: the tail then looks like plausible
                # audio to the causal convs instead of a hard edge. It is trimmed off either way,
                # but the transposed convs overlap, so a pathological tail is worth avoiding.
```


---

## `ttnn_voxtral_pipeline.py`

*the three blocks wired together*

### [pipe-01] `module docstring`

```text
"""End-to-end Voxtral-TTS on device: text ids + voice preset -> 24 kHz waveform.

Mirrors reference/voxtral_pipeline_ref.py's `generate`, with all three blocks on TTNN:

    ids + voice --embed--> Block 1 prefill --> h ─┐
                                                 ├─> Block 2 -> 37 codes ─┬─> embed_frame ─┐
                                                 │                        │                │
                                        Block 1 step <────────────────────┴────────────────┘
                                                 │
                              accumulated frames ┴─> Block 3 -> waveform @ 24 kHz

Stops on [END_AUDIO] as the semantic code, or at max_frames. Frame rate is 12.5 Hz, so 150 frames
is 12 s of audio.

WHAT RUNS WHERE. Blocks 1-3 all run on device. Three host steps remain, each deliberate:
  * the tekken tokenizer and prompt assembly (upstream of everything; see voxtral_tokenizer_ref)
  * `embed_frame` -- a 37-way embedding gather + sum, per frame; 57.3 us on the p150, and
    7.5x SLOWER on device (6.50). ttnn.embedding needs a bf16 table
    and these tables are large-valued, the same reasoning as the codec's semantic gather.
  * Block 2's FSQ quantise -- clamp/scale/round on [B,36]; 36 values is not worth a dispatch.
    (Block 2's semantic argmax USED to be here too. It is on device now, in fp32 -- worth 1.49
    ms/frame; see ttnn_voxtral_flow.semantic_dev for why fp32 and not bf16.)

FIDELITY, measured per block against the fp32 CPU reference, on real prompts -- never random ones,
which are a pessimistic proxy and cost a lot of time once (STATUS.md trap #12):
    Block 1 prefill  PCC 0.999924 / 0.999883  (last position -- all Block 2 consumes)
    Block 1 decode   PCC 0.99985+, mean worst-sample 0.86% over 44 teacher-forced frames
    Block 2 velocity PCC 0.99998522, semantic codes EXACT, 71/74 frame codes exact on synthetic h
    Block 3          real speech PCC 0.999984, worst sample 1.16% of peak

END TO END, the number to quote is **long-form WER: 1 wrong word in 298**, plus 15/15 natural
[END_AUDIO] and the voice-identity check. NOT the 340-word natural-text headline: that bucket
includes 42 words of 3-to-6-word clips where one Whisper disagreement is worth 17-50%, and the
SAME CODE at seeds 0/1/2 spans 0.88-2.06% on it. See STATUS.md 6.7 before quoting any WER, and use
the teacher-forced gates in tests/tt_gates.py to judge a numerical change.

PERFORMANCE, steady state on one N150, long-form cases:

    Block 1 decode      ~25.7 ms/frame   ~51%
    Block 2 flow        ~23.0 ms/frame   ~46%
    host embed_frame      0.2 ms/frame    0.4%
    TOTAL               49.0-52.5 ms/frame, mean 50.4 over the 15-case fixture
    prefill 0.1-1.5 s once; Block 3 codec 97 ms warm, i.e. 0.4% -- but SECONDS the first
    time a bucket length is seen, which is COMPILE cost, not compute (STATUS.md 6.10)
    RTF 0.62-0.71 on 14 of 15 cases   (RTF = generation / audio, lower is better)

The 15th is case 0 at RTF 1.89, and that is COLD-START, not a slow case: it pays the first codec
bucket's kernel compiles and the first prefill shape. Every later case with the same shapes runs
at 0.74-0.80. Quote the steady-state number, and re-run a case twice if it looks anomalous.

A frame is 80 ms of audio at 12.5 Hz, so RTF = ms_per_frame / 80.

WHERE THE REMAINING TIME IS. Both blocks now stream every weight matmul at the 194 GB/s DRAM
ceiling, so neither has a byte or a layout trick left; each module's docstring carries the per-line
map. What is left is different in each:
  * Block 1 is at its floor except for w2, the last bf16 weight and the pinned trigger of the hang
    documented below. ~21 of its 25.7 ms is pure weight streaming at the ceiling, and everything
    that is not a matmul now totals under 5 ms.
  * Block 2 sits ~1.6x above its 13.4 ms weight-read floor, and that gap is DEVICE-side per-kernel
    cost -- proven by tracing, which removes host dispatch and changes nothing (6.6). Fewer ops
    does not help; bigger kernels and L1-resident operands do. Its worst single line is
    nlp_create_qkv_heads, whose ~97 us is a FIXED cost (same at 10.7x the data).
  * Block 3 is NOT a target: 97 ms warm against a ~26 s generation, i.e. 0.4%. Its
    seconds-scale appearance in a fresh run is first-call kernel compilation per bucket,
    not compute. An earlier version of this file called it ~9% of wall; that was derived
    by subtraction and conflated the two. STATUS.md 6.10.

The one structural idea left is throughput, not latency: a 3-token sequence wastes 26 of 32 tile
rows and nothing in ONE utterance can fill them (Euler steps are sequential, frames are
autoregressive), but CONCURRENT REQUESTS fit exactly.

Against ign/voxtral_p150_qb2: their code cannot run on our tree, so their tt-metal was built
separately and measured here at 598 ms/frame. That is their Blackhole-targeted code on our
Wormhole card, which answers "can we adopt theirs" (no) and NOT "is their P150 slow" (unmeasured).
STATUS.md 6.5 has the setup and the two findings it did corroborate.
"""
```

### [pipe-02] `module level` — A HANG THAT SHAPED THE SHIPPED CONFIG, recorded because...

```text
# A HANG THAT SHAPED THE SHIPPED CONFIG, recorded because the workaround is gone and the trigger
# is still live in ttnn. Multi-utterance runs used to hang inside Block 3's decode and take the
# card down with it (recovery needs a tt-smi board reset). It required FIVE things at once:
#
#     all-BFP8 weights in Block 1    <- the one we control; the mixed default avoids it
#     Block 2 in the loop            raw Block 1 steps alone: completes
#     Block 3 on device              codec on CPU: completes
#     >= 2 distinct codec buckets    one bucket everywhere: completes
#     a generation BETWEEN two same-bucket decodes
#
# Minimal repro under all-BFP8, ~90 s: short gen + 128-bucket decode, long gen, 512-bucket decode,
# long gen, 512-bucket decode -- the last is a pure cache HIT and hangs. tt_transformers never saw
# it because it uses the same mixed precision we now do.
#
# ROOT-CAUSED 2026-08-04 with TT_METAL_WATCHER=10, which turns the hang into a clean abort with a
# device-side dump instead of a wedged card. Investigating this no longer costs a board reset.
#
#   EXACT OP     ttnn_voxtral_codec.py:589 -- the codec's OUTPUT projection,
#                _conv1d(x, "out", 1024, 240, kernel=7, stride=1, "reflect")  at L=4102
#   EXACT KERNEL ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/
#                halo_gather.cpp   (stuck on both BRISC and NCRISC)
#   EXACT FAULT  NCRISC on noc1 attempts a unicast write of 13,897,728 bytes from local
#                L1[0x15f0000] to virtual core 18-52 [addr=0x008ae800] -- a core that does not
#                exist. 13,897,728 = 3393 x 4096, and 4096 B is exactly one input row (1024 fp32
#                channels), so it is trying to push 3393 rows in ONE NOC transaction to nowhere.
#
# IT IS NOT A DEADLOCK. It is an out-of-range NOC write with an implausible size, i.e. a corrupted
# descriptor -- which is why memory looked flat and why no amount of program-cache or leak hunting
# found it.
#
# THE TRIGGER IS THE SECOND EXECUTION OF THAT EXACT SHAPE, a pure program-cache hit. Case 2 runs
# L=4102 and completes; case 3's byte-identical call faults. That is precisely the "pure cache HIT"
# in the minimal repro below, now with a named kernel.
#
# WHY BLOCK 1's w2 IS INVOLVED AT ALL: it is not a Block 1 bug. w2's dtype is simply the biggest
# lever we have on DRAM allocation ADDRESSES -- BFP8 frees ~690 MB across 26 layers, shifting every
# later allocation including the codec's conv buffers. So this is a latent address-dependent bug in
# ttnn's conv halo path that w2 happens to expose. Anything else that moves allocations by enough
# would do the same, which is why the five conditions below looked so arbitrary.
#
# UNTESTED IDEAS for dodging it, if upstream is slow: we already pad manually and pass padding=0, so
# a conv_config that avoids the halo path may exist; or the output projection could be expressed
# without conv1d at all, since a k=7 stride-1 conv over a pre-padded tensor is a sliding-window
# matmul. Worth ~2.5 ms/frame (w2 in BFP8) if either works.
#
# Measured and ELIMINATED, so none of these get retried: memory (flat, 8 GB free at the hang);
# program-cache COUNT (576 entries over 4 buckets completes, while we died at 310-341 and
# tt_transformers lived at 329); Block 3 length/content; a Block 1 leak (1400 steps clean); and
# every distinctive Block 1 op. The underlying ttnn failure -- a silent hang rather than an
# error -- is unreported upstream and still unexplained.
```

### [pipe-03] `__init__` — embed_frame is a host gather, so it needs the backbone's...

```text
        # embed_frame is a host gather, so it needs the backbone's audio embedding table. Load it
        # BEFORE the backbone and hand the same dict over: our Block 1 would otherwise load its own
        # ~13 GB fp32 copy of the same file.
```

### [pipe-04] `generate` — Only the last position conditions the first frame,...

```text
        # Only the last position conditions the first frame, matching the reference's
        # IncrementalBackbone.prefill which returns x[:, -1:]. Both backbones expose this as
        # prefill_last so the loop below does not care which one is running.
```

### [gpt-19] `_V_SHARD` — DELETED on Blackhole; the fused cache write loses here

**REVERSED. STATUS.md §6.44.** `paged_fused_update_cache` is **0.687 ms/step SLOWER** on the p150
than two plain `paged_update_cache` calls, so both it and `_V_SHARD` are gone — and with them the
silent failure mode below, where RoPE on a core whose cos/sin table lives elsewhere returns
3.4e38 from uninitialised L1 rather than raising. Bit-identical either way.

Same cause as §6.45: a small op costs 3.4x more here (67.7 us against the N150's ~20), and the
fused write buys one launch at the price of a reshard.

---

**N150 RECORD BELOW — historical on this fork.**

### [gpt-19-n150] V's parking space, and why V and not K

```text
 V's parking space, core (1,0). paged_fused_update_cache writes both caches in one kernel but
 refuses K and V on the same core, and nlp_create_qkv_heads_decode puts q, k and v all on (0,0), so
 ONE of them has to move. MOVE V, NOT K -- the choice matters more than it looks:
   * moving K is what overlap_qk_coregrid=False does. It is 0.047 ms/frame faster, and it costs two
     coupling hazards: it asserts a whole head per core (pinning _QKV_SHARD to 48 cores, load-bearing
     and non-obvious), and K then goes through RoPE on a core whose cos/sin table is elsewhere --
     which does NOT raise, it returns 3.4e38 from uninitialised L1.
   * V never touches RoPE and imposes nothing on the shard width. One reshard per layer, 26 a frame,
     ~2.1 us each because it is an 8 KB hop between adjacent cores.
 0.405 vs 0.452 ms/frame -- 0.09% of a frame to delete a silent-garbage failure mode. Taken from
 lserbedzija/xtts-gpt-ttnn, which does the same thing in xtts_v2/tt/ttnn_xtts_gpt.py.
```

### [gpt-20] `wo` — NO program config on Blackhole (p150 fork)

**DELETED ON THIS FORK. STATUS.md §6.43; the N150 record is kept below.**

§6.25 hand-tuned `_WO_PRG` for the N150 and it was worth +0.196 ms/frame there. On the p150 it
is worth nothing any instrument here can find, so it is gone along with `_WO_GRID`.

Removing it is **bit-exact**, which is what made the decision purely about speed: `torch.equal`
on the 26-layer decode output, and — the strong form — **all 45 utterances of a 15-case x 3-seed
run reproduced identical frame counts** (§6.32's exactness gate; free-running generation over
~500 autoregressive steps landing on the same termination frame every time).

On speed, two instruments and neither can see it:

```text
    isolated op          default 92.9 us / 144 GB/s   vs   8x4 ib=2  63.7 us / 210 GB/s
                         -> predicts +0.76 ms/frame over 26 layers if the cost were real
    whole 26-layer step  +0.174 ms  (spread 1.7-1.9, resolution ~0.3)   INCONCLUSIVE
    pipeline, 15x3       -0.81 / +1.20 / -2.11 ms per seed, mean -0.40   signs flip
```

The isolated cost is entirely overlapped away, the same way §6.27 found isolated ops summing to
29.9 ms against a 23.8 ms step. **A sweep also found configs far faster than either** — 12x2
`in0_block_w=8` reaches 354 GB/s against the shipped 210 and a ~360 GB/s ceiling — and that too
produced zero whole-step change. The isolated ranking of this op does not transfer at all.

One thing worth keeping from §6.25's method: every config in that sweep sat at the **same
2.854e-04 from an fp64 reference** as the default, several better, while none was bit-identical
to it. The shipped `ib=2` was chosen on bit-equality-vs-default — the criterion §6.25 itself
warns is not a correctness test.

---

**N150 RECORD BELOW — historical on this fork.**

```text
 wo's DECODE program config, and it is the only linear that needs one. Each decode linear splits
 into bytes/ceiling plus overhead, and only wo carried any: 85.9 us against a 68.9 floor, 20% of its
 time, where wqkv/w1/w3/w2 all sit at 98-103% of the ceiling with nothing to reclaim.

 THE KNOB IS in0_block_w, i.e. how many of K's 128 tiles are loaded per inner iteration. The default
 picks something that runs at 162 GB/s; 4 runs at 196, which is the ceiling. Both directions lose:

     in0_block_w    1      2      4      8     16     32
     us           152.0   83.2   68.1   73.7   80.5   89.4
     GB/s           88    161    196    182    166    150

 THE BIT-EXACT ONE SHIPS. in0_block_w=2 at 8x4 is 74.7 us (1.102x, +0.196 ms/frame) and byte-identical
 to the default on all six decode-gate columns. in0_block_w=4 is faster still -- 68.1 us, 1.209x,
 +0.370 -- but not bit-exact: mean/p90 worst-sample +0.02/+0.01 pp and max 3.05% -> 4.76%, which is
 0.054 pp of mean per ms, the same ballpark as the w2 trade rejected in STATUS.md 6.16. Change the two
 numbers below to take it. The core count barely matters either way (8x8 68.2, 8x6 68.1, 8x4 68.5);
 my original theory was that N=3072 splitting 1.5 tiles per core over 64 cores was the problem, and
 the sweep says otherwise.

 DECODE ONLY. per_core_M=1 assumes M is a single tile, which is true of one decode position and false
 of prefill's S rows -- see _layer, which deliberately has no program config.

 The earlier "hand-tuned matmul program configs measured SLOWER (169 vs 193 GB/s)" finding stands: it
 was measured on wq, which is already at 94% of its floor. Sweeping an op with no headroom finds none.
```

### [gpt-27] `_layer_step` / `_mlp` — the residual rides in as the matmul's bias

Both Block 1 residuals become `linear(..., bias=x)` instead of `add_(x, linear(...))`.
**−1.918 ms/step** against a 0.190 noise floor, WER and long-form MOS unchanged.

`bias=` is genuinely fused, +0.1 µs over a plain matmul — unlike `activation=`, which never fused
at all ([gpt-26]). **§6.47 rejected this at +0.069 ms and was right at the time**: the `wo` matmul
then took 92.7 µs and the separate add hid in its shadow at +2.5. [gpt-26] made it 40.3 µs and
exposed the add at +53.5. A correct measurement whose premise expired — see STATUS.md §6.62.

**DECODE ONLY.** A matmul bias is a row vector broadcast across rows, so it is the residual only
at M=1. Prefill has many rows, each with its own residual, and would be silently wrong. Block 2's
3-or-6 CFG-folded rows are the same hazard, which is why `_block` keeps the explicit add. Gated on
`prg` being non-empty, which is true only on the decode path.

### [gpt-26] decode matmul program configs — and `activation="silu"` is not fused

Every decode matmul in **both** blocks takes a `MatmulMultiCoreReuseMultiCast1DProgramConfig` on
one 12×6 grid. **−4.24 ms/frame** on the whole-block A/B (three runs: −4.66 / −4.38 / −4.24).

**Two independent findings, one fix.**

**(1) `activation="silu"` is NOT fused on this chip.** On the real w1 shape it measures 98.8 µs
against a plain matmul's 85.5 — the same **+14.9 µs** as writing `ttnn.silu()` as its own op,
which is evidently what it does. `UnaryWithParam` and `UnaryOpType` spellings behave identically
(100.6 / 100.2). **Only** a program config's `fused_activation` folds it in, at 88.1 µs. Across
the 47 w1 calls per frame that single op is worth **2.42 ms**, and it is *more* accurate besides
(PCC 0.9999984 vs 0.9999970) because the value never leaves the dest registers.

This invalidates the inherited claim in **[flow-12]**, which said SiLU "rides along on the w1
matmul instead of being its own op". It was its own op the whole time, on both chips.

**(2) The ttnn heuristic collapses on deep reductions.** Achieved bandwidth by K-tiles:

| matmul | Kt | default | tuned `in0_block_w` |
|---|---|---|---|
| `w1`/`w3` | 96 | 352 GB/s | 362 |
| `wqkv` | 96 | 281 | 346 |
| `wo` | 128 | **144** | 336 |
| `w2` | 288 | **147** | 355 |

At Kt=96 the heuristic is already near this chip's measured 367 GB/s ceiling; at Kt=128/288 it
delivers **under half the memory system**. §6.50 predicted exactly this shape of result — *"it was
measured on wq, which is already at 94% of its floor; sweeping an op with no headroom finds
none."* `w2` and `wo` are where the headroom was.

**ONE grid for every shape.** A per-shape set of isolated winners pinning two grids measured
37.04 ms against uniform 12×6's 36.99, under a 0.070 noise floor, so no shape earns its own grid.
13×10 (the full 130 cores) was 0.31 ms **worse**.

**THE ISOLATED NUMBERS ABOVE DO NOT PREDICT THE BLOCK, AND THE DIRECTION OF THE ERROR REVERSES.**
`w2` and `wo`, with 2.4× isolated wins, delivered **0.00 ms** in the block. `w1`, with a 1.03×
isolated win, delivered **2.42 ms**. The mechanism: a tight loop of *identical* ops pipelines, so
an isolated microbenchmark understates op cost — the silu op costs 12.2 µs isolated and ~54 µs
in-block (2.42 ms ÷ 47), near the full ~68 µs floor. This is §6.43's rule with a cause attached:
**isolated sweeps find pipelining, blocks find dispatch.**

**DECODE ONLY.** `per_core_M=1` and `fuse_batch=True` assume one tile of rows — true for Block 1's
1 and Block 2's 3-or-6, false for prefill. `_mlp` is shared with prefill, so the configs are
passed **in** as an argument rather than read from module scope; prefill passes nothing and keeps
the heuristic. `test_decode_matmul_configs_assume_one_tile_of_rows` guards this.

**Not bit-exact**: a paired `--gate codes` run reads 85/288 acoustic against the baseline's
86/288 (semantic 1 both). Codes move, so the frame-count A/B applies — see STATUS.md §6.52.

### [gpt-25] `_mlp` / `_layer_step` — in-place elementwise, worth 0.929 ms/step

The three elementwise ops in the MLP tail — `multiply(g, w3_out)` and both residual adds — run
in place. **+0.929 ms/step, bit-identical**, and the frame-count A/B confirms it end to end
(cases 0/2/3 reproduce 68/452/493 exactly).

**§6.37 measured this at +0.001 ms on the N150**, i.e. nothing, and it is ~1000x that here. The
reason is that in-place removes an ALLOCATION, not a launch: on Blackhole an op costs ~65 us
against the N150's ~20 (§6.45), and roughly 12 us of that is the allocator. Whatever the parent
branch measured as noise is a real line item on this chip.

**SAFETY, since in-place is where aliasing bugs live.** `add_(x, …)` mutates the layer input,
which is safe at both sites because `x` is dead the moment the layer returns — layer 0's comes
from a fresh `from_torch` per frame, layers 1–25's from the previous layer — and `_norm(x, …)` in
`_layer_step` is evaluated as an argument, i.e. BEFORE `_mlp` can mutate anything.
`multiply_(g, …)` mutates a fresh matmul output that has no other consumer. `_mlp` is shared with
prefill and the same argument holds there.

**A HARNESS TRAP THIS WALKED INTO, and §6.37 documents the same one.** The first measurement
showed in-place at maxabs 2.4e+01 — apparently broken. It was the benchmark: the timing loop
reused one `x0` across iterations, so layer 0's `add_` ate it. §6.37 hit this from the other side
and wrapped the operands in `ttnn.clone`, which made in-place look SLOWER. Clone the input per
iteration, not the operands.

**RESIDUAL-AS-BIAS WAS TESTED IN THE SAME PASS AND REJECTED.** `linear(a, wo, bias=x)` is
expressible (decode has M=1, so the residual is exactly a row-vector bias) and bit-identical, but
worth +0.069 ms for `wo` and +0.076 for `w2` against a 0.062 ms noise floor — and doing BOTH is
−0.148, i.e. worse than shipped. It is also anti-complementary with in-place: bias removes the
very adds that in-place accelerates, so the combination (+0.229) is worse than in-place alone
(+0.929). §6.27's N150 verdict therefore stands here, for a different reason. STATUS.md §6.47.

### [gpt-21] `_SDPA_PRG` — sdpa_decode's program config, and why only a position sweep is safe

```text
 sdpa_decode's program config. The shipped call passed none, and the default spends almost all its
 time on setup rather than on the cache: at pos=312 it reads 1.22 MB (a 6.6 us floor) in 68.6 us, and a
 31x bigger cache costs only 21% more time -- ~62 us of the 68.6 is FIXED. So it was never a bandwidth
 problem, and k_chunk/grid reach it:

     default            68.6 us          k=512 8x2   42.2 us  1.63x  <- this
     k=256 8x2  38.7 us (1.77x)          k=512 8x1   40.2 us  1.72x

 THE FASTER ONES ARE NOT SAFE, and only a position sweep shows it. Bit-exact vs the default at 11
 positions from 64 to 1000, spanning the k_chunk boundary at 511/512/513:

     k=512 8x2   11/11 exact   0 worse than default vs fp32   0.673 ms/frame   <- ships
     k=512 8x1    6/11         0 worse                        0.706 ms/frame
     k=256 8x2    3/11         3 WORSE than default vs fp32   0.825 ms/frame

 k=256 looks fine at pos=312 and degrades at 480, 511 and 700. The decode gate pins ONE position per
 case, so it would not have caught that -- ship a config only if it is exact at every position.

 DECODE ONLY: prefill computes attention with explicit matmuls in _attend, not sdpa.
```

### [gpt-22] `_mlp` — w1 and w3 stay separate; matmul bandwidth collapses past N~9216

```text
 w1 and w3 stay SEPARATE. Fusing them into one 3072x18432 weight is 4x SLOWER, not the
 small loss on record: identical 57.4 MB either way, but 48 GB/s fused against 192 separate.
 Matmul bandwidth collapses between 9216 and 18432 output columns -- 3072, 6144 and 9216 all
 hold the ceiling. Assume any fusion past N~9216 collapses. STATUS.md 6.24.
```

### [gpt-23] `_layer_step` — two rope calls, not ttnn's fused q+k rope

```text
 Two calls, not ttnn's fused q+k rope: that one implements the INTERLEAVED convention via a
 trans_mat, and our wq/wk are permuted to HALF-SPLIT at load. Measured 0.236 ms/frame for
 reverting that permute, disjoint q/k cores and losing bit-exactness -- STATUS.md 6.23.
```

### [gpt-24] `_layer_step` — one fused KV write instead of two

```text
 ONE fused write, not two: 26 launches a frame instead of 52, worth 0.405 ms and
 bit-identical. V is moved to core (1,0) first because the op refuses an overlap -- see
 _V_SHARD for why V and not K.
```

### [gpt-03b] `_layer_step` — no memory_config move on sdpa's output

```text
 No memory_config move here: sdpa_decode already emits o as interleaved DRAM, which is what
 the wo matmul wants. Routing it to L1 instead is NOT the win it looks like -- 0.999x, and
 the reason is worth reading before trying it: NOTES.md [gpt-03].
```

### [flow-21] `_solve` — project and reshape the llm conditioning once per frame, not per step

```text
 llm conditioning is constant across the solve, so project it ONCE rather than per step
 (it was n_steps identical 3072x3072 matmuls). Reshaped once here for the same reason:
 _trunk wants [B,1,3072] and p2 changes once a frame, not once a step -- NOTES.md [flow-19].
```

### [codec-20] attention bias — ALiBi + causal + sliding window as one additive term

```text
 Attention bias: ALiBi + causal + sliding window as ONE additive term.
 [1,H,S,S] on the unchunked path; [1,H,slab,slab] once chunking applies, which is the
 normal case and is why it stays small (4.2 MB) at any utterance length.
```
