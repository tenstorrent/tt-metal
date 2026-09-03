# Qwen3-8B long-context decode on Wormhole N300

What was tuned, what it bought, and what was tried and rejected — for generating tokens
one at a time from a 32,768-token context on an N300 board (two Wormhole chips).

All numbers here are measured on that configuration unless stated otherwise. Every
per-model setting is gated so that no other model's behaviour changes; see
[Configuration chosen](#configuration-chosen) for where each gate lives.

## Contents

- [Results](#results)
- [How to reproduce](#how-to-reproduce)
- [Configuration chosen](#configuration-chosen)
- [Where the time goes now](#where-the-time-goes-now)
- [Why each change worked](#why-each-change-worked)
- [What did not work: scheduling and layout sweeps](#what-did-not-work-scheduling-and-layout-sweeps)
- [What did not work: lower-precision weights](#what-did-not-work-lower-precision-weights)
- [Structural limits found](#structural-limits-found)
- [Measurement method, and the traps in it](#measurement-method-and-the-traps-in-it)
- [Open items](#open-items)

## Results

Wall clock per generated token, 32,768-token context, one user, board 0. Lower is better.

| stage | ms/token | tok/s/user |
|---|---|---|
| start of this work (`fd2c3290563`, benchmark added, nothing tuned) | 36.17 | 27.6 |
| after the scheduling and fidelity work (`563fe9b9fca`) | 32.71 | 30.6 |
| after picking the next token on the chip | 29.82 | 33.5 |
| after cutting to the real users before the chips exchange scores | **28.73** | **34.8** |

**Cumulative: 36.17 -> 28.73 ms/token, a 20.6% reduction.**

The rows are cumulative stages; the last one is what ships. Its 28.73 is the mean of a
20-run soak (range 28.67–28.77), and a confirming run of the shipped default measured
28.70. Against a baseline paired in the same session (33.14 ms/token), picking the token
on the chip is worth **3.32 ms/token (10.0%)**, and cutting before the exchange a further
**1.09 ms/token** over that soak (1.12 in the confirming pair).

Across user counts, paired in one session, before and after moving token selection onto
the chip (with the pre-gather cut). These are their own runs, so the batch-1 "after" of
28.75 is a third sample of the same configuration as the 28.73 and 28.70 above — all
within the 0.35 ms/token wall-clock spread.

| users | before | after | generated tokens identical? |
|---|---|---|---|
| 1 | 33.14 | 28.75 | 257 / 257 |
| 2 | 39.15 | 34.82 | 257 / 257 |
| 4 | 51.09 | 46.97 | 257 / 257 |

Token-for-token identical output at every batch size, which is the correctness evidence
for the change: the chip's arithmetic-maximum agrees with the host's on every one of 257
tokens, ties included.

## How to reproduce

Timing (trace on, so wall clock is meaningful):

```
TT_VISIBLE_DEVICES=0 HF_MODEL=Qwen/Qwen3-8B \
  pytest models/tt_transformers/tests/test_long_context.py -s -k 32k-b1
```

Read `decode ms` and `tok/s/u` from the end-of-run summary table. Drop `-k 32k-b1` to
sweep batch 1, 2 and 4.

Per-operation attribution (trace off — device times only; the multi-millisecond op-to-op
gaps in these captures are a profiler artefact, not real):

```
TT_VISIBLE_DEVICES=0 HF_MODEL=Qwen/Qwen3-8B python -m tracy -r -p -v -n <tag> \
  --op-support-count 40000 -m pytest \
  models/tt_transformers/tests/test_long_context.py -s -k 32k-b1 --tracy_decode
tt-perf-report generated/profiler/reports/<tag>/*/ops_perf_results_*.csv --start-signpost decode
```

`--op-support-count 40000` is required: the chip tracks 1000 operations by default, a
32,768-token prompt dispatches about 34,650, and the report is correctly refused as
incomplete without it.

Accuracy (teacher-forced against a full-precision reference, 1024 predictions taken from
the tail so they sit at contexts 31,744 -> 32,768):

```
pytest models/tt_transformers/tests/test_long_context.py -s -k "32k-b1" --accuracy
```

Attributing the per-token host cost (changes nothing about what runs):

```
HOST_STAGE_TIMING=1 TT_VISIBLE_DEVICES=0 HF_MODEL=Qwen/Qwen3-8B \
  pytest models/tt_transformers/tests/test_long_context.py -s -k 32k-b1
```

## Configuration chosen

Everything below is reached only by Qwen3-8B (or by this benchmark), so no other model
changes behaviour.

| # | setting | value | where it is set | worth |
|---|---|---|---|---|
| 1 | `sdpa_decode_k_chunk_size` | 256 | `_set_model_specific_params()`, gated on `model_type == "qwen3"` | 4.1% of attention |
| 2 | `use_tuned_decode_grids` | True | same gate; selects 8-core decode grids for the matmul inputs | 2.34 ms/token, with #3 |
| 3 | feed-forward result placement | produced directly into the next matmul's layout | `mlp.py`, gated on #2 | included above |
| 4 | decode attention accumulation | `HIFI2_NOFP32` on the three decode attention operators | `ModelOptimizations.performance()`, gated on `Qwen3-8B` | 0.139 ms/token |
| 5 | per-layer chip-to-chip collectives | 40 chunks per sync, 3 workers per link | `non_galaxy_ccl_configs`, keyed `Qwen3-8B` | 0.095 ms/token |
| 6 | next-token selection | on the chip (`allow_force_argmax: True`) | same table | 3.32 ms/token |
| 6a | cut to the real users before the chips exchange scores | `pre_slice_before_gather: True` | same table | 1.09 ms/token |
| 6b | gather for token selection | 40 chunks per sync, **4** workers per link | same table, `sampling_force_argmax` | 0.07 ms/token |
| 7 | feed-forward gate weights | 4-bit with low-fidelity arithmetic | framework default in the performance preset | pre-existing |
| 8 | KV-cache page block | 256 tokens | `PAGE_BLOCK_SIZE` in the benchmark | 0.59% |

Two of these are worth expanding.

**#6a is worth expanding, because the saving is nearly all avoided traffic.** The chip
computes on 32x32 blocks, so at one user the score block is padded to 32 rows of which
only row 0 is real. The exchange moved all of them: 9.7 MB to use 0.3 MB. It cannot simply
be cut first, because a tiled block interleaves its values and row 0 is not a contiguous
run of memory — no slice is possible below one 32-row tile. Untilizing converts to
row-by-row order, after which the cut is a real operation, and untilizing the local shard
is cheap (0.066 ms on 32x75968). So the order becomes untilize, cut, exchange 0.3 MB.

Enabled per model rather than globally. The guards in the code are generic (multi-chip,
decode, batch under 32, unpadded vocabulary), so a global default would also switch it on
for Llama-3.1-8B on Galaxy — a different board with 4 links and a ring topology instead of
1 link and a line — where it has never been run. `TT_SAMPLING_SLICE_BEFORE_GATHER`
overrides the model's setting either way: `1` forces it on, `0` forces the shipped order,
unset takes the model default.

Verified as shipped: 28.70 ms/token with it against 29.82 with it forced off, and the two
runs produced **byte-identical output** — 136,013 characters, 24,369 words, matching
hashes.

**#8 is a serving parameter, not a model one.** The framework only accepts a page block
size; it never chooses one. So this speeds up *this benchmark*, and a deployment gets it
only by setting its own block size to match. Worth passing to whoever owns serving
configuration for long-context workloads on this hardware. The measured curve, in
attention device time per two decode steps at 32k:

```
  32 -> 15,511 us    128 -> 15,321    256 -> 15,206 / 15,214    384 -> 15,324    512 -> 15,307
```

256 was run twice and reproduced to 7.7 us (0.05%), while its neighbours cluster about
105 us above it, so the dip is real. Why 256 specifically is **not understood** — run
length alone predicts 512 should win, and it does not. Trade-off: a page block is the
allocation unit, so a short conversation still occupies a whole block; good for long
context, wasteful for many concurrent short chats.

Two bug fixes are also on the branch, both of which change behaviour for other models and
are therefore worth calling out:

- **Distributed normalisation compared an enumeration value against the string
  `"decode"`**, so the comparison was always false and the decode path was never taken
  (`f7e7d71d64f`).
- **Reference generation restarted every 1,024-token chunk at position 0** on its
  HuggingFace path, with no cached keys/values and no position offset. Raising the
  requested length therefore did not produce a long-context reference — it produced N
  independent short windows, at any length. Fixed by carrying context across chunks, and
  by deriving the context-extension factor from the requested length instead of hardcoding
  4.0 (`673c6bb97c2`). Without this, no accuracy verdict at 32k was obtainable at all.

## Where the time goes now

Per-operation attribution at 32k, one user, from the profiler. **Read these as shares, not
as a device-time total**: profiling runs with trace disabled, which inflates the absolute
figures (they sum to 30.74 ms/token against a measured 28.73 ms of wall clock with trace
on).

| operation | ms/token | share |
|---|---|---|
| weight multiplications (all five groups) | 13.018 | 42.4% |
| attention over saved history | 7.565 | 24.6% |
| picking the next token on the chip | 2.182 | 7.1% |
| chip-to-chip reduce-scatter | 1.954 | 6.4% |
| chip-to-chip gather, per layer | 1.702 | 5.5% |
| normalisation (width-sharded) | 1.054 | 3.4% |
| chip-to-chip gather, for token selection | 0.798 | 2.6% |
| normalisation (interleaved) | 0.550 | 1.8% |
| everything else | 1.913 | 6.2% |

Host cost per token, measured in place with the `HOST_STAGE_TIMING` probe. The probe skips
no work, so the chip's work is identical; it splits waiting for the chip from moving bytes
by draining the queue first.

| host stage | before (host picks) | after (chip picks) |
|---|---|---|
| rebuild the next token's inputs | 0.611 ms x 258 calls | 0.601 ms x **3 calls** |
| copy those inputs to the chip | 0.818 ms x 257 calls | 0.146 ms x **2 calls** |
| transfer scores back | 2.593 ms | **0.210 ms** |
| host scans for the maximum | 1.786 ms | **0.142 ms** |

Two things are visible there. The transfer and the scan collapse because a single token
identifier comes back instead of 9.7 MB of scores. And the call counts for the first two
stages fall from once-per-token to three and two **for the whole run** — that is metal
trace: once the chip picks the token, the recorded command stream replays without the host
in the loop at all.

## Why each change worked

**Picking the next token on the chip (3.32 ms/token, plus 1.09 with the pre-slice
switch).** Qwen3-8B has a 151,936-entry vocabulary, which is 75,968 per chip. The
framework's top-k selection accepts at most 65,536 entries in one call, because its
multi-core sort addresses elements with 16-bit indices. Over that width there is no
top-k route, and selection fell all the way back to the host — meaning a full score
readback every single token. Greedy decoding never needs top-k: it needs only the
position of the largest value, and the arithmetic-maximum operator has no width ceiling
because it untilizes in tile-aligned chunks. So the fix was to permit the
maximum-only path for this model, which skips top-k entirely. Measured alternatives at
this width: maximum 1.38 ms/token, top-k 22.46 ms, host round trip 5.07 ms.

Two implementation notes that cost real debugging time:

- The result must be written **directly into the caller's output tensor**. A three-step
  pad/reshape/copy write-back is correct when run eagerly and returns zeros under trace
  replay. This produced 252 correct tokens out of 257 at full speed — only a token-by-token
  comparison caught it.
- Non-greedy requests at this width still have no on-chip route. That is surfaced to
  callers rather than silently taking the 22.46 ms path.

**Fixed 8-core decode grids and feed-forward result placement (2.34 ms/token).** These two
are one change. The gate matmuls want *few* cores: the core count sets how large a bite the
inner loop takes, so 8 cores give a bite of 8 where 64 give 2 — measured 95.8 -> 63.3 us
per call. But the elementwise combination that consumes their results wants *many* cores:
it is pure data movement with no arithmetic to hide latency, measured 3.3 us on 64 cores
against 21.8 on 8. Originally the second inherited the first's placement, so moving the
matmuls to 8 cores dragged the combination down with them and cost 1.99 ms/token — wiping
out the 2.35 ms gain. Producing the result directly into the layout the *next* matmul wants
decouples them and costs nothing extra, because the conversion that followed wanted that
layout anyway and becomes a genuine no-op.

**Dropping 32-bit accumulation in the three decode attention operators (0.139 ms/token).**
A 32-bit running total occupies twice the destination-register space, so 4 tiles fit per
pass where 8 otherwise would. Top-1 agreement at 32k is 85.74% with this against 84.18%
for the previous configuration, so it costs nothing measurable at the depth this model
serves. Applied to decode only; prefill is deliberately unchanged.

**The gather for token selection, tuned separately (0.07 ms/token).** This one call
moves 2.58 MB across the single chip-to-chip link once per token — roughly 20x what a
per-layer collective moves — and it was running the same settings as everything else.
Swept at that exact size over chunks {2,5,10,20,40} x workers {1,2,4} x buffers {2,4}
over 40 round-robin rounds: 40/4/2 runs 303.96 us against 416.23 for 10/2/2. In the
model, 0.41 -> 0.34 ms/token with every other operation flat. It is deliberately scoped
to this one call, because the per-layer collectives want the opposite direction — they
are setup-bound where this is bytes-bound, and applying the settings that looked best
for them standalone cost +0.66 and +0.03 ms/token in the model.

**Per-layer collectives at 3 workers per link (0.095 ms/token).** The per-layer gather went
1.770 -> 1.705 ms/token and the reduce-scatter 1.998 -> 1.953. That 0.095 ms is 5.6x the
0.017 ms spread between two otherwise identical profiling runs. The chunks-per-sync value
is flat once workers reach 3: 20, 40 and 80 were indistinguishable over 14 alternating
rounds, so 40 is an arbitrary pick among them.

## What did not work: scheduling and layout sweeps

Eleven items, none kept. Recorded because the negative results are what stop them being
re-tried.

| item | expected | measured | why |
|---|---|---|---|
| fast exponential in attention's softmax | the operator's own default is on | **+0.31 ms/token**, 0/3 paired wins | the model's override to *off* is correct for this shape; output was byte-identical over 256 tokens either way |
| attention output kept on-chip | 0.031 ms ceiling | **blocked** | `Sharded output not supported for GQA` — the permissive check is at one line, the guard that rejects it whenever a chip holds more than one key/value head is 300 lines later. This model holds 4. |
| reduce-scatter staging buffer to on-chip memory | claimed 1.95 ms | **-0.012 ms** | under the 0.017 ms floor; the claim had mis-sized the buffer as the result rather than the staging area |
| token-selection gather moved from its tuned 40/4 to the per-layer 40/3 | small | **+0.001 ms** | already at its own optimum; the per-layer settings do not transfer to it |
| chip-to-chip send buffer depth 2 -> 4 | unknown | **+0.067 ms** | — |
| chip-to-chip send buffer depth 2 -> 1 | unknown | under floor | shipped depth 2 stands |
| query chunk size auto -> 32 | ~0 | **+0.002 ms** | inert at one query row, as predicted |
| normalisation: full destination sync + wider bite | **-0.2 to -0.5 ms** | **+0.129 ms** | sign was predicted wrong; the two halves are confounded and were not separated, ceiling was too small to justify another run |
| KV pass size 128 / 512 / 1024 | larger should be better | 256 best (7.581 vs 8.023 / 7.637 / 7.750 ms/token) | fixed cost is 1.41–1.57 ms/token at *every* pass size, so pass count is not what it is made of |
| attention on 32 cores instead of 64 | **-0.106 ms/token**, 7/7 paired wins in isolation | **+0.159 ms/token in the model** | see below |
| fusing the two feed-forward gate matmuls | -0.178 ms/token | **+0.202 ms/token** | see below |

Four more were ruled out by reading the device operators rather than by measuring:
sharding the KV cache (three operators assert interleaved-only, and the width is 128, not a
multiple of the 12 memory banks); changing the sharding strategy (both operands must be
width-sharded on the memory-sharded path — a different program configuration is a different
algorithm, not a sweep); more chip-to-chip links (the board table caps N300 at 1 per
direction, which is already set); and the weight prefetcher (requires Blackhole; this is
Wormhole).

### The two instructive failures

**Attention on 32 cores.** Attention's cost splits into a context-independent part and
streaming. The context-independent part scales almost linearly with core count (64 cores
1.464 ms/token, 32 cores 1.265, 16 cores 0.811, 4 cores 0.447), which points at combining
each core's partial result. Streaming saturates first: 32 cores already reach 205.0 GB/s of
64 cores' 209.3. So halving the cores should cost ~2% of bandwidth and buy back half the
combination cost — and in isolation it did, winning 7 of 7 and then 5 of 5 alternating
rounds. **In the model it lost by 0.159 ms/token.** The cause is visible in the profiler's
own label: in the model the query arrives height-sharded, so its placement is tied to the
core grid, and changing the grid moves the query relative to the compute cores. The
isolated benchmark handed the operator an interleaved query and structurally could not see
this. The two profiler arms agreed on every other operation to 0.3 us, so the reversal is
real and not noise.

**Fusing the two feed-forward gate matmuls.** They are two calls of identical shape sharing
one input, so fusing them into one wider matmul removes 36 calls per token. Measured device
times: one 60 us, fused 118 us — so the fused call saves **2.00 us per layer**, not the
4.95 us the cross-shape fit suggested. Meanwhile the halves must be separated afterwards, at
**3.80 us per slice, twice per layer**. The split costs 3.8x what the fusion saves, for a net
**+0.202 ms/token**. The reason is placement again: today both results are width-sharded
across the same 64 cores, so core *j* holds column-block *j* of both and the combination is
entirely local. Fused, the first half lands on cores 0–31 and the second on 32–63, so
pairing them requires movement that does not exist today. **Even with a completely free
split the ceiling is 0.072 ms/token (0.25%)**, so no cleverer separation rescues it; it
would take a new device operator (a matmul writing two outputs, or a fused gated activation
consuming one concatenated tensor), which is not worth writing for that ceiling.

## What did not work: lower-precision weights

Weight streaming is what a single-user decode step is made of, so fewer bytes is the one
lever with large numbers behind it. All four were measured for both speed and accuracy;
accuracy is top-1 agreement over 1024 predictions at contexts 31,744 -> 32,768 against a
full-precision reference.

| change | ms/token saved | top-1 | delta | **points lost per ms** | verdict |
|---|---|---|---|---|---|
| baseline (8-bit) | — | 85.74% | — | — | — |
| vocabulary weights -> 4-bit | 0.470 | 85.64% | **-0.10** | **0.21** | free, but not shipped (see below) |
| three weight matrices -> 4-bit | 2.426 | 82.23% | -3.51 | 1.45 | open judgment call |
| saved history -> 4-bit | 2.562 | 52.15% | **-33.59** | **13.11** | **abandoned** |
| all of the above together | 5.461 | 49.61% | -36.13 | | not viable |

Damage is additive (bundle -36.13 against a sum of parts of -37.20), so dropping one group
recovers its cost.

**The saved history at 4 bits is the one to never revisit.** It looked like the best
candidate on speed and it is the worst trade by a wide margin — 9x worse per millisecond
than the weight matrices and 62x worse than the vocabulary. The reason is structural: it is
the only change whose error *accumulates with context*. Every one of ~31,744 stored entries
is degraded, and attention reads all of them at every step. Ranking these by speed alone was
the mistake; cost per millisecond spread **62x** across them while their speeds spread only
5x.

**The vocabulary weights at 4 bits are free and still not shipping.** -0.10 points is
exactly one prediction in 1024 — the smallest non-zero value the test can report — and
top-5 agreement actually improved (99.41% against 99.22%). It is blocked on plumbing, not
merit: the setting is applied as a constructor argument rather than through a per-tensor
group, so it bypasses the weight-cache identity check. A warm-cache run would build those
weights from uninitialised memory and silently ship garbage. **It must be routed through a
real per-tensor setting before it can ship.** This is the best-value item still open.

## Structural limits found

Both of the two large consumers were measured against the hardware's limits, and both are
close enough to them that scheduling work on either is finished.

**Weight multiplications are at 89.4% of peak memory bandwidth.** Fitting the profiler's
own timings against true byte counts across six matmul shapes:

```
  us/call = 4.95 + MB / 257.3 GB/s          (the chip's peak is 288 GB/s)
```

Effective bandwidth rises strictly with call size — 222.8, 235.9, 238.7, 245.3, 254.5,
255.9 GB/s — which is the signature of a fixed per-call cost, not of wasted bandwidth. The
total fixed cost is 0.901 ms/token across 182 calls, and the only way to reduce it is fewer
calls, which the fusion experiment above measured as a net loss. Note also that **12 cores
is structural, not a knob**: the memory-sharded matmul computes on the 12 cores adjacent to
the 12 memory banks and redistributes from there regardless. Arithmetic use sits at 54–62%
while bandwidth is at 89%, so memory is unambiguously the binding constraint — even though
at one user 31 of every 32 rows in a tile are padding and 97% of the arithmetic is wasted.

**Attention's streaming is also at its limit.** A 5-point context sweep (4k to 64k) at one
fixed number format fits `ms/call = 0.0414 + MB / 209.3 GB/s`, residuals under 2.8%. That
209.3 GB/s is faster than any plain reduction that could be run on the same chip
(121–186 GB/s). The 0.0414 ms/call — **1.489 ms/token** across 36 layers — is the
context-independent combination of per-core partial results discussed above, and the
shallow optimum means roughly 0.1 ms of it is reachable in principle and none of it in the
model.

A useful cross-check for the byte accounting: the profiler's own bandwidth figures
**undercount by 6.25% for 8-bit and 12.5% for 4-bit weights**, because it charges 1.0 and
0.5 bytes per weight and ignores the shared block exponent (really 1.0625 and 0.5625). Its
"% of peak" column is correspondingly pessimistic.

## Measurement method, and the traps in it

- **Run-to-run spread is 0.017 ms/token on device time and 0.35 ms/token on wall clock.**
  Anything under ~0.4 ms cannot be resolved by a single whole-model A/B; use the profiler's
  per-operation numbers, or paired alternating rounds.
- **Device time and wall clock are different numbers and must not be mixed.** This work
  began because commit messages quoted 36.0 tok/s/user (derived from device time) while the
  benchmark printed 30.4 (wall clock) for the same run. That 5.5 tok/s/user disagreement
  *was* the host gap — 5.07 ms/token, 15.5% of the 32.75 ms token it was measured in — and
  finding it produced the largest single win here. A formula in the benchmark that mixed a
  device-time intercept with a wall-clock slope has been corrected for the same reason.
- **An isolated benchmark can reverse a ranking.** It happened twice, in both directions:
  the 32-core attention grid won in isolation and lost in the model, and the fusion harness
  was host-bound and reported the combination at 101 us where the model measures 3.3. Both
  times the fix was to measure device time in the model. Treat isolated results as
  exploration; confirm in the model before believing a ranking.
- **A third instance, found earlier in this work:** tuning the per-layer collectives
  standalone was meaningless because those operations cost ~24 us each in the model while
  the standalone harness added ~47 us of its own per call, burying the differences. Only
  the one transfer large enough to survive that overhead gave a usable answer. Isolated
  harnesses have now misled this work three times, in three different ways: harness
  overhead swamping the signal, host-bound throughput replacing device time, and a
  memory-layout difference reversing a ranking.
- **Validate an isolated benchmark against a known in-model figure before trusting it.** The
  attention sweep reproduced 7.612 ms/token against the model's 7.561 (0.67%), and the
  fusion profile reproduced 60 us per gate matmul exactly. Without that check neither
  number would mean anything.
- **A control that does not exercise the code path is not a control.** An early probe
  claimed a 7.95 ms saving; its control ran a different, untraced code path, so the probe
  never activated in it, and it also froze the position counter, reducing device work.
- **The profiler needs `--op-support-count 40000`** at this context, and the join then takes
  about four minutes rather than seconds.
- **Kill stranded device processes before blaming a benchmark.** Two runs failed with
  hugepage allocation errors that were caused by an earlier stranded process of this work's
  own holding the chip and its memory.

## Open items

1. **Route the 4-bit vocabulary weights through a real per-tensor setting** so they feed the
   weight-cache identity check, then ship them. Measured at 0.470 ms/token for -0.10 points.
   Best remaining value on this model.
2. **Decide on the three weight matrices at 4 bits**: 2.426 ms/token (8.7%) for -3.51 points.
   Top-5 barely moves (98.44% against 99.22%), so it is the top pick that wavers rather than
   the right answer leaving contention. This is a product call, not a measurement question.
3. **An intermittent hang**, seen once in roughly 48 runs, in the cross-chip gather for token
   selection during prefill. A 20-run soak was clean, putting it near 2% (95% confidence
   0.05%–7.7%). Not transfer-size dependent — all ten large-transfer runs passed. The board
   survived the timeout-based kill.
4. **Time to first token is 12.9 s at 32k and has never been investigated.** Out of scope
   here because this work targets decode, but noted: there is no Qwen3-8B row in the
   prefill chunk-size table, so it falls back to the smallest value (4,096 tokens per pass,
   8 passes) while every comparable model on this board is set 8–16x higher. The framework
   itself warns about this at startup.
5. **Report a framework bug**: a KV pass size of 32 silently returns an all-zero tensor. It
   equals the KV page size and appears untested upstream. Not yet filed.
6. The 64k-context accuracy runs have not been done; the saved-history result should get
   *worse* with more context.
