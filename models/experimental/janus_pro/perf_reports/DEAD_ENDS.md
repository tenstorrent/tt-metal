# Dead ends

Companion to [PERF.md](../PERF.md). Everything here was tried on a Wormhole N150 and is **not** in
the tree, kept so nobody spends a day rediscovering it.

## How to read this

Every entry names the **stage** it was tried at — the row number in
[PERF.md's change log](../PERF.md#change-log), whose `kernel after` column gives the tower's state
at that moment. That is not bookkeeping. The tower's bottleneck moved as the work progressed, so
three of the levers below lost at one stage and *paid* at a later one; see
[Superseded](#superseded--lost-once-paid-later). A negative result here is evidence about a
tower at that kernel time, not a law.

Two caveats on the numbers:

- **The metric changed mid-campaign.** Stages up to 13 were driven on **span** (kernel time plus
  op-to-op latency); 14 onward on **kernel time**, which is the stabler of the two. Percentages
  below are against whatever the session measured, so a delta and the stage's `kernel after` are
  not the same denominator. PERF.md's *Vocabulary* section defines both.
- **Rows marked `pre` predate the campaign** and were measured on a different test
  (`test_vision_model.py`) with a different metric (DEVICE FW). They are the weakest evidence in
  this file, and one of them turned out to be wrong.

## Deliberately absent: the largest win found anywhere in this work

### bfloat8_b residual stream

Tried at stage 11. **14.4992 ms span against 15.8809, −8.70%** — the largest single improvement
measured in the whole project, and it is deliberately not in the tree.

Most of it was not the adds. Matmul alone gave −0.98 ms of the −1.38: both layer norms inherit
their output format from the residual, so the in0 multicast for `qkv` and `c_fc` halved.

It fails `test_vision_transformer` at PCC **0.9765 against that test's 0.99 gate**; reverting it
alone restores 0.99888. End-to-end fell to 0.98925 from 0.99537. The tower's own 0.95 gate
survived at 0.96145 — the tower metric was never the binding one, the per-module suite was.

Why it fails where every other bfloat8_b tensor passes is the **read-once rule**: the residual is
the one tensor summed across all 24 layers, so quantization error compounds. Every surviving
bfloat8_b tensor has exactly one consumer.

To reproduce: pass `dtype=ttnn.bfloat8_b` to both `ttnn.add` calls in
`janus_pro_image_block.py:forward` and change nothing else.

### It is not a binary — a suffix of it landed as change 31

**PCC is monotonic in how many layers carry the bfp8 residual**, so there is a boundary rather
than a cliff. Change 31 takes the last 12; the full sweep is in
[31-bfp8-residual-last12.md](31-bfp8-residual-last12.md) and
[CANDIDATES-2026-08-20.md](CANDIDATES-2026-08-20.md).

| suffix length | `test_vision_transformer` | |
|---:|---|---|
| 6 | 0.997891 | pass |
| **12** | **0.996674** | **landed as change 31** |
| 18 | 0.991522 | passes, and buys −0.089 ms, but leaves 1.5e-3 of the gate |
| 19 | 0.989845 | fail |
| 24 | 0.9765 | fail |

18 was measured and **deliberately not taken**: it spends 7.2e-3 of the gate's 8.8e-3 of slack
for 0.13% more tower time than 12 does, which prices every later encoder change out of the same
budget. 12 leaves 6.7e-3.

Also worth recording: scaling to all 24 layers gives roughly **−1.3%**, against the **−8.70%**
this section reports from stage 11. The lever did not shrink, the tower did — stage 11 measured
15.88 ms on span, before bfp8 weights, LoFi and the output sharding. Read the caveat at the top
of this file about a delta and a stage's `kernel after` not sharing a denominator; this is its
largest instance.

### Why it has to be a suffix

Interleaving is the obvious way to spend fewer layers, and it is **strictly worse than a suffix
of the same length** — at 12 layers it does not merely lose, it fails the gate:

| 12 of 24 blocks bfp8 | `test_vision_transformer` |
|---|---|
| suffix, blocks 12-23 | **0.996674** pass |
| every second block, 1,3,…,23 | 0.980624 **fail** |
| every second block, 0,2,…,22 | 0.977422 **fail** |

Both interleavings are worse than the *18*-block suffix despite narrowing six fewer blocks, and
the even one is close to all-24's 0.9765. Two mechanisms, and neither is about how many
quantizations happen — that count is 12 either way:

- **A bfloat16 add cannot restore bits the running sum has already dropped.** There is no
  recovery between steps, so alternating buys nothing; the intuition that a wide layer "repairs"
  the narrow one before it has no mechanism behind it.
- **Position dominates count.** Marginal cost of narrowing a six-block window: blocks 18-23 cost
  8.7e-4, blocks 12-17 cost 1.27e-3, blocks 6-11 cost **5.10e-3** — roughly six times, because
  an early block's error propagates through every block after it. Any interleaving reaches into
  block 0 or 1 and pays that rate. Starting at 1 rather than 0 is worth 3.2e-3 on its own, which
  is the same effect measured at one block of resolution.

**Where the 0.99 comes from: nowhere.** `test_vision_transformer.py:36` is
`pcc_required = 0.99`, bare. `git log --follow` traces the file to
`models/demos/multimodal/gemma3/tests/vision_tests/test_vision_transformer.py`, added by
**PR #26924 "Add Experimental Support for Gemma-3-4b-it"** (`85a33e8a34d`), where the same line
already reads `pcc_required = 0.99` with no comment and no derivation. Janus's copy inherited it;
`3cff7544b5c` only moved it. So the constant was chosen for **Gemma-3-4b's** SigLIP tower, not
this one.

The repo's gates are a tier convention by module depth, not per-model measurements:

| tier | tests |
|---|---|
| 0.9999 | aligner, patch embedding, vision embedding, decoder RMS |
| 0.999 | rope |
| 0.99 | **transformer**, transformer block, attention, mlp, layernorm, decoder |
| 0.95 | tower, vision model, pipeline, e2e |

That is the whole provenance. It is **not** a reason to move it — a gate with no derivation is as
likely to be too loose as too tight, and nothing here measures what this tower's downstream
actually tolerates. It is a reason to know that "0.99" carries no information about Janus, and
that the case for the bfp8 residual rests on a downstream measurement nobody has taken, not on
arguing with 0.99.

## Superseded — lost once, paid later

The most useful entries in this file, because each says the same lever is worth re-testing after
the surrounding cost structure changes — or in the last case, in a different geometry.

| lever | first tried | then | what changed |
|---|---|---|---|
| math fidelity below HiFi4 on the body matmuls | `pre`: 178 vs 177 us on the same shape, dismissed as inert | **change 22 took LoFi and gained 15-17%** | the matmuls were reader-bound at first, so math time never surfaced. Once bytes and dispatch had been cut it did |
| L1 block-sharded in0 on `qkv` | stage 17: +14.1% on that op, +2.1% on the tower, every validate passing | **change 23 shipped the same idea** | by then the *unshard* it removes had become the larger cost. The mcast fan-out was never cheap; the thing it replaced got more expensive |
| `c_fc` 1D → 2D reuse | stage 21: 10.923 vs 10.897, flat | **change 24 shipped it** | flat on its own, but 2D is what lets `c_fc` read `ln_2`'s shard in place. The gain was never in the matmul |
| sharding `c_fc`'s output | stage 19: **+1.6%** width-sharded | **change 26 took -1.42% block-sharded** | nothing about the tower changed — the *geometry* did. Width-sharding gave each core 18 rows x 2 columns; block-sharding gives 3 x 16. A rejected idea can be a rejected shard spec |

## Measured and rejected

Each row is a device measurement, not an opinion.

| stage | lever | measured | why it lost |
|---|---|---|---|
| 29 | the aligner's compute config `hifi2` → **`hifi2_fp16`**, i.e. `fp32_dest_acc_en` off at the same fidelity | **−0.036 ms (−0.39%)**, aligner fc1 235.8 → **213.4 us**, and aligner PCC 0.999955 → **0.999910** | it pays, and it is not in the tree. Against a **0.9999** gate that is 82% of the remaining 5.5e-5 of slack spent on 0.39%, leaving 1.0e-5 — no later aligner change could be measured against it. The only lever here whose cost is a gate rather than a clock. Its diagnostic value is kept in [PROFILER_NOTES](PROFILER_NOTES.md#the-aligner-characterised): the aligner's 3.27x against `c_fc` at an identical shape and config is mostly this flag |
| 29 | the sharded layer norm's math fidelity at **LoFi** | tower PCC **0.937403** against its 0.95 gate | the one LayerNorm knob [not in the swept list](#where-every-knob-has-already-been-swept), and it is an accuracy wall rather than a flat result. The norm's rsqrt is where the tower's precision actually lives |
| 29 | the same at **HiFi2** | 9.298 ms, −0.013 ms, but tower PCC 0.970880 → **0.962398** | it does pay, 19.23 → 18.61 us over 49 ops, so "fidelity cannot touch a cross-core reduction" was wrong. It loses on the trade: **8.5e-3 of tower PCC for 0.14%**, where the bfp8 residual on 18 layers buys 0.96% for 6.8e-3. Worst PCC-per-microsecond of anything measured on this tower |
| 29 | `out_subblock_w` 4 → 8 on the **aligner only**, with `fp32_dest_acc_en` off so the DST bound is 8 | 9.277 vs 9.275 flat; aligner fc1 213.4 → **220.9 us**, i.e. worse | the [stage-22 result](#measured-and-rejected) again — "DST was not the constraint" — now confirmed on a **writer-bound** op, which was the one reason to retest. A wider subblock widens the write burst and the write burst was not the cost either. `get_out_subblock_w`'s hardcoded 4 is not leaving anything on the table |
| 28 | the patch projection writing L1 **interleaved**, to cheapen the position add | **9.990 ms against 9.401**, +0.589 ms | an interleaved-L1 output propagates into the residual, and from there **47 new `InterleavedToSharded`** appear and the block adds go 3.9 -> 9.7 us. The milder form of the stage-7 result below. Block-sharding the same output, with a 2D config so the grid survives, is change 29 |
| 28 | the same output block-sharded but with **no** program config | 9.452 ms, +0.051 ms | the shard spec alone is not enough: ttnn's derivation collapsed the projection to **12 cores** and it went 43.7 -> 128.3 us. The consumer side worked exactly as intended (-28 us); the producer paid three times that |
| 28 | dropping the `qkv` output shard, so the matmul writes DRAM and no unshard is needed | **11.507 ms against 9.300**, +23.7% | re-tested because changes 18 and 19 were measured against an 11.7 ms tower, before LoFi and before the norm shard. The unshard costs 226 us; removing it costs the matmuls **+1.76 ms**. The margin is wider now, not narrower |
| 26 | narrowing the layer-norm output to bfloat8_b, so `qkv` and `c_fc` read a bfp8 in0 under LoFi | **+0.355 ms**, 295 -> 343 ops | the matmuls *do* gain — `c_fc` 79.2 -> 77.5 us, `qkv` 48.2 -> 47.6, **-0.042 ms** over all 99 — but the conversion cannot ride inside the norm, so it is 48 separate typecasts at **+0.396 ms**, 9.4x what it buys. Why it cannot ride inside: [below](#where-the-bfp8-norm-output-is-blocked) |
| 24 | `nlp_create_qkv_heads` / `nlp_concat_heads` as slice + reshape + permute | **+29.5 ms, 3.9x slower** | its own section [below](#long-form-the-head-reshape-rewrite) |
| 25 | `ttnn.linear(activation=...)` in place of a fused config on the aligner | the gelu came back as its own op, 0.123 ms | **`activation=` does not fuse.** With no program config it emits a separate unary op; folding into the matmul happens only through the config's `fused_activation` field |
| 25 | unsharding the aligner in0 to lift the `in0_block_w` bound | 9.975 against 9.984, reproducible to ±0.001 | real and trivial. 0.09% for an extra op, an extra conversion and another constant — measured three times on each side, then discarded |
| 25 | sweeping `in0_block_w` on the aligner while its in0 stays sharded | 2 gives +1.7%, 8 and 16 are **rejected outright** | `matmul_device_operation.cpp:1410` requires the in0 shard's width in tiles to divide `in0_block_w`. The aligner inherits a four-tile shard, so 4 is a ceiling rather than an optimum |
| 22 | SDPA `math_approx_mode=True` | 10.192, flat, PCC unchanged | does not reach the softmax kernel. Distinct from `exp_approx_mode` |
| 22 | `out_subblock` raised to the real DST bound | 10.207, flat and bit-identical | LoFi turns `fp32_dest_acc` off, doubling the DST bound 4 → 8 (`compute_kernel_config.cpp:152-161`), so `get_out_subblock_w`'s hardcoded 4 does leave half unused — but DST was not the constraint |
| 22 | the three 2D `in0_block_w` constants raised 8/16/32 | 10.309, +1.1% | retested at the *new* fidelity on the theory that LoFi shifts the compute/read balance. It does not |
| 22 | the same three lowered 2/4/8 | 10.507, +3.0% | same, in the other direction. Both sides now measured |
| 21 | `c_fc` `in0_block_w` swept off 8 | 4 gives +2.5%, 16 gives +12.3% | **two-sided optimum.** Below: more mcast rounds for the same bytes. Above: the circular buffers grow until `out_block_h` falls 18 → 9 and in1's 4.19 MB is read twice |
| 19 | width-sharded `c_fc` output | 11.759, +1.6% | its writer covers 36 tiles per core as 18 rows x 2 columns against `qkv`'s 3 x 12, so the burst along N is six times narrower. Sharding pays by write-burst width, not tile count. **Block-sharding the same output later paid -1.42%** (change 26) |
| 17 | `c_fc` intermediate to DRAM | 12.985 tower (+8.0%), `c_fc` 84.0 → 120.6 us (+43.6%) | `knowledge/matmul.md:31` warns L1 interleaved is slower than DRAM for matmul *inputs*; here the 2.36 MB *write* dominates |
| 17 | `ttnn.experimental.minimal_matmul` on `c_fc` | 30-104% slower across eight block-size configs | the ~2.5x claim at `mlp_1d.py:97-99` is a Llama down-proj. M = 18 tiles is too small to amortize its blocking. Note `bias_tensor` is keyword-only (`minimal_matmul_nanobind.cpp:141`) though the docstring reads positional |
| 11 | `transpose_mcast` on `qkv` | 17.0160, +5.8%; `qkv` 82.9 → 124.6 us with math at 101.09 | hurts the math engine as much as the reader. Reader and math move together through the CB pipeline, so there is no independent reader knob. **Retested at stage 29 and it no longer runs at all:** transposing swaps the axes, so `qkv`'s 8 N-blocks become `num_blocks_y` against a `grid.y` of 6 and `matmul_device_operation.cpp:795` rejects it. `grid_y` would have to be 8, and it is chosen as a divisor of M = 18 tiles. The stage-11 number was measurable only because no explicit config pinned the grid |
| 11 | bfloat4_b weights, at HiFi2 and at HiFi4 | PCC **0.85841** and **0.85656** against a 0.95 gate | the precision is missing from the *stored weight*; extra fidelity passes read mantissa bits that do not exist. HiFi4 measured marginally *worse*. 0.09 of PCC is not recoverable by partial application |
| 11 | `qkv` `in0_block_w` off its derived 4 | 16 gives +0.44%, 2 gives +3.7% | ttnn's derived value wins. The ~25 us reader overhang is invariant under every config change tried; cause never established |
| 7 | interleaved-L1 residual | **+134%**, 133.8 → 1195.6 us on `qkv` | not an L1 capacity problem: `TtLayerNorm` passes its input's memory config through, so `qkv`'s in0 became L1 and ttnn's heuristic flipped to a 1D 64-core strategy |
| 7 | L1 residual *and* L1 matmul outputs | **+149%** | same mechanism, compounded |
| 4 | SDPA `exp_approx_mode=True` | span 21.8951, bit-identical output, inside noise | unexplained — the flag does reach the kernel as a compile define |
| `pre` | `core_grid` value | 6x8 identical to 8x8 to the last digit of PCC | ttnn reaches 48 cores from either |
| `pre` | 1D reuse on `qkv` | +33% | 96 N-tiles cannot spread past 48 cores |
| `pre` | `c_fc`/`c_proj` fusion | predicted 3.66 ms from 216 MB / 59 GB/s, measured **0.240** | `c_proj` did not move at all — its DRAM read was already hidden behind compute. A bandwidth estimate is not a bound |

## Where the bfp8 norm output is blocked

The weights were never the problem — every weight in the tower is already bfloat8_b
(`janus_pro_layernorm.py:26`, `janus_pro_image_mlp.py:60,63`, `janus_pro_vision_aligner.py:54`), which
is why the report reads `BFP8 x BFP8`. What is bfloat16 is the *activation*: the norm output that
`qkv` and `c_fc` take as in0. It is produced at runtime, so there is nothing to preload.

**The capability exists. Only the entry point is missing.**

| layer | state |
|---|---|
| device op | `layernorm_device_operation.cpp:441` — `operation_attributes.dtype.value_or(input_tensor.dtype())`. Honoured, and specifically in the **sharded** branch, which is the one this tower takes. The interleaved branch at `:446` ignores it |
| validate | no constraint on the output dtype at all — the `TT_FATAL`s cover the input (`:49`), gamma (`:110`), beta (`:157`) and stats (`:213`) |
| params struct | `LayerNormParams::dtype` exists and is bound read-write to Python (`layernorm_nanobind.cpp:232`) |
| **wrapper** | **`layernorm.cpp:58` passes `std::nullopt, // dtype` — hardcoded.** This is the break |
| Python | dead end. `layer_norm_t` is only the callable's class, the same `ttnn.layer_norm`. `LayerNormDeviceOperation` binds `compute_program_hash`, `create_output_tensors`, `compute_output_specs`, `select_program_factory` — **no invoke**. The field is settable and unreachable |

So `LayerNormParams().dtype = ttnn.bfloat8_b` type-checks and does nothing: no call accepts it. There
is no Python workaround, and looking for one is the wasted day this entry exists to prevent.

Opening it is three lines of C++:

1. `layernorm.hpp:16` — add `const std::optional<DataType>& dtype = std::nullopt` to the signature
2. `layernorm.cpp:58` — forward it in place of the hardcoded `std::nullopt`
3. `layernorm_nanobind.cpp:196-210` — add the type to the `overload_cast` list and `nb::arg("dtype") = nb::none()`

**Worth `-0.42%` of the tower**, plus whatever the norm saves writing half the bytes (1.18 -> 0.59 MB
per call, unmeasured). That is the whole prize, and it is small: it edits a shared ttnn op that every
model in the repo calls, for four tenths of a percent on this one. The gap is real — the device op
supports what the wrapper drops — but the case for closing it is the API, not this tower's numbers.

## What `tt-perf-report`'s advice asks for, against what was measured

Its advice on the optimized tower makes two suggestions, and both are settled:

- **"If possible place input 0 in L1"** — true where it is asked, and the advice is asked of matmuls:
  **+14.1%** on `qkv` block-sharded (stage 17), **+134%** on the interleaved-L1 residual (stage 7),
  and 330.7 -> 354.9 us on the aligner's second projection. DRAM gives every core its own parallel
  read; a shard makes each core multicast out of its own piece, and the fan-out costs more than the
  read it replaces. **Do not generalise it past matmul in0** — see below.
- **"Use HiFi2 or HiFi4 with BF16 activations for improved accuracy"** — an accuracy suggestion, not a
  perf one. Taking it undoes change 22: **-0.693 ms for 0.0051 of tower PCC**. Every gate has slack;
  the tower sits 0.0165 above its own.

The inversion is the better idea and it is in the table above — if LoFi truncates anyway, feed it
bfp8 and halve the multicast bytes. It gains 1.7 us on `c_fc` and loses 8.25 us to the typecast that
produces it.

**`SLOW` on every matmul is a diagnosis, not a verdict.** `perf_report.py:1119-1126` prints it when
neither DRAM% nor FLOPs% reaches 65; both sit far below, which is true and is exactly where the
per-RISC split started. What the advice cannot say, because it does not measure it, is that BRISC
runs 99-100% of every one of these ops.


## L1 is not worse than DRAM — mcast is

The three measurements above read as "L1 loses here", and for two years of this tower's history that
is how they were applied: everything DRAM unless a matmul's own output could be sharded. Stages 27
and 28 show the rule was drawn one level too wide.

| tensor | consumer | L1 vs DRAM |
|---|---|---|
| `qkv` output as `c_fc`-style in0 | a matmul | **+14.1%** |
| the residual | a matmul's in0 | **+134%** |
| the aligner's intermediate | a matmul | +24 us |
| **q/k/v from `nlp_create_qkv_heads`** | **SDPA** | **-306.9 us, 48.0 -> 35.2 us each** |
| **SDPA's output** | **`nlp_concat_heads`** | **-104.8 us, 17.1 -> 12.7 us each** |
| **`nlp_concat_heads`'s output** | **`wo`, a matmul in0** | **-0.7 us per instance** |

The split is not the tensor and not its size. It is **what the consumer does to read it**. A 2D
matmul multicasts its in0 across the core grid, so an L1 source makes every core fan its own piece
out over the NOC, and that fan-out costs more than the parallel DRAM read it replaced. SDPA and the
head ops read what is in front of them; nothing multicasts, so an L1 write is simply local and an
L1 read is simply near.

The last row is the one that corrects the rule. `nlp_concat_heads`'s output was left in DRAM on the
reasoning that `wo` reads it as a matmul in0 -- and that was **wrong**: it gains from L1 like the
others. Every measurement in the top half of the table has a *sharded* in0, where each core has to
multicast its own piece. An **interleaved** L1 in0 under an explicit 2D config multicasts nothing;
it is simply a nearer read. The penalty is mcast, not L1, and the two are easy to confuse because
the only way to get an L1 in0 before change 27 was to shard it.

Stage 7's +134% looks like a counter-example and is not: there was no explicit program config then,
so an L1 input flipped ttnn's derivation to a 1D 64-core strategy. The layout was the trigger, not
the cost.

**The general form:** ask what reads the tensor before choosing where it lives. For a data-movement
op the write is most of what it costs, and an L1 write is the cheaper write.


## Closed without a device run

Not "slow" — unavailable. Each is closed by an API surface or by arithmetic, and no measurement
would change the answer.

| lever | what closes it |
|---|---|
| bfloat8_b layer-norm output, leaving the residual bfloat16 | `ttnn.layer_norm` has no output-dtype argument. The docstring's `dtype` is bound on `LayerNormParams`, i.e. the primitive (`layernorm_nanobind.cpp:232`), not the composite op; passing it raises `TypeError`. Norm output format is inseparable from the residual's through the public API |
| eliminating the head reshapes | SDPA requires `[B, num_heads, Sq, DH]` (`sdpa_device_operation.cpp:60-64`), so the split moves the head axis ahead of the sequence — a real transpose, not a view. No offline weight permutation replaces it |
| sharding `nlp_create_qkv_heads` | the sharded path requires `num_q_heads % num_cores == 0`; with 16 heads that caps at 16 cores, *fewer* than the 18 it already uses |
| running the head split on more than 18 cores, by any op | see [below](#the-head-split-is-capped-by-batch-not-by-a-missing-argument) — every op that would parallelise it further keys its parallelism to a dimension this tower does not have |
| fusing the residual add into layer norm | `FUSE_PRE_ADD` exists (`layernorm_op_multi_core.cpp:349`) and would delete 48 adds, but the sum lands in a circular buffer (`:471`) and is never returned. A pre-norm block needs that sum for its own residual |
| `gather_in0` on `c_fc` | `matmul_device_operation.cpp:714` requires input A sharded and `:719` a sub-device id. `c_fc`'s in0 arrives DRAM-interleaved from shared-code layer norm, so it would need a reshard per call — 24 x ~30 us against a gain bounded by `c_fc`'s 6.45 us overhang |
| padding the sequence 576 → 768 | arithmetic, not a measurement. 576 is 18 M-tiles and `find_prefill_grid` takes the largest divisor ≤ 8, so `grid_y` = 6 and `per_core_M` = 3. 768 is 24 tiles, `grid_y` = 8, `per_core_M` = 3 again. 33% more work meets 33% more grid rows, and every non-matmul op grows 33% with nothing to offset it |

## The head split is capped by batch, not by a missing argument

Tried at stage 24. The two fused ops cost 1.59 ms on 18 of 64 cores, and 18 is not a tuning choice:
`nlp_create_qkv_heads` fixes `num_blocks = shape[0]*[1]*[2]/TILE_HEIGHT`
(`nlp_create_qkv_heads_program_factory.cpp:72`), which is 18 tile-rows for 576 tokens.

Three routes out, all closed:

**A sibling op with a tunable grid.** `nlp_create_qkv_heads_vit` looks like the ViT-shaped answer
and is not — identical formula at `nlp_create_qkv_heads_vit_program_factory.cpp:54`, identical 18.

**`create_qkv_heads`**, the variant without the `nlp_` prefix, *does* take its core count from the
input's shard grid rather than from a formula (`create_qkv_heads_device_operation.cpp:84`), and it
wants exactly the BLOCK_SHARDED bfloat8_b tensor the projection already produces on 48 cores. It
still cannot run here, because of one line further down (`:62`):

```cpp
TT_FATAL(input_shape[0] == num_h_cores, "Batch size {} must be equal to num cores {}", ...)
```

The op gives **one batch per core row**. This tower runs batch 1, so its shard grid may have
exactly one row: on the projection's 8x6 grid it fails outright, and a forced 1x8 grid would give
8 cores against the 18 already in use. Splitting 576 tokens into 6 batches of 96 to feed it is not
a workaround — SDPA would then attend only within each 96-token group.

**Folding the split into the projection weight.** Closed by arithmetic. The matmul computes
`out[s,n] = Σ_k in0[s,k]·W[k,n]`, so permuting `W`'s columns can reorder `n` however you like, but
the head split needs `s` to move *inside* the head axis and `s` comes from `in0`. (The column
permutation `create_qkv_heads` would have needed — per-kv-head interleaved rather than blocked
q|k|v — was verified correct on torch, so this is a statement about the layout, not about the
permutation being hard.)

**The general rule:** every ttnn op that would spread this split past 18 cores parallelises over
batch, and a vision tower processing one image has none to spread.

### What the models that avoid this cost actually do

They do not make the split faster — they never run it. The head split rides inside the op in front
of it:

- **`tt_dit` (WAN2.2, LTX, Ideogram4)** fuse it into the QK norm.
  `ttnn.experimental.dit_fused_distributed_rmsnorm_*` takes `num_heads_per_device` and emits
  `[B, H, N, E]` directly, with RoPE folded into the same kernel
  (`tt_dit/layers/normalization.py:196-259`). Only V pays a standalone `nlp_create_qkv_heads`,
  because V has no norm to ride in — see the comment at `ltx/attention_ltx.py:453-465`.
- **Decode paths** (llama2_70b, gpt_oss, qwen3_vl) use `nlp_create_qkv_heads_decode`, where batch
  is the number of users, so the batch axis supplies the parallelism the prefill axis cannot.

Neither is available here. Janus-Pro-7B's vision config sets **`use_qk_norm = False`**, so SigLIP
attention has nothing between the projection and SDPA to fuse a split into — `ln_1` sits *before*
the projection, not after it. And there is one image, so the decode op's batch axis is empty.

The closest analogue in the repo accepts the same cost: `qwen3_vl/tt/vision_attention.py:398` is a
batch-1 vision encoder over the same family of shapes and calls plain `nlp_create_qkv_heads` with a
DRAM output.

**So the prerequisite is architectural, not a config:** removing this cost needs an op between the
projection and SDPA worth fusing into. Inserting a norm purely to host the split does not work —
RMSNorm divides by the data's own RMS, so even with unit weight it changes the values.

## Long-form: the head-reshape rewrite

Tried at stage 24, against a 10.023 ms tower. The two fused ops cost 1.59 ms and are pinned to
18 cores; the question was whether basic ttnn ops could use all 64.

**They can, and they lose anyway: +29.5 ms, 3.9x slower.** The replacement cost 31.06 ms in 672
extra device ops. Output was bit-identical, so the rewrite is correct — it is simply far more work.

Two things no reading of the API predicts:

- **`ttnn.reshape` on a tiled tensor is not a view**, even when the split falls on a tile boundary.
  96 of them cost 20.4 ms — twice the entire tower on their own.
- ttnn inserts **`Typecast` (384 instances) and `FillPad` (72)** that nothing in the signature
  suggests.

The core-count reasoning was sound: the fused op is capped at 18 by
`num_blocks = shape[0]*[1]*[2]/TILE_HEIGHT` (`nlp_create_qkv_heads_program_factory.cpp:72`), while
`permute` splits by tile count (`permute_tiled_program_factory.cpp:118`) and does reach 64. It got
there and still lost.

**One pass over the data on 18 cores beats twenty passes on 64.** Occupancy is not the objective.

## Where every knob has already been swept

Not dead ends — just surfaces with nothing left on them, recorded so a sweep is not repeated.

| op | swept |
|---|---|
| SDPA | `q_chunk_size` and `k_chunk_size` (see PERF.md changes 2 and 13), math fidelity (20), `fp32_dest_acc` (21), both approx-mode flags (above) |
| LayerNorm | `welford`, `inplace`, `legacy_reduction`, `legacy_rsqrt`, `subblock_w`, grid shape |
| the four body matmuls | `in0_block_w` in both directions at two fidelities, `out_subblock`, `core_grid`, 1D vs 2D, output memory config, output dtype. `transpose_mcast` is not a knob here at all — see the stage-11 row above |
| the head ops and SDPA | output memory config (changes 27 and 28 took L1 on all three; `nlp_concat_heads` keeps DRAM because `wo` reads it) |
| the patch projection | output memory config and layout: DRAM, L1 interleaved, block shard without a config, block shard with one (change 29 kept the last) |
