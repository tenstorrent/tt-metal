# Decode Optimization Comparison: Local Autoport vs In-Place Graph Opt

Compares two **different** agentic decode-optimization runs for
`meta-llama/Llama-3.1-8B-Instruct`:

- **LOCAL** — functional-decoder autoport in `tt-metal`
  (`models/autoports/meta_llama_llama_3_1_8b_instruct`, summary
  `doc/OPTIMIZATION_SUMMARY.md`). Pipeline: `$forge-functional-decoder` →
  `$optimize`. A fresh TTNN-native decoder was built from the forge emit, then
  tuned.
- **GRAPH** — in-place optimization of the emitted decode graph on
  [`ttnn-models` branch `...llama-3.1-8b-decode-opt`](https://github.com/svuckovicTT/ttnn-models/blob/mvasiljevic/agentic-research-llama-3.1-8b-decode-opt/meta-llama/Llama-3.1-8B-Instruct/model/graph_0/OPTIMIZATION_SUMMARY.md).
  No bringup stage; the tt-forge-emitted `graph_0` (`model_ttnn.py`,
  `consteval.py`, `main.py`) was legalized and tuned in place.

## Most important caveat: the two are not measured on the same thing

| Dimension | LOCAL (autoport) | GRAPH (in-place) |
| --- | --- | --- |
| Unit measured | **one** dense decoder layer | **full 32-layer** model |
| Batch | 1 | 32 |
| Decode shape | paged, prefix 16 | single-token step, 32 seqs |
| Headline metric | traced **host** latency (ms/layer) | traced **wall-clock** (s/step) |
| Device | single Wormhole | single Wormhole, 8x8 grid |
| Correctness gate | real-weight PCC ~0.99999 | PCC 1.000000 |

Because of this, the absolute numbers (LOCAL 1.289 ms/layer vs GRAPH 0.0791 s/step)
**must not be divided against each other**. Compare the *methods* and the
*per-change deltas within each run*, not the raw figures across runs.

## Scaled to the same basis: e2e begin → end

To put both on a full-model **decode-step** basis, LOCAL's single-layer traced
host latency is multiplied by 32 layers. **This is a naive estimate, not a
measurement:** it excludes embedding, final norm, and the LM head, it is host
(not device) time, and — most importantly — it is **batch 1**, whereas GRAPH is
**batch 32**. Decode is weight-bandwidth-bound, so a batch-32 run reads the same
weights per step while serving 32 tokens; batch cannot be scaled away. Treat the
LOCAL e2e column as an order-of-magnitude sanity figure only.

| Full-model decode step | Begin | End | Latency reduction |
| --- | ---: | ---: | ---: |
| **LOCAL** (1.845/1.289 ms × 32 layers, batch 1, excl. embed/norm/LM head) | ~59.0 ms | ~41.3 ms | **~30%** |
| **GRAPH** (measured, batch 32, full graph) | 83.1 ms | 79.1 ms | **~4.8%** |

Throughput on each run's own basis (again, **not** cross-comparable — the batch
difference dominates):

| | Begin | End |
| --- | ---: | ---: |
| LOCAL, batch 1 (1 user) | ~16.9 t/s | ~24.2 t/s |
| GRAPH, batch 32 aggregate | ~385 TPS | ~404 TPS |
| GRAPH, per user | ~12.0 t/s/u | ~12.6 t/s/u |

**The only defensible cross-run number is % latency reduction (30% vs 4.8%), and
even that compares different baselines** — see the next section for why the gap is
not "LOCAL optimized 6x harder."

## Is it one layer or the full model? (this is the main reason they differ)

| | What actually runs on device | Batch |
| --- | --- | --- |
| **LOCAL** | **one bare decoder layer** — input norm → attention → MLP → residuals. No embedding, no final norm, **no LM head**. | 1 (tile-padded to 32) |
| **GRAPH (device / reduced)** | **full-model harness with `num_layers=1`**: token embedding → **one** decoder layer → final norm → **LM head** (`profile_reduced.py`) | 32 |
| **GRAPH (e2e headline 0.0791 s)** | **full 32-layer model** + embedding + final norm + LM head | 32 |

So the device numbers compared earlier were **not the same object**: LOCAL is a
single decoder layer; the GRAPH reduced window is a whole one-layer *model*
including the LM head.

## Exact op-level comparison (both measured on this Wormhole)

Full per-op reports:
`doc/optimized_decoder/tt_perf_report_decode.txt` (LOCAL final) and
`doc/optimized_decoder/tt_perf_report_GRAPH_branch_reduced.txt` (GRAPH reduced).

| Op group | LOCAL final (batch 1) | GRAPH reduced (batch 32) |
| --- | ---: | ---: |
| Token embedding | — (not in layer) | ~61 μs |
| Input LayerNorm | 94 μs (1 core) | 17 μs (8 cores) |
| QKV projection | 102 μs (BFP4, DRAM-interleaved) | 108 μs (BFP8, **DRAM-sharded**) |
| Head creation + reshapes | ~60 μs (fused decode ops) | ~263 μs (reshape 148 + create-heads 115) |
| Rotary embedding | ~33 μs | ~338 μs (169: 72 + 181: 266) |
| Paged KV cache update | 28 μs | 53 μs |
| SDPA decode | 12 μs | 125 μs |
| O projection | 58 μs (BFP4) | 91 μs (BFP8) |
| Post-attn LayerNorm | 94 μs (1 core) | 14 μs (16 cores) |
| MLP gate + up | 528 μs (258 + 270) | 327 μs (170 + 157, BFP4) |
| MLP down | 134 μs (BFP4, DRAM-sharded) | 299 μs (BFP8, DRAM-sharded) |
| Final norm | — (not in layer) | 14 μs |
| **LM head** (`4096×128256`) | **— (not in layer)** | **2,941 μs (57.8%)** |
| **Total window** | **1,229 μs, 37 ops** | **5,019 μs, 69 ops** |

### Layer-to-layer, apples-to-apples

Strip the parts LOCAL doesn't have (embedding 61 + final norm 14 + LM head 2,941 +
tail reshard 57 = 3,073 μs) from the GRAPH window:

| | Decoder-layer-only device time |
| --- | ---: |
| LOCAL final (batch 1, BFP4/BFP4) | **1,229 μs** |
| LOCAL baseline (batch 1, BFP8/BFP8) | 1,817 μs |
| GRAPH decoder layer (batch 32, BFP8 attn / BFP4 MLP) | **~1,946 μs** |

Now they're comparable — and GRAPH's single layer (~1.95 ms) does **32× the tokens**
of LOCAL's (~1.23 ms). Per token, GRAPH's batched layer is far more efficient; the
raw `5,019 μs` only looked ~4× bigger because **58% of it is the one-time LM head**.

### What the op tables reveal about the two implementations

- **LOCAL wins on head-creation + rotary** (~93 μs vs ~601 μs): it uses fused
  decode head ops, while the emitted GRAPH does large explicit reshapes/rotary.
- **LOCAL loses on LayerNorm** (94 μs single-core ×2 vs GRAPH's 14–17 μs sharded).
- **QKV/down are DRAM-sharded in both**; GRAPH keeps BFP8 there, LOCAL uses BFP4.
- GRAPH's headline device number is dominated by the **LM head**, which is a
  full-model op, not a decoder-layer op.

## Device-level perf: baseline vs baseline (measured)

LOCAL rows were captured on this Wormhole with Tracy + `tt-perf-report` over the
`PERF_TRACE_DECODE` window: the baseline is the BFP8/BFP8 LoFi "first correct"
precision (see `optimized_decoder/tt_perf_report_baseline_decode.txt`) and the
final is the committed BFP4/BFP4 report (`tt_perf_report_decode.txt`). GRAPH rows
are from the branch summary's reduced single-layer device window (no committed
artifact on the branch).

| Metric | LOCAL baseline (BFP8/BFP8) | LOCAL final (BFP4/BFP4) | GRAPH baseline (emitted) | GRAPH final |
| --- | ---: | ---: | ---: | ---: |
| Device time | 1,817 μs | 1,229 μs | 8,435 μs | 5,034 μs |
| Device ops | 37 | 37 | 71 | 69 |
| DRAM BW | 122 GB/s | 91 GB/s | 81 GB/s | 136 GB/s |
| DRAM roofline | 42.2% | 31.6% | 28.2% | 47.2% |
| Device improvement | — | **−32%** | — | **−40%** |

**These absolute device numbers are NOT apples-to-apples** — three confounders:

1. **Batch.** LOCAL is batch 1 padded to a 32-row tile (1 real token/step); GRAPH
   is batch 32 (32 real tokens/step). Same matmul `M=32` shape, but GRAPH does
   32× the useful work per step.
2. **Scope.** LOCAL's window is a **bare decoder layer** (37 ops). GRAPH's reduced
   single-layer window **includes the one-time embedding / final-norm / LM-head
   tail** — that accounts for most of the 71-vs-37 op gap and a large share of its
   milliseconds (the LM head is a 4096×128256 matmul).
3. **Precision baseline.** LOCAL baseline is uniform BFP8; GRAPH baseline is the
   emitted mix (BFP8 attention/down/LM-head + BFP4 gate/up).

### Opposite roofline stories (the interesting part)

- **LOCAL got faster by moving *fewer bytes*.** BFP4 halves weight bytes vs BFP8,
  so device time fell 32% even though DRAM roofline **dropped** 42.2% → 31.6% and
  absolute GB/s fell 122 → 91. It is doing less work, not using DRAM harder.
- **GRAPH got faster by moving bytes *more efficiently*** at unchanged precision:
  DRAM width-sharding the QKV weights pushed roofline **up** 28.2% → 47.2% and
  GB/s up 81 → 136.

So the two runs improved decode through opposite mechanisms — LOCAL via precision
(fewer bytes), GRAPH via layout (higher bandwidth utilization).

## Change-to-change, normalized to % of each run's own baseline

Each change expressed as a fraction of that run's starting latency. LOCAL rows are
% of 1.845 ms; GRAPH rows are % of the 83.1 ms e2e step. LOCAL's A/B rows are vs
their alternative (not cumulative), so they **do not sum** to the 30% net — the
net is essentially all precision.

| Category | LOCAL change | LOCAL Δ | GRAPH change | GRAPH Δ |
| --- | --- | ---: | --- | ---: |
| **Precision** | BFP4 MLP + BFP4 attn weights | **−30% (net, dominant)** | left as emitted (BFP8/BFP4) | **0%** |
| **QKV matmul** | kept **packed** (vs separate) | −6.5% (vs alt) | **DRAM width-shard** (primary e2e driver) | ≈ the −4.8% e2e |
| **MLP down matmul** | **DRAM-shard KEPT** (`in0_block_w=14`) | −1.3% (vs 28) | **DRAM-shard REJECTED** (regressed) | — |
| **Gate/up matmul** | kept **separate** (vs packed) | −2.4% (vs alt) | unchanged | 0% |
| **KV cache** | **BF8_B** (vs BF16) | −0.5% | unchanged | 0% |
| **Attention** | paged SDPA decode | (in baseline) | unchanged emitted | 0% |
| **Graph tail (logits/LM head)** | n/a (single layer) | — | reshape + redundant-copy removal | big on 1-layer window, small e2e* |

\* The removed tail work shrank the **reduced single-layer** device window 40%
(8.435 → 5.034 ms), but that window over-weights the one-time tail; amortized
across 32 layers its e2e effect is small, which is why GRAPH's e2e only moved 4.8%.

## Why LOCAL's % is much larger than GRAPH's %

1. **Different baselines.** LOCAL's start is a correct-but-unpruned candidate at
   BFP8/BFP8; GRAPH's start is the emitted graph that **already ships BFP8
   attention/down/LM-head + BFP4 gate/up**. LOCAL still had the big precision win
   available; GRAPH had already spent it.
2. **LOCAL changed precision, GRAPH did not.** ~All of LOCAL's 30% is BFP4
   weights. GRAPH explicitly judged further precision lowering unsafe and left it.
3. **e2e amortization.** GRAPH's e2e spreads its wins across 32 layers plus a
   fixed tail, damping the percentage; LOCAL's per-layer % has no such damping.

So the 30% vs 4.8% gap is mostly "LOCAL still had precision headroom and spent it"
plus a measurement-basis effect — **not** evidence that one search was 6x better.

## Decode starting point and overall improvement

| | LOCAL | GRAPH |
| --- | --- | --- |
| Baseline definition | first **correct** traced candidate (BFP8 attn/MLP, LoFi) | **legalized** emitted graph (minimal fixes to run) |
| Start | 1.845 ms / layer | 0.0831 s / step (~385 TPS, ~12.0 t/s/u) |
| Final | 1.289 ms / layer | 0.0791 s / step (~404 TPS, ~12.6 t/s/u) |
| Improvement | ~0.556 ms (~30%, ~1.43x) | ~4.0 ms/step (~4.8%, +~19 TPS) |
| Device-window analog | profiler decode 1,229 us (37 ops) | reduced 1-layer 8.435 ms/71 ops → 5.034 ms/69 ops; roofline 28.2%→47.2% |

Note LOCAL had **no** pre-optimization decode baseline at all (the functional
decode path was a stub), so its "start" is the first numerically-correct traced
candidate. GRAPH's start is the emitted graph after only legalization.

## Per-change contribution — LOCAL

Precision ladder (cumulative, all LoFi, real-PCC pass; `precision_trials.csv`):

| Change | Before → After | Improvement |
| --- | ---: | ---: |
| **BFP4 MLP weights** | 1.845 → 1.373 ms | **−0.472 ms (~26%)** ← dominant |
| **BFP4 attention weights** | 1.373 → 1.287 ms | −0.086 ms (~5%) |

A/B choices at BFP4/BFP4 LoFi (each vs its alternative):

| Change | Kept | Alt | Improvement |
| --- | ---: | ---: | ---: |
| Decode Q/K/V | packed 1.294 ms | separate 1.413 ms | ~0.119 ms |
| Gate/up MLP | separate 1.289 ms | packed 1.333 ms | ~0.045 ms |
| Down proj `in0_block_w` | 14 → 1.286 ms | 28 → 1.310 ms | ~0.024 ms |
| KV cache dtype | BF8_B 1.276 ms | BF16 1.285 ms | ~0.009 ms |
| L1 input movement | rejected | L1 → 1.307 ms | −(would add +0.024 ms) |

**LOCAL's biggest lever is precision (BFP4 weights).**

## Per-change contribution — GRAPH

Kept changes (from the branch summary + verified in `model_ttnn.py`/`consteval.py`):

| Change | Effect | Evidence |
| --- | --- | --- |
| Legalize shard/core grids for 8x8 | precondition; defines the 0.0831 s baseline | run log |
| Harness entrypoint fix (`main()` vs `test_main()`) | makes the graph runnable at all | `main.py` |
| PCC gate `==` → `>=` | correctness-contract fix, not perf | `main.py` |
| Remove tail logits reshape + redundant L1 copy | reduced window **8.435 → 5.034 ms**, 71 → 69 ops | reduced profile |
| **QKV DRAM width-sharded matmul** | **primary win**: QKV row 128 → 107 us; roofline **28.2% → 47.2%**; full step 0.0831 → 0.0791 s | `consteval.py`, `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` |

Rejected changes:

| Change | Why rejected |
| --- | --- |
| Remove unused logits copy only | no warmed-trace change (0.0831 s both) |
| Reuse DRAM logits tensor for reshape | slower (0.0837 vs 0.0831 s) |
| `nlp_create_qkv_heads_decode` on sharded output | compile fail (needs 4D) |
| Pre-reshape sharded QKV to 4D | broke rotary (seq len 8 vs 1) |
| **MLP down-proj DRAM-sharding** | **regressed** reduced replay to 20.8 ms |
| Broad precision lowering | deemed "not an obvious safe win"; left as emitted |

**GRAPH's biggest lever is QKV DRAM width-sharding.**

## What is in BOTH

- **DRAM-sharded matmul as a technique** — but on **different targets, opposite
  conclusions** (see below).
- Correctness-gated (PCC) with every kept change required to hold accuracy.
- `tt-perf-report` / Tracy-driven candidate selection.
- Final validation under `TT_METAL_WATCHER=10`, single Wormhole, traced decode.

## Notable divergence: DRAM-sharding lands opposite

| | LOCAL | GRAPH |
| --- | --- | --- |
| DRAM-sharded **down** proj | **KEPT** (`in0_block_w=14`) | **REJECTED** (regressed to 20.8 ms) |
| DRAM-sharded **QKV** proj | not adopted (kept **packed** QKV) | **KEPT** (primary win) |

The same tool (`tt-perf-report`) pointed both runs at DRAM-bound projections, but
the profitable target differed — most likely because the graphs are structurally
different (fresh paged single-layer decoder vs emitted batch-32 full-model graph)
and start from different precision policies.

## What is ONLY in one

Only in **LOCAL**:

- Full precision retuning to **BFP4 attention + BFP4 MLP** (its dominant win).
  GRAPH deliberately left precision as emitted.
- **Paged KV cache (BF8_B)** + `paged_scaled_dot_product_attention_decode`.
- Projection-topology sweeps (packed vs separate QKV, gate/up).
- A prior **functional-decoder bringup** stage from the forge emit.
- Per-decision trial CSVs + work logs as committed evidence.

Only in **GRAPH**:

- **Graph legalization** for the 8x8 grid (bring-up on the emitted graph).
- **Harness fixes**: entrypoint bug, PCC-gate relaxation, host-side logits
  normalization.
- **Tail cleanup**: removing the terminal logits reshape and redundant L1 copy.
- **QKV DRAM width-sharding** as the primary compute win.
- **Full-model, batch-32** measurement (TPS and per-user t/s/u).
- Changes applied **in place** to `model_ttnn.py` / `consteval.py` / `main.py`.

## One-line takeaway

Both are single-Wormhole, PCC-gated, profiler-driven decode optimizations, but
they optimize different objects: **LOCAL** rebuilds and tunes a single decoder
layer and wins mostly through **BFP4 precision**; **GRAPH** legalizes and tunes
the emitted full-model graph in place and wins mostly through **QKV DRAM
width-sharding** — and the two even reach opposite verdicts on which projection
benefits from DRAM-sharding.
