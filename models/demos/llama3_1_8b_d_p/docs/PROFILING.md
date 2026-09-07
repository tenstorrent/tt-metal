<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama 3.1 8B prefill — profiling overview

Where the time goes in a warm prefill on the 8x4 Blackhole Galaxy, and what that says about which
optimisations are worth doing.

**Measured at the spec's own shapes.** Everything below uses `chunking.chunk_size = 4096` and
contexts up to the spec's full `context_length = 131072`. An earlier revision of this document
measured a 2048-token one-shot prefill throughout. That was a mistake, and not a harmless one: the
one-shot path and the chunked path **run different ops** (§3b), so the earlier numbers described a
code path the spec's configuration never takes.

**Headline: the prefill is host-dispatch-bound, not compute-bound.** Device kernels account for
~62.9 ms of an ~825 ms chunk — about **7.6%**. The other ~92% is per-op dispatch and
synchronisation. That is exactly what ttnn trace mode removes, which makes the missing
tensor-metadata form on the GQA ring SDPA (`docs/SPEC_NOTES.md` §8d, `tt/trace.py`) a **first-order
performance blocker** rather than a formality.

**Second headline: the spec's 4096 chunk costs 3.5x on full-context prefill** (§2c). That is the
largest single number in this document and it is a spec value, not a code defect — written up as
`docs/SPEC_NOTES.md` §8e.

Setup unless stated: real `Meta-Llama-3.1-8B-Instruct` weights, bf16 attention + MLP, 32 layers,
mesh 8x4 (SP=8 x TP=4), `FABRIC_1D` + Linear topology, one user, warm (a warmup prefill precedes
every measured one, so JIT and program-cache misses are outside the measurement).

---

## 1. Wall clock is noisy — but the noise is a small-shape artefact

The 2048-token one-shot configuration, best-of-3 within each run, run in five separate processes:

```
316 ms    777 ms    932 ms    940 ms    961 ms
```

A 3x spread on an identical configuration. That was the basis for the earlier rule "no wall-clock
difference below roughly 2x is meaningful on this box".

**At realistic shapes the spread largely disappears.** The full 131072-token context, chunk 4096,
measured in two independent processes:

```
27,031 ms    28,660 ms      (6% apart)
```

Longer runs average the jitter out, so **large-shape comparisons are far more trustworthy than the
2048 numbers were**. The 3.5x chunk-size result in §2c is well outside this noise; the SDPA-config
result below is not.

Two hypotheses died against controlled tests, both of which looked convincing on first samples:

| hypothesis from a single sample | controlled result |
|---|---|
| "chunks below the 2048 SDPA threshold are ~3x slower (64/64 vs 256/256 q/k chunking)" | forcing a 2048 chunk onto the 64/64 config gave **1106 / 724 ms** — overlapping the 256/256 runs. **Not supported.** |
| "~90% of a chunk is fixed per-chunk overhead" | the layer sweep below shows a **26 ms** intercept, i.e. ~2.7%. **Wrong; the cost is per-layer.** |

Reproduce: `PREFILL_SDPA_THRESHOLD` / `PREFILL_SDPA_{Q,K}_{SMALL,LARGE}` override the program config
(defaults are the spec's values), which is what makes that a controlled experiment rather than a
correlation.

## 2. Cost is linear in LAYERS, flat in TOKENS, and therefore driven by CHUNK COUNT

### 2a. Layers

Layer count swept at the spec's 4096 chunk, 16384-token context (4 chunks), so the SDPA program
config is identical at every point and layer count is the only variable:

| layers | best | ms/layer/chunk |
|---|---|---|
| 1 | 167.9 ms | 42.0 |
| 4 | 564.0 ms | 35.3 |
| 16 | 2137.5 ms | 33.4 |
| 32 | 3839.7 ms | 30.0 |

Least squares: **29.7 ms per layer per chunk, intercept 26.1 ms per chunk, R² = 0.997.** Per-chunk
fixed cost is ~2.7% of a 32-layer chunk; essentially all the time is per-layer and scales linearly.
(The earlier 2048 measurement gave 28.4 ms/layer with a 15.4 ms intercept — the same conclusion.)

### 2b. Tokens

Per-chunk cost as the KV cache grows, chunk fixed at 4096, 32 layers:

| context | chunks | best | tok/s | ms/chunk |
|---|---|---|---|---|
| 4,096 | 1 | 1,117.9 ms | 3,664 | 1118 |
| 16,384 | 4 | 3,304.3 ms | 4,958 | 826 |
| 65,536 | 16 | 14,740.7 ms | 4,446 | 921 |
| 131,072 | 32 | 27,031.0 ms | 4,849 | 845 |

**Per-chunk cost is flat from an empty cache to a 128K one.** Attention over the growing cache never
becomes the bottleneck at the spec's target context — the dispatch-bound conclusion is not an
artefact of a small context, it holds all the way out. Device-side, the ring SDPA *does* get more
expensive with a populated cache (§3a), but it is a small enough share that it does not move the
wall clock.

### 2c. Chunk count is the lever — and the spec's 4096 is leaving 3.5x on the table

Cost scales with **(layers x chunks)** and chunks = `context / chunk_size`. So halving the chunk
count halves the prefill. Measured at the spec's full 131072 context, 32 layers:

| `chunk_size` | chunks | best | tok/s | vs spec |
|---|---|---|---|---|
| **4096** (spec) | 32 | 28,659.9 ms | 4,573 | 1.00x |
| 8192 | 16 | 15,805.7 ms | 8,293 | **1.81x** |
| 16384 | 8 | 8,245.1 ms | 15,897 | **3.48x** |

Same ordering at a 16384 context (1,940 / 4,966 / 8,007 tok/s for 2048 / 4096 / 8192), so this is
not an artefact of one shape. **Full-context prefill drops from 28.7 s to 8.2 s.**

The tradeoff is real and belongs with the number: larger chunks raise peak activation memory and
coarsen scheduling granularity — a queued request waits for a chunk boundary, so time-to-first-token
for the *next* request rises with chunk size even as aggregate throughput improves. 16384 is a
throughput ceiling; 8192 is the low-risk move. See `docs/SPEC_NOTES.md` §8e.

## 3. Device-side breakdown

From `generated/profiler/reports/2026_09_07_16_24_08/` — a warm profiled run at the spec's chunked
shape (2 x 4096 chunks, 8 layers, 32 devices). Normalised per device per chunk-forward per layer;
devices run concurrently, so the per-device mean approximates the critical path when the mesh is
balanced.

**Tilize is excluded.** All 1376 of its calls (43/device) land before the first embedding op, so it
is one-time weight tilization at setup, not per-prefill work. Verified independently at both shapes.

**Total device kernel time: 1.96 ms per layer — 62.9 ms per 32-layer chunk.**

| op | ms/layer | % of device time |
|---|---|---|
| Matmul | 0.662 | 33.7 |
| **RingJointSDPA** | 0.536 | 27.3 |
| AllGatherAsync | 0.237 | 12.0 |
| ReduceScatterMinimalAsync | 0.221 | 11.3 |
| LayerNorm | 0.126 | 6.4 |
| BinaryNg (add/mul) | 0.084 | 4.3 |
| RotaryEmbeddingIndexed | 0.025 | 1.3 |
| NlpCreateHeads / ConcatHeads | 0.037 | 1.9 |
| Unary (silu) | 0.019 | 1.0 |
| Typecast | 0.011 | 0.6 |
| UpdatePaddedKvCache | 0.005 | 0.3 |

Grouped: **matmul 33.7%, attention 27.3%, collectives 23.3%.**

### 3a. The ring SDPA nearly doubles once the cache is real

Splitting the warm prefill into its two chunk-forwards:

| chunk | cache state | SDPA ms/layer | total ms/layer |
|---|---|---|---|
| 1 | empty | 0.381 | 1.792 |
| 2 | 4,096 tokens | **0.692** (+82%) | 2.137 |

A one-shot profile cannot see this at all — it only ever runs the empty-cache case. This is the
main reason the earlier revision of this document understated attention.

### 3b. The one-shot and chunked paths run DIFFERENT ops

Controlled: same 8 layers, same weights, only the shape differs.

| per layer | one-shot 2048 | chunked 4096 (spec path) |
|---|---|---|
| attention op | `SDPAOperation` | `RingJointSDPADeviceOperation` |
| all-gathers | 5.125 | **2.125** |
| reduce-scatters | 3 | **2** |
| matmul | 31.7% | 33.7% |
| collectives | **36.7%** | **23.3%** |
| attention | 16.7% | 27.3% |
| total | 1.824 ms/layer | 1.965 ms/layer |

The ring op carries its own communication internally, replacing explicit all-gathers — so the
chunked path issues **less than half** the collectives per layer. Total device time per layer is
within 8% between the two paths, but the composition is quite different.

**This retracts a conclusion.** The earlier revision reported "collectives cost more device time
than every matmul combined" (41.0% vs 29.6%). That was measured on the one-shot path. On the path
the spec actually runs, matmul (33.7%) exceeds collectives (23.3%), and the sharded-residual
optimisation described in §4 is correspondingly less valuable than previously claimed.

## 4. What this means for optimisation

**1. Raise the chunk size.** Free, no model change, 1.8x at 8192 and 3.5x at 16384 on full-context
prefill (§2c). Bounded by activation memory and the serving SLO for TTFT, not by correctness. This
is now the first thing to do, ahead of trace mode, because it requires no op work.

**2. Trace mode — still the big structural lever, still blocked.** 1.96 ms of device kernel work per
layer against ~26 ms of wall time means roughly **24 ms per layer is host dispatch and
synchronisation**, ~12x the device work. ~25 op dispatches per layer x 32 layers is ~800 dispatches
per chunk, each a host→device round trip across 32 chips. Trace replay collapses that to one
submission.

It is unavailable: capturing a trace requires the ring SDPA to read its cache offset from a device
tensor, and `ring_joint_scaled_dot_product_attention` has no such form (`ring_mla` does). The
model-side work is already done and validated — see `tt/trace.py` and
`tests/unit/test_trace_metadata_vs_ref.py`, where the metadata-tensor path is bit-identical to the
scalar path (PCC 1.0). What remains is confined to the op.

**3. Attention is the largest single op after matmul on the real path**, at 27.3% and rising with
cache occupancy (§3a). Worth measuring before assuming collectives are the target.

**4. Collectives are 23.3%, not 41%.** The replicated-residual contract this package uses (attention
and MLP each close with a full all-reduce) is still a candidate for the sharded-residual variant
MiniMax-M3 uses, but on the chunked path the ceiling on that optimisation is smaller than the
earlier revision claimed.

**5. Do not tune the SDPA chunk sizes on this evidence.** The controlled test found no effect
distinguishable from noise. The knobs exist (§1) if someone wants to measure properly.

## 5. Reproducing

```bash
source ttenv.sh
export HF_MODEL=/path/to/Meta-Llama-3.1-8B-Instruct

# full-context chunk-size sweep — the 3.5x result
PREFILL_PERF_SEQ_LEN=131072 PREFILL_PERF_CHUNK_SIZES=8192 \
  pytest models/demos/llama3_1_8b_d_p/tests/prefill_perf.py -k 8x4 -s

# context sweep at the spec chunk (per-chunk cost vs cache occupancy)
PREFILL_PERF_SEQ_LEN=65536 PREFILL_PERF_CHUNK_SIZES=4096 \
  pytest models/demos/llama3_1_8b_d_p/tests/prefill_perf.py -k 8x4 -s

# layer scaling (the measurement that identified the per-layer cost)
PREFILL_PERF_SEQ_LEN=16384 PREFILL_PERF_CHUNK_SIZES=4096 PREFILL_PERF_LAYERS=4 \
  pytest models/demos/llama3_1_8b_d_p/tests/prefill_perf.py -k 8x4 -s

# device op breakdown at the CHUNKED shape (use >=2 chunks or the ring SDPA never reads a cache)
PREFILL_PERF_SEQ_LEN=8192 PREFILL_PERF_CHUNK_SIZES=4096 PREFILL_PERF_LAYERS=8 \
PREFILL_PERF_WARMUP=1 PREFILL_PERF_ITERS=1 \
  python -m tracy -r -p -v -m pytest models/demos/llama3_1_8b_d_p/tests/prefill_perf.py -k 8x4
```

Each profiled chunk-forward writes several GB of ops-times CSV and post-processing holds it in a
pandas frame. 32 layers x 4 chunk-forwards **exceeds what the post-processor can join** (it fails
with `Device data missing: Op N not present in cpp_device_perf_report.csv`); 8 layers x 4
chunk-forwards works. Since the breakdown is normalised per layer, profile with `PREFILL_PERF_LAYERS`
reduced rather than trying to profile all 32.

## 6. Caveats

- Wall-clock and device-op numbers come from **different runs** (profiling perturbs wall clock), so
  the "7.6% device" figure is a ratio across runs, not a single measurement. The direction is far
  larger than the noise, but do not quote it to two significant figures.
- The device breakdown is measured at **8 layers**, the wall-clock numbers at 32. §3b controls for
  this by comparing both shapes at 8 layers; the 8-layer one-shot breakdown (attention 16.7%,
  collectives 36.7%) is close to the 32-layer one-shot figures the earlier revision reported
  (15.6% / 41.0%), so layer count is not distorting the comparison.
- All timings are bf16 attention + MLP. The package defaults (bfp8 / bfp4) will differ.
- Single user, single pipeline stage. Multi-user batching and pipeline-parallel stages are unmeasured.
- No performance TARGETS exist for this model to gate against (the spec's `performance_targets` is
  deliberately empty), so this document describes behaviour; it does not pass or fail anything.
