<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama 3.1 8B prefill — profiling overview

Where the time goes in a warm prefill on the 8x4 Blackhole Galaxy, and what that says about which
optimisations are worth doing.

**Headline: the prefill is host-dispatch-bound, not compute-bound.** Device kernels account for
~75 ms of a ~900 ms chunk — about **8%**. The other ~92% is per-op dispatch and synchronisation.
That is exactly what ttnn trace mode removes, which makes the missing tensor-metadata form on the
GQA ring SDPA (`docs/SPEC_NOTES.md` §8d, `tt/trace.py`) a **first-order performance blocker** rather
than a formality.

Setup unless stated: real `Meta-Llama-3.1-8B-Instruct` weights, bf16 attention + MLP, 32 layers,
mesh 8x4 (SP=8 x TP=4), `FABRIC_1D` + Linear topology, one user, warm (a warmup prefill precedes
every measured one, so JIT and program-cache misses are outside the measurement).

---

## 1. Wall clock is noisy here — read it as a distribution

The same configuration (one-shot, 2048 tokens, 32 layers, default SDPA config), best-of-3 within
each run, run in five separate processes:

```
316 ms    777 ms    932 ms    940 ms    961 ms
```

A 3x spread on an identical configuration. Load average was ~6 on 64 cores with a single pytest
running, so this is not host contention — it is intrinsic run-to-run variance.

**Consequence: no wall-clock difference below roughly 2x is meaningful on this box**, and any
conclusion resting on a single sample is unsafe. Two hypotheses died here, both of which looked
convincing on first samples:

| hypothesis from a single sample | controlled result |
|---|---|
| "chunks below the 2048 SDPA threshold are ~3x slower (64/64 vs 256/256 q/k chunking)" | forcing a 2048 chunk onto the 64/64 config gave **1106 / 724 ms** — overlapping the 256/256 runs. **Not supported.** |
| "~90% of a chunk is fixed per-chunk overhead" | the layer sweep below shows a **15 ms** intercept, i.e. ~1.7%. **Wrong; the cost is per-layer.** |

The reliable instrument is device-side op time (§3), which is measured on-device and does not see
host scheduling noise.

Reproduce: `PREFILL_SDPA_THRESHOLD` / `PREFILL_SDPA_{Q,K}_{SMALL,LARGE}` override the program config
(defaults are the spec's values), which is what makes that a controlled experiment rather than a
correlation.

## 2. Cost is linear in LAYERS and flat in TOKENS

Layer count swept at a fixed 2048-token chunk, so the SDPA program config is identical at every
point and layer count is the only variable:

| layers | best | ms/layer |
|---|---|---|
| 1 | 31.2 ms | 31.2 |
| 4 | 120.4 ms | 30.1 |
| 16 | 510.0 ms | 31.9 |
| 32 | 906.9 ms | 28.3 |

Least squares: **28.4 ms/layer, intercept 15.4 ms, R² = 0.996.** Per-chunk fixed cost is ~1.7% of a
32-layer chunk; essentially all the time is per-layer and scales linearly.

Token count swept at a fixed 32 layers, single chunk:

| tokens | best | tok/s |
|---|---|---|
| 512 | 941 ms | 544 |
| 1024 | 1011 ms | 1,013 |
| 2048 | 777–961 ms | ~2,200 |

**4x the tokens costs nothing measurable.** A layer processing 2048 tokens does 4x the arithmetic of
one processing 512 and takes the same wall time — the signature of a latency-bound pipeline, and the
reason throughput (tok/s) rises almost linearly with chunk size while latency stays flat.

Practical consequence for chunked prefill: 4 x 512 costs 3190 ms against ~900 ms for a single 2048
chunk — the same tokens for 3.5x the time, because cost is driven by (layers x chunks), not tokens.
**Use the largest chunk that fits.** The spec's 4096 is the right instinct.

## 3. Device-side breakdown

From `generated/profiler/reports/llama31_8b_prefill_warm/` (a warm profiled run: 32 devices x 2
prefill forwards, inferred from op counts — 10240 matmuls / 32 devices / 5 per layer / 32 layers).
Normalised per device per prefill; devices run concurrently, so the per-device mean approximates the
critical path when the mesh is balanced.

**Total device kernel time: 74.8 ms per prefill — 2.34 ms per layer.**

| op | calls/layer | ms/prefill | % of device time |
|---|---|---|---|
| Matmul | 5.00 | 18.48 | 24.7 |
| **AllGatherAsync** | 5.03 | 15.83 | 21.2 |
| **Tilize** | 2.55 | 12.29 | 16.4 |
| **ReduceScatterMinimalAsync** | 3.00 | 9.84 | 13.2 |
| SDPA | 1.00 | 9.75 | 13.0 |
| LayerNorm | 2.00 | 3.78 | 5.1 |
| BinaryNg (add/mul) | 4.00 | 2.16 | 2.9 |
| RotaryEmbeddingIndexed | 2.00 | 0.81 | 1.1 |
| NlpCreateHeads / ConcatHeads | 2.00 | 1.11 | 1.5 |
| Unary (silu) | 1.00 | 0.43 | 0.6 |
| Typecast | 3.05 | 0.19 | 0.3 |
| UpdatePaddedKvCache | 2.00 | 0.14 | 0.2 |

Grouped:

- **collectives 34.3%** (8 per layer: 5 all-gather + 3 reduce-scatter, i.e. 256 per 32-layer chunk)
- **matmul 24.7%** — the only line that is useful arithmetic
- **layout conversion 16.7%** (tilize + typecast) — pure overhead
- **SDPA 13.0%**

## 4. What this means for optimisation

**1. Trace mode — the big one, and it is blocked.** 2.34 ms of device kernel work per layer against
28.4 ms of wall time means roughly **26 ms per layer is host dispatch and synchronisation**, ~12x
the device work. ~32 op dispatches per layer x 32 layers is ~1000 dispatches per chunk, each a
host→device round trip across 32 chips. Trace replay collapses that to one submission.

This is the single largest lever available, and it is unavailable: capturing a trace requires the
ring SDPA to read its cache offset from a device tensor, and
`ring_joint_scaled_dot_product_attention` has no such form (`ring_mla` does). The model-side work is
already done and validated — see `tt/trace.py` and `tests/unit/test_trace_metadata_vs_ref.py`, where
the metadata-tensor path is bit-identical to the scalar path (PCC 1.0). What remains is confined to
the op.

**2. Collectives cost more device time than matmul does.** 34.3% vs 24.7%. Worth checking whether all
eight per-layer collectives are needed: this package deliberately uses the **replicated-residual**
contract (attention and MLP each close with a full all-reduce = reduce-scatter + all-gather).
MiniMax-M3's sharded-residual variant keeps the stream at emb/tp and removes two of the three
all-gathers per MoE block; the dense analogue would remove one all-gather per sublayer here. That
was consciously out of bring-up scope (`README.md`, "Reuse vs fresh") and is now quantified: it is
the second-largest lever.

**3. Layout conversion is 16.7% of device time.** 2.55 Tilize calls per layer is more than the math
needs; worth finding which op boundary forces the retilization.

**4. Do not tune the SDPA chunk sizes on this evidence.** The controlled test found no effect
distinguishable from noise. The knobs exist (§1) if someone wants to measure properly.

## 5. Reproducing

```bash
source ttenv.sh
export HF_MODEL=/path/to/Meta-Llama-3.1-8B-Instruct

# wall clock: one-shot vs chunked sweep (no profiler — the honest end-to-end number)
pytest models/demos/llama3_1_8b_d_p/tests/prefill_perf.py -k 8x4 -s

# layer scaling (the measurement that identified the per-layer cost)
PREFILL_PERF_SEQ_LEN=2048 PREFILL_PERF_CHUNK_SIZES=0 PREFILL_PERF_LAYERS=4 \
  pytest models/demos/llama3_1_8b_d_p/tests/prefill_perf.py -k 8x4 -s

# device op breakdown (adds per-op overhead: ignore its wall clock, trust DEVICE KERNEL DURATION)
python -m tracy -r -p -v -m pytest models/demos/llama3_1_8b_d_p/tests/prefill_perf.py -k 8x4
```

Each profiled prefill writes ~7 GB of ops-times CSV and post-processing holds it in a pandas frame,
so keep `PREFILL_PERF_WARMUP` + `PREFILL_PERF_ITERS` at 1 + 1 when profiling.

## 6. Caveats

- Wall-clock and device-op numbers come from **different runs** (profiling perturbs wall clock), so
  the "8% device" figure is a ratio across runs, not a single measurement. The direction is far
  larger than the noise, but do not quote it to two significant figures.
- The profiled run in `generated/profiler/reports/` was recorded on 2026-09-05, not in this session;
  its configuration is inferred from op counts (32 layers, 2 prefill forwards, 32 devices).
- All timings are bf16 attention + MLP. The package defaults (bfp8 / bfp4) will differ.
- Single user, single pipeline stage, 2048-token context. Nothing here has been measured at the
  spec's 4096 chunk or 131072 context.
- No performance TARGETS exist for this model to gate against (the spec's `performance_targets` is
  deliberately empty), so this document describes behaviour; it does not pass or fail anything.
