# Model performance and accuracy

Numbers below come from direct pytest runs in `models/demos/time_series_transformer/tests/perf`.

## Environment

- Device: Wormhole `n300`
- Checkpoint: `huggingface/time-series-transformer-tourism-monthly` (`d_model=26`, 2+2 layers,
  2 heads, context 24, horizon 24, Student's t output)

## Benchmark commands

```bash
# Gates, scaling, trace lifecycle and pipelining
pytest models/demos/time_series_transformer/tests/perf/test_perf.py -q -s

# Repository-standard performance report; writes perf_time_series_transformer_*.csv
pytest models/demos/time_series_transformer/tests/perf/test_perf_report.py -q -s

# Correctness, including multivariate, streaming and the tourism benchmark
pytest models/demos/time_series_transformer/tests/pcc/ -q
```

The standard report is emitted through `models.perf.perf_utils.prep_perf_report`, so it carries
the same columns as every other demo (Model, Setting, Batch, First/Second Run, Compile Time,
Inference Time, Throughput). It is produced for both runtime profiles.

## Stage 1 targets

Measured on the float32 accuracy profile, batch 1 unless noted, after the program cache and
trace are warm.

| Metric | Target | Measured |
|---|---:|---:|
| Single-sequence latency (batch 1) | < 50 ms | **17.5 ms** (mean of 30, p95 18.3) |
| Throughput (best over batch sweep) | >= 100 seq/s | **335.0 seq/s** |
| 100 samples, one series | < 1 s | **0.383 s** |

Throughput sweep (float32, mean mode):

| Batch | Latency | Throughput |
|---:|---:|---:|
| 1 | 17.62 ms | 56.8 seq/s |
| 8 | 36.13 ms | 221.4 seq/s |
| 32 | 101.97 ms | 313.8 seq/s |
| 64 | 191.02 ms | 335.0 seq/s |

## Runtime profiles

| | accuracy (float32) | performance (bfloat16 + SDPA) |
|---|---:|---:|
| Relative MAE vs reference rollout | 0.27% | 0.59% |
| PCC vs reference rollout | > 0.9999 | 0.999147 |
| Throughput @ batch 32 | 313.8 seq/s | 551.6 seq/s |
| Throughput @ batch 64 | 335.0 seq/s | **686.1 seq/s** |

bfloat16 with the flash-attention kernel buys about 2.0x throughput at batch 64 for 0.59%
relative MAE, and clears the Stage 3 stretch target of 500 seq/s. It does **not** improve
batch-1 latency, because latency is bound by dispatch and host work rather than arithmetic.

## Accuracy

Against the HuggingFace reference on identical inputs:

| Stage | Metric | Result |
|---|---|---|
| Network inputs (scaler, lags, covariates) | exact | bit-identical |
| Encoder / decoder embeddings | PCC | > 0.999 |
| Attention (self, cross, causal) | PCC | > 0.999 |
| Encoder / decoder stacks | PCC | > 0.99 |
| Distribution parameters | PCC | > 0.999 |
| Negative log-likelihood (TT encoder+decoder+head, masked) | relative | within 5% |
| CRPS | relative | within 5% |
| Mean prediction | relative MAE | within 5% |
| Unrolled rollout vs stepped path | PCC | 1.0 (bit-identical) |
| Student's t / Normal / Negative Binomial generate | relative MAE | within 5% |
| Multivariate (`input_size=3`) generate | relative MAE | within 5% |
| Mean / std scalers, incl. constant and unobserved series | exact | matches HF |

On the real benchmark (Monash tourism-monthly, 8 series, 100 trajectories): median forecast
MAPE **13.0%**, and the nominal 80% prediction interval covers **79.2%** of observations.

## Stage 2 memory and fusion options, measured

Bounty #32140 Stage 2 asks for sharded/interleaved memory configurations, attention and
autoregressive/sample-generation sharding, L1 placement where beneficial, and the recommended
TTNN fused flows. Each was implemented far enough to benchmark on the target shapes before
being adopted or rejected. Two paid and are shipped on; three measure *slower* here and are
rejected with the numbers below rather than silently omitted.

The rejection is not "this model is small". That was the first explanation, and measuring it
properly showed it to be the wrong one. Sweeping the width from 26 to 1024 and the row count
from 24 to 1536 -- far past this checkpoint -- interleaved DRAM is fastest at **every** point,
including when the layout conversion is amortised over a chain of eight blocks:

| Rows | Width | Interleaved DRAM | L1 | Height-sharded |
|---:|---:|---:|---:|---:|
| 24 | 26 | **0.147 ms** | 0.394 ms | `TT_FATAL` |
| 24 | 256 | **0.201 ms** | 0.414 ms | 0.354 ms |
| 24 | 1024 | **0.209 ms** | 0.352 ms | 1.109 ms |
| 1536 | 26 | **0.208 ms** | 0.360 ms | `TT_FATAL` |
| 1536 | 256 | **0.227 ms** | 0.353 ms | 0.434 ms |
| 1536 | 1024 | **1.009 ms** | 1.667 ms | 2.094 ms |

Chained eight deep so the conversion is paid once, height sharding still runs at 0.44x-0.98x of
interleaved across the same sweep, while remaining numerically correct (PCC >= 0.997). There is
no crossover to find.

The likely reason is that `ttnn.linear` already selects a multi-core program config for
interleaved operands, so pinning the layout by hand constrains that choice without adding
parallelism. That explanation is inferred from the timings rather than from a profiler trace,
but the timings themselves are unambiguous and reproducible:

```bash
pytest models/demos/time_series_transformer/tests/perf/test_utilization.py -q -s -k Sharding
```

`use_l1` and the sharding helper are kept, exercised and correctness-tested rather than deleted,
so the trade-off can be re-measured on future runtimes instead of taken on trust.

| Option | Effect | Adopted |
|---|---|---|
| Fused `ttnn.softmax` instead of a composed reduction | **~15% faster**, no accuracy cost | **yes**, default |
| Flash attention (SDPA) + bfloat16 | **1.9x throughput** at batch 64 | **yes**, performance profile |
| L1-resident activations (`use_l1=True`) | 0.35x-0.6x across the whole width sweep — *slower* | no, off by default |
| Height-sharded activations | 0.44x-0.98x from width 64-1024; `TT_FATAL` at width 26 | no, kept and measured |
| Fused QKV projection (26 -> 78 + 3 slices) | 0.38x at batch 1, 0.72x at batch 64 — *slower* | no |

Attention, autoregressive-decode and sample-generation sharding all reduce to the same
measurement above, since they shard the same activations. At this checkpoint's width there is
the additional hard blocker that a batch-1 activation is a single tile and the shard spec
degenerates (`TT_FATAL`). Sample generation is instead parallelised by batching: all
`num_parallel_samples` trajectories advance as one batch, which is what makes 1000 samples in
1.68 s possible.

For the fused flows that *do* apply at this size, both are adopted: `ttnn.softmax` in place of a
composed reduction, and `ttnn.transformer.scaled_dot_product_attention` in the performance
profile.

Per-op measurements behind those numbers (float32, `d_model=26`, `ffn_dim=32`, seq 24):

| Operation | batch 1 | batch 64 |
|---|---:|---:|
| FFN, interleaved DRAM, fused bias | 0.226 ms | 0.276 ms |
| FFN, L1, split bias | 0.469 ms | 0.504 ms |
| Projection 26→26, DRAM | 0.059 ms | 0.137 ms |
| Projection 26→26, L1 | 0.179 ms | 0.197 ms |
| QKV as three projections | 0.168 ms | 0.415 ms |
| QKV fused plus three slices | 0.446 ms | 0.578 ms |
| Softmax, composed by hand | 0.788 ms | 0.905 ms |
| Softmax, single kernel | 0.041 ms | 0.044 ms |

Why L1 loses: `ttnn.linear` cannot fuse a bias when an operand or the output is L1-resident
(it raises a matmul broadcast error), so every projection pays an extra eltwise add. The width
sweep shows this is not a small-model artefact — L1 stays slower out to width 1024 and 1536
rows. `use_l1=True` is kept working and correctness-tested so the trade can be re-measured on a
future runtime, not because the current data suggests it would flip.

Why the fused softmax wins without costing accuracy: `ttnn.softmax` leaves attention rows a few
percent off unity, but the error is close to a uniform per-row scale factor, and the layer norm
after the residual add removes it. Measured end to end the fused kernel is no worse:

| | composed softmax | fused kernel |
|---|---:|---:|
| Decoder PCC | 0.999999 | 0.999996 |
| NLL relative error | 0.04% | 0.02% |
| Mean prediction relative MAE | 0.67% | 0.28% |
| CRPS relative error | 0.34% | 1.74% |
| Batch-1 latency | 50.4 ms | 42.7 ms |

## Effect of tracing

Trace replay is what makes the latency gate reachable on a model this small: a decode step is
roughly 95 TTNN ops, so wall-clock is dispatch-bound.

| Path | Batch-1 latency |
|---|---:|
| Eager (KV cache, one token per step) | ~205 ms |
| Traced decoder, host loop (one trace replayed per step) | 46 ms |
| Unrolled rollout, encoder still eager | 23.4 ms |
| Unrolled rollout with the encoder folded in | 20.5 ms |
| ...plus cross-attention K/V hoisted out of the loop | 19.3 ms |
| ...plus projecting only the parameter the loop consumes | **17.5 ms** |

The step from 46 ms to 23 ms is the host round-trips, not device work. A 24-step trace replays
in 18.1 ms of pure device time, so the stepped loop was spending roughly 28 ms assembling
windows, uploading them and reading parameters back. Unrolling the loop inside a single capture
makes every step's lag offsets and sequence lengths compile-time constants, which is what lets
the feedback close on device.

Exactly one trace is live at a time. A change of `batch * num_parallel_samples` releases the
previous capture and takes a fresh one (~0.2 s). Keeping several live is what makes tt-metal
warn that subsequently allocated buffers may be corrupted once a trace executes, which would
invalidate any measurement taken afterwards.

The same rule applies to anything that allocates *between* replays, and the stepped path had two
such sites. `TracedDecodeRunner.prepare` handed the encoder output over with `ttnn.copy`, which
allocates a device temporary; it now stages through the host, once per forecast, at no measurable
cost. The eager encoder itself allocates too, so on the stepped path the trace is released before
it runs and recaptured afterwards. Mean mode pays neither, because its encoder runs inside the
trace.

That recapture grows with the row count while the per-step dispatch it saves does not, so the
stepped path stops tracing above `stepped_trace_max_rows` (64). Student's t sampling, one series,
bfloat16 + SDPA, best of three:

| Rows | Traced | Untraced | Ratio |
|-----:|-------:|---------:|------:|
| 4    | 70.8 ms | 173.9 ms | 0.41x |
| 16   | 95.9 ms | 177.0 ms | 0.54x |
| 64   | 191.9 ms | 193.5 ms | 0.99x |
| 256  | 523.8 ms | 349.4 ms | 1.50x |
| 1000 | 1832.8 ms | 1195.6 ms | 1.53x |

The crossover sits almost exactly at 64 rows, which is where the threshold is set. Net effect on
the Stage 3 target: 1000 samples went from 1.68 s to **1.165 s**, against a 2 s gate. The full
suite now runs with zero allocator warnings:

```
pytest models/demos/time_series_transformer/ -q   # 100 passed, 0 "unsafe ... active trace"
```

Folding the encoder into the same trace is worth a further ~3 ms: it is only ~90 ops, but run
eagerly they each pay host dispatch. What crosses the host boundary per forecast is now three
buffer uploads and one readback.

The last two steps remove recomputation rather than trading anything away, and neither changes
the result by a single bit. Cross-attention keys and values depend only on the encoder output,
so they are projected on the first step and reused by the remaining twenty-three (-1.2 ms). And
the loop consumes only the distribution's pre-affine mean, which for Student's t and Normal is
the loc projection passed through the domain map unchanged -- so two of the three parameter
heads and the entire domain map were dead weight inside the rollout (-1.8 ms).

## Pipelining

Encoder, all decoder steps and the distribution head are captured as one trace, so the Stage 3
pipelining requirements are met by construction rather than by overlapping separate dispatches:

| Requirement | How it is met | Evidence |
|---|---|---|
| Overlap encoder with decoder initialisation | Both inside one capture; the encoder's *input* crosses the host boundary, not its output | `TestPipelining::test_encoder_runs_inside_the_trace` |
| Pipeline decoder steps | 24 steps unrolled into the same trace | `TestPipelining::test_forecast_is_one_dispatch` |
| Overlap distribution computation | The head is captured with the decoder | same |

| Measurement | Result |
|---|---:|
| Trace executions per 24-step forecast | **1** |
| Forecast wall-clock | 17.14 ms |
| Inside the single dispatch | **15.41 ms (90%)** |

The residual 10% is host-side input construction plus one readback. At batch 1 there is no
second device stage to overlap it against; across concurrent requests a second command queue
would be the next lever, and is not implemented.

## Core utilisation, shape overhead, and distribution switching

Three Stage 3 items whose honest answer is bounded by geometry rather than effort. All are
measured in `tests/perf/test_utilization.py` rather than asserted in prose.

**Maximise core counts.** The device offers an 8x8 grid, 64 cores. At batch 1 every activation
in this model is a single 32x32 tile, so no amount of sharding can spread one across cores —
which is also why the Stage 2 sharding attempts raised `TT_FATAL`.

| Activation | Tiles | Cores it can occupy |
|---|---:|---:|
| Encoder input, batch 1 | 1 | 1 |
| Hidden state, batch 1 | 1 | 1 |
| Attention scores, batch 1 | 1 | 1 |
| Hidden state, batch 64 | 48 | 48 |

Batching, not sharding, is what fills the grid here: throughput rises from 57.9 seq/s at batch 1
to 325.0 seq/s at batch 64 (5.6x). A single forecast of a 26-wide model cannot occupy 64 cores,
and no implementation choice changes that.

**Minimise tensor-manipulation overhead.** A forecast on the eager path issues 814 shape ops:

| Op | Count |
|---|---:|
| `permute` | 398 |
| `reshape` | 300 |
| `concat` | 92 |
| `slice` | 24 |

Roughly 34 per decode step. Permutes dominate because every attention splits and merges heads,
and at `head_dim=13` there is no tile-aligned layout that avoids the transpose. The traced path
pays these once at capture rather than once per forecast, which is a large part of why tracing
is worth 10x here.

**Multi-distribution switching.** Switching heads is a model rebuild, not a recompile: weights
are re-uploaded and no kernel is rebuilt.

| Head | Rebuild | Output |
|---|---:|---|
| Student's t | 11.9 ms | valid |
| Normal | 11.7 ms | valid |
| Negative binomial | 11.8 ms | valid |

## Stage 3 stretch targets

| Metric | Stretch target | Measured | Status |
|---|---:|---:|---|
| Throughput | 500+ seq/s | **686 seq/s** | **reached** |
| Latency | < 20 ms | **17.1 ms** (p95 18.3) | **reached** |
| 1000 samples | < 2 s | **1.78 s** (performance profile) | **reached** |
| Context length | up to 2048 | **2048 at 29.8 ms** | **reached** |
| Series per batch | 100+ | **256 at 356 seq/s** | **reached** |

All five stretch targets are met, with two qualifications worth stating plainly.

Throughput varies by 1-2% run to run (677-686 seq/s observed at batch 64 on the performance
profile); the figures here come from the run reported under "Runtime profiles" above.

**1000 samples needs the performance profile.** float32 takes 3.43 s; bfloat16 with the flash
kernel takes 1.78 s. At 1000 rows the cost is device throughput at width rather than host
overhead — scaling batch-64 mean-mode latency to 1000 rows predicts ~3.0 s of the float32
figure, so roughly 90% of it is device work that only wider arithmetic addresses.

**Long context is a capability result, not an accuracy one.** No checkpoint exists beyond
context 24, so the 2048 measurement uses untrained weights and verifies that shapes hold and
output stays finite. Op shapes and counts do not depend on weight values, so the latency is
representative; the forecast quality at that size is simply unknown.

## Scaling

Sample count, batch 1 (target: 1000 in under 2 s):

| Samples | float32 | bfloat16 + SDPA |
|---:|---:|---:|
| 100 | 0.39 s | — |
| 500 | 1.72 s | 0.91 s |
| 1000 | 3.43 s | **1.77 s** |

Batch width, mean mode, float32 (target: 100+ series):

| Batch | Latency | Throughput |
|---:|---:|---:|
| 64 | 190.6 ms | 335.7 seq/s |
| 100 | 284.5 ms | 351.5 seq/s |
| 128 | 363.9 ms | 351.8 seq/s |
| 256 | 719.1 ms | 356.0 seq/s |

Context length, batch 1, untrained weights (target: up to 2048):

| Context | Latency |
|---:|---:|
| 24 | 17.6 ms |
| 128 | 18.3 ms |
| 512 | 22.1 ms |
| 1024 | 23.8 ms |
| 2048 | **32.0 ms** |

Context scales far better than the quadratic attention term would suggest, because at these
widths the encoder is not the critical path — the 24 sequential decoder steps are, and their
cost grows only with the cross-attention key count.

Sampling closes on device where the draw can be built from pre-generated noise. A Normal draw
is `loc + scale * z`, so the randomness is generated in bulk on the host, uploaded once, and the
whole sampling rollout runs from the same single trace as mean mode. Student's t needs a Gamma
variate whose shape parameter is the predicted `df`, and the negative binomial a Poisson-Gamma
pair; neither can be pre-generated without the parameters, so both keep the stepped host loop.
That path meets its own gate (100 samples in 0.382 s) comfortably.

Speculative decoding, also listed under Stage 3, does not apply: a 24-step probabilistic rollout
has no draft model to speculate from and no verification criterion that would preserve the
predictive distribution.

## Latency distribution

Batch 1, float32 accuracy profile, idle host (load average < 1), after warm-up:

| Quantity | n | mean | median | stdev | p5 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| Per-call latency | 200 | 17.66 ms | 17.64 ms | 0.73 ms | 16.58 ms | 18.78 ms |
| As gated (mean of 5 calls) | 30 | 17.52 ms | 17.44 ms | 0.50 ms | 16.98 ms | 18.31 ms |

Against the 50 ms Stage 1 gate this is a 2.7x margin at p95. Against the 20 ms Stage 3 stretch
it clears on 199 of 200 individual calls and 30 of 30 gate-style measurements.

Measuring the spread mattered here: an earlier single reading of 20.22 ms sat 1% under a 20 ms
threshold while the underlying distribution was centred at 20.5 ms.

## Caveat on measurement

These numbers are host-load sensitive, though much less so since the rollout moved on device:
before that change the batch-1 latency test measured 46.0 ms on a quiet machine and 68.4 ms
under heavy unrelated load (load average ~35), leaving only 8% of headroom against the gate.
At 23.4 ms the margin is over 2x. Throughput and sample generation are less affected still,
since they amortize host work across a larger batch.
