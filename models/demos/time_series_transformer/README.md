# Time Series Transformer (TTNN)

TTNN implementation of HuggingFace's `TimeSeriesTransformerForPrediction` — a vanilla
encoder-decoder transformer for probabilistic time-series forecasting.

## Platforms

- Wormhole (`n150`, `n300`)

## Overview

- Full encoder-decoder stack with masked decoder self-attention and cross-attention
- Value embedding over lag features, sinusoidal positions, temporal and static covariates
- Probabilistic head: Student's t (default), Normal, Negative Binomial
- Autoregressive sampling with all trajectories advanced as one batch
- Parity-checked against the HuggingFace reference at every stage
- Two runtime profiles: a float32 accuracy profile and a bfloat16 + flash-attention profile
- Univariate and multivariate (`input_size > 1`) inputs, with observed-mask handling
- Streaming (online) forecasting over a rolling window
- Scales to 2048-step context, 256 series per batch, and 1000 sampled trajectories

Reference checkpoint: [`huggingface/time-series-transformer-tourism-monthly`](https://huggingface.co/huggingface/time-series-transformer-tourism-monthly)
(`d_model=26`, 2 encoder + 2 decoder layers, 2 heads, context 24, horizon 24, 16 lags,
Student's t output).

## Directory layout

```text
time_series_transformer/
├── README.md
├── PERF.md                     measured latency, throughput and accuracy
├── requirements.txt
├── conftest.py
├── demo/
│   └── demo.py                 forecast with prediction intervals
├── reference/
│   ├── torch_reference.py      HF harness, golden capture, CRPS/PCC metrics
│   └── tourism.py              real Monash tourism-monthly observations
├── tt/
│   ├── config.py               runtime config + HF config conversion
│   ├── weights.py              state-dict loading helpers
│   ├── ops.py                  linear, softmax, masks, activation
│   ├── layers.py               LayerNorm, FeedForward
│   ├── embeddings.py           value / positional / combined embedding
│   ├── attention.py            multi-head attention, KV cache, SDPA path
│   ├── transformer.py          encoder & decoder layers and stacks
│   ├── distribution.py         parameter projection, domain maps, sampling
│   ├── inputs.py               scaler, lags, covariate assembly
│   ├── state_io.py             checkpoint discovery and loading
│   ├── streaming.py            rolling-window online forecasting
│   ├── trace.py                trace capture and replay for generation
│   └── model.py                end-to-end model, forward, generate
└── tests/
    ├── pcc/                    per-stage parity against HuggingFace
    │   ├── test_embeddings.py      inputs, scaler, lags, embeddings
    │   ├── test_attention.py       self / cross / causal, KV cache
    │   ├── test_layers.py          encoder and decoder stacks
    │   ├── test_distribution.py    all three distribution heads
    │   ├── test_e2e_model.py       forward, generate, runtime options
    │   ├── test_multivariate.py    input_size > 1 parity
    │   ├── test_streaming.py       online forecasting
    │   └── test_benchmark.py       forecast quality on tourism-monthly
    └── perf/                   latency / throughput / scaling gates
        ├── test_perf.py            gates, scaling, trace lifecycle, pipelining
        ├── test_utilization.py     core counts, shape overhead, head switching
        └── test_perf_report.py     repository-standard perf CSV
```

## Setup

```bash
./build_metal.sh
./create_venv.sh
source python_env/bin/activate

export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

# Bind the install to the interpreter that will run the tests. A bare `pip` can resolve to
# /usr/bin/pip and user-site packages even inside an activated environment.
python -m pip install -r models/demos/time_series_transformer/requirements.txt
```

Run everything below with the same activated environment. The checkpoint is pulled from the
Hub on first use and cached.

## Run the demo

```bash
python models/demos/time_series_transformer/demo/demo.py
python models/demos/time_series_transformer/demo/demo.py --batch 8 --samples 200
python models/demos/time_series_transformer/demo/demo.py --profile performance
```

It prints per-horizon p10/p50/p90 quantiles against the held-out truth, plus MAE, MAPE and
80% interval coverage.

`--data tourism` forecasts the real Monash tourism-monthly series this checkpoint was trained
on, read from the Hub's parquet conversion through `huggingface_hub` and `pandas` -- the
`datasets` package is deliberately not used, since it drops a heavyweight dependency and keeps
the demo working against current `huggingface_hub` releases. Time features are reconstructed as
the standard GluonTS monthly pair (month of year scaled to [-0.5, 0.5], plus a log-scaled age
counter).

The default `--data auto` prefers the real data and falls back to a self-contained synthetic
seasonal series if the Hub is unreachable, so the demo runs offline too.

## Tests

```bash
# Per-stage parity against HuggingFace (needs a device)
pytest models/demos/time_series_transformer/tests/pcc/ -q

# Latency / throughput / sample-generation gates
pytest models/demos/time_series_transformer/tests/perf/ -q -s

# Repository-standard performance report (writes perf_*.csv)
pytest models/demos/time_series_transformer/tests/perf/test_perf_report.py -q -s
```

Measured numbers and the environment they came from are in [PERF.md](PERF.md).

## Runtime profiles

| | accuracy (default) | performance |
|---|---|---|
| weights / activations | float32 | bfloat16 |
| attention | eager, fused softmax kernel | `ttnn.transformer.scaled_dot_product_attention` |
| relative MAE vs reference | 0.27% | 0.59% |
| best throughput | 335.0 seq/s | 686.1 seq/s |

Select with `TimeSeriesTransformer.from_pretrained(..., dtype="bfloat16", use_sdpa=True)`.

## Implementation notes

**`d_model=26` is not tile-aligned.** Neither is `head_dim=13`. TTNN reduces and matmuls over
the *logical* width, so the modules are written against logical shapes with no padding
bookkeeping — `ttnn.layer_norm` over 26 matches `torch.nn.functional.layer_norm` exactly, and
a `linear` with `K=26` is exact even on layer-norm output. The one exception is the SDPA
kernel, which rejects a last dim whose logical and padded extents differ; the performance
profile zero-pads `head_dim` to 32 before the call and slices the result back.

**The fused softmax kernel is used despite a bad-looking row-sum error.** `ttnn.softmax`
carries roughly 3.8% row-sum error on this model's score matrices, with or without
`numeric_stable`, and independently of tile alignment — a 32-wide row is as affected as a
24-wide one. That error is close to a uniform per-row scale factor, and the layer norm after
the residual add removes it, so end to end the fused kernel is no less accurate than composing
the reduction by hand (NLL 0.02% vs 0.04%, mean MAE 0.28% vs 0.67%) and is ~15% faster.
`use_exact_softmax=True` restores the composed version for diagnosing attention numerics.

**The Stage 2 memory-layout options are pessimizations, and not just because this model is
small.** Sweeping width from 26 to 1024 and rows from 24 to 1536, interleaved DRAM is fastest at
every point — height sharding runs at 0.44x–0.98x even with the layout conversion amortised over
a chain of blocks, and L1 residency stays slower throughout because `ttnn.linear` cannot fuse a
bias against an L1-resident operand. A fused QKV projection is likewise slower than three
separate ones. The likely explanation is that `ttnn.linear` already picks a multi-core program
config for interleaved operands, so pinning the layout by hand constrains it without adding
parallelism. All measured in [PERF.md](PERF.md), reproducible via
`tests/perf/test_utilization.py`; both `use_l1` and the sharding helper are kept and
correctness-tested so the trade can be re-measured later.

**Generation is dispatch-bound, not compute-bound.** A decode step is ~95 TTNN ops on a model
this small, so wall-clock is dominated by per-op dispatch and host round-trips, not arithmetic.
Tracing takes a single-sequence forecast from ~205 ms to ~17.5 ms.

**The whole mean-mode rollout is one trace.** tt-metal locks device allocations while any trace
exists, so a trace per decode step is not viable — the second capture allocates while the first
is live. Unrolling the loop inside a single capture solves that *and* removes the per-step host
round-trips: with the loop written out, every step's lag offsets and sequence lengths are
compile-time constants, so the feedback can close on device. The lag window is gathered with one
matmul against a constant one-hot selector, and because the running series is normalized the
value fed back is exactly the distribution's pre-affine mean, so the scaler statistics never
reach the device. The encoder runs inside the same trace too, so a forecast costs three buffer
uploads and one readback in total.

Inside the rollout two further savings come free, neither changing the result by a bit: the
cross-attention keys and values are projected once and reused, since the encoder output is
constant across the forecast; and only the parameter the loop actually consumes is projected,
because for Student's t and Normal the pre-affine mean is the loc head passed through the domain
map unchanged. Measured end to end: 46 ms stepped, 23.4 ms unrolled, 20.5 ms with the encoder
folded in, 17.5 ms with both.

Sampling keeps the stepped path — a Student's t variate needs a Gamma draw whose shape parameter
is data-dependent, so it cannot be produced on device. Both paths are checked against each other.

**The O(horizon²) recompute is free here.** At `d_model=26` a 24-row window and a single token
both pad to one 32-row tile, so a step costs the same either way — measured flat at 0.754 ms per
unrolled step. That is why the KV-cache path buys nothing at this size; it would matter at long
context, and it is kept and tested for that reason.

**What runs where.** Everything from the value embedding through the distribution parameters is
TTNN on device: embeddings, layer norms, all attention, the FFNs, the parameter projections and
the domain maps. In mean mode the autoregressive feedback is on device too, including the lag
gather. Three things stay in torch on the host, by design rather than omission:

- `create_network_inputs` — the scaler, the initial lag window, static-feature assembly and the
  categorical embedding lookup. Data preparation, once per forecast.
- Drawing samples. A Student's t variate needs a Gamma draw whose shape parameter is
  data-dependent, so it cannot be produced on device or pre-generated.
- De-normalizing the final forecast by the scaler statistics.

The test suite sets `ttnn.CONFIG.throw_exception_on_fallback = True`, so any TTNN op that
silently fell back to host execution would fail the run rather than quietly pass. All reported
latency and throughput figures are end-to-end `generate()` wall-clock, host work included — they
are not device-only timings.

**Input construction runs on the host, once per forecast.** Scaling, lag gathering and covariate
assembly mirror HuggingFace's `create_network_inputs` one-for-one, which keeps parity debugging
tractable; everything from the value embedding onward runs on device. The per-step lag gather —
the part that is actually in the hot loop — is on device for mean mode, and a single indexed
read rather than one slice per lag on the sampling path.

## Accuracy

Verified against the HuggingFace reference on identical inputs:

| stage | metric | result |
|---|---|---|
| network inputs (scaler, lags, covariates) | exact match | bit-identical |
| embeddings | PCC | > 0.999 |
| attention (self, cross, causal) | PCC | > 0.999 |
| encoder / decoder stacks | PCC | > 0.99 |
| distribution parameters | PCC | > 0.999 |
| negative log-likelihood | relative | within 5% |
| CRPS | relative | within 5% |
| mean prediction | relative MAE | within 5% |
| generate, all three distributions | relative MAE | within 5% |

All three distribution heads are checked end to end. The published checkpoint carries only a
Student's t head, so Normal and Negative Binomial are verified against reference models built
with the same geometry and a randomly-initialised head.

## Forecast quality on the benchmark

Parity says the port is faithful; this says the forecasts are good. Monash tourism-monthly,
8 series, 100 sampled trajectories, real observations:

| Metric | Result |
|---|---:|
| MAPE of the median forecast | **13.0%** |
| Coverage of the nominal 80% interval | **79.2%** |

Covered by `tests/pcc/test_benchmark.py`, which skips if the Hub is unreachable. A
seasonal-naive forecast on this dataset sits around 20-25% MAPE.

A two-layer stack is held to 0.99 rather than 0.999 because attention on this checkpoint tops
out near 0.9998 — the residual is device float32 matmul precision, already present in QK^T,
and compute-kernel fidelity does not move it (HiFi4 with fp32 accumulate measures the same as
the default).

## Streaming inference

`tt/streaming.py` wraps the model in a rolling window for online forecasting: seed it with
`past_length` observations, then call `observe()` as each new sample arrives and `forecast()`
whenever a forecast is wanted.

```python
from models.demos.time_series_transformer.tt.streaming import StreamingForecaster

stream = StreamingForecaster(model, past_values=..., past_time_features=...)
stream.observe(new_value, new_time_feature)
forecast = stream.forecast(future_time_features)
```

The window length is fixed, which is the point: every forecast presents identical shapes, so
the captured trace is reused across updates instead of being recaptured. `tests/pcc/
test_streaming.py` checks that rolling forward N steps gives the same forecast as presenting
the equivalent window directly, that the observed mask streams through to the scaler, and that
no update forces a recapture.

## Pipelining

Encoder, all `prediction_length` decoder steps, and the distribution head are captured as a
**single trace**, so a forecast is one dispatch rather than one per step, and there is no host
gap between the stages to overlap:

| Measurement | Result |
|---|---|
| Trace executions per 24-step forecast | **1** |
| Share of forecast wall-clock inside that dispatch | **90%** (15.4 ms of 17.1 ms) |

Asserted in `tests/perf/test_perf.py::TestPipelining`, which counts dispatches rather than
taking the claim on trust. The remaining 10% is host-side input construction and the single
readback; there is no second device stage left to overlap it with at batch 1.

## Multivariate support

`input_size > 1` is supported end to end and checked against HuggingFace in
`tests/pcc/test_multivariate.py`:

| Aspect | Coverage |
|---|---|
| Lag window | Flattened channel-major, verified equal to the full `get_lagged_subsequences` gather |
| Feature width | `feature_size` matches HF, including the per-channel `log1p(\|loc\|)` and `log(scale)` terms |
| Distribution | Channels are an event dimension (`Independent`), as in HF |
| Generation | Mean-mode rollout parity within 5% MAE; traced path matches eager |
| Output shape | `(batch, samples, horizon, channels)` |
| Observed mask | Masked channels change the scaler and still match HF |

The published tourism checkpoint is univariate, so multivariate parity is established against
reference models built with the same geometry and a wider input. There is no multivariate
checkpoint to measure forecast *quality* against — these tests establish numerical parity with
HuggingFace, not accuracy on a multivariate benchmark.

## Known limitations

- Multivariate parity is against constructed reference models, not a trained multivariate
  checkpoint; forecast quality at `input_size > 1` is therefore unmeasured.
- Sampling closes on device only for the Normal head, where a draw is `loc + scale * z` and the
  noise can be generated in bulk and uploaded once. Student's t needs a Gamma variate whose
  shape is the predicted `df`, and the negative binomial a Poisson-Gamma pair; both keep the
  stepped host loop. Mean-mode decoding is always on device.
- A single forecast cannot fill the 64-core grid: at `d_model=26` every batch-1 activation is
  one tile. Batching is the lever, not sharding — see PERF.md.
- L1 residency and manual sharding are slower than interleaved DRAM across the whole measured
  width range, not only at this checkpoint's size. Both are kept, exercised and correctness-
  tested, but off by default.
- Exactly one trace is live at a time. A new `batch * num_parallel_samples` releases the
  previous capture and pays a fresh one (~0.2 s). Holding several live traces is what makes
  tt-metal warn that later allocations may be corrupted, so the recapture is deliberate.
- Perf numbers are sensitive to host CPU load, since the model is dispatch-bound.
