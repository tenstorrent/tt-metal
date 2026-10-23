<!-- SPDX-FileCopyrightText: © 2026 Abror Shopulatov -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

# Chronos-2 on Blackhole

Experimental TTNN inference for [amazon/chronos-2](https://huggingface.co/amazon/chronos-2)
(120M parameters, Apache-2.0), an encoder-only T5-style probabilistic
forecaster that predicts 21 quantiles per step in a single forward pass.
The port targets univariate series with batch 1–4, contexts up to 512 points
and horizons of 1–64 steps. FP32 is the default; `bf16` and `bfp8_b` are
available as opt-in precisions.

All learned compute runs on the device. The host does the FP32 input scaling,
patching and output unscaling, which match chronos-forecasting exactly. The
graph is captured into a metal trace and replayed. Weight folds done once at
load time cut the op count from about 900 to 500 per forward.

## Layout

| Directory | Contents |
|---|---|
| `tt/` | Configuration, host preprocessing, weight loading/folds, TTNN executor, public `create_backend` API |
| `demo/` | CSV-in / quantile-CSV-out forecasting CLI |
| `reference/` | FP32 PyTorch implementation used as the test oracle |
| `tests/` | CPU tests, parity tests against chronos-forecasting, opt-in device tests |
| `benchmarks/` | Latency benchmark of the public forecast path |
| `docs/validation.md` | Environment, method, accuracy and latency results, limitations |

## Run a forecast

Use a built TT-Metal environment with TTNN, PyTorch, NumPy and safetensors.
Download `config.json` and `model.safetensors` of `amazon/chronos-2` at revision
`29ec3766d36d6f73f0696f85560a422f50e8498c`. Run commands from the TT-Metal root:

```sh
python -m models.experimental.chronos2.demo.demo \
  --checkpoint /path/to/chronos-2 --input series.csv --index-column ts \
  --output forecast.csv --prediction-length 64
```

The input CSV has one column per series (oldest row first; empty cells are
missing values). The output has one row per series and step, with one column
per quantile. From Python:

```python
import ttnn
from models.experimental.chronos2.tt import DEVICE_OPTIONS, create_backend

device = ttnn.open_device(device_id=0, **DEVICE_OPTIONS)  # the caller owns the device
backend = create_backend("/path/to/chronos-2", None, device)  # precision="fp32" by default
quantiles = backend.forecast(past_values, past_observed_mask, prediction_length=64)["quantiles"]
backend.release()
ttnn.close_device(device)
```

`past_values` and `past_observed_mask` are `[B, T]` float32 arrays (mask 1 =
observed); the result is `[B, prediction_length, 21]`.

## Test and benchmark

```sh
# CPU tests; checkpoint tests skip without CHRONOS2_CHECKPOINT
pytest models/experimental/chronos2/tests
# Parity with the official implementation needs `pip install chronos-forecasting`
# Device tests (fp32, bf16, bfp8_b) are opt-in:
CHRONOS2_CHECKPOINT=/path/to/chronos-2 CHRONOS2_TEST_DEVICE=0 pytest models/experimental/chronos2/tests

python -m models.experimental.chronos2.benchmarks.benchmark_forecast \
  --checkpoint /path/to/chronos-2 --reps 20 [--precision bf16] [--eager]
```

## Results

Measured on one Blackhole card against chronos-forecasting's `Chronos2Pipeline`
run in FP32 on CUDA. Details are in [docs/validation.md](docs/validation.md).

| | fp32 (default) | bf16 |
|---|---|---|
| Held-out suite (35 cases), worst quantile NRMSE | 0.0068, all pass | 34/35 pass; one short high-dynamic-range case at 0.069 |
| Weighted quantile loss vs the reference | within +0.31% | — |
| Latency, batch 1, context 512, horizon 64 | 6.8 ms | 6.7 ms |

The same graph without trace replay takes 24.5 ms per call. The first
straightforward TTNN version took about 41 ms.

## Limitations

- Univariate forecasting only: no multivariate groups or covariates.
- Validated for batch ≤ 4, context ≤ 512 and horizon ≤ 64.
- One captured trace at a time; a change of input shape re-captures (about 70 ms).
- bf16 can exceed the accuracy target on very short, high-dynamic-range contexts.
  Use fp32 when accuracy matters; it costs about 0.2 ms more per call.
