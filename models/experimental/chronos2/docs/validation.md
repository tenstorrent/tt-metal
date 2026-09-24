<!-- SPDX-FileCopyrightText: © 2026 Abror Shopulatov -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

# Chronos-2 validation on Blackhole

## Environment

- TT-Metal `fd80faa3b35fa6d38a92d08326205fc4168284ec`, one Blackhole card,
  Python 3.10.19, PyTorch 2.14.0+cpu, NumPy 1.26.4, safetensors 0.8.0.
- Checkpoint `amazon/chronos-2` at revision
  `29ec3766d36d6f73f0696f85560a422f50e8498c` (`model.safetensors` SHA-256
  `ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42`).
- Reference: chronos-forecasting's `Chronos2Pipeline` in FP32 on an NVIDIA
  A100 (PyTorch 2.14.0+cu126), with no autocast.
- CPU parity tests: chronos-forecasting 2.3.2, PyTorch 2.14.0,
  Transformers 5.17.0.

## Method

The quantile NRMSE of a series is the RMSE between TT and reference quantile
forecasts over all horizon steps and quantiles, divided by the RMS of the
reference forecast. A case passes when every series stays at or below 0.04.
Consistency cases compare the same series across different batch
compositions; they must agree within 0.01, and repeated identical calls must
be bit-identical. On the scored cases, the weighted quantile loss (WQL)
against the observed future may be at most 2% worse than the reference's.

Latency covers CPU float32 arrays in to CPU quantile arrays out. The device is
synchronized before and after each call. Weight loading and the first
(compiling) call are excluded.

The held-out suite has 35 cases:

- 7 ETTh1 columns at three windows each: context 512 / horizon 64,
  128 / 16 and 64 / 64;
- two synthetic series at the same windows: mixed sinusoids, and
  `sinh(4 sin(t))`, which alternates between tiny and huge magnitudes;
- a batch of four series, its single rows, a reversed batch and a repeated call;
- masked holes, a masked tail, and a constant series.

The development suites (1 and 8 cases) cover the same checks at smaller scale.

## Accuracy

| Precision | Held-out result | Worst NRMSE | WQL vs reference |
|---|---|---|---|
| fp32 | 35/35 pass | 0.0068 | at most +0.31% |
| bf16 | 34/35 pass | 0.069 on `sinh(4 sin)` at context 64 / horizon 64; all others ≤ 0.012 | — |

On the 8-case development suite, the worst NRMSE is 0.0024 for fp32, 0.0043
for bf16 and 0.0118 for bfp8_b. Every run has zero quantile crossings and
bit-identical repeated calls.

The FP32 PyTorch reference in `reference/` matches `Chronos2Pipeline.predict`
within 1e-3 NRMSE on CPU (`tests/test_components.py`). The device tests
compare the TTNN backend with that reference.

## Latency

`benchmarks/benchmark_forecast.py`, 20 repetitions, median, horizon 64:

| Shape | fp32 | bf16 | fp32 without trace |
|---|---|---|---|
| batch 2, context 512 | 9.65 ms | 9.41 ms | 24.9 ms |
| batch 1, context 512 | 6.83 ms | 6.66 ms | 24.5 ms |
| batch 1, context 65 | 5.72 ms | 5.60 ms | 24.4 ms |

Repeated benchmark runs vary by about ±0.5 ms.

On the development suite, the median case latency fell from 40.8 ms (first
TTNN version, fp32) to 7.8 ms with trace replay, and to 6.7 ms after op-count
reduction. For context, the reference pipeline on an A100 took 30.7 ms (fp32)
and 34.6 ms (bf16) per case on the same suite.

Weight memory on the device is about 457 MiB for fp32, 236 MiB for bf16 and
122 MiB for bfp8_b.

## Design choices

- **Trace replay.** Eager execution is dispatch-bound (about 500 small ops;
  latency barely changes with sequence length), so each static input shape is
  captured once and replayed.
- **One live trace.** Keeping traces for several shapes alive, and allocating a
  new shape's buffers while another trace existed, hung the device on a later
  replay. A new shape therefore releases the current trace first.
- **Op-count reduction.** These folds are exact in FP32 at load time:
  - RMSNorm weights fold into the following linears;
  - `rotate_half` folds into the q/k weight columns, so RoPE plus q/k/v is two
    linears and two elementwise ops;
  - group attention over independent series reduces to one `W_o @ W_v` linear.
- **Composed softmax and RMSNorm.** The attention softmax is composed from
  elementwise ops in FP32. At these masked, non-tile-aligned sequence lengths
  it was more accurate than `ttnn.softmax` (relative error 1.8e-2), and the
  same held for RMSNorm against `ttnn.rms_norm`.
- **bf16 policy.** Plain bf16 reached 0.041 on short-context, high-dynamic-range
  stress inputs. The shipped bf16 graph fixes most of that:
  - HiFi4 with FP32 accumulation on every matmul;
  - an FP32 residual stream;
  - FP32 for the final norm and quantile head.

  This brings those inputs to at most 0.017 at about 11% extra latency. HiFi4
  alone, the FP32 residual alone and HiFi2 variants were not enough.
- **bfp8_b.** Storing only the linear weights as BFP8_B halves weight memory
  but does not reduce latency, because replay is dispatch-bound. It increases
  error by about 1.3× compared with plain bf16.

## Limitations

- Univariate only. Multivariate groups and past or future covariates are not
  implemented. For fully masked time steps the fused group attention differs
  from the original, but those steps never reach the output.
- Validated for batch ≤ 4, context ≤ 512 and horizon ≤ 64. Longer contexts are
  truncated to the checkpoint's context length but are not validated.
- A change of input shape re-captures the trace (about 70 ms warm, seconds on
  the first compile). Workloads that alternate shapes should pad or bucket.
- bf16 fails one held-out stress case; use fp32 when accuracy matters.
- The first process start spends about 20 seconds compiling kernels.
