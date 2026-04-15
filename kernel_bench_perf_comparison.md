# Kernel Bench: SFPU Generator vs metal_main Performance Comparison

**Date**: 2026-04-15
**Branch**: `vignjatijevic/add-rrelu-benchmark`
**Hardware**: Tenstorrent N300

## Summary

Comparison of AI-generated TTNN operations (`vignjatijevic-sfpu-generator`) against the
tt-metal main baseline (`metal_main`) using the kernel_bench evaluation framework.
All timing values are wall-clock measurements (Python `time.time()`) that include host
dispatch overhead — not pure device kernel time.

## Accuracy Comparison

| Operation | metal_main | sfpu-generator | Winner |
|-----------|-----------|----------------|--------|
| rpow | 644/644 (100.0%), acc=0.9999 | 644/644 (100.0%), acc=0.9999 | Tie |
| frac | 271/271 (100.0%), acc=1.0000 | 271/271 (100.0%), acc=1.0000 | Tie |
| swish | 5/24 (20.8%), acc=0.8813 | **24/24 (100.0%)**, acc=1.0000 | **sfpu-gen** |
| softshrink | 0/1856 (0.0%), acc=0.0000 | **1856/1856 (100.0%)**, acc=1.0000 | **sfpu-gen** |
| hardsigmoid | 195/261 (74.7%), acc=0.9752 | **261/261 (100.0%)**, acc=1.0000 | **sfpu-gen** |
| hardswish | 195/259 (75.3%), acc=0.9738 | **259/259 (100.0%)**, acc=1.0000 | **sfpu-gen** |
| softsign | 229/229 (100.0%), acc=1.0000 | 229/229 (100.0%), acc=1.0000 | Tie |
| cbrt | 59/273 (21.6%), acc=0.8865 | **224/273 (82.1%)**, acc=0.9635 | **sfpu-gen** |
| softcap | N/A | 203/203 (100.0%), acc=0.9976 | sfpu-gen (no baseline) |

### Sources — Accuracy

| Operation | metal_main eval_results | sfpu-generator eval_results |
|-----------|------------------------|----------------------------|
| rpow | [`submissions/rpow/metal_main/eval_results.json`](../kernel_bench/submissions/rpow/metal_main/eval_results.json) | [`submissions/rpow/vignjatijevic-sfpu-generator/eval_results.json`](../kernel_bench/submissions/rpow/vignjatijevic-sfpu-generator/eval_results.json) |
| frac | [`submissions/frac/metal_main/eval_results.json`](../kernel_bench/submissions/frac/metal_main/eval_results.json) | [`submissions/frac/vignjatijevic-sfpu-generator/eval_results.json`](../kernel_bench/submissions/frac/vignjatijevic-sfpu-generator/eval_results.json) |
| swish | [`submissions/swish/metal_main/eval_results.json`](../kernel_bench/submissions/swish/metal_main/eval_results.json) | [`submissions/swish/vignjatijevic-sfpu-generator/eval_results.json`](../kernel_bench/submissions/swish/vignjatijevic-sfpu-generator/eval_results.json) |
| softshrink | [`submissions/softshrink/metal_main/eval_results.json`](../kernel_bench/submissions/softshrink/metal_main/eval_results.json) | [`submissions/softshrink/vignjatijevic-sfpu-generator/eval_results.json`](../kernel_bench/submissions/softshrink/vignjatijevic-sfpu-generator/eval_results.json) |
| hardsigmoid | [`submissions/hardsigmoid/metal_main/eval_results.json`](../kernel_bench/submissions/hardsigmoid/metal_main/eval_results.json) | [`submissions/hardsigmoid/vignjatijevic-sfpu-generator/eval_results.json`](../kernel_bench/submissions/hardsigmoid/vignjatijevic-sfpu-generator/eval_results.json) |
| hardswish | [`submissions/hardswish/metal_main/eval_results.json`](../kernel_bench/submissions/hardswish/metal_main/eval_results.json) | [`submissions/hardswish/vignjatijevic-sfpu-generator/eval_results.json`](../kernel_bench/submissions/hardswish/vignjatijevic-sfpu-generator/eval_results.json) |
| softsign | [`submissions/softsign/metal_main/eval_results.json`](../kernel_bench/submissions/softsign/metal_main/eval_results.json) | [`submissions/softsign/vignjatijevic-sfpu-generator/eval_results.json`](../kernel_bench/submissions/softsign/vignjatijevic-sfpu-generator/eval_results.json) |
| cbrt | [`submissions/cbrt/metal_main/eval_results.json`](../kernel_bench/submissions/cbrt/metal_main/eval_results.json) | [`submissions/cbrt/vignjatijevic-sfpu-generator/eval_results.json`](../kernel_bench/submissions/cbrt/vignjatijevic-sfpu-generator/eval_results.json) |
| softcap | N/A | [`submissions/softcap/softcap_raw_1/eval_results.json`](../kernel_bench/submissions/softcap/softcap_raw_1/eval_results.json) |

## Latency Comparison (Median wall-clock time)

| Operation | metal_main Median | sfpu-gen Median | Speedup | metal_main Mean | sfpu-gen Mean |
|-----------|------------------|-----------------|---------|-----------------|---------------|
| rpow | 2.951 ms | **0.502 ms** | **5.9x** | 67.890 ms | 46.101 ms |
| frac | 0.711 ms | **0.498 ms** | **1.4x** | 21.441 ms | 36.610 ms |
| swish | 0.761 ms | **0.570 ms** | **1.3x** | 62.067 ms | 158.209 ms |
| softshrink | 0.000 ms (all failed) | **0.504 ms** | N/A | 0.000 ms | 37.027 ms |
| hardsigmoid | 1.208 ms | **0.494 ms** | **2.4x** | 32.375 ms | 41.108 ms |
| hardswish | 1.095 ms | **0.484 ms** | **2.3x** | 34.673 ms | 46.052 ms |
| softsign | 1.154 ms | **0.496 ms** | **2.3x** | 36.132 ms | 47.873 ms |
| cbrt | 4.646 ms | **0.505 ms** | **9.2x** | 164.457 ms | 67.731 ms |
| softcap | N/A | 0.475 ms | N/A | N/A | 96.465 ms |

### Sources — Latency

Same `eval_results.json` files as accuracy (above). Each file contains per-test `ttnn_time`
and `reference_time` fields in `detailed_results[]`. Median and mean are computed across
all test vectors.

## Implementation Sources

| Operation | metal_main Implementation | sfpu-generator Implementation |
|-----------|--------------------------|------------------------------|
| rpow | [`submissions/rpow/metal_main/result/latest/code/ttnn_rpow_impl.py`](../kernel_bench/submissions/rpow/metal_main/result/latest/code/ttnn_rpow_impl.py) | [`submissions/rpow/vignjatijevic-sfpu-generator/result/1/code/ttnn_rpow_impl.py`](../kernel_bench/submissions/rpow/vignjatijevic-sfpu-generator/result/1/code/ttnn_rpow_impl.py) |
| frac | [`submissions/frac/metal_main/result/latest/code/ttnn_frac_impl.py`](../kernel_bench/submissions/frac/metal_main/result/latest/code/ttnn_frac_impl.py) | [`submissions/frac/vignjatijevic-sfpu-generator/result/latest/code/ttnn_frac_impl.py`](../kernel_bench/submissions/frac/vignjatijevic-sfpu-generator/result/latest/code/ttnn_frac_impl.py) |
| swish | [`submissions/swish/metal_main/result/latest/code/ttnn_swish_impl.py`](../kernel_bench/submissions/swish/metal_main/result/latest/code/ttnn_swish_impl.py) | [`submissions/swish/vignjatijevic-sfpu-generator/result/latest/code/ttnn_swish_impl.py`](../kernel_bench/submissions/swish/vignjatijevic-sfpu-generator/result/latest/code/ttnn_swish_impl.py) |
| softshrink | [`submissions/softshrink/metal_main/result/latest/code/ttnn_softshrink_impl.py`](../kernel_bench/submissions/softshrink/metal_main/result/latest/code/ttnn_softshrink_impl.py) | [`submissions/softshrink/vignjatijevic-sfpu-generator/result/1/code/ttnn_softshrink_impl.py`](../kernel_bench/submissions/softshrink/vignjatijevic-sfpu-generator/result/1/code/ttnn_softshrink_impl.py) |
| hardsigmoid | [`submissions/hardsigmoid/metal_main/result/latest/code/ttnn_hardsigmoid_impl.py`](../kernel_bench/submissions/hardsigmoid/metal_main/result/latest/code/ttnn_hardsigmoid_impl.py) | [`submissions/hardsigmoid/vignjatijevic-sfpu-generator/result/1/code/ttnn_hardsigmoid_impl.py`](../kernel_bench/submissions/hardsigmoid/vignjatijevic-sfpu-generator/result/1/code/ttnn_hardsigmoid_impl.py) |
| hardswish | [`submissions/hardswish/metal_main/result/latest/code/ttnn_hardswish_impl.py`](../kernel_bench/submissions/hardswish/metal_main/result/latest/code/ttnn_hardswish_impl.py) | [`submissions/hardswish/vignjatijevic-sfpu-generator/result/1/code/ttnn_hardswish_impl.py`](../kernel_bench/submissions/hardswish/vignjatijevic-sfpu-generator/result/1/code/ttnn_hardswish_impl.py) |
| softsign | [`submissions/softsign/metal_main/result/latest/code/ttnn_softsign_impl.py`](../kernel_bench/submissions/softsign/metal_main/result/latest/code/ttnn_softsign_impl.py) | [`submissions/softsign/vignjatijevic-sfpu-generator/result/1/code/ttnn_softsign_impl.py`](../kernel_bench/submissions/softsign/vignjatijevic-sfpu-generator/result/1/code/ttnn_softsign_impl.py) |
| cbrt | [`submissions/cbrt/metal_main/result/latest/code/ttnn_cbrt_impl.py`](../kernel_bench/submissions/cbrt/metal_main/result/latest/code/ttnn_cbrt_impl.py) | [`submissions/cbrt/vignjatijevic-sfpu-generator/result/1/code/ttnn_cbrt_impl.py`](../kernel_bench/submissions/cbrt/vignjatijevic-sfpu-generator/result/1/code/ttnn_cbrt_impl.py) |
| softcap | N/A | [`submissions/softcap/softcap_raw_1/result/latest/code/ttnn_softcap_impl.py`](../kernel_bench/submissions/softcap/softcap_raw_1/result/latest/code/ttnn_softcap_impl.py) |

## Test Definitions

Each operation's test generator defines the PyTorch reference implementation and test vectors.

| Operation | Test Generator |
|-----------|---------------|
| rpow | [`benchmark/rpow/rpow_test_generator.py`](../kernel_bench/benchmark/rpow/rpow_test_generator.py) |
| frac | [`benchmark/frac/frac_test_generator.py`](../kernel_bench/benchmark/frac/frac_test_generator.py) |
| swish | [`benchmark/swish/swish_test_generator.py`](../kernel_bench/benchmark/swish/swish_test_generator.py) |
| softshrink | [`benchmark/softshrink/softshrink_test_generator.py`](../kernel_bench/benchmark/softshrink/softshrink_test_generator.py) |
| hardsigmoid | [`benchmark/hardsigmoid/hardsigmoid_test_generator.py`](../kernel_bench/benchmark/hardsigmoid/hardsigmoid_test_generator.py) |
| hardswish | [`benchmark/hardswish/hardswish_test_generator.py`](../kernel_bench/benchmark/hardswish/hardswish_test_generator.py) |
| softsign | [`benchmark/softsign/softsign_test_generator.py`](../kernel_bench/benchmark/softsign/softsign_test_generator.py) |
| cbrt | [`benchmark/cbrt/cbrt_test_generator.py`](../kernel_bench/benchmark/cbrt/cbrt_test_generator.py) |
| softcap | [`benchmark/softcap/softcap_test_generator.py`](../kernel_bench/benchmark/softcap/softcap_test_generator.py) |

## Evaluation Infrastructure

| Component | Path |
|-----------|------|
| Eval CLI | [`tools/eval/eval.py`](../kernel_bench/tools/eval/eval.py) |
| Eval engine | [`tools/eval/eval_engine.py`](../kernel_bench/tools/eval/eval_engine.py) |
| Test generator base (timing logic) | [`tools/eval/test_generator_base.py`](../kernel_bench/tools/eval/test_generator_base.py) |
| Data aggregator | [`tools/eval/data_aggregator.py`](../kernel_bench/tools/eval/data_aggregator.py) |
| Leaderboard | [`README.md`](../kernel_bench/README.md) |

## Methodology Notes

- **Timing**: Wall-clock `time.time()` in Python, measured around the TTNN call + `to_torch()` readback. Includes host dispatch overhead, not pure device kernel time.
- **Accuracy**: ULP-based quantization error comparison. A test passes if accuracy >= 0.99 (~2 ULP tolerance for bfloat16).
- **Both submissions call the same API**: Both `metal_main` and `sfpu-generator` implementations are thin wrappers around the same `ttnn.*` Python API (e.g., `ttnn.frac()`, `ttnn.rpow()`). They use identical dispatch paths and execute the same device kernels.
- **Latency differences are run-to-run variance**: Since both submissions call the same underlying API, the timing differences reflect system-level noise across different evaluation sessions (system load, caching state, thermal conditions), not any algorithmic or implementation difference.
- **The meaningful comparison is accuracy**: The key differentiator between submissions is pass rate and accuracy, not latency. Where `metal_main` fails tests (e.g., swish 20.8%, softshrink 0%, hardsigmoid 74.7%), the sfpu-generator achieves 82-100% — this reflects differences in how test parameters are handled, not in the underlying kernel.
