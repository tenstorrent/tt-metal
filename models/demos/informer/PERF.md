# Model performance and accuracy

Performance and accuracy numbers for Informer are collected from direct pytest runs in `models/demos/informer/tests/perf`.

## Environment
- Device: Wormhole `n300`
- Firmware: `19.3.1`
- KMD: `2.5.0`
- Date: `2026-02-26`

## Benchmark command
```bash
pytest models/demos/informer/tests/perf/test_perf.py::TestInformerPerformance::test_benchmark_summary -v -s
```

## Summary metrics

| Metric | Value |
|---|---:|
| Throughput  | `1584.9 seq/s` |
| Latency  | `4.01 ms` |
| Correlation (TTNN vs torch ref) | `0.9964` |
| MSE (TTNN vs torch ref) | `0.000106` |
| MAE (TTNN vs torch ref) | `0.008176` |

## Perf coverage
Source suite: `tests/perf/test_perf.py::TestInformerPerformance`
- Throughput/latency sweep
- Sequence scaling
- Trace replay stability
- TTNN vs torch-reference parity
- Summary metrics report

## Advanced capability coverage
Source suite: `tests/perf/test_perf.py::TestInformerAdvancedCapabilities`
- ProbSparse query reduction
- Decode cache path parity
- Trace replay parity vs eager
- Streaming decoder parity
- Long-sequence support
- High-dimensional multivariate input support
- Multi-horizon consistency

## Real-data and HF parity coverage
Source suite: `tests/perf/test_accuracy_hf.py::TestInformerAccuracyHF`
- ETTh1 real-data metrics against ground truth
- HF checkpoint compatibility and parity checks
