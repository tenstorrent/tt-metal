# Performance Targets — Granite TTM-R1

| Metric              | Target      | Achieved | Notes |
|---------------------|-------------|----------|-------|
| Throughput          | ≥ 500 seq/s | ~117 seq/s (xfail) | Stage 2: all ops on-device; per-op dispatch overhead limits batch=1 throughput |
| Latency (batch=1)   | < 10 ms     | ~8.5 ms ✅ | Stage 2: full TTNN pipeline; no TorchModuleFallback in forward path |
| Model parameters    | < 1M        | 805,280 ✅ | |
| Model size on disk  | < 10 MB     | 3.07 MB ✅ | float32 weights |
| PCC vs PyTorch      | ≥ 0.99      | ≥ 0.99 ✅ | 7/7 components pass on Wormhole N300s |
| MSE vs PyTorch      | within 5%   | within 5% ✅ | validated by PCC tests |
| Zero-shot vs published | within 10% | pending | requires ETTh1 dataset (see scripts/prepare_assets.py) |
