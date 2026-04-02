# Performance Targets — Granite TTM-R1

| Metric              | Target      | Achieved | Notes |
|---------------------|-------------|----------|-------|
| Throughput          | ≥ 500 seq/s | ~40 seq/s (xfail) | Stage 1: patching/encoder/decoder via TorchModuleFallback (CPU) |
| Latency (batch=1)   | < 10 ms     | ~25 ms (xfail) | Stage 1: same as above |
| Model parameters    | < 1M        | 805,280 ✅ | |
| Model size on disk  | < 10 MB     | 3.07 MB ✅ | float32 weights |
| PCC vs PyTorch      | ≥ 0.99      | ≥ 0.99 ✅ | 7/7 components pass on Wormhole N300s |
| MSE vs PyTorch      | within 5%   | within 5% ✅ | validated by PCC tests |
| Zero-shot vs published | within 10% | pending | requires ETTh1 dataset (see scripts/prepare_assets.py) |
