# Performance Targets — Granite TTM-R1

| Metric              | Target      | Achieved | Notes |
|---------------------|-------------|----------|-------|
| Throughput          | ≥ 500 seq/s | —        | batch=1, context=512 |
| Latency (batch=1)   | < 10 ms     | —        | single forward pass |
| Model parameters    | < 1M        | —        | |
| Model size on disk  | < 10 MB     | —        | |
| PCC vs PyTorch      | ≥ 0.99      | —        | per-component and e2e |
| MSE vs PyTorch      | within 5%   | —        | |
| Zero-shot vs published | within 10% | —     | ETTh1 |
