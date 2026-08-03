# Advisor-challenger work log

- Stage checkpoint: `83198728053`
- Independent stage-review verdict: clean-pass
- Strict gate: passed
- Optimized-decoder suite: 21 passed, 1 skipped (native layer-4 checkpoint shards unavailable)
- Shipping decision: 32-core L1 width-sharded sparse-MoE RMSNorm
- Full-model device estimate: 24,949.218 → 22,397.946 ± 74.289 µs at decode batch 1

The follow-up artifact commit adds this log and force-includes the two compact
bounded profiler CSVs that the repository-wide `*.csv` ignore rule otherwise
omits. No raw profiler logs are retained.
