# OLMo-3.1-32B ISL Demo Perf Results

Run date: 2026-04-26
All 8 tests: PASSED

| Test | ISL (tokens) | Batch | TTFT (ms) | Decode tok/s/user | Throughput (tok/s) | Batched Prefill |
|------|-------------|-------|-----------|-------------------|--------------------|-----------------|
| batch-32     |    128 | 32 |   1189.30 | 30.86 |  987.55 | YES (9.9x TTFT speedup) |
| batch-1      |    167 |  1 |    365.88 | 30.75 |   30.75 | - |
| long-4k-b32  |   4096 | 32 |  38837.93 | 26.65 |  852.78 | no (sequential) |
| long-4k-b1   |   4000 |  1 |   1245.84 | 28.66 |   28.66 | - |
| long-8k-b1   |   8160 |  1 |   3659.23 | 28.38 |   28.38 | - |
| long-16k-b1  |  15938 |  1 |   6773.75 | 27.79 |   27.79 | - |
| long-32k-b1  |  32694 |  1 |  14955.23 | 26.71 |   26.71 | - |
| long-64k-b1  |  65168 |  1 |  37574.90 | 24.78 |   24.78 | - |

## Batched Prefill Notes
- Batched prefill is active for ISL=128, batch=32: all 32 users prefilled in one forward pass
- TTFT improvement: 11765ms (sequential) → 1189ms (batched) = **9.9x speedup**
- ISL=128/B32 uses a pre-compiled batched trace.
- ISL=4096/B32 uses sequential prefill. The 2×B16 traced experiment passed but did not improve TTFT, so it was removed.
- 4k/b32 KV cache is restored to 4096 blocks.
