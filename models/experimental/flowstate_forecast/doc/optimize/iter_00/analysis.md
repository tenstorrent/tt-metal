# Round 0 analysis: paper / accuracy (FlowState r1.1, chip 0)

Source: read-only analyst (Opus, low effort), working from perf_report.txt.gz, profile_zones.json, bench.json and b0_conv_route.json. Written up by the worker.

The device time is 374.9 ms (bench, traced). The reduced-layer profile estimates 375.9 ms (+0.3 %, within PLAN's 2 % limit).

## Device time by role (ms per forward)

| role | ms | notes |
|---|---|---|
| B projection (x6) | 63.6 | 17 % of the HiFi4 peak, 127 GB/s; default config |
| C projection (x5, plus a small last-layer op) | 67.9 | 13 %, 99 GB/s (fp32 in0); default config |
| route-K scan (x6) | 30.3 | uses 32 of 110 cores at Bp = 64 |
| gate (linear, sigmoid, mul) | 41.0 | linear at 10.5 % |
| D·u and add | 18.3 | |
| MLP (w1, silu, w2, mul, w3) | 121.0 | 12-19 %; silu runs as a separate op |
| norms | 14.1 | |
| residual adds | 18.1 | |
| embed and decoder | 1.7 | |

- Matmuls take about 266 ms (71 %). All of them run 2.5-4x above their bandwidth floor because they use the default program configs.
- The elementwise ops are already bandwidth-bound, so the only way to save that time is to fuse them away.

## Performance class

At paper it is no faster: 375.8 vs 374.9 ms. It changes only the gate and MLP weights and fidelity, and those matmuls are bound neither by fidelity nor by weight bandwidth.

## Suspicious items

1. `activation="silu"` without a program config runs as a separate unary op.
2. The performance-class LoFi/bfp8 settings have not been confirmed to reach the ops on a profile.
3. The profiler shows 4.8 ms of op-to-op gaps. These are an eager-profile artifact (the bench is traced).
4. The scan's core count scales with the batch: 16 cores at Bp = 32.
