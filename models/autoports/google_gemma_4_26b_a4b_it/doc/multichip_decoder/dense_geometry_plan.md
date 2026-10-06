# Current-profile dense geometry candidates

Unapplied runner patch `dense_geometry_runner.patch` prepares one-role trials;
`dense_geometry_provenance.json` records its base and syntax-only validation.
It does not modify the runtime defaults or optimized TP1 baseline. Run after
expert replay AutoFix; reject combinations with DRAM or fused AGMM backends.
Weights, fidelity, FP32 accumulation, inputs, collective policy and prefill are
unchanged. Candidate configs keep M1 and use N2/N4 output blocks/subblocks.

| Role | N tiles | Baseline grid / N per worker | N2 grid | N4 grid | K block |
|---|---:|---|---|---|---:|
| Sliding QKV | 64 | 8x8 /1 | 8x4 | 8x2 | 22 |
| Full QKV | 96 | 8x6 /2 | already baseline | 8x3 | 22 |
| WO, either kind | 88 | 11x8 /1 | 11x4 | 11x2 | 8 |
| Router, sliding/full | 4 | 4x1 /1 | 2x1 | 1x1 | 22 /44 |

All N values exactly divide their output width and use rectangular active grids;
all K blocks divide the current tiled input width (88 QKV/router,32 slidingWO,
64 fullWO). FP32 output subblocks1x4 are within the four-tile FP32 limit. Fewer
workers may lose bandwidth despite larger subblocks, so compare warmed complete
layer latency and all128 PCC/replay/cache checks. Router retains its BF16 weights
and HiFi4; the report's generic BFP8 recommendation does not describe that op.

The optional `--sharded-decode-rope` candidate requires the separate runtime
patch described in `sharded_rope_plan.md`. It should first be tested independently
from dense geometry. No result or improvement is claimed by these plans.

`combined_geometry_runner.patch` combines these options with the sparse geometry
trials and rejects incompatible DRAM/fused backends. Its candidate is syntax-
and Black-checked only; the original runner and runtime are unchanged.
