# Matmul Exploration Scoreboard (N150)

Use this file for exploration outcomes that are informative but not merge-ready under the strict gate.

Strict merge gate remains unchanged:
- Candidate is mergeable only if both aggregate `overall_p50` and `overall_p95` improve consistently across repeated runs.

| Date | Experiment | Scope | Mean p50 delta | Mean p95 delta | Mergeable under strict gate | Notes |
|---|---|---|---:|---:|---|---|
| 2026-03-20 | Adaptive boundary-memory (short/moderate shapes use L1) | `matmul_boundary_memory_n150_benchmark.py` | `+0.4%` | `-4.6%` | No | Tail improved, median regressed. |
| 2026-03-20 | Adaptive boundary-memory (L1 only at `dim==1536`) | `matmul_boundary_memory_n150_benchmark.py` | `-5.8%` | `+0.7%` | No | Strong median gain, slight tail regression. |
| 2026-03-20 | `MLP1D` subblock width cap `4 -> 8` | `models/common/modules/mlp/mlp_1d.py` + `mlp1d_prefill_n150_benchmark.py` | `+6.4%` | `+3.1%` | No | Regressed both metrics. |

## Candidate backlog

Keep promising mixed results here for future regime-specific gating:
- Boundary-memory adaptive (`dim==1536`) as decode/DM-bound candidate: promising p50, needs p95 stabilization.
