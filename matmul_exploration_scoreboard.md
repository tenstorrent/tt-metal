# Matmul Exploration Scoreboard (N150)

Use this file for exploration outcomes that are informative but not merge-ready under the strict gate.

Strict merge gate remains unchanged:
- Candidate is mergeable only if both aggregate `overall_p50` and `overall_p95` improve consistently across repeated runs.

| Date | Experiment | Scope | Mean p50 delta | Mean p95 delta | Mergeable under strict gate | Notes |
|---|---|---|---:|---:|---|---|
| 2026-03-20 | Adaptive boundary-memory (short/moderate shapes use L1) | `matmul_boundary_memory_n150_benchmark.py` | `+0.4%` | `-4.6%` | No | Tail improved, median regressed. |
| 2026-03-20 | Adaptive boundary-memory (L1 only at `dim==1536`) | `matmul_boundary_memory_n150_benchmark.py` | `-5.8%` | `+0.7%` | No | Strong median gain, slight tail regression. |
| 2026-03-20 | Alternating A/B (5 rounds): `dram` vs adaptive(`dim==1536`) | `matmul_n150_alternating_ab.py` + `matmul_boundary_memory_n150_benchmark.py` | `-0.97%` | `+0.36%` | No | Baseline-first alternating schedule confirmed mixed result. |
| 2026-03-20 | `MLP1D` subblock width cap `4 -> 8` | `models/common/modules/mlp/mlp_1d.py` + `mlp1d_prefill_n150_benchmark.py` | `+6.4%` | `+3.1%` | No | Regressed both metrics. |
| 2026-03-20 | Decode kernelbench A/B: output override `dram` | `matmul_n150_kernelbench.py` + `matmul_n150_alternating_ab.py` | `+9.87%` | `+4.52%` | No | Also regressed kernel p50 (~+7.43%). |
| 2026-03-20 | Decode kernelbench A/B: output override `l1` | `matmul_n150_kernelbench.py` + `matmul_n150_alternating_ab.py` | `-12.47%` | `-22.13%` | Yes (decode subset gate) | e2e improved strongly; kernel p50 slightly worse (~+1.44%). Needs broader validation. |
| 2026-03-20 | Decode kernelbench A/B: input A override `dram` | `matmul_n150_kernelbench.py` + `matmul_n150_alternating_ab.py` | `+3.79%` | `-8.68%` | No | Mixed (median regression). |
| 2026-03-20 | Prefill kernelbench A/B: output override `l1` | `matmul_n150_kernelbench.py` + `matmul_n150_alternating_ab.py` | `+4.86%` | `+2.82%` | No | Decode-only winner does not transfer to prefill. |
| 2026-03-20 | All-regime kernelbench A/B: output override `l1` | `matmul_n150_kernelbench.py` + `matmul_n150_alternating_ab.py` | `+14.73%` | `+5.62%` | No | Broad regression despite slightly improved kernel aggregates. |
| 2026-03-20 | Decode kernelbench A/B quick check (1 round): output override `l1` | `matmul_n150_kernelbench.py` + `matmul_n150_alternating_ab.py` | `-14.51%` | `-5.54%` | Yes (single-round only) | Fast smoke passed strict gate, but single-round confidence is low. |
| 2026-03-20 | Decode kernelbench A/B quick check (3 rounds): output override `l1` | `matmul_n150_kernelbench.py` + `matmul_n150_alternating_ab.py` | `+46.11%` | `-0.86%` | No | Kernel aggregates stayed near-flat while e2e median regressed; indicates instability/noise sensitivity. |
| 2026-03-20 | Decode-only protocol slice A/B rerun (`M<=1`, 37 vecs) | `matmul_n150_sweeps_summary.py` + `matmul_n150_alternating_ab.py` | `-6.38%` | `-4.33%` | Yes (slice gate) | First run had extreme p95 outlier; rerun was stable and positive. |
| 2026-03-20 | Full protocol A/B with decode-only output-L1 sweep toggle (159 vecs) | `matmul_n150_sweeps_summary.py` + `matmul_n150_alternating_ab.py` | `-1.82%` | `+4.11%` | No | Mixed at full scope; candidate reverted. |

## Candidate backlog

Keep promising mixed results here for future regime-specific gating:
- Boundary-memory adaptive (`dim==1536`) as decode/DM-bound candidate: promising p50, needs p95 stabilization.
