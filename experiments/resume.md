# Resume State

**Last updated:** 2026-03-11
**Task:** GLM-4.7-Flash modularization and optimization
**Current phase:** Phase 4 (Optimize) — full model profiled, baseline established

## Current State

- **Phase 1 (Understand):** Complete.
- **Phase 2 (Profile):** Complete. Full model profiled on 4x Wormhole.
- **Phase 3 (Modularize):** Complete. All modules extracted, hardware validated.
- **Phase 4 (Optimize):** In progress. Full model baseline: **1.98 tok/s**, 504.6 ms/token.
- **Code state:** Refactored and validated on hardware.
- **Venv/build:** Done and working.

## Full Model Performance Baseline

| Metric | Value |
|--------|-------|
| Decode throughput | 1.98 tok/s |
| Decode latency | 504.6 ms/token |
| Device kernel time | 44.2 ms/device |
| Host dispatch overhead | 91.2% of latency |
| Devices | 4x Wormhole (mesh 1,4) |

## What Has Been Done

| # | Step | Status |
|---|------|--------|
| 1 | Architecture analysis and op mapping | Done |
| 2 | Set up agentic workflow | Done |
| 3-8 | Extract 6 modules (config, linear, attn, mlp, mtp, trace) | Done |
| 9 | Wire modules into decoder_layer_tt.py (2113→1098 lines) | Done |
| 10 | Hardware validation (layer 0 + MoE) | Done |
| 11 | Single-layer Tracy profile | Done |
| 12 | Full model 4-device profile | Done |
| 13 | Per-op CSV and detailed report | Done |

## Per-Op Profile Summary (full model, device 0)

| Op | Count | Kernel (us) | % |
|---|---|---|---|
| MatmulDeviceOperation | 206 | 12,754 | 28.8 |
| FillPadDeviceOperation | 75 | 4,662 | 10.5 |
| TilizeDeviceOperation | 21 | 4,376 | 9.9 |
| BinaryNgDeviceOperation | 242 | 2,702 | 6.1 |
| SparseMatmulDeviceOperation | 62 | 2,521 | 5.7 |
| PermuteDeviceOperation | 24 | 2,174 | 4.9 |
| RepeatDeviceOperation | 48 | 1,975 | 4.5 |
| Other | 1,192 | 13,061 | 29.5 |
| **Total** | **1,870** | **44,225** | **100** |

## Next Steps Queue (Phase 4: Optimize)

| # | Step | Status | Rationale |
|---|------|--------|-----------|
| 1 | Enable metal trace capture/replay | **NEXT** | Host dispatch is 91.2% of latency — this is the single biggest win |
| 2 | Reduce data movement ops | pending | FillPad+Permute+Repeat+Clone+Transpose = 24.5% of kernel time |
| 3 | Eliminate Tilize overhead | pending | 9.9% of kernel time — pre-tilize or maintain tile layout |
| 4 | Optimize MoE kernel fusion | pending | SparseMatmul+Remap = 8.6% — fuse to reduce dispatch |
| 5 | DRAM-sharded matmuls for small projections | pending | Attention projections using few cores |

## Output Files

- `experiments/glm4_full_model_profile_report.md` — detailed analysis with bottlenecks and recommendations
- `experiments/glm4_full_model_ops_profile.csv` — raw per-op data (7,496 rows, all devices)
- `experiments/glm4_full_model_ops_summary.csv` — per-op summary by device
- `experiments/baseline.yaml` — all baseline numbers
