# Resume State

**Last updated:** 2026-03-10
**Task:** GLM-4.7-Flash modularization and optimization
**Current phase:** Phase 2 (Profile Baseline)

## Current State

- **Phase 1 (Understand):** Complete. Architecture documented in status.md.
- **Phase 2 (Profile):** Partial. Tracy profiled dense layers (1316 ops, 54.3 ms). Full model OOM on MoE expert weights.
- **Code state:** Unmodified (no refactoring started yet)
- **Venv/build:** Done and working

## What Has Been Done

| # | Step | Status |
|---|------|--------|
| 1 | Read and document model architecture | Done |
| 2 | Create op-mapping table | Done |
| 3 | Identify modularity problems | Done |
| 4 | Set up agentic workflow files | Done |

## Next Steps Queue

| # | Step | Status | Rationale |
|---|------|--------|-----------|
| 1 | Build venv and validate existing tests pass | Done | |
| 2 | Profile dense layers with Tracy | Done | 1316 ops, 54.3 ms. Full model OOM on MoE experts. |
| 3 | Record baselines in baseline.yaml | Done | Partial — dense layer baselines recorded |
| 4 | Extract runtime_config.py | **NEXT** | First refactoring step (lowest risk) |
| 6 | Extract linear_helpers.py | pending | Prerequisite for attention/mlp |
| 7 | Extract attention/ package | pending | Core modularity improvement |
| 8 | Extract mlp/ package | pending | Core modularity improvement |
| 9 | Simplify decoder_layer_tt.py | pending | Assembly of extracted modules |
| 10 | Split model_tt.py | pending | Final modularity step |

## Before Next Experiment

1. Build venv: `cd /home/ubuntu/agent/agentic/tt-metal && ./create_venv.sh`
2. Activate: `source python_env/bin/activate`
3. Run smoke test to verify environment works
4. Then proceed to profiling

## Key Files (for refactoring)

- `models/demos/glm4_moe_lite/tt/decoder_layer_tt.py` (2113 lines -> target ~150)
- `models/demos/glm4_moe_lite/tt/model_tt.py` (2685 lines -> target ~500)
- `models/demos/glm4_moe_lite/tt/moe_tt.py` (1768 lines -> split into router + experts)
