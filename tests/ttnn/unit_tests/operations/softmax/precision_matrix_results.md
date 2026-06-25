# Softmax Precision Matrix Results

**Date**: 2026-06-25
**Refinement**: 1 — Numerical configurability (dtypes + fp32-dest-only policy)
**Test file**: `test_softmax_precision_matrix.py`
**Total cases**: 192 (8 shapes × 3 dtypes × 4 fidelities × 2 distributions)
**Pass rate**: 192/192 (100%)

## Configuration

- **Dtypes**: float32, bfloat16, bfloat8_b
- **Math fidelities**: HiFi4, HiFi3, HiFi2, LoFi
- **fp32_dest_acc_en**: True (False is rejected by EXCLUSIONS — fp32-dest-only op)
- **Distributions**: uniform (rand), normal (randn)
- **PCC threshold**: 0.99 for all dtypes (per skill §11: "Precision matrix (all fidelities + fp32 acc)" threshold)

## Key findings

### bf16 (bfloat16)
- HiFi4/HiFi3: PCC ≥ 0.9999 on all shapes
- HiFi2: PCC ≥ 0.9995 on all shapes
- LoFi: PCC ≥ 0.9988 on all shapes (expected — LoFi trades precision for throughput)
- No NaN/Inf in any configuration

### bf8b (bfloat8_b)
- HiFi4/HiFi3: PCC ≥ 0.9998 on all shapes
- HiFi2: PCC ≥ 0.9990 on all shapes
- LoFi: PCC ≥ 0.9985 on all shapes
- Block-float precision is inherently lower but within 0.99 threshold

### fp32 (float32)
- HiFi4/HiFi3: PCC ≥ 0.9999 on all shapes
- HiFi2: PCC ≥ 0.9996 on all shapes
- LoFi: PCC ≥ 0.9989 on all shapes

## Implementation notes

- Zero compute kernel changes — helpers handle data-format reconfig automatically
- Intermediate CBs (cb_max, cb_exp, cb_recip_sum) set to Float32 format
  (accumulator precision preserved across phase boundaries)
- No UnpackToDestFp32 tagging (all intermediates feed FPU ops, not copy_tile-only)
- fp32_dest_acc_en=False rejected for all dtypes via EXCLUSIONS (fp32-dest-only policy)

## Skipped combinations

- `fp32_dest_acc_en=False` for all dtypes: rejected by EXCLUSIONS (op is fp32-dest-only)
