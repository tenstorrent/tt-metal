# AI-JOURNAL: brain/revert-44412-llk-scalar-bcast-precision

## Context
Sanity CI regression detected 2026-05-30 morning (UTC).
Failing: `ttnn-unit-tests / ttnn reduce group` → `test_var_fp32_doscale_wt_gt_1[Wt2-scalar=2.0]`
ATOL ~419, RTOL ~11.9 — large magnitude error, not a small precision issue.

## Root Cause
PR #44412 (`d7a3414`) merged 2026-05-30T05:32 UTC by markoradosavljevicTT.

The PR rewrote FP32 32-bit SCALAR broadcast in `llk_math_eltwise_unary_datacopy.h`:
- Removed dependency on bit 11 (RISCV_DEBUG_REG_DBG_FEATURE_DISABLE)  
- Replaced separate B-bank hi16/lo16 slots with ALU_FORMAT_SPEC_REG_SrcA_val toggling (Tf32/Float32)
- Changed TTI_MOVD2B/TTI_MOVB2D (fixed dest row 0) → TT_MOVD2B/TT_MOVB2D (dynamic tile_base = dst_index * 64)

## Suspected Bug
For SCALAR broadcast, the scalar value is placed at dest row 0 for the single-element input.
New code reads scalar from `dest[tile_base]` = `dest[dst_index * 64]`.
For Wt2, when `dst_index=1`, reads from row 64 which may not hold the scalar value.

Alternatively, the ALU_FORMAT_SPEC_REG_SrcA_val toggle (Tf32/Float32) between B2D calls
may need additional pipeline sync.

## This Branch
Simple revert of d7a34140 to restore sanity. The underlying bit-11 state-leak fix from 
#44412 is still needed; author should re-investigate and submit a corrected version.
