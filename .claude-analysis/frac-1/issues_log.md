# Issues Log: frac

## Configuration
- **Operation**: frac
- **Math definition**: frac(x) = x - trunc(x) (torch.frac semantics)
- **Source**: direct formula (corrected from x - floor(x) to x - trunc(x) during testing)
- **Output folder**: `.claude-analysis/frac-1/`
- **Date**: 2026-04-05

## Phase Timeline
| Phase | Description | Status | Duration | Issues |
|-------|-------------|--------|----------|--------|
| 1 | Reference Discovery | ok | ~30s | None |
| 2 | Reference Analysis | ok | ~2 min | None |
| 3 | Implementation | ok | ~5 min | tt_llk submodule empty, created standalone kernel |
| 4 | Testing & Debugging | ok | ~5 min | Semantic mismatch (floor vs trunc), fixed in iteration 2 |
| 5 | Documentation | ok | ~1 min | None |
| 6 | Self-Reflection | ok | ~1 min | None |

## Issues

### Issue 1: Semantic mismatch between math definition and golden function
- **Phase**: 4 (Testing)
- **Severity**: Medium
- **Description**: The math definition specified `frac(x) = x - floor(x)` (always non-negative), but the golden function uses `torch.frac(x)` which implements `x - trunc(x)` (preserves sign). For negative inputs, these differ: `floor(-2.3) = -3` while `trunc(-2.3) = -2`.
- **Resolution**: Changed SFPU kernel to implement `x - trunc(x)` semantics. Updated test assertions accordingly.

### Issue 2: Empty tt_llk submodule
- **Phase**: 3 (Implementation)
- **Severity**: Low
- **Description**: The tt_llk submodule is empty, so `floor_tile()`, `rounding_op_tile_init()`, and `_calculate_frac_()` are not available.
- **Resolution**: Created a standalone SFPU kernel using raw SFPI bit manipulation (exexp, reinterpret, shft, bitwise AND) to compute trunc(x) from scratch.

## File Manifest

### New Files
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h
- tt_metal/hw/inc/api/compute/eltwise_unary/frac.h
- tests/ttnn/unit_tests/operations/eltwise/test_frac.py

### Modified Files
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
- ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp
- ttnn/ttnn/operations/unary.py
