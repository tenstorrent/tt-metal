# Implementation Notes: frac

## Math Definition
frac(x) = x - trunc(x) -- the fractional part of x, preserving sign (matches torch.frac).

## SFPU Kernel Algorithm
The kernel implements frac using IEEE 754 bit manipulation:

1. Extract the debiased exponent E via `sfpi::exexp(v)`
2. Three cases:
   - **E >= 23**: The float has no fractional bits, result = 0.
   - **E < 0**: |x| < 1, so trunc(x) = 0, result = x (entire value is fractional).
   - **0 <= E < 23**: Build a bitmask to zero out fractional mantissa bits (trunc),
     compute result = x - trunc(x).

3. The mask is computed as: `0xFFFFFFFF << (23 - E)` which zeroes the lower (23-E) mantissa bits.
   This is applied to the raw IEEE 754 bits via `reinterpret<vInt>`, yielding trunc(x).

## Design Decisions
- **torch.frac vs x-floor(x)**: The initial implementation used `x - floor(x)` (always non-negative).
  However, the existing ttnn golden function uses `torch.frac()` which implements `x - trunc(x)`
  (preserves sign). Changed to match the golden function semantics.
- **No special init needed**: The kernel uses only basic SFPI instructions (exexp, reinterpret, shft, arithmetic),
  so no init callback is required. The LLK init simply calls the default `llk_math_eltwise_unary_sfpu_init`.

## Reference Operations Used
- **hardswish**: Primary pattern reference (simple no-param unary, same file structure)
- **softsign**: Secondary pattern reference (LLK dispatch pattern)
- **cbrt**: SFPI bit manipulation patterns (reinterpret, shift, exponent extraction)

## Test Results
- **5/5 tests PASS** (PCC=0.999 against torch.frac golden)
- Tests cover: multiple shapes (32x32, 320x384, 3x320x384), property verification (|frac| < 1), integer inputs (frac=0)
- Iteration 1: 3 pass, 1 fail (test checked non-negativity assuming x-floor(x) semantics)
- Iteration 2: 5 pass (fixed kernel to torch.frac semantics + fixed test assertions)

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

## Pre-existing Registrations (no changes needed)
- `UnaryOpType::FRAC` already in enum (unary_op_types.hpp line 103)
- `REGISTER_UNARY_OPERATION(frac, FRAC)` already in unary.hpp line 154
- `bind_unary_operation<"frac", &ttnn::frac>` already in unary_nanobind.cpp

## Known Limitations
- The SFPI `shft` operation with vector shift amount may have performance implications
- Nested v_if (2 levels) has moderate SFPU throughput impact
- Only tested with bfloat16 dtype; fp32 path not explicitly tested
