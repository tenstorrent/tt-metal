# Implementation Notes: selu

## Math Definition
SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))

Where:
- scale = 1.0507009873554804934193349852946
- alpha = 1.6732632423543772848170429916717

These are fixed constants (not user-configurable parameters).

Equivalently:
- For x >= 0: scale * x
- For x < 0: scale * alpha * (exp(x) - 1)

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h
tt_metal/hw/inc/api/compute/eltwise_unary/selu.h

### Modified Files
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
ttnn/ttnn/operations/unary.py
tt_metal/hw/sources.cmake

## Design Decisions

### Reference Operations Used
- **ELU** (most useful): ELU is the closest reference because SELU's negative branch is `scale * alpha * (exp(x) - 1)` which is structurally identical to ELU's `alpha * (exp(x) - 1)`. The SFPU kernel follows the same pattern: `_calculate_exponential_piecewise_` for the exp computation, conditional on `v < 0.0f`.
- **CELU**: Confirmed the `_calculate_exponential_piecewise_` / `_init_exponential_` pattern and the `Converter::as_float` usage for passing constants.
- **PReLU/SFPU**: Showed the raw TTI instruction pattern (not used, but confirmed the SFPI abstraction approach is preferred for new operations).
- **expm1**: Showed the full wiring pattern for compute API headers, LLK dispatch, and split includes.

### Key Design Choices
1. **No-parameter operation**: SELU uses fixed constants (scale, alpha) that are mathematically defined and never change. These are baked into the SFPU kernel as FP32 hex constants rather than passed as runtime parameters. This simplifies the dispatch chain (no `is_parametrized_type`, no parameter packing).

2. **Single conditional + unconditional multiply**: Instead of two v_if blocks (one for negative, one for positive), the kernel uses:
   - `v_if(v < 0.0f)`: Replace v with `alpha * (exp(v) - 1)` for negative lanes
   - `v_endif`
   - Unconditionally: `v = scale * v` for ALL lanes
   This is more efficient (one fewer conditional check) and correct because:
   - Positive lanes: `scale * original_x`
   - Negative lanes: `scale * alpha * (exp(x) - 1)`

3. **FP32 hex constants**: `alpha = 0x3FD63840` (~1.6733 in FP32), `scale = 0x3F868640` (~1.0507 in FP32). These are the closest FP32 representations of the exact constants. Using `Converter::as_float()` for consistent bitcasting.

4. **Custom init callback**: `selu_init` delegates to `_init_exponential_<APPROXIMATION_MODE, false, 0x3F800000>()` to set up reciprocal polynomial coefficients needed by the non-approximate exponential path. This follows the ELU pattern exactly.

5. **New SfpuType entry**: Added `SfpuType::selu` since this is a brand new LLK dispatch (not part of an existing family).

### Deviations from Standard Pattern
- None. The implementation follows the exact same pattern as ELU and other no-parameter unary SFPU operations.

## Known Limitations
- **Precision**: The scale and alpha constants are stored as FP32, which provides ~7 decimal digits of precision. The exact mathematical constants have ~31 digits, so there is inherent rounding.
- **Approximation mode**: When `APPROXIMATION_MODE=true`, the exponential uses a faster but less precise bit-manipulation approximation. This may reduce accuracy for the negative branch.
- **Input range**: For very large negative inputs (x < ~-88), exp(x) underflows to 0 in FP32, so SELU(x) ≈ scale * alpha * (-1) ≈ -1.7581. This is the correct mathematical limit.
- **bfloat16**: The operation processes data as stored in DEST (typically FP32 during computation). The bfloat16 truncation happens at pack time, which may introduce additional rounding in the output.
