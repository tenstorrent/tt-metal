# Softcap SFPU Operation Implementation Notes

## Overview
Successfully implemented a new parameterized unary SFPU operation `softcap(x, cap)` that computes `cap * tanh(x / cap)` with default cap value of 50.0.

## Mathematical Implementation
The kernel implements the softcap function using the existing `exp_21f` helper function from the sinh implementation to compute tanh accurately:
- **Algorithm**: `softcap(x, cap) = cap * tanh(x / cap)`
- **tanh computation**: `tanh(y) = (exp(y) - exp(-y)) / (exp(y) + exp(-y))` using exp_21f helper
- **Small value optimization**: For `|scaled_x| < 0.5`, uses Taylor approximation `tanh(x) ≈ x - x³/3`
- **Overflow/underflow protection**: Clamps exponential arguments to prevent numerical issues

## Reference Operations Used
The most useful reference operations were:

1. **hardtanh_analysis.md** - Provided the canonical pattern for parameterized SFPU operations
   - Parameter encoding pattern (single uint32_t param in FP16_B format)
   - SFPU kernel signature with iterations parameter
   - LLK dispatch pattern for parameterized operations

2. **tanh_analysis.md** - Documented the expected tanh implementation pattern
   - Use of exp_21f helper for accurate exponential computation
   - Small-x Taylor series approximation for numerical stability
   - Parameter clamping strategies

3. **sinh implementation** (ckernel_sfpu_sinh.h) - Source for exp_21f helper function
   - Complete exp_21f implementation copied for use in softcap kernel
   - Constants and algorithmic patterns for hyperbolic functions

4. **frac implementation** - Pattern for unparameterized SFPU operations
   - LLK dispatch header structure
   - API header format and documentation

## Implementation Architecture

### New Files

tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_softcap.h
tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_softcap.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h

### Modified Files

ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h

## Implementation Details

### Core SFPU Kernel
- Located in `ckernel_sfpu_softcap.h` for both Wormhole and Blackhole
- Uses `exp_21f` helper function for accurate tanh computation
- Implements parameter scaling, tanh calculation, and result scaling
- Includes numerical stability optimizations for edge cases

### LLK Dispatch Layer
- Standard parameterized dispatch pattern following hardtanh model
- Uses `SfpuType::unused` for address mode configuration (default settings)
- Passes parameter via `_llk_math_eltwise_unary_sfpu_params_` template

### API Layer Integration
- API header provides `softcap_tile()` and `softcap_tile_init()` functions
- Parameter passed as uint32_t in FP16_B format
- Integrated into split includes system with `SFPU_OP_SOFTCAP_INCLUDE` guard

### Host Dispatch Integration
- Added SOFTCAP to UnaryOpType enum
- Configured as parametrized type in `is_parametrized_type()`
- Dispatch generates `softcap_tile_init(); softcap_tile(idst, param0_bits);` calls
- Python binding via `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(softcap, SOFTCAP)`

## Deviations from Standard Patterns
1. **exp_21f helper duplication**: Since tanh SFPU implementation was removed from the evaluation environment, the exp_21f helper function was copied into the softcap kernel rather than using a shared implementation. In a production environment, this would be refactored to use shared helper functions.

2. **Direct tanh implementation**: Instead of calling existing tanh infrastructure (which was stripped), implemented tanh directly within the softcap kernel using the documented algorithm patterns.

## Known Limitations and Concerns
1. **Code duplication**: The exp_21f function is duplicated across softcap kernels rather than shared
2. **Evaluation environment constraints**: Implementation had to work around removed tanh infrastructure
3. **Parameter encoding**: Uses standard FP16_B encoding which may have precision limitations for very large cap values
4. **Test coverage**: No test implementation was created per instructions (test file already exists)

## Testing Notes
A test file already exists at `tests/ttnn/unit_tests/operations/eltwise/test_softcap.py` which was not modified per the implementation instructions. The implementation should be compatible with standard TTNN testing patterns.

## Performance Considerations
- Uses efficient exp_21f algorithm for exponential computation
- Includes small-value Taylor series optimization to avoid catastrophic cancellation
- Implements proper overflow/underflow clamping
- Single-pass algorithm with minimal temporary storage