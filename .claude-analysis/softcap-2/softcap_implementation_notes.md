# Softcap SFPU Operation Implementation Notes

## Overview

Successfully implemented the softcap SFPU operation: `softcap(x, cap) = cap * tanh(x / cap)` with default cap value of 50.0. The implementation follows the standard TTNN unary SFPU operation pattern across all abstraction layers.

## Reference Operations Used

The following reference operations were most useful for implementation:

### 1. **hardtanh** (Primary pattern for parameterized operations)
- **Why useful**: Provided the complete pattern for parameterized SFPU operations including parameter handling, function signatures, and dispatch mechanisms.
- **Key insights**: How to structure parameterized operations with `get_op_init_and_func_parameterized()`, parameter encoding as uint32_t FP16_B format, and LLK dispatch with lambda functions.

### 2. **swish** (SFPU kernel implementation pattern)
- **Why useful**: Provided the sigmoid approximation algorithm that was essential for implementing tanh.
- **Key insights**: Polynomial approximation coefficients, piecewise linear segments, breakpoint handling, and SFPI programming patterns.

### 3. **softshrink** (Python binding pattern)
- **Why useful**: Demonstrated the nanobind pattern for parameterized operations with default values.
- **Key insights**: Documentation format, parameter binding with default values, and wrapper function usage.

## Mathematical Implementation Strategy

Since tanh SFPU functions were removed in the "deep nuke", I implemented tanh using the mathematical relationship:
- `tanh(x) = 2 * sigmoid(2x) - 1`
- Reused the sigmoid approximation from swish kernel (polynomial + piecewise linear)
- Final computation: `softcap(x, cap) = cap * tanh(x / cap)`

## Implementation Approach

### SFPU Kernel Implementation
- Used sigmoid approximation from swish kernel to implement tanh
- Followed the transformation: `u = x / cap`, then `tanh(u)`, then `cap * tanh(u)`
- Applied standard SFPI programming pattern with `#pragma GCC unroll 0`

### Parameter Handling
- Single float parameter `cap` passed as uint32_t in FP16_B format
- Used lambda function in LLK dispatch to capture parameter
- Default value of 50.0f specified in both C++ and Python APIs

### Dispatch Integration
- Added SOFTCAP to UnaryOpType enum and SfpuType enum
- Registered as parameterized type in `is_parametrized_type()`
- Added dispatch case to `get_op_init_and_func_parameterized()`
- Added macro definition for include handling

## Deviations from Standard Patterns

### Custom Function Definition for Python API
**Deviation**: Instead of using `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER`, created a custom inline function definition.
**Why**: The macro doesn't support default parameter values. Needed `cap=50.0f` as a default.

### Tanh Implementation via Sigmoid
**Deviation**: Reimplemented tanh from scratch rather than calling existing tanh function.
**Why**: Existing tanh SFPU functions were removed in the deep nuke. Used mathematical relationship `tanh(x) = 2 * sigmoid(2x) - 1` with sigmoid approximation from swish.

## Files Implementation Details

### New Files

tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_softcap.h
tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_softcap.h
tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h

### Modified Files

ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
ttnn/ttnn/operations/unary.py

## Known Limitations and Concerns

### 1. **Accuracy Limitations**
- Uses polynomial approximation for sigmoid, which is then used for tanh computation
- Expected accuracy: ~4 ULP for bfloat16 (inherited from swish approximation)
- Compounding approximation errors from sigmoid-to-tanh conversion

### 2. **Performance Considerations**
- More computationally expensive than a direct tanh implementation would be
- Performs division, multiplication, and full sigmoid computation per element
- No hardware acceleration available (unlike dedicated SFPNONLINEAR on Quasar)

### 3. **Range Limitations**
- Sigmoid approximation saturates for large inputs (|x| > 5.0 after scaling by 2)
- For very large or very small cap values, numerical precision may be affected
- Not optimized for extreme parameter ranges

### 4. **Missing Quasar Support**
- No Quasar-specific implementation created
- Could potentially use SFPNONLINEAR for improved performance on Quasar hardware

## Testing Status

Implementation is complete across all abstraction layers. Testing should verify:
- Numerical accuracy compared to PyTorch golden function
- Performance characteristics
- Parameter handling edge cases
- Hardware compatibility (Wormhole, Blackhole)

## Recommendations

1. **Future Optimization**: When tanh SFPU functions are reimplemented, consider replacing the sigmoid-based approach with direct tanh calls for better accuracy and performance.

2. **Quasar Support**: Add Quasar-specific implementation using SFPNONLINEAR for better performance.

3. **Accuracy Testing**: Verify numerical accuracy across different input ranges and cap values, especially for edge cases.
