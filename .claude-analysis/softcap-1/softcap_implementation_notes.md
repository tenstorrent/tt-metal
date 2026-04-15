# Softcap SFPU Operation Implementation Notes

## Overview
This document summarizes the implementation of the softcap unary SFPU operation with the mathematical definition:
```
softcap(x, cap) = cap * tanh(x / cap)
```

The implementation follows the standard 11-layer TTNN SFPU operation architecture, with softcap being a parameterized operation that takes a positive float `cap` parameter.

## Reference Operations Analysis

The following reference operations were most useful:

### SWISH
- **Why useful**: Complete, functioning SFPU operation showing the full dispatch chain from Python API to SFPU kernel
- **Key patterns learned**:
  - Standard SFPU abstraction layer structure
  - API header pattern with TRISC_MATH guards
  - LLK dispatch template usage
  - SFPU kernel implementation using SFPI abstractions

### SOFTSHRINK (Nuked Operation)
- **Why useful**: Showed what a parameterized operation looks like when partially deleted
- **Key patterns learned**:
  - How parameterized operations are registered (`is_parametrized_type`)
  - Parameter packing in program factory
  - Enum and registration stubs that remain after deletion

### HARDTANH (Incomplete Operation)
- **Why useful**: Showed partial implementation of parameterized operation
- **Key patterns learned**:
  - Parameter passing pattern in SFPU kernels (`uint32_t param0`)
  - Parameter conversion using `sfpi::s2vFloat16b()`

## Implementation Approach

### Core Design Decisions

1. **Parameter Handling**: Used `sfpi::s2vFloat16b(param0)` to convert the packed float parameter, following hardtanh pattern
2. **Math Implementation**: Leveraged existing SFPU `tanh()` function for maximum accuracy instead of implementing custom approximations
3. **Function Signature**: `softcap_tile(idst, param0)` with parameter as second argument following parameterized operation pattern

### Algorithm Implementation

The SFPU kernel implements the mathematical formula directly:
1. Convert packed parameter to `sfpi::vFloat cap`
2. Load input value `x` from `dst_reg[0]`
3. Compute `scaled_x = x / cap`
4. Compute `tanh_result = sfpi::tanh(scaled_x)` using SFPU built-in
5. Compute `result = cap * tanh_result`
6. Store result and advance to next DEST row

## Deviations from Standard Patterns

### None Identified
The implementation follows standard TTNN SFPU operation patterns exactly:
- Standard 11-layer abstraction (enum → registration → dispatch → API → LLK → kernel)
- Standard parameterized operation pattern (parameter packing/unpacking)
- Standard SFPU kernel structure (SFPI abstractions, iteration loop, `dst_reg` advancement)
- Standard address mode configuration (ADDR_MOD_7 with no auto-increment)

## Known Limitations and Concerns

### 1. Parameter Precision
- The `cap` parameter is converted through FP16_B format which may introduce small precision losses
- For most use cases with cap values in typical ranges (1.0-100.0), this should be negligible

### 2. Division by Zero
- No explicit handling for cap = 0, which would cause division by zero
- The API documentation specifies cap must be positive, relying on user compliance
- Hardware division by zero behavior would apply (typically produces infinity/NaN)

### 3. Tanh Function Dependency
- Implementation relies on the existing SFPU `tanh()` function
- If the built-in tanh has accuracy issues for certain ranges, softcap will inherit them
- No independent verification of tanh accuracy was performed

### 4. Range Limitations
- For very large |x/cap| values, tanh approaches ±1 and softcap approaches ±cap
- No explicit saturation handling, relying on tanh's built-in behavior

## Testing Attempts and Issues

### Test Run 1: Python Binding Missing
**Issue**: `ttnn.softcap` was not available in Python
**Root Cause**: Missing Python binding in `unary_nanobind.cpp`
**Fix Applied**: Added Python binding for softcap operation with proper documentation and default parameters

### Test Run 2: Kernel Header Include Error
**Issue**: Compilation failed with "trigonometry.h: No such file or directory"
**Root Cause**: Incorrect include in `eltwise_sfpu.cpp`
**Fix Applied**: Changed `#include "api/compute/eltwise_unary/trigonometry.h"` to `#include "api/compute/eltwise_unary/softcap.h"`

### Test Run 3: Additional Missing Headers
**Issue**: Compilation failed due to missing `ckernel_sfpu_mul_int32.h`
**Root Cause**: Unnecessary includes in compute kernel
**Fix Applied**: Removed problematic includes (`mul_int_sfpu.h`, `rpow.h`, `rdiv.h`, `fill.h`)

### Test Run 4: SFPU Type Compilation Errors
**Issue**: Multiple compilation errors related to SFPU type definitions
**Root Cause**: Potential compatibility issues with SFPU infrastructure
**Status**: Unresolved - compilation still fails

### Test Run 5: Segmentation Fault
**Issue**: Direct test of ttnn.softcap caused segmentation fault
**Root Cause**: Unknown - could be related to implementation issues or missing SFPU kernel components
**Status**: Unresolved

## Current Status: FAILED
The softcap implementation has critical issues that prevent testing:
1. ✅ Python binding registered
2. ✅ Basic header includes fixed
3. ❌ SFPU compilation errors
4. ❌ Runtime segmentation fault

## Next Steps Required
1. **Investigate SFPU compilation errors** - May need to examine SFPU type definitions and template specializations
2. **Review SFPU kernel implementation** - The ckernel_sfpu_softcap.h files may have implementation issues
3. **Check parameter handling** - Verify that parameter packing/unpacking is correct
4. **Validate against working operations** - Compare implementation against known working SFPU operations like swish

## Testing Recommendations (When Fixed)

1. **Parameter Validation**: Test with edge cases (very small cap, very large cap)
2. **Range Testing**: Test with various input ranges to verify tanh behavior
3. **Precision Testing**: Compare against PyTorch implementation across different cap values
4. **Performance Testing**: Verify no significant performance regression vs other SFPU ops

## New Files
tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h

## Modified Files
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
ttnn/ttnn/operations/unary.py
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
