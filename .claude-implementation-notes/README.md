# SFPU Operation Implementation Notes

This directory contains comprehensive implementation notes for SFPU (Specialized Floating-Point Unit) operations in the TT-Metal project. Each operation is documented with full source code snippets for all implementation layers.

## Available Operations

### 1. hardsigmoid
- **Math**: hardsigmoid(x) = max(0, min(1, x/6 + 0.5))
- **File**: `hardsigmoid_implementation_notes.md`
- **Lines**: 339
- **Layers**: 12 (SFPU kernels, LLK wrappers, Compute API, enums, dispatch, golden functions, tests, registration)

### 2. rpow
- **Math**: rpow(x, base) = base^x
- **File**: `rpow_implementation_notes.md`
- **Lines**: 418
- **Layers**: 12 (includes parameterized SFPU kernel, one-parameter operation)
- **Special**: Logarithmic decomposition for overflow/underflow handling

### 3. softsign
- **Math**: softsign(x) = x / (1 + |x|)
- **File**: `softsign_implementation_notes.md`
- **Lines**: 354
- **Layers**: 12 (includes reciprocal-based implementation)
- **Special**: Input clamping to handle subnormal values

### 4. hardswish
- **Math**: hardswish(x) = x * max(0, min(1, x/6 + 0.5)) = x * hardsigmoid(x)
- **File**: `hardswish_implementation_notes.md`
- **Lines**: 368
- **Layers**: 12 (includes vec_min_max instruction usage)
- **Special**: Exhaustive test coverage for all bfloat16 values

## Structure of Each Implementation Note

Each file follows a consistent structure:

1. **Math Definition**: Clear mathematical definition of the operation
2. **Files Created**: Detailed source code for all 12 implementation layers:
   - Layer 1: SFPU Kernel (Wormhole)
   - Layer 2: SFPU Kernel (Blackhole)
   - Layer 3: LLK Wrapper (Wormhole) or Compute API Header
   - Layer 4: LLK Wrapper (Blackhole)
   - Layer 5: Compute API Header or SfpuType enum entry
   - Layer 6-12: Enum entries, conditional includes, dispatch code, golden functions, tests, registration

3. **Design Decisions**: Architectural choices and implementation rationale
4. **Debug Log**: Testing approach and coverage
5. **Test Results**: Verification criteria and accuracy metrics
6. **Known Limitations**: Edge cases and constraints

## Key Implementation Patterns

### Architecture Duplication
Wormhole and Blackhole implementations are typically identical, following the pattern where architectures share the same mathematical implementation.

### Template Parameters
All SFPU kernels use template parameters:
- `APPROXIMATION_MODE`: Selects fast vs. accurate implementation
- `ITERATIONS`: Controls tile loop unrolling

### Dispatch Patterns
Two main dispatch patterns are used:

1. **No-Parameter Operations** (hardsigmoid, softsign, hardswish):
   ```cpp
   case UnaryOpType::OP: return {"op_tile_init();", fmt::format("op_tile({});", idst)};
   ```

2. **Parameterized Operations** (rpow):
   ```cpp
   case UnaryOpType::RPOW: {
       return {"rpow_tile_init();", fmt::format("rpow_tile({}, {:#x}u);", idst, param)};
   }
   ```

### Golden Functions
Each operation has a Python golden function registered with ttnn for test validation:
```python
def _golden_function_operation(input_tensor_a, *args, **kwargs):
    import torch
    return torch.operation(input_tensor_a)

ttnn.attach_golden_function(ttnn.operation, golden_function=_golden_function_operation)
```

## File Paths Reference

### Source Code Locations

**SFPU Kernels:**
- Wormhole: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_{op}.h`
- Blackhole: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_{op}.h`

**LLK Wrappers:**
- Wormhole: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_{op}.h`
- Blackhole: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_{op}.h`

**Compute API:**
- `tt_metal/hw/inc/api/compute/eltwise_unary/{op}.h`

**Registration & Dispatch:**
- Enum: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- Includes: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
- LLK API: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`
- Dispatch: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- Golden: `ttnn/ttnn/operations/unary.py`
- Registration: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`

**Tests:**
- `tests/ttnn/unit_tests/operations/eltwise/test_{op}.py`

## Implementation Insights

### hardsigmoid
- Uses two separate v_if blocks for min/max clamping
- All constants embedded as bit-cast FP32 values
- No special initialization required

### rpow
- Logarithmic decomposition: base^x = 2^(x * log2(base))
- IEEE754 exponent/mantissa decomposition
- Minimax polynomial for log2(1+f)
- Parameterized operation with float base cast to uint32_t

### softsign
- Reciprocal-based implementation for efficiency
- Input clamping (to 1e30) to prevent subnormal results
- Requires reciprocal initialization
- Template ITERATIONS parameter for loop control

### hardswish
- Uses SFPU vec_min_max instruction for efficient min/max
- Inline hard-sigmoid computation
- No approximation mode (fixed to exact)
- Exhaustive bfloat16 test coverage

## Test Coverage Summary

All operations include comprehensive tests:
- Basic functionality validation
- Edge cases and boundary conditions
- Exhaustive bitpattern testing (for hardswish)
- ULP (Unit in Last Place) accuracy verification
- PCC (Pearson Correlation Coefficient) validation
- Multiple tensor shapes and input ranges

## Total Documentation

- **4 Operations Documented**: hardsigmoid, rpow, softsign, hardswish
- **1,479 Total Lines**: Comprehensive coverage with full source code
- **12 Layers Per Operation**: Complete stack from SFPU to Python API
- **100% File Coverage**: All created/modified files included with full code
