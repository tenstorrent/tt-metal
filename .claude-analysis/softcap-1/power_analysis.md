## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

**CRITICAL FINDING: The `power` SFPU kernel does not exist in this codebase version.** The entire device-side SFPU kernel chain -- from the LLK dispatch header through to the core SFPU ckernel implementation -- is absent. What follows documents the current state of each abstraction layer, what exists, what is missing, and what implications this has for the operation.

### Unary Dispatch Summary
- **UnaryOpType**: `POWER`
- **Compute kernel**: `eltwise_sfpu.cpp` (default, but POWER has no path to reach it)
- **SFPU_OP_CHAIN_0 expansion**: Would be `power_tile(0, param0)` -- but POWER is NOT handled in any `get_op_init_and_func` switch case, so the dispatch would `TT_FATAL` before reaching the compute kernel

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(POWER)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | N/A -- dispatch unreachable | `get_op_init_and_func()` -- POWER has no case in `get_op_init_and_func_default` and `is_parametrized_type(POWER)` returns false, so `get_op_init_and_func_parameterized` would `TT_FATAL` |
| Effective SFPU path | **Unreachable** -- host-side dispatch crashes before any kernel compilation occurs | N/A |

**Detailed dispatch failure path**: When `ttnn::power(tensor, scalar)` is called, the `UNARY_OP_SCALAR_VARIANT` macro (in `unary.hpp:80-95`) creates `EltwiseUnaryWithParam{UnaryOpType::POWER, param}` with a non-empty params vector. The `get_op_init_and_func<float>()` template (in `unary_op_utils.cpp:120-124`) sees non-empty params and calls `get_op_init_and_func_parameterized()`. This function calls `is_parametrized_type(POWER)` which returns `false` (only `HARDTANH` and `SOFTSHRINK` return true), triggering a `TT_FATAL` at `unary_op_utils.cpp:35-37`.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 337-344) -- `power_tile()` and `power_tile_init()` are declared here but reference LLK functions in a MISSING header |
| **LLK Dispatch** | **MISSING** -- `llk_math_eltwise_unary_sfpu_power.h` is `#include`d by `llk_math_unary_sfpu_api.h` (WH line 14, BH line 9) but the file does not exist on disk in any include path |
| **Core SFPU Implementation** | **MISSING** -- no `ckernel_sfpu_power.h` exists in the tt_llk submodule for any architecture (WH, BH, or Quasar) |
| **Parameters Dispatch** | **MISSING** -- would be defined in `llk_math_eltwise_unary_sfpu_power.h` |

### Call Chain
The intended call chain would be:
1. `power_tile(idst, param0)` (in `compute_kernel_api.h:337`) calls `llk_math_eltwise_unary_sfpu_power<APPROX, DST_ACCUM_MODE>(idst, param0)`
2. `llk_math_eltwise_unary_sfpu_power<>()` would be defined in `llk_math_eltwise_unary_sfpu_power.h` and would call `_llk_math_eltwise_unary_sfpu_params_<>()` from `llk_math_eltwise_unary_sfpu_params.h` with a lambda wrapping the core SFPU function
3. The core SFPU function (e.g., `_calculate_power_()`) would be defined in `ckernel_sfpu_power.h`

**None of steps 2 or 3 exist.** The chain is broken at the LLK dispatch layer.

### Parameters Dispatch Summary
N/A -- the parameters dispatch header `llk_math_eltwise_unary_sfpu_power.h` does not exist.

Based on the generic pattern in `llk_math_eltwise_unary_sfpu_params.h` (which does exist), the standard dispatch would be:
- **Vector mode**: `VectorMode::RC` (all 4 faces, 8 iterations per face)
- **Operation invocation**: The core SFPU function would be called 4 times (once per face), with `SETRWC` between faces to advance DEST addressing
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, dst_reg++ per iteration, SETRWC between faces). Address mode would be `ADDR_MOD_7` on Wormhole (per `llk_math_eltwise_unary_sfpu.h` default configuration)

### Annotated SFPU Kernel Source
**No SFPU kernel source exists for the `power` operation.**

The following files are the closest related code:

#### 1. API Declaration (exists but unreachable)
```cpp
// File: tt_metal/hw/inc/api/compute/compute_kernel_api.h (lines 321-344)
// These declarations reference llk_math_eltwise_unary_sfpu_power which does not exist.

// POWER : y = x^(const param0)
ALWI void power_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_power<APPROX, DST_ACCUM_MODE>(idst, param0)));
}

ALWI void power_tile_init() { MATH((llk_math_eltwise_unary_sfpu_power_init<APPROX>())); }

// POWER_ITERATIVE : y = x^(const param0) -- integer exponents only, iterative multiplication
ALWI void power_iterative_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_power_iterative<APPROX>(idst, param0)));
}

ALWI void power_iterative_tile_init() { MATH((llk_math_eltwise_unary_sfpu_power_iterative_init<APPROX>())); }
```

#### 2. Binary Power Stub (returns 0.0f)
```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h

// POW/DIV/XLOGY implementations removed -- depend on exp/log/recip primitives
// Generator must implement from SFPI instructions

sfpi_inline sfpi::vFloat _calculate_sfpu_binary_power_(sfpi::vFloat base, sfpi::vFloat pow)
{
    return 0.0f; // Stub -- no implementation
}

// In _calculate_sfpu_binary_ template:
// else if constexpr (BINOP == BinaryOp::POW)
// {
//     // Power removed -- depends on exp/log/recip primitives
// }
```

### SFPU Instructions Used
No SFPU instructions are used because the kernel does not exist.

For reference, the mathematical identity for power is `x^p = exp(p * ln(x))`, which would require:
- **SFPLOAD/SFPSTORE**: Load/store values from/to DEST registers
- **SFPMAD**: Fused multiply-add for polynomial evaluation (used in log and exp approximations)
- **SFPEXEXP**: Extract exponent (used in log approximation)
- **SFPEXMAN**: Extract mantissa (used in log approximation)
- **SFPSETEXP**: Set exponent (used in exp reconstruction)
- **SFPDIVP2**: Divide by power of 2 (used in range reduction for exp)
- **SFPSETCC/SFPENCC/SFPCOMPC**: Condition code manipulation for edge cases (x <= 0, NaN, Inf)
- **SFPLOADI**: Load immediate constants for polynomial coefficients

These would be the expected instructions if the kernel were implemented from SFPI primitives.

### SFPU Register Usage
N/A -- no kernel exists. The expected usage would involve:
- **dst_reg[0]**: Primary data register (load input, store output)
- **LREG0-LREG3**: Intermediate computation registers for log/exp pipeline
- **Programmable constants**: Polynomial coefficients for log and exp approximations

### Address Mode Configuration
N/A -- no kernel exists. The standard unary SFPU configuration from `llk_math_eltwise_unary_sfpu.h` would apply:
- **ADDR_MOD_7** (Wormhole/Blackhole): `{srca.incr=0, srcb.incr=0, dest.incr=0}` -- DEST address progression is handled by explicit `SETRWC` instructions in the params dispatch, not by auto-increment

## Missing Implementation Details

### What Would Need to Be Implemented

To create a working `power` SFPU kernel, the following files need to be created:

1. **Core SFPU implementation**: `ckernel_sfpu_power.h` in `tt_llk_wormhole_b0/common/inc/sfpu/` (and corresponding BH version)
   - Must implement `_calculate_power_<APPROXIMATION_MODE>(uint32_t param0)` with the `x^p = exp(p * ln(x))` identity
   - Must handle edge cases: x=0 (result=0 for p>0), x<0 (NaN for non-integer p), NaN/Inf inputs
   - For `power_iterative`, implement iterative multiplication loop for integer exponents

2. **LLK dispatch header**: `llk_math_eltwise_unary_sfpu_power.h` in the `llk_sfpu/` directory (for both WH and BH)
   - Must define `llk_math_eltwise_unary_sfpu_power<APPROX, DST_ACCUM>()` and `llk_math_eltwise_unary_sfpu_power_init<APPROX>()`
   - Must define `llk_math_eltwise_unary_sfpu_power_iterative<APPROX>()` and corresponding init

3. **Split API header**: `power.h` in `tt_metal/hw/inc/api/compute/eltwise_unary/`
   - With proper `SFPU_OP_POWER_INCLUDE` guard in `sfpu_split_includes.h`

4. **Unary op utils registration**: Add POWER cases to `unary_op_utils.cpp`:
   - `is_parametrized_type()`: Return true for POWER
   - `get_op_init_and_func_parameterized()`: Return `{"power_tile_init();", "power_tile(0, param0_bits);"}`
   - `get_macro_definition()`: Return appropriate include guard

### Related Working Implementations

The moreh kernel `moreh_abs_pow_kernel.cpp` implements power as a composite operation using multiple existing SFPU ops:
```
x^p = x^(integer_part) * exp(log(x) * fractional_part)
```
Where `x^(integer_part)` uses `power_iterative_tile()` (iterative multiplication) and the fractional part uses `log_tile()`, `mul_tiles()`, and `exp_tile()`. However, this moreh kernel also references the MISSING `power_iterative_tile()` through `compute_kernel_api.h` and would fail to compile in this codebase version.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Primary dispatch configuration for unary SFPU operations
   **Key Findings**: POWER has no case in `get_op_init_and_func_default`, `get_op_init_and_func_parameterized`, or `is_parametrized_type`. The dispatch would TT_FATAL for POWER.

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Declaration of dispatch functions and parametrized type check
   **Key Findings**: `is_parametrized_type()` only returns true for HARDTANH and SOFTSHRINK.

3. **File**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
   **Reason**: API-level declarations of `power_tile()` and `power_iterative_tile()`
   **Key Findings**: Both functions are declared and call `llk_math_eltwise_unary_sfpu_power<>()` and `llk_math_eltwise_unary_sfpu_power_iterative<>()` respectively, but the LLK header defining these functions does not exist.

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`
   **Reason**: Aggregator header that includes per-operation LLK headers
   **Key Findings**: Line 14 includes `llk_math_eltwise_unary_sfpu_power.h` which does not exist on disk. Same situation on blackhole (line 9).

5. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/` (directory)
   **Reason**: Location where per-op LLK headers and ckernel SFPU implementations live
   **Key Findings**: Only 4 unary ops have implementations here: atanh, frac, sinh, swish. No power-related files exist.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h`
   **Reason**: Contains the binary power stub `_calculate_sfpu_binary_power_`
   **Key Findings**: Returns `0.0f` unconditionally. Comment states "POW/DIV/XLOGY implementations removed -- depend on exp/log/recip primitives. Generator must implement from SFPI instructions."

7. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
   **Reason**: Conditional include mechanism for split SFPU API headers
   **Key Findings**: Only includes frac, swish, atanh, sinh. No power include guard or header.

8. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
   **Reason**: The unary factory's compute kernel
   **Key Findings**: Uses the new split API includes (eltwise_unary.h, sfpu_split_includes.h), NOT the old monolithic compute_kernel_api.h. The SFPU_OP_CHAIN_0 macro expansion is what drives the operation.

9. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: Python/C++ binding declarations for unary ops
   **Key Findings**: `UNARY_OP_SCALAR_VARIANT(power, POWER)` at line 185 creates the host-side entry point. Comment at lines 100-102 explicitly states "Stubs for nuked ops still referenced by other modules. The underlying SFPU kernels may not exist yet."

10. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
    **Reason**: Generic parameters dispatch template for all unary SFPU operations
    **Key Findings**: Defines `_llk_math_eltwise_unary_sfpu_params_()` with VectorMode::RC/R/C face iteration patterns. This would be used by the power LLK dispatch if it existed.

11. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
    **Reason**: Generic unary SFPU initialization and address mode configuration
    **Key Findings**: Default ADDR_MOD_7 configuration with `{srca.incr=0, srcb.incr=0, dest.incr=0}`.

12. **File**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
    **Reason**: Contains `power_tile_to_cb()` which shows how power is implemented as a composite
    **Key Findings**: Implements `x^p = x^(int_part) * exp(log(x) * frac_part)` using `power_iterative_tile()`, `log_tile()`, `mul_tiles()`, and `exp_tile()`. However, this also depends on the MISSING `power_iterative_tile()` via `compute_kernel_api.h`.

13. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative SFPU hardware model reference
    **Key Findings**: Provided instruction set reference and addressing model used to document expected instructions for a power kernel implementation.

14. **File**: `tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal.cpp`
    **Reason**: Runtime include path configuration for compute kernel JIT compilation
    **Key Findings**: Confirmed the include search order for compute kernels: `llk_api/`, `llk_api/llk_sfpu/`, `tt_llk/llk_lib/`, `tt_llk/common/inc/`. The missing `llk_math_eltwise_unary_sfpu_power.h` is not found in any of these paths.
