# SFPU Kernel Analysis: frac

## 1. Operation Overview

| Property | Value |
|---|---|
| **Operation** | `frac` |
| **Math Definition** | `frac(x) = x - trunc(x)` (fractional part, truncation toward zero) |
| **PyTorch Equivalent** | `torch.frac()` |
| **UnaryOpType Enum** | `UnaryOpType::FRAC` (value at line 99 of `unary_op_types.hpp`) |
| **SfpuType Enum** | `SfpuType::frac` (line 9 of `llk_sfpu_types.h`) |
| **Parameterized** | No (no runtime parameters) |
| **Approximate Mode** | Not used (`get_op_approx_mode` returns `false` by default) |
| **Supported Dtypes** | BFLOAT16, BFLOAT8_B, FLOAT32 |

## 2. File Inventory

### Layer 1: SFPU Kernel (core math)
| File | Path |
|---|---|
| **Wormhole B0** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h` |
| **Blackhole** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h` |

### Layer 2: LLK Math Wrapper
| File | Path |
|---|---|
| **Wormhole B0** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h` |
| **Blackhole** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h` |

### Layer 3: Compute Kernel API Header
| File | Path |
|---|---|
| **Tile API** | `tt_metal/hw/inc/api/compute/eltwise_unary/frac.h` |

### Layer 4: Split Include Guard
| File | Path |
|---|---|
| **Split includes** | `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` |

### Layer 5: SfpuType Enum
| File | Path |
|---|---|
| **Wormhole B0** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` |
| **Blackhole** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` |

### Layer 6: Compute Kernel (shared, dispatches via SFPU_OP_CHAIN_0)
| File | Path |
|---|---|
| **Legacy unary** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` |
| **Unary NG** | `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/kernels/compute/eltwise_sfpu.cpp` |

### Layer 7: Host-side dispatch (op_utils)
| File | Path |
|---|---|
| **Legacy unary** | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` |
| **Unary NG** | `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` |

### Layer 8: UnaryOpType Enum
| File | Path |
|---|---|
| **Enum definition** | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` |

### Layer 9: C++ API Registration
| File | Path |
|---|---|
| **TTNN C++ API** | `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` |

### Layer 10: Python Binding
| File | Path |
|---|---|
| **Nanobind** | `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` |

## 3. SFPU Kernel Deep Dive

### 3.1 Source Code (Wormhole B0 = Blackhole, identical)

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_frac() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Default: frac = 0 for integers (exp >= 23)
        sfpi::vFloat trunc_x = x;

        // Extract unbiased exponent
        sfpi::vInt exp = sfpi::exexp(x);

        // Case 1: |x| < 1 (exp < 0) -- trunc toward zero gives 0
        v_if(exp < 0) { trunc_x = 0.0f; }
        v_endif;

        // Case 2: 0 <= exp < 23 (has fractional bits in float32)
        v_if(exp >= 0 && exp < 23) {
            sfpi::vUInt shift = sfpi::vUInt(23 - exp);
            sfpi::vInt mask = sfpi::vInt(-1) << shift;

            sfpi::vInt xi = sfpi::reinterpret<sfpi::vInt>(x);
            trunc_x = sfpi::reinterpret<sfpi::vFloat>(xi & mask);
        }
        v_endif;

        // frac(x) = x - trunc(x)
        sfpi::dst_reg[0] = x - trunc_x;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

### 3.2 Algorithm Analysis

The kernel computes `frac(x) = x - trunc(x)` using IEEE 754 float32 bit manipulation:

**Three cases based on unbiased exponent `e`:**

1. **`e < 0` (|x| < 1):** The entire value is fractional. `trunc(x) = 0`, so `frac(x) = x`.

2. **`0 <= e < 23` (mixed integer + fractional bits):** The float32 mantissa has 23 bits. For exponent `e`, the lowest `(23 - e)` mantissa bits encode the fractional part. A bitmask `0xFFFFFFFF << (23 - e)` zeroes out the fractional bits, yielding `trunc(x)`. Then `frac(x) = x - trunc(x)`.

3. **`e >= 23` (already an integer):** All 23 mantissa bits encode integer magnitude. `trunc(x) = x`, so `frac(x) = 0`. This is the default path (trunc_x is initialized to x).

**Key insight:** The algorithm matches PyTorch `torch.frac()` semantics exactly -- it uses truncation toward zero (not floor), so for negative inputs the result is negative or zero (e.g., `frac(-3.7) = -0.7`).

### 3.3 SFPI Instructions Used

| Instruction/Intrinsic | Purpose | Count per iteration |
|---|---|---|
| `sfpi::dst_reg[0]` (load) | Read one element from DST register | 1 |
| `sfpi::exexp(x)` | Extract unbiased exponent from float | 1 |
| `sfpi::vUInt(23 - exp)` | Compute shift amount as vector uint | 1 |
| `sfpi::vInt(-1) << shift` | Create bitmask by left-shifting -1 | 1 |
| `sfpi::reinterpret<vInt>(x)` | Bitcast float to int for masking | 1 |
| `xi & mask` | Apply bitmask to zero fractional bits | 1 |
| `sfpi::reinterpret<vFloat>(...)` | Bitcast int back to float | 1 |
| `x - trunc_x` | Subtract to get fractional part | 1 |
| `sfpi::dst_reg[0] = ...` (store) | Write result back to DST register | 1 |
| `sfpi::dst_reg++` | Advance DST pointer to next row | 1 |
| `v_if` / `v_endif` | Predicated execution (2 predicates) | 2 |

### 3.4 Template Parameters

| Parameter | Default | Description |
|---|---|---|
| `APPROXIMATION_MODE` | (from `APPROX` define) | Not used in the algorithm (no approximation path) |
| `ITERATIONS` | 8 | Number of rows processed per call. 8 rows x 4 faces = 32 rows per tile |

### 3.5 Cross-Architecture Differences

**None.** The Wormhole B0 and Blackhole SFPU kernels are byte-for-byte identical.

## 4. LLK Wrapper Pattern

```cpp
// llk_math_eltwise_unary_sfpu_frac.h

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_frac_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::frac, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_frac(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_frac<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}
```

This is the standard LLK wrapper pattern:
- **Init function** calls `llk_math_eltwise_unary_sfpu_init` with the `SfpuType::frac` enum for SFPU initialization.
- **Compute function** calls `_llk_math_eltwise_unary_sfpu_params_` which handles DST addressing, stall management, and face iteration (RC mode = all 4 faces).

## 5. Compute Kernel API

```cpp
// tt_metal/hw/inc/api/compute/eltwise_unary/frac.h

ALWI void frac_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_frac<APPROX>(idst)));
}

ALWI void frac_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_frac_init<APPROX>()));
}
```

Standard tile API: `frac_tile_init()` initializes the SFPU for frac, `frac_tile(idst)` executes frac on tile at DST index `idst`. The `MATH((...))` macro ensures the code runs on the math thread (TRISC_MATH).

## 6. Host-Side Dispatch

### 6.1 Include Guard Mechanism

In `sfpu_split_includes.h`:
```cpp
#if SFPU_OP_FRAC_INCLUDE
#include "api/compute/eltwise_unary/frac.h"
#endif
```

The define `SFPU_OP_FRAC_INCLUDE` is set to `"1"` by `update_macro_defines()` when the op type is `UnaryOpType::FRAC`. This conditional include reduces compile time by only including the frac header when needed.

### 6.2 SFPU_OP_CHAIN_0 Construction

From `unary_op_utils.cpp`:
```cpp
// get_macro_definition:
case UnaryOpType::FRAC: return "SFPU_OP_FRAC_INCLUDE";

// get_op_init_and_func_default:
case UnaryOpType::FRAC: return {"frac_tile_init();", fmt::format("frac_tile({});", idst)};
```

The `get_block_defines()` function assembles these into preprocessor defines:
- `SFPU_OP_CHAIN_0_INIT_0` = `frac_tile_init();`
- `SFPU_OP_CHAIN_0_FUNC_0` = `frac_tile(0);`
- `SFPU_OP_CHAIN_0` = `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0`
- `SFPU_OP_FRAC_INCLUDE` = `1`

These defines are passed to the compiler when building the compute kernel `eltwise_sfpu.cpp`.

### 6.3 Compute Kernel Path

`get_compute_kernel_path()` returns `"eltwise_sfpu.cpp"` (the default shared compute kernel, not a custom one).

## 7. Registration Chain Summary

```
Python: ttnn.frac(tensor)
  --> nanobind: bind_unary_operation<"frac", &ttnn::frac>(...)
    --> C++ API: ttnn::frac() via REGISTER_UNARY_OPERATION(frac, FRAC)
      --> ttnn::detail::unary_impl(input, {UnaryWithParam{UnaryOpType::FRAC}}, ...)
        --> UnaryProgramFactory builds compute kernel with defines:
            SFPU_OP_CHAIN_0 = "frac_tile_init(); frac_tile(0);"
            SFPU_OP_FRAC_INCLUDE = 1
          --> eltwise_sfpu.cpp kernel_main():
              #ifdef SFPU_OP_CHAIN_0
              SFPU_OP_CHAIN_0  // expands to: frac_tile_init(); frac_tile(0);
              #endif
            --> frac_tile() calls llk_math_eltwise_unary_sfpu_frac<APPROX>(idst)
              --> _llk_math_eltwise_unary_sfpu_params_<APPROX>(calculate_frac<...>, ...)
                --> calculate_frac() executes SFPI instructions on SFPU
```

## 8. Key Implementation Characteristics

| Characteristic | Detail |
|---|---|
| **No runtime parameters** | Frac takes no extra parameters beyond the input tile |
| **No approximation path** | `APPROXIMATION_MODE` template parameter exists but is unused |
| **Pure bit-manipulation** | Uses `exexp`, bitwise AND, shift -- no FPU math functions, no LUTs |
| **Three-way branch** | `v_if` predicated execution for exp < 0, 0 <= exp < 23, and exp >= 23 |
| **IEEE 754 exact** | Bitmask approach is mathematically exact for float32 |
| **No special-case handling** | No explicit NaN/Inf handling (follows IEEE 754 naturally) |
| **Identical across architectures** | WH B0 and BH kernels are the same |
| **Shared compute kernel** | Uses the common `eltwise_sfpu.cpp`, not a custom compute kernel |
| **Standard unary registration** | Uses `REGISTER_UNARY_OPERATION` (no fast_and_approximate_mode parameter) |

## 9. Relevance for New Operation Implementation

When implementing a new SFPU unary operation modeled after frac, the key patterns to replicate are:

1. **SFPU kernel file** (`ckernel_sfpu_frac.h`): Template with `APPROXIMATION_MODE` and `ITERATIONS=8`, loop over `ITERATIONS`, read from `dst_reg[0]`, write to `dst_reg[0]`, increment `dst_reg++`.

2. **LLK wrapper** (`llk_math_eltwise_unary_sfpu_frac.h`): Init function calling `llk_math_eltwise_unary_sfpu_init<SfpuType::X>`, compute function calling `_llk_math_eltwise_unary_sfpu_params_`.

3. **Tile API header** (`frac.h` in `api/compute/eltwise_unary/`): `ALWI void X_tile(uint32_t idst)` and `ALWI void X_tile_init()` with `MATH((...))` wrapper.

4. **SfpuType enum entry** in `llk_sfpu_types.h` for both WH and BH.

5. **Split include guard** in `sfpu_split_includes.h` with `SFPU_OP_X_INCLUDE`.

6. **Host dispatch** in `unary_op_utils.cpp`: `get_macro_definition()` returning `"SFPU_OP_X_INCLUDE"` and `get_op_init_and_func_default()` returning `{X_tile_init, X_tile}`.

7. **UnaryOpType enum** entry in `unary_op_types.hpp`.

8. **C++ API** via `REGISTER_UNARY_OPERATION(x, X)` in `unary.hpp`.

9. **Python binding** via `bind_unary_operation<"x", &ttnn::x>(...)` in `unary_nanobind.cpp`.

SFPI intrinsics demonstrated by frac that are useful for new operations:
- `sfpi::exexp(x)` -- extract unbiased exponent
- `sfpi::reinterpret<vInt/vFloat>(...)` -- type-punning between float and int
- `sfpi::vInt(-1) << shift` -- dynamic bitmask construction
- `v_if(...) { ... } v_endif;` -- predicated (per-lane) execution
