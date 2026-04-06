# SFPU Kernel Analysis: hardswish

## 1. Math Definition

```
hardswish(x) = x * min(max(x + 3, 0), 6) / 6
             = x * hardsigmoid(x)
             = x * clamp(x/6 + 0.5, 0, 1)
```

Piecewise:
- `x <= -3` => `0`
- `x >= 3`  => `x`
- otherwise => `x * (x/6 + 0.5)`

This is equivalent to multiplying input `x` by the `hardsigmoid(x)` function. The operation is a common neural network activation function (PyTorch `torch.nn.functional.hardswish`).

## 2. SFPU Kernel Implementation

### Core Compute Function

**File (Wormhole):** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h`
**File (Blackhole):** `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h`

Both architectures share **identical** kernel code:

```cpp
namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardswish() {
    constexpr float one_sixth = 1.0f / 6.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat hsigmoid = x * one_sixth + 0.5f;

        // Clamp hardsigmoid to [0, 1]
        v_if(hsigmoid < 0.0f) { hsigmoid = 0.0f; }
        v_endif;
        v_if(hsigmoid > sfpi::vConst1) { hsigmoid = sfpi::vConst1; }
        v_endif;

        sfpi::dst_reg[0] = x * hsigmoid;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

### SFPI Instructions Used

| SFPI Primitive | Purpose |
|---|---|
| `sfpi::dst_reg[0]` | Load input datum from DST register |
| `sfpi::dst_reg[0] = ...` | Store result back to DST register |
| `sfpi::dst_reg++` | Advance DST register pointer to next face |
| `sfpi::vFloat` | SIMD float register type (32 elements wide) |
| `sfpi::vConst1` | Hardware constant representing `1.0f` |
| `v_if(...) / v_endif` | Predicated SIMD conditional (lane masking) |
| `*` (vFloat multiply) | SFPU vector multiply |
| `+` (vFloat + scalar) | SFPU vector-scalar add |
| `<`, `>` (vFloat compare) | SFPU vector comparison for predication |

### Algorithm Breakdown (per face iteration)

1. **Load**: Read input `x` from `dst_reg[0]`
2. **Linear transform**: Compute `hsigmoid = x * (1/6) + 0.5` (the hardsigmoid value)
3. **Clamp lower**: If `hsigmoid < 0`, set `hsigmoid = 0` (predicated)
4. **Clamp upper**: If `hsigmoid > 1`, set `hsigmoid = 1` (predicated, uses `vConst1`)
5. **Multiply**: Compute `result = x * hsigmoid`
6. **Store**: Write result to `dst_reg[0]`, advance pointer

### Relationship to hardsigmoid

The hardswish kernel inlines the hardsigmoid computation rather than calling it. Comparing with `ckernel_sfpu_hardsigmoid.h`:

- **hardsigmoid**: computes `clamp(x/6 + 0.5, 0, 1)` and stores it directly
- **hardswish**: computes the same `clamp(x/6 + 0.5, 0, 1)` into a local variable, then multiplies by `x` before storing

The inlining avoids the overhead of a second DST read/write round-trip.

### Template Parameters

- **`APPROXIMATION_MODE`**: Boolean flag (typically `false`). The kernel does not use this parameter to change behavior — the same exact code path runs in both modes. This is a placeholder for potential future approximation variants.
- **`ITERATIONS`**: Defaults to `8` (standard for processing all 8 faces of a 32x32 tile: 8 faces x 32 elements = 256 elements per tile, with 4 faces per row across 2 rows in bfloat16). The `#pragma GCC unroll 8` ensures the loop is fully unrolled at compile time.

### Instruction Count Estimate (per face)

| Operation | Est. Cycles |
|---|---|
| `dst_reg[0]` load | 1 |
| multiply (`x * one_sixth`) | 1 |
| add (`+ 0.5f`) | 1 |
| compare + predicated assign (clamp low) | 2 |
| compare + predicated assign (clamp high) | 2 |
| multiply (`x * hsigmoid`) | 1 |
| store + advance | 1 |
| **Total per face** | **~9** |
| **Total per tile (8 faces)** | **~72** |

The kernel is lightweight — no transcendental functions, no LUT lookups, no iterative approximations. All operations are basic arithmetic and predicated assignments.

## 3. LLK Integration Layer

### LLK Wrappers

**File (Wormhole):** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardswish.h`
**File (Blackhole):** `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardswish.h`

```cpp
namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_hardswish_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardswish, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardswish(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_hardswish<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

Key observations:
- Uses `SfpuType::hardswish` enum from `llk_sfpu_types.h`
- Dispatch via `_llk_math_eltwise_unary_sfpu_params_` (the standard no-extra-params dispatch template)
- Default `VectorMode::RC` — processes all rows and columns of the tile
- No extra runtime parameters — this is a pure parameterless unary op

## 4. Compute API (Top-Level Kernel API)

**File:** `tt_metal/hw/inc/api/compute/eltwise_unary/hardswish.h`

```cpp
ALWI void hardswish_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_hardswish<APPROX>(idst)));
}

ALWI void hardswish_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_hardswish_init<APPROX>()));
}
```

- `hardswish_tile_init()` — called once before the compute loop to initialize SFPU state
- `hardswish_tile(idst)` — called per tile, where `idst` is the index in the DST register buffer
- `APPROX` is a compile-time constant set by the build system
- `MATH(...)` macro ensures the code runs only on the math RISC-V thread

### Conditional Include (Split Includes)

**File:** `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`

```cpp
#if SFPU_OP_HARDSWISH_INCLUDE
#include "api/compute/eltwise_unary/hardswish.h"
#endif
```

The `SFPU_OP_HARDSWISH_INCLUDE` define is set to `1` by the program factory when hardswish is the active op, keeping compile times fast by only including the needed SFPU kernel.

## 5. Host-Side Registration and Dispatch

### UnaryOpType Enum

**File:** `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp:122`

```cpp
HARDSWISH,
```

Registered in the `UnaryOpType` enum alongside other unary operations.

### Macro Definition (Conditional Compile Flag)

**File:** `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp:21`

```cpp
case UnaryOpType::HARDSWISH: return "SFPU_OP_HARDSWISH_INCLUDE";
```

This returns the preprocessor define name that gates the kernel include.

### Op Init and Func Strings (SFPU_OP_CHAIN_0)

**File:** `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp:66`

```cpp
case UnaryOpType::HARDSWISH:
    return {"hardswish_tile_init();", fmt::format("hardswish_tile({});", idst)};
```

These strings are injected into the auto-generated compute kernel as `SFPU_OP_CHAIN_0` init and per-tile calls.

### C++ API Registration

**File:** `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:157`

```cpp
REGISTER_UNARY_OPERATION(hardswish, HARDSWISH)
```

This macro generates the `ttnn::hardswish` C++ function and wires it to `UnaryOpType::HARDSWISH`.

### Python Golden Function

**File:** `ttnn/ttnn/operations/unary.py:78-84`

```python
def _golden_function_hardswish(input_tensor_a, *args, **kwargs):
    import torch
    return torch.nn.functional.hardswish(input_tensor_a)

ttnn.attach_golden_function(ttnn.hardswish, golden_function=_golden_function_hardswish)
```

### Build System

**File:** `tt_metal/hw/sources.cmake:48`

```
inc/api/compute/eltwise_unary/hardswish.h
```

Listed in the hardware sources for compile tracking.

## 6. Unary NG (Next-Gen) Registration

The operation is also registered in the next-generation unary framework:

**File:** `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`

```cpp
case UnaryOpType::HARDSWISH: return "SFPU_OP_HARDSWISH_INCLUDE";  // line 22
case UnaryOpType::HARDSWISH: return {"hardswish_tile_init();", fmt::format("hardswish_tile({});", idst)};  // line 88
```

Both legacy (`unary/`) and NG (`unary_ng/`) factories share the same SFPU kernel code and init/func strings.

## 7. Backward Operation

**File:** `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp:915`

The backward operation (`hardswish_bw`) is implemented as a **composite** using existing tensor ops (not a custom SFPU kernel):

```
grad_result = where(input < -3, 0, where(input <= 3, grad * (input/3 + 0.5), grad))
```

This matches the mathematical derivative:
- `x < -3`: gradient = 0
- `-3 <= x <= 3`: gradient = `x/3 + 0.5`
- `x > 3`: gradient = 1

## 8. Test Coverage

**File:** `tests/ttnn/unit_tests/operations/eltwise/test_hardswish.py`

Two test cases:

1. **`test_hardswish`**: Parametrized over shapes `[1,1,32,32]`, `[1,1,320,384]`, `[1,3,320,384]` with `bfloat16` dtype. Validates against `torch.nn.functional.hardswish` with PCC >= 0.999.

2. **`test_hardswish_piecewise`**: Wide-range input (`*10`), validates:
   - Overall PCC >= 0.999
   - Exact zero output for `x <= -3`
   - Identity output for `x >= 3` (PCC check)

## 9. Complete File Inventory

| Layer | File Path |
|---|---|
| SFPU kernel (WH) | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h` |
| SFPU kernel (BH) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h` |
| LLK wrapper (WH) | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardswish.h` |
| LLK wrapper (BH) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardswish.h` |
| SfpuType enum (WH) | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` |
| SfpuType enum (BH) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` |
| Compute API header | `tt_metal/hw/inc/api/compute/eltwise_unary/hardswish.h` |
| Split includes | `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` |
| UnaryOpType enum | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` |
| Legacy dispatch | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` |
| NG dispatch | `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` |
| C++ API registration | `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` |
| Python golden | `ttnn/ttnn/operations/unary.py` |
| Build system | `tt_metal/hw/sources.cmake` |
| Backward impl | `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp` |
| Backward header | `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.hpp` |
| Backward binding | `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward_nanobind.cpp` |
| Unit test | `tests/ttnn/unit_tests/operations/eltwise/test_hardswish.py` |

## 10. Key Observations for Implementors

1. **No runtime parameters**: Hardswish is a zero-parameter unary op. No `float_params` or `int_params` are passed through the dispatch chain.

2. **Inlined hardsigmoid**: Rather than calling the hardsigmoid kernel, the hardswish kernel computes hardsigmoid inline. This avoids a double DST read/write.

3. **Efficient clamping**: Uses `vConst1` hardware constant for the upper clamp bound, avoiding loading `1.0f` from memory.

4. **No transcendentals**: The entire kernel uses only multiply, add, compare, and predicated assignment — no exp, log, or reciprocal. This makes it one of the cheapest activation functions.

5. **Identical across architectures**: Wormhole and Blackhole share the exact same SFPU kernel code, so no architecture-specific tuning is needed.

6. **Standard unary pattern**: Follows the canonical parameterless unary SFPU pattern — `_llk_math_eltwise_unary_sfpu_params_` dispatch, `SFPU_OP_*_INCLUDE` gating, `*_tile_init()` / `*_tile()` API.
