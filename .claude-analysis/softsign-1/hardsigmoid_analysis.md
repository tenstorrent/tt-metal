# SFPU Kernel Analysis: hardsigmoid

## 1. Operation Overview

**Math Definition:**
```
hardsigmoid(x) = max(0, min(1, x/6 + 0.5))
```

Piecewise linear approximation of sigmoid:
- `x <= -3` => `0`
- `x >= 3` => `1`
- otherwise => `x * (1/6) + 0.5`

**Operation Category:** Unary elementwise activation (piecewise linear, no transcendentals)

**UnaryOpType Enum Value:** `HARDSIGMOID` (in `unary_op_types.hpp:121`)

**SfpuType Enum Value:** `hardsigmoid` (in `llk_sfpu_types.h:11`)

---

## 2. SFPU Kernel Implementation

### 2.1 Core SFPU Kernel Function

**File (Wormhole):** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`
**File (Blackhole):** `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`

Both architectures share **identical** implementations:

```cpp
namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardsigmoid() {
    constexpr float one_sixth = 1.0f / 6.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = x * one_sixth + 0.5f;

        // Clamp to [0, 1]
        v_if(result < 0.0f) { result = 0.0f; }
        v_endif;
        v_if(result > sfpi::vConst1) { result = sfpi::vConst1; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

### 2.2 SFPI Instructions Used

| SFPI Construct | Purpose | Count per iteration |
|---|---|---|
| `sfpi::dst_reg[0]` (read) | Load element from DST register | 1 |
| `sfpi::vFloat` multiply (`x * one_sixth`) | Scalar-vector multiply (1/6 constant) | 1 |
| `sfpi::vFloat` add (`+ 0.5f`) | Scalar-vector add (0.5 constant) | 1 |
| `v_if(result < 0.0f)` | Conditional: clamp lower bound to 0 | 1 |
| `v_if(result > sfpi::vConst1)` | Conditional: clamp upper bound to 1 | 1 |
| `sfpi::vConst1` | Hardware constant for 1.0f | 1 |
| `sfpi::dst_reg[0]` (write) | Store result back to DST | 1 |
| `sfpi::dst_reg++` | Advance to next SFPU row | 1 |

**Key observations:**
- **No transcendental SFPU instructions** — purely arithmetic (multiply, add) plus conditional assignment
- Uses `sfpi::vConst1` hardware constant for the upper clamp (avoids loading 1.0f from memory)
- The `APPROXIMATION_MODE` template parameter is accepted but **not used** — the function is exact for its piecewise linear definition
- Two separate `v_if` blocks for clamping (lower bound, then upper bound) rather than a single min/max chain

### 2.3 Iteration Structure

- Default `ITERATIONS = 8` — processes 8 SFPU rows per call (standard for 32x32 tiles with 4-wide SIMD, covering 32 rows = 8 iterations x 4 elements)
- `#pragma GCC unroll 8` — full unroll hint for the iteration loop
- Each iteration is independent (no cross-row data dependencies)

---

## 3. LLK (Low-Level Kernel) Wrapper

**File (Wormhole):** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h`
**File (Blackhole):** `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h`

Both identical:

```cpp
namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_hardsigmoid_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardsigmoid, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardsigmoid(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_hardsigmoid<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

**Pattern:** Standard parameterless unary SFPU wrapper:
- `_init()` — calls generic `llk_math_eltwise_unary_sfpu_init` with the `SfpuType::hardsigmoid` tag
- Execution function — passes `calculate_hardsigmoid` as a function pointer to `_llk_math_eltwise_unary_sfpu_params_`
- No runtime parameters passed to the SFPU function (constants are compile-time)

---

## 4. Compute API (Tile-level)

**File:** `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h`

```cpp
ALWI void hardsigmoid_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>(idst)));
}

ALWI void hardsigmoid_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_hardsigmoid_init<APPROX>()));
}
```

**Pattern:** Standard tile-level API pair (`_tile` + `_tile_init`). The `APPROX` macro is resolved at compile time. The `MATH((...))` macro ensures execution only on the math RISC-V thread.

### Include registration

- **`sfpu_split_includes.h`:** Guarded by `#if SFPU_OP_HARDSIGMOID_INCLUDE` — conditional include for split-compile mode
- **`activations.h`:** Unconditionally includes `hardsigmoid.h` — aggregation header for activation functions
- **`llk_math_unary_sfpu_api.h`:** Includes `llk_math_eltwise_unary_sfpu_hardsigmoid.h` directly
- **`sources.cmake`:** Lists `inc/api/compute/eltwise_unary/hardsigmoid.h`

---

## 5. Host-Side Dispatch (UnaryProgramFactory)

### 5.1 UnaryOpType Registration

**File:** `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp:121`
```cpp
HARDSIGMOID,
```

### 5.2 SFPU_OP_CHAIN Dispatch

**File:** `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp:66`
```cpp
case UnaryOpType::HARDSIGMOID:
    return {"hardsigmoid_tile_init();", fmt::format("hardsigmoid_tile({});", idst)};
```

This generates the `SFPU_OP_INIT_0` and `SFPU_OP_CHAIN_0` defines that the generic `eltwise_sfpu.cpp` compute kernel uses. The hardsigmoid dispatch returns:
- Init string: `"hardsigmoid_tile_init();"`
- Op string: `"hardsigmoid_tile(0);"`

**Also registered in unary_ng dispatch** (`unary_ng_op_utils.cpp:89`):
```cpp
case UnaryOpType::HARDSIGMOID:
    return {"hardsigmoid_tile_init();", fmt::format("hardsigmoid_tile({});", idst)};
```

### 5.3 No Runtime Parameters

Hardsigmoid takes **no runtime parameters** — the alpha=1/6 and beta=0.5 are hardcoded constants in the SFPU kernel. This means:
- No `UnaryWithParam` needed
- Simple `REGISTER_UNARY_OPERATION(hardsigmoid, HARDSIGMOID)` macro in `unary.hpp:98`

---

## 6. Python Binding & API

### 6.1 C++ Nanobind Registration

**File:** `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp:1791-1795`
```cpp
bind_unary_operation<"hardsigmoid", &ttnn::hardsigmoid>(
    mod,
    R"doc(\text{hardsigmoid}(x) = \max(0, \min(1, x / 6 + 0.5)))doc",
    "",
    R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
```

### 6.2 Python API

**File:** `ttnn/ttnn/operations/unary.py:42`
```python
"hardsigmoid": torch.nn.functional.hardsigmoid,
```
Golden function mapped to `torch.nn.functional.hardsigmoid`.

**File:** `ttnn/ttnn/operations/unary.py:62`
```python
ttnn.hardsigmoid,
```
Listed in `TTNN_ELTWISE_UNARY_CPP_FUNCTIONS`.

**Supported dtypes:** BFLOAT16, BFLOAT8_B, FLOAT32

---

## 7. Compute Kernel Usage Patterns

### 7.1 Standard Unary Path (via UnaryProgramFactory)

The standard path uses `eltwise_sfpu.cpp` which:
1. Reads tile from CB c_0 into DST register
2. Applies `SFPU_OP_CHAIN_0` (expands to `hardsigmoid_tile(0);`)
3. Packs result from DST to CB c_2

### 7.2 Hardswish Composite Kernel (Custom Compute Kernel)

Two custom compute kernels use hardsigmoid as a building block for hardswish (`x * hardsigmoid(x)`):

**`hardswish_kernel.cpp`:** Uses `hardsigmoid_tile(0)` then `mul_binary_tile(0, 1, 0)` — copies input to two DST slots, applies hardsigmoid to slot 0, multiplies slots 0 and 1.

**`hardswish_kernel_sfpu.cpp`:** Uses `hardsigmoid_tile(0)` then `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>` — more efficient variant that reuses DST for the multiplication.

---

## 8. Test Coverage

**File:** `tests/ttnn/unit_tests/operations/eltwise/test_hardsigmoid.py`

Two tests:
1. **`test_hardsigmoid`** — Parametrized over shapes `[1,1,32,32]`, `[1,1,320,384]`, `[1,3,320,384]` with bfloat16 dtype. Compares against `torch.nn.functional.hardsigmoid` with PCC threshold 0.999.
2. **`test_hardsigmoid_output_range`** — Verifies output is always in `[0, 1]` using wide-range input (`*10`).

---

## 9. Architecture Summary

### File Inventory

| Layer | File Path | Role |
|---|---|---|
| SFPU kernel (WH) | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h` | Core math: `calculate_hardsigmoid()` |
| SFPU kernel (BH) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h` | Identical to WH |
| LLK wrapper (WH) | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h` | `llk_math_eltwise_unary_sfpu_hardsigmoid[_init]` |
| LLK wrapper (BH) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h` | Identical to WH |
| Compute API | `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h` | `hardsigmoid_tile()`, `hardsigmoid_tile_init()` |
| Split includes | `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | `SFPU_OP_HARDSIGMOID_INCLUDE` guard |
| Activations header | `tt_metal/hw/inc/api/compute/eltwise_unary/activations.h` | Aggregation include |
| SfpuType enum | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu_types.h` | `SfpuType::hardsigmoid` |
| LLK API header | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_math_unary_sfpu_api.h` | Includes LLK wrapper |
| CMake | `tt_metal/hw/sources.cmake` | Build system registration |
| UnaryOpType | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | `UnaryOpType::HARDSIGMOID` |
| Op dispatch | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | SFPU_OP_CHAIN generation |
| NG dispatch | `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` | Unary-NG dispatch |
| C++ API | `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` | `REGISTER_UNARY_OPERATION(hardsigmoid, HARDSIGMOID)` |
| Nanobind | `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` | Python binding |
| Python API | `ttnn/ttnn/operations/unary.py` | `ttnn.hardsigmoid` + golden function |
| Python activations | `ttnn/ttnn/operations/activations.py` | `UnaryOpType.HARDSIGMOID` mapping |
| Tests | `tests/ttnn/unit_tests/operations/eltwise/test_hardsigmoid.py` | Unit tests |

### Key Implementation Characteristics

- **Complexity:** Very simple — multiply, add, two conditional clamps. No transcendentals, no LUTs.
- **Parameters:** None at runtime. Alpha (1/6) and beta (0.5) are compile-time constants.
- **Architecture parity:** WH and BH implementations are byte-identical.
- **APPROXIMATION_MODE:** Template parameter exists but is unused — the operation is already exact for its piecewise-linear definition.
- **Hardware constants:** Uses `sfpi::vConst1` for the upper clamp value (1.0f), avoiding memory loads.
- **Conditional execution:** Uses SFPI predicated execution (`v_if`/`v_endif`) for the clamping — this is lane-wise conditional assignment on the SFPU vector.
- **Reuse in hardswish:** The `hardsigmoid_tile()` API is directly reused by hardswish compute kernels, making it a building block for composite operations.
