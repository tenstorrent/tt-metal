# SFPU Kernel Analysis: swish

## 1. Operation Identity

| Field | Value |
|-------|-------|
| **Operation name** | `swish` |
| **UnaryOpType enum** | `UnaryOpType::SWISH` |
| **SfpuType enum** | `SfpuType::swish` |
| **Math definition** | `swish(x) = x * sigmoid(x) = x / (1 + exp(-x))` |
| **Parameters** | None (no runtime parameters) |
| **Parameterized** | No |
| **Approximation mode used** | Template parameter `APPROXIMATION_MODE` exists but is not referenced in the kernel body (kernel always runs the same polynomial/piecewise path) |

## 2. File Manifest

### SFPU kernel (ckernel layer)
- **Wormhole**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
- Content is **identical** across both architectures.

### LLK wrapper (llk layer)
- **Wormhole**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
- Content is **identical** across both architectures.

### Compute API (tile-level)
- `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h`

### Split-include guard
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` (macro `SFPU_OP_SWISH_INCLUDE`)

### SfpuType enum
- **Wormhole**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`

### Host-side dispatch
- **UnaryOpType enum**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` (line 126: `SWISH`)
- **op_utils (old unary)**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
  - `get_macro_definition`: returns `"SFPU_OP_SWISH_INCLUDE"` (line 22)
  - `get_op_init_and_func_default`: returns `{"swish_tile_init();", "swish_tile({idst});"}` (line 50)
- **op_utils (unary_ng)**: `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` (line 89)
  - Same init/func strings as old unary path.
- **C++ API registration**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` (line 163)
  - `REGISTER_UNARY_OPERATION(swish, SWISH)` - no parameters, no `fast_and_approximate_mode`.

### Python binding
- **nanobind**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` (line 1824)
  - `bind_unary_operation<"swish", &ttnn::swish>(...)`
- **Golden function**: `ttnn/ttnn/operations/unary.py` (line 50-56)
  - `torch.nn.functional.silu(input_tensor_a)` (SiLU is the PyTorch name for swish)

### Test
- `tests/ttnn/unit_tests/operations/eltwise/test_swish.py`

## 3. SFPU Kernel Deep Dive

### Function signature
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_swish();
```

### Algorithm: Piecewise sigmoid approximation + multiplication

The kernel computes `swish(x) = x * sigmoid(x)` by first computing `sigmoid(|x|)` using a three-segment piecewise approximation, then applying the negative-x symmetry property, and finally multiplying by `x`.

#### Segment breakdown for `sigmoid(|x|)`:

| Segment | Range | Approximation | Max error |
|---------|-------|--------------|-----------|
| 0 | `|x| <= 2.5` | Degree-3 polynomial: `0.5 + |x| * (0.2533 + |x| * (-0.01479 + |x| * (-0.00747)))` | ~0.007 at `|x| ~ 2.0` |
| 1 | `2.5 < |x| <= 5.0` | Linear: `0.0276 * |x| + 0.855` | ~0.017 at `|x| ~ 4.0` |
| 2 | `|x| > 5.0` | Saturation: `1.0` | ~0.007 at `|x| = 5.0` |

#### Negative-x handling:
Uses the symmetry property `sigmoid(-t) = 1 - sigmoid(t)`:
```cpp
v_if(x < 0.0f) { sig_pos = vConst1 - sig_pos; }
```

### Constants used

| Name | Value | Purpose |
|------|-------|---------|
| `c1` | `0.2533f` | Polynomial coefficient (linear term) |
| `c2` | `-0.01479f` | Polynomial coefficient (quadratic term) |
| `c3` | `-0.00747f` | Polynomial coefficient (cubic term) |
| `lin_slope` | `0.0276f` | Linear segment slope |
| `lin_offset` | `0.855f` | Linear segment offset |
| `bp1` | `2.5f` | Breakpoint between polynomial and linear |
| `bp2` | `5.0f` | Breakpoint between linear and saturation |

All constants are `constexpr float` - no LUT, no runtime args, no `l_reg` usage.

### SFPI instructions and constructs used

| SFPI construct | Usage |
|----------------|-------|
| `sfpi::dst_reg[0]` | Read input value / write output value |
| `sfpi::dst_reg++` | Advance to next face row |
| `sfpi::abs(x)` | Compute absolute value `|x|` |
| `sfpi::vConst1` | Constant `1.0f` for saturation and symmetry |
| `sfpi::vFloat` | Vector float register type |
| `v_if(...) { ... } v_endif;` | Conditional (predicated) execution for piecewise segments |
| Arithmetic: `*`, `+`, `-`, `>`, `<` | Standard vFloat arithmetic and comparison |

### Data flow within the kernel

```
Input: dst_reg[0] = x  (one face row per iteration, 16 elements)

1. ax = abs(x)
2. sig_pos = 0.5 + ax*(c1 + ax*(c2 + ax*c3))   [polynomial, Horner form]
3. if (ax > 2.5): sig_pos = ax*0.0276 + 0.855   [linear override]
4. if (ax > 5.0): sig_pos = 1.0                  [saturation override]
5. if (x < 0.0):  sig_pos = 1.0 - sig_pos        [negative symmetry]
6. dst_reg[0] = x * sig_pos                       [final swish value]
7. dst_reg++                                       [next face row]

Output: dst_reg[0] = swish(x)
```

### Loop structure
- Outer loop: `for (int d = 0; d < ITERATIONS; d++)` with `ITERATIONS = 8` (default)
- Unrolled via `#pragma GCC unroll 8`
- Each iteration processes one face row (16 elements) of a 32x32 tile
- 8 iterations = 128 elements = half a tile face; the program factory calls this twice for a full tile

### Register pressure analysis
- `x`: 1 vFloat register (input, kept for final multiply)
- `ax`: 1 vFloat register (absolute value)
- `sig_pos`: 1 vFloat register (sigmoid approximation, overwritten in-place)
- Total: **3 vFloat registers** - very low pressure, well within SFPU register limits

### Conditional execution count
- 3 `v_if/v_endif` blocks:
  1. `ax > bp1` (linear override)
  2. `ax > bp2` (saturation override)
  3. `x < 0.0f` (negative symmetry)
- All conditions are independent (no nesting); each applies a predicated override

## 4. LLK Wrapper Layer

### Init function
```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_swish_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROXIMATE>();
}
```
- Uses the standard `llk_math_eltwise_unary_sfpu_init` template
- No custom init callback - no LUT, no special register setup needed

### Compute function
```cpp
template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_swish(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_swish<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}
```
- Dispatches via `_llk_math_eltwise_unary_sfpu_params_` (the standard no-extra-params dispatcher)
- `vector_mode` defaults to `VectorMode::RC` (process all rows and columns of tile)
- No runtime parameters passed to the SFPU function

## 5. Compute API Layer

```cpp
ALWI void swish_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)));
}

ALWI void swish_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_swish_init<APPROX>()));
}
```
- Standard tile-level API pattern
- Guarded by `#ifdef TRISC_MATH` (only compiled for the math RISC-V thread)
- `APPROX` is a compile-time constant set by the program factory
- `ALWI` = `__attribute__((always_inline))` - forced inline at call site

## 6. Split-Include Mechanism

In `sfpu_split_includes.h`:
```cpp
#if SFPU_OP_SWISH_INCLUDE
#include "api/compute/eltwise_unary/swish.h"
#endif
```

The host-side dispatch sets `SFPU_OP_SWISH_INCLUDE=1` as a preprocessor define, which causes the compute kernel to include the swish tile API. This avoids pulling in all SFPU operation headers and keeps compile times down.

The macro is set in `unary_op_utils.cpp`:
```cpp
case UnaryOpType::SWISH: return "SFPU_OP_SWISH_INCLUDE";
```

## 7. Host-Side Dispatch Chain

```
ttnn.swish(tensor)                                    # Python API
  -> ttnn::swish(tensor)                              # C++ via REGISTER_UNARY_OPERATION(swish, SWISH)
    -> ttnn::detail::unary_impl(tensor, {UnaryWithParam{UnaryOpType::SWISH}})
      -> UnaryProgramFactory                          # standard unary program factory
        -> get_op_init_and_func_default(SWISH, idst)
           returns ("swish_tile_init();", "swish_tile({idst});")
        -> update_macro_defines(SWISH, defines)
           sets SFPU_OP_SWISH_INCLUDE=1
        -> compile compute kernel with defines:
           SFPU_OP_CHAIN_0 -> "SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0"
           SFPU_OP_CHAIN_0_INIT_0 -> "swish_tile_init();"
           SFPU_OP_CHAIN_0_FUNC_0 -> "swish_tile(idst);"
           SFPU_OP_SWISH_INCLUDE -> "1"
```

### Approximation mode
`get_op_approx_mode(UnaryOpType::SWISH)` falls through to default, returning `false`. The `APPROXIMATION_MODE` template parameter is always `false` but is unused in the kernel body.

## 8. Numerical Properties

| Property | Value |
|----------|-------|
| **Output range** | `(-inf, +inf)` (approaches `x` for large `x`, approaches `0` for large negative `x`) |
| **ULP error (bfloat16)** | ~2 ULP (test threshold) |
| **ULP error (float32)** | ~3 ULP (test threshold) |
| **Allclose rtol (bfloat16)** | 1.6e-2 |
| **Allclose atol (bfloat16)** | 1e-2 |
| **Allclose rtol (float32)** | 1e-3 |
| **Allclose atol (float32)** | 1e-4 |
| **Subnormal handling** | Hardware flushes subnormals to zero; test replicates this |
| **Special values** | NaN/Inf filtered from comparison; hardware produces finite results for finite inputs |

## 9. Key Implementation Patterns for Reuse

### Pattern: Piecewise function approximation
The swish kernel demonstrates the canonical pattern for approximating a smooth nonlinear function on the SFPU:

1. **Fold to positive domain**: Use `sfpi::abs(x)` to compute on `|x|` only
2. **Core segment (polynomial)**: Use Horner-form evaluation for the main range where the function has the most curvature
3. **Transition segment (linear)**: Linear interpolation for the transition region
4. **Saturation segment**: Clamp to known asymptotic value
5. **Unfold**: Apply symmetry/antisymmetry to recover the negative-x result
6. **Final transform**: Multiply by `x` (or apply whatever final operation combines sigmoid with input)

### Pattern: No-parameter SFPU operation wiring
The swish operation is the simplest wiring pattern:
- No runtime parameters
- No custom init callback
- No LUT setup
- Standard `REGISTER_UNARY_OPERATION` macro (no `_WITH_FLOAT_PARAMETER` variant)
- Standard split-include guard (`SFPU_OP_SWISH_INCLUDE`)
- Dispatch via `get_op_init_and_func_default` (not parameterized)

### Pattern: Conditional override (v_if cascade)
The three `v_if` blocks form a cascade of overrides:
```
default: polynomial result
override 1: if |x| > 2.5, replace with linear
override 2: if |x| > 5.0, replace with 1.0
override 3: if x < 0, flip via 1 - sig
```
This is efficient because later `v_if` blocks overwrite earlier results for the affected lanes, avoiding complex nested conditionals.

## 10. Relationship to SiLU

Swish and SiLU are mathematically identical:
- **swish(x)** = x * sigmoid(x)
- **SiLU(x)** = x * sigmoid(x)

PyTorch uses `torch.nn.functional.silu()` for this function. The golden function in `unary.py` correctly uses `silu`. The codebase maintains both `ttnn.swish` and `ttnn.silu` as separate `UnaryOpType` entries (`SWISH` and `SILU`), but both compute the same mathematical function. The SFPU kernel implementations may differ (SILU likely uses a different approximation strategy).

## 11. Summary Table

| Layer | File | Key symbol |
|-------|------|-----------|
| SFPU kernel | `ckernel_sfpu_swish.h` | `calculate_swish<APPROX, ITER>()` |
| LLK wrapper | `llk_math_eltwise_unary_sfpu_swish.h` | `llk_math_eltwise_unary_sfpu_swish<APPROX>(dst_index, vector_mode)` |
| LLK init | `llk_math_eltwise_unary_sfpu_swish.h` | `llk_math_eltwise_unary_sfpu_swish_init<APPROX>()` |
| Compute API | `swish.h` | `swish_tile(idst)`, `swish_tile_init()` |
| Split include | `sfpu_split_includes.h` | `SFPU_OP_SWISH_INCLUDE` |
| SfpuType enum | `llk_sfpu_types.h` | `SfpuType::swish` |
| UnaryOpType enum | `unary_op_types.hpp` | `UnaryOpType::SWISH` |
| Macro define | `unary_op_utils.cpp` | `"SFPU_OP_SWISH_INCLUDE"` |
| Init+func strings | `unary_op_utils.cpp` | `"swish_tile_init();"` / `"swish_tile({idst});"` |
| C++ API | `unary.hpp` | `REGISTER_UNARY_OPERATION(swish, SWISH)` |
| Python binding | `unary_nanobind.cpp` | `bind_unary_operation<"swish", &ttnn::swish>` |
| Golden function | `unary.py` | `torch.nn.functional.silu(x)` |
| Test | `test_swish.py` | `test_swish(device, is_fp32)` |
