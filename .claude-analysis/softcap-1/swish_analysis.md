# SFPU Kernel Analysis: swish

## 1. Math Definition

```
swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

This is equivalent to PyTorch's `torch.nn.functional.silu(x)`. The operation is parameterless (no runtime scalars).

## 2. File Inventory

| Layer | File | Purpose |
|-------|------|---------|
| SFPU kernel (Blackhole) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` | Core SFPI math: `calculate_swish<>()` |
| SFPU kernel (Wormhole) | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` | Identical copy for Wormhole B0 |
| LLK wrapper (Blackhole) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` | `llk_math_eltwise_unary_sfpu_swish()` + `_init()` |
| LLK wrapper (Wormhole) | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` | Identical copy for Wormhole B0 |
| Compute API header | `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h` | `swish_tile()` / `swish_tile_init()` |
| Conditional include guard | `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | `SFPU_OP_SWISH_INCLUDE` gate |
| SfpuType enum (Blackhole) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` | `SfpuType::swish` |
| SfpuType enum (Wormhole) | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` | `SfpuType::swish` |
| UnaryOpType enum | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | `UnaryOpType::SWISH` (line 126) |
| Op dispatch (legacy) | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Maps `SWISH` to macro `SFPU_OP_SWISH_INCLUDE` and to `swish_tile_init()`/`swish_tile(idst)` |
| Op dispatch (unary_ng) | `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` | Same mapping for next-gen unary path |
| C++ API registration | `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` | `REGISTER_UNARY_OPERATION(swish, SWISH)` (line 163) |
| Python nanobind | `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` | `bind_unary_operation<"swish", &ttnn::swish>(...)` |
| Python golden function | `ttnn/ttnn/operations/unary.py` | `_golden_function_swish` using `torch.nn.functional.silu` |
| Compute kernel entry | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | Shared unary SFPU compute kernel dispatching via `SFPU_OP_CHAIN_0` |
| Test | `tests/ttnn/unit_tests/operations/eltwise/test_swish.py` | Exhaustive bfloat16 bitpattern + float32 test |

## 3. SFPU Kernel Deep Dive

### 3.1 Function Signature

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_swish();
```

- **APPROXIMATION_MODE**: Template parameter threaded from `APPROX` compile-time constant. Not used to select between different code paths in this kernel (both modes use the same polynomial approximation).
- **ITERATIONS**: Default 8, corresponding to 8 face-rows of a 32x32 tile (each iteration processes one row of 16 elements via SIMD).

### 3.2 Algorithm: Hybrid Polynomial + Piecewise-Linear Sigmoid Approximation

The kernel does **not** use hardware `exp` or `sigmoid` primitives. Instead, it approximates `sigmoid(|x|)` using a 3-segment piecewise function over `|x|`, then reconstructs `sigmoid(x)` for negative inputs, and finally multiplies by `x`.

#### Segment 0: Polynomial for |x| <= 2.5

```
sigmoid(t) ~ 0.5 + t * (0.2533 + t * (-0.01479 + t * (-0.00747)))
```

A degree-3 Horner-form polynomial fitted to minimize max error over [0, 2.5]. Evaluated in 4 SFPU multiply-add operations (Horner scheme). Max error ~0.007 at t ~ 2.0.

#### Segment 1: Linear for 2.5 < |x| <= 5.0

```
sigmoid(t) ~ 0.0276 * t + 0.855
```

A linear interpolation. Max error ~0.017 at t ~ 4.0.

#### Segment 2: Saturation for |x| > 5.0

```
sigmoid(t) = 1.0
```

Uses `sfpi::vConst1` for exact 1.0. Max error ~0.007 at t = 5.0.

#### Negative input handling

```
sigmoid(x) = 1 - sigmoid(|x|)   for x < 0
```

Uses the identity that sigmoid is antisymmetric around 0.5.

#### Final computation

```
swish(x) = x * sigmoid(x)
```

One final SFPU multiply.

### 3.3 SFPI Instructions Used

| SFPI Construct | Usage | Count per iteration |
|----------------|-------|---------------------|
| `sfpi::dst_reg[0]` (load) | Read input tile element | 1 |
| `sfpi::abs()` | Compute `|x|` | 1 |
| `vFloat` arithmetic (`*`, `+`) | Horner polynomial, linear segment, final multiply | ~8 |
| `sfpi::vConst1` | Constant 1.0f | 2 uses |
| `v_if` / `v_endif` | Predicated execution for segment selection and sign handling | 3 conditional blocks |
| `sfpi::dst_reg[0] =` (store) | Write result | 1 |
| `sfpi::dst_reg++` | Advance to next face-row | 1 |

### 3.4 Register Pressure

- **vFloat registers**: 3 live simultaneously (`x`, `ax`, `sig_pos`)
- **No vInt registers** used
- **No LUT/LREG** usage — all constants are float literals loaded as immediates
- Register pressure is moderate: well within the SFPU's ~8 vFloat register budget

### 3.5 Control Flow

Three `v_if`/`v_endif` blocks implement predicated SIMD lanes:

1. `v_if(ax > 2.5f)` — override polynomial with linear segment
2. `v_if(ax > 5.0f)` — override with saturation to 1.0
3. `v_if(x < 0.0f)` — flip sigmoid for negative inputs

These are nested in a fallthrough pattern (not truly nested — each `v_if` overwrites `sig_pos` if the condition holds, so the last matching segment wins). The ordering is critical: segment 0 is the default, segment 1 overrides for mid-range, segment 2 overrides for large |x|.

### 3.6 Loop Structure

```cpp
#pragma GCC unroll 8
for (int d = 0; d < ITERATIONS; d++) {
    // ... compute ...
    sfpi::dst_reg++;
}
```

The loop is fully unrolled with `#pragma GCC unroll 8` for the default 8 iterations. Each iteration processes one face-row (16 elements) via SIMD, covering all 8 face-rows of a 32x16 half-tile (or equivalent portion).

## 4. Dispatch Chain

### 4.1 How `SFPU_OP_CHAIN_0` is Constructed

At compile time, the host-side `unary_op_utils.cpp` generates defines like:

```cpp
// From get_macro_definition():
case UnaryOpType::SWISH: return "SFPU_OP_SWISH_INCLUDE";

// From get_op_init_and_func_default():
case UnaryOpType::SWISH:
    return {"swish_tile_init();", fmt::format("swish_tile({});", idst)};
```

These are injected as compile-time defines:
- `SFPU_OP_SWISH_INCLUDE=1` — gates `#include "api/compute/eltwise_unary/swish.h"` in `sfpu_split_includes.h`
- `SFPU_OP_CHAIN_0` — set to `swish_tile_init(); swish_tile(0);` (the init + compute calls)

### 4.2 Full Call Chain

```
SFPU_OP_CHAIN_0 (in eltwise_sfpu.cpp)
  -> swish_tile_init()                              [swish.h]
       -> llk_math_eltwise_unary_sfpu_swish_init()  [llk_math_eltwise_unary_sfpu_swish.h]
            -> llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROX>()
  -> swish_tile(idst)                               [swish.h]
       -> llk_math_eltwise_unary_sfpu_swish(idst)   [llk_math_eltwise_unary_sfpu_swish.h]
            -> _llk_math_eltwise_unary_sfpu_params_<APPROX>(
                   calculate_swish<APPROX, 8>, dst_index, vector_mode)
                 -> calculate_swish<APPROX, 8>()    [ckernel_sfpu_swish.h]
```

### 4.3 Compute Kernel Flow

The shared `eltwise_sfpu.cpp` kernel:
1. Calls `init_sfpu(c_0, c_2)` once at startup
2. For each block: reserves output CB `c_2`, then per tile:
   - `tile_regs_acquire()` — lock DST registers
   - `cb_wait_front(c_0, 1)` — wait for input tile
   - `copy_tile(c_0, 0, 0)` — copy input to DST
   - `SFPU_OP_CHAIN_0` — execute `swish_tile_init(); swish_tile(0);`
   - `tile_regs_commit()` / `tile_regs_wait()` — hand off DST to packer
   - `pack_tile(0, c_2)` — pack result to output CB
   - `cb_pop_front(c_0, 1)` — release input tile
   - `tile_regs_release()`
3. Pushes back completed output block

## 5. Parameters

**No runtime parameters.** Swish is a parameterless operation — no scalars are passed via runtime args or compile-time args beyond the standard `per_core_block_cnt` and `per_core_block_dim`.

The `UnaryOpType::SWISH` enum uses `REGISTER_UNARY_OPERATION(swish, SWISH)` (no `_WITH_FLOAT_PARAMETER` variant).

## 6. Accuracy Profile

| Metric | bfloat16 | float32 |
|--------|----------|---------|
| ULP threshold | 2 | 3 |
| rtol | 1.6e-2 | 1e-3 |
| atol | 1e-2 | 1e-4 |
| Max sigmoid approx error | ~0.017 (at |x| ~ 4.0) | same |
| Test coverage | All 65536 bfloat16 bitpatterns | All bf16 bitpatterns cast to f32 |

The polynomial approximation introduces the dominant error source. The piecewise-linear segment (|x| in [2.5, 5.0]) has the highest individual error (~0.017 for sigmoid), but since `swish(x) = x * sigmoid(x)` and `sigmoid` is close to 1.0 in that range, the multiplicative effect on the final result is bounded.

Subnormal values are flushed to zero on both input and output to match hardware behavior.

## 7. Cross-Architecture Notes

- The `ckernel_sfpu_swish.h` files for Blackhole and Wormhole B0 are **identical** — same algorithm, same constants, same code.
- The `llk_math_eltwise_unary_sfpu_swish.h` wrappers are also **identical** across architectures.
- Both architectures use `SfpuType::swish` in their respective `llk_sfpu_types.h` enum.

## 8. Relationship to SILU

`swish(x)` with beta=1 is identical to `silu(x) = x * sigmoid(x)`. The Python golden function confirms this: `_golden_function_swish` calls `torch.nn.functional.silu()`. The codebase has a separate `UnaryOpType::SILU` enum entry (line 74 in `unary_op_types.hpp`), but `SWISH` provides its own independent SFPU implementation via the polynomial approximation rather than reusing the SILU path.

## 9. Key Patterns for Reuse

1. **Piecewise approximation pattern**: The 3-segment approach (polynomial + linear + saturation) is a reusable template for any bounded monotonic function like sigmoid, tanh, etc.
2. **Symmetry exploitation**: Computing `f(|x|)` then correcting for negative `x` via `1 - f(|x|)` halves the approximation domain.
3. **No LUT/LREG usage**: All constants are float literals — no setup overhead for LUT loads.
4. **Conditional include guard**: The `SFPU_OP_SWISH_INCLUDE` pattern ensures the kernel header is only compiled when needed, reducing compile time for other unary ops.
5. **No runtime parameters**: Simplest dispatch pattern — no scalar packing into compile-time or runtime args.
