# Reference Selection for RReLU

## Target Operation

**RReLU** (Randomized Rectified Linear Unit)

```
f(x) = x          if x >= 0
f(x) = a * x      if x < 0

Eval mode:  a = (lower + upper) / 2
Train mode: a ~ Uniform(lower, upper) per element

Parameters: lower (default=0.125), upper (default=1/3), training (bool, default=False)
```

## Math Decomposition

RReLU decomposes into these primitive sub-operations:
1. **Sign-based conditional**: branch on `x >= 0` vs `x < 0` (like relu, leaky_relu)
2. **Scalar parameter computation**: `a = (lower + upper) / 2` in eval mode
3. **Scalar-element multiplication**: `a * x` for the negative branch
4. **Two float parameters** passed through dispatch: `lower` and `upper`

The SFPU kernel (eval mode) reduces to:
```
result = x                         if x >= 0
result = ((lower + upper) / 2) * x  if x < 0
```

## Selection Criteria

Operations were ranked by:
1. **Structural similarity** to RReLU's sign-based conditional pattern
2. **Parameter passing pattern** (scalar params through runtime args / compile-time defines)
3. **Completeness** of surviving implementation (ckernel + LLK + dispatch + bindings + test)
4. **SFPI instruction usage** relevant to RReLU (v_if, dst_reg, vFloat multiply)

## Top 5 Reference Operations

### 1. SWISH (Highest Relevance)

**Math**: `swish(x) = x * sigmoid(x)`

**Why selected**: Best end-to-end reference for SFPU kernel structure. The kernel uses `v_if(x < 0.0f)` — the exact conditional pattern RReLU needs. Complete surviving pipeline: ckernel SFPU kernel, LLK wrapper, dispatch via `SFPU_OP_SWISH_INCLUDE`, nanobind binding, and exhaustive test. Demonstrates all SFPI primitives needed: `sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`, arithmetic operators.

**Key files**:
- SFPU kernel: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
- LLK wrapper: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
- Dispatch (old): `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` (line 51: `swish_tile_init/swish_tile`)
- Dispatch (ng): `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` (line 89)
- Registration: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` (line 163: `REGISTER_UNARY_OPERATION(swish, SWISH)`)
- Nanobind: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` (line 1824)
- Python golden: `ttnn/ttnn/operations/unary.py` (line 50)
- Test: `tests/ttnn/unit_tests/operations/eltwise/test_swish.py`

**What to learn**:
- SFPI kernel template: `template <bool APPROXIMATION_MODE, int ITERATIONS = 8> inline void calculate_swish()`
- Iteration loop with `#pragma GCC unroll 8` and `sfpi::dst_reg++`
- Sign-based conditional: `v_if(x < 0.0f) { ... } v_endif;`
- Custom SFPU include macro: `SFPU_OP_SWISH_INCLUDE`
- LLK wrapper calling `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(...)`

---

### 2. HARDSHRINK (High Relevance)

**Math**: `hardshrink(x, lambda) = x * 1(x + lambda < 0) + x * 1(x - lambda > 0)`

**Why selected**: Best reference for parameterized conditional operation. Shows how scalar parameters are packed and passed through runtime args (`pack_scalar_runtime_arg`), how the program factory allocates a temporary CB (`cb_tmp0` / `CBIndex::c_1`), and how custom compute kernels use conditional tile operations (`ltz_tile`, `gtz_tile`, `mul_binary_tile`). Although it uses old-style binary tile ops instead of SFPI, the parameter-passing pipeline through the program factory is essential for RReLU (which needs `lower` and `upper`).

**Key files**:
- Compute kernel (SFPU variant): `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel_sfpu.cpp`
- Compute kernel (binary variant): `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel.cpp`
- Program factory (old): `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` (lines 70, 129-131)
- Program factory (ng): `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/unary_ng_program_factory.cpp` (lines 45, 54)
- Registration: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` (line 166: `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(hardshrink, HARDSHRINK)`)

**What to learn**:
- Scalar parameter packing: `pack_scalar_runtime_arg(op, 0, input_dtype)`
- Runtime arg unpacking in kernel: `get_arg_val<uint32_t>(0)` + `reinterpret_cast<const float*>(&packed_scalar)`
- Custom compute kernel path (different from standard `eltwise_sfpu.cpp`)
- Temporary CB allocation for intermediate results: `needs_tmp0_cb()` pattern
- Conditional tile operations: `ltz_tile()`, `gtz_tile()`, `fill_tile()`, `mul_binary_tile()`

---

### 3. FRAC (Medium-High Relevance)

**Math**: `frac(x) = x - trunc(x)`

**Why selected**: Clean, complete SFPU kernel demonstrating conditional SFPI patterns with `v_if`/`v_endif`. Shows bit-level manipulation with `sfpi::exexp()`, `sfpi::reinterpret<>()`, and `sfpi::setexp()`. The conditional structure (`v_if(exp < 0)`, `v_if(exp >= 0 && exp < 23)`) demonstrates how to handle multi-branch logic in SFPI. Complete pipeline with custom include macro.

**Key files**:
- SFPU kernel: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h`
- LLK wrapper: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`
- Dispatch: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` (line 49)
- Registration: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` (line 154)
- Test: `tests/ttnn/unit_tests/operations/eltwise/test_frac.py`

**What to learn**:
- Simpler SFPI kernel structure (no sigmoid approximation, just conditionals)
- SFPI bit manipulation intrinsics for reference
- Custom `SFPU_OP_FRAC_INCLUDE` macro pattern
- Clean `calculate_frac<APPROXIMATION_MODE, ITERATIONS>()` template

---

### 4. SINH (Medium Relevance)

**Math**: `sinh(x) = (exp(x) - exp(-x)) / 2`

**Why selected**: Complete SFPU kernel showing helper function composition (`exp_21f`), conditional override for small values (`v_if(abs_x < v_half)`), and bfloat16 rounding via `sfpi::float_to_fp16b()`. Demonstrates a custom `sinh_init()` function (though empty). The `#pragma GCC unroll 0` usage (vs `#pragma GCC unroll 8` in swish) shows when to disable unrolling for larger kernels.

**Key files**:
- SFPU kernel: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
- LLK wrapper: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
- Dispatch: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` (line 52)
- Registration: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` (line 116)
- Test: `tests/ttnn/unit_tests/operations/eltwise/test_sinh.py`

**What to learn**:
- Helper function pattern (`exp_21f<APPROXIMATION_MODE>()`)
- Conditional override for numerical edge cases
- bfloat16 explicit rounding: `sfpi::float_to_fp16b(y, 0)`
- Custom init function template: `template <bool APPROXIMATION_MODE> inline void sinh_init()`

---

### 5. ATANH (Medium Relevance)

**Math**: `atanh(x) = 0.5 * ln((1+x)/(1-x))`

**Why selected**: Demonstrates programmable constant registers (`sfpi::vConstFloatPrgm0/1/2`) in the init function — a mechanism for precomputing values once and using them across all iterations. RReLU could use this to precompute `(lower + upper) / 2` in the init function rather than computing it per-element. Also shows the full ckernel + LLK + dispatch pipeline.

**Key files**:
- SFPU kernel: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
- LLK wrapper: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
- Dispatch: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` (line 51)
- Registration: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` (line 136)

**What to learn**:
- Programmable constant registers: `sfpi::vConstFloatPrgm0 = value` in init
- Using precomputed constants in kernel body: `sfpi::vConstFloatPrgm0`
- Init function with register programming: `template <bool APPROXIMATION_MODE> inline void atanh_init()`
- Complete LLK init wrapper calling the init function

---

## Implementation Strategy Notes

Based on the reference analysis, the recommended approach for RReLU:

1. **Kernel structure**: Follow SWISH pattern — simple `calculate_rrelu()` template with `v_if(x < 0.0f)` conditional
2. **Parameter passing**: Follow HARDSHRINK pattern for passing `lower` and `upper` as packed scalar runtime args. Or follow ATANH pattern to precompute `alpha = (lower + upper) / 2` in the init function using `vConstFloatPrgm0`
3. **Registration**: Use `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER` or a custom signature with two float params (like `hardtanh` at `unary.hpp:282`)
4. **Include macro**: Create `SFPU_OP_RRELU_INCLUDE` following SWISH/FRAC/SINH pattern
5. **Test**: Follow `test_swish.py` pattern with exhaustive bfloat16 bitpattern testing

## Codebase State Notes

This is a deep-nuked evaluation branch. The following are **missing/gutted**:
- All relu family SFPU kernels (relu, leaky_relu, prelu, relu_max, relu_min, elu, selu, celu)
- All exp/sigmoid/tanh primitives (`ckernel_sfpu_exp.h`, `ckernel_sfpu_sigmoid.h`, etc.)
- All log/recip primitives
- All rounding primitives

The following **survive intact** and can be used as references:
- SFPI instruction set (`sfpi.h`) — the raw building blocks
- 4 custom SFPU kernels: swish, frac, sinh, atanh (complete pipelines)
- HARDSHRINK compute kernels (old-style binary tile ops, not SFPI)
- Standard compute kernel API: comparison tiles (ltz_tile, gtz_tile), binary tiles, fill_tile
- Full dispatch infrastructure (program factories, nanobind, Python)
