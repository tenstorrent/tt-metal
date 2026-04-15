# Reference Operation Selection for softcap

## Target Operation
- **Name**: softcap
- **Definition**: `cap * tanh(x / cap)` where `cap` is a float parameter
- **Component operations identified**: scalar division (multiply by `1/cap`), tanh (transcendental), scalar multiply (by `cap`), float parameter passing to SFPU kernel

## Selected References (ranked by relevance)

### 1. sinh
- **Why selected**: `sinh` is the most recently implemented custom hyperbolic SFPU operation in this codebase and traverses the complete 5-layer implementation stack that `softcap` must follow: `ckernel_sfpu_sinh.h` (SFPU microcode) → `llk_math_eltwise_unary_sfpu_sinh.h` (LLK wrapper) → `llk_math_eltwise_unary_sfpu_params.h` (params dispatch) → `api/compute/eltwise_unary/sinh.h` (tile API) → `sfpu_split_includes.h` (conditional include guard `SFPU_OP_SINH_INCLUDE`) → `eltwise_sfpu.cpp` (compute kernel via `SFPU_OP_CHAIN_0`). It also uses the `exp_21f` helper that a tanh implementation would reuse, and shows how to register a new op in `unary_op_utils.cpp` (`get_op_init_and_func_default`, `get_macro_definition`, `UnaryOpType::SINH`). The architecture of softcap's SFPU ckernel — a vectorized loop over `dst_reg` elements with `#pragma GCC unroll 0` and an early-exit path for numerical stability — should directly match sinh's pattern.
- **Relevance**: high — entire implementation stack is directly reusable as a template; softcap follows the same registration and dispatch path

### 2. atanh
- **Why selected**: `atanh` is the other recently implemented hyperbolic function sharing the exact same 5-layer stack as sinh, but additionally shows how to use `vConstFloatPrgm0/1/2` in `atanh_init()` to preload polynomial coefficients into SFPU programmable registers before the compute loop. Softcap needs `1/cap` (a runtime float parameter) available inside the SFPU microcode loop. The `atanh_init<APPROXIMATE>()` function registered in `llk_math_eltwise_unary_sfpu_atanh_init` demonstrates the precise mechanism for precomputing scalar constants into `vConstFloatPrgm`. Additionally, atanh's SFPU code uses `sfpi::exexp`, `sfpi::setexp`, and `int32_to_float` for IEEE 754 decomposition — techniques that may appear in an accurate tanh implementation.
- **Relevance**: high — demonstrates float parameter preloading via `vConstFloatPrgm*` in the init function, and shows another hyperbolic op's complete SFPU implementation

### 3. tanhshrink
- **Why selected**: `tanhshrink` (both `tanhshrink_sfpu_kernel.cpp` and `tanhshrink_kernel.cpp`) directly calls `tanh_tile_init()` and `tanh_tile()` — the exact LLK API calls that softcap's compute kernel must use for the tanh step. These compute kernel files are the only ones in the repository that explicitly call `tanh_tile_init()` and `tanh_tile()`, making them the essential reference for how tanh is invoked at the kernel level. `tanhshrink_sfpu_kernel.cpp` also shows the `SFPU_OP_CHAIN_0`-based approach (matching `eltwise_sfpu.cpp`), while `tanhshrink_kernel.cpp` shows a dedicated custom kernel that applies tanh then a binary subtract — the same pattern softcap needs (apply tanh, then scale by scalar).
- **Relevance**: high — direct usage of `tanh_tile_init()` and `tanh_tile()` which softcap must call in its compute kernel

### 4. hardtanh
- **Why selected**: `hardtanh` is one of only two operations in `is_parametrized_type()` (the other being `softshrink`), and it is the operation most semantically similar to softcap — it is a tanh-bounding operation that clips activations. It is registered in `unary.hpp` with two float parameters (`min_val`, `max_val`) passed via `UnaryWithParam{UnaryOpType::HARDTANH, min_val, max_val}`. Its `ckernel_sfpu_hardtanh.h` shows the `s2vFloat16b(param)` technique for loading FP16_B-encoded scalar parameters into SFPU vector registers, and the `_calculate_hardtanh_` function signature `(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)` shows the exact parameter passing convention for the parameterized SFPU kernel path through `get_op_init_and_func_parameterized`. Softcap needs this same single-float-parameter pattern.
- **Relevance**: high — shows the parameterized op registration, float-to-uint32 encoding, and `s2vFloat16b` usage inside SFPU for scalar parameters

### 5. swish
- **Why selected**: `swish` computes `x * sigmoid(x)`, structurally analogous to `cap * tanh(x * (1/cap))` — both are "scalar × f(scaled_input)" patterns. `ckernel_sfpu_swish.h` shows how to hold the input `x` in an SFPU register, compute a multi-step piecewise function over it, then multiply the result back by `x` (or a scalar). The piecewise approximation with breakpoints and `v_if/v_elseif/v_endif` conditionals is a useful reference for implementing tanh via piecewise polynomial if hardware tanh is not available. Swish is also registered via the `SFPU_OP_SWISH_INCLUDE` macro path in `sfpu_split_includes.h` and `get_macro_definition()`, showing another macro-guarded include registration that softcap must replicate.
- **Relevance**: medium — structural analogy of scalar-times-transformed-input, piecewise-conditional SFPU pattern, and macro-guarded include registration

## Implementation Notes

The full stack for softcap should be:
1. `ckernel_sfpu_softcap.h` — SFPU microcode implementing `cap * tanh(x / cap)`, loading `1/cap` and `cap` via `vConstFloatPrgm*` in the init function (see atanh) or via `s2vFloat16b(param)` (see hardtanh)
2. `llk_math_eltwise_unary_sfpu_softcap.h` — LLK wrapper calling `_llk_math_eltwise_unary_sfpu_params_` (see sinh/atanh LLK headers)
3. `api/compute/eltwise_unary/softcap.h` — tile API exposing `softcap_tile_init()` and `softcap_tile()` (see sinh.h / atanh.h)
4. `sfpu_split_includes.h` — add `#if SFPU_OP_SOFTCAP_INCLUDE` guard (see existing entries)
5. `unary_op_utils.cpp` — register in `get_op_init_and_func_parameterized`, `get_macro_definition`, and add `UnaryOpType::SOFTCAP` to `unary_op_types.hpp`
6. `unary_program_factory.cpp` — add `case UnaryOpType::SOFTCAP:` to pack the `cap` parameter as a runtime arg (see HARDSHRINK pattern, but single param)
7. `unary.hpp` — expose `softcap(tensor, cap)` using `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER` macro
