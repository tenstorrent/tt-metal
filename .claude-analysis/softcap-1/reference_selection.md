# Reference Operation Selection for softcap

## Target Operation
- **Name**: softcap
- **Definition**: `cap * tanh(x / cap)` where `cap` is a positive float scalar parameter (default 50.0)
- **Component operations identified**:
  - Division by scalar parameter `cap` (equivalently: multiply by `1/cap`)
  - `tanh` function — the core nonlinearity (NOTE: `ckernel_sfpu_tanh.h` was deleted by DEEP_NUKE_MANIFEST.md; implementor must write tanh from raw SFPI)
  - Multiplication by scalar parameter `cap`
  - Parameter handling: a single float scalar (`cap`) must be passed at runtime and used both as divisor and multiplier

## Context: What Was Nuked

Per `DEEP_NUKE_MANIFEST.md` (Phase 1), `ckernel_sfpu_tanh.h` was deleted along with exp, sigmoid, and other family primitives. The implementor must write a tanh approximation from raw SFPI instructions. The four Wave 3 generated ops — `swish`, `atanh`, `sinh`, `frac` — represent the surviving custom SFPU kernels that demonstrate exactly how to write new ckernel_sfpu implementations from scratch.

## Selected References (ranked by relevance)

### 1. swish
- **Why selected**: `swish(x) = x * sigmoid(x)` is the closest structural analog to softcap. Both are composite nonlinearities that could not use pre-existing primitives (sigmoid/tanh nuked), so swish was implemented as a full custom `ckernel_sfpu_swish.h` with polynomial piecewise approximation from raw SFPI. The file demonstrates the exact template structure `template <bool APPROXIMATION_MODE, int ITERATIONS = 8>`, the use of `sfpi::dst_reg[0]`, `v_if`/`v_endif` for piecewise logic, and the pattern of loading constants as `constexpr float`. It also shows the full dispatch chain: `ckernel_sfpu_swish.h` → `llk_math_eltwise_unary_sfpu_swish.h` → `SFPU_OP_SWISH_INCLUDE` in `get_macro_definition()` → `swish_tile_init()`/`swish_tile()` pair → `REGISTER_UNARY_OPERATION(swish, SWISH)` in `unary.hpp`.
- **Relevance**: HIGH — exact implementation pattern to follow; softcap needs the same custom-kernel structure since tanh primitives are absent

### 2. tanhshrink
- **Why selected**: `tanhshrink(x) = x - tanh(x)` directly uses `tanh_tile()` at the compute kernel layer (see `tanhshrink_sfpu_kernel.cpp`). It shows the pattern for a specialized compute kernel (not `eltwise_sfpu.cpp`) that calls `tanh_tile_init()` and `tanh_tile()`. The `tanhshrink_sfpu_kernel.cpp` version shows how to use `SFPU_OP_CHAIN` macros while `tanhshrink_kernel.cpp` shows the direct `tanh_tile`/`binary_dest_reuse_tiles` pattern. Crucially, this is the only surviving example of a kernel that directly calls the tanh tile function, making it the essential reference for how tanh is consumed at the compute kernel level.
- **Relevance**: HIGH — shows the tanh_tile_init/tanh_tile call pattern and how to build a kernel that uses tanh as a sub-operation; softcap needs to do the same

### 3. atanh
- **Why selected**: `atanh(x) = 0.5 * ln((1+x)/(1-x))` is implemented in `ckernel_sfpu_atanh.h` as a custom SFPU kernel that computes a logarithm via IEEE 754 decomposition with cubic polynomial approximation. The `atanh_init()` function loads polynomial coefficients into `vConstFloatPrgm0/1/2` registers — this is the canonical pattern for storing approximation constants that are used across all SFPU iterations. Since tanh approximation typically uses a polynomial or rational approximation with similar initialization, the `atanh_init()` + `vConstFloatPrgm` pattern is directly applicable. Additionally, atanh is the mathematical inverse of tanh, so the approximation techniques are intimately related.
- **Relevance**: HIGH — `vConstFloatPrgm` init pattern and polynomial approximation structure are directly reusable; shows IEEE 754 exponent/mantissa manipulation via `sfpi::exexp`/`sfpi::setexp`

### 4. sinh
- **Why selected**: `sinh(x) = (exp(x) - exp(-x)) / 2` is implemented in `ckernel_sfpu_sinh.h` and contains the `exp_21f<APPROXIMATION_MODE>()` helper function that computes `2^z` using the Moroz et al. 2022 algorithm. This exp-via-addexp approach is the only surviving SFPU exponential implementation in the codebase after the nuke. Since tanh can be computed as `(exp(2x)-1)/(exp(2x)+1)` or `1 - 2/(exp(2x)+1)`, the `exp_21f` helper can be directly reused inside a softcap implementation. The sinh kernel also demonstrates the piecewise/Taylor fallback pattern for small values, which is relevant for numerical stability.
- **Relevance**: HIGH — `exp_21f` is the only remaining exp primitive; softcap's tanh implementation will likely depend on it

### 5. hardtanh
- **Why selected**: `hardtanh(x) = clamp(x, min_val, max_val)` is the primary example of a **parameterized SFPU op** in the codebase (listed in `is_parametrized_type()` in `unary_op_utils.hpp`). The `ckernel_sfpu_hardtanh.h` implementation shows how to load multiple scalar parameters via `sfpi::s2vFloat16b(param0/1/2)` into vFloat vectors, and how to pass these through the runtime dispatch. The `hardtanh` function in `unary.hpp` shows the `UnaryWithParam{UnaryOpType::HARDTANH, min_val, max_val}` constructor with two float params. Softcap needs to pass `cap` as a float parameter, and while it only has one parameter (unlike hardtanh's two), the dispatch mechanism and parameter packing through `get_op_init_and_func_parameterized()` is the same pattern softcap must follow.
- **Relevance**: MEDIUM — scalar parameter passing pattern is essential for softcap; `is_parametrized_type` registration and `get_op_init_and_func_parameterized()` dispatch are the exact hooks softcap needs
