# Reference Operation Selection for softcap

## Target Operation
- **Name**: softcap
- **Definition**: cap * tanh(x / cap)
- **Component operations identified**:
  - Scalar divide: `x / cap` (equivalently `x * (1.0f / cap)`)
  - Nonlinear function: `tanh(...)` applied to scaled input
  - Scalar multiply: `cap * result`
  - One configurable float parameter: `cap`

## Selected References (ranked by relevance)

### 1. swish
- **Why selected**: Swish (`x * sigmoid(x)`) is the closest structural match to softcap. Both are composite operations that apply a nonlinear function and then multiply by a scalar (swish multiplies by x itself; softcap multiplies by cap). The full implementation stack is present in this codebase: `ckernel_sfpu_swish.h` contains `calculate_swish<>` with the inner SFPU loop; `llk_math_eltwise_unary_sfpu_swish.h` wraps it via `_llk_math_eltwise_unary_sfpu_params_`; and `api/compute/eltwise_unary/swish.h` exposes `swish_tile()` / `swish_tile_init()`. The macro-based registration in `unary_op_utils.cpp` (`SFPU_OP_SWISH_INCLUDE`, `swish_tile_init()`, `swish_tile(idst)`) gives the complete blueprint for adding softcap to the same infrastructure.
- **Relevance**: high — provides the complete layer-by-layer template (ckernel → llk → api → utils registration) for a composite unary SFPU op; the pattern `vFloat y = nonlinear(x); dst_reg[0] = scalar * y;` in the inner loop directly maps to softcap's structure.

### 2. sinh
- **Why selected**: Sinh (`(exp(x) - exp(-x)) / 2`) demonstrates the critical pattern of bookending a nonlinear computation with scalar multiplications inside the SFPU loop. The implementation in `ckernel_sfpu_sinh.h` shows: (a) a compile-time scalar constant (`v_half = 0.5f`) applied as a multiply at the end of the computation — exactly how `cap` will be applied as the final multiply in softcap; (b) a scalar applied to the input before the nonlinear step (`x * log2e`) — exactly how `1/cap` will be applied to scale x before tanh; (c) use of the `exp_21f` helper template. Also shows the `sinh_init()` pattern (no programmable constants needed), which softcap may adopt or adapt.
- **Relevance**: high — two scalar multiplies bracketing a nonlinear function (one pre-nonlinear, one post) is the exact computation shape of softcap; the compile-time constant approach in sinh informs whether cap should be a compile-time or runtime arg.

### 3. atanh
- **Why selected**: Atanh (`0.5 * (ln(1+x) - ln(1-x))`) is the most important reference for scalar parameter passing via `vConstFloatPrgm` registers. The `atanh_init<>()` function loads polynomial coefficients into `vConstFloatPrgm0/1/2` — this is the pattern for loading `cap` and `1/cap` into programmable constant registers before the main SFPU loop. The compute loop then reads those registers directly via `sfpi::vConstFloatPrgm0/1/2`. Additionally, atanh ends with a scalar multiply (`* 0.5f`), reinforcing the scalar-multiply-as-final-step pattern. The `llk_math_eltwise_unary_sfpu_atanh_init()` showing `sfpu::atanh_init<APPROXIMATE>` as the init callback is the direct model for softcap's init function.
- **Relevance**: high — the `vConstFloatPrgm` register loading in `atanh_init` is the standard mechanism for baking a float scalar (cap) into the SFPU pipeline; softcap will need to load `1.0f/cap` and `cap` into these registers before the computation loop.

### 4. tanhshrink
- **Why selected**: Tanhshrink (`x - tanh(x)`) is the only existing operation in this codebase that directly calls `tanh_tile()` in a dedicated compute kernel. The `tanhshrink_sfpu_kernel.cpp` file shows exactly how to call `tanh_tile_init()` and `tanh_tile(idst)` within a tile processing loop, including the `init_sfpu(cb_input, cb_output)` pattern, `tile_regs_acquire/commit/wait/release` sequencing, and `copy_tile` → `tanh_tile` → `pack_tile` pipeline. Since softcap requires tanh as its central operation, this is the direct template for the compute kernel structure.
- **Relevance**: high — the tanh invocation pattern in the compute kernel is directly reusable; softcap's kernel will differ only in adding a scalar pre-multiply (x/cap) before tanh and a scalar post-multiply (cap) after tanh.

### 5. hardtanh
- **Why selected**: Hardtanh is the best reference for how a parametrized SFPU operation passes scalar runtime arguments from the host to the SFPU kernel. The `_calculate_hardtanh_` function in `ckernel_sfpu_hardtanh.h` takes `param0`, `param1`, `param2` as `uint32_t` and converts them via `s2vFloat16b()` — this is the pattern for receiving `cap` as a packed float. The `unary_op_utils.hpp` also marks HARDTANH in `is_parametrized_type()` showing the exact registration needed for a parameterized op type. The `unary_program_factory.cpp` shows how `pack_scalar_runtime_arg()` extracts and packs the scalar before passing it as a runtime argument — softcap needs this same mechanism for `cap`.
- **Relevance**: medium — does not use tanh internally, but provides the complete host-side infrastructure blueprint (parametrized type registration, runtime arg packing, SFPU param unpacking) that softcap needs since `cap` is a user-provided runtime float parameter.
