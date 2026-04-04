# Reference Operation Selection for softsign

## Target Operation
- **Name**: softsign
- **Definition**: x / (1 + |x|)
- **Component operations identified**:
  1. `sfpi::abs(v)` — absolute value of the input element
  2. Add scalar 1: `sfpi::abs(v) + sfpi::vConst1` — constant addition
  3. `sfpu_reciprocal<APPROXIMATION_MODE>(tmp)` — reciprocal of the denominator
  4. Scalar multiply: `v * tmp` — multiply original x by the reciprocal

## Selected References (ranked by relevance)

### 1. hardsigmoid
- **Why selected**: `hardsigmoid` is the most structurally complete template within the worktree. It lives in `ckernel_sfpu_hardsigmoid.h`, is registered in `unary_op_utils.cpp` (both `get_op_init_and_func_default` and `get_macro_definition` with `SFPU_OP_HARDSIGMOID_INCLUDE`), and is wired through `sfpu_split_includes.h`. It is a newly generated operation from the same codegen wave, so it demonstrates the exact file layout, include structure, LLK wrapper pattern (`llk_math_eltwise_unary_sfpu_hardsigmoid.h` using `_llk_math_eltwise_unary_sfpu_params_`), and compute API header pattern (`api/compute/eltwise_unary/hardsigmoid.h`) that `softsign` must replicate. Its inner loop uses `sfpi::dst_reg[0]`, `sfpi::vConst1`, arithmetic on `vFloat`, and a `sfpi::dst_reg++` advance — the same basic skeleton softsign uses.
- **Relevance**: High — provides the complete end-to-end file skeleton (ckernel, LLK wrapper, compute API header, unary_op_utils registration, sfpu_split_includes guard) that softsign must follow

### 2. cbrt
- **Why selected**: `ckernel_sfpu_cbrt.h` is the only existing custom kernel in the worktree that calls `sfpi::abs(a)` directly — precisely the SFPU intrinsic needed to compute `|x|` in softsign. It also uses `sfpi::setsgn(d, a)` to restore the sign of the result, which demonstrates how the sign of the original `x` is preserved alongside absolute-value arithmetic. The function signature `calculate_cube_root<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>` and the separate `cube_root_init<APPROXIMATION_MODE>` function show how to split the init and calculate functions. It also shows `#pragma GCC unroll 8` and the `dst_reg++` loop structure.
- **Relevance**: High — directly demonstrates `sfpi::abs()` usage in a custom worktree kernel, and shows the init/calculate split pattern

### 3. silu
- **Why selected**: `ckernel_sfpu_silu.h` (from the build reference) implements `silu(x) = x * sigmoid(x)`, which is structurally analogous to `softsign(x) = x * recip(1 + |x|)`. Both are of the form `output = x * f(x)` where `f(x)` is a normalizing function. Critically, `silu` uses `sfpu_reciprocal<APPROXIMATION_MODE>` indirectly (via `_sfpu_sigmoid_`) and its `silu_init()` calls `_init_sfpu_reciprocal_<APPROXIMATION_MODE>()` — exactly the reciprocal init pattern that softsign's `init_softsign()` must use. The multiply-after-transform pattern (`x * result`) mirrors softsign's final `v * tmp`.
- **Relevance**: High — demonstrates the `x * f(x)` multiply structure and the `sfpu_reciprocal` init pattern that softsign reuses

### 4. sigmoid
- **Why selected**: `ckernel_sfpu_sigmoid.h` (from the build reference) computes `1 / (1 + exp(-x))`. The denominator `sfpi::vConst1 + exp_neg_x` and subsequent `_sfpu_reciprocal_<>(denominator)` call is the closest sub-expression to softsign's `sfpi::abs(v) + sfpi::vConst1` followed by `sfpu_reciprocal<>()`. Both operations share the pattern: compute a positive quantity (sigmoid: `1 + exp(-x)`, softsign: `1 + |x|`) and then take its reciprocal using the same SFPU reciprocal instruction sequence. Sigmoid's `sigmoid_init()` calling `_init_reciprocal_<false, false>()` is the exact reciprocal initialization that softsign's init must mirror.
- **Relevance**: High — the denominator-plus-reciprocal sub-expression is structurally identical; softsign's implementation directly parallels sigmoid's computation, replacing `1 + exp(-x)` with `1 + |x|`

### 5. hardtanh
- **Why selected**: `ckernel_sfpu_hardtanh.h` is the only other parameterized custom kernel in the worktree. While softsign is not parameterized, hardtanh shows the complete reference for how the worktree-local custom kernel headers use `sfpi.h` (not `ckernel.h + ckernel_defs.h`), use `cstdint` for `std::uint32_t` param passing, and follow the `namespace ckernel { namespace sfpu { ... } }` namespace nesting. It also shows how to handle `Converter::as_float(param)` for bitcast parameters if softsign ever needs runtime params. As a worktree-resident kernel from the same generation wave, it confirms the file structure and header guards used in this custom kernel set.
- **Relevance**: Medium — confirms the include style (`sfpi.h` directly), namespace nesting pattern, and `calculate_*` / init function naming convention used by the worktree custom kernels
