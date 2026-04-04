# Reference Operation Selection for selu

## Target Operation
- **Name**: `selu`
- **Definition**: `scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))`, where:
  - `scale = 1.0507009873554804934193349852946` (fixed constant)
  - `alpha = 1.6732632423543772848170429916717` (fixed constant)
- **Simplified form**:
  - `scale * x`                        if `x >= 0`
  - `scale * alpha * (exp(x) - 1)`    if `x < 0`
- **Component operations identified**:
  1. Conditional branching on sign of `x` (`v_if(v >= 0.0f)` / `v_else`)
  2. `exp(x)` computation (using `_sfpu_exp_21f_bf16_`)
  3. Subtraction: `exp(x) - 1`
  4. Multiply by `alpha` constant (runtime parameter)
  5. Multiply by `scale` constant (runtime parameter)
  6. Two runtime `uint32_t` parameters (`scale`, `alpha`) via bit-reinterpreted floats
  7. `is_fp32_dest_acc_en` precision path with `float_to_fp16b` rounding
- **Key structural notes**:
  - Uses `Converter::as_float()` to recover floating-point values from packed `uint32_t` args
  - Both branches write to `sfpi::dst_reg[0]` before advancing `sfpi::dst_reg++`
  - Custom LLK header pattern (`ckernel_sfpu_unary_selu.h`) — not in the common LLK library

---

## Selected References (ranked by relevance)

### 1. celu
- **Why selected**: CELU has the **identical structural formula** to SELU: `max(0, x) + min(0, alpha * (exp(x/alpha) - 1))`. The SFPU kernel uses the exact same two-branch `v_if(v < 0.0f)` / `v_endif` pattern, the same `exp(...) - sfpi::vConst1` idiom for the negative branch, and the same `is_fp32_dest_acc_en` + `float_to_fp16b` precision handling. CELU also takes **two runtime parameters** (alpha and alpha_recip), making it the closest analog for the two-parameter (scale, alpha) SELU registration.
- **Relevance**: **High** — provides the complete compute kernel template, the two-param `_llk_math_eltwise_unary_sfpu_params_` dispatch pattern, the custom LLK header layout (`ckernel_sfpu_celu.h`), the `compute API` layer style (`celu_tile` / `celu_tile_init`), and the `unary_op_utils.cpp` init/compute string definitions. Every abstraction layer of CELU is directly adaptable to SELU.
- **Key source files**:
  - `ckernel_sfpu_celu.h` — SFPU kernel
  - `llk_math_eltwise_unary_sfpu_activations.h` — LLK wrapper
  - `api/compute/eltwise_unary/activations.h` — Compute API tile functions
  - `docs/sfpu_operations/key_notes/celu_key_notes.md` — Formula reference

### 2. elu
- **Why selected**: ELU is the **mathematical parent** of SELU. Mathematically, `SELU(x) = scale * ELU(x, alpha)` where both scale and alpha are fixed SELU constants. ELU's SFPU kernel (`ckernel_sfpu_elu.h`) shows how `_calculate_exponential_piecewise_` is called in the negative branch with a single slope parameter, producing `slope * (exp(x) - 1.0f)`. SELU extends this with an outer scale multiply on both branches. The `_init_elu_` function shows the exponential initialization pattern.
- **Relevance**: **High** — demonstrates the single-parameter exp-branch activation pattern that SELU's negative branch directly extends. The `elu_tile` / `elu_tile_init` API shows the simpler one-param SFPU dispatch macro. The `SFPU_OP_ELU_INCLUDE` define pattern in `sfpu_split_includes.h` is directly mirrored by `SFPU_OP_SELU_INCLUDE`.
- **Key source files**:
  - `ckernel_sfpu_elu.h` — SFPU kernel with `_calculate_elu_` and `_init_elu_`
  - `api/compute/eltwise_unary/elu.h` — Compute API tile functions
  - `docs/sfpu_operations/key_notes/elu_key_notes.md` — Formula reference

### 3. prelu_sfpu
- **Why selected**: PRELU is the **simplest conditional-multiply activation** in the codebase and provides the clearest, most readable template for the SFPU registration chain. Its `calculate_prelu` kernel (`ckernel_sfpu_prelu.h`) reduces to `v_if(a < 0.0f) { a = a * weight; } v_endif` — the same core conditional pattern used in SELU (plus the exp computation). PRELU's registration stack (SFPU ckernel → LLK wrapper → Compute API → `unary_op_utils` init/compute strings) mirrors SELU's stack exactly. PRELU uses one runtime parameter, making it a simpler reference for the parameter-passing machinery.
- **Relevance**: **High** — best reference for understanding the full SFPU registration pipeline (from `calculate_prelu` through `prelu_tile_init` to `SFPU_OP_PRELU_INCLUDE`) at its simplest. The `ckernel_sfpu_prelu.h` file lives in the same custom LLK layer as SELU's kernel, unlike the LLK-library-level ELU kernel.
- **Key source files**:
  - `ckernel_sfpu_prelu.h` — SFPU kernel (minimal conditional-multiply template)
  - `api/compute/eltwise_unary/prelu.h` — Compute API tile functions
  - `docs/sfpu_operations/key_notes/prelu_sfpu_key_notes.md` — Formula reference

### 4. rrelu
- **Why selected**: RRELU is a **recently added two-parameter custom-LLK operation** that lives at the same layer as SELU (`ckernel_sfpu_rrelu.h` / `llk_math_eltwise_unary_sfpu_rrelu.h`). Its LLK wrapper demonstrates the `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(fn, dst_index, vector_mode, lower, upper)` two-argument dispatch signature that SELU also uses (`scale`, `alpha`). The eval mode (`calculate_rrelu_eval`) is structurally similar to SELU's positive branch (simple conditional multiply), and shows the `Converter::as_float()` pattern for both parameters. Also shows `rrelu_init(seed)` as an init function with a parameter.
- **Relevance**: **High** — best reference for the exact two-param `_llk_math_eltwise_unary_sfpu_params_` call pattern used in SELU's LLK wrapper, and for understanding the custom-header LLK architecture that SELU follows. The `SFPU_OP_RRELU_INCLUDE` → `api/compute/eltwise_unary/rrelu.h` chain also models the SELU include guard pattern.
- **Key source files**:
  - `ckernel_sfpu_rrelu.h` — SFPU kernel (two-param, custom LLK layer)
  - `llk_math_eltwise_unary_sfpu_rrelu.h` — LLK wrapper with two-param dispatch
  - `docs/sfpu_operations/key_notes/rrelu_key_notes.md` — Formula reference

### 5. expm1
- **Why selected**: EXPM1 implements `exp(x) - 1` which is the **core mathematical sub-expression** of SELU's negative branch. The `ckernel_sfpu_expm1.h` shows how `_sfpu_exp_21f_bf16_<true>(val)` is called and then `sfpi::vConst1` is subtracted — exactly what SELU's kernel does inline. More importantly, it shows the precision-sensitive `_sfpu_expm1_<is_fp32_dest_acc_en>` wrapper with Taylor series for small values. The `expm1_init` function shows what polynomial coefficients must be loaded into SFPU constant registers for the exponential computation.
- **Relevance**: **Medium** — the sub-expression `exp(x) - 1` in SELU is handled inline (like CELU does), not by calling `expm1`. But the `expm1` kernel is the canonical reference for understanding *why* and *how* the `_sfpu_exp_21f_bf16_<true>` + subtract pattern works in bfloat16 context, including the `is_fp32_dest_acc_en` precision guard. The `expm1_init` coefficient setup can clarify what the SELU init function should do.
- **Key source files**:
  - `ckernel_sfpu_expm1.h` — SFPU kernel with `_sfpu_expm1_` and `expm1_init`
  - `llk_math_eltwise_unary_sfpu_expm1.h` — LLK wrapper
  - `docs/sfpu_operations/key_notes/expm1_key_notes.md` — Formula reference

---

## Summary of Relevance by Component

| SELU Component | Primary Reference | Secondary Reference |
|---|---|---|
| `v_if(v >= 0) { x * scale }` conditional | `prelu_sfpu` | `rrelu` (eval mode) |
| `exp(x) - 1` negative branch | `elu` | `expm1` |
| Two-path `max(0,x) + min(0, alpha*(exp-1))` | `celu` | `elu` |
| Two `uint32_t` runtime params | `celu` | `rrelu` |
| `Converter::as_float()` for params | `celu`, `rrelu` | `prelu_sfpu` |
| `is_fp32_dest_acc_en` + `float_to_fp16b` | `celu` | `expm1` |
| Custom LLK header (not common/inc) | `prelu_sfpu`, `rrelu` | `celu` |
| `SFPU_OP_*_INCLUDE` pattern | `elu` | `rrelu` |
| `_llk_math_eltwise_unary_sfpu_params_` 2-arg | `rrelu` | `celu` |
