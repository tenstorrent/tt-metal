# Reference Operation Selection for lgamma

## Target Operation
- **Name**: lgamma
- **Definition**: ln(|Gamma(x)|)
- **Component operations identified**:
  - Natural logarithm (the "ln" outer function)
  - Absolute value (|Gamma(x)| - for handling negative x)
  - Polynomial approximation (Stirling's series or Lanczos approximation for ln(Gamma(x)))
  - Piecewise/conditional logic (reflection formula: lgamma(x) = log(pi/|sin(pi*x)|) - lgamma(1-x) for x < 0, or argument reduction for small x)
  - Floating-point constant storage in vConstFloatPrgm registers (polynomial coefficients)
  - Exponent extraction and manipulation (for argument normalization in log)

## Selected References (ranked by relevance)

### 1. cbrt
- **Why selected**: `cbrt` is the most structurally similar operation in the worktree: it implements a non-trivial multi-step approximation (magic-constant initial guess + Newton-Raphson refinement polynomial) using `vConstFloatPrgm0/1/2` to hold polynomial coefficients, extracts and manipulates integer bit representations of floats (via `reinterpret<vInt>`/`reinterpret<vFloat>` and `sfpi::shft`), and branches on `is_fp32_dest_acc_en` to emit different code paths for fp32 vs fp16b accumulation. Lgamma will follow the same template: store polynomial coefficients in vConstFloatPrgm registers, evaluate a Stirling-series polynomial in `calculate_lgamma()`, and provide a `lgamma_init()` that loads those constants. The SFPU_OP_CBRT_INCLUDE / `get_macro_definition` pattern and `llk_math_eltwise_unary_sfpu_cbrt.h` wrapper are direct templates for lgamma's scaffold files.
- **Relevance**: high — polynomial coefficient loading via `vConstFloatPrgm`, fp32/fp16b accumulation branching, SFPU_OP_*_INCLUDE macro, and `_init_` / `calculate_*` naming convention are all directly reusable

### 2. cosh
- **Why selected**: `cosh` is a composite formula — (exp(x) + exp(-x)) / 2 — that shows how to accumulate multiple SFPU sub-expression results into a single output. Lgamma's Stirling series accumulates multiple polynomial terms plus a log call (e.g. `(x-0.5)*log(x) - x + 0.5*log(2*pi) + correction_poly(x)`), following the same pattern of composing primitives arithmetically. `cosh` also illustrates how to include a vendor-provided helper (`sfpu/ckernel_sfpu_exp.h`) and call `_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>`, which is the model for calling `_calculate_log_body_no_init_()` inside `calculate_lgamma`. The `cosh_init()` delegate to `_init_exponential_` mirrors how `lgamma_init()` will delegate to `_init_log_()`.
- **Relevance**: high — composite multi-term formula structure, delegation to vendor SFPU primitive (exp/log), and template parameter forwarding for fp32 accumulation are directly applicable

### 3. selu
- **Why selected**: `selu` demonstrates the piecewise conditional pattern (v_if / v_endif) that lgamma requires for the reflection formula on negative inputs (x < 0 requires special handling: lgamma(x) is computed via the reflection relation, involving sin(pi*x) and lgamma(1-x)). The selu code structure — load value, branch on sign, compute exp/polynomial in the negative branch, unconditionally multiply by scale after the branch — is exactly the structure lgamma needs for its piecewise positive/negative domain handling. Selu also shows the `SFPU_INIT_KERNEL_CALL(selu, ckernel::sfpu::selu_init, APPROX)` macro usage in `selu.h` (the compute API wrapper), and the `#pragma GCC unroll 0` annotation appropriate for loops with conditional branches.
- **Relevance**: high — v_if/v_endif conditional branching pattern for piecewise domain handling, and the init macro call pattern in the compute API wrapper are directly reusable

### 4. hardsigmoid
- **Why selected**: `hardsigmoid` provides the cleanest example of a piecewise-linear operation that uses `v_if/v_endif` for clamping, has a parameter-free `hardsigmoid_tile_init()` (no constants needed at init because the slope `1/6` is a compile-time constexpr), and uses the `SFPU_OP_HARDSIGMOID_INCLUDE` macro in `sfpu_split_includes.h`. Lgamma has no runtime parameters, so its `UnaryOpType::LGAMMA` entry falls through to `get_op_init_and_func_default()` — exactly the path taken by hardsigmoid. The `update_macro_defines` call for `SFPU_OP_LGAMMA_INCLUDE` follows the same registration pattern. The `llk_math_eltwise_unary_sfpu_hardsigmoid.h` wrapper (init delegates to `llk_math_eltwise_unary_sfpu_init<SfpuType::hardsigmoid, APPROXIMATE>()` without a custom callback) shows the minimal variant for ops that do not need coefficient preloading.
- **Relevance**: medium — no-parameter default dispatch path in `get_op_init_and_func_default`, SFPU_OP_*_INCLUDE macro registration in sfpu_split_includes.h, and the LLK wrapper pattern for init-without-constants are directly applicable

### 5. hardtanh
- **Why selected**: `hardtanh` is the only operation in this worktree that passes runtime floating-point constants to the compute kernel via `uint32_t` bitcast (`std::bit_cast<uint32_t>(min_val)`) and receives them back via `Converter::as_float(param0)`. While lgamma has no user parameters, implementing it may require passing computed boundary constants (e.g., the value of the reflection cutoff) in the same way if the implementation uses a parameterized variant. More importantly, `hardtanh` is the sole representative of `get_op_init_and_func_parameterized()`, and its `SFPU_OP_HARDTANH_INCLUDE` / `hardtanh_tile_init()` / `hardtanh_tile(idst, param0, param1)` structure in the compute API header shows how to attach runtime args to the tile call — useful as a contrast model to confirm lgamma does not need this path. The `#pragma GCC unroll 8` annotation in `calculate_hardtanh` (vs `#pragma GCC unroll 0` in selu) illustrates the trade-off between unroll strategies when there are vs. are not conditional branches.
- **Relevance**: medium — demonstrates the contrast between parameterized and non-parameterized paths in unary_op_utils, and shows the Converter::as_float / uint32_t bitcast pattern for floating-point constant handling that may be useful for any lgamma boundary constants
