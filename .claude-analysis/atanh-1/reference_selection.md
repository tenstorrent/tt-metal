# Reference Operation Selection for atanh

## Target Operation
- **Name**: atanh
- **Definition**: atanh(x) = 0.5 * ln((1+x)/(1-x))
- **Component operations identified**:
  - `ln` (natural logarithm) — applied to the ratio (1+x)/(1-x)
  - Addition: `1 + x`
  - Subtraction: `1 - x`
  - Division: `(1+x) / (1-x)` (implicit via multiplication after reciprocal, or direct ratio)
  - Scalar multiply: result * 0.5
  - Domain guard: |x| < 1; values at or beyond ±1 produce ±inf or NaN

## Selected References (ranked by relevance)

### 1. cbrt
- **Why selected**: `cbrt` is the closest structural template in the codebase. It is a no-parameter SFPU operation that: (a) uses a `_init_` function to load polynomial constants into `vConstFloatPrgm0/1/2`, (b) implements a multi-step arithmetic algorithm in `calculate_cube_root` using raw sfpi vector arithmetic (`sfpi::vFloat`, `sfpi::addexp`, `sfpi::reinterpret`, etc.), (c) uses the `sfpi::abs` and `sfpi::setsgn` pattern for sign handling, (d) has a `if constexpr (is_fp32_dest_acc_en)` branch for fp32 vs bfloat16 output rounding with `sfpi::float_to_fp16b`, and (e) registers its `SfpuType::cbrt` enum entry and uses `SFPU_TWO_TEMPLATE_PARAM_INIT` macro for LLK init dispatch. The `atanh` kernel will need the same `vConstFloatPrgm` constant loading (for `0.5`) and the same fp16b rounding pattern.
- **Relevance**: high — provides the exact template for: custom init loading a constant (0.5), fp32/bfloat16 dispatch branch, multi-step arithmetic loop, and the `SfpuType` + `SFPU_INIT_KERNEL_CALL` wiring pattern.

### 2. rpow
- **Why selected**: `rpow` shows how to perform a multi-step computation involving logarithm-like arithmetic on SFPU hardware (`log2` via float bit manipulation). It demonstrates: (a) scalar precomputation before the per-element loop, (b) integer bit manipulation on float values using `sfpi::reinterpret<sfpi::vInt>` and `sfpi::addexp`, (c) special-case handling with `v_if`/`v_endif` conditionals, and (d) the `Converter::as_float(param)` pattern for decoding IEEE 754 bit-packed parameters. While atanh does not need `log2` specifically, the float bit arithmetic shown here directly informs how to apply the log/exp primitive to implement `ln((1+x)/(1-x))` using whatever log primitive is available from the tt_llk submodule.
- **Relevance**: high — the integer-exponent/mantissa manipulation in `calculate_rpow` is the exact SFPU idiom used to implement transcendental functions when no direct LLK primitive is available; also shows `ckernel_sfpu_converter.h` usage needed if the converter header is included.

### 3. hardsigmoid
- **Why selected**: `hardsigmoid` is the simplest no-parameter unary SFPU operation in the codebase with a fully functional end-to-end wiring. It shows the minimal scaffold for a new operation: `calculate_hardsigmoid()` reads `dst_reg[0]`, performs arithmetic with `v_if`/`v_endif` for clamping, writes back, and increments `dst_reg`. There is no custom init function (just an inline empty body), and the LLK header (`llk_math_eltwise_unary_sfpu_hardsigmoid.h`) shows the `SFPU_UNARY_KERNEL_INIT` macro pattern for the simplest init path. The `SFPU_OP_HARDSIGMOID_INCLUDE` split-include guard shows how to register a custom include path if needed. Since atanh also has no parameters, this is a direct structural template.
- **Relevance**: high — minimal complete example of a no-parameter op from `ckernel_sfpu_*.h` through LLK wrapper through compute API header through `sfpu_split_includes.h`, exactly matching atanh's parameter profile.

### 4. hardtanh
- **Why selected**: `hardtanh` is the closest existing function in the tanh family — it is literally `clamp(x, min, max)` which defines the tanh operation's saturation region. It shows: (a) the `SFPU_OP_HARDTANH_INCLUDE` split-include path pattern, (b) how to use `Converter::as_float(param0)` to decode IEEE 754 bit-packed min/max float values, (c) the two-parameter LLK dispatch via `llk_math_eltwise_unary_sfpu_hardtanh` with `uint param0, uint param1`, and (d) the `v_if(val < min_val) { val = min_val; }` clamping idiom with `vFloat` locals. The atanh operation will need to handle the domain boundary (|x| >= 1 → ±inf), and the hardtanh pattern of conditional assignment is directly reusable for that edge-case guard.
- **Relevance**: medium — the tanh-family naming convention and the conditional-assignment pattern for domain clamping are directly relevant; the parameter-decode pattern is relevant if atanh needs special-value handling.

### 5. softshrink
- **Why selected**: `softshrink` demonstrates the full end-to-end wiring of a parameterized operation that uses pure sfpi arithmetic (no LLK transcendental primitive). It shows: (a) `calculate_softshrink(uint32_t param0)` receiving a parameter at the calculate step, (b) a three-branch conditional with `v_if`/`v_endif` covering distinct numeric regions, (c) the `SfpuType::softshrink` enum entry using `llk_math_eltwise_unary_sfpu_init<SfpuType::softshrink, APPROXIMATE>()` with no init callback (just the type registration), (d) the LLK wrapper dispatching with `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_softshrink<APPROXIMATE, ITERATIONS>, ...)`, and (e) the `unary_op_utils.cpp` entry including the bitcast-to-uint32 float parameter encoding. While atanh has no parameters, the multi-region numeric handling in `softshrink` informs how to structure atanh's special-case handling for x near ±1.
- **Relevance**: medium — the three-region conditional structure and the `is_parametrized_type` / `get_op_init_and_func_parameterized` plumbing in `unary_op_utils.cpp` shows the dispatch pattern; the arithmetic-only kernel (no LLK primitive) is directly relevant if atanh is implemented via pure sfpi arithmetic rather than a log primitive.
