# Reference Operation Selection for star_relu

## Target Operation
- **Name**: star_relu
- **Definition**: `s * relu(x)^2 + b` where `s` (scale) defaults to 1.0 and `b` (bias) defaults to 0.0
- **Component operations identified**:
  1. `relu(x)` — conditional zero-clamp: `max(0, x)`
  2. `square` — multiply result by itself: `relu(x) * relu(x)`
  3. `mul_unary_sfpu` (scalar scale) — multiply by parameter `s`
  4. `add_unary_sfpu` (scalar bias) — add parameter `b`
  - Overall shape: two-parameter composite (scale, bias) with a conditional guard and a squaring step

## Selected References (ranked by relevance)

### 1. SQUARE
- **Why selected**: The squaring step `relu(x)^2` is the arithmetically distinctive core of star_relu. The `_calculate_square_` implementation in `/localdev/vignjatijevic/tt-metal/tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_square.h` (wrapped in `ckernel_sfpu_square.h`) shows exactly how to multiply a destination register value by itself using direct SFPU instructions (`TTI_SFPLOAD`, `TTI_SFPMUL`, `TTI_SFPSTORE`). The star_relu compute loop must replicate this multiplication after the relu clamp.
- **Relevance**: High — exact sub-operation needed for the `x^2` step; tile init (`square_tile_init`) and tile function (`square_tile`) patterns are reused directly.

### 2. SELU
- **Why selected**: SELU (`scale * (max(0,x) + min(0, alpha*(exp(x)-1)))`) is the closest structural match: it applies a conditional branch to separate positive from negative inputs, and the positive branch is `v * scale_value` — combining relu with a scalar multiply in one kernel loop. The implementation in `ckernel_sfpu_unary_selu.h` shows the two-parameter (`scale`, `alpha`) kernel pattern with `Converter::as_float` param unpacking, a `v_if(v >= 0.0f) / v_else` conditional, and an explicit multiply-by-scalar result write. star_relu only needs the positive branch of SELU (squaring, not exp), but the two-parameter init, parameter passing convention, and single-pass loop structure are all directly applicable.
- **Relevance**: High — provides the two-scalar-parameter kernel template (`uint scale, uint alpha` → `Converter::as_float`) and the relu-conditional combined with a scalar multiply in one SFPU loop.

### 3. PRELU
- **Why selected**: PRELU (`x < 0 ? x*slope : x`) in `ckernel_sfpu_prelu.h` is the simplest existing parametrized kernel that uses a `v_if(a < 0.0f) { a = a * init; } v_endif` conditional multiply pattern. Its structure — load one float param, single tile loop, conditional branch, multiply and writeback — is a clean, easy-to-follow template for the conditional relu branch in star_relu. Unlike leaky_relu (which delegates to `_calculate_lrelu_`), prelu has its full implementation inline, making it easier to adapt.
- **Relevance**: High — simplest complete inline example of a conditional + scalar-multiply SFPU kernel with one parameter; good structural starting point before layering in the square and bias steps.

### 4. XIELU
- **Why selected**: xIELU (`alpha_p * x^2 + beta*x` for `x > 0`) in `ckernel_sfpu_xielu.h` contains the sub-expression `alpha_p * x * x + beta_mul_x`, which is algebraically equivalent to star_relu's `s * x^2 + b` (for the positive case, with `b` being a fixed offset rather than `beta*x`). The helper `_xielu_mad_` shows the multiply-accumulate pattern `mul_a * mul_b + addend` needed for combining scale, square, and bias in one expression. xIELU also demonstrates two-parameter passing (`param0 = alpha_p`, `param1 = alpha_n`) and `is_fp32_dest_acc_en` precision handling.
- **Relevance**: High — contains the exact algebraic sub-expression `alpha * x^2 + offset` in the positive branch, plus a two-parameter init and MAD (multiply-add) helper directly applicable to star_relu.

### 5. LEAKY_RELU
- **Why selected**: Leaky ReLU (`max(0,x) + slope*min(0,x)`) in `ckernel_sfpu_relu.h` (registered under `SFPU_OP_RELU_FAMILY_INCLUDE`) and its `unary_op_utils.cpp` entry `leaky_relu_tile_init(); leaky_relu_tile(idst, slope_bits)` is the canonical example of how a parametrized relu-family op is wired through the full TTNN stack — from `UnaryOpType::LEAKY_RELU` in `get_macro_definition`, through `get_op_init_and_func_parameterized`, to the SFPU kernel. star_relu belongs in the same relu family and should follow the same registration path (`SFPU_OP_RELU_FAMILY_INCLUDE` or a new macro), parameter unpacking with `std::bit_cast<uint32_t>(param0)`, and init/compute naming conventions.
- **Relevance**: Medium-High — less directly relevant at the arithmetic level (uses `_calculate_lrelu_` delegate, no squaring), but essential as the end-to-end registration and naming template for a two-parameter relu-family SFPU operation in the TTNN layer.
