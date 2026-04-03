# Reference Operation Selection for rrelu

## Target Operation
- **Name**: rrelu
- **Definition**: RReLU(x) = x if x >= 0, a*x if x < 0. Training mode: a ~ Uniform(lower, upper). Eval mode: a = (lower+upper)/2. Defaults: lower=1/8, upper=1/3.
- **Component operations identified**:
  - Piecewise conditional: `v_if(x < 0.0f)` branch with slope multiply
  - Scalar multiply of input by float parameter `a`
  - Uniform random number generation from `[lower, upper)` for training mode
  - PRNG seed initialization
  - Two configurable float parameters (`lower`, `upper`) passed as kernel arguments

---

## Selected References (ranked by relevance)

### 1. prelu_sfpu
- **Why selected**: The eval mode of rrelu (fixed `a = (lower+upper)/2`) is structurally identical to PReLU: both apply `a = a * slope` only when `a < 0`, and pass the slope as a single float parameter. The kernel at `/localdev/vignjatijevic/tt-metal-5/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h` is a direct template — rrelu eval mode is prelu with a precomputed slope. The TTNN registration pattern (`prelu_tile_init()` + `prelu_tile(idst, param_bits)` in `unary_op_utils.cpp` line 439-441) shows exactly how to wire a single-param conditional-multiply activation.
- **Relevance**: High — the rrelu eval-mode SFPU kernel body can be directly adapted from `calculate_prelu`. The `Converter::as_float(value)` pattern for loading a float parameter from a uint bitcast is directly reusable.

### 2. leaky_relu
- **Why selected**: LeakyReLU applies `slope * x` when `x < 0`, which is the same piecewise conditional as rrelu. The LLK-level implementation at `/localdev/vignjatijevic/tt-metal-5/tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h` shows `_calculate_lrelu_` using `SFPSETCC` / `SFPMUL` / `SFPENCC` instruction sequences — the low-level SFPU instruction pattern for a conditional multiply. The TTNN registration (`leaky_relu_tile_init()` + `leaky_relu_tile(idst, slope_bits)` in `unary_op_utils.cpp` line 179-182) plus its `SFPU_OP_RELU_FAMILY_INCLUDE` macro grouping illustrate how to place a new relu-family op in the same include family.
- **Relevance**: High — provides both the SFPU instruction-level pattern (for the training-mode conditional multiply path) and the compile-time macro/include structure to follow.

### 3. rand
- **Why selected**: The `rand` operation is the only existing implementation that uses `rand_tile_init(seed)` followed by `rand_tile(dst, from_bits, scale_bits)` to generate uniform random floats in a range `[from, from+scale)`. This is precisely what rrelu training mode requires: draw `a ~ Uniform(lower, upper)` per element. The kernel at `/localdev/vignjatijevic/tt-metal-5/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h` shows the full SFPU sequence: `TTI_SFPMOV` to generate a pseudo-random uint32, `SFPSETSGN`/`SFPSETEXP` to normalize to `[1,2)`, subtract 1 to get `[0,1)`, then `SFPMAD` to scale to `[from, from+scale)`. The program factory at `/localdev/vignjatijevic/tt-metal-5/ttnn/cpp/ttnn/operations/rand/device/rand_program_factory.cpp` and compute kernel at `.../kernels/compute_uniform.cpp` demonstrate how to pass `seed`, `from`, and `scale` as runtime args and call `rand_tile_init` + `rand_tile`.
- **Relevance**: High — the training mode of rrelu can directly incorporate the `rand` SFPU sequence for per-element slope sampling, then apply the conditional multiply from prelu.

### 4. dropout
- **Why selected**: Dropout demonstrates the pattern of seeding the PRNG (`_init_dropout_(seed)` which calls `init_prng_seed`) and then using `TTI_SFPMOV(0, 9, LREG3, 8)` to generate a pseudo-random integer per element inside the tile loop. This is the canonical existing pattern for PRNG-based per-element operations in SFPU kernels. The inner implementation at `/localdev/vignjatijevic/tt-metal-5/tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h` shows how to combine PRNG generation with a conditional value override — structurally analogous to rrelu training mode (generate random value, conditionally use it). The `dropout_init(seed)` wrapper in the outer header also shows the two-phase init + compute structure.
- **Relevance**: Medium — the PRNG seed init and per-element random generation pattern is directly reusable, though rrelu uses the random value as a multiply slope rather than a drop mask.

### 5. selu
- **Why selected**: SELU is `scale * max(0, x) + scale * alpha * (exp(x) - 1)` for `x < 0`, implemented with `v_if(v >= 0.0f) { ... } v_else { ... }`. It takes two float parameters (`scale` and `alpha`) registered as `selu_tile(idst, scale_bits, alpha_bits)` in `unary_op_utils.cpp` lines 553-563. This is the clearest existing model for a two-parameter activation op: how to declare two params in `is_parametrized_type`, how to extract `param0` and `param1` in `get_op_init_and_func_parameterized`, how to format the tile call with two bit-cast float args, and how the kernel at `/localdev/vignjatijevic/tt-metal-5/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_selu.h` receives them as `uint scale, uint alpha`. For rrelu eval mode, `lower` and `upper` are two parameters whose midpoint forms the slope — the two-parameter registration pattern from selu is the right template.
- **Relevance**: Medium — the two-parameter op registration plumbing (type enum, `is_parametrized_type`, `get_op_init_and_func_parameterized`, tile call format) is directly applicable to rrelu's `lower` and `upper` parameters.
