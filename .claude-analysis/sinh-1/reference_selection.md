# Reference Operation Selection for sinh

## Target Operation
- **Name**: sinh
- **Definition**: sinh(x) = (exp(x) - exp(-x)) / 2
- **Component operations identified**: exp(x), exp(-x), subtraction, multiply-by-0.5 (scalar multiply), symmetric two-exp composition

---

## Selected References (ranked by relevance)

### 1. cosh
- **Why selected**: `cosh` is structurally nearly identical to `sinh`. Both compute `(exp(x) OP exp(-x)) * 0.5` — the only difference is `+` (cosh) vs `-` (sinh). The implementation in `ckernel_sfpu_cosh.h` uses exactly the same primitives that `sinh` will require: two calls to `_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(v)` and `_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(-v)`, combined with `* 0.5f`, and initialized via `_init_exponential_<APPROXIMATION_MODE, false, p_sfpu::kCONST_1_FP16B>()`. The template signature `template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>` is exactly what `sinh` should use. The tile API file `tt_metal/hw/inc/api/compute/eltwise_unary/cosh.h` also provides the exact pattern for `sinh_tile_init()` and `sinh_tile()` wrappers.
- **Relevance**: High — copy-and-adapt with one character change (`+` to `-`) in the core expression; all init, template, and API scaffolding is directly reusable.

### 2. selu
- **Why selected**: `selu` in `ckernel_sfpu_selu.h` demonstrates the pattern of calling `_calculate_exponential_piecewise_` (the exp internal used by the older LLK ops) and performing `alpha * (exp(x) - 1.0f)` — combining an exp call with subtraction of a constant and scalar multiplication. While `sinh` uses `_sfpu_exp_21f_bf16_` (the newer exp_21f algorithm from cosh), `selu` shows the exp-subtract-scale composition idiom. It also shows how to initialize exp via `_init_exponential_<APPROXIMATION_MODE, FAST_APPROX, EXP_BASE_SCALE_FACTOR>()` with a scalar constant, which is used by `cosh_init` and will be used by `sinh_init`. The tile API wrapper structure in `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h` is another direct template.
- **Relevance**: High — exp + constant subtraction + scalar multiply composition idiom; exp init pattern reused directly by sinh.

### 3. elu
- **Why selected**: `elu` (in `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_elu.h`) is the simplest example of `scale * (exp(v) - 1.0f)` — a focused, easy-to-follow reference for the exp-minus-constant-times-scalar pattern. Its `_init_elu_` function shows exactly the same `_init_exponential_<APPROXIMATION_MODE, FAST_APPROX, EXP_BASE_SCALE_FACTOR>()` call that `cosh_init` and therefore `sinh_init` will use. Being simpler than selu (no conditional branch, just the core exp computation), it is an easier template to follow for the implementor learning the exp init/compute split.
- **Relevance**: High — clearest direct example of exp(x) - constant pattern with the precise init function signature that sinh reuses.

### 4. lgamma
- **Why selected**: `lgamma` in `ckernel_sfpu_lgamma.h` demonstrates a kernel that computes multiple arithmetic sub-expressions (several reciprocal and log calls) and combines them into a single final result using additions, subtractions, and multiplications. It is the best local example of the "compute two independent terms and combine them arithmetically" structural pattern that `sinh` requires (compute `exp(x)` and `exp(-x)` independently, then combine via subtract and scale). It also illustrates how to handle multiple intermediate `sfpi::vFloat` temporaries within a single tile loop, which is good practice for the implementor writing `calculate_sinh`.
- **Relevance**: Medium — demonstrates multi-term combination and intermediate variable management within a tile loop; math is different but structure is instructive.

### 5. rpow
- **Why selected**: `rpow` in `ckernel_sfpu_rpow.h` directly exposes the internals of the `exp_21f` algorithm (the Moroz et al. 2022 exponential approximation) that underlies `_sfpu_exp_21f_bf16_`. Reading `rpow` alongside `cosh` gives the implementor a deep understanding of what `_sfpu_exp_21f_bf16_` is doing internally — the `addexp`, `exexp`, `exman9`, `_float_to_int32_positive_`, and Horner polynomial steps — which is essential for debugging and understanding numerical precision of the `sinh` result. It also demonstrates the `SFPU_OP_*_INCLUDE` macro pattern and the `unary_op_utils.cpp` parameterized registration path (the only parameterized op in the custom ops set), which is a useful contrast to the simpler non-parameterized registration `sinh` will use.
- **Relevance**: Medium — provides deep understanding of the exp_21f algorithm that sinh relies on via cosh's `_sfpu_exp_21f_bf16_` calls; useful for precision analysis and debugging.
