# Reference Operation Selection for softcap

## Target Operation
- **Name**: softcap
- **Definition**: `cap * tanh(x / cap)` where `cap` is a float parameter
- **Component operations identified**: scalar multiply (`x * (1/cap)`), tanh, scalar multiply (`result * cap`), float parameter passing

## Selected References (ranked by relevance)

### 1. swish
- **Why selected**: swish (`x * sigmoid(x)`) is the closest structural match. It is a Wave 3 generated operation with the exact same code organization as softcap will need: a dedicated `ckernel_sfpu_swish.h` implementing the math, an LLK wrapper `llk_math_eltwise_unary_sfpu_swish.h`, an API header `swish.h`, entries in `unary_op_types.hpp`, `unary_op_utils.cpp`, and `sfpu_split_includes.h`. The pattern `scalar_operation * transcendental(x)` mirrors `cap * tanh(x / cap)`. The sigmoid approximation loop structure (iterate over ITERATIONS, read from `sfpi::dst_reg[0]`, write back) is the direct template for the softcap kernel loop.
- **Relevance**: high — provides the complete Wave 3 file structure template including the SFPU compute loop, the LLK init/compute wrapper pair, and the API header pattern

### 2. tanhshrink
- **Why selected**: tanhshrink (`x - tanh(x)`) calls `tanh_tile` directly in a dedicated kernel (`tanhshrink_sfpu_kernel.cpp`). This is the only operation in the codebase that calls the existing `tanh_tile` / `tanh_tile_init` API within a custom compute kernel. Since softcap will need to call `tanh_tile` internally (or replicate the tanh SFPU logic), tanhshrink shows exactly how to set up the kernel, initialize SFPU, copy tiles, call `tanh_tile_init()` + `tanh_tile(idst)`, and pack the result.
- **Relevance**: high — direct reference for integrating `tanh_tile` into a new kernel; shows the `init_sfpu` + `tanh_tile_init` + `copy_tile` + `tanh_tile` call sequence

### 3. atanh
- **Why selected**: atanh is a Wave 3 generated operation with an explicit `atanh_init()` function that loads three polynomial coefficients into `sfpi::vConstFloatPrgm0/1/2`. This init pattern is the reference for how softcap should load `cap` (and `1/cap`) into programmable constant registers so they are available as vector constants inside the SFPU compute loop, avoiding repeated scalar-to-vector broadcast each iteration.
- **Relevance**: high — provides the `_init_` function pattern using `vConstFloatPrgm` registers for parameter passing, and the full Wave 3 layered structure (ckernel header + llk_init wrapper calling `llk_math_eltwise_unary_sfpu_init<SfpuType::..., APPROXIMATE>(init_func)`)

### 4. hardtanh
- **Why selected**: hardtanh is the canonical example of a parameterized SFPU operation that takes float parameters encoded as `uint32_t param0/1/2` in FP16_B format. `_calculate_hardtanh_` shows the exact pattern for converting these packed parameters via `sfpi::s2vFloat16b(paramN)` into `sfpi::vFloat` values used in the SFPU loop. Since softcap takes a float `cap` parameter, this is the reference for how to receive, unpack, and use the runtime float parameter in the SFPU kernel.
- **Relevance**: high — demonstrates the float parameter encoding and decoding pattern (`s2vFloat16b`) and how parameterized SFPU kernels are structured with `const int iterations` and `uint32_t paramN` arguments

### 5. sinh
- **Why selected**: sinh (`(exp(x) - exp(-x)) / 2`) is a Wave 3 generated operation that applies a scalar multiply (`* 0.5f`) after computing a transcendental function, mirrors the `* cap` final multiply in softcap. It also uses the `exp_21f` helper which applies a scale factor (`x * log2e`) before computing the exponential — the same conceptual pattern as `x * (1/cap)` before tanh. The `#pragma GCC unroll 0` + `ITERATIONS` template loop with `sfpi::dst_reg[0]` read/write is the direct boilerplate for a Wave 3 SFPU compute function.
- **Relevance**: medium — provides the pre-scale-then-transcendental-then-post-scale computation pattern, and the Wave 3 function template with `APPROXIMATION_MODE` and `ITERATIONS` template parameters
