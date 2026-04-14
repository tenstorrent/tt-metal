# Reference Operation Selection for softcap

## Target Operation
- **Name**: softcap
- **Definition**: cap * tanh(x / cap)
- **Component operations identified**: tanh (core nonlinearity), scalar division by cap (x / cap), scalar multiplication by cap (result * cap), single float runtime parameter

## Selected References (ranked by relevance)

### 1. sinh
- **Why selected**: `sinh` is the most structurally complete Wave 3 generated op in the same file hierarchy (`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`). It implements a hyperbolic function — sinh(x) = (exp(x) - exp(-x)) / 2 — that is mathematically adjacent to tanh (tanh = sinh/cosh), using the same `exp_21f` helper for high-precision 2^z computation. The `sinh` source shows the full Wave 3 pattern: `calculate_sinh<APPROX, ITERATIONS>()` + `sinh_init()`, the `#pragma GCC unroll 8` loop over `dst_reg`, clamp-to-prevent-underflow logic with `v_if`, the Taylor series override for small arguments, and bfloat16 rounding via `sfpi::float_to_fp16b`. A tanh implementation for softcap will follow nearly the same structure. The LLK glue (`llk_math_eltwise_unary_sfpu_sinh.h`), API header (`api/compute/eltwise_unary/sinh.h`), and registration in `unary_op_utils.cpp` provide the complete end-to-end template.
- **Relevance**: high — provides the complete Wave 3 file structure template, the exp_21f helper, the tanh sub-expression building block (exp pos and neg), and the small-argument Taylor fallback pattern.

### 2. swish
- **Why selected**: `swish` is the other Wave 3 generated op in the same directory (`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`). It computes `x * sigmoid(x)`, which has the same outer structure as softcap: apply a nonlinearity to the input, then multiply the result by a scalar derived from the input. The key structural analogy is that swish stores the original `x`, applies a nonlinear transform, then multiplies back — exactly what softcap does with `cap * tanh(x/cap)` (save cap, scale x, apply tanh, multiply by cap). The swish source also shows polynomial approximation of a sigmoid-like function and piecewise saturation for large inputs, relevant to how tanh saturates to ±1.
- **Relevance**: high — provides the `save x, transform x, multiply x by result` structural template directly applicable to softcap's `save cap, compute tanh(x/cap), multiply by cap`. Also shows how to handle saturation boundaries.

### 3. tanhshrink
- **Why selected**: `tanhshrink` (`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_sfpu_kernel.cpp`) directly demonstrates how to use `tanh_tile_init()` and `tanh_tile(idst)` as building blocks inside a custom SFPU compute kernel. Softcap must invoke tanh as its core operation, and tanhshrink_sfpu_kernel.cpp is the closest existing model for a dedicated compute kernel that calls tanh on a DST register tile. It also demonstrates `init_sfpu(cb_input, cb_output)`, `tile_regs_acquire/commit/wait/release`, `copy_tile_init/copy_tile`, and multi-op DST register usage.
- **Relevance**: high — directly shows how to call tanh_tile_init()/tanh_tile() in a custom compute kernel, which is exactly what softcap's kernel will do after scaling the input.

### 4. atanh
- **Why selected**: `atanh` is a Wave 3 generated op (`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`) that uses the same init-time constant storage pattern via `sfpi::vConstFloatPrgm0`, `vConstFloatPrgm1`, `vConstFloatPrgm2`. A high-accuracy tanh implementation needs polynomial coefficients precomputed in `init()`, and `atanh` shows precisely this pattern — `atanh_init()` stores cubic polynomial coefficients into the programmable float constant registers. The atanh compute loop also shows how to use `sfpi::exexp(v)`, `sfpi::setexp(v, 127)`, `sfpi::int32_to_float()`, and Horner polynomial evaluation — all primitives that a tanh polynomial approximation would reuse.
- **Relevance**: high — provides the vConstFloatPrgm0/1/2 init pattern and Horner polynomial evaluation loop that a high-accuracy tanh kernel will use for its Taylor or minimax polynomial.

### 5. hardtanh
- **Why selected**: `hardtanh` (`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`) is the most relevant parameterized SFPU op in the LLK library. It takes two float parameters (`min_val`, `max_val`) packed as FP16_B uint32 values and decodes them inside the kernel using `sfpi::s2vFloat16b(param0)`. Softcap needs to receive the float scalar `cap` as a runtime argument packed as uint32 and decode it in the SFPU kernel. Hardtanh also shows how `is_parametrized_type()` is set in `unary_op_utils.hpp` and how the parameter is registered via `get_op_init_and_func_parameterized()` in `unary_op_utils.cpp`. This is the critical reference for the parameter-passing infrastructure.
- **Relevance**: high — provides the only existing example of a parameterized SFPU LLK kernel with float parameters passed as packed uint32, and shows the full registration pattern (is_parametrized_type, get_op_init_and_func_parameterized) required for softcap's `cap` parameter.
