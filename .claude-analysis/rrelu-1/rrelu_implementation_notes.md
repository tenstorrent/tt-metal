# RReLU Implementation Notes

## Operation Summary
RReLU (Randomized Leaky ReLU) computes:
- `f(x) = x` when `x >= 0`
- `f(x) = a * x` when `x < 0`

Where `a` is:
- **Eval mode**: `a = (lower + upper) / 2` (fixed slope)
- **Training mode**: `a ~ Uniform(lower, upper)` per element (random slope via SFPU PRNG)

Default parameters: `lower = 1/8 (0.125)`, `upper = 1/3 (0.333...)`, `training = False`.

## Reference Operations Used
- **leaky_relu / prelu_sfpu** (most useful): The eval mode is mathematically identical to leaky_relu. The kernel structure (v_if on negative, multiply by slope) was taken directly from the reconstructed `_calculate_lrelu_` pattern. The parameter passing pattern (bit-cast float to uint32) follows prelu_sfpu.
- **dropout** (critical for training mode): The SFPU PRNG mechanism (`SFPMOV` with `instr_mod1=8`, `lreg_c=9`) was adapted from dropout. The `init_prng_seed()` initialization pattern was also taken from dropout's `_init_dropout_`.
- **swish**: Used as the structural template for the SFPI-based kernel, LLK dispatch, and API header file organization.
- **hardtanh**: Used as the reference for multi-parameter dispatch through `get_op_init_and_func_parameterized` and the `is_parametrized_type` registration pattern.

## Deviations from Standard Patterns
1. **PRNG via SFPI**: The training mode uses `__builtin_rvtt_sfpmov(sfpi::vConst0.get(), SFPMOV_MOD1_CONFIG)` to access the hardware PRNG from within SFPI code. This is a novel approach not used by any existing operation. Dropout uses raw TTI for PRNG; we use the SFPI builtin to generate random bits, then `abs() + setexp(127) - 1.0` to produce uniform floats in [0, 1).
2. **Dual code paths**: The kernel has separate eval/training code paths selected by a runtime `if (seed_u != 0)` branch. This is uncommon for SFPU kernels but necessary because training mode requires PRNG initialization and per-element random slope generation.
3. **Seed generation**: The PRNG seed is deterministically derived from `lower ^ upper ^ 0xDEADBEEF` on the host side. For production training use, the seed should be randomized by the caller.

## Known Limitations
1. **Training mode randomness**: The PRNG seed is deterministic per (lower, upper) pair, meaning the same inputs produce the same random slopes. For true stochastic training, the host should vary the seed across forward passes.
2. **Precision of random slopes**: The SFPU PRNG generates random mantissa bits in the internal FP19 format (10 mantissa bits), giving ~1024 distinct slope values in [lower, upper). This exceeds bfloat16 precision (7 mantissa bits, 128 distinct values per binade).
3. **Training mode performance**: The training path generates PRNG values for ALL elements unconditionally (before the `v_if` conditional), then applies slopes only to negative elements. This wastes PRNG cycles on non-negative elements but avoids potential issues with CC-guarded PRNG state advancement.

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h

### Modified Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
ttnn/ttnn/operations/unary.py
