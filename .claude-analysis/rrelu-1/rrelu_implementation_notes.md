# RReLU Implementation Notes

## Overview

Implemented the RReLU (Randomized Leaky ReLU) unary SFPU operation across all abstraction layers:
- **Math definition**: `RReLU(x) = x if x >= 0, a * x if x < 0`
- **Eval mode** (default): `a = (lower + upper) / 2` (deterministic)
- **Training mode**: `a ~ Uniform(lower, upper)` per element (uses PRNG)

## Which Reference Operations Were Most Useful and Why

1. **leaky_relu** (most useful): RReLU eval mode is structurally identical to leaky_relu — same conditional multiply pattern with a single slope parameter. The SFPI `v_if (v < 0.0F) { dst_reg[0] = v * slope; } v_endif;` pattern was directly reused.

2. **dropout**: Essential for training mode implementation. Dropout's PRNG usage pattern (`TTI_SFPMOV(0, 9, LREG3, 8)` for random number generation, `init_prng_seed(seed)` for initialization) was adapted for generating random slopes.

3. **threshold**: Confirmed the `Converter::as_float()` pattern for uint32_t-to-float parameter conversion and the standard SFPI kernel structure.

4. **prelu_sfpu**: Validated the single-parameter conditional multiply pattern and standalone `SFPU_OP_INCLUDE` macro approach.

5. **hardtanh**: Demonstrated the two-parameter dispatch pattern through `get_op_init_and_func_parameterized()` and the technique of precomputing derived values on the host to simplify SFPU kernel logic.

## Implementation Details

### Eval Mode (training=False)
- Host computes `slope = (lower + upper) / 2.0` and passes as single uint32_t parameter
- SFPU kernel is identical to leaky_relu: pure SFPI with `v_if`/`v_endif`
- Uses `Converter::as_float()` for parameter decoding
- Works on all platforms (WH, BH, Quasar)

### Training Mode (training=True)
- Host passes `lower` and `upper` as two float-to-uint32_t parameters
- PRNG seeded during init via `init_prng_seed(seed)`
- SFPU kernel uses raw TTI instructions for PRNG + bit manipulation:
  1. Generate random uint32 via PRNG (`TTI_SFPMOV` special mode)
  2. Construct uniform float in [1.0, 2.0) by masking mantissa + setting exponent
  3. Use precomputed constants A=2*lower-upper, B=upper-lower to compute `slope = A + rand * B`
  4. Conditionally multiply negative elements by the random slope
- Uses `SFPAND`/`SFPOR` for bit manipulation (not available on Quasar with same API)
- Training mode is WH/BH only (Quasar has different instruction signatures)

### Dispatch Architecture
- `UnaryOpType::RRELU` added to enum
- Registered as parameterized type in `is_parametrized_type()`
- `get_op_init_and_func_parameterized()` selects mode based on param count:
  - 1 param → eval mode: `rrelu_tile_init(); rrelu_tile(idst, slope);`
  - 2 params → training mode: `rrelu_tile_init(seed); rrelu_tile(idst, lower, upper);`
- Uses dedicated `SFPU_OP_RRELU_INCLUDE` macro for conditional include

## Deviations from Standard Patterns

1. **Overloaded compute API functions**: `rrelu_tile()` has two overloads (1-param eval, 2-param training) and `rrelu_tile_init()` has two overloads (no-param eval, 1-param training with seed). This is unusual but follows C++ function overloading conventions.

2. **Mixed SFPI/TTI implementation**: Eval mode uses pure SFPI abstractions, while training mode uses raw TTI instructions. This is because training mode needs PRNG access (only available via raw `TTI_SFPMOV` special mode) and bit manipulation (`TTI_SFPAND`/`TTI_SFPOR`).

3. **Quasar training mode not implemented**: Quasar has different instruction signatures for `SFPAND`/`SFPOR` (2 params vs 4 params on WH/BH). Only eval mode kernel is provided for Quasar. Training mode would require a Quasar-specific implementation.

4. **PRNG seed derivation**: Training mode uses `lower_uint ^ upper_uint ^ 0xDEADBEEF` as the seed. This is deterministic per parameter combination. A production implementation would want a true random seed from the host.

## Known Limitations

1. **Training mode PRNG quality**: The uniform float generation uses mantissa bit masking to create floats in [1.0, 2.0), giving 23-bit mantissa precision. This is sufficient for BFloat16 output (which only has 7 mantissa bits).

2. **Quasar training mode**: Not implemented due to different SFPAND/SFPOR instruction signatures.

3. **Training mode determinism**: The PRNG seed is deterministic per (lower, upper) pair. Different tiles processed by different cores will use the same seed, producing correlated random sequences. For true randomness, each core would need a unique seed (e.g., seeded from core coordinates).

4. **Lower parameter sign assumption**: Training mode's `SFPSETSGN` approach for negation assumes `lower >= 0`. If lower is negative, the sign bit forcing would produce incorrect results. The standard RReLU definition requires `0 <= lower <= upper`, so this is valid for correct usage.

### New Files
tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_rrelu.h
tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_rrelu.h
tt_metal/third_party/tt_llk/tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_rrelu.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h

### Modified Files
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
ttnn/ttnn/experimental_loader/golden_functions.py
