# RReLU (Randomized Leaky ReLU) Implementation Notes

## Operation Definition
- **Evaluation mode**: `output = x if x >= 0; output = ((lower + upper) / 2) * x if x < 0`
- **Training mode**: `output = x if x >= 0; output = a * x if x < 0, where a ~ U(lower, upper)`
- **Parameters**: lower (default 0.125), upper (default 1/3), seed (0=eval, non-zero=training)

## Implementation Summary

RReLU is implemented as a standard UnaryOpType going through the UnaryProgramFactory pipeline.
It supports both evaluation mode (deterministic midpoint slope) and training mode (per-element
PRNG-generated random slopes).

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h`

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` (added `rrelu` to SfpuType enum)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` (added `rrelu` to SfpuType enum)
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` (added SFPU_OP_RRELU_INCLUDE guard)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` (added RRELU to UnaryOpType enum)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` (added RRELU to is_parametrized_type)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` (added get_macro_definition, get_op_init_and_func, string_to_unary_with_param)
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` (added rrelu function declaration)
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp` (added rrelu function implementation)
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` (added Python binding via bind_unary_composite_floats_with_default)
- `ttnn/ttnn/operations/unary.py` (added golden function for rrelu)

## Reference Operations Used

1. **leaky_relu** (most useful): Provided the core pattern for conditional multiplication of negative elements using SFPSETCC + CC-guarded SFPMUL + SFPENCC. The eval mode SFPU kernel directly follows this pattern using SFPI abstractions.

2. **prelu** (useful): Demonstrated the SFPI abstraction pattern (`vFloat`, `v_if`, `v_endif`) for the same conditional multiply. The eval mode kernel mirrors prelu's SFPI code.

3. **dropout/rand** (useful for training mode): Provided the PRNG infrastructure pattern:
   - `TTI_SFPMOV(0, 9, LREG, 8)` for hardware PRNG generation
   - `SFPSETSGN` + `SFPSETEXP(127)` + `SFPADDI(-1.0)` for normalizing to [0, 1)
   - `SFPMAD` for scaling to [lower, upper)
   - `init_prng_seed()` for PRNG seeding

4. **selu** (useful): Demonstrated the 2-parameter registration pattern through the unary pipeline (LLK dispatch with params, `get_op_init_and_func` with multiple hex params). Extended to 3 params for rrelu.

## Design Decisions

### Dual-mode SFPU kernel
- **Eval mode**: Uses SFPI abstractions (`vFloat`, `v_if`, `v_endif`) for simplicity and compiler optimization. Computes midpoint = (lower + upper) * 0.5 once, then applies as leaky_relu slope.
- **Training mode**: Uses raw TTI instructions for direct PRNG control. Cannot use SFPI for PRNG because there's no SFPI abstraction for `SFPMOV(PRNG)`.

### PRNG seeding
- The `rrelu_tile_init(seed)` is called per-tile via SFPU_OP_CHAIN_0. A `static bool` guard in the LLK dispatch ensures `init_prng_seed()` (which takes 600 SFPNOP cycles) is only called once.

### Parameter encoding
- 3 float params passed through the standard unary pipeline: lower, upper, seed
- `seed == 0.0f` → evaluation mode (deterministic midpoint)
- `seed != 0.0f` → training mode (seed value bit-cast to uint32 for PRNG)
- The Python API exposes `lower` and `upper` with defaults; the C++ layer currently hardcodes seed=0 (eval mode).

### Address modes
- WH training mode uses `ADDR_MOD_3` (following leaky_relu/dropout convention)
- BH training mode uses `ADDR_MOD_7` (following BH convention)
- Eval mode uses SFPI abstractions which handle address modes automatically

## Known Limitations

1. **Training mode not exposed in Python API**: The current `ttnn.rrelu()` Python binding only supports evaluation mode (seed=0). Training mode with PRNG can be accessed by constructing `UnaryWithParam{UnaryOpType::RRELU, {lower, upper, seed_float}}` directly in C++.

2. **Same seed across all cores**: In training mode, all cores receive the same PRNG seed (baked into compile-time defines). This means all cores start with identical PRNG state. The random patterns will diverge as each core processes different tiles, but initial elements may correlate.

3. **WH pipeline stalls**: The WH training mode kernel includes `TTI_SFPNOP` after `SFPADDI` and `SFPMAD` instructions (required for Wormhole's pipeline). BH does not need these stalls.
