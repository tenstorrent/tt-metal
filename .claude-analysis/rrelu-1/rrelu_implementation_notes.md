# Implementation Notes: rrelu

## Math Definition
RReLU(x) = x if x >= 0; a*x if x < 0
- Training mode (seed != 0): a is randomly sampled from Uniform(lower, upper) per element using hardware PRNG
- Eval mode (seed == 0): a = (lower + upper) / 2 (handled by passing midpoint as both lower and upper, or by the Python-level golden function)
- Default: lower=0.125, upper=0.333333

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h

### Modified Files
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
ttnn/ttnn/operations/unary.py

## Design Decisions

### Reference Operations Used
- **leaky_relu**: Most useful reference for the conditional branch pattern (CC set on sign bit, CC-guarded multiply, CC clear). The raw TTI instruction sequence (SFPSETCC/SFPMUL/SFPENCC) was directly adapted for the negative-element multiplication. HIGH usefulness.
- **rand**: Critical reference for PRNG access pattern (SFPMOV from RS[9], SFPSETSGN, SFPSETEXP, SFPADDI to construct uniform [0,1) float). The rand kernel's technique of reading PRNG -> setting exponent to 127 -> subtracting 1.0 was directly reused. HIGH usefulness.
- **selu**: Used as the template for multi-parameter registration (2+ params in get_op_init_and_func_parameterized, LLK dispatch with multiple args, custom C++ function declaration). MEDIUM usefulness.
- **prelu**: Confirmed the pattern for SFPI v_if conditional application of slope to negative values. Initially attempted SFPI-based approach before switching to raw TTI. LOW usefulness (superseded by leaky_relu raw TTI pattern).

### Key Design Choices
1. **Raw TTI kernel instead of SFPI**: The initial implementation mixed SFPI abstractions (v_if/v_endif, vFloat) with raw TTI instructions (SFPMOV for PRNG). This risked register allocation conflicts (SFPI compiler uses LREG0-3 internally). The final implementation uses purely raw TTI instructions, following the leaky_relu and rand patterns. This is safer and more predictable.

2. **PRNG seeding via rrelu_tile_init(seed)**: The init function calls init_prng_seed(seed) which is called per-tile (since it's part of SFPU_OP_CHAIN_0). This means the PRNG is re-seeded each tile with the same seed, producing the same random pattern per tile. This is a known limitation but is acceptable for deterministic testing. For production use, the seed should be varied per tile.

3. **3-parameter design**: lower (float), upper (float), seed (uint32_t bitcast to float). The seed is transported through the UnaryWithParam float parameter system using std::bit_cast<float>(seed). The init function receives the seed, and the tile function receives lower and upper.

4. **Subtraction via SFPMAD**: Computing range = upper - lower requires subtraction. Since there's no TTI subtract instruction, we load -1.0 into LREG3 and compute `lower * (-1.0) + upper` via SFPMAD. The result is then moved to LREG2 via SFPMOV to free LREG3 for the loop.

5. **Architecture differences**: Wormhole B0 uses ADDR_MOD_3 and requires TTI_SFPNOP after SFPADDI/SFPMAD. Blackhole uses ADDR_MOD_7 and omits the NOPs (improved pipeline forwarding).

## Known Limitations
- PRNG is re-seeded every tile (since rrelu_tile_init is called per tile via SFPU_OP_CHAIN_0). This means every tile gets the same random slope pattern. For truly different random values per tile, a custom compute kernel (like rand/dropout) would be needed.
- The eval mode (seed=0) still seeds the PRNG with 0 and generates random slopes. To get true eval behavior (fixed midpoint slope), the Python-level code should pass lower = upper = midpoint when seed == 0. The golden function handles this distinction.
- No bfloat16 rounding in the raw TTI path (the raw TTI SFPSTORE uses IMPLIED format which handles format conversion automatically via the DEST accumulator format).
- Each lane's PRNG advances independently, but all lanes advance unconditionally (even for positive elements where the random value is not used). This is by design and ensures consistent PRNG state progression.
