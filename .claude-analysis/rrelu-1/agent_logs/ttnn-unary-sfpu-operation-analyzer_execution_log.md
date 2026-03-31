# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Operation: rand
## Date: 2026-03-31

### Summary
Analyzed the SFPU kernel implementation for the `rand` compute API. This is a non-standard operation -- it is not a `UnaryOpType` but rather a standalone compute API (`rand_tile`/`rand_tile_init`) used by custom compute kernels.

### Key Findings
1. **PRNG Mechanism**: The hardware PRNG is accessed via `SFPMOV` instruction with `instr_mod1=8` and `lreg_c=9`, which generates a pseudorandom uint32 into the destination LREG.
2. **Normalization Strategy**: Random bits are normalized to [0, 1) via IEEE754 manipulation: set sign=0 (SFPSETSGN), set exponent=127 (SFPSETEXP), subtract 1.0 (SFPADDI). This is more efficient than division-based normalization.
3. **Wormhole vs Blackhole**: Blackhole omits the SFPNOP pipeline stalls after SFPADDI and SFPMAD, and uses named constants in SFPSTORE. The core algorithm is identical.
4. **PRNG Initialization**: `init_prng_seed()` writes to a config register and then executes 600 NOPs to settle the PRNG state -- this is a significant latency cost paid once.

### Files Read
- `tt_metal/hw/inc/api/compute/eltwise_unary/rand.h` (API header)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h` (WH SFPU kernel)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h` (BH SFPU kernel)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (macro definitions)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h` (init dispatch)
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (WH params dispatch)
- `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (BH params dispatch)
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h` (WH SFPU init/done)
- `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h` (BH SFPU init/done)
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel.h` (init_prng_seed WH)
- `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel.h` (init_prng_seed BH)
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_instr_params.h` (p_sfpu constants)
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h` (PRNG reference)
- `ttnn/cpp/ttnn/operations/uniform/device/kernels/compute_uniform.cpp` (usage example)
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/instructions/assembly.yaml` (instruction definitions)

### External Services
- **DeepWiki**: Unavailable (HTTP 429 Too Many Requests) -- all 3 attempts failed
- **Confluence**: Not consulted
- **Glean**: Not consulted

### Verification
All SFPU function names, instruction names, and file paths were verified via grep. All passed.

### Output
- Analysis file: `.claude-analysis/rrelu-1/rand_analysis.md`
