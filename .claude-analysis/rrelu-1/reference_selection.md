# Reference Selection: rrelu

## Target Operation
- **Name**: rrelu (Randomized Leaky ReLU)
- **Math**: x if x >= 0, a * x if x < 0; training: a ~ Uniform(lower, upper); eval: a = (lower + upper) / 2
- **Key features**: conditional (piecewise), parameterized (lower, upper), RNG in training mode

## Selected References

### 1. threshold
- **Rationale**: Closest structural match -- conditional comparison (v_if) with parameter passing. Uses `Converter::as_float()` for uint32_t-to-float conversion. Shows the standard pattern for a simple conditional SFPU kernel with template parameters.
- **SFPU kernel**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_threshold.h`

### 2. dropout
- **Rationale**: Contains hardware RNG via `SFPMOV` instruction (instr_mod1=8, lreg_c=9), which is needed for rrelu's training mode. Also shows `init_prng_seed()` for PRNG initialization. Uses raw TTI instructions for performance-critical RNG path.
- **SFPU kernel**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h`

### 3. hardtanh
- **Rationale**: Multi-parameter conditional (3 params) with v_if branching. Shows how to pass multiple uint32_t parameters and convert them to vFloat via `s2vFloat16b()`. Demonstrates the pre-computed parameter optimization pattern.
- **SFPU kernel**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`

### 4. clamp
- **Rationale**: Multi-branch conditional (v_if/v_elseif/v_endif) with parameter conversion. Uses both `s2vFloat16a()` and `s2vFloat16b()` conversion functions, showing the difference between FP16_A and FP16_B formats.
- **SFPU kernel**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_clamp.h`

### 5. fill
- **Rationale**: Simplest SFPU kernel pattern, showing basic iteration, dst_reg access, and the Converter utility. Good baseline for understanding the minimal SFPU kernel structure.
- **SFPU kernel**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h`

## SELECTED_REFERENCES: threshold, dropout, hardtanh, clamp, fill
