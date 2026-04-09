# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Operation: atanh
## Date: 2026-04-09

### Summary
Analyzed the SFPU kernel implementation for the `atanh` unary operation. The kernel implements `atanh(x) = 0.5 * (ln(1+x) - ln(1-x))` using IEEE 754 exponent decomposition and a cubic minimax polynomial approximation for `ln(m)` on `[1, 2)`.

### Key Findings
1. **Compute kernel**: `eltwise_sfpu.cpp` (standard unary dispatch)
2. **SFPU_OP_CHAIN_0**: `atanh_tile(0)` with `SFPU_OP_ATANH_INCLUDE` split-include mechanism
3. **Approximation mode**: `false` (default), but kernel has no branching on this parameter
4. **Algorithm**: Decomposes `ln(y) = e * ln(2) + P(m)` where `P(m)` is a cubic minimax polynomial
5. **SFPU instructions per iteration**: ~18 (2x SFPEXEXP, 2x SFPSETEXP, 2x SFPCAST, ~10 SFPMAD, 1x SFPLOAD, 1x SFPSTORE)
6. **WH/BH identical**: Both hardware targets use the same ckernel_sfpu_atanh.h
7. **Address mode**: ADDR_MOD_7 (all increments = 0), standard DEST progression

### Files Created
- `.claude-analysis/softcap-1/atanh_analysis.md`

### Verification Steps
- Verified `calculate_atanh` function exists in both WH and BH ckernel directories
- Verified `atanh_init` function exists in both WH and BH ckernel directories
- Verified all file paths in abstraction layers table exist
- Verified SFPI intrinsic-to-instruction mappings via sfpi_lib.h

---

## Operation: sinh
## Date: 2026-04-09

### Summary
Analyzed the SFPU kernel implementation for the `sinh` unary operation. The kernel implements `sinh(x) = (exp(x) - exp(-x)) / 2` using:
1. A `exp_21f` helper (Moroz et al. 2022 algorithm for fast 2^z)
2. Taylor series fallback `sinh(x) ~ x + x^3/6` for `|x| < 0.5` to avoid catastrophic cancellation
3. Final BF16 rounding via `float_to_fp16b` for deterministic output

### Key Findings
1. **Compute kernel**: `eltwise_sfpu.cpp` (standard unary dispatch)
2. **SFPU_OP_CHAIN_0**: `sinh_tile(0)` with `SFPU_OP_SINH_INCLUDE` split-include mechanism
3. **Approximation mode**: `false` (default), kernel has no branching on APPROXIMATION_MODE
4. **Undefined symbol**: `_float_to_int32_positive_` is called twice in `exp_21f` but has no definition -- compilation blocker
5. **SFPU instructions per iteration**: Heavy SFPMAD (all float arithmetic), SFPDIVP2/SFPEXEXP/SFPEXMAN/SFPSETEXP (IEEE 754 manipulation), SFPCAST, SFPSTOCHRND, and CC instructions
6. **WH/BH identical**: Both hardware targets use the same ckernel_sfpu_sinh.h
7. **Address mode**: ADDR_MOD_7 (all increments = 0), standard DEST progression

### Files Created
- `.claude-analysis/softcap-1/sinh_analysis.md`

### Verification Steps
- Verified `calculate_sinh` function exists in both WH and BH ckernel directories
- Verified `sinh_init` function exists in both WH and BH ckernel directories
- Verified all file paths in abstraction layers table exist
- Verified SFPI intrinsic-to-instruction mappings via sfpi_lib.h
- Confirmed `_float_to_int32_positive_` is undefined across entire codebase (0 definitions found)

---

## Operation: hardtanh
## Date: 2026-04-09

### Summary
Analyzed the SFPU kernel implementation for the `hardtanh` unary operation. The kernel implements `clamp(x, min_val, max_val)` using an algebraic decomposition: three SFPMAD additions (shift by -min, shift by (min-max), shift by +max) interleaved with two conditional zeroing operations (v_if/v_endif for lanes below min and above max). This avoids explicit min/max instructions.

### Key Findings
1. **Compute kernel**: `eltwise_sfpu.cpp` (standard unary dispatch)
2. **SFPU_OP_CHAIN_0**: `hardtanh_tile_init(); hardtanh_tile(0, param0, param1)` (nuked -- inferred from pattern)
3. **Approximation mode**: `false` (default), APPROXIMATION_MODE template param accepted but unused (no conditional branches)
4. **Parametrized type**: `is_parametrized_type(HARDTANH) = true`, takes min_val and max_val floats from host
5. **Parameter encoding**: 3 FP16_B-packed uint32 params. param2 source comment has possible sign error vs mathematical derivation.
6. **SFPU instructions per iteration**: SFPLOAD(1) + SFPLOADI(2 conditional) + SFPMAD(3) + SFPSETCC(2) + SFPENCC(4) + SFPSTORE(1)
7. **WH/BH identical**: Both hardware targets use byte-identical ckernel_sfpu_hardtanh.h
8. **Address mode**: ADDR_MOD_7 (all increments = 0), standard DEST progression
9. **Nuked layers**: API header, LLK dispatch, and dispatch switch case all deleted. Core SFPU implementation preserved in tt_llk.

### Files Created
- `.claude-analysis/softcap-1/hardtanh_analysis.md`

### Verification Steps
- Verified `_calculate_hardtanh_` function exists in both WH and BH via grep
- Verified all file paths in abstraction layers table exist
- Verified all SFPU instruction patterns confirmed in source code
- Confirmed WH and BH ckernel files are byte-identical via diff
