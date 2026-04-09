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
