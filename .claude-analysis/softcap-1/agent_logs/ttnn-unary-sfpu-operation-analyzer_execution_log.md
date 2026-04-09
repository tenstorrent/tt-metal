# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Operation: swish
## Date: 2026-04-09

### Summary
Analyzed the SFPU kernel implementation for the `swish` unary operation (`UnaryOpType::SWISH`). The kernel computes `swish(x) = x * sigmoid(x)` using a piecewise approximation of sigmoid with three segments: a degree-3 polynomial for `|x| <= 2.5`, linear interpolation for `2.5 < |x| <= 5.0`, and saturation to 1.0 for `|x| > 5.0`.

### Key Findings
1. **Kernel style**: SFPI-based (uses `vFloat`, `dst_reg`, `v_if`/`v_endif`, `sfpi::abs`, `sfpi::vConst1`)
2. **Approximation mode**: `APPROXIMATION_MODE=false` (no explicit case in `get_op_approx_mode`, default returns false). The kernel does not branch on this parameter.
3. **Hardware implementations**: Wormhole B0 and Blackhole are identical for all layers (LLK dispatch, core SFPU, params dispatch address mode config).
4. **SFPU instructions**: SFPLOAD, SFPSTORE, SFPABS, SFPLOADI, SFPMAD, SFPSETCC, SFPENCC, SFPPUSHC, SFPPOPC
5. **Address mode**: `ADDR_MOD_7` with all-zero increments (software-managed via `dst_reg++` and `SETRWC`)
6. **CC stack depth**: Maximum 1 (three sequential `v_if` blocks, none nested)

### Files Produced
- `.claude-analysis/softcap-1/swish_analysis.md` -- SFPU kernel analysis

### Verification Steps
- All function names verified via grep: `calculate_swish`, `llk_math_eltwise_unary_sfpu_swish`, `llk_math_eltwise_unary_sfpu_swish_init`
- All file paths verified to exist
- SFPI-to-instruction mappings verified: `abs()` -> `SFPABS`, `vConst1` -> Fixed Const 2 (1.0)
