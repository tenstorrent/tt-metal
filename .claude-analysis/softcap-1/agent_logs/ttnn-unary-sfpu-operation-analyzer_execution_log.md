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

---

## Operation: hardshrink
## Date: 2026-04-09

### Summary
Analyzed the SFPU kernel implementation for the `hardshrink` unary operation (`UnaryOpType::HARDSHRINK`). This is a non-standard operation that uses a custom compute kernel (`hardshrink_kernel_sfpu.cpp`) rather than the standard `eltwise_sfpu.cpp` with `SFPU_OP_CHAIN_0`. The kernel implements `x if |x| > lambda, 0 otherwise` using a two-pass algorithm with multiple SFPU operations.

### Key Findings
1. **Non-standard dispatch**: Uses custom compute kernel, not `SFPU_OP_CHAIN_0` pattern
2. **Two compute kernel variants**: `hardshrink_kernel_sfpu.cpp` (SFPU binary path) and `hardshrink_kernel.cpp` (FPU binary_dest_reuse path)
3. **Two-pass algorithm**: Pass 1 computes `a * 1(a + lambda < 0)` via tmp CB. Pass 2 computes `a * 1(a - lambda > 0)` and adds Pass 1 result.
4. **SFPU operations composed**: `fill_tile`, `ltz_tile`, `gtz_tile`, `add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`
5. **Core SFPU functions**: `_calculate_fill_`, `_calculate_comp_` (templated for ltz/gtz variants), `_calculate_sfpu_binary_` (for ADD/SUB/MUL)
6. **Nuked files**: API headers (comp.h, fill.h) and LLK dispatch layers removed, but ckernel implementations survive in tt_llk
7. **Approximation mode**: Irrelevant -- all SFPU functions used ignore `APPROXIMATION_MODE`

### Files Produced
- `.claude-analysis/softcap-1/hardshrink_analysis.md` -- SFPU kernel analysis

### Verification Steps
- All function names verified via grep: `_calculate_comp_`, `_calculate_fill_`, `_calculate_sfpu_binary_`, `_calculate_comp_init_flag_`, `_sfpu_is_fp16_zero_`
- All 12+ file paths verified to exist
- SFPU instruction patterns verified in kernel source files
