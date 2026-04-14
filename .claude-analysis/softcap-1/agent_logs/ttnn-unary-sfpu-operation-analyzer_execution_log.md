# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Operation: swish
## Date: 2026-04-14

### Summary
Successfully analyzed the SFPU kernel implementation for the `swish` unary operation. The kernel uses SFPI abstractions (Style A) with a hybrid polynomial+linear sigmoid approximation. Wormhole and Blackhole implementations are identical.

### Key Findings
1. **Kernel style**: SFPI-based (vFloat, dst_reg, v_if/v_endif, sfpi::abs)
2. **Algorithm**: Piecewise sigmoid approximation with 3 segments:
   - Degree-3 polynomial for |x| <= 2.5
   - Linear interpolation for 2.5 < |x| <= 5.0
   - Saturation to 1.0 for |x| > 5.0
   - Symmetry: sigmoid(x) = 1 - sigmoid(|x|) for x < 0
3. **APPROXIMATION_MODE**: Template parameter is present but entirely unused (no branching on it)
4. **Address mode**: ADDR_MOD_7 with all increments = 0 (SFPI handles progression internally)
5. **Instructions**: SFPLOAD, SFPSTORE, SFPABS, SFPMAD, SFPLOADI, SFPSETCC, SFPENCC, SFPPUSHC, SFPPOPC, SFPCOMPC, SFPMOV

### Files Produced
- `.claude-analysis/softcap-1/swish_analysis.md` -- Main SFPU analysis document
- `.claude-analysis/softcap-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl` -- Event breadcrumbs
- `.claude-analysis/softcap-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_execution_log.md` -- This file

### Verification Steps
- Verified all function names via grep: `calculate_swish`, `llk_math_eltwise_unary_sfpu_swish`, `llk_math_eltwise_unary_sfpu_swish_init`
- Verified all file paths exist for both WH and BH architectures
- Verified `sfpi::abs` emits SFPABS via `__builtin_rvtt_sfpabs`
- Verified `vConst1` maps to CREG_IDX_1 (Fixed Const 2 = 1.0)
- Confirmed WH and BH implementations are byte-for-byte identical

---

## Operation: frac
## Date: 2026-04-14

### Summary
Successfully analyzed the SFPU kernel implementation for the `frac` unary operation. The kernel computes `frac(x) = x - trunc(x)` using IEEE 754 mantissa bit masking with three conditional branches based on the unbiased exponent. Uses SFPI abstractions (Style A). Wormhole and Blackhole implementations are identical.

### Key Findings
1. **Kernel style**: SFPI-based (vFloat, vInt, vUInt, dst_reg, v_if/v_endif, exexp, reinterpret)
2. **Algorithm**: IEEE 754 mantissa bit masking with 3 cases:
   - exp < 0 (|x| < 1): trunc(x) = 0, frac = x
   - 0 <= exp < 23: mask fractional mantissa bits, frac = x - trunc(x)
   - exp >= 23: x is already integer, frac = 0
3. **APPROXIMATION_MODE**: Template parameter is present but entirely unused (no branching on it)
4. **Address mode**: ADDR_MOD_7 with all increments = 0 (SFPI handles progression internally)
5. **Instructions**: SFPLOAD, SFPSTORE, SFPEXEXP, SFPIADD, SFPSHFT, SFPAND, SFPADD/SFPMAD, SFPLOADI, SFPSETCC, SFPENCC, SFPPUSHC, SFPPOPC, SFPCOMPC
6. **Notable**: Uses compound conditional `v_if(exp >= 0 && exp < 23)` requiring CC stack operations

### Files Produced
- `.claude-analysis/softcap-1/frac_analysis.md` -- Main SFPU analysis document

### Verification Steps
- Verified function name `calculate_frac` exists in both WH and BH ckernel dirs
- Verified all 7 file paths exist (API header, LLK dispatch x2, core SFPU x2, params dispatch x2)
- Verified `sfpi::exexp` maps to `__builtin_rvtt_sfpexexp` (SFPEXEXP) via `sfpi_lib.h`
- Verified `sfpi::reinterpret` is a zero-cost type cast (no instruction emitted) via `sfpi_lib.h`
- Verified `vInt << vUInt` maps to `__builtin_rvtt_sfpshft_v` (SFPSHFT) via `sfpi.h`
- Verified `vInt & vInt` maps to `__builtin_rvtt_sfpand` (SFPAND) via `sfpi.h`
- Confirmed WH and BH implementations are identical

---

## Operation: hardtanh
## Date: 2026-04-14

### Summary
Analyzed the SFPU kernel implementation for the `hardtanh` unary operation. Found that while the ckernel SFPU implementation (`_calculate_hardtanh_`) exists in both Wormhole and Blackhole LLK directories, the full TTNN dispatch chain is incomplete -- there is no API header, no LLK dispatch function, and no case in `get_op_init_and_func_parameterized()`. The operation would throw at runtime if invoked through TTNN.

### Key Findings
1. **Dispatch chain incomplete**: HARDTANH has `UnaryOpType::HARDTANH` enum, Python binding, and ckernel SFPU implementation, but missing API header, LLK dispatch, and utils switch case.
2. **Kernel style**: SFPI-based (Style A) -- uses vFloat, dst_reg, v_if/v_endif abstractions.
3. **Algorithm**: Three-step arithmetic clamping using negated threshold offsets. Avoids direct comparison against thresholds; instead shifts values to test against zero.
4. **APPROXIMATION_MODE**: Template parameter accepted but unused -- single code path regardless.
5. **WH/BH parity**: Implementations are byte-for-byte identical.
6. **ADDR_MOD**: ADDR_MOD_7 with all-zero increments (standard unary default).
7. **Instructions**: SFPLOADI, SFPLOAD, SFPMAD, SFPSETCC, SFPENCC, SFPPUSHC, SFPPOPC, SFPSTORE

### Files Produced
- `.claude-analysis/softcap-1/hardtanh_analysis.md` -- Main SFPU analysis document

### Verification Steps
- Verified `_calculate_hardtanh_` exists in both WH and BH ckernel dirs
- Verified ckernel_sfpu_hardtanh.h file paths exist for both architectures
- Verified that no `hardtanh_tile` / `hardtanh_tile_init` API functions exist in `tt_metal/hw/inc/api/compute/eltwise_unary/`
- Verified that no `llk_math_eltwise_unary_sfpu_hardtanh.h` exists in either WH or BH llk_lib
- Confirmed WH and BH implementations are byte-for-byte identical
