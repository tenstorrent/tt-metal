# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Operation: atanh
## Status: SUCCESS

## Summary
Analyzed the SFPU kernel implementation for the `atanh` unary operation. The kernel computes `atanh(x) = 0.5 * (ln(1+x) - ln(1-x))` using IEEE 754 exponent decomposition and a cubic minimax polynomial approximation for the natural logarithm. The implementation uses SFPI abstractions throughout (Style A) and is identical across Wormhole and Blackhole architectures.

## Key Findings
- **Compute kernel**: `eltwise_sfpu.cpp` (standard unary dispatch)
- **SFPU_OP_CHAIN_0**: `atanh_tile(0)`
- **Approximation mode**: `APPROX=false`, but kernel has no branching on this parameter
- **Kernel style**: SFPI-based (vFloat, vInt, dst_reg, exexp, setexp, int32_to_float)
- **SFPU instructions**: SFPLOAD, SFPSTORE, SFPMAD (x14+ per iteration), SFPEXEXP (x2), SFPSETEXP (x2), SFPCAST (x2), SFPLOADI (x3 in init)
- **Address mode**: ADDR_MOD_7 with all-zero increments (same for WH and BH)
- **Polynomial coefficients**: c0=-1.5828, c1=2.3110, c2=-0.8691, c3=0.1416 (cubic minimax for ln(m) on [1,2))

## Output
- Analysis file: `.claude-analysis/softcap-1/atanh_analysis.md`

## Verification Steps Taken
1. Verified `calculate_atanh` function exists in both WH and BH ckernel files
2. Verified `atanh_init` function exists in both WH and BH ckernel files
3. Verified all SFPU intrinsics (exexp, setexp, int32_to_float, dst_reg, vConst1, vConstFloatPrgm0/1/2) appear in the kernel source
4. Verified all file paths cited in the abstraction layers table exist
5. Confirmed WH and BH implementations are identical (byte-for-byte same content)

---

## Operation: sinh
## Status: SUCCESS

## Summary
Analyzed the SFPU kernel implementation for the `sinh` unary operation. The kernel implements `sinh(x) = (exp(x) - exp(-x)) / 2` using a custom `exp_21f` helper (Moroz et al. 2022 2^z algorithm) with a Taylor fallback `x + x^3/6` for |x| < 0.5 to avoid catastrophic cancellation. Uses SFPI abstractions throughout (Style A) and is identical across Wormhole and Blackhole.

## Key Findings
- **Compute kernel**: `eltwise_sfpu.cpp` (default for SINH)
- **SFPU_OP_CHAIN_0**: `sinh_tile_init(); sinh_tile(0);`
- **math_approx_mode**: `false` (default switch)
- **APPROXIMATION_MODE template**: `false`; has no branching effect inside the kernel body
- **Kernel style**: SFPI-based (Style A)
- **SFPU instructions**: SFPLOAD, SFPSTORE, SFPMAD, SFPIADD, SFPLOADI, SFPDIVP2, SFPEXEXP, SFPEXMAN, SFPSETEXP, SFPSETSGN, SFPCAST, SFP_STOCH_RND, SFPSETCC, SFPENCC, SFPMOV
- **Address mode**: ADDR_MOD_7 (dest.incr=0) for both WH and BH -- same configuration
- **Key design**: Dual-path (exp-based for |x|>=0.5, Taylor for |x|<0.5); explicit BF16 output rounding via SFP_STOCH_RND; `_float_to_int32_positive_` is an IEEE bit-reinterpret used in the Moroz exp_21f algorithm

## Output
- Analysis file: `.claude-analysis/softcap-1/sinh_analysis.md`

## Verification Steps Taken
1. Verified `calculate_sinh` function exists in both WH and BH ckernel files
2. Verified `exp_21f` helper exists in both WH and BH ckernel files
3. Verified all SFPI intrinsics (addexp, exexp, exman9, setexp, setsgn, int32_to_float, float_to_fp16b) used in the kernel appear in sfpi_lib.h with correct instruction mappings
4. Verified all file paths cited in the abstraction layers table exist
5. Confirmed WH and BH ckernel_sfpu_sinh.h implementations are byte-for-byte identical
6. Confirmed `SfpuType::sinh` is registered in both WH and BH llk_sfpu_types.h
7. Confirmed ADDR_MOD_7 (not ADDR_MOD_6) is the only configured slot for sinh in eltwise_unary_sfpu_configure_addrmod

---

## Operation: tanhshrink
## Status: SUCCESS

## Summary
Analyzed the SFPU kernel implementation for the `tanhshrink` unary operation. Tanhshrink computes `x - tanh(x)` and is a composite operation: it first applies tanh via the unary SFPU path, then subtracts using either FPU binary or SFPU binary subtraction. The operation is in a partially-nuked state: the dispatch integration was removed from `unary_op_utils.cpp`, and the underlying tanh LLK/ckernel files (`ckernel_sfpu_tanh.h`, `llk_math_eltwise_unary_sfpu_tanh.h`) were deleted. Two orphaned compute kernel files exist but would not compile.

## Key Findings
- **Compute kernel**: Two orphaned dedicated kernel files (not using standard `eltwise_sfpu.cpp` dispatch)
  - `tanhshrink_kernel.cpp` -- FPU binary subtraction via `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>`
  - `tanhshrink_sfpu_kernel.cpp` -- SFPU binary subtraction via `sub_binary_tile(0, 1, 0)`
- **SFPU_OP_CHAIN_0**: Not applicable (dedicated kernels, not standard dispatch)
- **math_approx_mode**: `false` (default)
- **tanh_tile template**: `fast_and_approx=false` (default)
- **Kernel style**: SFPI-based (Style A) for the subtraction phase
- **SFPU instructions (subtraction)**: SFPLOAD, SFPMAD (vFloat subtraction), SFPSTORE
- **SFPU instructions (tanh, nuked)**: Would have used SFPNONLINEAR InstrMod=5 (hardware-accelerated tanh)
- **Address mode**: ADDR_MOD_7 with all-zero increments (same for WH and BH)

## Output
- Analysis file: `.claude-analysis/softcap-1/tanhshrink_analysis.md`

## Verification Steps Taken
1. Verified `_calculate_sfpu_binary_` function exists in both WH and BH ckernel_sfpu_binary.h
2. Verified `_sfpu_binary_init_` function exists in both WH and BH ckernel_sfpu_binary.h
3. Confirmed tanh LLK (`llk_math_eltwise_unary_sfpu_tanh`) does NOT exist anywhere in the codebase
4. Confirmed `ckernel_sfpu_tanh.h` does NOT exist anywhere in the codebase
5. Verified all file paths cited in the abstraction layers table exist
6. Confirmed WH and BH ckernel_sfpu_binary.h implementations are identical
7. Verified ADDR_MOD_7 configuration in llk_math_eltwise_binary_sfpu.h
