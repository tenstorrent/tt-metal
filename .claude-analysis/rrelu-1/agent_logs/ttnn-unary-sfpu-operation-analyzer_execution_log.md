# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Operation: prelu
## Date: 2026-03-31

### Summary
Analyzed the SFPU kernel implementation for the `prelu` (PRELU_SFPU) unary operation. The kernel uses SFPI abstractions (vFloat, dst_reg, v_if/v_endif) and implements a simple conditional multiply: elements less than zero are scaled by a user-provided slope parameter.

### Key Findings
- **Compute kernel**: `eltwise_sfpu.cpp` (default path)
- **SFPU kernel**: `ckernel_sfpu_prelu.h` (identical logic for WH and BH, only `#pragma GCC unroll` differs: 8 for WH, 0 for BH)
- **Kernel style**: SFPI-based (Style A) -- clean, readable SFPI code
- **APPROXIMATION_MODE**: `false` (from `get_op_approx_mode`), but unused -- the kernel has no approximation-dependent branches
- **Vector mode**: `VectorMode::RC` (all 4 faces processed)
- **Core instructions**: SFPLOAD, SFPLOADI, SFPXFCMPS, SFPPUSHC, SFPMUL, SFPPOPC, SFPSTORE, INCRWC
- **Address mode**: ADDR_MOD_7 with all zero increments (explicit advancement via dst_reg++ and SETRWC/inc_dst_addr between faces)

### External Service Issues
- DeepWiki returned HTTP 429 (rate limited). All analysis was performed from source code.

### Files Analyzed
1. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- dispatch configuration
2. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` -- parametrized type check
3. `tt_metal/hw/inc/api/compute/eltwise_unary/prelu.h` -- API header
4. `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h` -- BH SFPU kernel
5. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h` -- WH SFPU kernel
6. `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` -- macro definitions
7. `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` -- BH params dispatch
8. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` -- WH params dispatch
9. `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h` -- BH init/addrmod
10. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h` -- WH init/addrmod
11. `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_converter.h` -- Converter::as_float
12. `runtime/sfpi/include/sfpi.h` -- SFPI abstraction definitions (vFloat, dst_reg, v_if, operators)

### Verification
All SFPU function names, instruction names, and file paths were verified via grep. All passed.

### Output
- `.claude-analysis/rrelu-1/prelu_analysis.md`

---

## Operation: leaky_relu
## Date: 2026-03-31

### Summary
Analyzed the SFPU kernel implementation for the `leaky_relu` (LEAKY_RELU) unary operation. The kernel uses raw TTI instructions with a simple CC manipulation pattern (SFPSETCC for LT0 detection, CC-guarded SFPMUL, SFPENCC reset). Negative input values are multiplied by a slope parameter; non-negative values pass through unchanged.

### Key Findings
- **Compute kernel**: `eltwise_sfpu.cpp` (default path)
- **SFPU kernel**: `ckernel_sfpu_relu.h` -- function `_calculate_lrelu_` (identical logic for WH and BH; only ADDR_MOD slot differs: ADDR_MOD_3 on WH, ADDR_MOD_7 on BH)
- **Kernel style**: B_raw_TTI (raw TTI instructions with simple CC manipulation)
- **APPROXIMATION_MODE**: `false` (from `get_op_approx_mode`), but unused -- the kernel has no approximation-dependent branches
- **Vector mode**: `VectorMode::RC` (all 4 faces processed)
- **Core instructions**: TT_SFPLOADI, TTI_SFPLOAD, TTI_SFPSETCC, TTI_SFPMUL, TTI_SFPENCC, TTI_SFPSTORE
- **CC pattern**: Simple LT0 guard per iteration (SFPSETCC -> SFPMUL guarded -> SFPENCC reset)
- **Address mode**: ADDR_MOD_3 (WH) / ADDR_MOD_7 (BH), both with dest.incr=0 (explicit advancement via dst_reg++)
- **Instructions per tile**: ~162 (2 SFPLOADI + 4 faces x 8 iterations x 5 instructions)

### External Service Issues
- DeepWiki returned HTTP 429 (rate limited). All analysis was performed from source code.

### Files Analyzed
1. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- dispatch configuration
2. `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h` -- API header
3. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h` -- WH metal wrapper
4. `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h` -- BH metal wrapper
5. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h` -- WH core SFPU kernel
6. `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h` -- BH core SFPU kernel
7. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` -- WH params dispatch
8. `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` -- BH params dispatch
9. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h` -- WH init/addrmod
10. `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h` -- BH init/addrmod
11. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_instr_params.h` -- p_sfpu register constants
12. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_ops.h` -- TTI instruction definitions
13. `runtime/sfpi/include/sfpi_constants.h` -- SFPSETCC_MOD1 constant definitions

### Verification
All SFPU function names (`_calculate_lrelu_`, `calculate_lrelu`), instruction names (TTI_SFPLOAD, TTI_SFPSETCC, TTI_SFPMUL, TTI_SFPENCC, TTI_SFPSTORE, TT_SFPLOADI), and file paths were verified via grep. All passed.

### Output
- `.claude-analysis/rrelu-1/leaky_relu_analysis.md`
