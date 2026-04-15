# Execution Log: ttnn-unary-sfpu-operation-analyzer (atanh)

## Session Summary
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Operation**: atanh
- **Final Status**: SUCCESS
- **Output File**: `.claude-analysis/softcap-1/atanh_analysis.md`

## Analysis Steps

### 1. Dispatch Tracing
- Read `unary_op_utils.cpp` to find ATANH dispatch configuration
- Compute kernel: `eltwise_sfpu.cpp` (default case)
- Approx mode: `false` (default case in `get_op_approx_mode`)
- Include guard: `SFPU_OP_ATANH_INCLUDE`
- SFPU_OP_CHAIN_0 expansion: `atanh_tile_init()` / `atanh_tile(idst)`

### 2. Abstraction Layer Tracing
- API header: `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`
  - `atanh_tile(idst)` -> `llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)`
  - `atanh_tile_init()` -> `llk_math_eltwise_unary_sfpu_atanh_init<APPROX>()`
- LLK dispatch: `llk_math_eltwise_unary_sfpu_atanh.h` (identical on WH and BH)
  - Calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_atanh<APPROXIMATE, 8>` and `VectorMode::RC`
- Core SFPU: `ckernel_sfpu_atanh.h` (identical on WH and BH)
  - SFPI-style kernel using vFloat, dst_reg, exexp, setexp, int32_to_float
- Params dispatch: `llk_math_eltwise_unary_sfpu_params.h` (WH and BH variants)
  - VectorMode::RC: loops 4 faces, calls sfpu_func once per face

### 3. SFPU Kernel Analysis
- **Algorithm**: atanh(x) = 0.5 * (ln(1+x) - ln(1-x))
- **Logarithm approximation**: IEEE 754 decomposition + cubic minimax polynomial
  - y = 2^e * m, ln(y) = e*ln(2) + P(m), P(m) = c0 + m*(c1 + m*(c2 + m*c3))
- **Init function**: Programs 3 constant registers with polynomial coefficients
- **Instructions emitted**: SFPLOAD, SFPMAD (chain), SFPEXEXP, SFPSETEXP, SFPCAST, SFPSTORE, SFPCONFIG (in init)
- **No condition code usage**: Straight-line computation, no branching
- **ADDR_MOD_7**: All-zero increments on both WH and BH

### 4. Verification
- All function names verified: `calculate_atanh` (2 matches), `atanh_init` (2 matches)
- All file paths verified to exist
- SFPI intrinsic-to-instruction mappings verified via `sfpi_lib.h`

## Key Findings
- The kernel is a clean SFPI implementation with no hardware-generation differences
- APPROXIMATION_MODE template parameter is accepted but unused (no branching on it)
- The main computation is dominated by SFPMAD instructions (Horner polynomial evaluation for two logarithms)
- Each iteration processes 32 elements, 8 iterations per face, 4 faces per tile = 1024 elements
