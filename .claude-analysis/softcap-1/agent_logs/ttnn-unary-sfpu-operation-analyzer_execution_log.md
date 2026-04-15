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

---

# Execution Log: ttnn-unary-sfpu-operation-analyzer (swish)

## Session Summary
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Operation**: swish
- **Final Status**: SUCCESS
- **Output File**: `.claude-analysis/softcap-1/swish_analysis.md`

## Analysis Steps

### 1. Dispatch Tracing
- Read `unary_op_utils.cpp` to find SWISH dispatch configuration
- Compute kernel: `eltwise_sfpu.cpp` (default case)
- Approx mode: `false` (default case in `get_op_approx_mode`)
- Include guard: `SFPU_OP_SWISH_INCLUDE`
- SFPU_OP_CHAIN_0 expansion: `swish_tile_init()` / `swish_tile(idst)`

### 2. Abstraction Layer Tracing
- API header: `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h`
  - `swish_tile(idst)` -> `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)`
  - `swish_tile_init()` -> `llk_math_eltwise_unary_sfpu_swish_init<APPROX>()`
- LLK dispatch: `llk_math_eltwise_unary_sfpu_swish.h` (identical on WH and BH)
  - Calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_swish<APPROXIMATE, 8>` and `VectorMode::RC`
- Core SFPU: `ckernel_sfpu_swish.h` (identical on WH and BH)
  - SFPI-style kernel using vFloat, dst_reg, abs, v_if/v_endif, vConst1
- Params dispatch: `llk_math_eltwise_unary_sfpu_params.h` (WH and BH variants)
  - VectorMode::RC: loops 4 faces, calls sfpu_func once per face

### 3. SFPU Kernel Analysis
- **Algorithm**: swish(x) = x * sigmoid(x), with sigmoid approximated piecewise:
  - Segment 0 (|x| <= 2.5): degree-3 polynomial: sigmoid(|x|) = 0.5 + |x| * (0.2533 + |x| * (-0.01479 + |x| * -0.00747))
  - Segment 1 (2.5 < |x| <= 5.0): linear interpolation: sigmoid(|x|) = 0.0276 * |x| + 0.855
  - Segment 2 (|x| > 5.0): saturate sigmoid to 1.0
  - For x < 0: sigmoid(x) = 1 - sigmoid(|x|)
- **Instructions emitted**: SFPLOAD, SFPABS, SFPLOADI, SFPMAD (chain via mul/add), SFPPUSHC, SFPXFCMPS, SFPSETCC, SFPPOPC, SFPREADLREG, SFPSTORE
- **Three v_if conditional blocks**: ax > 2.5, ax > 5.0, x < 0.0 -- sequential (not nested)
- **ADDR_MOD_7**: All-zero increments on both WH and BH

### 4. Verification
- All function names verified: `calculate_swish` (2 matches WH/BH), `llk_math_eltwise_unary_sfpu_swish` (2 matches), `llk_math_eltwise_unary_sfpu_swish_init` (2 matches)
- All file paths verified to exist
- SFPI intrinsic-to-instruction mappings verified via `sfpi.h` and `sfpi_lib.h`

## Key Findings
- Swish uses a piecewise sigmoid approximation (polynomial + linear + saturation) rather than computing exp()
- Sign symmetry exploited: sigmoid(x) = 1 - sigmoid(|x|) for negative inputs
- APPROXIMATION_MODE template parameter is accepted but unused (no branching on it)
- WH and BH implementations are identical
- Three sequential v_if/v_endif blocks provide piecewise function selection via SFPU condition code stack
- Each iteration processes 32 elements, 8 iterations per face, 4 faces per tile = 1024 elements
