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

---

# Execution Log: ttnn-unary-sfpu-operation-analyzer (sinh)

## Session Summary
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Operation**: sinh
- **Final Status**: SUCCESS
- **Output File**: `.claude-analysis/softcap-1/sinh_analysis.md`

## Analysis Steps

### 1. Dispatch Tracing
- Read `unary_op_utils.cpp` to find SINH dispatch configuration
- Compute kernel: `eltwise_sfpu.cpp` (default case)
- Approx mode: `false` (default case in `get_op_approx_mode`)
- Include guard: `SFPU_OP_SINH_INCLUDE`
- SFPU_OP_CHAIN_0 expansion: `sinh_tile_init()` / `sinh_tile(idst)`

### 2. Abstraction Layer Tracing
- API header: `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h`
  - `sinh_tile(idst)` -> `llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)`
  - `sinh_tile_init()` -> `llk_math_eltwise_unary_sfpu_sinh_init<APPROX>()`
- LLK dispatch: `llk_math_eltwise_unary_sfpu_sinh.h` (identical on WH and BH)
  - Calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_sinh<APPROXIMATE, 8>` and `VectorMode::RC`
- Core SFPU: `ckernel_sfpu_sinh.h` (identical on WH and BH)
  - SFPI-style kernel using vFloat, vInt, dst_reg, v_if/v_endif, addexp, exexp, exman9, setexp, setsgn
- Params dispatch: `llk_math_eltwise_unary_sfpu_params.h` (WH and BH variants)
  - VectorMode::RC: loops 4 faces, calls sfpu_func once per face

### 3. SFPU Kernel Analysis
- **Algorithm**: sinh(x) = (exp(x) - exp(-x)) / 2
  - Exponentiation via `exp_21f` helper: computes 2^z using Moroz et al. 2022 fast exp2 algorithm
    - Scale z by 2^23 (addexp), add IEEE bias, decompose into exponent (exexp) and mantissa (exman9)
    - Polynomial refinement on mantissa, then reconstruct via setexp
  - Two calls to exp_21f: one for exp(x) via z_pos = x*log2(e), one for exp(-x) via z_neg = -z_pos
  - Underflow clamping: z values clamped to >= -127 to prevent 2^z underflow
  - Taylor fallback: for |x| < 0.5, uses sinh(x) ~ x + x^3/6 to avoid catastrophic cancellation
  - Final output: converted to bfloat16 via float_to_fp16b for deterministic rounding
- **Init function**: `sinh_init()` is empty -- no programmable constants needed
- **Instructions emitted**: SFPLOAD, SFPSTORE, SFPMAD (arithmetic), SFPLOADI (constants), SFPDIVP2 (addexp), SFPEXEXP (extract exp), SFPEXMAN (extract mantissa), SFPSETEXP (reconstruct), SFPSETSGN (abs value), SFPCAST (int32_to_float), SFP_STOCH_RND (float_to_fp16b), SFPIADD (integer add), CC instructions (v_if/v_endif)
- **Three v_if/v_endif blocks**: z_pos underflow clamp, z_neg underflow clamp, Taylor fallback for small |x|
- **ADDR_MOD_7**: All-zero increments on both WH and BH
- **Critical issue**: `_float_to_int32_positive_()` called twice in exp_21f but not defined anywhere in codebase

### 4. Verification
- All function names verified: `calculate_sinh` (2 matches WH/BH), `exp_21f` (2 matches WH/BH), `sinh_init` (2 matches WH/BH)
- All file paths verified to exist
- SFPI intrinsic-to-instruction mappings verified via `sfpi_lib.h`
- `_float_to_int32_positive_` NOT found in any file outside ckernel_sfpu_sinh.h -- marked as [UNVERIFIED]

## Key Findings
- The sinh kernel implements a two-branch approach: exp-based for |x| >= 0.5, Taylor for |x| < 0.5
- The exp_21f helper is a custom fast 2^z implementation (Moroz et al. 2022) using bit manipulation
- APPROXIMATION_MODE template parameter is accepted but unused -- no `if constexpr (APPROXIMATION_MODE)` branches exist
- WH and BH implementations are identical
- **Missing dependency**: `_float_to_int32_positive_()` is called but not defined anywhere in the codebase, which would cause a compilation error
- Each iteration processes 32 elements, 8 iterations per face, 4 faces per tile = 1024 elements

---

# Execution Log: ttnn-unary-sfpu-operation-analyzer (hardtanh)

## Session Summary
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Operation**: hardtanh
- **Final Status**: SUCCESS
- **Output File**: `.claude-analysis/softcap-1/hardtanh_analysis.md`

## Analysis Steps

### 1. Dispatch Tracing
- Read `unary_op_utils.hpp` and `.cpp` for HARDTANH dispatch configuration
- Compute kernel: `eltwise_sfpu.cpp` (default case in `get_compute_kernel_path`)
- Approx mode: `false` (default case in `get_op_approx_mode`)
- **Dispatch chain INCOMPLETE**: HARDTANH is in `is_parametrized_type()` but has no case in `get_op_init_and_func_parameterized()` -- would throw at runtime
- No `hardtanh_tile()` API header exists
- No `SfpuType::hardtanh` in the production `llk_sfpu_types.h`
- Core SFPU kernel `_calculate_hardtanh_` exists at tt_llk level

### 2. Abstraction Layer Tracing
- API header: Does not exist
- LLK dispatch: Does not exist
- Core SFPU: `ckernel_sfpu_hardtanh.h` (identical on WH and BH)
  - SFPI-style kernel with shifted-addition clamping algorithm
- Params dispatch: Generic `llk_math_eltwise_unary_sfpu_params.h` (WH and BH)

### 3. SFPU Kernel Analysis
- **Algorithm**: `hardtanh(x) = clamp(x, low, high)` implemented via shifted-addition clamping
  - Three adds with two conditional-zero steps reconstruct the clamped value
  - Takes 3 pre-computed FP16_B parameters
- **SFPI-style kernel**: Uses `vFloat`, `dst_reg`, `v_if`/`v_endif`
- **Instructions emitted** (via SFPI compiler): SFPLOAD, SFPLOADI, SFPMAD (3 per iter), SFPSETCC (2 per iter), SFPENCC (4 per iter), SFPPUSHC/SFPPOPC (2 pairs per iter), SFPMOV (2 per iter), SFPSTORE
- **APPROXIMATION_MODE**: Accepted but unused (no branching)
- **ADDR_MOD_7**: dest.incr=0 on both WH and BH

### 4. Verification
- `_calculate_hardtanh_` verified: 2 matches (WH and BH)
- All file paths verified to exist
- Algorithm correctness verified via algebraic trace (param comments appear incorrect; p2 must be positive for correctness)

## Key Findings
- HARDTANH has a core SFPU kernel but the full dispatch chain is not wired in this worktree
- The kernel uses an elegant shifted-addition clamping technique that avoids explicit comparisons against threshold values
- WH and BH implementations are byte-for-byte identical
- The source code comment `param2 = -(pos_threshold)` is inconsistent with the algorithm producing correct results; the actual encoding likely has param2 = pos_threshold (positive)
- APPROXIMATION_MODE template parameter is accepted but has zero effect on execution

---

# Execution Log: ttnn-unary-sfpu-operation-analyzer (tanhshrink)

## Session Summary
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Operation**: tanhshrink
- **Final Status**: SUCCESS
- **Output File**: `.claude-analysis/softcap-1/tanhshrink_analysis.md`

## Analysis Steps

### 1. Dispatch Tracing
- Read `unary_op_utils.cpp` -- TANHSHRINK not in any switch cases
- `get_compute_kernel_path()` returns `eltwise_sfpu.cpp` (default) -- no TANHSHRINK case
- `get_op_init_and_func_default()` has no TANHSHRINK case -- would throw
- Found dedicated compute kernel: `tanhshrink_kernel.cpp` (does not use SFPU_OP_CHAIN)
- Approx mode: `false` (default)

### 2. Abstraction Layer Tracing
- API header: `compute_kernel_api.h` -- `tanh_tile<false>()` calls `llk_math_eltwise_unary_sfpu_tanh<false>(idst)`
- LLK dispatch: **DELETED** -- `llk_math_eltwise_unary_sfpu_tanh.h` removed in deep nuke
- Core SFPU: **DELETED** -- `ckernel_sfpu_tanh.h` removed in deep nuke Phase 1
- Binary subtract API: `eltwise_binary.h` -- `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>`
- Params dispatch: `llk_math_eltwise_unary_sfpu_params.h` (still exists)

### 3. SFPU/FPU Kernel Analysis
- **Algorithm**: `tanhshrink(x) = x - tanh(x)` implemented as:
  - Phase 1 (SFPU): `copy_tile` + `tanh_tile(0)` -- computes tanh(x) in DEST
  - Phase 2 (FPU): `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>` -- moves DEST (tanh(x)) to SRCB, unpacks x from CB to SRCA, FPU computes x - tanh(x)
- **Tanh SFPU impl deleted**: `ckernel_sfpu_tanh.h` was in exponential-composition family, described as "sigmoid via tanh"
- **No SFPNONLINEAR on WH/BH**: Hardware tanh (`p_sfpnonlinear::TANH_MODE = 0x5`) is Quasar-only
- **FPU subtract**: Uses `MathFidelity::LoFi` (sufficient for exact subtraction)

### 4. Verification
- `tanh_tile()` API verified in `compute_kernel_api.h`
- `binary_dest_reuse_tiles()` API verified in `eltwise_binary.h`
- `tanhshrink_kernel.cpp` verified to exist
- `ckernel_sfpu_tanh.h` confirmed DELETED (DEEP_NUKE_MANIFEST.md Phase 1)
- `SFPNONLINEAR` confirmed absent from WH/BH instruction sets
- `p_sfpnonlinear::TANH_MODE` confirmed in Quasar `ckernel_instr_params.h`

## Key Findings
- TANHSHRINK is a hybrid SFPU+FPU operation: SFPU computes tanh, FPU computes the subtraction
- The dedicated compute kernel `tanhshrink_kernel.cpp` exists but references deleted LLK functions
- Dispatch routing is broken: `get_compute_kernel_path()` has no TANHSHRINK case
- Double breakage: even if routing were fixed, the SFPU tanh implementation is deleted
- The `DEST_TO_SRCB` reuse mechanism is a key design choice -- it avoids extra circular buffer tiles by re-reading the input from the same CB
- Hardware tanh acceleration is Quasar-only (`SFPNONLINEAR TANH_MODE`)
