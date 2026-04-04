# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Metadata
- **Operation**: hardtanh
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output file**: `.claude-analysis/lgamma-1/hardtanh_analysis.md`

## Input Interpretation
- **Operation name**: `hardtanh` (confidence: HIGH -- explicitly provided)
- **Output directory**: `.claude-analysis/lgamma-1/` (confidence: HIGH -- explicitly provided)
- **Output filename**: `hardtanh_analysis.md` (confidence: HIGH -- explicitly provided)

## Execution Timeline

### Phase 1: Dispatch Tracing
- Read `unary_op_utils.cpp` and `.hpp` to find HARDTANH dispatch configuration
- Found: `eltwise_sfpu.cpp` compute kernel, `hardtanh_tile_init()` / `hardtanh_tile(idst, param0, param1)` SFPU chain
- Parameterized type with `min_val` (default -1.0f) and `max_val` (default 1.0f)
- Include guard: `SFPU_OP_HARDTANH_INCLUDE`
- Approximation mode: `false` (default case in `get_op_approx_mode`)

### Phase 2: Abstraction Layer Tracing
- API header: `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h`
- LLK dispatch: `llk_math_eltwise_unary_sfpu_hardtanh.h` (identical for WH and BH)
- Core SFPU: `ckernel_sfpu_hardtanh.h` (identical for WH and BH)
- Params dispatch: `llk_math_eltwise_unary_sfpu_params.h` in tt_llk (slightly different WH vs BH but same logic)

### Phase 3: Kernel Analysis
- Kernel style: SFPI-based (Style A)
- Key pattern: Two `v_if` conditional clamp blocks
- SFPU instructions: SFPLOADI, SFPLOAD, SFPMAD, SFPSETCC, SFPPUSHC, SFPPOPC, SFPMOV, SFPSTORE, SFPENCC
- Address mode: ADDR_MOD_7 with all-zero increments (no special cases for hardtanh)

### Phase 4: Verification
- All function names verified via grep (calculate_hardtanh, llk_math_eltwise_unary_sfpu_hardtanh, llk_math_eltwise_unary_sfpu_hardtanh_init)
- All file paths verified to exist
- WH and BH implementations confirmed identical

## Recovery Summary
No errors or recovery needed.

## Deviations
None.

## Artifacts
- `.claude-analysis/lgamma-1/hardtanh_analysis.md` -- main analysis output

## SFPU-Specific Analysis Summary
- **Kernel complexity**: Low (simple clamp operation)
- **CC usage**: Two independent `v_if` blocks, no nesting, max stack depth 1
- **APPROXIMATION_MODE**: Template parameter exists but is unused in kernel logic
- **Cross-architecture differences**: None -- WH and BH implementations are identical
- **Special considerations**: Parameters passed as bitcast uint32_t (IEEE 754 float bits)

---

# Execution Log: ttnn-unary-sfpu-operation-analyzer (selu)

## Metadata
- **Operation**: selu
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output file**: `.claude-analysis/lgamma-1/selu_analysis.md`

## Input Interpretation
- **Operation name**: `selu` (confidence: HIGH -- explicitly provided)
- **Output directory**: `.claude-analysis/lgamma-1/` (confidence: HIGH -- explicitly provided)
- **Output filename**: `selu_analysis.md` (confidence: HIGH -- explicitly provided)

## Execution Timeline

### Phase 1: Dispatch Tracing
- Read `unary_op_utils.cpp` to find SELU dispatch configuration
- Found: `eltwise_sfpu.cpp` compute kernel, `selu_tile_init()` / `selu_tile(idst)` SFPU chain
- Non-parameterized type (no extra params)
- Include guard: `SFPU_OP_SELU_INCLUDE`
- Approximation mode: `false` (default case in `get_op_approx_mode`)

### Phase 2: Abstraction Layer Tracing
- API header: `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h`
- LLK dispatch: `llk_math_eltwise_unary_sfpu_selu.h` (WH and BH identical)
- Core SFPU: `ckernel_sfpu_selu.h` (WH and BH identical)
- Params dispatch: `llk_math_eltwise_unary_sfpu_params.h` in tt_llk

### Phase 3: Kernel Analysis
- Kernel style: SFPI-based (Style A)
- Key pattern: Single `v_if(v < 0)` block for negative inputs, unconditional scale multiply
- Sub-functions: `_calculate_exponential_piecewise_` (APPROXIMATION_MODE=false path), `_sfpu_exp_` (Horner series + repeated squaring), `_sfpu_reciprocal_<2>` (quadratic estimate + 2 Newton-Raphson iterations)
- SFPU instructions: SFPLOAD, SFPSTORE, SFPMAD, SFPSETCC, SFPENCC, SFPPUSHC, SFPPOPC, SFPCOMPC, SFPLOADI, SFPEXEXP, SFPSETEXP, SFPNOT, SFPSETMAN, SFPSETSGN
- Address mode: ADDR_MOD_7 with all-zero increments

### Phase 4: Verification
- All function names verified via grep (calculate_selu, selu_init)
- All file paths verified to exist
- All helper functions verified (_calculate_exponential_piecewise_, _sfpu_exp_, _sfpu_reciprocal_, _init_exponential_, _init_sfpu_reciprocal_)
- WH and BH implementations confirmed identical

## Recovery Summary
No errors or recovery needed.

## Deviations
None.

## Artifacts
- `.claude-analysis/lgamma-1/selu_analysis.md` -- main analysis output

## SFPU-Specific Analysis Summary
- **Kernel complexity**: High (involves exponential + reciprocal sub-functions with nested CC)
- **CC usage**: Nested v_if blocks: outer `v_if(v < 0)` in calculate_selu, inner `v_if(in < 0)` in piecewise exp, `v_if(exp >= 0)` with `v_and` narrowing in _sfpu_exp_. Max CC stack depth ~3.
- **APPROXIMATION_MODE**: false -- takes the precise path in _calculate_exponential_piecewise_ (exp(|x|) + reciprocal for negative inputs)
- **Cross-architecture differences**: None -- WH and BH implementations are byte-identical
- **Special considerations**: Fixed constants alpha=1.6732632 and scale=1.0507009 are hardcoded (not user-configurable). Init configures vConstFloatPrgm0/1/2 for reciprocal polynomial.

---

# Execution Log: ttnn-unary-sfpu-operation-analyzer (hardsigmoid)

## Metadata
- **Operation**: hardsigmoid
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output file**: `.claude-analysis/lgamma-1/hardsigmoid_analysis.md`

## Input Interpretation
- **Operation name**: `hardsigmoid` (confidence: HIGH -- explicitly provided)
- **Output directory**: `.claude-analysis/lgamma-1/` (confidence: HIGH -- explicitly provided)
- **Output filename**: `hardsigmoid_analysis.md` (confidence: HIGH -- explicitly provided)

## Execution Timeline

### Phase 1: Dispatch Tracing
- Read `unary_op_utils.cpp` to find HARDSIGMOID dispatch configuration
- Found: `eltwise_sfpu.cpp` compute kernel, `hardsigmoid_tile_init()` / `hardsigmoid_tile(idst)` SFPU chain
- Non-parameterized type (no extra params)
- Include guard: `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` (default)
- Approximation mode: `false` (default case in `get_op_approx_mode`)

### Phase 2: Abstraction Layer Tracing
- API header: `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h`
- LLK dispatch: `llk_math_eltwise_unary_sfpu_hardsigmoid.h` (WH and BH identical)
- Core SFPU: `ckernel_sfpu_hardsigmoid.h` (WH and BH identical)
- Params dispatch: `llk_math_eltwise_unary_sfpu_params.h` in tt_llk

### Phase 3: Kernel Analysis
- Kernel style: SFPI-based (Style A)
- Key pattern: SFPMAD for linear transform, two `v_if` conditional clamp blocks for [0, 1] clamping
- SFPU instructions: SFPLOAD, SFPMAD, SFPSETCC, SFPPUSHC, SFPENCC, SFPLOADI, SFPPOPC, SFPMOV, SFPSTORE
- Address mode: ADDR_MOD_7 with all-zero increments

### Phase 4: Verification
- All function names verified via grep (calculate_hardsigmoid, llk_math_eltwise_unary_sfpu_hardsigmoid, llk_math_eltwise_unary_sfpu_hardsigmoid_init)
- All file paths verified to exist
- WH and BH implementations confirmed identical

## Recovery Summary
No errors or recovery needed.

## Deviations
None.

## Artifacts
- `.claude-analysis/lgamma-1/hardsigmoid_analysis.md` -- main analysis output

## SFPU-Specific Analysis Summary
- **Kernel complexity**: Low (piecewise linear with two clamp blocks)
- **CC usage**: Two independent `v_if` blocks, no nesting, max CC stack depth 1
- **APPROXIMATION_MODE**: Template parameter exists but is unused in kernel logic (no `if constexpr` branches)
- **Cross-architecture differences**: None -- WH and BH implementations are identical
- **Special considerations**: Uses `sfpi::vConst1` (Fixed Const 2 = 1.0f) for upper clamp. Algorithm is `max(0, min(1, x/6 + 0.5))`.
