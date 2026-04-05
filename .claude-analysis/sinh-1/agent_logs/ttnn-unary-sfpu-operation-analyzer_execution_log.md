# Execution Log: ttnn-unary-sfpu-operation-analyzer (cosh)

## 1. Metadata
- **Operation**: cosh
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output file**: `.claude-analysis/sinh-1/cosh_analysis.md`

## 2. Execution Timeline

### Phase 1: Dispatch Tracing
- Read `unary_op_utils.cpp` to determine compute kernel path, init/func strings, and approximation mode
- Found: `eltwise_sfpu.cpp`, `cosh_tile_init()` / `cosh_tile(idst)`, `APPROX=false`, macro `SFPU_OP_COSH_INCLUDE`

### Phase 2: Abstraction Layer Tracing
- Traced from `cosh.h` API header through `SFPU_THREE_PARAM_KERNEL_FP32_FIRST` macro to `_llk_math_eltwise_unary_sfpu_params_`
- Identified LLK params dispatch in `tt_llk` for both WH and BH
- Found core SFPU implementation in `ckernel_sfpu_cosh.h` (identical on WH and BH)

### Phase 3: Kernel Source Analysis
- Read `calculate_cosh()` -- simple formula: `(exp(v) + exp(-v)) * 0.5`
- Traced dependency to `_sfpu_exp_21f_bf16_` in shared `ckernel_sfpu_exp.h`
- Read `_float_to_int32_for_exp_21f_` helper
- Read `PolynomialEvaluator::eval` in `ckernel_sfpu_polyval.h` -- Horner's method
- Read `_init_exponential_` -- confirmed `FAST_APPROX=false` means no LOADMACRO setup

### Phase 4: Instruction Analysis
- Mapped all SFPI abstractions to hardware instructions via `sfpi_lib.h`
- Identified 11 distinct SFPU instruction types used
- Confirmed ADDR_MOD_7 (all increments = 0) on both WH and BH

### Phase 5: Verification
- Verified `calculate_cosh` and `cosh_init` function names exist in both WH and BH ckernel files
- Verified all 9 cited file paths exist on disk
- Confirmed WH and BH implementations are identical

### Phase 6: Analysis Writing
- Wrote complete analysis to `.claude-analysis/sinh-1/cosh_analysis.md`
- All required sections populated

## 3. Recovery Summary
No errors or recovery needed. Straightforward analysis.

## 4. Artifacts
| File | Action |
|------|--------|
| `.claude-analysis/sinh-1/cosh_analysis.md` | Created |
| `.claude-analysis/sinh-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl` | Appended |
| `.claude-analysis/sinh-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_execution_log.md` | Created |

## 5. Key Findings
- `cosh` uses the `_sfpu_exp_21f_bf16_` helper (Moroz et al. 2022 exp_21f algorithm) called twice per iteration
- No LOADMACRO fast-path is used (FAST_APPROX=false in init)
- WH and BH implementations are identical
- Kernel style: A_sfpi (pure SFPI abstractions, no raw TTI instructions)
- 11 distinct SFPU instructions emitted through SFPI compiler
