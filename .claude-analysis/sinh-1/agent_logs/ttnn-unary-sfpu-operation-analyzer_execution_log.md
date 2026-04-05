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

---

# Execution Log: ttnn-unary-sfpu-operation-analyzer (selu)

## 1. Metadata
- **Operation**: selu
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output file**: `.claude-analysis/sinh-1/selu_analysis.md`

## 2. Execution Timeline

### Phase 1: Dispatch Tracing
- Read `unary_op_utils.cpp` to find SELU dispatch configuration
- Found: `get_macro_definition(SELU)` returns `SFPU_OP_SELU_INCLUDE`
- Found: `get_op_init_and_func_default(SELU)` returns `selu_tile_init()` / `selu_tile({idst})`
- Found: `get_op_approx_mode(SELU)` returns `false` (default case)
- Found: `get_compute_kernel_path(SELU)` returns `eltwise_sfpu.cpp` (default case)

### Phase 2: Abstraction Layer Tracing
- Read API header: `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h`
- Read LLK dispatch: `llk_math_eltwise_unary_sfpu_selu.h` (WH and BH identical)
- Read core SFPU: `ckernel_sfpu_selu.h` (WH and BH identical)
- Read macros: `llk_math_eltwise_unary_sfpu_macros.h` for `SFPU_UNARY_NO_PARAM_KERNEL_FN` and `SFPU_INIT_KERNEL_CALL`

### Phase 3: Kernel Source Analysis
- Read `calculate_selu()` -- conditional: positive lanes get `scale * x`, negative lanes get `scale * alpha * (exp(x) - 1)`
- Traced dependency to `_calculate_exponential_piecewise_` in shared `ckernel_sfpu_exp.h`
- Read `_sfpu_exp_` (Horner series with repeated squaring) in same file
- Read `_sfpu_reciprocal_<2>` (Newton-Raphson, 2 iterations) in `ckernel_sfpu_recip.h`
- Read `_init_exponential_<false,false,0x3F800000>` -- confirmed non-approx path calls `_init_sfpu_reciprocal_<false>()`

### Phase 4: Instruction Analysis
- Kernel style: A_sfpi (SFPI abstractions: vFloat, dst_reg, v_if/v_endif)
- Identified 14 distinct SFPU instruction types emitted by SFPI compiler
- CC pattern: v_if/v_endif with nested conditionals in helpers (exp, reciprocal)
- ADDR_MOD_7 (all increments = 0) on both WH and BH

### Phase 5: Verification
- Verified all function names via grep: calculate_selu, selu_init, _calculate_exponential_piecewise_, _sfpu_exp_, _sfpu_reciprocal_, _init_exponential_, _init_sfpu_reciprocal_
- Verified all file paths exist on disk
- Confirmed WH and BH implementations identical for ckernel_sfpu_selu.h

### Phase 6: Analysis Writing
- Wrote complete analysis to `.claude-analysis/sinh-1/selu_analysis.md`
- All required sections populated

## 3. Recovery Summary
No errors or recovery needed.

## 4. Artifacts
| File | Action |
|------|--------|
| `.claude-analysis/sinh-1/selu_analysis.md` | Created |
| `.claude-analysis/sinh-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl` | Appended |
| `.claude-analysis/sinh-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_execution_log.md` | Updated |

## 5. Key Findings
- SELU uses the non-approximate exponential path: `_sfpu_exp_()` + `_sfpu_reciprocal_<2>()` for negative inputs
- Fixed constants (not user-configurable): alpha=1.6732632 (0x3FD63840), scale=1.0507009 (0x3F868640)
- Init loads reciprocal polynomial coefficients into programmable constant registers (k0=0.3232325, k1=1.4545459, k2=2.1212124)
- WH and BH implementations are identical
- Kernel style: A_sfpi (pure SFPI abstractions, no raw TTI instructions)
- 14 distinct SFPU instructions emitted through SFPI compiler
