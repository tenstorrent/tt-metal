# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Metadata
- **Operation**: hardtanh
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output file**: `.claude-analysis/hardswish-1/hardtanh_analysis.md`

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|------------|
| Operation name | hardtanh | HIGH |
| UnaryOpType | HARDTANH | HIGH |
| Output location | `.claude-analysis/hardswish-1/` | HIGH (explicitly specified) |

## Execution Timeline

1. **Dispatch trace**: Found HARDTANH in `unary_op_utils.cpp` -- parameterized type with `min_val`/`max_val`, uses `eltwise_sfpu.cpp`, macro `SFPU_OP_HARDTANH_INCLUDE`, approx_mode=false.
2. **API header read**: `hardtanh.h` delegates to `llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)`.
3. **LLK dispatch read**: Both WH and BH identical. Uses `_llk_math_eltwise_unary_sfpu_params_` with `VectorMode::RC` and two uint32 params.
4. **Core SFPU kernel read**: Both WH and BH identical. Simple SFPI-based clamping kernel -- load, two v_if comparisons for min/max clamping, store.
5. **Params dispatch read**: Confirmed 4-face loop with SETRWC-based face advancement (WH) / inc_dst_addr-based (BH).
6. **Init/addr_mod read**: ADDR_MOD_7 with all increments=0 for both architectures (no special case for hardtanh).
7. **Identifier verification**: All function names and file paths verified via grep.
8. **Analysis written**: Complete markdown file with all required sections.

## Recovery Summary
No errors or recoveries needed. Straightforward analysis of a simple SFPI kernel.

## Deviations
None. Standard analysis flow followed.

## Artifacts
| File | Action |
|------|--------|
| `.claude-analysis/hardswish-1/hardtanh_analysis.md` | Created |
| `.claude-analysis/hardswish-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl` | Appended |
| `.claude-analysis/hardswish-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_execution_log.md` | Created |

## Key Observations
- HARDTANH is one of the simplest SFPU kernels: two sequential v_if clamping blocks per iteration.
- The `APPROXIMATION_MODE` template parameter is accepted but never referenced in the kernel body.
- WH and BH implementations are byte-for-byte identical for both the core kernel and LLK dispatch.
- The kernel is parameterized (takes two runtime uint32 args for min/max bounds) unlike most unary ops.

---

# Execution Log: ttnn-unary-sfpu-operation-analyzer (hardsigmoid)

## Metadata
- **Operation**: hardsigmoid
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output file**: `.claude-analysis/hardswish-1/hardsigmoid_analysis.md`

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|------------|
| Operation name | hardsigmoid | HIGH |
| UnaryOpType | HARDSIGMOID | HIGH |
| Output directory | .claude-analysis/hardswish-1/ | HIGH |

## Execution Timeline

1. **Initialized breadcrumbs** and read reference files (sfpu-hardware-model.md, logging specs, diagram templates)
2. **Traced dispatch path**: `get_op_init_and_func()` returns `hardsigmoid_tile_init()` / `hardsigmoid_tile(idst)`. Compute kernel: `eltwise_sfpu.cpp`. Approx mode: `false`.
3. **Read API header**: `hardsigmoid.h` -- wraps `llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>(idst)`.
4. **Read LLK dispatch**: Both WH and BH identical -- delegates to `_llk_math_eltwise_unary_sfpu_params_` with `calculate_hardsigmoid<APPROX, 8>`.
5. **Read core SFPU kernel**: `ckernel_sfpu_hardsigmoid.h` -- piecewise linear: `max(0, min(1, x/6 + 0.5))`. Uses SFPI abstractions. WH and BH identical.
6. **Read params dispatch**: `llk_math_eltwise_unary_sfpu_params.h` for both architectures. WH uses `TTI_SETRWC`, BH uses `inc_dst_face_addr`.
7. **Read init/addr_mod**: `ADDR_MOD_7` with all-zero increments for both architectures.
8. **Verified all identifiers**: All file paths exist, all function names confirmed via grep.
9. **Wrote analysis file**: `hardsigmoid_analysis.md` with all required sections.

## Recovery Summary
No errors or recovery needed. Analysis was straightforward.

## Deviations
- The `SFPU_OP_HARDSIGMOID_INCLUDE` guard exists in `sfpu_split_includes.h`, but `get_macro_definition()` for HARDSIGMOID falls through to the default `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE`. This means the hardsigmoid header is included via the standard compute API path rather than the split includes mechanism.

## Artifacts
| File | Action | Description |
|------|--------|-------------|
| `.claude-analysis/hardswish-1/hardsigmoid_analysis.md` | Created | SFPU kernel analysis for hardsigmoid |
| `.claude-analysis/hardswish-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_execution_log.md` | Updated | Appended hardsigmoid execution log |

## Key Findings
- **Kernel style**: SFPI abstractions (Style A) -- `vFloat`, `dst_reg`, `v_if`/`v_endif`
- **Math**: Piecewise linear with FMA: `result = x * (1/6) + 0.5`, clamped to [0, 1]
- **Instructions**: SFPLOAD, SFPMAD, SFPLOADI, SFPSETCC, SFPENCC, SFPPUSHC, SFPPOPC, SFPMOV, SFPSTORE
- **CC pattern**: Two sequential `v_if` blocks for clamping (lower bound at 0, upper bound at 1)
- **APPROXIMATION_MODE**: Accepted as template parameter but never used in any conditional branch
- **Architecture parity**: WH and BH implementations are byte-identical

---

# Execution Log: ttnn-unary-sfpu-operation-analyzer (selu)

## Metadata
- **Operation**: selu
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output file**: `.claude-analysis/hardswish-1/selu_analysis.md`

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|------------|
| Operation name | selu | HIGH |
| UnaryOpType | SELU | HIGH |
| Output directory | .claude-analysis/hardswish-1/ | HIGH |

## Execution Timeline

1. **Initialized breadcrumbs** and read reference files (sfpu-hardware-model.md, logging specs, diagram templates)
2. **Traced dispatch path**: `get_op_init_and_func_default()` returns `selu_tile_init()` / `selu_tile(idst)`. Compute kernel: `eltwise_sfpu.cpp`. Approx mode: `false`. Include guard: `SFPU_OP_SELU_INCLUDE`.
3. **Read API header**: `selu.h` -- uses `SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_selu, RC, APPROX, idst)` and `SFPU_INIT_KERNEL_CALL(selu, ckernel::sfpu::selu_init, APPROX)`.
4. **Read LLK dispatch**: `llk_math_eltwise_unary_sfpu_selu.h` -- both WH and BH identical. Delegates to `_llk_math_eltwise_unary_sfpu_params_`.
5. **Read core SFPU kernel**: `ckernel_sfpu_selu.h` -- WH and BH identical. SFPI-based kernel implementing SELU formula with conditional negative branch.
6. **Read exponential sub-function**: `ckernel_sfpu_exp.h` from tt_llk -- `_calculate_exponential_piecewise_<false,false,false>()` takes non-approximate path using `_sfpu_exp_()` + `_sfpu_reciprocal_<2>()`.
7. **Read reciprocal sub-function**: `ckernel_sfpu_recip.h` from tt_llk -- quadratic initial estimate + 2 Newton-Raphson iterations.
8. **Read init function**: `selu_init()` calls `_init_exponential_<false,false,0x3F800000>()` which initializes reciprocal polynomial constants.
9. **Read params dispatch and ADDR_MOD config**: ADDR_MOD_7 with all-zero increments, standard VectorMode::RC 4-face loop.
10. **Verified all identifiers**: All function names and file paths confirmed via grep.
11. **Wrote analysis file**: `selu_analysis.md` with all required sections.

## Recovery Summary
No errors or recovery needed. The tt_llk submodule was empty in the worktree so shared SFPU files were read from the main repo at `/localdev/vignjatijevic/tt-metal/tt_metal/third_party/tt_llk/`.

## Deviations
- The worktree's `tt_metal/third_party/tt_llk/` submodule directory was empty. All shared tt_llk files (ckernel_sfpu_exp.h, ckernel_sfpu_recip.h) were read from the main repository instead.

## Artifacts
| File | Action | Description |
|------|--------|-------------|
| `.claude-analysis/hardswish-1/selu_analysis.md` | Created | SFPU kernel analysis for selu |
| `.claude-analysis/hardswish-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl` | Appended | 6 breadcrumb events (start, dispatch_traced, kernel_source_read, instruction_analysis_complete, analysis_written, complete) |
| `.claude-analysis/hardswish-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_execution_log.md` | Updated | Appended selu execution log |

## Key Findings
- **Kernel style**: SFPI abstractions (Style A) -- `vFloat`, `dst_reg`, `v_if`/`v_endif`
- **Math**: SELU(x) = scale * x for x >= 0; scale * alpha * (exp(x) - 1) for x < 0
- **Constants**: alpha = 1.6732632 (0x3FD63840), scale = 1.0507009 (0x3F868640) -- both hardcoded in FP32
- **Sub-function depth**: 3 levels: `calculate_selu` -> `_calculate_exponential_piecewise_` -> `_sfpu_exp_` + `_sfpu_reciprocal_<2>`
- **CC nesting**: Up to 3-4 levels deep due to nested conditional branches in the exponential sub-functions
- **Instructions**: SFPLOAD, SFPSTORE, SFPMAD, SFPLOADI, SFPSETCC, SFPENCC, SFPPUSHC, SFPPOPC, SFPCOMPC, SFPEXEXP, SFPSETEXP, SFPSETMAN, SFPNOT, SFPSETSGN, SFPIADD
- **Init path**: `_init_exponential_<false,false,0x3F800000>()` -> `_init_sfpu_reciprocal_<false>()` sets vConstFloatPrgm0/1/2 to reciprocal polynomial coefficients
- **Architecture parity**: WH and BH kernel implementations are identical. Only platform difference is in the params dispatch layer (WH uses set_addr_mod_base, BH does not).

---

# Execution Log: ttnn-unary-sfpu-operation-analyzer (softsign)

## Metadata
- **Operation**: softsign
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output file**: `.claude-analysis/hardswish-1/softsign_analysis.md`
- **Commit**: cf5ccac99be

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|------------|
| Operation name | softsign | HIGH |
| UnaryOpType | SOFTSIGN | HIGH |
| Output directory | .claude-analysis/hardswish-1/ | HIGH |

## Execution Timeline

1. **Initialized breadcrumbs** and read reference files (sfpu-hardware-model.md, logging specs, diagram templates).
2. **Traced dispatch path**: `get_op_init_and_func_default()` returns `softsign_tile_init()` / `softsign_tile(idst)`. Compute kernel: `eltwise_sfpu.cpp`. Macro: `SFPU_OP_SOFTSIGN_INCLUDE`. Approx mode: `false`.
3. **Read API header**: `softsign.h` delegates to `llk_math_eltwise_unary_sfpu_softsign<APPROX>(idst)`.
4. **Read LLK dispatch**: Both WH and BH identical. Dispatches to `_llk_math_eltwise_unary_sfpu_params_` with `calculate_softsign<APPROX, 8>` and `VectorMode::RC`. Init calls `softsign_init<APPROX>`.
5. **Read core SFPU kernel**: `ckernel_sfpu_softsign.h` -- WH and BH identical. Computes `x / (1 + |x|)` using `abs`, `vConst1`, `_sfpu_reciprocal_<2>`, and multiply.
6. **Read reciprocal helper (WH)**: `ckernel_sfpu_recip.h` -- quadratic polynomial initial estimate with `setman`/`SFPNOT` + 2 Newton-Raphson iterations. Init programs `vConstFloatPrgm0/1/2` with Sollya-optimized coefficients.
7. **Read reciprocal helper (BH)**: `ckernel_sfpu_recip.h` -- hardware `SFPARECIP` + 2 NR iterations with NaN guard via `v_if (t < 0)`. Init programs `vConstFloatPrgm0 = 2.0f`.
8. **Read params dispatch**: Standard 4-face iteration with SETRWC/inc_dst_addr between faces.
9. **Read addr_mod config**: `SfpuType::softsign` only configures `ADDR_MOD_7` (dest.incr=0).
10. **Verified all identifiers**: Function names, file paths, and SFPI intrinsic mappings all confirmed via grep.
11. **Wrote analysis file**: Complete with all required sections including architecture-specific reciprocal implementations.

## Recovery Summary
No errors or recovery needed. The `sfpu/ckernel_sfpu_recip.h` file was not found in the worktree (tt_llk submodule not checked out), but was located in the parent repo at `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_recip.h`.

## Deviations
None. Standard analysis flow followed.

## Artifacts
| File | Action | Description |
|------|--------|-------------|
| `.claude-analysis/hardswish-1/softsign_analysis.md` | Created | SFPU kernel analysis for softsign |
| `.claude-analysis/hardswish-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl` | Appended | Breadcrumb events |
| `.claude-analysis/hardswish-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_execution_log.md` | Updated | Appended softsign execution log |

## Key Findings
- **Kernel style**: SFPI abstractions (Style A) -- `vFloat`, `dst_reg`, `sfpi::abs`
- **Math**: `softsign(x) = x / (1 + |x|)` = `x * reciprocal(|x| + 1)`
- **Architecture divergence**: The core softsign kernel is identical on WH and BH, but the reciprocal helper differs significantly:
  - WH: Pure-software quadratic polynomial estimate + 2 Newton-Raphson iterations (uses SFPSETMAN, SFPNOT, SFPSETSGN, many SFPMADs)
  - BH: Hardware SFPARECIP instruction + 2 NR iterations with NaN guard (uses SFPARECIP, SFPMAD, v_if/v_endif CC manipulation)
- **APPROXIMATION_MODE**: `false` -- not used in softsign kernel body. Passed through to `_init_sfpu_reciprocal_<false>()` which on BH programs `vConstFloatPrgm0 = 2.0f`.
- **Init constants differ by architecture**: WH programs 3 polynomial coefficients; BH programs a single constant (2.0).
