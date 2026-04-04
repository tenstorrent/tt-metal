# Execution Log: ttnn-unary-sfpu-operation-analyzer (silu)

## Metadata
- **Operation**: silu
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output file**: `.claude-analysis/hardswish-1/silu_analysis.md`

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|------------|
| Operation name | silu | HIGH |
| UnaryOpType | SILU | HIGH |
| Output directory | `.claude-analysis/hardswish-1/` | HIGH (explicit override) |

## Execution Timeline

### Phase 1: Dispatch Tracing
- Read `unary_op_utils.cpp` to find `get_op_init_and_func_default()` returns `silu_tile_init()` / `silu_tile(idst)`
- Confirmed compute kernel: `eltwise_sfpu.cpp` (default case)
- Confirmed approx_mode: `false` (default case in `get_op_approx_mode`)
- Confirmed macro define: `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` (default case)

### Phase 2: Abstraction Layer Tracing
- API header: `compute_kernel_api.h` lines 463-465
- LLK dispatch: Build-generated `llk_math_eltwise_unary_sfpu_silu.h` (not in source tree; found in `build_Debug/`)
- Core SFPU: Build-generated `ckernel_sfpu_silu.h` (differs from tt_llk source)
- Parameters dispatch: `llk_math_eltwise_unary_sfpu_params.h` in tt_llk

### Phase 3: Kernel Source Analysis
- **Critical finding**: The build-generated `ckernel_sfpu_silu.h` uses `_sfpu_sigmoid_()` (exp + reciprocal), NOT the `_sigmoid_piecewise_linear_positive_()` from the tt_llk source files
- Traced full dependency tree: silu -> sigmoid -> exp (exp_21f for bf16, exp_f32_accurate for fp32) -> reciprocal (WH: polynomial + NR, BH: SFPARECIP + NR)
- Read reciprocal implementations for both WH and BH, noting architectural differences (WH uses quadratic estimate, BH uses hardware SFPARECIP instruction)

### Phase 4: Identifier Verification
- All function names verified via grep: `calculate_silu`, `silu_init`, `_calculate_silu_`, `_sfpu_sigmoid_`, `_sfpu_reciprocal_`, `_sfpu_exp_21f_`, `_sfpu_exp_improved_`
- All file paths verified to exist on disk

## Recovery Summary
No errors or recoveries needed.

## Deviations
- **Source discrepancy discovered**: The tt_llk source tree has a different silu implementation than what the build system produces. Documented both, emphasizing the build version as the deployed one.

## Artifacts
| File | Action |
|------|--------|
| `.claude-analysis/hardswish-1/silu_analysis.md` | Created |
| `.claude-analysis/hardswish-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_silu_execution_log.md` | Created |

## Handoff Notes
- The silu operation is a composition: `silu(x) = x * sigmoid(x)` where `sigmoid(x) = 1 / (1 + exp(-x))`
- This involves significant instruction depth due to the exp and reciprocal sub-operations
- The WH and BH implementations differ in the reciprocal path: WH uses software quadratic estimate, BH uses hardware SFPARECIP
- The exp path differs based on dest accumulation mode: FP32 uses Cody-Waite + 7th-order Taylor, BF16 uses exp_21f algorithm
