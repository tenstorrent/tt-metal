# Execution Log: ttnn-unary-sfpu-operation-analyzer -- SILU

## Metadata
- **Operation**: SILU
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output File**: `.claude-analysis/softsign-1/silu_analysis.md`

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|------------|
| Operation name | SILU | HIGH |
| UnaryOpType | `SILU` | HIGH |
| Dispatch path | `unary_ng` (not legacy unary) | HIGH |
| Compute kernel | `eltwise_sfpu.cpp` | HIGH |
| Approximation mode | `false` | HIGH |

## Execution Timeline

### Phase 1: Dispatch Tracing
- Located SILU in `unary_ng_op_utils.cpp` (line 60): `silu_tile_init()` / `silu_tile(idst)`
- Confirmed compute kernel: `eltwise_sfpu.cpp` (default case)
- Confirmed `get_op_approx_mode()` returns `false` for all ops
- No template parameters on init/func calls

### Phase 2: Abstraction Layer Tracing
- API Header: `compute_kernel_api.h` lines 463-465
- LLK Dispatch: `llk_math_eltwise_unary_sfpu_silu.h` (build-generated)
- Core SFPU: `ckernel_sfpu_silu.h` (build-generated, differs from third_party source)
- Params Dispatch: `llk_math_eltwise_unary_sfpu_params.h`

### Phase 3: Core SFPU Analysis
- **Key discovery**: Build-output `ckernel_sfpu_silu.h` uses `x * _sfpu_sigmoid_(x)` (clean sigmoid composition), while the third_party source uses an older piecewise-linear sigmoid approximation. The build version is authoritative.
- Traced sigmoid sub-function chain: `_sfpu_sigmoid_` -> `_sfpu_exp_accurate_` + `_sfpu_reciprocal_`
- Documented WH vs BH differences in reciprocal implementation (polynomial vs SFPARECIP hardware)
- Documented WH vs BH differences in silu_init (WH passes APPROXIMATION_MODE; BH always uses `sigmoid_init<false>()`)

### Phase 4: Instruction Verification
- All function names verified via grep
- All file paths verified via `test -f`
- No SFPABS in the active SILU path (removed from instruction table note)

## Recovery Summary
No errors or blockers encountered.

## Artifacts
| File | Action |
|------|--------|
| `.claude-analysis/softsign-1/silu_analysis.md` | Created |
| `.claude-analysis/softsign-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_silu_execution_log.md` | Created |

## Deviations
- The third_party tt_llk source for `ckernel_sfpu_silu.h` (WH/BH) contains a piecewise-linear sigmoid approximation (`_sigmoid_piecewise_linear_positive_` with POLYVAL5), which is NOT the version used in the actual build. The build-generated version uses `_sfpu_sigmoid_` (exact computation via exp + reciprocal). The analysis documents the build version as authoritative.

## Handoff Notes
- SILU is a composition operation: its SFPU kernel is thin (just `x * sigmoid(x)`) and delegates all complexity to the sigmoid sub-function.
- The sigmoid sub-function itself chains exp and reciprocal, making SILU one of the most instruction-heavy unary operations.
- Both WH and BH share the same top-level `calculate_silu` implementation; differences only emerge in the reciprocal sub-function (software polynomial on WH vs SFPARECIP hardware on BH).
