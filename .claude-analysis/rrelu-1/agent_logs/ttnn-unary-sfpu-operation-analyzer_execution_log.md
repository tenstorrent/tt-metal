# Execution Log: ttnn-unary-sfpu-operation-analyzer (prelu_sfpu)

## Metadata
- **Operation**: prelu_sfpu
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Start time**: 2026-04-03T09:18:00+00:00 (approx)
- **End time**: 2026-04-03T09:20:30+00:00 (approx)

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|------------|
| Operation name | prelu_sfpu | HIGH |
| UnaryOpType | PRELU_SFPU | HIGH |
| Output location | .claude-analysis/rrelu-1/ | HIGH |

## Execution Timeline

### Phase 1: Dispatch Tracing
- Read `unary_op_utils.cpp` to find include guard (`SFPU_OP_PRELU_INCLUDE`) and compute kernel path (default: `eltwise_sfpu.cpp`)
- Read `get_op_init_and_func()` to find the SFPU_OP_CHAIN_0 expansion: `prelu_tile_init()` / `prelu_tile(idst, param0)`
- Confirmed `get_op_approx_mode()` returns `false` (default case)

### Phase 2: Kernel Source Read
- Read API header: `tt_metal/hw/inc/api/compute/eltwise_unary/prelu.h`
  - `prelu_tile()` calls `SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_prelu, RC, APPROX, idst, param0)`
  - `prelu_tile_init()` calls `SFPU_UNARY_KERNEL_INIT(prelu, APPROX)`
- Read macros: `llk_math_eltwise_unary_sfpu_macros.h` -- traced macro expansion to `_llk_math_eltwise_unary_sfpu_params_<false>(...)`
- Read params dispatch: `llk_math_eltwise_unary_sfpu_params.h` -- confirmed VectorMode::RC with 4-face loop
- Read init: `llk_math_eltwise_unary_sfpu_init.h` -> `_llk_math_eltwise_unary_sfpu_init_<SfpuType::prelu>()`
- Read addr_mod config: `llk_math_eltwise_unary_sfpu.h` -- only ADDR_MOD_7 (all zeros) configured for prelu
- Read SFPU kernel: `ckernel_sfpu_prelu.h` for both Wormhole and Blackhole

### Phase 3: Instruction Analysis
- Kernel style: SFPI (vFloat, dst_reg, v_if/v_endif)
- Traced SFPI C++ abstractions to underlying SFPU instructions via `sfpi.h`:
  - `Converter::as_float(value)` + vFloat constructor -> SFPLOADI
  - `dst_reg[0]` read -> SFPLOAD
  - `a < 0.0f` -> SFPSETCC (LT0 sign bit test)
  - `v_if` -> SFPPUSHC + SFPENCC
  - `a * init` -> SFPMUL
  - `v_endif` -> SFPPOPC
  - `dst_reg[0] = a` -> SFPSTORE
  - `dst_reg++` -> INCRWC

### Phase 4: Analysis Written
- Output file: `.claude-analysis/rrelu-1/prelu_sfpu_analysis.md`
- All sections populated: dispatch summary, call chain, annotated source, instruction table, register usage, address mode config

## Recovery Summary
No errors or recovery needed. Straightforward SFPI kernel analysis.

## Deviations
- DeepWiki was unavailable (repository not indexed). All analysis was performed from source code.

## Artifacts
| File | Action |
|------|--------|
| `.claude-analysis/rrelu-1/prelu_sfpu_analysis.md` | Created |
| `.claude-analysis/rrelu-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl` | Appended |
| `.claude-analysis/rrelu-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_execution_log.md` | Created |

## SFPU Analysis Summary
- **Kernel style**: A_sfpi (pure SFPI abstractions)
- **Core logic**: PReLU -- pass through positive values unchanged, multiply negative values by a scalar slope
- **Instructions**: SFPLOADI, SFPLOAD, SFPSETCC (LT0), SFPMUL, SFPSTORE, SFPPUSHC, SFPPOPC, SFPENCC, INCRWC
- **CC pattern**: Simple v_if guard on sign bit (LT0)
- **WH vs BH difference**: Only unroll pragma differs (8 vs 0); logic is identical
- **Approximation mode**: Has no effect -- kernel has no APPROXIMATION_MODE branches
