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

---

# Execution Log: ttnn-unary-sfpu-operation-analyzer (selu)

## Metadata
- **Operation**: selu
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|-----------|
| Operation name | selu | HIGH |
| UnaryOpType | SELU | HIGH |
| Output directory | .claude-analysis/rrelu-1/ | HIGH |
| Output filename | selu_analysis.md | HIGH |

## Execution Timeline

### Phase 1: Dispatch Tracing
- Read `unary_op_utils.cpp` to find `get_compute_kernel_path()` -- SELU uses default `eltwise_sfpu.cpp`
- Read `get_op_init_and_func()` -- SELU case at line 553: `selu_tile_init()` / `selu_tile(idst, param0, param1)`
- Read `get_op_approx_mode()` -- SELU falls through to `default: return false`
- Identified include guard: `SFPU_OP_SELU_INCLUDE`

### Phase 2: Abstraction Layer Tracing
- API Header: `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h`
- LLK Dispatch: `llk_math_eltwise_unary_sfpu_selu.h` (WH/BH identical)
- Core SFPU: `ckernel_sfpu_unary_selu.h` (WH/BH identical)
- Params Dispatch: `llk_math_eltwise_unary_sfpu_params.h` (from tt_llk)

### Phase 3: Kernel Source Analysis
- Read both WH and BH `ckernel_sfpu_unary_selu.h` -- confirmed identical
- Key finding: SFPI-based kernel (Style A), uses `v_if(v >= 0.0f)` / `v_else` branching
- Positive branch: simple `v * scale`
- Negative branch: calls `_sfpu_exp_21f_bf16_<true>(v)`, computes `(exp(x)-1) * alpha * scale`
- Read `ckernel_sfpu_exp.h` for the exp_21f algorithm (Moroz et al. 2022)
- Read `ckernel_sfpu_polyval.h` for PolynomialEvaluator (Horner's method)
- Traced SFPI intrinsics to `__builtin_rvtt_sfp*` builtins via `sfpi_lib.h`

### Phase 4: Instruction Mapping
- Mapped 17 distinct SFPU instructions used across the kernel and its helpers
- Confirmed no raw TTI instructions in the SELU kernel itself
- All SFPU instructions emitted by the SFPI compiler from high-level C++ abstractions

### Phase 5: Address Mode Analysis
- `SfpuType::selu` does not match any special-case branches in `eltwise_unary_sfpu_configure_addrmod`
- Only `ADDR_MOD_7` is configured: `{srca.incr=0, srcb.incr=0, dest.incr=0}`
- Identical on both Wormhole B0 and Blackhole

## External Services
| Service | Status | Fallback |
|---------|--------|----------|
| DeepWiki | FAILED (repository not indexed) | Source code analysis only |
| Confluence | Not consulted | N/A |
| Glean | Not consulted | N/A |

## Artifacts
| File | Action |
|------|--------|
| `.claude-analysis/rrelu-1/selu_analysis.md` | Created |
| `.claude-analysis/rrelu-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl` | Appended |
| `.claude-analysis/rrelu-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_execution_log.md` | Updated |

## SFPU Analysis Summary
- **Kernel style**: A_sfpi (pure SFPI abstractions)
- **Core logic**: SELU splits into two CC-guarded branches: positive (v * scale) and negative ((exp(v)-1) * alpha * scale)
- **Instructions**: 17 distinct SFPU instructions including SFPLOAD, SFPSTORE, SFPMAD, SFPLOADI, SFPSETCC, SFPENCC, SFPCOMPC, SFPPUSHC, SFPPOPC, SFPEXEXP, SFPEXMAN, SFPSHFT, SFPSETEXP, SFPSWAP, SFPCAST, SFP_STOCH_RND, SFPMUL
- **CC pattern**: v_if(v >= 0.0f) / v_else with GTE0 sign bit test
- **WH vs BH difference**: None -- implementations are identical
- **Approximation mode**: APPROXIMATION_MODE template parameter is not used in the function body; exp helper is always called with is_fp32_dest_acc_en=true
- **Notable**: The exp helper `_sfpu_exp_21f_bf16_` is called with hardcoded `is_fp32_dest_acc_en=true` to avoid premature FP16B rounding, even though the outer function defaults to `is_fp32_dest_acc_en=false`. Final rounding to FP16B is done explicitly after the full SELU formula.

---

# Execution Log: ttnn-unary-sfpu-operation-analyzer (leaky_relu)

## Metadata
- **Operation**: leaky_relu
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output File**: `.claude-analysis/rrelu-1/leaky_relu_analysis.md`

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|------------|
| Operation name | leaky_relu | HIGH |
| UnaryOpType | LEAKY_RELU | HIGH |
| Compute kernel | eltwise_sfpu.cpp (default) | HIGH |
| Include guard | SFPU_OP_RELU_FAMILY_INCLUDE | HIGH |
| Init function | leaky_relu_tile_init() | HIGH |
| Tile function | leaky_relu_tile(idst, slope) | HIGH |
| Approx mode | false | HIGH |

## Execution Timeline

### Phase 1: Dispatch Tracing
- Read `unary_op_utils.cpp` to find `get_macro_definition()` -> `SFPU_OP_RELU_FAMILY_INCLUDE`
- Read `get_op_init_and_func()` -> `leaky_relu_tile_init()` / `leaky_relu_tile(idst, param0u)`
- Read `get_compute_kernel_path()` -> falls through to default `eltwise_sfpu.cpp`
- Read `get_op_approx_mode()` -> falls through to `default: return false`

### Phase 2: Abstraction Layer Tracing
- Read API header `relu.h` -> `SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_lrelu, RC, APPROX, idst, slope)`
- Read macros header `llk_math_eltwise_unary_sfpu_macros.h` -> expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_lrelu<APPROX>, ...)`
- Read params dispatch (WH and BH) -> `VectorMode::RC` loop over 4 faces
- Read init function -> `eltwise_unary_sfpu_configure_addrmod()` sets ADDR_MOD_7 with zero increments

### Phase 3: Core SFPU Kernel Analysis
- Read WH `ckernel_sfpu_relu.h` (tt_llk) -> `_calculate_lrelu_` with raw TTI instructions
- Read BH `ckernel_sfpu_relu.h` (tt_llk) -> identical except ADDR_MOD_7 vs ADDR_MOD_3
- Identified 6 SFPU instructions: SFPLOADI, SFPLOAD, SFPSETCC, SFPMUL, SFPENCC, SFPSTORE
- Analyzed CC pattern: simple LT0 guard (SFPSETCC enables CC for negative lanes, SFPMUL is guarded, SFPENCC resets)
- Verified ADDR_MOD remapping on WH via `set_addr_mod_base()`

### Phase 4: Verification
- Verified all function names exist in source (grep)
- Verified all SFPU instructions exist in the kernel file (grep: 9 matches)
- Verified all cited file paths exist

## External Services
| Service | Status | Notes |
|---------|--------|-------|
| DeepWiki | UNAVAILABLE | Repository not indexed |
| Confluence | Not needed | SFPU Hardware Model Reference was sufficient |
| Glean | Not needed | All information found in source code |

## Artifacts
| File | Action |
|------|--------|
| `.claude-analysis/rrelu-1/leaky_relu_analysis.md` | Created |
| `.claude-analysis/rrelu-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl` | Appended |
| `.claude-analysis/rrelu-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_execution_log.md` | Updated (appended) |

## SFPU Analysis Summary
- **Kernel style**: B_raw_TTI (raw TTI instructions with simple CC pattern)
- **Core logic**: Leaky ReLU -- pass through non-negative values unchanged, multiply negative values by a scalar slope
- **Instructions**: SFPLOADI (x2, slope load), SFPLOAD, SFPSETCC (LT0), SFPMUL, SFPENCC (EU_R1), SFPSTORE
- **CC pattern**: Simple LT0 guard (SFPSETCC -> guarded SFPMUL -> SFPENCC reset)
- **WH vs BH difference**: Only ADDR_MOD numbering differs (WH: ADDR_MOD_3 remapped via base to slot 7; BH: ADDR_MOD_7 directly)
- **Approximation mode**: Has no effect -- kernel has no APPROXIMATION_MODE branches
