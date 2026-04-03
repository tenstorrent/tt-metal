# Execution Log: ttnn-unary-sfpu-operation-analyzer (prelu)

## Metadata
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Operation**: prelu (UnaryOpType::PRELU_SFPU)
- **Status**: SUCCESS
- **Start time**: 2026-04-03T17:07:28+00:00
- **Model**: Claude Opus 4.6 (1M context)

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|------------|
| Operation name | prelu | HIGH |
| UnaryOpType | PRELU_SFPU | HIGH |
| Compute kernel | eltwise_sfpu.cpp (default) | HIGH |
| Output location | .claude-analysis/rrelu-1/prelu_analysis.md | HIGH |

## Execution Timeline

### Phase 1: Dispatch Tracing
1. Read `unary_op_utils.cpp` to find `get_macro_definition(PRELU_SFPU)` -> `"SFPU_OP_PRELU_INCLUDE"`
2. Read `get_op_init_and_func()` case for `PRELU_SFPU` -> `prelu_tile_init()` / `prelu_tile(idst, param0)`
3. Confirmed `get_compute_kernel_path()` falls through to default -> `eltwise_sfpu.cpp`
4. Confirmed `get_op_approx_mode()` falls through to default -> `false`
5. Traced `APPROX` constant generation in `genfiles.cpp:394` -> `constexpr bool APPROX = false`

### Phase 2: Kernel Source Reading
1. Read API header `prelu.h` -- found `SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_prelu, RC, APPROX, idst, param0)`
2. Read macro expansion in `llk_math_eltwise_unary_sfpu_macros.h` -- `_llk_math_eltwise_unary_sfpu_params_<APPROX>(calculate_prelu<APPROX>, idst, RC, param0)`
3. Read LLK dispatch in `llk_math_eltwise_unary_sfpu_params.h` (both WH and BH)
4. Read core SFPU implementation `ckernel_sfpu_prelu.h` (both WH and BH) -- identified SFPI-based kernel
5. Read `Converter::as_float` helper in `ckernel_sfpu_converter.h`
6. Read address mode configuration in `llk_math_eltwise_unary_sfpu.h` (both WH and BH)

### Phase 3: Instruction Analysis
1. Identified kernel style as SFPI-based (Style A) -- uses vFloat, dst_reg, v_if/v_endif
2. Traced v_if/v_endif mechanism in `runtime/sfpi/include/sfpi.h` -- SFPPUSHC + SFPSETCC(CC_LT) + SFPPOPC
3. Identified full instruction sequence: SFPLOADI x2, SFPLOAD, SFPPUSHC, SFPSETCC, SFPMAD, SFPPOPC, SFPSTORE
4. Confirmed ADDR_MOD_7 with dest.incr=0 on both WH and BH

### Phase 4: Analysis Writing
1. Wrote complete analysis to `.claude-analysis/rrelu-1/prelu_analysis.md`
2. All sections filled: dispatch summary, approx mode, abstraction layers, call chain, params dispatch, annotated source, instructions, register usage, addr mode

## Verification Summary
| Check | Result |
|-------|--------|
| `calculate_prelu` function exists (WH) | PASS |
| `calculate_prelu` function exists (BH) | PASS |
| All cited file paths exist | PASS |
| SFPU instructions verified in kernel source | PASS |

## External Service Results
| Service | Status | Fallback |
|---------|--------|----------|
| DeepWiki | UNAVAILABLE (repo not indexed) | Direct source code analysis |
| Confluence | Not needed | N/A |
| Glean | Not needed | N/A |

## Artifacts
- `.claude-analysis/rrelu-1/prelu_analysis.md` -- SFPU kernel analysis for prelu operation

## Handoff Notes
The prelu kernel is one of the simplest SFPU kernels -- it uses clean SFPI abstractions with a single v_if/v_endif block. The APPROXIMATION_MODE template parameter is accepted but unused (no conditional branches on it). The only difference between WH and BH is the pragma unroll directive.

---

# Execution Log: ttnn-unary-sfpu-operation-analyzer (leaky_relu)

## Metadata
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Operation**: leaky_relu (UnaryOpType::LEAKY_RELU)
- **Status**: SUCCESS
- **Model**: Claude Opus 4.6 (1M context)

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|------------|
| Operation name | leaky_relu | HIGH |
| UnaryOpType | LEAKY_RELU | HIGH |
| Compute kernel | eltwise_sfpu.cpp (default) | HIGH |
| Output location | .claude-analysis/rrelu-1/leaky_relu_analysis.md | HIGH |

## Execution Timeline

### Phase 1: Dispatch Tracing
1. Read `unary_op_utils.cpp` -- LEAKY_RELU falls to `default` in `get_compute_kernel_path()` -> `eltwise_sfpu.cpp`
2. Found `get_op_init_and_func()` case: `leaky_relu_tile_init()` / `leaky_relu_tile(idst, slope_u32)`
3. Confirmed `get_op_approx_mode()` -- `default: return false`
4. Found `get_macro_definition()` returns `SFPU_OP_RELU_FAMILY_INCLUDE`
5. Identified `SfpuType::lrelu` as the init template parameter

### Phase 2: Abstraction Layer Tracing
1. Read API header `relu.h` -- `leaky_relu_tile()` uses `SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_lrelu, RC, APPROX, idst, slope)`
2. Traced macro expansion to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_lrelu<APPROX>, ...)`
3. Read params dispatch for both WH and BH
4. Read metal wrappers `ckernel_sfpu_relu.h` (both WH and BH)
5. Read init function -- ADDR_MOD_7 configured with dest.incr=0

### Phase 3: Core SFPU Kernel Analysis
1. Read `ckernel_sfpu_relu.h` from both WH and BH tt_llk directories
2. Identified kernel style as B_raw_TTI
3. Analyzed CC mechanism: SFPSETCC(InstrMod=0, LT0 test) -> CC-guarded SFPMUL -> SFPENCC(reset)
4. Analyzed ADDR_MOD: WH ADDR_MOD_3 remapped to phys 7 via base, BH ADDR_MOD_7 direct

### Phase 4: External Research
1. DeepWiki -- unavailable (repository not indexed)
2. Confluence SFPU ISA page (1170505767) -- successfully retrieved
3. Verified SFPSETCC, SFPMUL, SFPENCC semantics from ISA spec

### Phase 5: Verification
1. Verified `_calculate_lrelu_` exists in WH and BH tt_llk files
2. Verified `calculate_lrelu` exists in metal wrapper files
3. Verified all 6 SFPU instructions present in kernel file
4. Verified all 8 cited file paths exist

## Verification Summary
| Check | Result |
|-------|--------|
| `_calculate_lrelu_` function exists (WH tt_llk) | PASS |
| `_calculate_lrelu_` function exists (BH tt_llk) | PASS |
| `calculate_lrelu` function exists (WH metal) | PASS |
| `calculate_lrelu` function exists (BH metal) | PASS |
| TT_SFPLOADI in kernel | PASS (2 occurrences) |
| TTI_SFPLOAD in kernel | PASS (2 occurrences) |
| TTI_SFPSETCC in kernel | PASS (1 occurrence) |
| TTI_SFPMUL in kernel | PASS (1 occurrence) |
| TTI_SFPENCC in kernel | PASS (1 occurrence) |
| TTI_SFPSTORE in kernel | PASS (2 occurrences) |
| All cited file paths exist | PASS (8/8) |

## External Service Results
| Service | Status | Fallback |
|---------|--------|----------|
| DeepWiki | UNAVAILABLE (repo not indexed) | Source code + Confluence ISA |
| Confluence | SUCCESS (SFPSETCC, SFPMUL, SFPENCC verified) | N/A |
| Glean | Not needed | N/A |

## Artifacts
- `.claude-analysis/rrelu-1/leaky_relu_analysis.md` -- SFPU kernel analysis for leaky_relu

## Key Findings
- Leaky relu is one of the simplest SFPU operations: 6 unique instructions, no polynomial approximation
- Uses raw TTI instructions (not SFPI abstractions) for direct CC manipulation
- CC pattern: one SFPSETCC per iteration (negative test) -> CC-guarded SFPMUL -> SFPENCC reset
- Slope loaded as 32-bit float via two SFPLOADI (16-bit each) before the loop
- WH and BH implementations are functionally identical; only ADDR_MOD index differs
