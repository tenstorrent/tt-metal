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

---

# Execution Log: ttnn-unary-sfpu-operation-analyzer (dropout)

## Metadata
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Operation**: dropout
- **Status**: SUCCESS
- **Model**: Claude Opus 4.6 (1M context)

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|------------|
| Operation name | dropout | HIGH |
| Output directory | `.claude-analysis/rrelu-1/` | HIGH (explicit override) |
| Breadcrumb path | `.claude-analysis/rrelu-1/` | HIGH (explicit override) |

## Execution Timeline

### Phase 1: Dispatch Tracing
1. Searched `unary_op_types.hpp` -- DROPOUT found in enum at line 108
2. Searched `unary_op_utils.cpp` for DROPOUT handling -- NOT found
3. **Non-standard discovery**: dropout has its own experimental program factory at `ttnn/cpp/ttnn/operations/experimental/dropout/`
4. Traced: `dropout_kernel.cpp` -> `dropout_tile()` -> `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(calculate_dropout, RC, APPROX, ...)` -> `_llk_math_eltwise_unary_sfpu_params_` -> `_calculate_dropout_`
5. Resolved `math_approx_mode = false` (hardcoded in program factory line 240)
6. Traced `APPROX` JIT generation: `genfiles.cpp:394` -> `constexpr bool APPROX = false`

### Phase 2: Kernel Source Reading
1. Read `ckernel_sfpu_dropout.h` for both WH and BH -- **implementations are identical**
2. Identified kernel style: `B_raw_TTI` (raw TTI instructions with CC manipulation)
3. Identified PRNG access via `SFPMOV(mod1=8, VC=9)` reading from RS[9] (PRNG Counter)
4. Read init function: `_init_dropout_` -> `init_prng_seed(seed)` (writes seed to PRNG_SEED config register + 600 SFPNOP delay)
5. Read LLK wrapper `ckernel_sfpu_dropout.h` in `hw/ckernels/` -- thin wrapper calling `_calculate_dropout_<APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, probability, scale)`

### Phase 3: Instruction Semantics Research (Confluence SFPU ISA)
1. **SFPMOV (PRNG mode)**: InstrMod=8 reads from RS view; RS[9] = PRNG Counter; reading advances PRNG by one step
2. **RS View table**: Confirmed RS[9] maps to PRNG Counter
3. **PRNG specification**: 32-bit LFSR per lane, XNOR taps at 31/30/10/0, period >= 2^32-1
4. **SFPSETSGN**: InstrMod[0]=1 sets sign from Imm12[0]; used to clear sign bit for unsigned comparison
5. **SFPIADD**: InstrMod=10 = CC_GTE0 | ARG_2SCOMP = subtraction with inverted CC sense
6. **SFPENCC**: InstrMod=0 sets CC.Res=1, keeps CC.En unchanged
7. **CC State analysis**: Extensive investigation of CC.En behavior. ISA states SFPIADD "Sets CC Enable? N", but the kernel relies on SFPIADD CC updates for conditional execution. Documented discrepancy.

### Phase 4: Verification
1. Verified `_calculate_dropout_` exists in both WH and BH (2 results)
2. Verified `_init_dropout_` exists in both WH and BH (2 results)
3. Verified all 7 cited file paths exist (7/7)
4. Verified all SFPU instructions in kernel file: TT_SFPLOADI(4), TTI_SFPLOAD(1), TTI_SFPMUL(1), TTI_SFPMOV(2), TTI_SFPSETSGN(1), TTI_SFPIADD(1), TTI_SFPENCC(1), TTI_SFPSTORE(1)

## Verification Summary
| Check | Result |
|-------|--------|
| `_calculate_dropout_` function exists (WH) | PASS |
| `_calculate_dropout_` function exists (BH) | PASS |
| `_init_dropout_` function exists (WH) | PASS |
| `_init_dropout_` function exists (BH) | PASS |
| All cited file paths exist | PASS (7/7) |
| All SFPU instructions verified in kernel | PASS |

## External Service Results
| Service | Status | Notes |
|---------|--------|-------|
| DeepWiki | UNAVAILABLE | Repository not indexed |
| Confluence SFPU ISA | SUCCESS | Retrieved SFPMOV, SFPSETSGN, SFPIADD, SFPENCC, PRNG, RS View documentation |
| Glean | Not used | N/A |

## Artifacts
- `.claude-analysis/rrelu-1/dropout_analysis.md` -- SFPU kernel analysis for dropout

## Key Findings
- Dropout uses a **non-standard dispatch** (experimental program factory, not UnaryProgramFactory)
- The SFPU kernel uses PRNG hardware (RS[9]) for per-lane random number generation
- CC mechanism uses SFPIADD with inverted sense (CC_GTE0) to mark lanes for zeroing
- ISA documentation inconsistency: SFPIADD "Sets CC Enable? N" but the kernel relies on CC-guarded execution after SFPIADD -- suggests implicit CC enable not documented in ISA table
- WH and BH implementations are **identical** -- no architecture-specific differences
- APPROXIMATION_MODE template parameter is accepted but completely unused (no conditional branches)
