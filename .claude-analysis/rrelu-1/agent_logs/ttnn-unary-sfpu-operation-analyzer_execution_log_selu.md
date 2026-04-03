# Execution Log: ttnn-unary-sfpu-operation-analyzer (selu)

## Metadata
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Operation**: selu (UnaryOpType::SELU)
- **Status**: SUCCESS
- **Model**: Claude Opus 4.6 (1M context)

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|------------|
| Operation name | selu | HIGH |
| UnaryOpType | SELU | HIGH |
| Compute kernel | eltwise_sfpu.cpp (default) | HIGH |
| Output location | .claude-analysis/rrelu-1/selu_analysis.md | HIGH |

## Execution Timeline

### Phase 1: Dispatch Tracing
1. Read `unary_op_utils.cpp` to find `get_macro_definition(SELU)` -> `"SFPU_OP_SELU_INCLUDE"`
2. Read `get_op_init_and_func()` case for `SELU` -> `selu_tile_init()` / `selu_tile(idst, param0, param1)`
3. Confirmed `get_compute_kernel_path()` falls through to default -> `eltwise_sfpu.cpp`
4. Confirmed `get_op_approx_mode()` falls through to default -> `false`
5. Confirmed SELU takes 2 runtime parameters: scale and alpha (bit-cast as uint32)

### Phase 2: Kernel Source Reading
1. Read API header `selu.h` -- `llk_math_eltwise_unary_sfpu_selu<APPROX, DST_ACCUM_MODE>(idst, param0, param1)`
2. Read LLK dispatch `llk_math_eltwise_unary_sfpu_selu.h` (both WH and BH) -- uses `_llk_math_eltwise_unary_sfpu_params_` with `calculate_selu` function pointer
3. Read core SFPU implementation `ckernel_sfpu_unary_selu.h` (both WH and BH) -- identified SFPI-based kernel, WH and BH identical
4. Read exp helper `_sfpu_exp_21f_bf16_` in `ckernel_sfpu_exp.h` (tt_llk WH)
5. Read `_float_to_int32_for_exp_21f_` helper
6. Read `PolynomialEvaluator::eval` in `ckernel_sfpu_polyval.h`
7. Verified SFPI instruction mappings in `runtime/sfpi/include/sfpi_lib.h`
8. Read address mode configuration in `llk_math_eltwise_unary_sfpu.h` (both WH and BH)

### Phase 3: Instruction Analysis
1. Identified kernel style as SFPI-based (Style A) -- uses vFloat, dst_reg, v_if/v_else/v_endif
2. Mapped all SFPI abstractions to SFPU instructions:
   - `dst_reg[0]` read/write -> SFPLOAD/SFPSTORE
   - `v * scale_value` -> SFPMAD
   - `exp_calc - vConst1` -> SFPMAD (with addend sign inversion)
   - `v_if(v >= 0.0f)` -> SFPENCC + SFPSETCC(GTE0) + SFPPUSHC
   - `v_else` -> SFPCOMPC
   - `v_endif` -> SFPPOPC + SFPENCC
   - `vec_min_max` -> SFPSWAP
   - `exexp_nodebias/exexp` -> SFPEXEXP
   - `exman8/exman9` -> SFPEXMAN
   - `shft` -> SFPSHFT
   - `setexp` -> SFPSETEXP
   - `int32_to_float` -> SFPCAST
   - `float_to_fp16b` -> SFP_STOCH_RND
   - `PolynomialEvaluator::eval` -> chain of SFPMADs
3. Confirmed ADDR_MOD_7 with dest.incr=0 on both WH and BH

### Phase 4: Analysis Writing
1. Wrote complete analysis to `.claude-analysis/rrelu-1/selu_analysis.md`
2. All sections filled: dispatch summary, approx mode, abstraction layers, call chain, params dispatch, annotated source, instructions, register usage, addr mode

## Verification Summary
| Check | Result |
|-------|--------|
| `calculate_selu` function exists (WH) | PASS |
| `calculate_selu` function exists (BH) | PASS |
| `llk_math_eltwise_unary_sfpu_selu` exists (WH) | PASS |
| `llk_math_eltwise_unary_sfpu_selu` exists (BH) | PASS |
| `llk_math_eltwise_unary_sfpu_selu_init` exists (WH) | PASS |
| `llk_math_eltwise_unary_sfpu_selu_init` exists (BH) | PASS |
| All cited file paths exist | PASS |
| SFPU instructions verified in kernel source | PASS |

## External Service Results
| Service | Status | Fallback |
|---------|--------|----------|
| DeepWiki | UNAVAILABLE (repo not indexed) | Direct source code analysis |
| Confluence | Not needed | N/A |
| Glean | Not needed | N/A |

## Artifacts
- `.claude-analysis/rrelu-1/selu_analysis.md` -- SFPU kernel analysis for selu operation

## Handoff Notes
The SELU kernel is moderately complex due to its dependency on the `_sfpu_exp_21f_bf16_` exponential helper. The kernel itself is clean SFPI (Style A) with a v_if/v_else/v_endif branch for positive vs. negative inputs. The positive branch is trivial (multiply by scale), while the negative branch invokes the full exp21f algorithm followed by `(exp(x) - 1) * alpha * scale`. A key implementation detail is that `_sfpu_exp_21f_bf16_` is called with `is_fp32_dest_acc_en=true` (hardcoded) to avoid premature BF16 rounding -- the rounding is deferred to the end of the SELU computation. WH and BH implementations are identical.
