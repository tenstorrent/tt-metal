# Execution Log: ttnn-unary-sfpu-operation-analyzer

---

## Run 1: rrelu

### Summary
- **Operation**: rrelu
- **Status**: FAILED — operation not found in current codebase
- **Output file**: `.claude-analysis/selu-1/rrelu_analysis.md`

### Timeline

1. **Start**: Searched for RRELU in `unary_op_types.hpp` — not found in `UnaryOpType` enum
2. **Broad search**: Searched entire codebase for `rrelu`/`RRELU` — only found in:
   - `docs/sfpu_operations/key_notes/rrelu_key_notes.md` (formula documentation, marked as "NEW - to be implemented")
   - `.claude-analysis/selu-1/reference_selection.md` (referenced as existing op for selu, but files no longer exist)
   - `.claude-analysis/selu-1/agent_logs/` (previous pipeline breadcrumbs)
3. **File verification**: Confirmed absence of all expected SFPU source files:
   - `ckernel_sfpu_rrelu.h` — not found
   - `llk_math_eltwise_unary_sfpu_rrelu.h` — not found
   - `api/compute/eltwise_unary/rrelu.h` — not found
4. **Git history check**: Found commit `24376c2fcb9` (implement rrelu) and `db3f683e0a5` (batch nuke). Per instructions, did not retrieve deleted code.
5. **Analysis written**: Documented absence with structural notes for future implementation based on related operations.

### Key Decision
The operation was previously implemented and then removed. Since the instructions prohibit retrieving deleted code from git history, the analysis documents the absence and provides structural context from related operations (PReLU, Leaky ReLU, CELU, SELU) that share the same conditional-multiply SFPU pattern.

---

## Run 2: prelu_sfpu

### Summary
- **Operation**: prelu_sfpu
- **Status**: PARTIAL — SFPU kernel nuked, enum retained, structurally identical reference available
- **Output file**: `.claude-analysis/selu-1/prelu_sfpu_analysis.md`

### Timeline

1. **Dispatch Lookup**: Searched `unary_op_types.hpp` for `PRELU_SFPU` — found at line 110. Searched `unary_op_utils.cpp` — no case for PRELU_SFPU in any switch statement (nuked).

2. **SFPU Kernel Search**: Searched for `ckernel_sfpu_prelu.h` across `tt_metal/hw/ckernels/` and `tt_metal/third_party/tt_llk/` — not found (nuked). Confirmed via git log that commit `db3f683e0a5` removed 109 SFPU operations.

3. **SfpuType Enum Check**: Metal build enum nuked to `{ unused = 0 }`. LLK test helpers enum has `SfpuType::prelu` at line 101.

4. **Structural Reference Discovery**: Found `_calculate_lrelu_` in `ckernel_sfpu_relu.h` — structurally identical to PReLU (both compute `x * slope` for `x < 0`). This kernel is in the shared LLK library (submodule) and was NOT nuked.

5. **Reference Kernel Analysis**: Analyzed `_calculate_lrelu_` in detail:
   - Raw TTI_ instruction style (not SFPI abstractions)
   - CC pattern: SFPSETCC → SFPMUL (guarded) → SFPENCC
   - Registers: LREG0 (working), LREG2 (slope param), LCONST_0 (zero addend)
   - Address mode: ADDR_MOD_3 for SFPLOAD/SFPSTORE

6. **Infrastructure Verification**: Confirmed shared infrastructure files still present (params dispatch, init, macros).

7. **Documentation Sources**: Read `prelu_sfpu_key_notes.md`, `unary_eltwise_sfpu_list.md` (macro group: `SFPU_OP_PRELU_INCLUDE`).

8. **Identifier Verification**: Grepped all cited function names and instructions — all verified present.

### Key Decision
Used `_calculate_lrelu_` (Leaky ReLU) as the definitive structural reference since PReLU and Leaky ReLU are computationally identical at the SFPU level. The nuked `ckernel_sfpu_prelu.h` would have had the same logic with a different function name.

---

## Run 3: celu

### Summary
- **Operation**: celu
- **Status**: COMPLETE — SFPU kernel exists at ckernel level but not wired to compute API/LLK dispatch
- **Output file**: `.claude-analysis/selu-1/celu_analysis.md`

### Timeline

1. **Dispatch Lookup**: Searched `unary_op_types.hpp` — found `UnaryOpType::CELU` at line 126. Searched `unary_op_utils.cpp` — NO case for CELU in `get_op_init_and_func_parameterized()` or `get_op_init_and_func_default()`. Would throw `TT_THROW("unexpected parameterized op type")`.

2. **Compute API Search**: Searched all `*.h` files for `celu_tile` — only found in documentation (`celu_tile.rst`), NOT as an actual function definition. No compute API function exists.

3. **SFPU Kernel Discovery**: Searched `tt_metal/third_party/tt_llk/` for `celu` and `ActivationType::Celu` — found full implementation in `ckernel_sfpu_activations.h` (both WH and BH, identical). Uses `ActivationImpl<APPROXIMATION_MODE, ActivationType::Celu>` template specialization.

4. **Helper Function Tracing**:
   - `_calculate_exponential_body_<false>()` in `ckernel_sfpu_exp.h:289-326` — non-approximate path: `_sfpu_exp_(setsgn(in, 0))` + conditional `_sfpu_reciprocal_<2>()`
   - `_sfpu_exp_()` in `ckernel_sfpu_exp.h:259-287` — Horner-form polynomial + repeated squaring
   - `_sfpu_reciprocal_<2>()` in `ckernel_sfpu_recip.h:23-76` — quadratic estimate + 2 Newton-Raphson iterations
   - `Converter::as_float()` in `ckernel_sfpu_converter.h:15-24` — union-based uint32→float reinterpretation

5. **LLK Dispatch Analysis**: Confirmed `_llk_math_eltwise_unary_sfpu_params_` in `llk_math_eltwise_unary_sfpu_params.h` would be the intended dispatch path. Documented VectorMode::RC face iteration with SETRWC.

6. **Addr Mod Configuration**: Documented `ADDR_MOD_7` (dest.incr=0) from `llk_math_eltwise_unary_sfpu.h`. CELU has no special SfpuType, so no `ADDR_MOD_6` is configured. Same on WH and BH.

7. **WH/BH Comparison**: `diff` confirmed `ckernel_sfpu_activations.h` is identical between WH and BH.

8. **Identifier Verification**: Grepped all cited function names — all verified present:
   - `_calculate_activation_` — found in `ckernel_sfpu_activations.h`
   - `ActivationType::Celu` — found in `ckernel_sfpu_activations.h` and `ckernel_defs.h`
   - `_calculate_exponential_body_` — found in `ckernel_sfpu_exp.h`
   - `_sfpu_exp_` — found in `ckernel_sfpu_exp.h`
   - `_sfpu_reciprocal_` — found in `ckernel_sfpu_recip.h`

### Key Decision
Although CELU is not wired to the compute API or host dispatch, the SFPU kernel implementation is complete and well-documented. The analysis focuses on the existing ckernel-level implementation and describes the intended dispatch path based on the `_llk_math_eltwise_unary_sfpu_params_` / `_calculate_activation_` pattern used by other activations.

---

## Run 4: expm1

### Summary
- **Operation**: expm1
- **Status**: COMPLETE — Full SFPU kernel analysis with potential init issue identified
- **Output file**: `.claude-analysis/selu-1/expm1_analysis.md`

### Timeline

1. **Dispatch Lookup**: Found `UnaryOpType::EXPM1` in `unary_op_types.hpp` at line 54. Confirmed `get_compute_kernel_path` returns `eltwise_sfpu.cpp` (default), `get_op_approx_mode` returns `false` (default). EXPM1 not in `get_op_init_and_func_default` — init/func strings generated by `tt_ops_code_gen`.

2. **API Layer**: Found `expm1_tile<bool approx = false>(uint32_t idst)` and `expm1_tile_init<bool approx = false>()` in `compute_kernel_api.h` (lines 435–446). Calls `llk_math_eltwise_unary_sfpu_expm1<approx, DST_ACCUM_MODE>(idst)`.

3. **LLK Dispatch**: Found `llk_math_eltwise_unary_sfpu_expm1.h` at `/localdev/adjordjevic/work/tt-metal/tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/` (not in local worktree — files generated by `tt_ops_code_gen`). Identical for WH and BH. Calls `llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(calculate_expm1<APPROXIMATE>, dst_index, VectorMode::RC)`.

4. **Core SFPU Kernel**: Read `ckernel_sfpu_expm1.h` — simple 8-iteration loop: `v = calculate_exponential_body_improved<false>(v); dst_reg[0] = v - 1.0f;`.

5. **Exp Helper Tracing**: `calculate_exponential_body_improved<false>` (from arch-specific `ckernel_sfpu_exp.h`): non-approx path calls `sfpu_exp(setsgn(val, 0))` → `_sfpu_exp_()` (Horner series), then `sfpu_reciprocal(out)` → `_sfpu_reciprocal_<3>()` for negative inputs.

6. **Init Issue Discovery**: `expm1_init<false>` sets `vConstFloatPrgm{0,1,2}` to `{1.442695, 2.0, 0.863281}`, but `_sfpu_reciprocal_` expects `{0.323, 1.455, 2.121}` (from `_init_sfpu_reciprocal_`). The `_sfpu_exp_` function uses built-in constants and does NOT read vConstFloatPrgm. Documented this mismatch as a potential accuracy issue for negative inputs.

7. **Architecture Comparison**: WH and BH versions of expm1 kernel are identical. Addr mode config identical (ADDR_MOD_7 with zero increments).

8. **Identifier Verification**: All cited function names and file paths verified via grep.

### Key Decision
Documented the potential initialization mismatch between `expm1_init<false>` and `_sfpu_reciprocal_` requirements. This may affect accuracy for negative inputs in non-approximate mode. The issue does not exist in the approximate mode path which only uses bit manipulation (not reciprocal).
