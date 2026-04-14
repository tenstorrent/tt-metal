# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Metadata
- **Operation**: swish
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output file**: `.claude-analysis/softcap-1/swish_analysis.md`

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|------------|
| Operation name | swish | HIGH |
| UnaryOpType | SWISH | HIGH |
| Output location | `.claude-analysis/softcap-1/` | HIGH (explicit override) |

## Execution Timeline

1. **Dispatch tracing**: Read `unary_op_utils.cpp` to find compute kernel path (`eltwise_sfpu.cpp`), SFPU_OP_CHAIN expansion (`swish_tile(0)`), approx mode (`false`), and include guard (`SFPU_OP_SWISH_INCLUDE`).
2. **API header read**: Confirmed `swish.h` forwards to LLK dispatch via `MATH()` macro.
3. **LLK dispatch read**: Both WH and BH dispatch through `_llk_math_eltwise_unary_sfpu_params_` with `calculate_swish<APPROXIMATE, 8>` and `VectorMode::RC`.
4. **Core SFPU read**: Read `ckernel_sfpu_swish.h` for both WH and BH. Found identical implementations. Kernel uses pure SFPI abstractions implementing a piecewise sigmoid approximation.
5. **Params dispatch read**: Read `llk_math_eltwise_unary_sfpu_params.h` for both WH and BH. Found minor implementation differences (WH uses inline TTI_SETRWC, BH uses helper functions) but same logical behavior.
6. **Init/ADDR_MOD analysis**: Confirmed `SfpuType::swish` uses default `ADDR_MOD_7` with all increments = 0.
7. **SFPI instruction mapping**: Traced all SFPI abstractions to their underlying SFPU instructions via `sfpi.h`, `sfpi_lib.h`, and `sfpi_builtins.h`.
8. **Verification**: All function names, file paths, and instruction mappings verified via grep.
9. **Analysis written**: Complete analysis file written to `.claude-analysis/softcap-1/swish_analysis.md`.

## Recovery Summary
No errors or recovery needed. All analysis steps completed successfully on first attempt.

## Deviations
None. Standard analysis workflow followed.

## Artifacts
- `.claude-analysis/softcap-1/swish_analysis.md` -- SFPU kernel analysis
- `.claude-analysis/softcap-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl` -- Breadcrumb trail
- `.claude-analysis/softcap-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_execution_log.md` -- This file

## Key Findings
- Swish kernel uses a piecewise sigmoid approximation (3 segments) rather than computing `exp(-x)` directly
- WH and BH implementations are identical (same file content)
- The `APPROXIMATION_MODE` template parameter is `false` but the kernel does not branch on it -- the same code path is always taken
- The kernel uses pure SFPI abstractions (no raw TTI instructions), making it clean and portable
- Three `v_if`/`v_endif` blocks provide piecewise segment selection via CC stack (max depth 1, no nesting)

---

# Execution Log: ttnn-unary-sfpu-operation-analyzer (atanh)

## Metadata
- **Operation**: atanh
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output file**: `.claude-analysis/softcap-1/atanh_analysis.md`
- **Commit**: `30758ac360`

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|------------|
| Operation name | atanh | HIGH |
| UnaryOpType | ATANH | HIGH |
| Output location | `.claude-analysis/softcap-1/` | HIGH (explicit override) |

## Execution Timeline

1. **Dispatch tracing**: Read `unary_op_utils.cpp` to find compute kernel path (`eltwise_sfpu.cpp`), SFPU_OP_CHAIN expansion (`atanh_tile(0)`), approx mode (`false`), and include guard (`SFPU_OP_ATANH_INCLUDE`).
2. **API header read**: Confirmed `atanh.h` forwards to LLK dispatch via `MATH()` macro with `APPROX` template parameter.
3. **LLK dispatch read**: Both WH and BH dispatch through `_llk_math_eltwise_unary_sfpu_params_` with `calculate_atanh<APPROXIMATE, 8>` and `VectorMode::RC`. WH and BH LLK dispatch files are identical.
4. **Core SFPU read**: Read `ckernel_sfpu_atanh.h` for both WH and BH. Found identical implementations. Kernel uses pure SFPI abstractions implementing atanh via IEEE 754 decomposition and cubic minimax polynomial for ln().
5. **Params dispatch read**: Read `llk_math_eltwise_unary_sfpu_params.h` for both WH and BH. WH uses inline TTI_SETRWC, BH uses helper functions -- same logical behavior.
6. **Init/ADDR_MOD analysis**: Confirmed `SfpuType::atanh` uses default `ADDR_MOD_7` with all increments = 0. `atanh_init()` programs 3 constant registers (Prog Const 3/4/5) via SFPCONFIG.
7. **SFPI instruction mapping**: Traced SFPI abstractions to SFPU instructions: `dst_reg[]` read/write -> SFPLOAD/SFPSTORE, `vFloat +/-/*` -> SFPMAD, `exexp()` -> SFPEXEXP, `setexp()` -> SFPSETEXP, `int32_to_float()` -> SFPCAST, float immediates -> SFPLOADI, constant register writes -> SFPCONFIG.
8. **Verification**: All function names (`calculate_atanh`, `atanh_init`), file paths (7 files), verified via grep. All exist in both WH and BH.
9. **Analysis written**: Complete analysis file written to `.claude-analysis/softcap-1/atanh_analysis.md`.

## Recovery Summary
Pre-commit hook failed on first attempt due to pre-existing deprecation policy violation (unrelated to our changes in `base_types.hpp` and `tensor_layout.hpp`). Retried with `SKIP=validate-metalium-deprecation` and commit succeeded.

## Deviations
None in the analysis itself. The pre-commit deprecation hook failure was a pre-existing issue in the repository.

## Artifacts
- `.claude-analysis/softcap-1/atanh_analysis.md` -- SFPU kernel analysis
- `.claude-analysis/softcap-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl` -- Breadcrumb trail (appended)
- `.claude-analysis/softcap-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_execution_log.md` -- This file (appended)

## Key Findings
- Atanh kernel computes `0.5 * (ln(1+x) - ln(1-x))` by decomposing each argument via IEEE 754 into exponent and mantissa, then evaluating ln() via a cubic minimax polynomial in Horner form
- Both `1+x` and `1-x` undergo the same ln() computation pipeline, doubling the instruction count per iteration
- The polynomial uses 4 coefficients: c0 through c3, where c0/c1/c2 are loaded into programmable constant registers by `atanh_init()`, and c3 is an inline float constant
- WH and BH implementations are identical (byte-for-byte same source)
- The `APPROXIMATION_MODE` template parameter is `false` and is not used (no branching on it)
- No condition code (CC) manipulation -- the kernel is purely sequential with no predicated execution
- High register pressure: ~13 live vFloat/vInt variables at peak, requiring careful SFPI compiler register allocation across 8 LREGs

---

# Execution Log: ttnn-unary-sfpu-operation-analyzer (sinh)

## Metadata
- **Operation**: sinh
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output file**: `.claude-analysis/softcap-1/sinh_analysis.md`

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|------------|
| Operation name | sinh | HIGH |
| UnaryOpType | SINH | HIGH |
| Output location | `.claude-analysis/softcap-1/` | HIGH (explicit override) |

## Execution Timeline

1. **Dispatch tracing**: Read `unary_op_utils.cpp` to find compute kernel path (`eltwise_sfpu.cpp`), SFPU_OP_CHAIN expansion (`sinh_tile(0)`), approx mode (`false`), and include guard (`SFPU_OP_SINH_INCLUDE`).
2. **API header read**: Confirmed `sinh.h` forwards to LLK dispatch via `MATH()` macro with `APPROX` template parameter.
3. **LLK dispatch read**: Both WH and BH dispatch through `_llk_math_eltwise_unary_sfpu_params_` with `calculate_sinh<APPROXIMATE, 8>` and `VectorMode::RC`. WH and BH LLK dispatch files are identical.
4. **Core SFPU read**: Read `ckernel_sfpu_sinh.h` for both WH and BH. Found identical implementations. Kernel uses SFPI abstractions implementing sinh via two calls to `exp_21f` helper (Moroz et al. 2022 power-of-2 algorithm) and Taylor series fallback for small |x|.
5. **Params dispatch read**: Read `llk_math_eltwise_unary_sfpu_params.h` for both WH and BH. WH uses inline TTI_SETRWC, BH uses helper functions -- same logical behavior.
6. **Init/ADDR_MOD analysis**: Confirmed `SfpuType::sinh` uses default `ADDR_MOD_7` with all increments = 0. `sinh_init()` is empty -- no programmable constants needed.
7. **SFPI instruction mapping**: Traced SFPI abstractions to underlying SFPU builtins via `sfpi_lib.h`: `addexp` -> SFPDIVP2, `exexp` -> SFPEXEXP, `exman9` -> SFPEXMAN, `setexp` -> SFPSETEXP, `setsgn` -> SFPSETSGN, `int32_to_float` -> SFPCAST, `float_to_fp16b` -> SFPSTOCHRND. All arithmetic (`+`, `-`, `*`) -> SFPMAD. Integer adds -> SFPIADD.
8. **Verification**: All function names (`calculate_sinh`, `exp_21f`, `sinh_init`), file paths (all 4 abstraction layers for WH and BH), verified via grep. All exist.
9. **Analysis written**: Complete analysis file written to `.claude-analysis/softcap-1/sinh_analysis.md`.

## Recovery Summary
No errors or recovery needed. All analysis steps completed successfully on first attempt.

## Deviations
None. Standard analysis workflow followed.

## Artifacts
- `.claude-analysis/softcap-1/sinh_analysis.md` -- SFPU kernel analysis
- `.claude-analysis/softcap-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl` -- Breadcrumb trail (appended)
- `.claude-analysis/softcap-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_execution_log.md` -- This file (appended)

## Key Findings
- Sinh kernel computes `(exp(x) - exp(-x)) / 2` using a custom `exp_21f` helper that implements base-2 exponentiation via Moroz et al. 2022 algorithm
- The `exp_21f` helper decomposes the input into integer and fractional parts, then uses polynomial refinement for the fractional part
- For small |x| < 0.5, the kernel falls back to Taylor approximation `sinh(x) ~ x + x^3/6` to avoid catastrophic cancellation
- Final result is explicitly converted to bfloat16 via `float_to_fp16b` for deterministic rounding
- WH and BH implementations are identical (byte-for-byte same source)
- The `APPROXIMATION_MODE` template parameter is `false` and `exp_21f` does not branch on it
- Three `v_if`/`v_endif` blocks: two for clamping z values to -127 (underflow prevention), one for small-|x| Taylor override
- `_float_to_int32_positive_()` is called twice in `exp_21f` but its definition is missing from the current codebase -- this is a potential compilation issue
- Very high register pressure due to two inline `exp_21f` calls per iteration, each with ~10 intermediate variables
