# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Session Info
- **Operation**: swish
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Start time**: 2026-04-15T10:43:30+00:00
- **Status**: SUCCESS

## Analysis Steps

1. **Read unary_op_utils.cpp** - Found SWISH dispatch: `swish_tile_init()` / `swish_tile(idst)`, compute kernel `eltwise_sfpu.cpp`, macro `SFPU_OP_SWISH_INCLUDE`, approx_mode=false.
2. **Read API header** (`swish.h`) - Confirmed `swish_tile()` calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)`.
3. **Read LLK dispatch** (`llk_math_eltwise_unary_sfpu_swish.h`) - Both WH and BH identical. Forwards to `_llk_math_eltwise_unary_sfpu_params_` with `calculate_swish<APPROXIMATE, 8>`.
4. **Read core SFPU kernel** (`ckernel_sfpu_swish.h`) - Both WH and BH identical. SFPI-based kernel. Piecewise sigmoid approximation: polynomial (|x|<=2.5), linear (2.5<|x|<=5), saturate (|x|>5). Then swish = x * sigmoid(x).
5. **Read params dispatch** (`llk_math_eltwise_unary_sfpu_params.h`) - VectorMode::RC, 4 faces, standard DEST progression.
6. **Read init/addrmod** (`llk_math_eltwise_unary_sfpu.h`) - SfpuType::swish has no special case; only ADDR_MOD_7 configured (zero increments).
7. **Verified APPROX generation** (`genfiles.cpp`) - `constexpr bool APPROX = {math_approx_mode}` from JIT.
8. **Verified all identifiers** - `calculate_swish`, `llk_math_eltwise_unary_sfpu_swish`, `llk_math_eltwise_unary_sfpu_swish_init` all found in both WH and BH ckernels. All file paths verified.
9. **Wrote analysis** to `.claude-analysis/softcap-1/swish_analysis.md`.

## Key Findings
- Swish is implemented entirely using SFPI abstractions (no raw TTI instructions)
- The `APPROXIMATION_MODE` template parameter is accepted but never tested -- same code runs regardless
- Sigmoid is approximated via a 3-segment piecewise approach: degree-3 polynomial, linear, and saturation
- WH and BH implementations are identical at all layers
- No special ADDR_MOD configuration needed (only ADDR_MOD_7 with zero increments)

## Output
- **Analysis file**: `.claude-analysis/softcap-1/swish_analysis.md`

---

## Session Info (atanh)
- **Operation**: atanh
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Start time**: 2026-04-15
- **Status**: SUCCESS

## Analysis Steps (atanh)

1. **Read unary_op_utils.cpp** - Found ATANH dispatch: `atanh_tile_init()` / `atanh_tile(idst)`, compute kernel `eltwise_sfpu.cpp`, macro `SFPU_OP_ATANH_INCLUDE`, approx_mode=false.
2. **Read API header** (`atanh.h`) - Confirmed `atanh_tile()` calls `llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)`.
3. **Read LLK dispatch** (`llk_math_eltwise_unary_sfpu_atanh.h`) - Both WH and BH identical. Forwards to `_llk_math_eltwise_unary_sfpu_params_` with `calculate_atanh<APPROXIMATE, 8>`.
4. **Read core SFPU kernel** (`ckernel_sfpu_atanh.h`) - Both WH and BH identical. SFPI-based kernel. Computes atanh(x) = 0.5*(ln(1+x)-ln(1-x)) via IEEE 754 decomposition and cubic minimax polynomial for ln(m).
5. **Read params dispatch** (`llk_math_eltwise_unary_sfpu_params.h`) - VectorMode::RC, 4 faces, standard DEST progression.
6. **Read init/addrmod** (`llk_math_eltwise_unary_sfpu.h`) - SfpuType::atanh has no special case; only ADDR_MOD_7 configured (zero increments).
7. **Verified APPROX generation** (`genfiles.cpp`) - `constexpr bool APPROX = {math_approx_mode}` from JIT.
8. **Verified all identifiers** - `calculate_atanh`, `atanh_init` found in both WH and BH ckernels. All SFPU instructions verified present in the kernel file. All file paths verified.
9. **Wrote analysis** to `.claude-analysis/softcap-1/atanh_analysis.md`.

## Key Findings (atanh)
- Atanh is implemented entirely using SFPI abstractions (no raw TTI instructions, no CC manipulation)
- The `APPROXIMATION_MODE` template parameter is accepted but never tested -- same code runs regardless
- Natural log is computed via IEEE 754 decomposition: ln(y) = e*ln(2) + P(m), where P is a cubic minimax polynomial
- The polynomial is evaluated twice per element (once for ln(1+x), once for ln(1-x))
- SFPU instructions: SFPMAD (dominant, all arithmetic), SFPEXEXP (exponent extraction x2), SFPSETEXP (mantissa normalization x2), SFPCAST (int-to-float x2), SFPLOAD/SFPSTORE (I/O)
- WH and BH implementations are identical at all layers
- No special ADDR_MOD configuration needed (only ADDR_MOD_7 with zero increments)

## Output (atanh)
- **Analysis file**: `.claude-analysis/softcap-1/atanh_analysis.md`
