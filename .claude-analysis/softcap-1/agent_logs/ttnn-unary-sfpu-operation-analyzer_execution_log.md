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

---

## Session Info (sinh)
- **Operation**: sinh
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Start time**: 2026-04-15
- **Status**: SUCCESS

## Analysis Steps (sinh)

1. **Read unary_op_utils.cpp** - Found SINH dispatch: `sinh_tile_init()` / `sinh_tile(idst)`, compute kernel `eltwise_sfpu.cpp`, macro `SFPU_OP_SINH_INCLUDE`, approx_mode=false.
2. **Read API header** (`sinh.h`) - Confirmed `sinh_tile()` calls `llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)`.
3. **Read LLK dispatch** (`llk_math_eltwise_unary_sfpu_sinh.h`) - Both WH and BH identical. Forwards to `_llk_math_eltwise_unary_sfpu_params_` with `calculate_sinh<APPROXIMATE, 8>`.
4. **Read core SFPU kernel** (`ckernel_sfpu_sinh.h`) - Both WH and BH identical. SFPI-based kernel. Computes sinh(x) via exp_21f helper (Moroz 2022 fast 2^z) with Taylor approximation fallback for |x| < 0.5.
5. **Read params dispatch** (`llk_math_eltwise_unary_sfpu_params.h`) - VectorMode::RC, 4 faces, standard DEST progression.
6. **Read init/addrmod** (`llk_math_eltwise_unary_sfpu.h`) - SfpuType::sinh has no special case; only ADDR_MOD_7 configured (zero increments).
7. **Verified APPROX generation** (`genfiles.cpp`) - `constexpr bool APPROX = {math_approx_mode}` from JIT.
8. **Verified all identifiers** - `calculate_sinh`, `sinh_init`, `exp_21f` found in both WH and BH ckernels. All SFPU instruction mappings verified against sfpi_lib.h. All file paths verified.
9. **Wrote analysis** to `.claude-analysis/softcap-1/sinh_analysis.md`.

## Key Findings (sinh)
- Sinh is implemented entirely using SFPI abstractions (no raw TTI instructions)
- The `APPROXIMATION_MODE` template parameter is accepted but never branched on -- same code runs regardless
- Uses exp_21f helper function implementing Moroz et al. 2022 fast 2^z algorithm via IEEE 754 bit manipulation
- Taylor approximation (sinh(x) ~ x + x^3/6) for |x| < 0.5 to avoid catastrophic cancellation in exp(x)-exp(-x)
- BF16 rounding at output via float_to_fp16b for deterministic results
- 12 distinct SFPU instructions used: SFPLOAD, SFPSTORE, SFPMAD (dominant), SFPLOADI, SFPDIVP2, SFPEXEXP, SFPEXMAN, SFPSETSGN, SFPSETEXP, SFPCAST, SFPSTOCHRND, SFPSETCC
- `_float_to_int32_positive_` is referenced but not defined in any reachable header -- potential compilation issue
- WH and BH implementations are identical at all layers
- No special ADDR_MOD configuration needed (only ADDR_MOD_7 with zero increments)

## Output (sinh)
- **Analysis file**: `.claude-analysis/softcap-1/sinh_analysis.md`

---

## Session Info (hardtanh)
- **Operation**: hardtanh
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Start time**: 2026-04-15
- **Status**: SUCCESS

## Analysis Steps (hardtanh)

1. **Read unary_op_utils.cpp** - HARDTANH has no case in `get_op_init_and_func_parameterized` (TT_THROW), `get_op_init_and_func_default` (TT_THROW), `get_compute_kernel_path` (default: eltwise_sfpu.cpp), `get_op_approx_mode` (default: false). Dispatch chain is NOT WIRED.
2. **Read unary_op_utils.hpp** - `is_parametrized_type(HARDTANH)` returns true, confirming it expects min_val/max_val parameters.
3. **Read unary.hpp** - `hardtanh(input, min_val=-1.0f, max_val=1.0f)` passes both params as `UnaryWithParam{UnaryOpType::HARDTANH, min_val, max_val}`.
4. **Checked API layer** - No `hardtanh_tile()` function exists in `tt_metal/hw/inc/api/compute/eltwise_unary/`.
5. **Checked LLK layer** - No `llk_math_eltwise_unary_sfpu_hardtanh.h` exists in either WH or BH llk_api directories.
6. **Checked metal SfpuType** - `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` has only frac/swish/atanh/sinh; no hardtanh entry.
7. **Read core SFPU kernel** (`ckernel_sfpu_hardtanh.h`) - Both WH and BH identical. SFPI-based. Three-step clamping: add(-min), clamp-below, add(-(max-min)), clamp-above, add(max). Takes 3 uint32 FP16_B params.
8. **Confirmed test SfpuType** - `llk_sfpu_types.h` has `hardtanh` at position 1 and `ActivationType::Hardtanh = 3` exists in `ckernel_defs.h`.
9. **Analyzed SFPU instructions** - SFPLOADI (3x param load + 2x zero), SFPLOAD (1x/iter), SFPMAD (3x/iter adds), SFPSETCC (2x/iter CC_LT and CC_GTE), SFPPUSHC/SFPPOPC (2x/iter each), SFPSTORE (1x/iter).
10. **Verified all identifiers** - `_calculate_hardtanh_` found in both WH and BH. All file paths verified.
11. **Wrote analysis** to `.claude-analysis/softcap-1/hardtanh_analysis.md`.

## Key Findings (hardtanh)
- SFPU kernel exists and is complete but the full dispatch chain (API header, LLK dispatch, get_op_init_and_func case, metal SfpuType) is NOT yet wired
- Kernel uses SFPI abstractions with a clever three-step clamping algorithm using additions and predicated zeroing
- APPROXIMATION_MODE template parameter is unused -- same code for both true/false
- Takes 3 pre-computed FP16_B parameters: -min_val, -(max_val-min_val), max_val
- Source comment for param2 says "-(pos_threshold)" but mathematical analysis shows it must be positive max_val for correct results
- WH and BH implementations are byte-for-byte identical
- No special ADDR_MOD configuration needed (default ADDR_MOD_7 with zero increments)

## Output (hardtanh)
- **Analysis file**: `.claude-analysis/softcap-1/hardtanh_analysis.md`
