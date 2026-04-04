# Execution Log: ttnn-unary-sfpu-operation-analyzer — elu

## Summary
- **Operation**: elu
- **Status**: Complete
- **Output file**: `.claude-analysis/selu-1/elu_analysis.md`

## Key Findings
1. ELU SFPU kernel exists in tt_llk: `ckernel_sfpu_elu.h` (identical on WH and BH)
2. Core functions: `_calculate_elu_<APPROXIMATION_MODE, ITERATIONS>(slope)` and `_init_elu_<APPROXIMATION_MODE>()`
3. Kernel uses SFPI abstractions (Style A) — no raw TTI instructions
4. Formula: `alpha * (exp(x) - 1)` for x < 0, identity for x >= 0
5. Non-approximate path uses `_sfpu_exp_` (Horner + repeated squaring) + `_sfpu_reciprocal_<2>` (Newton-Raphson)
6. Init delegates to `_init_exponential_<false, false, 0x3F800000>()` which calls `_init_sfpu_reciprocal_<false>()`
7. **Dispatch stack NOT wired**: No `elu_tile()` API, no LLK dispatch header, no `unary_op_utils.cpp` case

## Timeline
1. Read `unary_op_utils.cpp` — confirmed ELU defaults: approx_mode=false, kernel=eltwise_sfpu.cpp, no init/func case
2. Located SFPU kernel in `tt_metal/third_party/tt_llk/.../sfpu/ckernel_sfpu_elu.h`
3. Confirmed WH/BH kernel identical via diff
4. Traced `_calculate_exponential_piecewise_` → `_sfpu_exp_` → `_sfpu_reciprocal_<2>` chain
5. Traced `_init_elu_` → `_init_exponential_` → `_init_sfpu_reciprocal_` chain
6. Read params dispatch (`llk_math_eltwise_unary_sfpu_params.h`) and address mode config
7. Verified all cited functions and file paths exist
8. Wrote analysis file
