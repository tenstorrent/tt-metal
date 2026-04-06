# ttnn-unary-sfpu-operation-analyzer Execution Log
**Operation**: softplus
**Date**: 2026-04-06
**Agent**: ttnn-unary-sfpu-operation-analyzer

## Summary
Analyzed SFPU kernel implementation for the `softplus` unary operation. Output saved to `.claude-analysis/swish-1/softplus_analysis.md`.

## Steps Taken
1. Initialized breadcrumbs
2. Searched for softplus references in codebase — found SFPU kernel files only in worktree (new operation under development)
3. Read `unary_op_utils.cpp` — confirmed compute kernel (`eltwise_sfpu.cpp`), approx mode (`false`), and SFPU chain expansion
4. Read API header `softplus.h` — traced `SFPU_UNARY_THREE_PARAM_KERNEL_FN` macro expansion
5. Read core SFPU kernel `ckernel_sfpu_softplus.h` (Wormhole + Blackhole — identical) — documented hybrid algorithm
6. Read `_sfpu_exp_21f_bf16_` from `ckernel_sfpu_exp.h` — documented exp21f algorithm
7. Read `PolynomialEvaluator` from `ckernel_sfpu_polyval.h` — documented Horner's method dispatch
8. Read `llk_math_eltwise_unary_sfpu_params.h` — documented per-face iteration and TTI_SETRWC between faces
9. Read `llk_math_eltwise_unary_sfpu.h` — confirmed ADDR_MOD_7 (dest.incr=0) for softplus
10. Verified `jit_build/genfiles.cpp` — confirmed `APPROX = false` code generation
11. Wrote `softplus_analysis.md`

## Key Findings
- Algorithm: 3-region hybrid: `_sfpu_exp_21f_bf16_` (x < -5), 8-term Remez polynomial (x ∈ [-5, 4)), pass-through (x ≥ 4)
- `APPROXIMATION_MODE` template param is present but unused — no behavioral difference
- `is_fp32_dest_acc_en` always `false` — BF16 rounding applied in exp21f path
- VectorMode: RC (all 4 faces processed)
- ITERATIONS: 8 per face (standard)
- Both Wormhole and Blackhole share identical ckernel_sfpu_softplus.h

## Files Produced
- `.claude-analysis/swish-1/softplus_analysis.md`
- `.claude-analysis/swish-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_execution_log.md`
- `.claude-analysis/swish-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_breadcrumbs.jsonl`
