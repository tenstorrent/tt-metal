# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Operation: softsign

### Summary
- **Status**: SUCCESS
- **Output**: `.claude-analysis/swish-2/softsign_analysis.md`
- **Key Finding**: The softsign SFPU kernel is STUBBED OUT on both Wormhole and Blackhole. All dispatch wiring (API header, LLK dispatch, ckernel header) is in place, but `calculate_softsign()` and `softsign_init()` have empty bodies. The comment states the implementation was "removed -- depends on recip primitive (Family 3)".

### Analysis Steps
1. Confirmed `SOFTSIGN` exists in `UnaryOpType` enum (line 124 of `unary_op_types.hpp`)
2. Found SOFTSIGN dispatch in `unary_ng_op_utils.cpp` (line 90): `softsign_tile_init()` / `softsign_tile(idst)`
3. Verified SOFTSIGN is NOT in the legacy `unary_op_utils.cpp` dispatch
4. Traced API header: `softsign.h` -> `llk_math_eltwise_unary_sfpu_softsign<APPROX>(idst)`
5. Traced LLK dispatch: uses `_llk_math_eltwise_unary_sfpu_params_` with `calculate_softsign<APPROXIMATE, ITERATIONS>` functor
6. Read core SFPU implementation: empty stub on both WH and BH
7. Read params dispatch function from tt_llk submodule (parent repo, since worktree submodule is empty)
8. Read init/addrmod configuration: ADDR_MOD_7 with all-zero increments
9. Verified all function names and file paths exist in codebase

### Challenges
- The `tt_llk` submodule is empty in the worktree. Had to read the params dispatch file from the parent repo at `/localdev/vignjatijevic/tt-metal/tt_metal/third_party/tt_llk/`.
- The `llk_math_eltwise_unary_sfpu_params.h` and `llk_math_eltwise_unary_sfpu.h` files are only available in the tt_llk submodule, not in the `hw/ckernels/` directory.

### Timing
- Start: Analysis session start
- End: Analysis session end
- Total: Single pass, no retries needed

---

## Operation: hardswish

### Summary
- **Status**: SUCCESS
- **Output**: `.claude-analysis/swish-2/hardswish_analysis.md`
- **Key Finding**: Pure SFPI kernel computing `hardswish(x) = x * clamp(x/6 + 0.5, 0, 1)`. APPROXIMATION_MODE is false but unused by the kernel. WH and BH implementations are identical.

### Analysis Steps
1. Read `unary_op_utils.cpp` — Found HARDSWISH dispatch: compute kernel `eltwise_sfpu.cpp`, define `SFPU_OP_HARDSWISH_INCLUDE`, init `hardswish_tile_init()`, func `hardswish_tile({idst})`, approx_mode=false
2. Read API header (`hardswish.h`) — Confirmed forwarding to `llk_math_eltwise_unary_sfpu_hardswish<APPROX>()`
3. Read `sfpu_split_includes.h` — Confirmed conditional include via `SFPU_OP_HARDSWISH_INCLUDE`
4. Read LLK dispatch (both WH and BH) — Identical files calling `_llk_math_eltwise_unary_sfpu_params_` with `calculate_hardswish<APPROXIMATE, 8>` and `VectorMode::RC`
5. Read core SFPU kernel (both WH and BH) — Identical SFPI-style kernels computing `x * clamp(x/6 + 0.5, 0, 1)`
6. Read params dispatch (from tt_llk submodule in main repo) — WH uses direct TTI_SETRWC, BH uses helper functions; both loop 4 faces
7. Read init/addrmod (from tt_llk submodule) — Only ADDR_MOD_7 (all zeros) configured for hardswish
8. Read genfiles.cpp — Confirmed `APPROX` is `constexpr bool` generated from `math_approx_mode`
9. Verified all identifiers — `calculate_hardswish` found in both WH/BH ckernel files; all file paths verified to exist
10. Wrote analysis file to `.claude-analysis/swish-2/hardswish_analysis.md`

### Challenges
- The `tt_llk` submodule is empty in the worktree. Had to read params dispatch and init files from the parent repo at `/localdev/vignjatijevic/tt-metal/tt_metal/third_party/tt_llk/`.

### Timing
- Start: Analysis session start
- End: Analysis session end
- Total: Single pass, no retries needed

---

## Operation: hardsigmoid

### Summary
- **Status**: SUCCESS
- **Output**: `.claude-analysis/swish-2/hardsigmoid_analysis.md`
- **Commit**: 85131da7a2
- **Key Finding**: The hardsigmoid SFPU kernel is a simple piecewise-linear function using SFPI abstractions. Computes `max(0, min(1, x/6 + 0.5))` using a single SFPMAD for the linear portion and two v_if blocks for clamping. APPROXIMATION_MODE template parameter is unused. WH and BH implementations are identical.

### Analysis Steps
1. Read `unary_op_utils.cpp` — Confirmed HARDSIGMOID dispatch: compute kernel `eltwise_sfpu.cpp`, init `hardsigmoid_tile_init()`, func `hardsigmoid_tile(idst)`, approx_mode=false (default)
2. Found include path: `get_macro_definition(HARDSIGMOID)` returns `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` (default), meaning hardsigmoid is included via `llk_math_unary_sfpu_api.h` directly, NOT through the `sfpu_split_includes.h` conditional guard
3. Read API header (`hardsigmoid.h`) — Confirmed forwarding to `llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>()`
4. Read LLK dispatch (both WH and BH) — Identical files calling `_llk_math_eltwise_unary_sfpu_params_` with `calculate_hardsigmoid<APPROXIMATE, 8>` and `VectorMode::RC`
5. Read core SFPU kernel (both WH and BH) — Identical SFPI-style kernels
6. Read params dispatch (from tt_llk submodule in main repo) — WH uses direct TTI_SETRWC, BH uses helper functions; both loop 4 faces
7. Read init/addrmod (from tt_llk submodule) — Only ADDR_MOD_7 (all zeros) configured
8. Verified all identifiers and file paths exist
9. Wrote analysis file, committed with hash 85131da7a2

### Challenges
- The `tt_llk` submodule is empty in the worktree. Had to read params dispatch and init files from the parent repo at `/localdev/vignjatijevic/tt-metal/tt_metal/third_party/tt_llk/`.

### Timing
- Start: Analysis session start
- End: Analysis session end
- Total: Single pass, no retries needed

---

## Operation: cbrt

### Summary
- **Status**: SUCCESS
- **Output**: `.claude-analysis/swish-2/cbrt_analysis.md`
- **Key Finding**: The cbrt SFPU kernel implements a fast cube root approximation based on Moroz et al.'s magic-constant method. It uses `int32_to_float` + `SFPMAD` + `SFPSHFT` for the initial guess and Horner polynomial refinement via programmable constants. The FP32 path adds an extra Halley iteration for precision; the FP16b path truncates with `float_to_fp16b`. APPROXIMATION_MODE template parameter is accepted but never referenced. WH and BH implementations are identical.

### Analysis Steps
1. Confirmed `CBRT` exists in `UnaryOpType` enum (line 129 of `unary_op_types.hpp`)
2. Found CBRT is NOT in the legacy `unary_op_utils.cpp` dispatch (neither `get_op_init_and_func_default` nor `get_op_init_and_func_parameterized`)
3. Found CBRT is also NOT in the `unary_ng_op_utils.cpp` dispatch
4. Verified CBRT is registered via `REGISTER_UNARY_OPERATION(cbrt, CBRT)` in `unary.hpp`
5. Traced API header: `cbrt.h` → `llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst)`
6. Traced LLK dispatch: both WH and BH are identical, using `_llk_math_eltwise_unary_sfpu_params_` with `calculate_cube_root<APPROXIMATE, fp32_dest_acc_en, ITERATIONS>`
7. Read core SFPU implementation: Moroz et al. magic-constant method with Horner polynomial refinement
8. Read params dispatch from tt_llk submodule (parent repo)
9. Read init/addrmod configuration: ADDR_MOD_7 with all-zero increments
10. Read SFPI library functions (`sfpi_lib.h`) to map intrinsics to hardware instructions
11. Verified all function names and file paths exist in codebase
12. Wrote analysis file to `.claude-analysis/swish-2/cbrt_analysis.md`

### Challenges
- The `tt_llk` submodule is empty in the worktree. Had to read params dispatch and init files from the parent repo.
- CBRT is not in any dispatch switch statement in `unary_op_utils.cpp` or `unary_ng_op_utils.cpp` — it must rely on the compute kernel API include mechanism (`SFPU_OP_COMPUTE_KERNEL_API_INCLUDE`).
- `SfpuType::cbrt` is referenced in the LLK dispatch but does not exist in the `SfpuType` enum in the worktree, indicating the enum needs to be extended for compilation.

### Timing
- Start: Analysis session start
- End: Analysis session end
- Total: Single pass, no retries needed

---

## Operation: rpow

### Summary
- **Status**: SUCCESS
- **Output**: `.claude-analysis/swish-2/rpow_analysis.md`
- **Key Finding**: The rpow SFPU kernel implements `base^x = 2^(x * log2(base))` using the exp_21f algorithm from Moroz et al. 2022. It precomputes `log2(base)` as scalar RISC-V math, then vectorizes the 2^z computation on SFPU. The operation is work-in-progress: `SfpuType::rpow` is missing from the enum, `_float_to_int32_positive_()` is undefined, and RPOW is not in `is_parametrized_type()`. APPROXIMATION_MODE template parameter is declared but unused (no code path divergence). WH and BH implementations are identical.

### Analysis Steps
1. Confirmed `RPOW` exists in `UnaryOpType` enum (line 128 of `unary_op_types.hpp`)
2. Found RPOW registered as `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(rpow, RPOW)` in `unary.hpp` (takes a base parameter)
3. Discovered RPOW is NOT in `is_parametrized_type()` or `get_op_init_and_func_parameterized()` — integration gap
4. Read API header `rpow.h` — defines `rpow_tile_init()` and `rpow_tile(idst, base_val)` taking base as IEEE 754 uint32_t
5. Read LLK dispatch (both WH and BH) — identical, uses `_llk_math_eltwise_unary_sfpu_params_` with `calculate_rpow<APPROXIMATE, ITERATIONS>` and `base_val` forwarded as extra arg
6. Read core SFPU kernel (both WH and BH) — identical SFPI-style kernel implementing exp_21f algorithm
7. Identified `_float_to_int32_positive_()` is called but never defined anywhere in the codebase — compilation gap
8. Read `SfpuType` enum — `rpow` not present (only unused, hardsigmoid, hardtanh, hardswish, softshrink)
9. Read params dispatch from tt_llk submodule (parent repo) — standard 4-face RC dispatch with forwarded base_val arg
10. Read SFPI library functions — mapped all intrinsics (addexp→SFPDIVP2, exexp→SFPEXEXP, exman9→SFPEXMAN, setexp→SFPSETEXP, setsgn→SFPSETSGN, int32_to_float→SFPCAST, float_to_fp16b/float_to_int16→SFP_STOCH_RND)
11. Verified all function names and file paths exist in codebase
12. Wrote analysis file to `.claude-analysis/swish-2/rpow_analysis.md`

### Challenges
- Multiple integration gaps: SfpuType enum missing rpow, _float_to_int32_positive_ undefined, is_parametrized_type doesn't include RPOW
- The `tt_llk` submodule is empty in the worktree. Had to read params dispatch and init files from the parent repo.

### Timing
- Start: Analysis session start
- End: Analysis session end
- Total: Single pass, no retries needed
