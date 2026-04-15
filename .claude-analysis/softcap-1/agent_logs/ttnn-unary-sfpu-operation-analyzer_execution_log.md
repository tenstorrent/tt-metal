# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Session Summary
- **Operation**: rsub (reverse subtraction)
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output File**: `.claude-analysis/softcap-1/rsub_analysis.md`

## Key Findings

### Non-Standard Dispatch Path
RSUB exists as `UnaryOpType::RSUB` in the unary enum but is **NOT implemented** in the unary SFPU dispatch chain. The `get_op_init_and_func_parameterized` function in `unary_op_utils.cpp` would TT_FATAL when called with RSUB, because `is_parametrized_type(RSUB)` returns false.

The actual functional SFPU implementation is via `BinaryOpType::RSUB` in the **binary_ng** pipeline, which maps to `SfpuBinaryOp::RSUB` and produces `rsub_binary_tile` API calls.

### SFPU Kernel Characteristics
- **Style**: SFPI-based (uses `sfpi::vFloat`, `sfpi::dst_reg`)
- **Core operation**: `result = in1 - in0` (simple reverse subtraction)
- **Instructions**: SFPLOAD (x2), SFPMAD (x1 for subtraction), SFPSTORE (x1) per iteration
- **Iterations**: 8 per face, 4 faces per tile = 32 total iterations
- **Approximation mode**: Unused (the RSUB branch has no approximation-dependent logic)
- **Init function**: Empty (`_sfpu_binary_init_` is a no-op)
- **Hardware variants**: Wormhole B0 and Blackhole implementations are identical

### Naming Inconsistency
The LLK dispatch layer (`llk_math_eltwise_binary_sfpu_binop.h`) references `ckernel::sfpu::calculate_sfpu_binary` and `ckernel::sfpu::sfpu_binary_init` (without underscore prefix/suffix), but the actual function definitions in `ckernel_sfpu_binary.h` use `_calculate_sfpu_binary_` and `_sfpu_binary_init_` (with underscores). This naming mismatch would cause compilation failures in the metal build path.

## Timeline
1. Searched for RSUB in unary_op_utils -- found in enum but not in dispatch
2. Traced to binary_ng pipeline via BinaryOpType::RSUB -> SfpuBinaryOp::RSUB
3. Read API header (eltwise_binary_sfpu.h) -> rsub_binary_tile
4. Read LLK dispatch (llk_math_eltwise_binary_sfpu_binop.h)
5. Read parameters dispatch (llk_math_eltwise_binary_sfpu_params.h)
6. Read core SFPU implementation (ckernel_sfpu_binary.h) for WH and BH
7. Read init/addrmod configuration (llk_math_eltwise_binary_sfpu.h)
8. Verified all SFPU identifiers by grep
9. Wrote analysis file

---

## Session Summary (hardtanh)
- **Operation**: hardtanh
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output File**: `.claude-analysis/softcap-1/hardtanh_analysis.md`

## Key Findings (hardtanh)

### Integration Gap
HARDTANH has a complete core SFPU kernel (`_calculate_hardtanh_`) in both WH and BH `ckernel_sfpu_hardtanh.h`, but the dispatch chain is broken:
- No case in `get_op_init_and_func_parameterized()` (would TT_THROW)
- No compute API header (`hardtanh_tile()` does not exist)
- No LLK dispatch function (`llk_math_eltwise_unary_sfpu_hardtanh.h` does not exist)

### SFPU Kernel Characteristics
- **Style**: SFPI-based (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`)
- **Core algorithm**: Additive-shift-and-clamp (not direct min/max comparison)
- **Instructions per iteration**: SFPLOAD (1), SFPMAD (3), SFPSETCC (2), SFPPUSHC (2), SFPPOPC (2), SFPLOADI (2 CC-guarded), SFPSTORE (1)
- **Parameters**: 3 FP16_B-encoded values (shifted/negated thresholds)
- **Approximation mode**: Template parameter accepted but never branched on
- **Hardware variants**: WH and BH implementations are byte-identical

### Parameter Comment Discrepancy
Source code comments state `param2 = -(pos_threshold)`, but mathematical analysis proves the algorithm only produces correct results when `param2 = +pos_threshold = +max_val`. Since the host-side parameter encoding has never been implemented (dispatch is not wired), this discrepancy has not been exposed at runtime.

## Timeline (hardtanh)
1. Read `unary_op_utils.cpp` -- found HARDTANH in `is_parametrized_type()` but no dispatch case
2. Searched broadly for hardtanh across codebase -- found `ckernel_sfpu_hardtanh.h` in WH and BH
3. Verified no API header, no LLK dispatch, no compute API exists
4. Read core SFPU kernel source (both architectures, identical)
5. Analyzed SFPI-to-SFPU instruction mapping via `sfpi.h` compiler abstractions
6. Performed mathematical derivation and correctness proof for the shift-and-clamp algorithm
7. Discovered param2 comment discrepancy through mathematical verification
8. Verified all SFPU identifiers via grep
9. Wrote analysis file with full annotated source and mathematical proof

---

## Session Summary (softshrink)
- **Operation**: softshrink
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS (with caveat: SFPU kernel is nuked -- analysis documents the missing state)
- **Commit**: f55803b727
- **Output File**: `.claude-analysis/softcap-1/softshrink_analysis.md`

## Key Findings (softshrink)

### Complete Nuke
SOFTSHRINK's SFPU kernel was completely deleted in DEEP_NUKE_MANIFEST Phase 1 (commit `efdc0ad853`). Every layer is missing:
- Compute API header (`softshrink.h`) -- deleted
- LLK dispatch (`llk_math_eltwise_unary_sfpu_softshrink.h`) -- deleted
- Core SFPU kernel (`ckernel_sfpu_softshrink.h`) -- deleted
- Host dispatch (`get_op_init_and_func_parameterized`) -- no SOFTSHRINK case
- SfpuType enum -- no softshrink entry

### Surviving Infrastructure
- `UnaryOpType::SOFTSHRINK` enum value (`unary_op_types.hpp:113`)
- `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(softshrink, SOFTSHRINK)` macro
- `is_parametrized_type()` returning true for SOFTSHRINK
- Python nanobind binding
- Tests (would fail at runtime with TT_THROW)

### Family Classification
Softshrink is classified as "Piecewise Linear" family in the DEEP_NUKE_MANIFEST, same family as hardtanh, hardsigmoid, hardswish. The expected implementation would use SFPI conditionals (`v_if`/`v_endif`) with a single float parameter (lambda).

### No Special Program Factory Handling
Unlike HARDSHRINK (which allocates CB c_1 and packs runtime args), SOFTSHRINK has no special handling in `unary_program_factory.cpp`.

## Timeline (softshrink)
1. Read `unary_op_utils.cpp` -- found no SOFTSHRINK in `get_op_init_and_func_parameterized`
2. Read `unary_op_utils.hpp` -- confirmed SOFTSHRINK is parametrized type
3. Searched `tt_metal/hw/ckernels/` and `tt_metal/third_party/tt_llk/` -- no kernel files
4. Read `DEEP_NUKE_MANIFEST.md` -- confirmed Phase 1 deletion
5. Read analogous hardtanh kernel for expected pattern reference
6. Read shared dispatch infrastructure (params, init, macros)
7. Wrote analysis documenting nuked state with expected patterns
8. Committed analysis and breadcrumbs

---

## Session Summary (power)
- **Operation**: power
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS (documented missing implementation)
- **Output File**: `.claude-analysis/softcap-1/power_analysis.md`

## Key Findings (power)

### Complete Absence of SFPU Kernel Chain
The `power` operation's entire device-side SFPU kernel chain is absent from this codebase:
- **LLK dispatch header**: `llk_math_eltwise_unary_sfpu_power.h` -- included by `llk_math_unary_sfpu_api.h` but DOES NOT EXIST on disk
- **Core SFPU kernel**: `ckernel_sfpu_power.h` -- DOES NOT EXIST
- **Split API header**: `power.h` -- DOES NOT EXIST in `eltwise_unary/` directory
- **Host dispatch**: POWER has no case in any `unary_op_utils.cpp` switch statement

### Binary Power Also Stubbed
The binary power function `_calculate_sfpu_binary_power_()` in `ckernel_sfpu_binary.h` returns `0.0f` unconditionally. Comment states: "POW implementations removed -- depend on exp/log/recip primitives. Generator must implement from SFPI instructions."

### Dispatch Failure Path
When `ttnn::power(tensor, scalar)` is called:
1. `UNARY_OP_SCALAR_VARIANT` creates `EltwiseUnaryWithParam{POWER, param}` with non-empty params
2. `get_op_init_and_func<float>()` sees non-empty params, calls `get_op_init_and_func_parameterized()`
3. `is_parametrized_type(POWER)` returns false
4. **TT_FATAL** at line 35-37

### Mathematical Identity
`x^p = exp(p * ln(x))` would require SFPLOAD, SFPSTORE, SFPMAD (polynomial eval for log/exp), SFPEXEXP, SFPEXMAN, SFPSETEXP, SFPDIVP2, and CC instructions for edge cases.

### Moreh Composite Reference
`moreh_common.hpp::power_tile_to_cb()` implements power as: `x^p = x^(int_part) * exp(log(x) * frac_part)`, but also depends on the MISSING `power_iterative_tile()`.

## Timeline (power)
1. Read `unary_op_utils.cpp` -- POWER has no case in any switch statement
2. Read `unary_op_utils.hpp` -- `is_parametrized_type(POWER)` returns false
3. Read `compute_kernel_api.h` -- `power_tile()` declared, calls missing LLK function
4. Searched all include paths for `llk_math_eltwise_unary_sfpu_power.h` -- not found in 8 directories
5. Searched tt_llk submodule (WH, BH, Quasar) -- no `ckernel_sfpu_power.h`
6. Found binary power stub returning 0.0f in `ckernel_sfpu_binary.h`
7. Read `sfpu_split_includes.h` -- no power include guard
8. Read `eltwise_sfpu.cpp` -- uses split API, not old monolithic API
9. Read `moreh_common.hpp` -- composite power implementation reference
10. Verified all 13 existing file paths and confirmed all 5 expected-missing files
11. Wrote analysis documenting the missing state with implementation requirements

---

## Session Summary (tanh)
- **Operation**: tanh
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS (with caveat: SFPU kernel files are missing from worktree)
- **Output File**: `.claude-analysis/softcap-1/tanh_analysis.md`

## Key Findings (tanh)

### Missing SFPU Implementation
The tanh SFPU kernel implementation files (`ckernel_sfpu_tanh.h`, `llk_math_eltwise_unary_sfpu_tanh.h`) were stripped from this worktree as part of the generator evaluation environment. The API-level header `compute_kernel_api.h` still defines `tanh_tile<fast_and_approx=false>()` and `tanh_tile_init<false>()`, referencing the undefined `llk_math_eltwise_unary_sfpu_tanh()`.

### Dispatch Chain Status
- **API Header**: EXISTS (`compute_kernel_api.h`, lines 154-180)
- **LLK Dispatch**: MISSING (no `llk_math_eltwise_unary_sfpu_tanh.h`)
- **Core SFPU Kernel**: MISSING (no `ckernel_sfpu_tanh.h`)
- **Host Dispatch**: BROKEN (no case for `UnaryOpType::TANH` in `get_op_init_and_func_default()`)
- **SfpuType Enum**: No `tanh` entry in metal `SfpuType` (only frac, swish, atanh, sinh)
- **Integration Test**: Still references `"tanh_tile_init(); tanh_tile(0);"` in `test_sfpu_compute.cpp`

### Approximation Mode
- `math_approx_mode` returns `false` for TANH (default case)
- `fast_and_approx` template parameter defaults to `false`
- LLK tests explicitly skip tanh with approximation mode: "Metal tanh does not support approximation mode"

### Hardware Architecture Differences
- **Quasar**: Has hardware-accelerated tanh via `SFPNONLINEAR` with `TANH_MODE=0x5` (1 ULP max error on FP16_B)
- **Wormhole/Blackhole**: No hardware tanh support. Must be computed in software using exponential building blocks

### Reconstructed Algorithm
Based on the sibling `sinh` implementation (which uses `exp_21f` for 2^z approximation), tanh would compute:
```
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
         = (2^(x*log2e) - 2^(-x*log2e)) / (2^(x*log2e) + 2^(-x*log2e))
```
Key differences from sinh:
1. Division instead of halving (requires reciprocal via SFPNONLINEAR or Newton-Raphson)
2. Small-x Taylor: `tanh(x) ~ x - x^3/3` (vs sinh's `x + x^3/6`)
3. Large-x saturation: clamp to +/-1.0 for `|x| > ~9`

## Timeline (tanh)
1. Read `unary_op_utils.cpp` -- found TANH in UnaryOpType enum but not in dispatch
2. Confirmed `get_macro_definition()` returns `"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"` (default)
3. Read `compute_kernel_api.h` -- found API signatures for `tanh_tile<>()` and `tanh_tile_init<>()`
4. Searched for `llk_math_eltwise_unary_sfpu_tanh` -- found only in API header (reference, not definition)
5. Confirmed no `ckernel_sfpu_tanh.h` exists anywhere in the codebase
6. Read sibling implementations: `ckernel_sfpu_sinh.h` (exp_21f helper), `ckernel_sfpu_atanh.h` (log polynomial)
7. Read LLK dispatch pattern from `llk_math_eltwise_unary_sfpu_sinh.h`
8. Read parameters dispatch infrastructure (`llk_math_eltwise_unary_sfpu_params.h`)
9. Read SFPU hardware model reference (SFPNONLINEAR tanh mode, instruction semantics)
10. Verified `Quasar TANH_MODE=0x5` for hardware comparison
11. Verified LLK test skip for tanh approximation mode
12. Verified all SFPU identifiers via grep
13. Wrote analysis with annotated sinh source as reference and reconstructed tanh structure
