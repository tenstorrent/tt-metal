# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Session Info
- **Operation**: softshrink
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Date**: 2026-04-15
- **Output file**: `.claude-analysis/softcap-2/softshrink_analysis.md`

## Execution Timeline

### 1. Initialization
- Read reference files: `sfpu-hardware-model.md`, `diagram-templates.md`, `sfpu-operation-analyzer.md` logging spec
- Initialized breadcrumbs at `.claude-analysis/softcap-2/agent_logs/`

### 2. Dispatch Tracing
- Read `unary_op_utils.hpp`: Found `SOFTSHRINK` in `is_parametrized_type()` returning `true`
- Read `unary_op_utils.cpp`: Found **NO case** for `SOFTSHRINK` in `get_op_init_and_func_parameterized()` or `get_op_init_and_func_default()`. Both fall to `TT_THROW`.
- Read `unary_op_types.hpp`: Confirmed `SOFTSHRINK` exists in enum at line 113
- Read `unary.hpp`: Confirmed `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(softshrink, SOFTSHRINK)` at line 165
- Read `unary_program_factory.cpp`: No special handling for SOFTSHRINK (unlike HARDSHRINK)
- `get_compute_kernel_path()` returns default `eltwise_sfpu.cpp` but is unreachable
- `get_op_approx_mode()` returns `false` (default)

### 3. Kernel Source Search
- Searched entire codebase for `softshrink_tile`, `_calculate_softshrink`, `ckernel_sfpu_softshrink` -- **zero matches**
- Searched `tt_metal/third_party/tt_llk/` for `softshrink` -- **zero matches**
- Searched `tt_metal/hw/` for `softshrink` -- **zero matches**
- Searched all `.h` and `.hpp` files for `softshrink` -- **zero matches** in kernel/LLK files

### 4. Conclusion
**SOFTSHRINK has no SFPU kernel implementation.** It is a type-system stub:
- Enum entry exists
- TTNN C++ API registration exists
- Nanobind Python binding exists
- BUT: No dispatch case, no tile-level API, no LLK function, no ckernel SFPU implementation

Attempting to call `ttnn.softshrink()` at runtime would throw `TT_THROW("unexpected parameterized op type SOFTSHRINK")`.

### 5. Context
Per `docs/sfpu_operations/wave2_instructions.md`, SOFTSHRINK was planned for Wave 2 SFPU generation but the kernel was never completed. The registration stubs were added but the compute infrastructure was not.

## Final Status
- **Status**: SUCCESS (analysis complete -- documented that no SFPU kernel exists)
- **Output**: `.claude-analysis/softcap-2/softshrink_analysis.md`
- **Breadcrumbs**: 6 events logged

---

## Session Info (xielu)
- **Operation**: xielu
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Date**: 2026-04-15
- **Output file**: `.claude-analysis/softcap-2/xielu_analysis.md`

## Execution Timeline (xielu)

### 1. Initialization
- Initialized breadcrumbs at `.claude-analysis/softcap-2/agent_logs/`
- Read reference files: `sfpu-hardware-model.md`, `sfpu-operation-analyzer.md` logging spec

### 2. Dispatch Tracing
- Read `unary_op_types.hpp`: Found `XIELU` at line 124
- Read `unary_op_utils.hpp`: `is_parametrized_type(XIELU)` returns `false` -- XIELU not listed (only HARDTANH, SOFTSHRINK)
- Read `unary_op_utils.cpp`: **NO case** for XIELU in `get_op_init_and_func_parameterized()` (hits `default: TT_THROW`) or `get_op_init_and_func_default()` (also `default: TT_THROW`)
- Since `ttnn::xielu()` passes `{alpha_p, alpha_n}` as parameters, `params` is non-empty -> routes to `get_op_init_and_func_parameterized()` -> CRASH
- `get_compute_kernel_path()` returns default `eltwise_sfpu.cpp` but is unreachable
- `get_op_approx_mode()` returns `false` (default)
- `get_macro_definition()` returns default `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` (XIELU has no explicit case)

### 3. Kernel Source Search
- Searched `tt_metal/` and `tt_metal/third_party/tt_llk/` for `xielu` -- **zero matches**
- Searched all `compute_kernel_api/` headers for `xielu_tile` -- **zero matches**
- No `ckernel_sfpu_xielu.h` file exists anywhere in the codebase

### 4. Nanobind Status
- `bind_xielu()` function exists in `unary_nanobind.cpp:966-1012`
- It is inside `#if 0` block (started at line 913) with comment "disabled binding functions - reference nuked operations"
- `bind_xielu()` is **never called** from `py_module()`
- Python binding for `ttnn.xielu` is fully disabled

### 5. Mathematical Formula (from docstring)
xIELU: piecewise function with beta=0.5, eps=-1e-6:
- x > 0: `alpha_p * x^2 + 0.5 * x`
- x <= 0: `alpha_n * expm1(min(x, eps)) - alpha_n * x + 0.5 * x`

### 6. Conclusion
**XIELU has no SFPU kernel implementation.** Similar to SOFTSHRINK, it is a post-nuke stub:
- Enum entry exists
- C++ API function exists (`ttnn::xielu()`)
- Nanobind binding definition exists but is disabled (`#if 0`)
- Test exists but would fail (Python binding not registered)
- BUT: No dispatch case, no tile-level API, no LLK function, no ckernel SFPU implementation

## Final Status (xielu)
- **Status**: FAILED (no SFPU kernel exists to analyze)
- **Output**: `.claude-analysis/softcap-2/xielu_analysis.md` (documents unimplemented status)
- **Breadcrumbs**: 6 events logged

---

## Session Info (tanhshrink)
- **Operation**: tanhshrink
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Date**: 2026-04-15
- **Output file**: `.claude-analysis/softcap-2/tanhshrink_analysis.md`

## Execution Timeline (tanhshrink)

### 1. Initialization
- Read reference files: `sfpu-hardware-model.md`, `diagram-templates.md`, `sfpu-operation-analyzer.md` logging spec, `common.md`
- Initialized breadcrumbs at `.claude-analysis/softcap-2/agent_logs/`

### 2. Dispatch Tracing
- Read `unary_op_types.hpp`: Confirmed `TANHSHRINK` enum exists at line 111
- Read `unary_op_utils.cpp`: `get_op_init_and_func_default()` has NO case for TANHSHRINK -- falls to `default: TT_THROW("unexpected op type")`
- Read `unary_op_utils.hpp`: `is_parametrized_type()` does not include TANHSHRINK
- Read `unary.hpp`: `REGISTER_UNARY_OPERATION(tanhshrink, TANHSHRINK)` creates C++ inline function at line 158
- Read `unary.cpp`: `unary_impl()` dispatches through `prim::unary()` standard path
- Read `unary_device_operation.cpp`: Standard `UnaryProgramFactory` selected for non-sharded inputs
- Read `unary_program_factory.cpp`: Calls `utils::get_block_defines()` which leads to TT_THROW
- `get_compute_kernel_path()` returns default `eltwise_sfpu.cpp` but is unreachable
- `get_op_approx_mode()` returns `false` (default)

### 3. Kernel Source Search
- Searched entire codebase for `ckernel_sfpu_tanhshrink` -- **zero matches**
- Searched for `tanhshrink_tile` or `tanh_shrink_tile` -- **zero matches**
- Searched `tt_metal/` for `tanhshrink` -- **zero matches**
- Searched nanobind files for `tanhshrink` -- only backward op `tanhshrink_bw` is bound

### 4. Composite/Alternative Path Check
- Read `unary_composite_op.cpp`: Only comment at line 501 (`// tanhshrink(x) = x - tanh(x)`) -- no active code
- Read `unary_composite.hpp`: No `tanhshrink` function declared
- Backward op uses host-level composition: `ttnn::square(ttnn::tanh(input)) * grad`

### 5. Conclusion
**TANHSHRINK has no SFPU kernel implementation.** It is a type-system stub:
- `UnaryOpType::TANHSHRINK` enum entry exists
- `REGISTER_UNARY_OPERATION` macro creates C++ inline function
- BUT: No dispatch case in `get_op_init_and_func_default()`, no tile-level API, no LLK function, no ckernel SFPU implementation, no nanobind forward binding
- Was previously a composite op (`x - tanh(x)`) that was removed during refactoring

Attempting to call `ttnn.tanhshrink()` at runtime would throw `TT_THROW("unexpected op type TANHSHRINK")`.

## Final Status (tanhshrink)
- **Status**: SUCCESS (analysis complete -- documented that no SFPU kernel exists)
- **Output**: `.claude-analysis/softcap-2/tanhshrink_analysis.md`
- **Breadcrumbs**: 6 events logged (start, dispatch_traced, kernel_source_read, instruction_analysis_complete, analysis_written, complete)

---

## Session Info (hardtanh)
- **Operation**: hardtanh
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Date**: 2026-04-15
- **Output file**: `.claude-analysis/softcap-2/hardtanh_analysis.md`

## Execution Timeline (hardtanh)

### 1. Initialization
- Initialized breadcrumbs at `.claude-analysis/softcap-2/agent_logs/`
- Read reference files: `sfpu-hardware-model.md`, `sfpu-operation-analyzer.md` logging spec

### 2. Dispatch Tracing
- Read `unary_op_utils.hpp`: Found `HARDTANH` in `is_parametrized_type()` returning `true`
- Read `unary_op_utils.cpp`: `get_op_init_and_func_parameterized()` has NO case for HARDTANH (falls to `default: TT_THROW`)
- `get_compute_kernel_path()` returns default `eltwise_sfpu.cpp`
- `get_op_approx_mode()` returns `false` (default)
- `get_macro_definition()` returns default `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE`
- Read `unary.hpp`: `hardtanh(input, min_val=-1.0f, max_val=1.0f)` creates `UnaryWithParam{HARDTANH, min_val, max_val}`

### 3. Kernel Source Discovery
- Searched for `_calculate_hardtanh_` across codebase: found in 2 files
  - `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
  - `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
- Both included via `ckernel_sfpu.h` -> `#include "sfpu/ckernel_sfpu_hardtanh.h"`
- Verified implementations are identical across WH and BH
- API header `hardtanh.h` does NOT exist in `tt_metal/hw/inc/api/compute/eltwise_unary/`
- LLK dispatch `llk_math_eltwise_unary_sfpu_hardtanh.h` does NOT exist in `tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/`

### 4. SFPU Kernel Analysis
- Kernel style: A_sfpi (pure SFPI abstractions: `vFloat`, `dst_reg`, `v_if`/`v_endif`)
- Template params: `APPROXIMATION_MODE` (unused), `ITERATIONS` (default 8)
- Runtime params: 3 `uint32_t` in FP16_B format -- pre-negated thresholds
  - `param0 = -(neg_threshold)`, `param1 = -(pos_threshold - neg_threshold)`, `param2 = -(pos_threshold)`
- Algorithm: additive threshold clamping -- adds pre-negated thresholds and zeroes values that exceed bounds, then adds final offset to restore correct output
- Two independent `v_if` blocks per iteration (no nesting, CC stack depth = 1)
- Instructions emitted (by SFPI compiler): SFPLOADI, SFPLOAD, SFPMAD, SFPSETCC, SFPENCC, SFPPUSHC, SFPPOPC, SFPSTORE

### 5. Parameters Dispatch Analysis
- Read `llk_math_eltwise_unary_sfpu_params.h` (WH and BH)
- Standard `VectorMode::RC` dispatch: 4 faces, function called once per face
- `ADDR_MOD_7` with all-zero increments (hardtanh not in any special-case branches)
- DEST advancement: `dst_reg++` within loop + `TTI_SETRWC(CR_D, 8)` x2 between faces

### 6. Comparison with Similar Operation
- Read `ckernel_sfpu_clamp.h`: clamp uses direct min/max comparison with `v_if`/`v_elseif`
- hardtanh uses additive approach -- different algorithm, same 3-param signature

### 7. Verification
- Confirmed `_calculate_hardtanh_` exists in both WH and BH via grep
- Verified all 9 cited file paths exist on disk
- Confirmed no raw TTI instructions in kernel (pure SFPI)

## Final Status (hardtanh)
- **Status**: SUCCESS (analysis complete -- SFPU kernel exists and is fully documented)
- **Output**: `.claude-analysis/softcap-2/hardtanh_analysis.md`
- **Note**: The core SFPU kernel is fully implemented but the dispatch chain (API header, LLK dispatch, host-side parameterized case) is not yet wired. This is a partially-implemented operation.
- **Breadcrumbs**: 8 events logged
