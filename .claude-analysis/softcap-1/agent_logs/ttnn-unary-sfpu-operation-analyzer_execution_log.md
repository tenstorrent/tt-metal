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
