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
