# Implementation Notes: sinh

## Math Definition
sinh(x) = (exp(x) - exp(-x)) / 2

## Implementation Strategy
Direct mirror of `cosh` implementation, changing only the arithmetic operator from addition to subtraction.

## Reference Operations Used
- **cosh** (PRIMARY): Nearly identical structure. The sinh kernel is a copy of cosh with `+` changed to `-`.
- **atanh**: Used for understanding the full registration stack pattern.

## Files Created/Modified

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h`
- `tests/ttnn/unit_tests/operations/eltwise/test_sinh.py`

### Modified Files
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`
- `ttnn/ttnn/operations/unary.py`

## Design Decisions
1. Used same exp helper (`_sfpu_exp_21f_bf16_`) as cosh - this provides good precision across the input range
2. Used same exponential init (`_init_exponential_`) as cosh
3. Used the `SFPU_OP_SINH_INCLUDE` split-include macro pattern, consistent with cosh's `SFPU_OP_COSH_INCLUDE`
4. Used same macro dispatch pattern (`SFPU_INIT_KERNEL_CALL`, `SFPU_THREE_PARAM_KERNEL_FP32_FIRST`)
5. Golden function uses `torch.sinh` directly
6. UnaryOpType::SINH and REGISTER_UNARY_OPERATION(sinh, SINH) already existed in the codebase

## Known Limitations
- For very large |x| values (beyond ~10), the exp computation may overflow in bfloat16, leading to inf results. This matches PyTorch behavior.
- Input range [-4, 4] is used for testing to stay within reasonable precision bounds.

## Test Results
(to be filled after testing)
