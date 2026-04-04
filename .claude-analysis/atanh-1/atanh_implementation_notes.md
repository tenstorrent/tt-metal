# atanh Implementation Notes

## Math Definition
`atanh(x) = 0.5 * ln((1+x) / (1-x))`

Inverse hyperbolic tangent. Domain: |x| < 1. Returns NaN for |x| > 1, signed infinity for |x| == 1.

## Implementation Strategy
Used the canonical implementation from `tt_llk`'s `ckernel_sfpu_trigonometry.h` (found in build artifacts). The implementation:
1. Computes `1/(1-x)` using `_sfpu_reciprocal_`
2. Multiplies by `(1+x)` to get `(1+x)/(1-x)`
3. Applies `_calculate_log_body_no_init_` for the natural log
4. Multiplies by 0.5

### Key Design Decisions
- Uses `_sfpu_reciprocal_<2>` (2 Newton-Raphson iterations) for non-approximate mode to get fp32 precision
- Uses `_sfpu_reciprocal_<0>` for approximate mode
- Has `is_fp32_dest_acc_en` template parameter: when false and non-approximate, truncates reciprocal to fp16b precision via `float_to_fp16b()`
- Uses `SFPU_THREE_PARAM_KERNEL_FP32_FIRST` dispatch macro (like atan, cosh in tt_llk) since the kernel has 3 template params: `APPROXIMATION_MODE`, `is_fp32_dest_acc_en`, `ITERATIONS`
- Init function `atanh_init` calls `_init_sfpu_reciprocal_` which sets the polynomial coefficients for reciprocal approximation

### Reference Operations Most Useful
1. **cosh** (in worktree) - Provided the end-to-end file structure pattern (ckernel -> compute API -> split_includes -> utils)
2. **atanh in tt_llk build artifacts** - Provided the exact kernel algorithm and API pattern
3. **selu** (in worktree) - Provided the nanobind and golden function registration pattern

## New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`

## Modified Files
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
- `ttnn/ttnn/operations/unary.py`

## Known Limitations
- Domain restricted to |x| < 1 (mathematical definition)
- Precision limited by the log approximation (3rd order Chebyshev polynomial in `_calculate_log_body_no_init_`)
- bfloat16 precision may have higher ULP error near domain boundaries (|x| close to 1) due to reciprocal truncation
