# Reference Selection: atanh

## Operation
- **Name**: atanh
- **Math Definition**: atanh(x) = 0.5 * ln((1+x)/(1-x)) for |x| < 1

## Selected References (Top 5)

### 1. acosh / asinh (ckernel_sfpu_trigonometry.h in tt_llk)
- **Rationale**: Same inverse hyperbolic function family. Uses `_calculate_log_body_no_init_()` and `_calculate_sqrt_body_()`. Shows exact pattern for combining log with arithmetic in SFPU. Located in `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_trigonometry.h`.
- **Key patterns**: Log + arithmetic composition, boundary condition handling (NaN, zero), `_init_inverse_hyperbolic_()` init function.

### 2. softsign (ckernel_sfpu_softsign.h in metal layer)
- **Rationale**: Uses reciprocal for division pattern: x / (1 + |x|). Shows how to create new SFPU kernel with separate LLK dispatch file. Most recent pattern (2026 copyright). Located in `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h`.
- **Key patterns**: `_sfpu_reciprocal_<2>()` usage, LLK dispatch file structure, compute API file structure.

### 3. log (ckernel_sfpu_log.h in tt_llk)
- **Rationale**: Core log function that atanh will use. Shows `_calculate_log_body_no_init_()` helper and `_init_log_()`. Located in `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_log.h`.
- **Key patterns**: Log computation via Chebyshev approximation, exponent extraction/normalization.

### 4. cosh (ckernel_sfpu_cosh.h in metal layer)
- **Rationale**: Hyperbolic function with clean compute API pattern. Shows SFPU_THREE_PARAM_KERNEL_FP32_FIRST macro usage, init function pattern. Located in `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h`.
- **Key patterns**: Compute API structure, init function pattern, `is_fp32_dest_acc_en` template parameter.

### 5. selu (ckernel_sfpu_selu.h in metal layer)
- **Rationale**: Shows conditional SFPU logic (v_if/v_else/v_endif), constant loading via Converter::as_float(), exp usage. Good general reference for complex SFPU operations. Located in `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h`.
- **Key patterns**: Conditional execution, constant handling, complete registration pattern.

SELECTED_REFERENCES: acosh, softsign, log, cosh, selu
