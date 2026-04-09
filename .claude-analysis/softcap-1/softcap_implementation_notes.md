# Softcap Implementation Notes

## Operation Definition
`softcap(x, cap) = cap * tanh(x / cap)`

Parameters: `cap` (positive float, default 50.0)

## Algorithm

The implementation uses range reduction (`u = x / cap`) followed by a piecewise approximation for `tanh(u)`:

1. **Segment 0** (`|u| <= 1.0`): 9th-degree Taylor series in Horner form
   - `tanh(u) = u * (1 + u^2*(-1/3 + u^2*(2/15 + u^2*(-17/315 + u^2*62/2835))))`
   - Max error ~0.006 at |u|=1.0

2. **Segment 1** (`1.0 < |u| <= 2.0`): Quadratic Lagrange interpolation
   - Fitted through `(1.0, tanh(1.0)), (1.5, tanh(1.5)), (2.0, tanh(2.0))`
   - Coefficients: `-0.16936*u^2 + 0.71052*u + 0.22043`
   - Max interpolation error ~0.005

3. **Segment 2** (`2.0 < |u| <= 3.0`): Quadratic Lagrange interpolation
   - Fitted through `(2.0, tanh(2.0)), (2.5, tanh(2.5)), (3.0, tanh(3.0))`
   - Coefficients: `-0.02828*u^2 + 0.17242*u + 0.73231`
   - Max interpolation error ~0.001

4. **Segment 3** (`|u| > 3.0`): Exact saturation to `+/-cap`
   - `tanh(3.0) = 0.9951`, so error from saturation < 0.005

The computation works on `|u|` (positive tanh values) and applies the sign of `x` at the end: `softcap(-x) = -softcap(x)`.

## Parameter Passing

- `cap` and `1/cap` are precomputed on the host and packed as float32 bit patterns into uint32 hex literals
- These are embedded in the `SFPU_OP_CHAIN` init macro string
- The init function `softcap_init()` decodes them and stores in programmable constant registers (`vConstFloatPrgm0` = cap, `vConstFloatPrgm1` = inv_cap)
- The per-tile function `calculate_softcap()` reads the constants from these registers

This approach preserves full float32 precision for the cap parameter.

## Reference Operations Used

1. **swish** (most useful): Provided the piecewise approximation pattern with non-nested v_if/v_endif cascade for segment selection. The swish kernel approximates sigmoid with polynomial + linear + saturation segments using the exact same SFPI control flow pattern.

2. **atanh** (second most useful): Provided the pattern for using programmable constant registers (`vConstFloatPrgm0/1/2`) to pass precomputed values from the init function to the per-tile SFPU kernel. Also demonstrated Horner-form polynomial evaluation in SFPI.

3. **hardtanh**: Showed the parametrized operation pattern with `s2vFloat16b` for decoding packed parameters, and the `is_parametrized_type` registration. Also showed the non-nested v_if cascade pattern for piecewise functions.

4. **sinh**: Confirmed the standard LLK dispatch pattern and the `llk_math_eltwise_unary_sfpu_init` overload that accepts an init callback with forwarded args.

5. **tanhshrink**: Provided context on the tanhshrink/tanh computation patterns in the codebase.

## Deviations from Standard Patterns

1. **Two-parameter init**: Unlike most ops that have zero or one init parameter, softcap passes two uint32 params (cap and inv_cap) to the init function. This works because `llk_math_eltwise_unary_sfpu_init` uses variadic templates for forwarding init callback arguments.

2. **Piecewise tanh instead of pure Taylor**: The Taylor series for tanh has a convergence radius of pi/2 (~1.57), so it diverges for |u| > 1.5. The implementation adds quadratic polynomial segments for the middle range [1.0, 3.0] to maintain accuracy. This is consistent with the spec's "extended Taylor series" + "exact saturation" approach.

3. **Union type punning in init**: The init function uses a union to convert uint32 bit patterns back to float for assignment to programmable constant registers. This is technically UB in C++ but is the standard embedded pattern and works correctly on all RISC-V GCC versions used by this project.

## Known Limitations

- The Taylor series is only used for |u| <= 1.0 due to the convergence radius limitation. For 1.0 < |u| <= 3.0, quadratic fits are used instead.
- For very small cap values (cap < 0.01), the inv_cap value becomes very large, which may cause overflow in the `u = x * inv_cap` computation for large inputs. However, the saturation at |u| > 3.0 handles this gracefully.
- The quadratic fits introduce up to ~0.005 absolute error in tanh, which translates to ~0.005 * cap error in the final output. For cap=50, this is ~0.25, which is ~1 ULP in BF16.

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h

### Modified Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
ttnn/ttnn/operations/unary.py
