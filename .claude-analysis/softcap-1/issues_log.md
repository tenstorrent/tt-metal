# Issues Log: softcap

## Configuration
- **Operation**: softcap
- **Math definition**: cap * tanh(x / cap)
- **Source**: direct formula
- **Output folder**: `.claude-analysis/softcap-1/`
- **Date**: 2026-04-14

## Phase Timeline
| Phase | Description | Status | Duration | Issues |
|-------|-------------|--------|----------|--------|
| 1 | Reference Discovery | ok | ~234s | None |
| 2 | Reference Analysis | ok | ~744s | Some commit delays from parallel agents |
| 3 | Implementation | ok | ~542s | Initial kernel used unavailable SFPI operations |
| 4 | Testing & Debugging | partial | ~1636s | fp32 tests fail due to Pade approximation limits |
| 5 | Documentation | ok | ~60s | None |

## Issues
1. **SFPI register overflow**: Newton-Raphson reciprocal inside loop body causes "maximum number of generated reload insns" compiler error. Resolved by pre-computing cap reciprocal outside loop and using Pade reformulation.
2. **SFPI unary negation ICE**: `-vFloat` causes internal compiler error. Resolved by using `sfpi::setsgn()`.
3. **FP32 precision**: Pade [5,4] approximation insufficient for fp32 2-ULP accuracy in tanh transition region |u|=2.5-4.0. Root cause: SFPI register limit prevents higher-order approximations.
4. **sfpi::vConst1 type mismatch**: `sfpi::setsgn(sfpi::vConst1, u)` fails to compile. Must assign vConst1 to local vFloat first.
5. **param0 format mismatch**: Initially used `s2vFloat16b` to interpret params but they're passed as raw float32 bits. Fixed to use `reinterpret<vFloat>(vInt(param0))`.

## File Manifest
### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h`

### Modified Files
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
