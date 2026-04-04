# Reference Analysis: cbrt (CBRT)

## Overview
cbrt(x) = x^(1/3). Simple non-parameterized operation showing the full 12-layer pattern.

## File Structure (complete layer list)

### Layer 1: SFPU Kernel
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h`

### Layer 2: LLK Dispatch
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h`

### Layer 3: LLK API Include
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h` (include added)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_unary_sfpu_api.h` (include added)

### Layer 4: SfpuType Enum
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`

### Layer 5: Compute API Header
- `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h`

### Layer 6: SFPU Split Includes
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`

### Layer 7: UnaryOpType Enum (already exists)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`

### Layer 8: Op Utils - get_macro_definition, get_op_init_and_func_default
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

### Layer 9: Op Utils Header - is_parametrized_type (N/A for non-param ops)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`

### Layer 10: C++ API Registration (already exists)
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`

### Layer 11: Python Nanobinding
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`

### Layer 12: Python Golden Function
- `ttnn/ttnn/operations/unary.py`
