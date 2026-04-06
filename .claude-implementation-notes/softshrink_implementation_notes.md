# Implementation Notes: softshrink

## Math Definition

The `softshrink` operation (also known as soft thresholding) is defined as:

```
softshrink(x, lambda) = {
    x - lambda,  if x > lambda
    0,           if -lambda <= x <= lambda
    x + lambda,  if x < -lambda
}
```

This operation shrinks values toward zero by a threshold amount `lambda`. For values within the [-lambda, lambda] range, the output is zero. For values outside this range, they are shifted toward zero by exactly `lambda`.

## Files Created

### Layer 1: SFPU Kernel (Wormhole)

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h`

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softshrink(uint32_t param0) {
    // param0 = bit_cast<uint32_t>(lambda)
    sfpi::vFloat lambda = Converter::as_float(param0);
    sfpi::vFloat neg_lambda = -lambda;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat result = 0.0f;  // default: 0 for -lambda <= x <= lambda

        v_if(v > lambda) { result = v - lambda; }
        v_endif;

        v_if(v < neg_lambda) { result = v + lambda; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

### Layer 2: SFPU Kernel (Blackhole)

**File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h`

Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h`

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softshrink(uint32_t param0) {
    // param0 = bit_cast<uint32_t>(lambda)
    sfpi::vFloat lambda = Converter::as_float(param0);
    sfpi::vFloat neg_lambda = -lambda;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat result = 0.0f;  // default: 0 for -lambda <= x <= lambda

        v_if(v > lambda) { result = v - lambda; }
        v_endif;

        v_if(v < neg_lambda) { result = v + lambda; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

### Layer 3: LLK Wrapper (Wormhole)

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softshrink.h`

**Status**: Not present. The wrapper is handled via macro-based dispatch.

### Layer 4: LLK Wrapper (Blackhole)

**File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softshrink.h`

**Status**: Not present. The wrapper is handled via macro-based dispatch.

### Layer 5: Compute API Header

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/softshrink.h`

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_softshrink.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

ALWI void softshrink_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_softshrink, RC, APPROX, idst, param0));
}

ALWI void softshrink_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(softshrink, APPROX)); }

}  // namespace ckernel
```

### Layer 6: SfpuType Enum Entry

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`

The enum entry is defined in the SfpuType enumeration. softshrink is registered as a variant with parameter support (float lambda).

### Layer 7: sfpu_split_includes.h

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`

The sfpu_split_includes.h file conditionally includes softshrink.h when softshrink operation is required. The exact include directive depends on the build configuration and is handled through the unified include mechanism.

### Layer 8: llk_math_unary_sfpu_api.h

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`

The softshrink operation uses the macro-based dispatch system. Dispatch is handled through SFPU_UNARY_ONE_PARAM_KERNEL_FN for parameterized operations.

### Layer 9: Dispatch (unary_op_utils.cpp)

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

Relevant case statements for softshrink dispatch:

```cpp
case UnaryOpType::SOFTSHRINK:
    return {"softshrink_tile_init();", fmt::format("softshrink_tile({}, {}u);", idst, param0_bits)};
```

### Layer 10: Python Golden (unary.py)

**File**: `ttnn/ttnn/operations/unary.py`

```python
def _golden_function_softshrink(input_tensor_a, *args, lambd=0.5, **kwargs):
    import torch

    return torch.nn.functional.softshrink(input_tensor_a, lambd=lambd)


ttnn.attach_golden_function(ttnn.softshrink, golden_function=_golden_function_softshrink)
```

### Layer 11: Test File

**File**: `tests/ttnn/unit_tests/operations/eltwise/test_softshrink.py`

```python
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_allclose

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
@pytest.mark.parametrize("lambd", [0.5, 1.0, 0.0])
def test_softshrink_bfloat16(device, input_shape, lambd):
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16) * 3.0

    torch_output = torch.nn.functional.softshrink(torch_input, lambd=lambd)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softshrink(tt_input, lambd=lambd)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, pcc=0.99)
    assert_allclose(torch_output, tt_output, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 32, 32],
        [1, 3, 320, 384],
    ],
)
def test_softshrink_default_lambd(device, input_shape):
    torch.manual_seed(42)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16) * 3.0

    torch_output = torch.nn.functional.softshrink(torch_input, lambd=0.5)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softshrink(tt_input)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, pcc=0.99)
    assert_allclose(torch_output, tt_output, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
    ],
)
@pytest.mark.parametrize("lambd", [0.5, 1.0])
def test_softshrink_memory_config(device, input_shape, lambd):
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16) * 3.0

    torch_output = torch.nn.functional.softshrink(torch_input, lambd=lambd)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softshrink(tt_input, lambd=lambd, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, pcc=0.99)
    assert_allclose(torch_output, tt_output, rtol=1.6e-2, atol=1e-2)
```

### Layer 12: Registration

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`

```cpp
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(softshrink, SOFTSHRINK)
```

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`

```cpp
bind_function<"softshrink">(
    mod, R"doc(Performs softshrink function on input_tensor with threshold lambda.)doc",
    &unary_4param_to_5param_wrapper<&ttnn::softshrink>,
    // ... parameter documentation
);
```

## Design Decisions

- **Parameterized Kernel**: Unlike simpler unary operations, softshrink takes a `lambda` parameter which is bit-cast as a uint32_t and passed to the SFPU kernel.
- **Architecture-agnostic**: The kernel implementation is identical across Wormhole and Blackhole architectures, supporting hardware portability.
- **Conditional Clamping**: The implementation uses conditional operations (v_if/v_endif) to handle the three regions of the softshrink function.

## Test Results

- **bfloat16 tests**: Validates softshrink across different lambda values (0.0, 0.5, 1.0) with PCC >= 0.99 and relative tolerance 1.6e-2.
- **Memory configuration tests**: Ensures correct output when using L1_MEMORY_CONFIG.
- **Default parameter test**: Confirms lambda defaults to 0.5 when not specified.

## Known Limitations

None documented. The operation correctly implements PyTorch's softshrink semantics with compatible floating-point accuracy.
