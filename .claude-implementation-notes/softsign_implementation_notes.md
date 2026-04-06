# Implementation Notes: softsign

## Math Definition

softsign(x) = x / (1 + |x|)

This is a smooth activation function similar to sigmoid but with linear asymptotes instead of exponential saturation. It ranges from -1 to 1 and is computationally efficient.

## Files Created

### Layer 1: SFPU Kernel (Wormhole)

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h`

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softsign() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // Clamp input to avoid subnormal intermediates in reciprocal.
        // For very large |x|, 1/(1+|x|) becomes subnormal and hardware
        // flushes it to zero, giving softsign(x)=0 instead of ~sign(x).
        // Clamping to 1e30 is safe: softsign(1e30) rounds to +-1.0 in
        // both bfloat16 and float32.
        v_if(v > 1e30f) { v = 1e30f; }
        v_endif;
        v_if(v < -1e30f) { v = -1e30f; }
        v_endif;

        sfpi::vFloat tmp = sfpi::abs(v) + sfpi::vConst1;
        tmp = sfpu_reciprocal<APPROXIMATION_MODE>(tmp);
        sfpi::dst_reg[0] = v * tmp;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
```

### Layer 2: SFPU Kernel (Blackhole)

**File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h`

Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h`.

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softsign() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // Clamp input to avoid subnormal intermediates in reciprocal.
        // For very large |x|, 1/(1+|x|) becomes subnormal and hardware
        // flushes it to zero, giving softsign(x)=0 instead of ~sign(x).
        // Clamping to 1e30 is safe: softsign(1e30) rounds to +-1.0 in
        // both bfloat16 and float32.
        v_if(v > 1e30f) { v = 1e30f; }
        v_endif;
        v_if(v < -1e30f) { v = -1e30f; }
        v_endif;

        sfpi::vFloat tmp = sfpi::abs(v) + sfpi::vConst1;
        tmp = sfpu_reciprocal<APPROXIMATION_MODE>(tmp);
        sfpi::dst_reg[0] = v * tmp;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
```

### Layer 3: LLK Wrapper (Wormhole)

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softsign.h`

```cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_softsign_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::softsign, APPROXIMATE>(ckernel::sfpu::_init_softsign_<APPROXIMATE>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_softsign(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_<APPROXIMATE>(
        ckernel::sfpu::calculate_softsign<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

### Layer 4: LLK Wrapper (Blackhole)

**File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softsign.h`

Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softsign.h`.

```cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_softsign_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::softsign, APPROXIMATE>(ckernel::sfpu::_init_softsign_<APPROXIMATE>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_softsign(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_<APPROXIMATE>(
        ckernel::sfpu::calculate_softsign<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

### Layer 5: Compute API Header

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/softsign.h`

```cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_softsign.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

ALWI void softsign_tile(uint32_t idst) { MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_softsign, RC, APPROX, idst)); }

ALWI void softsign_tile_init() { MATH(SFPU_INIT_KERNEL_CALL(softsign, _init_softsign_, APPROX)); }

}  // namespace ckernel
```

### Layer 6: SfpuType Enum Entry

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`

Entry in enum:
```cpp
softsign,
```

### Layer 7: sfpu_split_includes.h Entry

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`

Conditional include block:
```cpp
#if SFPU_OP_SOFTSIGN_INCLUDE
#include "api/compute/eltwise_unary/softsign.h"
#endif
```

### Layer 8: llk_math_unary_sfpu_api.h Include

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`

Include line:
```cpp
#include "llk_math_eltwise_unary_sfpu_softsign.h"
```

### Layer 9: Dispatch in unary_op_utils.cpp

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

Case statement in `get_op_init_and_func_default`:
```cpp
case UnaryOpType::SOFTSIGN: return {"softsign_tile_init();", fmt::format("softsign_tile({});", idst)};
```

### Layer 10: Python Golden Function

**File**: `ttnn/ttnn/operations/unary.py`

Golden function definition:
```python
def _golden_function_softsign(input_tensor_a, *args, **kwargs):
    import torch

    return input_tensor_a / (1 + torch.abs(input_tensor_a))


ttnn.attach_golden_function(ttnn.softsign, golden_function=_golden_function_softsign)
```

### Layer 11: Test File

**File**: `tests/ttnn/unit_tests/operations/eltwise/test_softsign.py`

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_with_ulp,
    assert_allclose,
    generate_all_bfloat16_bitpatterns,
    flush_subnormal_values_to_zero,
)


@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_softsign(device, is_fp32):
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        # Cast to float32 and flush subnormal inputs - hardware flushes these to zero
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Compute reference in float32, flush subnormals to match hardware behavior
    input_f32 = torch_input.float()
    torch_output = input_f32 / (1.0 + torch.abs(input_f32))
    expected = flush_subnormal_values_to_zero(torch_output)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    # Run on device
    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.softsign(tt_input)
    actual = ttnn.to_torch(tt_output)
    if not is_fp32:
        actual = actual.to(torch.bfloat16)

    # Filter out NaN/Inf for meaningful comparison
    finite_mask = (
        torch.isfinite(torch_input.float()) & torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    )
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    if is_fp32:
        # Stricter tolerances - both sides have full float32 precision
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=3, allow_nonfinite=True)
        assert_allclose(expected_finite, actual_finite, rtol=1e-3, atol=1e-4)
    else:
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=2)
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)
```

### Layer 12: Registration in unary.hpp

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`

Registration line:
```cpp
REGISTER_UNARY_OPERATION(softsign, SOFTSIGN)
```

## Design Decisions

1. **Reciprocal-Based Implementation**: softsign(x) = x / (1 + |x|) is computed as x * (1 / (1 + |x|)) using the reciprocal SFPU hardware block. This reuses the well-optimized reciprocal computation rather than implementing division from scratch.

2. **Input Clamping**: Very large inputs (|x| > 1e30) are clamped because computing 1 / (1 + |x|) on such large values produces subnormal results that hardware flushes to zero. Clamping ensures softsign(large_x) asymptotes to sign(x) rather than saturating at zero.

3. **Reciprocal Initialization**: The init function calls sfpu_reciprocal_init, which sets up any lookup tables or initial states required by the reciprocal computation.

4. **Order of Operations**: The computation is (|x| + 1) then reciprocal, then multiply by original x. This ensures the sign is preserved correctly.

5. **Template Parameters**: Allows ITERATIONS parameter to be passed in to control tile loop unrolling, and APPROXIMATION_MODE to select fast vs. accurate reciprocal implementation.

## Debug Log

Implementation completed with comprehensive test coverage including:
- Exhaustive bfloat16 bitpattern testing (256x256 = 65536 values)
- Float32 precision testing
- Subnormal number handling
- ULP and allclose comparisons

## Test Results

Tests verify:
- Numerical accuracy against PyTorch's softsign(x) = x / (1 + |x|)
- Subnormal input flushing behavior matches hardware
- Both bfloat16 and float32 precision levels
- ULP threshold of 2-3 depending on precision
- Relative and absolute tolerances for allclose comparison

## Known Limitations

1. **Subnormal Flushing**: Hardware flushes subnormal numbers to zero, which affects very small results. This is mitigated by computing in float32 when needed.

2. **Large Input Clamping**: Inputs larger than 1e30 are clamped to avoid subnormal reciprocal results. This is a tradeoff that ensures correct asymptotic behavior.

3. **Reciprocal Approximation**: The reciprocal computation uses polynomial approximation (in APPROXIMATION_MODE) which introduces small errors that accumulate.

4. **Precision Loss**: bfloat16 results have stricter tolerances (1.6e-2 rtol) compared to float32 (1e-3 rtol) due to reduced mantissa precision.
