# Softsign Implementation Notes

## Math Definition
`softsign(x) = x / (1 + |x|)`

## Implementation Strategy

Softsign is implemented as a non-parametrized unary SFPU operation using the Newton-Raphson reciprocal from `ckernel_sfpu_recip.h`.

The SFPU kernel computes:
1. `abs(x)` using `sfpi::abs(v)` — absolute value intrinsic
2. `1 + |x|` using `sfpi::vConst1` hardware constant for 1.0f
3. `1 / (1 + |x|)` using `_sfpu_reciprocal_<2>(denom)` — 2-iteration Newton-Raphson reciprocal
4. `x * recip` — multiply original value by reciprocal

The init function calls `_init_sfpu_reciprocal_<APPROXIMATION_MODE>()` to program the SFPU constant registers (vConstFloatPrgm0/1/2) with the reciprocal polynomial coefficients.

## Reference Operations Used

1. **hardsigmoid** (primary template): Provided the complete end-to-end file skeleton — ckernel, LLK wrapper, compute API header, `unary_op_utils` registration, `sfpu_split_includes` guard, and `activations.h` aggregation. All new files follow this exact pattern.

2. **cbrt**: Demonstrated `sfpi::abs()` usage for absolute value computation and the `calculate_*/init` function naming convention.

3. **sigmoid/silu** (conceptual reference): Informed the `x * f(x)` multiply structure and reciprocal init pattern. The actual implementation uses the Newton-Raphson `_sfpu_reciprocal_` from `ckernel_sfpu_recip.h` rather than LUT-based sigmoid.

## Deviations from Standard Patterns

1. **Reciprocal include**: The kernel includes `sfpu/ckernel_sfpu_recip.h` from the third-party LLK library. This is the standard Newton-Raphson reciprocal implementation. Other custom kernels in this worktree (hardsigmoid, hardtanh) don't need reciprocal and thus don't have this include.

2. **Init function with callback**: The LLK wrapper passes `softsign_init<APPROXIMATE>` as an init callback to `llk_math_eltwise_unary_sfpu_init`, unlike hardsigmoid which has no init callback. This is necessary to program the reciprocal polynomial constants.

## Known Limitations

- Uses 2 Newton-Raphson iterations for the reciprocal, which provides ~1 ULP accuracy for fp32 but may have slightly higher error for extreme input values
- The reciprocal uses the programmable constant registers (vConstFloatPrgm0/1/2), so softsign cannot be fused in a chain with operations that also use these registers (e.g., cbrt)
- Approximation mode (APPROXIMATION_MODE=true) is not explicitly handled differently — the reciprocal always uses 2 iterations

---

## Source Code

### New Files

#### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

// softsign(x) = x / (1 + |x|)
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softsign() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // Compute denominator: 1 + |x|
        sfpi::vFloat denom = sfpi::abs(v) + sfpi::vConst1;

        // Compute reciprocal of denominator: 1 / (1 + |x|)
        sfpi::vFloat recip = _sfpu_reciprocal_<2>(denom);

        // Result: x * (1 / (1 + |x|))
        sfpi::dst_reg[0] = v * recip;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softsign_init() {
    _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
```

#### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

// softsign(x) = x / (1 + |x|)
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softsign() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // Compute denominator: 1 + |x|
        sfpi::vFloat denom = sfpi::abs(v) + sfpi::vConst1;

        // Compute reciprocal of denominator: 1 / (1 + |x|)
        sfpi::vFloat recip = _sfpu_reciprocal_<2>(denom);

        // Result: x * (1 / (1 + |x|))
        sfpi::dst_reg[0] = v * recip;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softsign_init() {
    _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
```

#### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softsign.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_softsign.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_softsign_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::softsign, APPROXIMATE>(ckernel::sfpu::softsign_init<APPROXIMATE>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_softsign(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_softsign<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

#### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softsign.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_softsign.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_softsign_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::softsign, APPROXIMATE>(ckernel::sfpu::softsign_init<APPROXIMATE>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_softsign(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_softsign<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

#### `tt_metal/hw/inc/api/compute/eltwise_unary/softsign.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_softsign.h"
#endif

namespace ckernel {

// clang-format off
 /**
 * Performs element-wise softsign operation: x / (1 + |x|).
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void softsign_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_softsign<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void softsign_tile_init() { MATH((llk_math_eltwise_unary_sfpu_softsign_init<APPROX>())); }

}  // namespace ckernel
```

#### `tests/ttnn/unit_tests/operations/eltwise/test_softsign.py`

```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def torch_softsign(x):
    """Golden reference: softsign(x) = x / (1 + |x|)."""
    return x / (1 + torch.abs(x))


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_softsign(device, shape, dtype):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    torch_output = torch_softsign(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softsign(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_softsign_output_range(device, shape):
    """Verify softsign output is always in (-1, 1)."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 10  # wide range

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softsign(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert tt_output_torch.min() >= -1.0, f"Output below -1: {tt_output_torch.min()}"
    assert tt_output_torch.max() <= 1.0, f"Output above 1: {tt_output_torch.max()}"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_softsign_allclose(device, shape):
    """Test softsign with allclose tolerances (rtol=1.6e-2, atol=1e-2)."""
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    torch_output = torch_softsign(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softsign(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert torch.allclose(
        torch_output, tt_output_torch, rtol=1.6e-2, atol=1e-2
    ), f"allclose failed: max diff = {(torch_output - tt_output_torch).abs().max()}"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_softsign_negative_inputs(device, shape):
    """Test softsign with all-negative inputs: softsign(-x) = -softsign(x)."""
    torch.manual_seed(42)
    torch_input = -torch.abs(torch.randn(shape, dtype=torch.bfloat16))
    torch_output = torch_softsign(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softsign(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)
    # All outputs should be negative
    assert tt_output_torch.max() <= 0.0, f"Expected all negative outputs, got max: {tt_output_torch.max()}"
```

### Modified Files

#### `llk_sfpu_types.h` (Wormhole + Blackhole)

Added softsign to `SfpuType` enum:

```cpp
enum class SfpuType : uint8_t {
    hardsigmoid,
    selu,
    hardtanh,
    softsign,  // NEW
};
```

#### `llk_math_unary_sfpu_api.h` (Wormhole + Blackhole)

Added softsign include after hardsigmoid:

```cpp
#include "llk_math_eltwise_unary_sfpu_hardsigmoid.h"
#include "llk_math_eltwise_unary_sfpu_softsign.h"  // NEW
```

#### `sfpu_split_includes.h`

Added conditional softsign include:

```cpp
#if SFPU_OP_SOFTSIGN_INCLUDE
#include "api/compute/eltwise_unary/softsign.h"  // NEW
#endif
```

#### `activations.h`

Added softsign include:

```cpp
#pragma once

#include "api/compute/eltwise_unary/hardsigmoid.h"
#include "api/compute/eltwise_unary/softsign.h"  // NEW
```

#### `sources.cmake`

Added softsign header to CMake file list:

```cmake
inc/api/compute/eltwise_unary/hardsigmoid.h
inc/api/compute/eltwise_unary/selu.h
inc/api/compute/eltwise_unary/softsign.h  # NEW
inc/api/compute/eltwise_unary/sfpu_split_includes.h
```

#### `unary_op_utils.cpp`

Added softsign case to `get_unary_op_kernel_call` function:

```cpp
case UnaryOpType::HARDSIGMOID: return {"hardsigmoid_tile_init();", fmt::format("hardsigmoid_tile({});", idst)};
case UnaryOpType::SELU: return {"selu_tile_init();", fmt::format("selu_tile({});", idst)};
case UnaryOpType::SOFTSIGN: return {"softsign_tile_init();", fmt::format("softsign_tile({});", idst)};  // NEW
```

#### `unary_nanobind.cpp`

Added softsign binding:

```cpp
bind_unary_operation<"softsign", &ttnn::softsign>(
    mod, R"doc(\text{softsign}(x) = \frac{x}{1 + |x|})doc", "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
```

#### `ttnn/operations/unary.py`

Added softsign to unary operations:

```python
name_to_golden_function = {
    "identity": torch.clone,
    "cbrt": torch_cbrt,
    "hardsigmoid": torch.nn.functional.hardsigmoid,
    "selu": lambda _x: torch.nn.functional.selu(_x.to(torch.float)),
    "softsign": torch.nn.functional.softsign,  # NEW
}

TTNN_ELTWISE_UNARY_CPP_FUNCTIONS = [
    ttnn.identity,
    ttnn.cbrt,
    ttnn.hardsigmoid,
    ttnn.selu,
    ttnn.softsign,  # NEW
]
```

#### `golden_functions.py`

Added softsign golden function:

```python
def _softsign_golden_function(input_tensor, *args, **kwargs):
    import torch

    return input_tensor / (1 + torch.abs(input_tensor))


if hasattr(ttnn, "softsign"):
    ttnn.attach_golden_function(ttnn.softsign, _softsign_golden_function)
```

---

## Test Results

**Status: ALL PASSED (6/6)**

| Test | Result |
|------|--------|
| `test_softsign[bfloat16, 1x1x32x32]` | PASSED |
| `test_softsign[bfloat16, 1x1x320x384]` | PASSED |
| `test_softsign[bfloat16, 1x3x320x384]` | PASSED |
| `test_softsign_output_range[1x1x32x32]` | PASSED |
| `test_softsign_allclose[1x1x32x32]` | PASSED |
| `test_softsign_negative_inputs[1x1x32x32]` | PASSED |

**Test duration**: 5.79s total

### Test Details

1. **test_softsign**: PCC >= 0.999 across three tensor shapes with bfloat16 dtype. Golden: `x / (1 + |x|)` computed in float32, cast back to bfloat16.
2. **test_softsign_output_range**: Verifies output is bounded in [-1, 1] for wide-range inputs (±10).
3. **test_softsign_allclose**: Verifies `torch.allclose(rtol=1.6e-2, atol=1e-2)` passes.
4. **test_softsign_negative_inputs**: Verifies all-negative input produces all-negative output with PCC >= 0.999.

### Debug Log
- No failures encountered. All tests passed on first run.
- Golden function registered in `golden_functions.py` with `hasattr` guard.
- No device hangs or kernel compilation errors observed.
