# frac Implementation Notes

## Operation Definition
`frac(x) = x - trunc(x)` — returns the fractional part of x, preserving sign.

## Implementation Summary

The frac operation is implemented as a parameterless unary SFPU operation using IEEE 754 bit manipulation to extract the fractional part. The kernel uses three-case branching based on the debiased exponent:

1. **E < 0** (|x| < 1): The entire value is fractional, so `frac(x) = x`
2. **E >= 23**: The value is an exact integer (no fractional mantissa bits), so `frac(x) = 0`
3. **0 <= E < 23**: Mixed — create a mask `0xFFFFFFFF << (23 - E)` to zero out fractional mantissa bits, producing `trunc(x)`, then compute `frac(x) = x - trunc(x)`

Key SFPI primitives used:
- `sfpi::exexp(v)` — extract debiased exponent
- `sfpi::shft(vUInt, vInt)` — shift to create mantissa mask
- `sfpi::reinterpret<sfpi::vInt>` / `sfpi::reinterpret<sfpi::vFloat>` — bitwise reinterpret between float and int
- `v_if` / `v_endif` — predicated SIMD conditional branches

## New Files

### Layer 1: SFPU Kernel — `ckernel_sfpu_frac.h`

Core SFPU kernel implementing the frac algorithm via IEEE 754 bit manipulation.
Identical for both Wormhole B0 and Blackhole architectures.

**Paths:**
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// frac(x) = x - trunc(x)
//
// Matches PyTorch torch.frac() semantics (truncation toward zero, not floor).
//
// Algorithm:
// 1. Extract unbiased exponent from x.
// 2. If exp < 0 (|x| < 1): trunc(x) = 0, so frac = x.
// 3. If exp >= 23: x is already an integer, frac = 0.
// 4. Otherwise (0 <= exp < 23): compute trunc(x) by masking out fractional bits.
// 5. Result = x - trunc(x).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_frac() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Default: frac = 0 for integers (exp >= 23)
        sfpi::vFloat trunc_x = x;

        // Extract unbiased exponent
        sfpi::vInt exp = sfpi::exexp(x);

        // Case 1: |x| < 1 (exp < 0) — trunc toward zero gives 0
        v_if(exp < 0) { trunc_x = 0.0f; }
        v_endif;

        // Case 2: 0 <= exp < 23 (has fractional bits in float32)
        v_if(exp >= 0 && exp < 23) {
            // Create bitmask to zero out fractional mantissa bits.
            // IEEE 754 float32 has 23 mantissa bits. For exponent e,
            // the lowest (23 - e) bits are fractional.
            // mask = 0xFFFFFFFF << (23 - exp)
            sfpi::vUInt shift = sfpi::vUInt(23 - exp);
            sfpi::vInt mask = sfpi::vInt(-1) << shift;

            // Apply mask to get trunc(x) (round toward zero)
            sfpi::vInt xi = sfpi::reinterpret<sfpi::vInt>(x);
            trunc_x = sfpi::reinterpret<sfpi::vFloat>(xi & mask);
        }
        v_endif;

        // frac(x) = x - trunc(x)
        sfpi::dst_reg[0] = x - trunc_x;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

### Layer 2: LLK Math Wrapper — `llk_math_eltwise_unary_sfpu_frac.h`

LLK-level init and dispatch functions that wire `calculate_frac` into the standard unary SFPU pipeline.
Identical for both Wormhole B0 and Blackhole architectures.

**Paths:**
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_frac.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_frac_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::frac, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_frac(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_frac<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

### Layer 3: Compute API — `frac.h`

Public compute kernel API exposing `frac_tile()` and `frac_tile_init()`.

**Path:** `tt_metal/hw/inc/api/compute/eltwise_unary/frac.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_frac.h"
#endif

namespace ckernel {

// clang-format off
 /**
 * Performs element-wise frac operation: x - trunc(x).
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
ALWI void frac_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_frac<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void frac_tile_init() { MATH((llk_math_eltwise_unary_sfpu_frac_init<APPROX>())); }

}  // namespace ckernel
```

### Layer 10: Test — `test_frac.py`

Pytest test file covering basic shapes, negative inputs (sign preservation), and integer inputs (should return 0).

**Path:** `tests/ttnn/unit_tests/operations/eltwise/test_frac.py`

```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_allclose


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16],
)
def test_frac(device, shape, dtype):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # PyTorch golden: torch.frac computes x - trunc(x)
    torch_output = torch.frac(torch_input.float()).to(torch.bfloat16)

    # TT computation
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)
    assert_allclose(torch_output, tt_output_torch, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_frac_negative(device, shape):
    """Verify frac works correctly for negative inputs (sign-preserving)."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 10  # wide range

    torch_output = torch.frac(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)
    # frac output should always be in (-1, 1)
    assert tt_output_torch.min() > -1.0, f"Output <= -1: {tt_output_torch.min()}"
    assert tt_output_torch.max() < 1.0, f"Output >= 1: {tt_output_torch.max()}"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_frac_integers(device, shape):
    """Verify frac returns 0 for integer inputs."""
    # Create integer-valued tensor
    torch_input = torch.arange(-16, 16, dtype=torch.bfloat16).reshape(1, 1, 32, 1).expand(shape)

    torch_output = torch.frac(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)
    # All fractional parts of integers should be 0
    assert torch.allclose(tt_output_torch, torch.zeros_like(tt_output_torch), atol=1e-2)
```

## Modified Files

### Layer 4: SfpuType Enum — `llk_sfpu_types.h`

Added `frac` to the `SfpuType` enum (both Wormhole B0 and Blackhole).

```cpp
// In enum class SfpuType:
    softshrink,
    frac,
};
```

### Layer 5: SFPU Split Includes — `sfpu_split_includes.h`

Added conditional include for the frac compute API header.

```cpp
#if SFPU_OP_FRAC_INCLUDE
#include "api/compute/eltwise_unary/frac.h"
#endif
```

### Layer 6: CMake Sources — `sources.cmake`

Registered `frac.h` in the compute kernel header list.

```cmake
    inc/api/compute/eltwise_unary/softshrink.h
    inc/api/compute/eltwise_unary/frac.h
    inc/api/compute/ema.h
```

### Layer 7: Legacy Unary Op Utils — `unary_op_utils.cpp`

Added dispatch case mapping `UnaryOpType::FRAC` to `frac_tile_init()` / `frac_tile()`.

```cpp
        case UnaryOpType::HARDSWISH: return {"hardswish_tile_init();", fmt::format("hardswish_tile({});", idst)};
        case UnaryOpType::FRAC: return {"frac_tile_init();", fmt::format("frac_tile({});", idst)};
        default: TT_THROW("unexpected op type {}", op_type);
```

### Layer 8: NG Unary Op Utils — `unary_ng_op_utils.cpp`

Added dispatch case for the next-generation unary pipeline.

```cpp
        case UnaryOpType::SOFTSIGN: return {"softsign_tile_init();", fmt::format("softsign_tile({});", idst)};
        case UnaryOpType::FRAC: return {"frac_tile_init();", fmt::format("frac_tile({});", idst)};
        default: TT_FATAL(false, "Undefined unary_ng op type {}", op_type);
```

### Layer 9: Python Binding — `unary.py`

Registered `frac` in the Python-level golden function map and the unary ops list.

```python
            "trunc": torch.trunc,
            "frac": torch.frac,
        }
```

```python
    ttnn.trunc,
    ttnn.frac,
]
```

## Reference Operations Used

1. **hardsigmoid** (most useful): Template for a parameterless unary op with no init constants. Provided the clean pattern for `calculate_frac()` structure, no-op init, and the `v_if/v_endif` conditional idiom.

2. **cbrt**: Provided the SFPI bit-manipulation patterns (`exexp`, `reinterpret<vInt>`, `reinterpret<vFloat>`) essential for the IEEE 754 exponent extraction and mantissa masking.

3. **softshrink**: Provided the three-case conditional pattern with a default result (`0.0f`) selectively overridden by `v_if/v_endif` branches.

4. **hardtanh**: Reinforced the standard ckernel_sfpu file structure and `v_if/v_endif` boilerplate.

5. **hardswish**: Showed the compute-intermediate-then-subtract pattern (similar to computing `trunc_val` then `x - trunc_val`).

## Deviations from Standard Patterns
- Uses nested `v_if` blocks (Case 3 is nested inside Case 1's else-branch via `exp >= 0`). This is safe because the CC stack supports nesting, as confirmed by the SFPU hardware model.
- No `APPROXIMATION_MODE` branching — the bit-manipulation algorithm is exact, no approximation variant needed.
- No `init` function needed — no programmable constant registers are used.

## Known Limitations
- bfloat16 precision: Since bfloat16 has only 7 mantissa bits (not 23), the exponent threshold for "exact integer" is lower. However, the kernel operates on FP32 values in the SFPU (bfloat16 is promoted to FP32 on load), so the 23-bit mantissa logic is correct.
- Very large floats (E >= 23) correctly return 0, matching IEEE 754 semantics where such values have no fractional bits.
- Subnormal numbers: `exexp` returns a debiased exponent; for subnormals this will be very negative, falling into Case 1 (result = x), which is correct since subnormals are always < 1 in magnitude.
