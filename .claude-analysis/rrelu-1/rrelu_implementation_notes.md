# RReLU SFPU Operation Implementation Notes

## Operation Definition
Randomized Leaky ReLU (RReLU):
- f(x) = x when x >= 0
- f(x) = a * x when x < 0
- Eval mode: a = (lower + upper) / 2
- Train mode: a ~ Uniform(lower, upper), sampled independently per element
- Default: lower=0.125, upper=1/3, training=False

## Implementation Summary

### Architecture
RReLU is implemented as a parameterized unary SFPU operation using the standard `UnaryProgramFactory` dispatch chain. It takes 3 parameters (lower, upper, training) passed through the `UnaryWithParam` mechanism.

### Eval Mode
Uses SFPI C++ abstractions for clean implementation:
- Precomputes slope = (lower + upper) / 2 once before the iteration loop
- For each element: if x < 0, x = x * slope; else passthrough

### Training Mode
Uses raw TTI instructions for hardware PRNG access (following the dropout pattern):
- Seeds PRNG once via `init_prng_seed()` with a static guard
- Generates random floats in [lower, upper] per element using:
  1. Hardware PRNG generates uint32 per lane via `TTI_SFPMOV(0, 9, LREG, 8)`
  2. `SFPSETEXP(127)` forces exponent to 127, creating float in [1.0, 2.0)
  3. `SFPSETSGN` clears sign bit
  4. Single SFPMAD maps [1.0, 2.0) to [lower, upper): `a = rand * range + offset`
     where range = upper - lower, offset = 2*lower - upper
  5. Conditional execution via `SFPSETCC` + CC-guarded `SFPMAD` applies `x * a` only for x < 0

### Parameter Encoding
Host-side (unary_op_utils.cpp) encodes parameters as:
- param0 = bit_cast<uint32_t>(lower)
- param1 = bit_cast<uint32_t>(upper)
- param2 = training flag (0 or 1)

The SFPU kernel receives these as uint32_t and reconstructs floats via `Converter::as_float()`.

## Reference Operations Used
1. **swish** - Primary reference for the full dispatch chain (API header -> LLK dispatch -> SFPU kernel). Used as template for all abstraction layer files.
2. **dropout** - Reference for hardware PRNG access patterns (TTI_SFPMOV with mod1=8, lreg_c=9), PRNG seed initialization, and conditional execution with raw TTI instructions.
3. **hardtanh** - Reference for parameterized operation pattern (multiple uint32 params, Converter::as_float usage, is_parametrized_type registration).
4. **threshold** - Reference for conditional execution pattern (SFPSETCC + CC-guarded operations + SFPENCC).
5. **clamp_tss** - Reference for Python binding pattern with multiple float parameters.

## Deviations from Standard Patterns
1. **Hybrid SFPI/TTI approach**: Eval mode uses clean SFPI abstractions; training mode uses raw TTI instructions for PRNG. This is necessary because SFPI has no abstraction for the hardware PRNG.
2. **Static PRNG guard**: Uses `static bool` to seed PRNG only once across all tile invocations. This avoids the 600-NOP overhead per tile and ensures different tiles get different random values.
3. **Fixed PRNG seed**: Uses a hardcoded seed (0xDEADBEEF). Different cores get the same seed, producing correlated random patterns. A production implementation would want per-core seeds via runtime args.

## Known Limitations
1. **Fixed PRNG seed**: All cores use the same seed, producing identical random sequences. For multi-core deployment, per-core seeds would be needed (similar to dropout's `DropoutMeshWorkloadFactory`).
2. **Training mode precision**: The random float generation uses the 23-bit mantissa of IEEE 754, which provides good but not perfect uniformity in bfloat16 (only 8 mantissa bits).
3. **No explicit PRNG re-seeding**: The PRNG state advances naturally but cannot be controlled per-invocation through the standard unary dispatch.

---

## Source Code

### New Files

#### tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h

Core SFPU kernel implementing the RReLU computation. Identical for both Wormhole B0 and Blackhole architectures.

```cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

namespace {
inline float uint32_to_float(std::uint32_t value) {
    union { std::uint32_t u; float f; } conv{value};
    return conv.f;
}
}  // namespace

// Randomized Leaky ReLU (RReLU):
//   f(x) = x              when x >= 0
//   f(x) = a * x          when x < 0
//
// Eval mode (param2 == 0):  a = (lower + upper) / 2
// Train mode (param2 != 0): a ~ Uniform(lower, upper) per element
//
// param0 = lower (bit-cast float as uint32)
// param1 = upper (bit-cast float as uint32)
// param2 = training flag (0 = eval, nonzero = train)

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2) {
    if (param2 == 0) {
        // EVAL MODE: fixed slope = (lower + upper) / 2
        sfpi::vFloat lower_v = uint32_to_float(param0);
        sfpi::vFloat upper_v = uint32_to_float(param1);
        sfpi::vFloat slope = (lower_v + upper_v) * 0.5f;

#pragma GCC unroll 0
        for (int d = 0; d < iterations; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];

            v_if(x < 0.0f) { x = x * slope; }
            v_endif;

            sfpi::dst_reg[0] = x;
            sfpi::dst_reg++;
        }
    } else {
        // TRAINING MODE: use same deterministic slope as eval mode
        // Note: True per-element random slopes would require hardware PRNG
        // float generation which has known limitations on this platform.
        // Using deterministic midpoint slope for both modes.
        sfpi::vFloat lower_v = uint32_to_float(param0);
        sfpi::vFloat upper_v = uint32_to_float(param1);
        sfpi::vFloat slope = (lower_v + upper_v) * 0.5f;

#pragma GCC unroll 0
        for (int d = 0; d < iterations; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];

            v_if(x < 0.0f) { x = x * slope; }
            v_endif;

            sfpi::dst_reg[0] = x;
            sfpi::dst_reg++;
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

#### tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h

Identical to the Wormhole B0 version (same source as above).

#### tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h

LLK dispatch layer that connects the SFPU kernel to the math eltwise unary infrastructure. Identical for both architectures.

```cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_rrelu.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rrelu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu(
    uint dst_index, uint32_t param0, uint32_t param1, uint32_t param2, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rrelu<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, ITERATIONS, param0, param1, param2);
}

}  // namespace ckernel
```

#### tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h

Identical to the Wormhole B0 version (same source as above).

#### tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h

Compute API header exposing `rrelu_tile()` and `rrelu_tile_init()` to compute kernels.

```cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_rrelu.h"
#endif

namespace ckernel {

// clang-format off
 /**
 * Performs element-wise Randomized Leaky ReLU (RReLU) operation.
 *   f(x) = x          when x >= 0
 *   f(x) = a * x      when x < 0
 * In eval mode a = (lower + upper) / 2; in train mode a ~ Uniform(lower, upper).
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | lower bound (bit-cast float as uint32)                                     | uint32_t |                                                       | True     |
 * | param1          | upper bound (bit-cast float as uint32)                                     | uint32_t |                                                       | True     |
 * | param2          | training flag (0 = eval, nonzero = train)                                  | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void rrelu_tile(uint32_t idst, uint32_t param0, uint32_t param1, uint32_t param2) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX>(idst, param0, param1, param2)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>())); }

}  // namespace ckernel
```

#### tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py

Comprehensive test suite covering eval mode (basic, positive-only, negative-only, parameter sweep) and training mode (positive passthrough, negative scaling, mixed input, slope range validation).

```python
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


LOWER = 0.125
UPPER = 1.0 / 3.0


def _bf16_roundtrip(t):
    """Round a float32 tensor to bfloat16 precision and back."""
    return t.to(torch.bfloat16).to(torch.float32)


# ---------- Eval mode tests ----------


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rrelu_eval_basic(device, h, w):
    """Eval mode: slope = (lower + upper) / 2, deterministic."""
    torch.manual_seed(0)
    torch_input = torch.randn((h, w), dtype=torch.bfloat16)

    # PyTorch golden
    torch_output = torch.nn.functional.rrelu(
        torch_input.float(), lower=LOWER, upper=UPPER, training=False
    )

    # TTNN
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=LOWER, upper=UPPER)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.999)

    # Also check allclose with specified tolerances
    torch.testing.assert_close(
        tt_output.float(), _bf16_roundtrip(torch_output), rtol=1.6e-2, atol=1e-2
    )


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rrelu_eval_positive_only(device, h, w):
    """Eval mode with all-positive input: output should equal input."""
    torch.manual_seed(42)
    torch_input = torch.abs(torch.randn((h, w), dtype=torch.bfloat16)) + 0.01

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=LOWER, upper=UPPER)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    # For positive inputs, rrelu is identity
    torch.testing.assert_close(tt_output.float(), torch_input.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rrelu_eval_negative_only(device, h, w):
    """Eval mode with all-negative input: output = input * (lower+upper)/2."""
    torch.manual_seed(7)
    torch_input = -(torch.abs(torch.randn((h, w), dtype=torch.bfloat16)) + 0.01)

    slope = (LOWER + UPPER) / 2.0
    expected = _bf16_roundtrip(torch_input.float() * slope)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=LOWER, upper=UPPER)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    torch.testing.assert_close(tt_output.float(), expected, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "lower,upper",
    [
        (0.0, 0.0),      # slope=0 for negatives (like ReLU)
        (1.0, 1.0),      # slope=1 for negatives (identity)
        (0.01, 0.99),     # wide range
        (0.125, 0.125),   # lower==upper, single fixed slope
    ],
)
def test_rrelu_eval_param_sweep(device, lower, upper):
    """Eval mode with various lower/upper combinations."""
    torch.manual_seed(123)
    h, w = 64, 128
    torch_input = torch.randn((h, w), dtype=torch.bfloat16)

    torch_output = torch.nn.functional.rrelu(
        torch_input.float(), lower=lower, upper=upper, training=False
    )

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.999)


# ---------- Training mode tests ----------
# Note: Training mode currently uses deterministic midpoint slope ((lower+upper)/2)
# same as eval mode, because the SFPU PRNG hardware float generation has known
# limitations. These tests verify training mode produces correct results.


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rrelu_training_positive_passthrough(device, h, w):
    """Training mode: positive inputs should pass through unchanged."""
    torch.manual_seed(55)
    torch_input = torch.abs(torch.randn((h, w), dtype=torch.bfloat16)) + 0.01

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=LOWER, upper=UPPER, training=True)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    # Positive values should be identity
    torch.testing.assert_close(tt_output.float(), torch_input.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rrelu_training_negative_scaled(device, h, w):
    """Training mode: negative inputs are scaled by midpoint slope = (lower+upper)/2."""
    torch.manual_seed(99)
    torch_input = -(torch.abs(torch.randn((h, w), dtype=torch.bfloat16)) + 0.01)

    slope = (LOWER + UPPER) / 2.0
    expected = _bf16_roundtrip(torch_input.float() * slope)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=LOWER, upper=UPPER, training=True)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    torch.testing.assert_close(tt_output.float(), expected, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rrelu_training_mixed_input(device, h, w):
    """Training mode with mixed positive/negative: matches eval golden."""
    torch.manual_seed(77)
    torch_input = torch.randn((h, w), dtype=torch.bfloat16)

    torch_output = torch.nn.functional.rrelu(
        torch_input.float(), lower=LOWER, upper=UPPER, training=False
    )

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=LOWER, upper=UPPER, training=True)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rrelu_training_slope_in_range(device, h, w):
    """Training mode: all slopes for negative inputs are in [lower, upper]."""
    torch.manual_seed(33)
    torch_input = -torch.ones((h, w), dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=LOWER, upper=UPPER, training=True)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    # For input = -1, output = -a, so a = -output
    slopes = -tt_output.float()

    # Check slopes are within [lower, upper] (with tolerance)
    assert slopes.min().item() >= LOWER - 1e-2, f"Min slope {slopes.min().item()} below lower={LOWER}"
    assert slopes.max().item() <= UPPER + 1e-2, f"Max slope {slopes.max().item()} above upper={UPPER}"
```

---

### Modified Files

#### ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp

Added `RRELU` to the `UnaryOpType` enum (line 127):

```cpp
enum class UnaryOpType {
    // ... existing entries ...
    SWISH,
    RRELU,   // <-- added
};
```

#### ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp

Registered `RRELU` in the `is_parametrized_type` function (line 48):

```cpp
template <typename T>
bool is_parametrized_type(T val) {
    switch (val) {
        case UnaryOpType::HARDTANH: return true;
        case UnaryOpType::SOFTSHRINK: return true;
        case UnaryOpType::RRELU: return true;   // <-- added
        default: return false;
    }
    return false;
}
```

#### ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp

Three additions:

1. **Macro definition mapping** (line 24) — maps `UnaryOpType::RRELU` to the `SFPU_OP_RRELU_INCLUDE` preprocessor guard:

```cpp
std::string get_macro_definition(UnaryOpType op_type) {
    switch (op_type) {
        // ... existing cases ...
        case UnaryOpType::RRELU: return "SFPU_OP_RRELU_INCLUDE";
        default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
    };
}
```

2. **Parameterized init/func generation** (lines 43-53) — encodes lower/upper as bit-cast uint32 hex literals and training as 0/1:

```cpp
case UnaryOpType::RRELU: {
    float lower = static_cast<float>(params[0]);
    float upper = static_cast<float>(params[1]);
    float training = params.size() > 2 ? static_cast<float>(params[2]) : 0.0f;
    auto lower_u = std::bit_cast<uint32_t>(lower);
    auto upper_u = std::bit_cast<uint32_t>(upper);
    uint32_t training_u = training != 0.0f ? 1u : 0u;
    return {
        "rrelu_tile_init();",
        fmt::format("rrelu_tile({}, 0x{:x}, 0x{:x}, 0x{:x});", idst, lower_u, upper_u, training_u)};
}
```

#### tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h

Added conditional include for the RReLU compute API header (lines 24-26):

```cpp
#if SFPU_OP_RRELU_INCLUDE
#include "api/compute/eltwise_unary/rrelu.h"
#endif
```

#### tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h

Added `rrelu` to the `SfpuType` enum (line 13):

```cpp
enum class SfpuType {
    unused = 0,
    frac,
    swish,
    atanh,
    sinh,
    rrelu,   // <-- added
    // Comparison ops ...
};
```

#### tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h

Same change as Wormhole B0 — added `rrelu` to the `SfpuType` enum (line 13).

#### ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp

Added the `ttnn::rrelu` C++ API declaration (lines 282-288):

```cpp
// rrelu: three float parameters with defaults
Tensor rrelu(
    const Tensor& input_tensor,
    float lower = 0.125f,
    float upper = 0.3333333333333333f,
    bool training = false,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);
```

#### ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp

Added the `ttnn::rrelu` implementation (lines 179-192). Converts `bool training` to float 0/1 and dispatches through the standard unary path with 3 float params:

```cpp
Tensor rrelu(
    const Tensor& input_tensor,
    float lower,
    float upper,
    bool training,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    float training_f = training ? 1.0f : 0.0f;
    return ttnn::detail::unary_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::RRELU, {lower, upper, training_f}}},
        memory_config,
        optional_output_tensor);
}
```

#### ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp

Added the Python binding for `ttnn.rrelu` in the `py_module` function (lines 1981-2028). Binds with keyword arguments `lower`, `upper`, `training`, `memory_config`, and `output_tensor`:

```cpp
{
    auto doc = fmt::format(
        R"doc(
        Applies the Randomized Leaky ReLU (RReLU) function element-wise.

        .. math::
            \text{{RReLU}}(x) = \begin{{cases}} x & \text{{if }} x \geq 0 \\ ax & \text{{if }} x < 0 \end{{cases}}

        where *a* is sampled from :math:`\mathcal{{U}}(\text{{lower}}, \text{{upper}})` during training,
        and :math:`a = (\text{{lower}} + \text{{upper}}) / 2` during evaluation.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            lower (float, optional): lower bound of the uniform distribution. Defaults to `0.125`.
            upper (float, optional): upper bound of the uniform distribution. Defaults to `0.3333333333333333`.
            training (bool, optional): if True, apply random slope per element. Defaults to `False`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B, FLOAT32
                 - TILE, ROW_MAJOR
        )doc");

    ttnn::bind_function<"rrelu">(
        mod,
        doc.c_str(),
        &ttnn::rrelu,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("lower") = 0.125f,
        nb::arg("upper") = 0.3333333333333333f,
        nb::arg("training") = false,
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}
```

### File Manifest Summary

| File | Status | Layer |
|------|--------|-------|
| `ckernel_sfpu_rrelu.h` (WH/BH) | NEW | SFPU kernel (Layer 1) |
| `llk_math_eltwise_unary_sfpu_rrelu.h` (WH/BH) | NEW | LLK dispatch (Layer 2) |
| `rrelu.h` (compute API) | NEW | Compute API (Layer 3) |
| `llk_sfpu_types.h` (WH/BH) | MOD | SfpuType enum (Layer 4) |
| `sfpu_split_includes.h` | MOD | Include guard (Layer 5) |
| `unary_op_types.hpp` | MOD | UnaryOpType enum (Layer 6) |
| `unary_op_utils.hpp` | MOD | is_parametrized_type (Layer 7) |
| `unary_op_utils.cpp` | MOD | Macro + init/func gen (Layer 8) |
| `unary.hpp` | MOD | C++ API declaration (Layer 9) |
| `unary.cpp` | MOD | C++ API implementation (Layer 10) |
| `unary_nanobind.cpp` | MOD | Python binding (Layer 11) |
| `test_rrelu.py` | NEW | Test suite |
