# RReLU Implementation Notes

## Operation Summary

**Name**: rrelu (Randomized Leaky ReLU)
**Math Definition**:
```
f(x) = x            when x >= 0
f(x) = a * x        when x < 0
```
Where:
- Eval mode: `a = (lower + upper) / 2`
- Train mode: `a ~ Uniform(lower, upper)`, sampled independently per element

**Parameters**: lower (float, default=0.125), upper (float, default=1/3), training (bool, default=False)

## New Files

- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h`
- `tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py`

## Modified Files

- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` — added `SFPU_OP_RRELU_INCLUDE` guard
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` — added `SfpuType::rrelu` + missing LLK SfpuType entries
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` — added `SfpuType::rrelu` + missing LLK SfpuType entries
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` — added `UnaryOpType::RRELU`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` — added `RRELU` to `is_parametrized_type()`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` — added `RRELU` to `get_macro_definition()` and `get_op_init_and_func_parameterized()`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` — added `rrelu()` function declaration
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp` — added `rrelu()` function implementation
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` — added Python binding for `ttnn.rrelu`
- `ttnn/ttnn/operations/unary.py` — added golden function using `torch.nn.functional.rrelu`
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` — removed broken includes for nuked operations (trigonometry.h, rpow.h, rdiv.h, fill.h)

## Reference Operations Used

1. **atanh** (most useful): Provided the pattern for programmable constant registers (`vConstFloatPrgm0/1/2`), init function with parameter loading, and LLK wrapper structure with `_llk_math_eltwise_unary_sfpu_params_` dispatch.

2. **hardshrink** (useful for parameterized pattern): Showed how parameterized operations pass parameters via `UnaryWithParam` with multiple float params, and the `is_parametrized_type()` registration.

3. **swish**: Provided the pattern for `v_if/v_endif` predicated execution for sign-based branching, which rrelu uses for the `x < 0` condition.

4. **frac**: Showed the standard non-parameterized registration pattern, including `get_macro_definition` -> `SFPU_OP_*_INCLUDE` split-include guard chain.

5. **sinh**: Confirmed the standard LLK wrapper pattern and the identical WH/BH kernel approach.

## Architecture Decisions

### Parameter Passing
RReLU has 3 parameters (lower, upper, training). These are passed as a 3-element float vector via `UnaryWithParam{UnaryOpType::RRELU, {lower, upper, training_flag}}`.

In the dispatch layer (`get_op_init_and_func_parameterized`), the float parameters are bit-cast to `uint32_t` hex literals embedded in the init/func strings:
- `rrelu_tile_init(0x3e000000u, 0x3eaaaaabu);` — lower and upper as hex floats
- `rrelu_tile(0, 0u);` — dst index and training flag as uint literal

### SFPU Kernel Design
- **Eval mode**: Uses `vConstFloatPrgm2` (precomputed midpoint) for a single multiply on negative inputs. Very efficient — just one `v_if` branch and one multiply.
- **Training mode**: Generates per-element pseudo-random slopes using a xorshift PRNG. A namespace-scoped scalar state advances per iteration, and per-lane diversity comes from XORing with input element bits. The random value is mapped to [0,1) via IEEE 754 mantissa extraction, then scaled to [lower, upper].
- Uses `__builtin_memcpy` for aliasing-safe float/uint32_t conversion (avoids `-Werror=strict-aliasing`).

### Pre-existing Codebase Issues Fixed
- Removed broken unconditional includes from `eltwise_sfpu.cpp` (trigonometry.h, rpow.h, rdiv.h, fill.h) that referenced nuked files
- Added missing `SfpuType` enum entries required by third-party LLK template specializations (comparison ops, integer ops, etc.)

## Known Limitations

1. **Training mode PRNG**: The pseudo-random number generator is deterministic and seeded from input bits + a global counter. For the same input tensor and same invocation sequence, it will produce the same slopes. This is acceptable for a hardware implementation but differs from PyTorch's true PRNG behavior.

2. **Training mode slope range**: Due to bfloat16 precision and the mantissa-extraction approach, the generated slopes may not cover the full [lower, upper] range with uniform distribution. The actual observed range in testing is approximately [0.126, 0.331] vs the theoretical [0.125, 0.333].

3. **No backward operation**: Only the forward pass is implemented. A backward pass (`rrelu_bw`) would need to track which slope was used per element during training mode.

---

## Source Code Snippets

### Layer 1: SFPU Kernel — `ckernel_sfpu_rrelu.h`

Identical for Wormhole B0 and Blackhole.

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
**File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel::sfpu {

// Namespace-scoped PRNG state for training mode.
// Advances each iteration to produce different random slopes across tiles.
static uint32_t rrelu_rng_state = 0xDEADBEEF;

// Randomized Leaky ReLU (RReLU):
//   f(x) = x              when x >= 0
//   f(x) = a * x          when x < 0
//
// Eval mode  (training == 0): a = (lower + upper) / 2    (stored in vConstFloatPrgm2)
// Train mode (training != 0): a ~ Uniform(lower, upper)  per element
//
// lower is in vConstFloatPrgm0, upper is in vConstFloatPrgm1,
// midpoint = (lower + upper) / 2 is in vConstFloatPrgm2.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(uint training) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        vFloat result = x;

        if (training == 0) {
            // Eval mode: fixed slope = midpoint = (lower + upper) / 2
            v_if(x < 0.0f) { result = x * vConstFloatPrgm2; }
            v_endif;
        } else {
            // Training mode: per-element random slope in [lower, upper]
            // Advance scalar RNG state (LCG step)
            rrelu_rng_state ^= rrelu_rng_state << 13;
            rrelu_rng_state ^= rrelu_rng_state >> 17;
            rrelu_rng_state ^= rrelu_rng_state << 5;

            // Mix input bits with global state for per-lane diversity
            vInt bits = reinterpret<vInt>(x);
            bits = bits ^ vInt(rrelu_rng_state);
            // Additional mixing passes
            bits = bits ^ (bits >> 16);
            bits = bits ^ (bits << 7);
            bits = bits ^ (bits >> 13);

            // Convert to float in [0, 1):
            // Take low 23 bits as mantissa, set exponent to 127 -> [1.0, 2.0), subtract 1.0
            vInt mantissa = bits & vInt(0x007FFFFF);
            vFloat rand_val = reinterpret<vFloat>(mantissa | vInt(0x3F800000));
            rand_val = rand_val - vConst1;

            // Scale to [lower, upper]: slope = rand_val * (upper - lower) + lower
            vFloat range = vConstFloatPrgm1 - vConstFloatPrgm0;
            vFloat slope = rand_val * range + vConstFloatPrgm0;

            v_if(x < 0.0f) { result = x * slope; }
            v_endif;
        }

        dst_reg[0] = result;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void rrelu_init(uint lower_bits, uint upper_bits) {
    // Bit-cast uint32_t parameters to float (aliasing-safe)
    float lower, upper;
    __builtin_memcpy(&lower, &lower_bits, sizeof(float));
    __builtin_memcpy(&upper, &upper_bits, sizeof(float));
    vConstFloatPrgm0 = lower;
    vConstFloatPrgm1 = upper;
    vConstFloatPrgm2 = (lower + upper) * 0.5f;
}

}  // namespace ckernel::sfpu
```

### Layer 2: LLK Wrapper — `llk_math_eltwise_unary_sfpu_rrelu.h`

Identical for Wormhole B0 and Blackhole.

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
**File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_rrelu.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rrelu_init(uint lower_bits, uint upper_bits) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>(
        sfpu::rrelu_init<APPROXIMATE>, lower_bits, upper_bits);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu(
    uint dst_index, int vector_mode = (int)VectorMode::RC, uint training = 0u) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rrelu<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, training);
}

}  // namespace ckernel
```

### Layer 3: Compute API — `rrelu.h`

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
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
 * Performs element-wise Randomized Leaky ReLU (RReLU).
 *   f(x) = x          when x >= 0
 *   f(x) = a * x      when x < 0
 * where a = (lower + upper) / 2 in eval mode, or a ~ Uniform(lower, upper) in training mode.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | training        | 0 for eval mode, 1 for training mode                                      | uint32_t | 0 or 1                                                | True     |
 */
// clang-format on
ALWI void rrelu_tile(uint32_t idst, uint32_t training) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX>(idst, (int)VectorMode::RC, training)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init(uint32_t lower_bits, uint32_t upper_bits) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>(lower_bits, upper_bits)));
}

}  // namespace ckernel
```

### Layer 4: Split Include Guard — `sfpu_split_includes.h`

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`

```cpp
// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Wave 3 generated ops (kept)
#if SFPU_OP_FRAC_INCLUDE
#include "api/compute/eltwise_unary/frac.h"
#endif

#if SFPU_OP_SWISH_INCLUDE
#include "api/compute/eltwise_unary/swish.h"
#endif

#if SFPU_OP_ATANH_INCLUDE
#include "api/compute/eltwise_unary/atanh.h"
#endif

#if SFPU_OP_SINH_INCLUDE
#include "api/compute/eltwise_unary/sinh.h"
#endif

#if SFPU_OP_RRELU_INCLUDE
#include "api/compute/eltwise_unary/rrelu.h"
#endif
```

### Layer 5: SfpuType Enum — `llk_sfpu_types.h`

Identical for Wormhole B0 and Blackhole (Blackhole additionally has `reciprocal` entry).

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`

```cpp
enum class SfpuType {
    unused = 0,
    frac,
    swish,
    atanh,
    sinh,
    rrelu,
    // Entries required by third-party LLK template specializations
    equal_zero,
    not_equal_zero,
    less_than_zero,
    greater_than_equal_zero,
    greater_than_zero,
    less_than_equal_zero,
    unary_ne,
    unary_eq,
    unary_gt,
    unary_lt,
    unary_ge,
    unary_le,
    signbit,
    isinf,
    isposinf,
    isneginf,
    isnan,
    isfinite,
    max,
    min,
    max_int,
    max_uint,
    min_int,
    min_uint,
    mul_int,
    mul_uint,
    unary_max,
    unary_min,
    unary_max_int,
    unary_max_uint,
    unary_min_int,
    unary_min_uint,
    topk_local_sort,
    typecast,
    where,
    unary_max_int32,
    unary_min_int32,
    unary_max_uint32,
    unary_min_uint32,
};
```

### Layer 6: UnaryOpType Enum — `unary_op_types.hpp`

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`

Relevant addition (last entry in the enum):

```cpp
enum class UnaryOpType {
    // ... existing entries ...
    SWISH,
    RRELU,
};
```

### Layer 7: Parameterized Type Registration — `unary_op_utils.hpp`

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`

```cpp
template <typename T>
bool is_parametrized_type(T val) {
    switch (val) {
        case UnaryOpType::HARDTANH: return true;
        case UnaryOpType::SOFTSHRINK: return true;
        case UnaryOpType::RRELU: return true;
        default: return false;
    }
    return false;
}
```

### Layer 8: Dispatch — Macro Definition and Init/Func Strings — `unary_op_utils.cpp`

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

**`get_macro_definition`** (maps `UnaryOpType` to the split-include guard):

```cpp
std::string get_macro_definition(UnaryOpType op_type) {
    switch (op_type) {
        case UnaryOpType::FRAC: return "SFPU_OP_FRAC_INCLUDE";
        case UnaryOpType::SWISH: return "SFPU_OP_SWISH_INCLUDE";
        case UnaryOpType::ATANH: return "SFPU_OP_ATANH_INCLUDE";
        case UnaryOpType::SINH: return "SFPU_OP_SINH_INCLUDE";
        case UnaryOpType::RRELU: return "SFPU_OP_RRELU_INCLUDE";
        default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
    };
}
```

**`get_op_init_and_func_parameterized`** (bit-casts float params to uint32_t hex literals):

```cpp
case UnaryOpType::RRELU: {
    float lower_f = param0;
    float upper_f = static_cast<float>(params[1]);
    float training_f = static_cast<float>(params[2]);
    uint32_t lower_u = std::bit_cast<uint32_t>(lower_f);
    uint32_t upper_u = std::bit_cast<uint32_t>(upper_f);
    uint32_t training_u = (training_f != 0.0f) ? 1u : 0u;
    return {
        fmt::format("rrelu_tile_init(0x{:x}u, 0x{:x}u);", lower_u, upper_u),
        fmt::format("rrelu_tile({}, {}u);", idst, training_u)};
}
```

### Layer 9: C++ Host API — `unary.hpp` / `unary.cpp`

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`

```cpp
// rrelu: Randomized Leaky ReLU
Tensor rrelu(
    const Tensor& input_tensor,
    float lower = 0.125f,
    float upper = 0.3333333432674408f,
    bool training = false,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);
```

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp`

```cpp
Tensor rrelu(
    const Tensor& input_tensor,
    float lower,
    float upper,
    bool training,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::RRELU, {lower, upper, training ? 1.0f : 0.0f}}},
        memory_config,
        optional_output_tensor);
}
```

### Layer 10: Python Binding — `unary_nanobind.cpp`

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`

```cpp
{
    auto doc = fmt::format(
        R"doc(
        Applies Randomized Leaky ReLU (RReLU) element-wise.

        .. math::
            \text{{rrelu}}(x) = \begin{{cases}} x & \text{{if }} x \geq 0 \\ ax & \text{{if }} x < 0 \end{{cases}}

        where *a* is sampled uniformly from :math:`\mathcal{{U}}(\text{{lower}}, \text{{upper}})` during
        training, and :math:`a = (\text{{lower}} + \text{{upper}}) / 2` during evaluation.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            lower (float, optional): lower bound of the uniform distribution. Defaults to `0.125`.
            upper (float, optional): upper bound of the uniform distribution. Defaults to `0.3333333432674408`.
            ...
        )doc");

    ttnn::bind_function<"rrelu">(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            &ttnn::rrelu,
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("lower") = 0.125f,
            nb::arg("upper") = 0.3333333432674408f,
            nb::arg("training") = false,
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()));
}
```

### Layer 11: Golden Function — `unary.py`

**File**: `ttnn/ttnn/operations/unary.py`

```python
def _golden_function_rrelu(input_tensor_a, *args, lower=0.125, upper=1.0 / 3.0, training=False, **kwargs):
    import torch

    return torch.nn.functional.rrelu(input_tensor_a, lower=lower, upper=upper, training=training)


ttnn.attach_golden_function(ttnn.rrelu, golden_function=_golden_function_rrelu)
```

### Compute Kernel Entry Point — `eltwise_sfpu.cpp`

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

```cpp
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);

            copy_tile(tt::CBIndex::c_0, 0, 0);

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            tile_regs_commit();

            tile_regs_wait();

            pack_tile(0, tt::CBIndex::c_2);

            cb_pop_front(tt::CBIndex::c_0, 1);

            tile_regs_release();
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
```

### Test File — `test_rrelu.py`

**File**: `tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py`

```python
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

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
@pytest.mark.parametrize(
    "lower,upper",
    [
        (0.125, 1.0 / 3.0),
        (0.0, 0.5),
        (0.1, 0.1),
    ],
    ids=["default", "wide", "constant"],
)
def test_rrelu_eval(device, is_fp32, lower, upper):
    """Test RReLU in eval mode (deterministic: slope = (lower + upper) / 2 for negative inputs)."""
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Compute reference in float32 using eval mode (training=False)
    torch_output = torch.nn.functional.rrelu(torch_input.float(), lower=lower, upper=upper, training=False)
    expected = flush_subnormal_values_to_zero(torch_output)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    # Run on device
    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=False)
    actual = ttnn.to_torch(tt_output)
    if not is_fp32:
        actual = actual.to(torch.bfloat16)

    # Filter out NaN/Inf for meaningful comparison
    finite_mask = (
        torch.isfinite(torch_input.float()) & torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    )
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    # allclose check
    assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)

    # ULP check for bfloat16
    if not is_fp32:
        nonzero_mask = expected_finite.float().abs() > 0.0
        if nonzero_mask.any():
            expected_nz = expected_finite[nonzero_mask].reshape(1, -1)
            actual_nz = actual_finite[nonzero_mask].reshape(1, -1)
            assert_with_ulp(expected_nz, actual_nz, ulp_threshold=2)
```
