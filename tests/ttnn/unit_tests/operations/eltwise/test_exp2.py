# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import numpy as np
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_allclose


def test_exp2_arange_masking(device):
    # Exp2 Working range - Overflow from 128(inf), Underflow till -127(<0)
    low = -126.0
    high = 127.0

    # Generate all possible bit patterns for bf16
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor_f32 = input_tensor.to(torch.float32)

    # masking to working range
    mask = (input_tensor_f32 >= low) & (input_tensor_f32 <= high)
    input_tensor = input_tensor[mask]

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.exp2)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.exp2(tt_in)
    result = ttnn.to_torch(tt_result)
    assert_with_ulp(golden, result, 1)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 2, 64, 120])),
        (torch.Size([1, 3, 320, 320])),
    ),
)
@pytest.mark.parametrize(
    "low, high",
    [
        (-5, 5),
        (-126, 127),
    ],
)
def test_exp2_ULP(input_shapes, low, high, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.exp2)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.exp2(tt_in)
    result = ttnn.to_torch(tt_result)
    assert_with_ulp(golden, result, 1)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 2, 64, 120])),
        (torch.Size([1, 3, 320, 320])),
    ),
)
@pytest.mark.parametrize(
    "low, high",
    [
        (-127, -126),
    ],
)
def test_exp2_atol(input_shapes, low, high, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.exp2)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.exp2(tt_in)
    result = ttnn.to_torch(tt_result)
    assert_allclose(tt_result, golden, rtol=1e-2, atol=1e-3)


# Targeted edge-case coverage for the optimised exp2 (see #44507).
# The optimised implementation handles overflow / underflow / ±inf explicitly,
# so we verify each special case produces the IEEE-correct value rather than
# relying on randomly sampled inputs to hit these boundaries.
#
# NaN is intentionally excluded: ttnn.bfloat16 host-side packing collapses NaN
# to +inf before the tensor reaches the SFPU (same root cause as the xfail in
# tests/.../test_fmod.py: "NaN is packed as inf for ttnn.bfloat16"), so a
# device-side NaN-propagation test is not expressible at this dtype.
def test_exp2_special_values(device):
    # Tile must be 32x32; pack edge-case scalars then pad the rest of the tile
    # with a value (0.0) whose result is known and stable.
    special_inputs = [
        0.0,  # exp2(0)    = 1.0
        1.0,  # exp2(1)    = 2.0
        -1.0,  # exp2(-1)   = 0.5
        10.0,  # exp2(10)   = 1024.0
        -10.0,  # exp2(-10)  ≈ 0.0009766
        127.0,  # near-max finite output
        -126.0,  # near-min normal output
        float("inf"),  # exp2(+inf) = +inf
        float("-inf"),  # exp2(-inf) = 0.0
        128.0,  # exact overflow boundary
        -127.0,  # exact underflow boundary
    ]

    pad_count = 32 * 32 - len(special_inputs)
    torch_input = torch.tensor(special_inputs + [0.0] * pad_count, dtype=torch.bfloat16).reshape(1, 1, 32, 32)

    golden_function = ttnn.get_golden_function(ttnn.exp2)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.exp2(tt_in)
    result = ttnn.to_torch(tt_result).reshape(-1)

    golden_flat = golden.reshape(-1)

    # Walk each special input three ways:
    #   1. Bit-exact checks for ±inf (ULP is undefined here).
    #   2. Subnormal goldens (|g| < smallest normal bf16 = 2^-126) must flush
    #      to zero — the SFPU stack uses IEEE FTZ semantics on output, so any
    #      subnormal result is rounded to ±0 (matches the kernel's documented
    #      behaviour at x = -127).
    #   3. Remaining finite (input, golden, result) triples are checked with
    #      assert_with_ulp — a principled bound regardless of magnitude.
    SMALLEST_NORMAL_BF16 = 2.0**-126
    finite_indices = []
    for i, x in enumerate(special_inputs):
        g = golden_flat[i].item()
        r = result[i].item()

        if np.isinf(g):
            assert np.isinf(r) and (np.sign(r) == np.sign(g)), f"exp2({x}): expected {g}, got {r}"
        elif 0.0 < abs(g) < SMALLEST_NORMAL_BF16:
            assert r == 0.0, f"exp2({x}): subnormal golden {g} expected to flush to 0, got {r}"
        else:
            finite_indices.append(i)

    if finite_indices:
        idx = torch.tensor(finite_indices)
        assert_with_ulp(golden_flat[idx], result[idx], 1)
