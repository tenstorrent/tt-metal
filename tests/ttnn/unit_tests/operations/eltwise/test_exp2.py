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
# The optimised implementation handles overflow / underflow / NaN explicitly,
# so we verify each special case produces the IEEE-correct value rather than
# relying on randomly sampled inputs to hit these boundaries.
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
        float("nan"),  # exp2(NaN)  = NaN
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

    # Walk each special input twice:
    #   1. Bit-exact checks for NaN / ±inf (ULP is undefined here).
    #   2. Collect finite (input, golden, result) triples and check them with
    #      assert_with_ulp at the end — a principled bound regardless of
    #      magnitude (handles exp2(127) ≈ 1.7e38 the same as exp2(-10)).
    finite_indices = []
    for i, x in enumerate(special_inputs):
        g = golden_flat[i].item()
        r = result[i].item()

        if np.isnan(g):
            assert np.isnan(r), f"exp2({x}): expected NaN, got {r}"
        elif np.isinf(g):
            assert np.isinf(r) and (np.sign(r) == np.sign(g)), f"exp2({x}): expected {g}, got {r}"
        else:
            finite_indices.append(i)

    if finite_indices:
        idx = torch.tensor(finite_indices)
        assert_with_ulp(golden_flat[idx], result[idx], 1)
