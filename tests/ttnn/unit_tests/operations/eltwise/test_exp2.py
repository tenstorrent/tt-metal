# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import matplotlib.pyplot as plt
import os
from tests.ttnn.utils_for_testing import assert_with_ulp


def compare_tensors(input_tensor, calculated_tensor, expected_tensor):
    mismatch_indices = torch.nonzero(calculated_tensor != expected_tensor)
    for idx in mismatch_indices:
        idx_tuple = tuple(idx.tolist())
        print(f"Input tensor value: {input_tensor[idx_tuple]}")
        print(f"  Calculated tensor value: {calculated_tensor[idx_tuple]}")
        print(f"  Expected tensor value: {expected_tensor[idx_tuple]}")
        print("=" * 50)


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
        (-100, 100),
        (-126, 126),
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
    print(golden, result)
    assert_with_ulp(golden, result, 0)


@pytest.mark.parametrize(
    "low, high, step",
    [
        (-127.0, -120.1, 0.1),
        (-120.0, -110.1, 0.1),
        (-110.0, -100.1, 0.1),
        (-100.0, -90.1, 0.1),
        (-90.0, -80.1, 0.1),
        (-80.0, -70.1, 0.1),
        (-70.0, -60.1, 0.1),
        (-60.0, -50.1, 0.1),
        (-50.0, -40.1, 0.1),
        (-40.0, -30.1, 0.1),
        (-30.0, -20.1, 0.1),
        (-20.0, -10.1, 0.1),
        (-10.0, 0.1, 0.1),
        (0.0, 10.1, 0.1),
        (10.0, 20.1, 0.1),
        (20.0, 30.1, 0.1),
        (30.0, 40.1, 0.1),
        (40.0, 50.1, 0.1),
        (50.0, 60.1, 0.1),
        (60.0, 70.1, 0.1),
        (70.0, 80.1, 0.1),
        (80.0, 90.1, 0.1),
        (90.0, 100.1, 0.1),
        (100.0, 110.1, 0.1),
        (110.0, 120.1, 0.1),
        (120.0, 127.1, 0.1),
    ],
)
def test_exp2(low, high, step, device):
    ttnn.set_printoptions(profile="full")

    input_tensor = torch.arange(low, high, step, dtype=torch.bfloat16)  # to detect fractional differences

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
    assert_with_ulp(golden, result, 0)
