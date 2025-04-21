# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import ttnn

from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    compare_equal,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shape_a, input_shape_b",
    [
        (torch.Size([1, 2, 32]), torch.Size([1, 2, 32])),
        (torch.Size([1]), torch.Size([1, 5, 12])),
        (torch.Size([1, 2, 32, 64, 125]), torch.Size([1, 2, 32, 1, 1])),
        (torch.Size([]), torch.Size([])),
        (torch.Size([5]), torch.Size([1])),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (0, 100, 0, 300),
        (1000, 10000, 500, 1000),
    ],
)
def test_binary_add_uint16_bcast(input_shape_a, input_shape_b, low_a, high_a, low_b, high_b, device):
    num_elements = max(int(torch.prod(torch.tensor(input_shape_a)).item()), 1)
    torch_input_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int16)
    torch_input_a = torch_input_a[:num_elements].reshape(input_shape_a).nan_to_num(0.0)

    num_elements = max(int(torch.prod(torch.tensor(input_shape_b)).item()), 1)
    torch_input_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int16)
    torch_input_b = torch_input_b[:num_elements].reshape(input_shape_b).nan_to_num(0.0)

    golden_function = ttnn.get_golden_function(ttnn.add)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_result = ttnn.add(tt_in_a, tt_in_b, use_legacy=False)
    result = ttnn.to_torch(tt_result)
    assert_with_pcc(golden, result, 0.999)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "input_a_val, input_b_val",
    [
        # (756, 100),
        # (32768, 65535),
        (11, 1),
    ],
)
def test_binary_add_fill_val_uint16(input_shapes, input_a_val, input_b_val, device):
    torch_input_a = torch.ones(input_shapes, dtype=torch.int16) * input_a_val
    torch_input_b = torch.ones(input_shapes, dtype=torch.int16) * input_b_val

    golden_function = ttnn.get_golden_function(ttnn.add)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.add(tt_in_a, tt_in_b)
    result = ttnn.to_torch(tt_result)
    torch.set_printoptions(threshold=10000)
    print(golden)
    print(result)

    comp_pass = compare_equal([tt_result], [golden])
    assert comp_pass
