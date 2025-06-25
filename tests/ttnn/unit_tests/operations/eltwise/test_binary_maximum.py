# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_equal
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 1, 2])),),
)
@pytest.mark.parametrize(
    "testing_dtype",
    ["bfloat16", "float32"],
)
def test_fmod(input_shapes, testing_dtype, device):
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)
    torch_input_a = torch.ones(input_shapes, dtype=torch_dtype) * 1.0
    torch_input_b = torch.ones(input_shapes, dtype=torch_dtype) * 0.0

    golden_function = ttnn.get_golden_function(ttnn.fmod)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.fmod(tt_in_a, tt_in_b)
    output_tensor = ttnn.to_torch(tt_result)
    print()
    print("dtype : ", testing_dtype)
    print("Expected : ", golden[0, 0, 0, 0])
    print("TTNN.    : ", output_tensor[0, 0, 0, 0])
    assert torch.equal(golden, output_tensor)
