# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.uint16, ttnn.uint32])
def test_eq(device, h, w, output_dtype):
    torch.manual_seed(0)

    same = 50
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_a[0, 0] = same
    torch_input_tensor_a[0, 1] = same
    torch_input_tensor_a[0, 2] = same

    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b[0, 0] = same
    torch_input_tensor_b[0, 1] = same
    torch_input_tensor_b[0, 2] = same

    golden_function = ttnn.get_golden_function(ttnn.eq)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    output_tensor = ttnn.eq(input_tensor_a, input_tensor_b, dtype=output_dtype)
    assert output_tensor.get_dtype() == output_dtype
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device)) - 1
    output_tensor = ttnn.to_torch(output_tensor)
    assert_equal(torch_output_tensor.float(), output_tensor.float())

    # EQ with a preallocated output tensor
    output_tensor_preallocated_bfloat16 = ttnn.ones(
        [h, w], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    output_tensor_preallocated = output_tensor_preallocated_bfloat16
    if output_dtype != ttnn.bfloat16:
        output_tensor_preallocated = ttnn.typecast(
            output_tensor_preallocated_bfloat16, output_dtype, memory_config=ttnn.L1_MEMORY_CONFIG
        )

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.eq(input_tensor_a, input_tensor_b, dtype=output_dtype, output_tensor=output_tensor_preallocated)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))
    torch_output_tensor_preallocated = ttnn.to_torch(output_tensor_preallocated)
    assert_equal(torch_output_tensor.float(), torch_output_tensor_preallocated.float())
