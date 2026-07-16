# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc, data_gen_with_range


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_gelu_default(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)

    tt_output_tensor_on_device = ttnn.gelu_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    assert torch.allclose(golden_tensor[0].to(torch.bfloat16), ttnn.to_torch(tt_output_tensor_on_device[0]), atol=0.01)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "approximate, atol_value",
    (
        ("none", 0.01),
        ("tanh", 0.01),
    ),
)
def test_bw_gelu_opt_output(input_shapes, approximate, atol_value, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    input_grad = torch.zeros(input_shapes, dtype=torch.bfloat16)
    input_grad = ttnn.from_torch(
        input_grad, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate, input_grad=input_grad)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    tt_output_tensor_on_device = [input_grad]

    golden_function = ttnn.get_golden_function(ttnn.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    assert torch.allclose(
        golden_tensor[0].to(torch.bfloat16), ttnn.to_torch(tt_output_tensor_on_device[0]), atol=atol_value
    )
