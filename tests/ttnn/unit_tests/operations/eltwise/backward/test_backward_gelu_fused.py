# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc

INPUT_SHAPES = (
    (torch.Size([32])),
    (torch.Size([25, 34])),  # not aligned by tile size
    (torch.Size([32, 32])),
    (torch.Size([1, 32, 32])),
    (torch.Size([1, 1, 32, 32])),
    (torch.Size([1, 1, 320, 384])),
    (torch.Size([1, 3, 320, 384])),
    (torch.Size([1, 3, 323, 389])),  # not aligned by tile size
)


def gen_data(input_shapes, low, high, device, required_grad=False, is_row_major=False, seed=213919):
    assert high > low, "Incorrect range provided"
    torch.manual_seed(seed)
    pt_tensor = torch.rand(input_shapes, requires_grad=required_grad).bfloat16() * (high - low) + low
    if is_row_major:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16)
        tt_tensor = ttnn.to_layout(tt_tensor, layout=ttnn.ROW_MAJOR_LAYOUT).to(device)
    else:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16)
        tt_tensor = ttnn.to_layout(tt_tensor, layout=ttnn.TILE_LAYOUT).to(device)

    return pt_tensor, tt_tensor


@pytest.mark.parametrize(
    "input_shapes",
    INPUT_SHAPES,
)
@pytest.mark.parametrize(
    "approximate",
    (
        "none",
        "tanh",
    ),
)
def test_bw_gelu(input_shapes, approximate, device):
    in_data, input_tensor = gen_data(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = gen_data(input_shapes, -5, 5, device)

    tt_output_tensor_on_device = ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate=approximate)

    golden_function = ttnn.get_golden_function(ttnn.experimental.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc([tt_output_tensor_on_device], golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    INPUT_SHAPES,
)
def test_bw_gelu_default(input_shapes, device):
    in_data, input_tensor = gen_data(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = gen_data(input_shapes, -5, 5, device)

    tt_output_tensor_on_device = ttnn.experimental.gelu_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.experimental.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc([tt_output_tensor_on_device], golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    INPUT_SHAPES,
)
@pytest.mark.parametrize(
    "approximate",
    (
        "none",
        "tanh",
    ),
)
def test_bw_gelu_opt_output(input_shapes, approximate, device):
    in_data, input_tensor = gen_data(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = gen_data(input_shapes, -5, 5, device)
    input_grad = torch.zeros(input_shapes, dtype=torch.bfloat16)
    input_grad = ttnn.from_torch(
        input_grad, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    cq_id = 0
    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate=approximate, input_grad=input_grad, queue_id=cq_id)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    tt_output_tensor_on_device = [input_grad]

    golden_function = ttnn.get_golden_function(ttnn.experimental.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    INPUT_SHAPES,
)
def test_bw_gelu_default_opt_output(input_shapes, device):
    in_data, input_tensor = gen_data(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = gen_data(input_shapes, -5, 5, device)
    input_grad = torch.zeros(input_shapes, dtype=torch.bfloat16)
    input_grad = ttnn.from_torch(
        input_grad, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    cq_id = 0
    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.experimental.gelu_bw(grad_tensor, input_tensor, input_grad=input_grad, queue_id=cq_id)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    tt_output_tensor_on_device = [input_grad]

    golden_function = ttnn.get_golden_function(ttnn.experimental.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
