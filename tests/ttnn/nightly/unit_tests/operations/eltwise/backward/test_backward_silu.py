# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_silu(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -1e4, 1e4, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1e4, 1e4, device)

    tt_output_tensor_on_device = ttnn.silu_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.silu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)

    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_silu_opt_tensor(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -1e4, 1e4, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1e4, 1e4, device)

    _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.silu_bw(grad_tensor, input_tensor, input_grad=input_grad)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    golden_function = ttnn.get_golden_function(ttnn.silu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    tt_output_tensor_on_device = [input_grad]
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)

    assert comp_pass


@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.float32,
        ttnn.bfloat16,
    ),
)
def test_bw_silu_saturation_and_infinities(dtype, device):
    """Verify silu_bw does not produce NaN at +/-inf or saturated tails."""
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    input_shapes = torch.Size([1, 1, 32, 32])

    in_data = torch.zeros(input_shapes, dtype=torch_dtype)
    in_data[..., 0, 0] = float("-inf")
    in_data[..., 0, 1] = float("inf")
    in_data[..., 0, 2] = -100.0
    in_data[..., 0, 3] = 100.0
    in_data[..., 0, 4] = 0.0

    grad_data = torch.ones(input_shapes, dtype=torch_dtype)

    input_tensor = ttnn.from_torch(in_data, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = ttnn.to_torch(ttnn.silu_bw(grad_tensor, input_tensor)[0])

    assert not torch.isnan(tt_output[..., 0, :5]).any()
    assert tt_output[..., 0, 0].item() == pytest.approx(0.0)
    assert tt_output[..., 0, 1].item() == pytest.approx(1.0)
    assert tt_output[..., 0, 2].item() == pytest.approx(0.0)
    assert tt_output[..., 0, 3].item() == pytest.approx(1.0)
    assert tt_output[..., 0, 4].item() == pytest.approx(0.5, rel=1e-2)
