# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from numpy import absolute
import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range_dtype


def torch_to_device(pt_tensor: torch.Tensor, ttnn_dtype: ttnn.types, device: ttnn.Device) -> ttnn.Tensor:
    return ttnn.Tensor(pt_tensor, ttnn_dtype).to(ttnn.TILE_LAYOUT).to(device)


def reference_softmax_backward_output(y: torch.Tensor, grad: torch.Tensor, axis: int) -> torch.Tensor:
    dot = (y * grad).sum(dim=axis, keepdim=True)
    return y * (grad - dot)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.float32],
)
@pytest.mark.parametrize(
    "range",
    [20, 200],
)
def test_bw_softmax(input_shapes, dtype, range, device):
    grad_data, grad_tensor = data_gen_with_range_dtype(input_shapes, -range, range, device, ttnn_dtype=dtype)
    in_data, input_tensor = data_gen_with_range_dtype(
        input_shapes, -range, range, device, required_grad=True, ttnn_dtype=dtype
    )

    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    pt_softmax_tensor = torch.softmax(in_data, dim=3, dtype=torch_dtype)
    tt_softmax_tensor = torch_to_device(pt_softmax_tensor, dtype, device)

    tt_output_tensor_on_device_composite = ttnn.softmax_backward(tt_softmax_tensor, grad_tensor, dim=3)
    pt_output_tensor_on_device_composite = ttnn.to_torch(tt_output_tensor_on_device_composite)
    pt_output_tensor_reference = reference_softmax_backward_output(pt_softmax_tensor, grad_data, axis=3)

    relative_tolerance = min(0.03, (1 / range))
    absolute_tolerance = range / 200
    assert torch.allclose(
        pt_output_tensor_on_device_composite,
        pt_output_tensor_reference,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
