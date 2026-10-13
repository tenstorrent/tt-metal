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
def test_bw_tanh(input_shapes, device):
    # tt tan supports input range [-1.45, 1.45]
    in_data, input_tensor = data_gen_with_range(input_shapes, -1.45, 1.45, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1e4, 1e4, device)

    tt_output_tensor_on_device = ttnn.tanh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.tanh_bw)
    golden_tensor = golden_function(grad_data, in_data)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, 0.95)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_tanh_with_output(input_shapes, device):
    # tt tan supports input range [-1.45, 1.45]
    in_data, input_tensor = data_gen_with_range(input_shapes, -1.45, 1.45, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1e4, 1e4, device)
    input_grad = None

    _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_output_tensor_on_device = ttnn.tanh_bw(
        grad_tensor,
        input_tensor,
        input_grad=input_grad,
        queue_id=cq_id,
    )

    golden_function = ttnn.get_golden_function(ttnn.tanh_bw)
    golden_tensor = golden_function(grad_data, in_data)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, 0.95)
    assert status


@pytest.mark.parametrize(
    "grad_dtype, input_dtype",
    (
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.float32, ttnn.float32),
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.float32, ttnn.bfloat16),
    ),
)
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_tanh_mixed_operand_dtypes(input_shapes, grad_dtype, input_dtype, device):
    # tanh_bw passes the input's dtype as the output dtype, so its validation accepts a
    # grad_output of any supported dtype. The device program has to hold up its end: the
    # circular buffer formats must follow the buffers bound to them, and float32 DEST values
    # must only be requested when DEST accumulates in float32. Before those fixes the two
    # mixed rows returned ~3.9e6 relative error and inf respectively, silently.
    torch.manual_seed(0)
    # tt tanh supports input range [-1.45, 1.45]
    in_data = torch.rand(input_shapes) * 2.9 - 1.45
    grad_data = torch.rand(input_shapes) * 2e4 - 1e4

    input_tensor = ttnn.from_torch(in_data, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, dtype=grad_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output_tensor_on_device = ttnn.tanh_bw(grad_tensor, input_tensor)

    output = ttnn.to_torch(tt_output_tensor_on_device[0])
    assert output.isfinite().all(), f"tanh_bw returned non-finite values for {grad_dtype} grad, {input_dtype} input"

    # Compared against the operands as the device holds them, so the bar measures the kernel
    # rather than the rounding from_torch already applied.
    golden_function = ttnn.get_golden_function(ttnn.tanh_bw)
    golden_tensor = golden_function(ttnn.to_torch(grad_tensor).float(), ttnn.to_torch(input_tensor).float())

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, 0.999)
    assert status
