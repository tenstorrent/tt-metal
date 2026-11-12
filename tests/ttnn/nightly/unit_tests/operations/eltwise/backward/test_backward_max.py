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
def test_bw_max(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, -1, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, 1, 2, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, -1, device, True)

    tt_output_tensor_on_device = ttnn.max_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.max_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


# max_bw used to blend its gradients with 0/1 mask multiplies, and in float32 0 * inf and 0 * nan are
# NaN, so a non-finite grad poisoned both outputs. torch selects: grad to the winner, an exact 0 to the loser.
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("grad_value", [float("inf"), float("-inf"), float("nan")], ids=["pos_inf", "neg_inf", "nan"])
def test_bw_max_non_finite_grad(device, dtype, grad_value):
    if grad_value != grad_value and dtype == ttnn.bfloat16:
        pytest.skip(
            "bfloat16 loses a NaN operand on the device, returning an infinity: "
            "https://github.com/tenstorrent/tt-metal/issues/31406"
        )
    shape = torch.Size([1, 1, 32, 32])

    def to_device(value):
        return ttnn.from_torch(torch.full(shape, value), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    grad_input, grad_other = (
        ttnn.to_torch(t).float() for t in ttnn.max_bw(to_device(grad_value), to_device(2.0), to_device(1.0))
    )

    if grad_value != grad_value:
        assert torch.isnan(grad_input).all(), f"winner gradient should be NaN, got {grad_input[0, 0, 0, 0].item()}"
    else:
        assert torch.equal(grad_input, torch.full(shape, grad_value)), f"winner got {grad_input[0, 0, 0, 0].item()}"
    assert torch.equal(
        grad_other, torch.zeros(shape)
    ), f"loser gradient should be 0, got {grad_other[0, 0, 0, 0].item()}"
