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
def test_bw_min(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, -1, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, 1, 2, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, -1, device, True)

    tt_output_tensor_on_device = ttnn.min_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.min_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)
    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


# min_bw used to blend its gradients with 0/1 mask multiplies, and in float32 0 * inf and 0 * nan are
# NaN, so a non-finite grad poisoned both outputs. torch selects: grad to the winner, an exact 0 to the loser.
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("grad_value", [float("inf"), float("-inf"), float("nan")], ids=["pos_inf", "neg_inf", "nan"])
def test_bw_min_non_finite_grad(device, dtype, grad_value):
    if grad_value != grad_value and dtype == ttnn.bfloat16:
        pytest.skip(
            "bfloat16 loses a NaN operand on the device, returning an infinity: "
            "https://github.com/tenstorrent/tt-metal/issues/31406"
        )
    shape = torch.Size([1, 1, 32, 32])

    def to_device(value):
        return ttnn.from_torch(torch.full(shape, value), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    grad_input, grad_other = (
        ttnn.to_torch(t).float() for t in ttnn.min_bw(to_device(grad_value), to_device(1.0), to_device(2.0))
    )

    if grad_value != grad_value:
        assert torch.isnan(grad_input).all(), f"winner gradient should be NaN, got {grad_input[0, 0, 0, 0].item()}"
    else:
        assert torch.equal(grad_input, torch.full(shape, grad_value)), f"winner got {grad_input[0, 0, 0, 0].item()}"
    assert torch.equal(
        grad_other, torch.zeros(shape)
    ), f"loser gradient should be 0, got {grad_other[0, 0, 0, 0].item()}"


# A tie used to be detected through other - input, which is NaN for two equal infinities, so both gradients were 0.
# torch splits the gradient in half at a tie, infinities included.
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["float32", "bfloat16"])
def test_bw_min_ties_match_torch(device, dtype):
    inf = float("inf")
    pairs = [(inf, inf), (-inf, -inf), (2.0, 2.0), (inf, -inf), (-inf, inf), (inf, 3.0), (-inf, 3.0), (3.0, -inf)]
    shape = torch.Size([1, 1, 32, 32])
    a = torch.tensor([p[0] for p in pairs] * 128).reshape(shape)
    b = torch.tensor([p[1] for p in pairs] * 128).reshape(shape)
    grad = torch.full(shape, 2.0)

    a_ref, b_ref = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
    torch.minimum(a_ref, b_ref).backward(grad)

    def to_device(t):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    grad_a, grad_b = (ttnn.to_torch(t).float() for t in ttnn.min_bw(to_device(grad), to_device(a), to_device(b)))

    assert torch.equal(grad_a, a_ref.grad), f"input gradient differs from torch: {grad_a.flatten()[:8]}"
    assert torch.equal(grad_b, b_ref.grad), f"other gradient differs from torch: {grad_b.flatten()[:8]}"
