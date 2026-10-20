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
@pytest.mark.parametrize(
    "beta",
    [0.5, -3, 1, 4, 0],
)
@pytest.mark.parametrize(
    "threshold",
    [-20, -10, 10, 20, 5, 0],
)
def test_bw_softplus(input_shapes, beta, threshold, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.softplus_bw(grad_tensor, input_tensor, beta=beta, threshold=threshold)

    golden_function = ttnn.get_golden_function(ttnn.softplus_bw)
    golden_tensor = golden_function(grad_data, in_data, beta, threshold)

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
def test_bw_default_softplus(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.softplus_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.softplus_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize("beta", [1.0, 2.0])
@pytest.mark.parametrize("threshold", [100, 120])
def test_bw_softplus_saturating_tail(beta, threshold, device):
    """beta * input in the saturating tail but still below threshold takes the sigmoid branch.

    The gradient there is grad * sigmoid(beta * input), which tends to grad. The reference is
    computed analytically in float64 rather than through the torch golden, because torch's own
    float32 softplus backward overflows in this range and returns NaN, so it cannot score it.
    """
    # all below threshold, so the branch under test is the sigmoid one, and all far enough out
    # that sigmoid(beta * input) is 1.0 to well within bfloat16
    values = [85.0, 88.0, 90.0, 95.0, 99.0]
    xs = [v / beta for v in values]

    in_data = torch.tensor(xs, dtype=torch.float32).repeat(1024 // len(xs) + 1)[:1024].reshape(1, 1, 32, 32)
    grad_data = torch.full((1, 1, 32, 32), 2.0, dtype=torch.float32)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out = ttnn.softplus_bw(grad_tensor, input_tensor, beta=beta, threshold=threshold)
    got = ttnn.to_torch(tt_out[0]).to(torch.float64)

    expected = grad_data.to(torch.float64) * torch.sigmoid(beta * in_data.to(torch.float64))

    # the failure this pins returned exactly 0.0 here, from exp overflowing to inf and
    # inf * (1 / inf) collapsing, so assert the gradient survived at all before comparing
    assert not torch.any(got == 0.0), "gradient vanished in the saturating tail"
    assert torch.allclose(got, expected, rtol=1e-2, atol=1e-2)
