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
@pytest.mark.parametrize("lambd", [2.2, 0.5, 1.0, 5.5, 9.9])
def test_bw_hardshrink(input_shapes, lambd, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.hardshrink_bw(grad_tensor, input_tensor, lambd=lambd)

    golden_function = ttnn.get_golden_function(ttnn.hardshrink_bw)
    golden_tensor = golden_function(grad_data, in_data, lambd)

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
def test_bw_hardshrink_default(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.hardshrink_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.hardshrink_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize("input_shapes", ((torch.Size([1, 1, 32, 32])),))
@pytest.mark.parametrize("lambd", [0.5, -0.5])
def test_bw_hardshrink_zero_input_negative_lambd(input_shapes, lambd, device):
    """Gradient at x == 0 must follow |x| > lambd, not the forward output.

    ``hardshrink(0.0, lambd)`` is 0.0 for either sign of lambd, so a predicate rebuilt from the
    forward output (``hardshrink(x) == 0``) wrongly zeroes the gradient at x == 0 when lambd < 0,
    where the forward pass actually passes 0.0 through and the gradient must be ``grad``.
    """
    in_data = torch.zeros(input_shapes)
    in_data[0, 0, 0, 0] = 0.0
    in_data[0, 0, 0, 1] = 0.25
    in_data[0, 0, 0, 2] = -0.25
    in_data[0, 0, 0, 3] = 2.0
    grad_data = torch.ones(input_shapes)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out = ttnn.to_torch(ttnn.hardshrink_bw(grad_tensor, input_tensor, lambd=lambd)[0]).float()

    golden_function = ttnn.get_golden_function(ttnn.hardshrink_bw)
    golden = golden_function(grad_data, in_data, lambd)[0].float()

    assert torch.equal(tt_out, golden), f"lambd={lambd} got={tt_out.flatten()[:4]} want={golden.flatten()[:4]}"
    assert tt_out.flatten()[0].item() == (1.0 if lambd < 0 else 0.0)


@pytest.mark.parametrize("input_shapes", ((torch.Size([1, 1, 32, 32])),))
def test_bw_hardshrink_nan_input_zero_grad(input_shapes, device):
    """NaN input must produce a zero gradient, not the incoming gradient.

    ``hardshrink_bw`` expresses its predicate as ``-lambd <= x <= lambd``; every comparison
    against NaN is false, so without an explicit NaN term the input falls through to the
    "pass the gradient" branch. PyTorch evaluates the derivative as ``(x < -lambd) || (x > lambd)``,
    which is also false for NaN, so the gradient there is zero for both dtypes.
    """
    in_data = torch.zeros(input_shapes)
    in_data[0, 0, 0, 0] = float("nan")
    in_data[0, 0, 0, 1] = float("inf")
    in_data[0, 0, 0, 2] = 0.25
    in_data[0, 0, 0, 3] = 2.0
    grad_data = torch.ones(input_shapes)

    for dtype in (ttnn.bfloat16, ttnn.float32):
        input_tensor = ttnn.from_torch(in_data, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        grad_tensor = ttnn.from_torch(grad_data, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        tt_out = ttnn.to_torch(ttnn.hardshrink_bw(grad_tensor, input_tensor, lambd=0.5)[0]).float().flatten()

        assert tt_out[0].item() == 0.0, f"{dtype} NaN gradient: got {tt_out[0].item()}"
        assert tt_out[1].item() == 1.0, f"{dtype} +inf gradient: got {tt_out[1].item()}"
        assert tt_out[2].item() == 0.0, f"{dtype} in-range gradient: got {tt_out[2].item()}"
        assert tt_out[3].item() == 1.0, f"{dtype} out-of-range gradient: got {tt_out[3].item()}"
