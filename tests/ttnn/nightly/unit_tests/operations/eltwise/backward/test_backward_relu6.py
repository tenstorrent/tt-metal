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
def test_bw_relu6(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 110, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -200, 199, device, required_grad=True)

    tt_output_tensor_on_device = ttnn.relu6_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.relu6_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize("dtype", (ttnn.float32, ttnn.bfloat16), ids=["float32", "bfloat16"])
@pytest.mark.parametrize("grad_value", (2.0, float("inf"), float("nan")), ids=["grad_2", "grad_inf", "grad_nan"])
@pytest.mark.parametrize(
    "input_value, expected_grad_multiplier",
    (
        (-5.0, 0.0),
        # -0.0 is out of range, not a special case of zero: the mask is `input > 0`, which -0.0
        # fails like any other non-positive value, so the gradient is zero. The sign of zero never
        # reaches the result either -- device and golden both return +0.0 for it, bit for bit, in
        # float32 and bfloat16 -- so nothing here depends on how -0.0 is signed.
        (-0.0, 0.0),
        (0.0, 0.0),
        (3.0, 1.0),
        (6.0, 0.0),
        (100.0, 0.0),
        (float("inf"), 0.0),
        (float("-inf"), 0.0),
        (float("nan"), 0.0),
    ),
    ids=["below", "at_negative_zero", "at_zero", "in_range", "at_six", "above", "pos_inf", "neg_inf", "nan"],
)
def test_bw_relu6_boundaries(device, dtype, input_value, expected_grad_multiplier, grad_value):
    """relu6_bw is grad inside (0, 6) and zero everywhere else, including for a NaN input.

    NaN is the case that regressed: grad_result was seeded with 6.0f for input > 0 and the later
    branches only overwrite it where a comparison holds. Every comparison is false for NaN, so the
    seed survived and relu6_bw returned 6.0 regardless of grad. The other rows pin the boundaries
    that the seeding happened to get right, so a future rewrite cannot trade one for the other.

    The gradient is swept over a finite value, inf and NaN because outside the range the result is
    a literal zero, not grad multiplied by a zero mask: an arithmetic lowering would give NaN
    there, since both 0 * inf and 0 * NaN are NaN. inf carries that invariant in both dtypes; NaN
    is skipped in range for bfloat16 only.
    """
    if expected_grad_multiplier and grad_value != grad_value and dtype == ttnn.bfloat16:
        # #31406: a bfloat16 NaN does not survive a format conversion on the device and comes back
        # as an infinity. Nothing relu6_bw does; it selects grad through a where and performs no
        # arithmetic on it. Delete this skip when that issue closes.
        pytest.skip("#31406: bfloat16 loses a NaN operand on the device, returning an infinity")

    shape = torch.Size([1, 1, 32, 32])
    in_data = torch.full(shape, input_value, dtype=torch.float32)
    grad_data = torch.full(shape, grad_value, dtype=torch.float32)

    input_tensor = ttnn.from_torch(in_data, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.to_torch(ttnn.relu6_bw(grad_tensor, input_tensor)[0])

    # Out of range the op returns a literal zero, so a NaN gradient does not propagate there.
    expected_value = grad_value if expected_grad_multiplier else 0.0
    if expected_value != expected_value:
        assert torch.isnan(output).all(), (
            f"relu6_bw(grad=NaN, input={input_value}) in {dtype}: " f"expected NaN but got {output[0, 0, 0, 0].item()}"
        )
    else:
        expected = torch.full(shape, expected_value, dtype=output.dtype)
        assert torch.equal(output, expected), (
            f"relu6_bw(grad={grad_value}, input={input_value}) in {dtype}: "
            f"expected {expected_value} but got {output[0, 0, 0, 0].item()}"
        )
