# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc


def test_clamp_bw_tensor_both_bounds_in_range_gradient(device):
    """Regression test for inverted ge/le in the tensor both-bounds branch of clamp_bw.

    With controlled inputs we can verify analytically:
      input  = [-5, -1, 0, 1, 5] (32-element tile padded)
      min    = [-2, -2, -2, -2, -2]
      max    = [ 2,  2,  2,  2,  2]
      grad   = [ 1,  1,  1,  1,  1]
    Expected: gradient passes through only where min <= input <= max (positions 1,2,3),
    zeros at positions 0 and 4 (out of range). Bug produced all-zero result instead.
    """
    shape = torch.Size([1, 1, 32, 32])
    # Build a flat pattern repeated to fill the tile
    base_input = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0] * (32 * 32 // 5) + [-5.0] * ((32 * 32) % 5))
    base_min = torch.full((32 * 32,), -2.0)
    base_max = torch.full((32 * 32,), 2.0)
    base_grad = torch.ones(32 * 32)

    input_torch = base_input.reshape(shape).bfloat16()
    min_torch = base_min.reshape(shape).bfloat16()
    max_torch = base_max.reshape(shape).bfloat16()
    grad_torch = base_grad.reshape(shape).bfloat16()

    # Golden: gradient only where min <= input <= max
    in_range = (input_torch >= min_torch) & (input_torch <= max_torch)
    expected = grad_torch * in_range.bfloat16()

    # Device tensors
    input_tt = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    min_tt = ttnn.from_torch(min_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    max_tt = ttnn.from_torch(max_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tt = ttnn.from_torch(grad_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    result_tt = ttnn.clamp_bw(grad_tt, input_tt, min_tt, max_tt)
    result_torch = ttnn.to_torch(result_tt[0])

    assert torch.allclose(
        result_torch.float(), expected.float(), atol=1e-2
    ), f"clamp_bw tensor both-bounds: got\n{result_torch}\nexpected\n{expected}"


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "min_val, max_val",
    [
        (-10.0, 10.0),
        (10.0, -10.0),
        (1, -1),
        (0, 0),
        (-1.0, None),
        (None, 1.0),
        (None, None),
        (-0.5, None),
        (None, -0.5),
        (1.0, 0.0),
        (0.0, 1.0),
        ("tensor", None),
        (None, "tensor"),
        ("tensor", "tensor"),
    ],
)
def test_unary_bw_clamp_ttnn(input_shapes, min_val, max_val, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, -1, device)
    if min_val == "tensor":
        min, min_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    elif min_val is None:
        min, min_tensor = None, None
    else:
        min, min_tensor = min_val, min_val

    if max_val == "tensor":
        max, max_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    elif max_val is None:
        max, max_tensor = None, None
    else:
        max, max_tensor = max_val, max_val

    if min is None and max is None:
        pytest.xfail("Only one of 'min' or 'max' can be None. Please provide one value")
    else:
        tt_output_tensor_on_device = ttnn.clamp_bw(grad_tensor, input_tensor, min_tensor, max_tensor)
        golden_function = ttnn.get_golden_function(ttnn.clamp_bw)
        golden_tensor = golden_function(grad_data, in_data, min, max)
        comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
        assert comp_pass
