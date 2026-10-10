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
def test_bw_atanh(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, required_grad=True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.atanh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.atanh_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


# A zero gradient was replaced by NaN (inf on device) at every input, not only at the singular
# points |x| == 1, so masked positions poisoned the whole backward graph (#57356). The singular
# points themselves are not asserted: the 0/0 NaN written there reads back as inf in bfloat16.
def test_bw_atanh_zero_grad(device):
    x = torch.tensor([0.5, -0.5, 0.0, 0.9, -0.99, 1.0, -1.0, 3.0]).repeat(128).reshape(1, 1, 32, 32)
    g = torch.zeros_like(x)
    to_tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    (got,) = ttnn.atanh_bw(to_tt(g), to_tt(x))
    got = ttnn.to_torch(got).float()
    regular = x.abs() != 1
    assert torch.equal(got[regular], torch.zeros_like(got[regular])), got[regular].unique()
