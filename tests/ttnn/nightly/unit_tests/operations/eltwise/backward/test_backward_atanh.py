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


def test_bw_atanh_negative_grad_overflow(device):
    # Verifies that negative gradient on |x| > 1 preserves +inf on overflow (#54695)
    grad_data = torch.tensor([-1e38], dtype=torch.bfloat16).reshape(1, 1, 1, 1)
    in_data = torch.tensor([1.0078125], dtype=torch.bfloat16).reshape(1, 1, 1, 1)

    golden_function = ttnn.get_golden_function(ttnn.atanh_bw)
    golden_tensor = golden_function(grad_data, in_data)

    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    tt_out = ttnn.atanh_bw(grad_tensor, input_tensor)
    res = ttnn.to_torch(tt_out[0])
    assert torch.equal(res, golden_tensor[0])
