# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 32])),
        (torch.Size([1, 1, 1, 64])),
    ),
)
def test_non_zero_indices_ttnn(input_shapes, device):
    torch.manual_seed(0)

    torch_input_tensor = torch.ones(input_shapes)
    torch_input_tensor[..., ::2] = 0

    torch_output_tensor = torch.nonzero(torch_input_tensor, as_tuple=True)
    torch_output_tensor = torch_output_tensor[3].unsqueeze(0).unsqueeze(0).unsqueeze(0)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.nonzero(input_tensor, queue_id=0)

    output_tensor1 = ttnn.to_layout(output_tensor[0], ttnn.ROW_MAJOR_LAYOUT)
    output_tensor1 = ttnn.from_device(output_tensor1)
    output_tensor1 = ttnn.to_torch(output_tensor1)
    no_of_non_zero_indices = output_tensor1[..., 0].item()

    output_tensor2 = ttnn.to_layout(output_tensor[1], ttnn.ROW_MAJOR_LAYOUT)
    output_tensor2 = ttnn.from_device(output_tensor2)
    output_tensor2 = ttnn.to_torch(output_tensor2)
    tt_output_tensor = output_tensor2[:, :, :, :no_of_non_zero_indices]

    assert_equal(torch_output_tensor, tt_output_tensor)
