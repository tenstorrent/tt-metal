# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch


@pytest.mark.parametrize(
    "shape",
    [
        (30, 30),
        (3, 30, 30),
        (2, 3, 30, 30),
        (2, 2, 3, 30, 30),
        (2, 2, 2, 3, 30, 30),
        (2, 2, 2, 2, 3, 30, 30),
        (2, 2, 2, 2, 2, 3, 30, 30),
    ],
)
def test_tensor_ranks(shape, device):
    torch_input_tensor = torch.randint(low=0, high=100, size=shape).to(torch.bfloat16)

    tt_tensor = ttnn.Tensor(torch_input_tensor, ttnn.bfloat16)
    tt_tensor = tt_tensor.pad_to_tile(0.0)
    tt_tensor = tt_tensor.to(ttnn.TILE_LAYOUT)
    tt_tensor = tt_tensor.to(device)
    tt_tensor = tt_tensor.cpu()
    tt_tensor = tt_tensor.to(ttnn.ROW_MAJOR_LAYOUT)
    tt_tensor = tt_tensor.unpad_from_tile(shape)

    assert tt_tensor.shape.rank == len(shape)

    torch_output_tensor = tt_tensor.to_torch()

    assert torch.allclose(torch_input_tensor, torch_output_tensor)
