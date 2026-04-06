# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 1, 32, 32],
        [1, 1, 64, 64],
        [1, 1, 32, 256],
    ],
)
def test_sinh(device, input_shapes):
    torch_input = torch.linspace(-4.0, 4.0, steps=input_shapes[-2] * input_shapes[-1]).reshape(input_shapes).bfloat16()
    torch_expected = torch.sinh(torch_input)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.sinh(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert torch.allclose(
        tt_output_torch, torch_expected, atol=0.2, rtol=0.05
    ), f"Max diff: {(tt_output_torch - torch_expected).abs().max().item()}"
