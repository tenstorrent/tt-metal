# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 32),
        (8, 1),
        (32, 1),
    ],
    ids=["1d", "2d", "l2d"],
)
@pytest.mark.parametrize(
    "output_shape",
    [
        (-1, 32),
        (32, -1, 32),
        (32, 32, -1, 32),
        (32, 32, 32, -1, 32),
        (4, 4, 4096, -1, 32),
    ],
    ids=["2d", "3d", "4d", "5d", "random_large_2d"],
)
def test_expand(input_shape, output_shape, device, use_program_cache):
    torch.manual_seed(2024)
    input = torch.rand(input_shape, dtype=torch.float32)
    ref_output = input.expand(output_shape)

    dev_input = ttnn.from_torch(input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # We also test the callback for good measure.
    for _ in range(3):
        dev_output = ttnn.expand(dev_input, output_shape)

    output = ttnn.to_torch(dev_output)
    assert torch.allclose(output, ref_output)
