# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import os


@pytest.mark.parametrize(
    "shape",
    [
        (2**6, 2**13, 2**13 + 32),  # 8Gb + 16Mb
        (96 * 96, 1, 32 * 228),  # 4Gb + 8Mb after padding
    ],
)
@pytest.mark.slow
def test_large_tensor_creation_sd(device, shape):
    slow_dispatch = os.environ.get("TT_METAL_SLOW_DISPATCH_MODE")
    shape = (2**6, 2**13, 2**13 + 32)
    if slow_dispatch is None or slow_dispatch == "0":
        pytest.skip("Requires slow dispatch, skipping test")

    torch_input = torch.full(shape, 1).bfloat16()
    torch_output = torch_input

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    output_tensor = ttnn.from_device(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor).bfloat16()

    assert torch_output.shape == output_tensor.shape
    assert torch.all(torch_output == output_tensor)
