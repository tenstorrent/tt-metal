# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch", [1, 5])
@pytest.mark.parametrize("sequence", [1, 2])
@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [64, 32])
def test_concatenate_heads(device, batch, sequence, height, width):
    torch_input_tensor = torch.rand((batch, sequence, height, width), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.transformer.concatenate_heads)
    torch_output_tensor = golden_function(torch_input_tensor)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.transformer.concatenate_heads(input_tensor)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
