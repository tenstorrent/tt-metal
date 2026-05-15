# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from ttnn.operations.toy_tilize_to_bfp8 import toy_tilize_to_bfp8


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


SHAPES = [
    pytest.param([32, 32], id="1tile"),
    pytest.param([32, 128], id="1x4tiles"),
    pytest.param([64, 64], id="2x2tiles"),
    pytest.param([128, 128], id="4x4tiles"),
    pytest.param([96, 256], id="3x8tiles"),
    pytest.param([32, 1024], id="wide"),
    pytest.param([1024, 32], id="tall"),
]


@pytest.mark.parametrize("shape", SHAPES)
def test_toy_tilize_to_bfp8(device, shape):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = toy_tilize_to_bfp8(tt_input)

    assert tt_output.dtype == ttnn.bfloat8_b
    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert list(tt_output.shape) == shape

    torch_output = ttnn.to_torch(tt_output)

    correlation = pcc(torch_output, torch_input)
    assert correlation > 0.999, f"PCC too low: {correlation:.6f}"

    max_abs = torch_input.float().abs().max().item()
    atol = max_abs / 64.0
    diff = (torch_output.float() - torch_input.float()).abs().max().item()
    assert diff < atol, f"max abs diff {diff:.4f} exceeds atol {atol:.4f}"
