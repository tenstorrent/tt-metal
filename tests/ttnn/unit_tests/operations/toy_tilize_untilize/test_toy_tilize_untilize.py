# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from ttnn.operations.toy_tilize_untilize import toy_tilize_untilize


def pcc(a, b):
    """Pearson correlation coefficient between two tensors."""
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# Heights: 1, small prime, non-aligned <32, aligned, prime >32, non-aligned >32, aligned >64, prime >64
HEIGHTS = [1, 7, 16, 32, 37, 48, 64, 97]

# Widths: 1-tile minimum (16 bf16 = 32B DRAM align), small, prime, aligned, non-aligned,
#         multi-tile aligned, prime >tile, 4-tile aligned, >256 (exceeds DEST), large prime
WIDTHS = [16, 17, 32, 48, 64, 67, 96, 128, 288, 331]

DTYPES = [
    pytest.param(ttnn.bfloat16, id="bf16"),
    pytest.param(ttnn.float32, id="fp32"),
]

GRANULARITIES = [
    pytest.param(False, id="tile"),
    pytest.param(True, id="row"),
]

# Build cross-product shape list with readable IDs
SHAPES = [pytest.param((h, w), id=f"H{h}_W{w}") for h in HEIGHTS for w in WIDTHS]


@pytest.mark.parametrize("use_row_granularity", GRANULARITIES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_toy_tilize_untilize(device, shape, dtype, use_row_granularity):
    """Identity test: tilize then untilize should return original data."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = toy_tilize_untilize(ttnn_input, use_row_granularity=use_row_granularity)

    assert list(ttnn_output.shape) == list(shape)
    torch_output = ttnn.to_torch(ttnn_output)

    if dtype == ttnn.bfloat16:
        # bf16 tilize+untilize is purely data reordering — must be exact
        assert torch.equal(
            torch_output, torch_input
        ), f"bf16 mismatch. Max diff: {(torch_output - torch_input).abs().max()}"
    else:
        # fp32 tilize+untilize with fp32 dest accumulation — should be near-lossless
        correlation = pcc(torch_output, torch_input)
        assert correlation > 0.999, f"fp32 PCC too low: {correlation:.6f}"
