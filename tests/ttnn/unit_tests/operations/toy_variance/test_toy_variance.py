# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for toy_variance — streaming-reduce helpers on a wide tensor.

Verifies bf16 per-row population variance against torch.var(unbiased=False)
across a few shapes, including the headline 32 x 64000 case which is too
wide to fit in L1 and so exercises the streaming chunking.
"""

import pytest
import torch
import ttnn

from ttnn.operations.toy_variance import toy_variance


@pytest.mark.parametrize(
    "shape",
    [
        # Tile-aligned W
        pytest.param((1, 1, 32, 256), id="W=256_aligned"),
        pytest.param((1, 1, 32, 1024), id="W=1024_aligned"),
        pytest.param((1, 1, 32, 8192), id="W=8192_aligned"),
        pytest.param((1, 1, 32, 64000), id="W=64000_wide_aligned"),
        # Non-aligned W — exercises the partial-scaler path
        pytest.param((1, 1, 32, 33), id="W=33_partial=1"),
        pytest.param((1, 1, 32, 100), id="W=100_partial=4"),
        pytest.param((1, 1, 32, 257), id="W=257_partial=1"),
        pytest.param((1, 1, 32, 1023), id="W=1023_partial=31"),
        # Non-aligned H — output rows beyond origin_H are garbage and sliced off
        pytest.param((1, 1, 33, 64), id="H=33_W=64"),
        pytest.param((1, 1, 33, 100), id="H=33_W=100_both_partial"),
    ],
)
@pytest.mark.parametrize("std_dev", [False, True], ids=["variance", "std_dev"])
def test_toy_variance(device, shape, std_dev):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 0.5

    if std_dev:
        torch_expected = torch.std(torch_input.float(), dim=-1, keepdim=True, unbiased=False)
    else:
        torch_expected = torch.var(torch_input.float(), dim=-1, keepdim=True, unbiased=False)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Fill implicit tile padding with non-zero garbage. If the partial scaler
    # is doing its job, the contaminated values are zeroed out in the reduce
    # and the result still matches torch. If it's broken, (99 - mean)^2 ≈ 9801
    # would dominate the variance and the comparison would fail by orders of
    # magnitude — this is the actual partial-scaler correctness check.
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, 99.0)

    ttnn_output = toy_variance(ttnn_input, std_dev=std_dev)
    torch_output = ttnn.to_torch(ttnn_output)

    # Result lives in column 0 of each output tile. For non-aligned H, rows
    # beyond origin_H in the output tile are padded garbage — slice them off.
    H = shape[-2]
    actual = torch_output[..., :H, :1].float()
    expected = torch_expected.float()

    W = shape[-1]
    atol = max(0.05, 0.001 * (W / 256))
    rtol = 0.10

    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"Mismatch for shape={shape}, std_dev={std_dev}:\n"
        f"  max abs diff = {(actual - expected).abs().max().item():.6f}\n"
        f"  actual[:4]   = {actual.flatten()[:4].tolist()}\n"
        f"  expected[:4] = {expected.flatten()[:4].tolist()}"
    )
