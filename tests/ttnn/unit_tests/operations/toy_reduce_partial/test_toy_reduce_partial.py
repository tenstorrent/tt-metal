# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for toy_reduce_partial — partial scaler pattern for both REDUCE_ROW and REDUCE_COL.

Test shapes exercise non-tile-aligned dimensions where the partial scaler
must correctly zero out padded elements.
"""

import pytest
import torch
import ttnn

from ttnn.operations.toy_reduce_partial import toy_reduce_partial


@pytest.mark.parametrize(
    "shape, dim",
    [
        # REDUCE_ROW (W) — non-aligned W, various partials
        pytest.param((1, 1, 32, 48), -1, id="reduce_w__W=48_partial=16"),
        pytest.param((1, 1, 32, 100), -1, id="reduce_w__W=100_partial=4"),
        pytest.param((1, 1, 32, 37), -1, id="reduce_w__W=37_partial=5"),
        pytest.param((1, 1, 32, 45), -1, id="reduce_w__W=45_partial=13"),
        pytest.param((1, 1, 32, 55), -1, id="reduce_w__W=55_partial=23"),
        pytest.param((1, 1, 32, 59), -1, id="reduce_w__W=59_partial=27"),
        pytest.param((1, 1, 32, 33), -1, id="reduce_w__W=33_partial=1"),
        pytest.param((1, 1, 32, 63), -1, id="reduce_w__W=63_partial=31"),
        pytest.param((1, 1, 64, 80), -1, id="reduce_w__H=64_W=80_partial=16"),
        # REDUCE_ROW (W) — aligned baselines
        pytest.param((1, 1, 32, 64), -1, id="reduce_w__W=64_aligned"),
        pytest.param((1, 1, 32, 32), -1, id="reduce_w__W=32_single_tile"),
        # REDUCE_COL (H) — non-aligned H, various partials
        pytest.param((1, 1, 48, 32), -2, id="reduce_h__H=48_partial=16"),
        pytest.param((1, 1, 100, 32), -2, id="reduce_h__H=100_partial=4"),
        pytest.param((1, 1, 37, 32), -2, id="reduce_h__H=37_partial=5"),
        pytest.param((1, 1, 45, 32), -2, id="reduce_h__H=45_partial=13"),
        pytest.param((1, 1, 55, 32), -2, id="reduce_h__H=55_partial=23"),
        pytest.param((1, 1, 59, 32), -2, id="reduce_h__H=59_partial=27"),
        pytest.param((1, 1, 33, 32), -2, id="reduce_h__H=33_partial=1"),
        pytest.param((1, 1, 63, 32), -2, id="reduce_h__H=63_partial=31"),
        pytest.param((1, 1, 80, 64), -2, id="reduce_h__H=80_W=64_partial=16"),
        # REDUCE_COL (H) — aligned baselines
        pytest.param((1, 1, 64, 32), -2, id="reduce_h__H=64_aligned"),
        pytest.param((1, 1, 32, 32), -2, id="reduce_h__H=32_single_tile"),
    ],
)
def test_toy_reduce_partial(device, shape, dim):
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # PyTorch reference: max over the target dimension
    torch_expected = torch.max(torch_input.float(), dim=dim, keepdim=True).values

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Fill implicit padding with non-zero garbage so the test actually verifies
    # the partial scaler is excluding padded elements (not just relying on zero padding)
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, 99.0)

    ttnn_output = toy_reduce_partial(ttnn_input, dim=dim)
    torch_output = ttnn.to_torch(ttnn_output)

    # Extract only the meaningful slice (first row or first column of the output tile)
    if dim == -1:
        # REDUCE_ROW: result is in column 0 of each output tile
        actual = torch_output[..., :1].float()
        expected = torch_expected.float()
    else:
        # REDUCE_COL: result is in row 0 of each output tile
        actual = torch_output[..., :1, :].float()
        expected = torch_expected.float()

    assert torch.allclose(actual, expected, rtol=0.05, atol=0.5), (
        f"Mismatch for shape={shape}, dim={dim}:\n"
        f"  max abs diff = {(actual - expected).abs().max().item():.4f}\n"
        f"  actual[:4]   = {actual.flatten()[:4].tolist()}\n"
        f"  expected[:4] = {expected.flatten()[:4].tolist()}"
    )
