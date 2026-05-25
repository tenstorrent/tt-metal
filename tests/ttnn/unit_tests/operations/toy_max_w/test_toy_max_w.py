# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for toy_max_w — streaming accumulate_reduce<MAX, REDUCE_ROW> with the
transpose-on-reload workaround.

Verifies bf16 per-row max against torch.amax across shapes that exercise
both single-block (NUM_BLOCKS=1, no reload) and multi-block (NUM_BLOCKS>1,
reload + transpose_wh_dest) paths. Capped at ~10 W-tiles for fast sim runs.
"""

import pytest
import torch
import ttnn

from ttnn.operations.toy_max_w import toy_max_w


@pytest.mark.parametrize(
    "shape",
    [
        # Tile-aligned W, single block (NUM_BLOCKS=1, no reload exercised)
        pytest.param((1, 1, 32, 32), id="W=32_single_tile"),
        pytest.param((1, 1, 32, 64), id="W=64_two_tiles"),
        # Tile-aligned W, multi-block (NUM_BLOCKS > 1 → reload + transpose path)
        pytest.param((1, 1, 32, 128), id="W=128_4tiles"),
        pytest.param((1, 1, 32, 320), id="W=320_10tiles"),
        # Non-aligned W — exercises the partial-scaler path on the last block
        pytest.param((1, 1, 32, 33), id="W=33_partial=1"),
        pytest.param((1, 1, 32, 100), id="W=100_partial=4"),
        pytest.param((1, 1, 32, 257), id="W=257_partial=1"),
        # Non-aligned H — output rows beyond origin_H are garbage and sliced off
        pytest.param((1, 1, 33, 64), id="H=33_W=64"),
        pytest.param((1, 1, 33, 100), id="H=33_W=100_both_partial"),
    ],
)
def test_toy_max_w(device, shape):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    torch_expected = torch.amax(torch_input.float(), dim=-1, keepdim=True)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Fill implicit tile padding with a large positive value. If the partial
    # scaler is doing its job for MAX (padded positions filled with -inf),
    # this large value is ignored. If broken, 99.0 would win every row and
    # the comparison would fail loudly — this is the actual partial-scaler
    # correctness check.
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, 99.0)

    ttnn_output = toy_max_w(ttnn_input)
    torch_output = ttnn.to_torch(ttnn_output)

    H = shape[-2]
    actual = torch_output[..., :H, :1].float()
    expected = torch_expected.float()

    assert torch.allclose(actual, expected, rtol=0.01, atol=0.01), (
        f"Mismatch for shape={shape}:\n"
        f"  max abs diff = {(actual - expected).abs().max().item():.6f}\n"
        f"  actual[:4]   = {actual.flatten()[:4].tolist()}\n"
        f"  expected[:4] = {expected.flatten()[:4].tolist()}"
    )
