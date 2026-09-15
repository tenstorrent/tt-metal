# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression coverage for untilize_with_unpadding when it crops the WIDTH of a wide tensor.

For an input wider than `threshold_row_block` (32) tiles, untilize_with_unpadding selects
UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory. That factory splits work over the
*input* padded width (common.cpp, make_block_plan, BlockDirection::Untilize), so a core can be
handed a block lying entirely past the unpadded output width. The writer kernel clamps its write
against the output width, and that clamp used to underflow for such a block, producing a ~4 GB
noc_async_write that never retired: the device hung with no assert and the board needed `tt-smi -r`.

Plain untilize shares the same writer kernel but cannot hit this, because there the wrap modulus and
the clamp bound are the same value; only the unpadding variant has the two disagree.

The cases below cover all three kinds of block:
  - wholly inside the unpadded width          (always worked)
  - straddling the boundary -> partial write  (always worked; guards against skipping too much)
  - wholly at or past the boundary -> skipped (this is what used to hang)
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

TILE = 32

# (height, input_padded_width, output_width)
WIDTH_CROP_CASES = [
    # 32 tiles/row: below the threshold, so a different factory handles it.
    pytest.param(128, 1024, 512, id="32tiles_cropped"),
    # 33 tiles/row but no crop: the block factory runs, the write is never clamped.
    pytest.param(128, 1056, 1056, id="33tiles_uncropped"),
    # 33 tiles/row + crop: the smallest shape over the threshold that also discards whole blocks.
    pytest.param(128, 1056, 512, id="33tiles_cropped"),
    # The shape this was originally found on (qkv projection of a GDN prefill chunk).
    pytest.param(128, 4128, 2560, id="129tiles_cropped"),
    # The crops above land on a block boundary, so every block is wholly kept or wholly discarded.
    # Straddle only: the crop falls inside the final (cliff) block, so no block is entirely past it.
    pytest.param(128, 1056, 1050, id="33tiles_straddle_only"),
    # Straddle plus wholly-discarded blocks: exercises both branches in one program.
    pytest.param(128, 1056, 500, id="33tiles_straddle_and_overhang"),
    pytest.param(128, 4128, 2500, id="129tiles_straddle_and_overhang"),
]


@pytest.mark.parametrize("height, padded_width, out_width", WIDTH_CROP_CASES)
def test_untilize_with_unpadding_width_crop(device, height, padded_width, out_width):
    assert padded_width % TILE == 0, "input width is the already-padded width"

    torch.manual_seed(42)
    torch_input = torch.randn(1, height, padded_width, dtype=torch.bfloat16)

    tile_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # output_tensor_end is inclusive
    untilized = ttnn.untilize_with_unpadding(
        tile_tensor,
        [0, height - 1, out_width - 1],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.to_torch(untilized)
    expected = torch_input[:, :, :out_width]

    assert tuple(result.shape) == tuple(expected.shape)
    # A bf16 tilize/untilize round trip is an identity, so this is exact.
    assert_equal(result, expected)


def test_untilize_with_unpadding_width_crop_matches_slice_then_to_layout(device):
    """The two-op form computes the same thing, and was the workaround while this hung."""
    height, padded_width, out_width = 128, 1056, 512

    torch.manual_seed(0)
    torch_input = torch.randn(1, height, padded_width, dtype=torch.bfloat16)
    tile_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    sliced = ttnn.slice(tile_tensor, (0, 0, 0), (1, height, out_width))
    two_op = ttnn.to_torch(ttnn.to_layout(sliced, ttnn.ROW_MAJOR_LAYOUT))

    one_op = ttnn.to_torch(
        ttnn.untilize_with_unpadding(
            tile_tensor,
            [0, height - 1, out_width - 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    )

    assert_equal(one_op, two_op)
