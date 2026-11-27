# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""rows_tile_scatter on the alignments SUPPORTED claims beyond the Phase 0 production cell.

dims (0, 2, 1, 3) keeps W, so a half-row carries its padding columns into the output's padding
columns, and a tile-aligned B only makes every input tile fully live. These cases hold the index map
to that claim (spec rule: every shape validate() accepts), bit for bit, with the input's tile padding
poisoned so a padding value that reached a logical position would show.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_permute import g6_permute, validate

ROWS = (0, 2, 1, 3)

# (shape, expected alignment tag)
CASES = [
    ((1, 32, 64, 64), "tile_aligned"),
    ((2, 64, 32, 96), "tile_aligned"),
    ((1, 6, 16, 40), "w_non_aligned"),  # W and H both non-aligned: W wins
    ((1, 8, 6, 8), "w_non_aligned"),  # second half-row is all padding
    ((2, 34, 64, 100), "w_non_aligned"),  # B tile-aligned, W not
    ((1, 16, 1, 2), "w_non_aligned"),  # one live row, two live columns
]


def _shape_id(shape):
    return "x".join(str(d) for d in shape)


@pytest.mark.parametrize("shape, alignment", CASES, ids=[_shape_id(c[0]) for c in CASES])
def test_alignment_bit_exact(shape, alignment, device):
    torch.manual_seed(11)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    assert validate(ttnn_input, ROWS)["alignment"] == alignment
    if alignment != "tile_aligned":
        ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, float("inf"), memory_config=ttnn_input.memory_config())

    ttnn_output = g6_permute(ttnn_input, ROWS)
    expected = torch.permute(torch_input, ROWS).contiguous()
    assert list(ttnn_output.shape) == list(expected.shape)
    assert ttnn_output.layout == ttnn.TILE_LAYOUT
    assert ttnn_output.memory_config().buffer_type == ttnn.BufferType.DRAM
    actual = ttnn.to_torch(ttnn_output)
    actual_bits = actual.to(torch.bfloat16).contiguous().view(torch.int16)
    expected_bits = expected.contiguous().view(torch.int16)
    mismatch = actual_bits != expected_bits
    assert not bool(
        mismatch.any()
    ), f"{int(mismatch.sum())} of {mismatch.numel()} differ; first at {torch.nonzero(mismatch)[0].tolist()}"
