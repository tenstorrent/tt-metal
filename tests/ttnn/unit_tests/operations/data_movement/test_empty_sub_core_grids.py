# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

# An empty sub_core_grids used to divide by its core count on the host (untilize, SIGFPE) or index its empty core
# list (the tilize family and untilize_with_unpadding, IndexError).

EMPTY = ttnn.CoreRangeSet([])


def _tensor(device, layout, shape=(1, 1, 32, 256)):
    return ttnn.from_torch(torch.randn(shape).bfloat16(), dtype=ttnn.bfloat16, layout=layout, device=device)


@pytest.mark.parametrize(
    "run",
    [
        lambda d: ttnn.untilize(_tensor(d, ttnn.TILE_LAYOUT), sub_core_grids=EMPTY),
        lambda d: ttnn.untilize_with_unpadding(_tensor(d, ttnn.TILE_LAYOUT), [0, 0, 31, 255], sub_core_grids=EMPTY),
        lambda d: ttnn.tilize(_tensor(d, ttnn.ROW_MAJOR_LAYOUT), sub_core_grids=EMPTY),
        lambda d: ttnn.to_layout(_tensor(d, ttnn.ROW_MAJOR_LAYOUT), ttnn.TILE_LAYOUT, sub_core_grids=EMPTY),
        lambda d: ttnn.tilize_with_val_padding(
            _tensor(d, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 30, 256)), [1, 1, 32, 256], 0.0, sub_core_grids=EMPTY
        ),
        lambda d: ttnn.tilize_with_zero_padding(
            _tensor(d, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 30, 256)), sub_core_grids=EMPTY
        ),
    ],
    ids=[
        "untilize",
        "untilize_with_unpadding",
        "tilize",
        "to_layout",
        "tilize_with_val_padding",
        "tilize_with_zero_padding",
    ],
)
def test_empty_sub_core_grids_raises(device, expect_error, run):
    with expect_error(RuntimeError, "sub_core_grids must contain at least one core"):
        run(device)
