# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the sharded decode RMSNorm grid selection."""

import pytest

from models.demos.muse_glimmer_30b.tt.rms_norm import rms_norm_path_info


@pytest.mark.parametrize(
    "dim,grid,expected",
    [
        (6656, (12, 10), (8, 2, 13, 1)),
        (6656, (11, 10), (8, 2, 13, 1)),
        (6656, (8, 8), (8, 2, 13, 1)),
        (6656, (13, 10), (13, 8, 2, 2)),
        (1664, (12, 10), (4, 1, 13, 1)),
        (1664, (13, 10), (13, 4, 1, 1)),
        (832, (12, 10), (2, 1, 13, 1)),
    ],
)
def test_path_info(dim, grid, expected):
    info = rms_norm_path_info((1, 1, 32, dim), grid)
    assert info == expected
    gx, gy, bw, sb = info
    assert gx <= grid[0] and gy <= grid[1]
    assert (gx * gy) * bw == dim // 32
    assert bw % sb == 0 and sb in (1, 2, 3, 4)


def test_unaligned():
    assert rms_norm_path_info((1, 1, 32, 100), (12, 10)) is None
