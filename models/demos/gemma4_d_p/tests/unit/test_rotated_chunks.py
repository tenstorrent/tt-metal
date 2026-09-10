# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.common.prefill.chunk_layout import chunk_positions, chunk_row_for_position, pack_chunk_tokens


@pytest.mark.parametrize("sp,chunk", [(8, 8192), (4, 4096), (8, 16384)])
@pytest.mark.parametrize("start", [0, 32, 6976, 8192, 8992, 25312])
def test_rotated_mapping_is_a_bijection_with_local_suffix_padding(sp, chunk, start):
    positions = chunk_positions(start, chunk, sp)
    assert sorted(positions) == list(range(start, start + chunk))
    end = start + chunk // 3 + 7
    packed = pack_chunk_tokens(list(range(start, end)), start, end, chunk, sp, pad_token=-1)
    local = chunk // sp
    for rank in range(sp):
        indices = positions[rank * local : (rank + 1) * local]
        assert indices == sorted(indices)
        values = packed[rank * local : (rank + 1) * local]
        real = [p for p in indices if p < end]
        assert values == real + [-1] * (local - len(real))
    assert positions[chunk_row_for_position(end - 1, start, chunk, sp)] == end - 1
    if start % chunk == 0:
        assert positions == list(range(start, start + chunk))


def test_boundary_crossing_example():
    tokens = pack_chunk_tokens(list(range(6976, 9000)), 6976, 9000, 8192, 8, pad_token=-1)
    counts = [sum(t != -1 for t in tokens[r * 1024 : (r + 1) * 1024]) for r in range(8)]
    assert counts == [808, 0, 0, 0, 0, 0, 192, 1024]


def test_short_source_is_rejected(expect_error):
    with expect_error(ValueError, "shorter"):
        pack_chunk_tokens([1], 6976, 9000, 8192, 8)
