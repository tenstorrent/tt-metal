# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import (
    MAX_CONTEXT,
    MIN_QSA_CACHE_CAPACITY,
    QSA_CACHE_CAPACITY_ALIGNMENT,
    Qwen38TTNNQSA,
    validate_qsa_cache_capacity,
)


@pytest.mark.parametrize("capacity", (MIN_QSA_CACHE_CAPACITY, 32768, MAX_CONTEXT))
def test_qsa_physical_capacity_accepts_aligned_bounds_and_resident_default(capacity: int) -> None:
    assert validate_qsa_cache_capacity(capacity) == capacity


@pytest.mark.parametrize(
    "capacity",
    (
        True,
        2048.0,
        MIN_QSA_CACHE_CAPACITY - 1,
        MIN_QSA_CACHE_CAPACITY + 1,
        MAX_CONTEXT + QSA_CACHE_CAPACITY_ALIGNMENT,
    ),
)
def test_qsa_physical_capacity_rejects_wrong_type_range_or_alignment(expect_error, capacity) -> None:
    with expect_error((TypeError, ValueError)):
        validate_qsa_cache_capacity(capacity)


def test_complete_block_query_row_covers_repeated_four_token_close_cadence() -> None:
    """The front-padded placement (the fallback inside the cache's last tile row) keeps its causal row."""

    for complete_blocks in range(1, 65):
        close_token_index = complete_blocks * 4 - 1
        valid_padded = math.ceil(complete_blocks / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        chunk_start = valid_padded - ttnn.TILE_SIZE
        query_row = complete_blocks - 1 - chunk_start
        front_padding = query_row
        back_padding = ttnn.TILE_SIZE - query_row - 1

        assert close_token_index % 4 == 3
        assert 0 <= query_row < ttnn.TILE_SIZE
        assert 1 + front_padding + back_padding == ttnn.TILE_SIZE

    assert (0, 1, 2) == tuple((complete_blocks - 1) % ttnn.TILE_SIZE for complete_blocks in (1, 2, 3))
    assert (3, 7, 11) == tuple(complete_blocks * 4 - 1 for complete_blocks in (1, 2, 3))


def test_score_complete_blocks_views_the_query_tile_and_front_pads_only_in_the_last_tile_row() -> None:
    source = inspect.getsource(Qwen38TTNNQSA._score_complete_blocks)
    view = source.index("query_view = valid_padded + ttnn.TILE_SIZE <= self.allocated_compressed_blocks")
    reshape = source.index("query_tile = ttnn.reshape(index_query, tile_shape, tile_shape)", view)
    fallback = source.index("query_tile = self._front_padded_query(index_query, query_row)", reshape)
    indexer = source.index("ttnn.experimental.indexer_score_dsa(", fallback)
    assert view < reshape < fallback < indexer
    assert "ttnn.pad(" not in source and "ttnn.to_layout(" not in source
    assert "chunk_start = valid_padded\n            kv_len = valid_padded + ttnn.TILE_SIZE" in source
    assert "chunk_start = valid_padded - ttnn.TILE_SIZE\n            kv_len = valid_padded" in source
    assert "query_row = complete_blocks - 1 - chunk_start" in source
    assert "if not query_view:\n            _deallocate(query_tile)" in source  # the view shares index_query's buffer


def test_front_padded_query_pads_only_after_row_major_conversion() -> None:
    source = inspect.getsource(Qwen38TTNNQSA._front_padded_query)
    row_major = source.index("query_row_major = ttnn.to_layout(")
    pad = source.index("query_padded_row_major = ttnn.pad(")
    tile = source.index("query_padded = ttnn.to_layout(", pad)

    assert row_major < pad < tile
    assert "index_query,\n            ttnn.ROW_MAJOR_LAYOUT," in source
    assert "query_padded_row_major,\n            ttnn.TILE_LAYOUT," in source
    assert "ttnn.pad(\n            index_query," not in source
    assert "(query_row, ttnn.TILE_SIZE - query_row - 1)" in source


def test_pinned_runtime_tile_pad_rejects_front_padding_but_row_major_path_supports_it() -> None:
    repository = Path(inspect.getfile(Qwen38TTNNQSA)).resolve().parents[5]
    pad_source = (repository / "ttnn/cpp/ttnn/operations/data_movement/pad/pad.cpp").read_text()
    row_major = pad_source[pad_source.index("ttnn::Tensor invoke_rm(") : pad_source.index("ttnn::Tensor invoke_tile(")]
    tile = pad_source[pad_source.index("ttnn::Tensor invoke_tile(") :]

    assert "pad_impl(input_tensor, padding_vec, value" in row_major
    assert 'TT_FATAL(front_padding_is_zero, "ttnn.pad: on device tile padding does not support front padding")' in tile
