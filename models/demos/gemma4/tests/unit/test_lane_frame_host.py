# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The lane engine's decode-frame shapes (``tt/lane_frame.py``), host-only: the engine's own host tensors pass, the
window-5 mistake (a ``[1, rows]`` tensor handed to the paged cache update as its positions) and every other
frame-vs-row slip are refused with the tensor named.

    pytest models/demos/gemma4/tests/unit/test_lane_frame_host.py -p no:cacheprovider
"""

import pytest

from models.demos.gemma4.tt.lane_frame import check_lane_decode_frame, lane_decode_frame_shapes

LANES, ROWS, BLOCKS, HIDDEN = 4, 32, 3, 2560


def _engine_shapes():
    return {name: host for name, (host, _col, _dim) in lane_decode_frame_shapes(LANES, ROWS, BLOCKS, HIDDEN).items()}


def test_the_engines_host_tensors_shard_to_the_column_frame():
    columns = check_lane_decode_frame(LANES, ROWS, **_engine_shapes())
    assert columns == {
        "hidden": (1, 1, ROWS, HIDDEN),
        "rope_positions": (1, ROWS),
        "cache_positions": (ROWS,),
        "page_table": (ROWS, BLOCKS),
    }
    # the (2,4) block: 4 lanes of 32 rows as well; one lane of 32 rows is the smallest frame
    assert check_lane_decode_frame(1, ROWS, cache_positions=(ROWS,))["cache_positions"] == (ROWS,)


def test_two_dimensional_cache_positions_are_the_window_5_mistake(expect_error):
    shapes = _engine_shapes()
    shapes["cache_positions"] = (1, ROWS)  # one row per column: the update reads ONE index for 32 rows
    with expect_error(ValueError, "cache_positions: rank 2"):
        check_lane_decode_frame(LANES, ROWS, **shapes)


@pytest.mark.parametrize(
    ("name", "shape", "message"),
    [
        ("rope_positions", (1, ROWS), "rope_positions: 32 rows on the lane-shard dim 1"),  # per column, not the frame
        ("page_table", (ROWS, BLOCKS), "page_table: 32 rows on the lane-shard dim 0"),
        ("hidden", (1, 1, ROWS, HIDDEN), "hidden: 32 rows on the lane-shard dim 2"),
        ("page_table", (LANES * ROWS,), "page_table: rank 1"),
    ],
)
def test_per_column_or_misranked_tensors_are_refused(expect_error, name, shape, message):
    shapes = _engine_shapes()
    shapes[name] = shape
    with expect_error(ValueError, message):
        check_lane_decode_frame(LANES, ROWS, **shapes)


def test_unknown_tensor_names_are_refused(expect_error):
    with expect_error(ValueError, "unknown decode-frame tensor"):
        check_lane_decode_frame(LANES, ROWS, masks=(1, ROWS))
