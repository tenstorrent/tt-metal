# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The host MRoPE position builder and row select (``mrope.py``): no device, no checkpoint."""

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.mrope import (
    IMAGE_TOKEN_ID,
    MROPE_AXIS_OF_COLUMN,
    MROPE_AXIS_OF_PAIR,
    MROPE_SECTION,
    ROPE_DIM,
    VIDEO_TOKEN_ID,
    VISION_END_TOKEN_ID,
    VISION_START_TOKEN_ID,
    Qwen38ImageGrid,
    Qwen38MRoPEPositions,
    expand_image_pads,
    image_groups,
    mrope_positions,
    mrope_rows,
    rope_inverse_frequency,
    rope_row,
    rope_table,
)

TEXT = [1000, 1001, 1002, 1003, 1004]


def _prompt(*parts):
    ids = []
    for part in parts:
        if isinstance(part, Qwen38ImageGrid):
            ids += [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * part.merged_tokens + [VISION_END_TOKEN_ID]
        else:
            ids += list(part)
    return ids


def test_interleaved_section_assignment():
    assert MROPE_SECTION == (11, 11, 10)
    assert [pair for pair in range(32) if MROPE_AXIS_OF_PAIR[pair] == 0] == list(range(0, 33, 3))
    assert [pair for pair in range(32) if MROPE_AXIS_OF_PAIR[pair] == 1] == list(range(1, 33, 3))
    assert [pair for pair in range(32) if MROPE_AXIS_OF_PAIR[pair] == 2] == list(range(2, 30, 3))
    assert MROPE_AXIS_OF_COLUMN.shape == (ROPE_DIM,)
    assert torch.equal(MROPE_AXIS_OF_COLUMN[:32], MROPE_AXIS_OF_COLUMN[32:])


@pytest.mark.parametrize(
    "grid, tokens, span, delta",
    [
        ((1, 16, 16), 64, 8, -56),
        ((1, 28, 28), 196, 14, -182),
        ((1, 64, 64), 1024, 32, -992),
        ((1, 68, 120), 2040, 60, -1980),
        ((1, 256, 256), 16384, 128, -16256),
        ((1, 2, 6), 3, 3, 0),
    ],
)
def test_known_grids(grid, tokens, span, delta):
    image = Qwen38ImageGrid(*grid)
    assert image.merged_tokens == tokens
    assert image.position_span == span
    positions = mrope_positions(_prompt(image), [image])
    assert positions.length == tokens + 2
    assert positions.delta == delta
    assert positions.shift == -delta


def test_text_only_is_the_plain_arange():
    ids = TEXT * 7
    positions = mrope_positions(ids)
    assert positions.text_only
    assert positions.delta == 0
    assert torch.equal(positions.axes, torch.arange(len(ids)).reshape(1, -1).expand(3, -1))
    assert positions.generated(len(ids)) == len(ids)
    assert mrope_positions([]).length == 0


def test_image_positions_follow_the_grid_and_text_continues_past_the_span():
    image = Qwen38ImageGrid(1, 4, 6)  # 2 x 3 merged tokens, span 3
    ids = _prompt(TEXT, image, TEXT)
    positions = mrope_positions(ids, [image])
    # TEXT and <|vision_start|> occupy indices 0 .. 5 at positions 0 .. 5; the pads start at index p0 = 6.
    p0 = len(TEXT) + 1
    assert positions.axes[:, p0 - 1].tolist() == [p0 - 1] * 3
    expected_t = [p0] * 6
    expected_h = [p0 + 0] * 3 + [p0 + 1] * 3
    expected_w = [p0, p0 + 1, p0 + 2] * 2
    assert positions.axes[0, p0 : p0 + 6].tolist() == expected_t
    assert positions.axes[1, p0 : p0 + 6].tolist() == expected_h
    assert positions.axes[2, p0 : p0 + 6].tolist() == expected_w
    # <|vision_end|> (index p0 + 6) continues at p0 + span, then the trailing text.
    end = p0 + 6
    assert positions.axes[:, end].tolist() == [p0 + 3] * 3
    assert positions.axes[:, end + 1 : end + 6].tolist() == [[p0 + 4 + i for i in range(5)]] * 3
    assert positions.delta == (p0 + 8 + 1) - len(ids) == -3
    assert positions.generated(len(ids)) == p0 + 9
    assert not positions.text_only
    assert torch.equal(positions.rows(end, end + 2), positions.axes[:, end : end + 2])


def test_two_images_accumulate_the_delta():
    first = Qwen38ImageGrid(1, 8, 4)  # 8 tokens, span 4
    second = Qwen38ImageGrid(1, 4, 12)  # 12 tokens, span 6
    ids = _prompt(TEXT, first, TEXT, second, TEXT)
    positions = mrope_positions(ids, [first, second])
    single_first = mrope_positions(_prompt(TEXT, first, TEXT), [first])
    assert single_first.delta == 4 - 8
    assert positions.delta == (4 - 8) + (6 - 12)
    # The second image starts where the text after the first left off.
    second_start = len(TEXT) + 1 + 8 + 1 + len(TEXT) + 1
    p0 = second_start - single_first.shift
    assert positions.axes[:, second_start].tolist() == [p0, p0, p0]
    assert positions.axes[2, second_start : second_start + 6].tolist() == [p0 + i for i in range(6)]


def test_refusals(expect_error):
    image = Qwen38ImageGrid(1, 4, 4)
    with expect_error(ValueError, match="video"):
        mrope_positions(TEXT + [VIDEO_TOKEN_ID])
    with expect_error(ValueError, match="image pad runs"):
        mrope_positions(_prompt(image), [])
    with expect_error(ValueError, match="merged tokens"):
        mrope_positions(_prompt(image), [Qwen38ImageGrid(1, 4, 6)])
    with expect_error(ValueError, match="multiple of 2"):
        Qwen38ImageGrid(1, 3, 4)
    with expect_error(ValueError, match="positive"):
        Qwen38ImageGrid(0, 4, 4)
    with expect_error(ValueError, match="delta"):
        Qwen38MRoPEPositions(torch.zeros((3, 2), dtype=torch.int64), 1)
    with expect_error(ValueError, match="int64"):
        Qwen38MRoPEPositions(torch.zeros((2, 2), dtype=torch.int64), 0)
    positions = mrope_positions(TEXT)
    with expect_error(ValueError, match="inside"):
        positions.generated(2)
    with expect_error(ValueError, match="outside"):
        positions.rows(3, 9)


def test_image_groups_and_pad_expansion(expect_error):
    grids = [Qwen38ImageGrid(1, 4, 4), Qwen38ImageGrid(1, 2, 8)]
    rendered = TEXT + [VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, VISION_END_TOKEN_ID] + TEXT[:2]
    rendered += [VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, VISION_END_TOKEN_ID]
    expanded = expand_image_pads(rendered, grids)
    assert len(expanded) == len(rendered) - 2 + 4 + 4
    assert expanded.count(IMAGE_TOKEN_ID) == 8
    assert image_groups(expanded) == [(len(TEXT) + 1, len(TEXT) + 5), (len(TEXT) + 5 + 1 + 2 + 1, len(TEXT) + 13)]
    assert mrope_positions(expanded, grids).delta == (2 - 4) + (4 - 4)
    with expect_error(ValueError, match="image pads"):
        expand_image_pads(rendered, grids[:1])
    assert expand_image_pads(TEXT, []) == TEXT
    assert image_groups(TEXT) == []
    assert image_groups([IMAGE_TOKEN_ID]) == [(0, 1)]


def test_rows_select_the_axis_per_column_and_text_rows_are_table_rows(expect_error):
    inverse_frequency = rope_inverse_frequency()
    cos_table, sin_table = rope_table(40, inverse_frequency)
    assert cos_table.shape == (40, ROPE_DIM) and cos_table.dtype == torch.bfloat16
    for position in (0, 1, 17, 39):
        cos, sin = rope_row(position, inverse_frequency)
        assert torch.equal(cos.reshape(-1), cos_table[position])
        assert torch.equal(sin.reshape(-1), sin_table[position])
    # Text rows: three equal positions read the table row bitwise.
    text = torch.tensor([[3, 7, 39], [3, 7, 39], [3, 7, 39]], dtype=torch.int64)
    cos, sin = mrope_rows(text, cos_table, sin_table)
    assert torch.equal(cos, cos_table[[3, 7, 39]]) and torch.equal(sin, sin_table[[3, 7, 39]])
    # Distinct axes: column c reads the row of axis (c % 32) % 3.
    axes = torch.tensor([[10, 5], [20, 5], [30, 6]], dtype=torch.int64)
    cos, sin = mrope_rows(axes, cos_table, sin_table)
    for column in range(ROPE_DIM):
        axis = MROPE_AXIS_OF_PAIR[column % 32]
        assert cos[0, column] == cos_table[axes[axis, 0], column]
        assert sin[1, column] == sin_table[axes[axis, 1], column]
    with expect_error(ValueError, match=r"\[0, 40\)"):
        mrope_rows(torch.tensor([[40], [1], [1]], dtype=torch.int64), cos_table, sin_table)
    with expect_error(ValueError, match="int64"):
        mrope_rows(torch.zeros((3, 1), dtype=torch.int32), cos_table, sin_table)


def test_rope_row_refusals(expect_error):
    inverse_frequency = rope_inverse_frequency()
    with expect_error(ValueError):
        rope_row(-1, inverse_frequency)
    with expect_error(ValueError):
        rope_row(True, inverse_frequency)
    with expect_error(ValueError):
        rope_table(0, inverse_frequency)
