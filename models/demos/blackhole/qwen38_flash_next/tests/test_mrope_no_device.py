# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The host MRoPE position builder and row select (``mrope.py``): no device, no checkpoint.  With
``QWEN38_TRANSFORMERS_SRC`` naming the pinned ``transformers`` source, also against its ``Qwen4Exp`` implementation."""

import functools
import random
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.mrope import (
    IMAGE_TOKEN_ID,
    MROPE_AXIS_OF_COLUMN,
    MROPE_AXIS_OF_PAIR,
    MROPE_SECTION,
    ROPE_DIM,
    ROPE_THETA,
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
from models.demos.blackhole.qwen38_flash_next.tests.helpers.model import (
    mrope_prompt,
    qwen4_exp_transformers,
    requires_transformers,
)

TEXT = [1000, 1001, 1002, 1003, 1004]


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
    positions = mrope_positions(mrope_prompt(image), [image])
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
    ids = mrope_prompt(TEXT, image, TEXT)
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
    ids = mrope_prompt(TEXT, first, TEXT, second, TEXT)
    positions = mrope_positions(ids, [first, second])
    single_first = mrope_positions(mrope_prompt(TEXT, first, TEXT), [first])
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
        mrope_positions(mrope_prompt(image), [])
    with expect_error(ValueError, match="merged tokens"):
        mrope_positions(mrope_prompt(image), [Qwen38ImageGrid(1, 4, 6)])
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


def test_rows_select_the_axis_per_column_text_rows_are_table_rows_and_refusals(expect_error):
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
    with expect_error(ValueError):
        rope_row(-1, inverse_frequency)
    with expect_error(ValueError):
        rope_row(True, inverse_frequency)
    with expect_error(ValueError):
        rope_table(0, inverse_frequency)


# -- against the pinned Transformers Qwen4Exp --------------------------------------------------------------------------

HEAD_DIM = 256
ROPE_PARAMETERS = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25,
    "rope_theta": ROPE_THETA,
    "rope_type": "default",
}


def _reference_rope_index(ids: list[int], grids: list[Qwen38ImageGrid]):
    """``Qwen4ExpModel.get_rope_index`` on a stub self (no weights): the 3D ids ``[3, 1, L]`` and the delta."""

    model = qwen4_exp_transformers()[1].Qwen4ExpModel
    stub = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    stub.get_vision_position_ids = functools.partial(model.get_vision_position_ids, stub)
    input_ids = torch.tensor([ids], dtype=torch.long)
    token_types = (input_ids == IMAGE_TOKEN_ID).to(torch.int32)
    grid_thw = torch.tensor([grid.as_tuple() for grid in grids], dtype=torch.long) if grids else None
    return model.get_rope_index(stub, input_ids, token_types, image_grid_thw=grid_thw)


def _random_prompt(seed: int):
    generator = random.Random(seed)
    images = generator.randrange(0, 4)
    grids = []
    ids = [generator.randrange(1000, 200_000) for _ in range(generator.randrange(1, 12))]
    for _ in range(images):
        t = 1 if generator.random() < 0.8 else generator.randrange(1, 4)
        grid = Qwen38ImageGrid(t, 2 * generator.randrange(1, 24), 2 * generator.randrange(1, 24))
        grids.append(grid)
        ids += mrope_prompt(grid, [generator.randrange(1000, 200_000) for _ in range(generator.randrange(0, 9))])
    return ids, grids


# A grid with t > 1 (a video frame pair as an image grid): its t positions run p0 .. p0 + t - 1, the text after it
# continues at p0 + max(h, w) // 2 whatever t is (the reference's current_pos rule), and the delta follows.
TEMPORAL_GRIDS = (Qwen38ImageGrid(3, 2, 2), Qwen38ImageGrid(2, 4, 8), Qwen38ImageGrid(4, 6, 2))
ADJACENT_GRIDS = [Qwen38ImageGrid(1, 4, 4), Qwen38ImageGrid(1, 6, 2)]


@requires_transformers
@pytest.mark.parametrize(
    "ids, grids",
    [_random_prompt(seed) for seed in range(12)]
    + [(mrope_prompt([7], grid, [8, 9]), [grid]) for grid in TEMPORAL_GRIDS]
    + [(mrope_prompt([5], *ADJACENT_GRIDS), ADJACENT_GRIDS)],  # separated by the markers alone
    ids=[f"random-{seed}" for seed in range(12)]
    + ["temporal-" + "x".join(map(str, grid.as_tuple())) for grid in TEMPORAL_GRIDS]
    + ["adjacent-images"],
)
def test_positions_and_delta_match_get_rope_index(ids, grids):
    positions = mrope_positions(ids, grids)
    reference_ids, reference_delta = _reference_rope_index(ids, grids)
    assert reference_ids.shape == (3, 1, len(ids))
    assert torch.equal(positions.axes, reference_ids[:, 0, :].to(torch.int64))
    assert positions.delta == int(reference_delta.reshape(-1)[0].item())
    assert positions.text_only == (not grids)


def _reference_rotary():
    configuration, modeling = qwen4_exp_transformers()
    config = configuration.Qwen4ExpTextConfig(
        head_dim=HEAD_DIM, rope_parameters=dict(ROPE_PARAMETERS), max_position_embeddings=262_144
    )
    rotary = modeling.Qwen4ExpTextRotaryEmbedding(config)
    assert rotary.inv_freq.shape == (ROPE_DIM // 2,)
    assert torch.equal(rotary.inv_freq, rope_inverse_frequency())
    return rotary


@requires_transformers
def test_rows_match_the_interleaved_rotary_embedding_in_fp32():
    rotary = _reference_rotary()
    generator = random.Random(7)
    length = 300
    axes = torch.tensor([[generator.randrange(0, length) for _ in range(64)] for _ in range(3)], dtype=torch.int64)
    reference_cos, reference_sin = rotary(torch.zeros((1, 64, HEAD_DIM), dtype=torch.float32), axes[:, None, :])
    assert reference_cos.shape == (1, 64, ROPE_DIM)
    # The fp32 rows built the way the port builds its BF16 tables (per-position products, cos in fp32).
    inverse_frequency = rope_inverse_frequency()
    products = torch.outer(torch.arange(length, dtype=torch.float32), inverse_frequency)
    table = torch.cat((products, products), dim=-1)
    cos, sin = mrope_rows(axes, table.cos(), table.sin())
    torch.testing.assert_close(cos, reference_cos[0], rtol=0, atol=2e-6)
    torch.testing.assert_close(sin, reference_sin[0], rtol=0, atol=2e-6)
    # Distinct axes really are selected: swapping the h axis for the t axis changes the h columns only.
    swapped = axes.clone()
    swapped[1] = axes[0]
    cos_swapped, _ = mrope_rows(swapped, table.cos(), table.sin())
    changed = (cos_swapped != cos).any(dim=0)
    h_columns = torch.tensor([column % 32 % 3 == 1 for column in range(ROPE_DIM)])
    assert not changed[~h_columns].any()
    assert changed[h_columns].any()


@requires_transformers
def test_bf16_rows_match_the_reference_for_text_positions_and_image_positions():
    rotary = _reference_rotary()
    inverse_frequency = rope_inverse_frequency()
    cos_table, sin_table = rope_table(64, inverse_frequency)
    text = torch.arange(7, dtype=torch.int64).reshape(1, -1).expand(3, -1)
    image = torch.tensor([[9, 9, 9, 9], [9, 9, 10, 10], [9, 10, 9, 10]], dtype=torch.int64)
    for axes in (text, image):
        cos, sin = mrope_rows(axes, cos_table, sin_table)
        reference_cos, reference_sin = rotary(
            torch.zeros((1, axes.shape[1], HEAD_DIM), dtype=torch.bfloat16), axes[:, None, :]
        )
        # The reference casts fp32 rows to BF16 as the port does; the per-position product path and the batched
        # matmul path agree to within one BF16 unit in the last place.
        torch.testing.assert_close(cos.float(), reference_cos[0].float(), rtol=2**-7, atol=2**-9)
        torch.testing.assert_close(sin.float(), reference_sin[0].float(), rtol=2**-7, atol=2**-9)
        exact = (cos == reference_cos[0]).float().mean().item()
        assert exact > 0.95, f"only {exact:.3f} of the BF16 cos entries are bitwise the reference's"
