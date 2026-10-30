# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The host side of the vision splice (``vision_splice.py``), the decode-tail rule of the rotary positions and the
committed prefix of an image prompt as a reuse key: no device, no checkpoint."""

from types import SimpleNamespace

import torch

from models.demos.blackhole.qwen38_flash_next.mrope import IMAGE_TOKEN_ID, Qwen38ImageGrid, mrope_positions
from models.demos.blackhole.qwen38_flash_next.tests.helpers.model import mrope_prompt
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import (
    Qwen38ChatSession,
    Qwen38PromptSnapshot,
    vision_spans_compatible,
)
from models.demos.blackhole.qwen38_flash_next.vision_splice import (
    HIDDEN_SIZE,
    NEGATIVE_ZERO_BF16_BITS,
    Qwen38VisionPrompt,
    clean_feature_rows,
    feature_rows_image,
    image_lanes,
    image_spans,
    sentinel_token_rows,
    split_features,
)

TEXT = [1000, 1001, 1002, 1003]


def test_image_lanes_and_sentinel_rows(expect_error):
    ids = TEXT + [IMAGE_TOKEN_ID, IMAGE_TOKEN_ID] + TEXT
    assert image_lanes(ids) == [4, 5]
    rows = torch.tensor(ids, dtype=torch.float32).reshape(1, 1, 1, -1)
    with_sentinel = sentinel_token_rows(rows, [4, 5])
    assert with_sentinel.reshape(-1)[4:6].tolist() == [-1.0, -1.0]
    assert torch.equal(with_sentinel.reshape(-1)[:4], rows.reshape(-1)[:4])
    assert torch.equal(rows.reshape(-1)[4:6], torch.tensor([float(IMAGE_TOKEN_ID)] * 2))  # the input is untouched
    with expect_error(ValueError, match="outside"):
        sentinel_token_rows(rows, [len(ids)])


def test_clean_rows_are_negative_zero_everywhere(expect_error):
    rows = clean_feature_rows(32, hidden=8)
    assert rows.shape == (1, 1, 32, 8) and rows.dtype == torch.bfloat16
    assert (rows.view(torch.int16) == NEGATIVE_ZERO_BF16_BITS).all()
    assert (rows == 0).all()  # -0.0 == 0.0
    # The identity of the add: x + (-0.0) == x bitwise for +0.0, -0.0 and ordinary values.
    values = torch.tensor([0.0, -0.0, 1.5, -2.25, 1e-3], dtype=torch.bfloat16)
    summed = values + rows[0, 0, 0, :5]
    assert torch.equal(summed.view(torch.int16), values.view(torch.int16))
    with expect_error(ValueError):
        clean_feature_rows(0)


def test_feature_rows_place_the_features_at_the_image_lanes(expect_error):
    ids = TEXT[:2] + [IMAGE_TOKEN_ID] * 3 + TEXT[:3]
    features = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8).to(torch.bfloat16)
    rows = feature_rows_image(ids, features, hidden=8)
    assert rows.shape == (1, 1, 8, 8)
    assert torch.equal(rows[0, 0, 2:5], features)
    text = torch.cat([rows[0, 0, :2], rows[0, 0, 5:]])
    assert (text.view(torch.int16) == NEGATIVE_ZERO_BF16_BITS).all()
    assert feature_rows_image(TEXT, None, hidden=8) is None
    assert feature_rows_image(TEXT, torch.zeros((0, 8), dtype=torch.bfloat16), hidden=8) is None
    with expect_error(ValueError, match="3 image pads in the chunk vs 2"):
        feature_rows_image(ids, features[:2], hidden=8)
    with expect_error(ValueError, match="BF16"):
        feature_rows_image(ids, features.float(), hidden=8)


def test_split_features_walks_the_chunks(expect_error):
    features = torch.zeros((5, HIDDEN_SIZE), dtype=torch.bfloat16)
    for row in range(5):
        features[row, 0] = row
    first = [IMAGE_TOKEN_ID] * 2 + TEXT
    second = TEXT + [IMAGE_TOKEN_ID] * 3
    part, cursor = split_features(first, features, 0)
    assert cursor == 2 and part[:, 0].tolist() == [0.0, 1.0]
    none, cursor = split_features(TEXT, features, cursor)
    assert none is None and cursor == 2
    part, cursor = split_features(second, features, cursor)
    assert cursor == 5 and part[:, 0].tolist() == [2.0, 3.0, 4.0]
    with expect_error(ValueError, match="3 image pads but 0 feature rows remain"):
        split_features(second, features, cursor)
    with expect_error(ValueError, match="2 image pads but 0 feature rows remain"):
        split_features(first, None, 0)


def test_shift_at_and_the_decode_tail_rule(expect_error):
    image = Qwen38ImageGrid(1, 8, 8)  # 16 pads, span 4: shift 12
    ids = mrope_prompt(TEXT, image, TEXT[:4])
    positions = mrope_positions(ids, [image])
    assert positions.shift == 12
    assert positions.shift_at(positions.length) == 12
    assert positions.shift_at(len(TEXT)) == 0  # before the image: plain text
    assert positions.shift_at(len(TEXT) + 1 + 1) == 0  # the first pad sits at its own index
    assert positions.shift_at(len(TEXT) + 1 + 16) == 12  # after the last pad: the whole shift
    with expect_error(ValueError):
        positions.shift_at(positions.length + 1)
    # The tail: <|vision_end|> and four text tokens follow the last pad.  Token length - 1 = index 25, its block
    # starts at 24 (= <|vision_end|> + 3): every token from 24 to 25 is plain text at index - 12.
    assert positions.tail_is_plain(positions.length)
    # A prompt whose last image ends inside the last index block: <|vision_end|> alone after the pads.
    short = mrope_positions(mrope_prompt(TEXT, image), [image])
    end = len(TEXT) + 1 + 16  # the <|vision_end|> index (21); its block start 20 is a pad
    assert not short.tail_is_plain(end + 1)
    # The two decode formulas disagree exactly when the shift is not a multiple of four: (P & ~3) - S is the block's
    # first token's row for a plain tail, (P - S) & ~3 is not.
    P = positions.length + 5
    S = positions.shift
    assert (P & ~3) - S == int(positions.axes[0, P & ~3 - 0].item()) if (P & ~3) < positions.length else True
    odd = Qwen38ImageGrid(1, 6, 6)  # 9 pads, span 3: shift 6
    odd_positions = mrope_positions(mrope_prompt(TEXT, odd, TEXT[:4]), [odd])
    assert odd_positions.shift == 6 and odd_positions.tail_is_plain(odd_positions.length)
    P = odd_positions.length + 2
    block_start_token_row = (
        int(odd_positions.axes[0, (P & ~3)].item()) if (P & ~3) < odd_positions.length else (P & ~3) - 6
    )
    assert (P & ~3) - 6 == block_start_token_row
    assert ((P - 6) & ~3) != (P & ~3) - 6 or (P & ~3) % 4 == 0 and 6 % 4 == 0


def test_tail_rule_keeps_the_shift_within_the_block_start():
    """The device rule S <= P & ~3 (contracts._require_rope_shift) follows from tail_is_plain at the prompt end."""

    for grid in (Qwen38ImageGrid(1, 8, 8), Qwen38ImageGrid(1, 6, 6), Qwen38ImageGrid(1, 2, 30)):
        for tail in (4, 5, 9):
            positions = mrope_positions(mrope_prompt(TEXT, grid, TEXT[:tail]), [grid])
            length = positions.length
            assert positions.tail_is_plain(length)
            for consumed in (length - 1, length):  # the chunks' end and the first decode position
                assert positions.shift_at(consumed) <= consumed & ~3


def test_image_spans_and_the_prompt_objects_keyed_spans(expect_error):
    grid = Qwen38ImageGrid(1, 4, 4)  # 4 merged tokens
    ids = mrope_prompt(TEXT, grid, TEXT[:2], grid, TEXT)
    first = len(TEXT) + 1
    second = first + 4 + 1 + 2 + 1
    assert image_spans(ids) == [(first, first + 4), (second, second + 4)]
    assert image_spans(TEXT) == [] and image_spans([IMAGE_TOKEN_ID] * 3) == [(0, 3)]
    positions = mrope_positions(ids, [grid, grid])
    features = torch.zeros((8, 2560), dtype=torch.bfloat16)
    keyed = Qwen38VisionPrompt(positions, features, "request", ("a", "b"))
    assert keyed.spans(ids) == ((first, first + 4, "a"), (second, second + 4, "b"))
    # without per-image digests every span carries the request digest (correct, less reusable)
    plain = Qwen38VisionPrompt(positions, features, "request")
    assert plain.spans(ids) == ((first, first + 4, "request"), (second, second + 4, "request"))
    keyed.validate_prompt(ids)
    with expect_error(ValueError, match="image digests"):
        Qwen38VisionPrompt(positions, features, "request", ("a",)).validate_prompt(ids)


def test_vision_prompt_validation(expect_error):
    image = Qwen38ImageGrid(1, 4, 4)
    ids = mrope_prompt(TEXT, image, TEXT[:4])
    positions = mrope_positions(ids, [image])
    features = torch.zeros((image.merged_tokens, HIDDEN_SIZE), dtype=torch.bfloat16)
    prompt = Qwen38VisionPrompt(positions, features, digest="abc")
    prompt.validate_prompt(ids)
    with expect_error(ValueError, match="rotary positions"):
        prompt.validate_prompt(ids + TEXT)
    with expect_error(ValueError, match="feature rows"):
        Qwen38VisionPrompt(positions, features[:-1]).validate_prompt(ids)
    with expect_error(ValueError, match="BF16"):
        Qwen38VisionPrompt(positions, features.float())
    short_ids = mrope_prompt(TEXT, image)
    short = Qwen38VisionPrompt(mrope_positions(short_ids, [image]), features)
    with expect_error(ValueError, match="last index block"):
        short.validate_prompt(short_ids)


# -- the committed prefix as a reuse key (tools/qwen38_chat_session.py) ------------------------------------------------
# The session keys its committed ids and its prompt-end snapshot on the images' pad spans with their digests beside the
# ids, so a follow-up turn on the same image extends (or restores) instead of resetting, a prompt with another image of
# the same grid (identical pad ids) resets, and a text request never extends into an image it does not carry.

GRID = Qwen38ImageGrid(1, 8, 8)  # 16 merged tokens
TURNS = list(range(1000, 1012))  # TURNS[:4] == TEXT


def _vision(ids: list[int], *digests: str) -> Qwen38VisionPrompt:
    grids = [GRID] * len(digests)
    features = torch.zeros((GRID.merged_tokens * len(digests), 2560), dtype=torch.bfloat16)
    return Qwen38VisionPrompt(mrope_positions(ids, grids), features, "+".join(digests), tuple(digests))


def _session() -> Qwen38ChatSession:
    chain = SimpleNamespace(capture_prompt_snapshot=lambda position: None, restore_prompt_snapshot=lambda: None)
    return Qwen38ChatSession(chain, SimpleNamespace())


def test_spans_compatibility_rule() -> None:
    a = ((5, 21, "A"),)
    assert vision_spans_compatible(a, a, 30) and vision_spans_compatible((), (), 30)
    assert not vision_spans_compatible(a, ((5, 21, "B"),), 30)  # the same grid, another image
    assert not vision_spans_compatible(a, (), 30) and not vision_spans_compatible((), a, 30)
    assert vision_spans_compatible(a, ((5, 21, "A"), (40, 56, "B")), 30)  # a new image beyond the prefix
    assert not vision_spans_compatible((), ((5, 21, "A"),), 10)  # a span the prefix cuts through
    assert vision_spans_compatible((), ((40, 56, "B"),), 30)


def test_a_follow_up_turn_on_the_same_image_extends_and_another_image_of_the_grid_resets() -> None:
    session = _session()
    first = mrope_prompt(TEXT, GRID, TURNS[4:8])
    vision_a = _vision(first, "A")
    # the committed state after the first turn's prefill and its reply
    session.committed = first + TURNS[8:10]
    session.committed_vision = vision_a.spans(first)
    second = first + TURNS[8:10] + TURNS[10:12]
    assert session.reusable_prefix(second, _vision(second, "A").spans(second)) == (len(session.committed), "extends")
    # the same ids with another image (the same grid renders the same pads): a reset, never a reuse
    assert session.reusable_prefix(second, _vision(second, "B").spans(second)) == (0, "reset")
    # a text prompt equal to the ids so far cannot exist without the vision inputs; a text prefix reuse is untouched
    session.committed_vision = ()
    session.committed = TURNS[:6]
    assert session.reusable_prefix(TURNS[:8]) == (6, "extends")


def test_the_prompt_end_snapshot_of_an_image_prompt_is_keyed_on_its_images_too() -> None:
    session = _session()
    first = mrope_prompt(TEXT, GRID, TURNS[4:8])
    spans_a = _vision(first, "A").spans(first)
    session.snapshot = Qwen38PromptSnapshot(tuple(first[:-1]), None, "chunked", spans_a)
    session.committed = first + TURNS[8:11]  # the committed state moved on (a reply); the snapshot still serves
    other = first[:-1] + [TURNS[11]] + TURNS[8:10]  # a prompt extending the snapshot's ids with another question
    assert session.reusable_prefix(other, _vision(other, "A").spans(other)) == (len(first) - 1, "snapshot")
    assert session.reusable_prefix(other, _vision(other, "B").spans(other)) == (0, "reset")
    # the record carries the spans; a restore brings them back as the committed images
    session._restore_prompt_snapshot()
    assert session.committed == list(first[:-1]) and session.committed_vision == spans_a
    # a text prompt's snapshot keeps an empty record and a text request that extends it restores
    session.snapshot = Qwen38PromptSnapshot(tuple(TURNS[:6]), None, "chunked")
    session.committed = TURNS[:8]
    assert session.reusable_prefix(TURNS[:6] + TURNS[9:12]) == (6, "snapshot")
    assert session.snapshot.vision == ()
