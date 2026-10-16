# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The committed prefix of an image prompt as a reuse key (no device): the session keys its committed ids and its
prompt-end snapshot on the images' pad spans with their digests beside the ids, so a follow-up turn on the same image
extends (or restores) instead of resetting, a prompt with another image of the same grid (identical pad ids) resets,
a text request never extends into an image it does not carry, and the extension's forced tokens never hold a pad;
the source pins: the snapshot is taken for image prompts, the extension carries the features of its own pads."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from models.demos.blackhole.qwen38_flash_next.mrope import (
    IMAGE_TOKEN_ID,
    VISION_END_TOKEN_ID,
    VISION_START_TOKEN_ID,
    Qwen38ImageGrid,
    mrope_positions,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import (
    Qwen38ChatSession,
    Qwen38PromptSnapshot,
    vision_spans_compatible,
)
from models.demos.blackhole.qwen38_flash_next.vision_splice import Qwen38VisionPrompt

GRID = Qwen38ImageGrid(1, 8, 8)  # 16 merged tokens
TEXT = list(range(1000, 1012))


def _image_prompt(question: list[int], grid: Qwen38ImageGrid = GRID) -> list[int]:
    return TEXT[:4] + [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * grid.merged_tokens + [VISION_END_TOKEN_ID] + question


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
    first = _image_prompt(TEXT[4:8])
    vision_a = _vision(first, "A")
    # the committed state after the first turn's prefill and its reply
    session.committed = first + TEXT[8:10]
    session.committed_vision = vision_a.spans(first)
    second = first + TEXT[8:10] + TEXT[10:12]
    assert session.reusable_prefix(second, _vision(second, "A").spans(second)) == (len(session.committed), "extends")
    # the same ids with another image (the same grid renders the same pads): a reset, never a reuse
    assert session.reusable_prefix(second, _vision(second, "B").spans(second)) == (0, "reset")
    # a text prompt equal to the ids so far cannot exist without the vision inputs; a text prefix reuse is untouched
    session.committed_vision = ()
    session.committed = TEXT[:6]
    assert session.reusable_prefix(TEXT[:8]) == (6, "extends")


def test_the_prompt_end_snapshot_of_an_image_prompt_is_keyed_on_its_images_too() -> None:
    session = _session()
    first = _image_prompt(TEXT[4:8])
    spans_a = _vision(first, "A").spans(first)
    session.snapshot = Qwen38PromptSnapshot(tuple(first[:-1]), None, "chunked", spans_a)
    session.committed = first + TEXT[8:11]  # the committed state moved on (a reply); the snapshot still serves
    other = first[:-1] + [TEXT[11]] + TEXT[8:10]  # a prompt extending the snapshot's ids with another question
    assert session.reusable_prefix(other, _vision(other, "A").spans(other)) == (len(first) - 1, "snapshot")
    assert session.reusable_prefix(other, _vision(other, "B").spans(other)) == (0, "reset")
    # the record carries the spans; a restore brings them back as the committed images
    session._restore_prompt_snapshot()
    assert session.committed == list(first[:-1]) and session.committed_vision == spans_a
    # a text prompt's snapshot keeps an empty record and a text request that extends it restores as before
    session.snapshot = Qwen38PromptSnapshot(tuple(TEXT[:6]), None, "chunked")
    session.committed = TEXT[:8]
    assert session.reusable_prefix(TEXT[:6] + TEXT[9:12]) == (6, "snapshot")
    assert session.snapshot.vision == ()


def test_complete_keys_the_prefix_on_the_spans_slices_the_extensions_features_and_keeps_the_snapshot() -> None:
    source = inspect.getsource(Qwen38ChatSession.complete)
    assert "spans = () if vision is None else vision.spans(token_ids)" in source
    assert "common, reuse = self.reusable_prefix(token_ids, spans)" in source
    # the forced tokens of an extension (its alignment steps, or a short tail) never hold a pad: a reset instead
    assert "if vision_splice.image_lanes(head[:aligned] if chunkable else head):" in source
    # the extension's features are those of its own pads; the prompt-end snapshot is taken for image prompts too
    assert "vision.features[pads_before : pads_before + pads_in]," in source
    assert "self.committed_vision = spans" in source
    assert "No prompt-end snapshot of an image prompt" not in source and "capture = lambda: None" not in source
    capture = inspect.getsource(Qwen38ChatSession._capture_prompt_snapshot)
    assert "Qwen38PromptSnapshot(tuple(self.committed), self.ple_context, schedule, self.committed_vision)" in capture
    reset = inspect.getsource(Qwen38ChatSession.reset)
    assert "self.committed_vision = ()" in reset
    assert "committed_vision_digest" not in inspect.getsource(session_module)
