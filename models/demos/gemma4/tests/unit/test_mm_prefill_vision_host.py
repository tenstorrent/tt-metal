# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host tests of the vision-span carrier for chunked and cached-prefix prefill (no device)."""

import pytest
import torch

from models.demos.gemma4.tt.attention import mm_prefill_vision as mv
from models.demos.gemma4.tt.attention.prefill import MASK_NEG, mm_allowed_matrix, vision_block_ids

IMG, VID, TXT = 262144, 262145, 7


def _prompt(spans):
    ids = []
    for tok, n in spans:
        ids += [tok] * n
    return torch.tensor(ids, dtype=torch.long)


def test_spans_of_a_suffix_are_absolute():
    full = _prompt([(TXT, 700), (IMG, 260), (TXT, 100), (VID, 40), (TXT, 100)])
    assert mv.vision_spans_of_prompt(full, IMG, VID) == ((700, 960), (1060, 1100))
    # the uncached suffix cut at 640 names the same positions
    assert mv.vision_spans_of_prompt(full[640:], IMG, VID, base=640) == ((700, 960), (1060, 1100))
    # audio placeholders and padding ids open no span
    assert mv.vision_spans_of_prompt(torch.zeros(64, dtype=torch.long), IMG) == ()


@pytest.mark.parametrize("window", [None, 512])
@pytest.mark.parametrize("abs_start", [0, 640, 4096 - 1024])
def test_absolute_square_matches_hf_mask_functions(window, abs_start):
    from transformers.masking_utils import (
        blockwise_overlay,
        causal_mask_function,
        or_masks,
        sliding_window_causal_mask_function,
    )

    # a 1200-token prompt with two vision runs; the square under test covers [abs_start, abs_start + 320)
    prompt = _prompt([(TXT, 700), (IMG, 260), (TXT, 100), (VID, 40), (TXT, 100)])
    spans = mv.vision_spans_of_prompt(prompt, IMG, VID)
    length = 320
    base = causal_mask_function if window is None else sliding_window_causal_mask_function(window)
    blocks = vision_block_ids(prompt, [IMG, VID]).view(1, -1)
    fn = or_masks(base, blockwise_overlay(blocks))
    ref = torch.zeros(length, length, dtype=torch.bool)
    for r in range(length):
        for c in range(length):
            q, k = abs_start + r, abs_start + c
            ref[r, c] = (
                bool(fn(0, 0, torch.tensor(q), torch.tensor(k)))
                if q < 1200 and k < 1200
                else (k <= q and (window is None or k > q - window))
            )
    got = mv.allowed_matrix_abs(abs_start, length, window, spans)
    assert torch.equal(got, ref)


def test_square_at_zero_equals_the_single_chunk_builder():
    prompt = _prompt([(TXT, 5), (IMG, 6), (TXT, 4), (VID, 3), (TXT, 2)])
    spans = mv.vision_spans_of_prompt(prompt, IMG, VID)
    blocks = vision_block_ids(prompt, [IMG, VID])
    for window in (None, 4):
        assert torch.equal(mv.allowed_matrix_abs(0, 20, window, spans), mm_allowed_matrix(blocks, 20, window))
    mask = mv.additive_mask(mv.allowed_matrix_abs(0, 20, None, spans))
    assert tuple(mask.shape) == (1, 1, 20, 20) and mask.dtype == torch.bfloat16
    neg = float(torch.tensor(MASK_NEG).to(torch.bfloat16))
    assert float(mask[0, 0, 5, 10]) == 0.0 and float(mask[0, 0, 0, 1]) == neg


def test_tail_rows_before_the_prompt_are_causal_only():
    # a [tail | chunk] square starting 64 rows before position 0 (a short first prefix): the negative rows
    # belong to no span and keep the plain causal + window rule
    spans = ((40, 60),)
    allowed = mv.allowed_matrix_abs(-64, 128, 32, spans)
    assert allowed[70, 10] is not None
    assert bool(allowed[104, 123]) and bool(allowed[123, 104])  # rows 104..123 = positions 40..59: one span
    assert not bool(allowed[70, 10]) and bool(allowed[70, 69]) and not bool(allowed[70, 71])


def test_chunk_starts_keep_the_grid_without_spans():
    assert mv.plan_chunk_starts_outside_vision_spans(0, 16384, 4096, 12288, ()) == [0, 4096, 8192, 12288]
    assert mv.plan_chunk_starts_outside_vision_spans(640, 8192, 4096, 640 + 4096, ()) == [640, 4736]


def test_a_boundary_inside_a_span_moves_down_to_the_span_start():
    # a 300-row image at [4000, 4300) crosses the 4096 boundary: chunk 1 ends at 3968, chunk 2 holds the image;
    # the grid continues from the moved boundary and a short chunk precedes the planned last start
    starts = mv.plan_chunk_starts_outside_vision_spans(0, 16384, 4096, 12288, ((4000, 4300),))
    assert starts == [0, 3968, 8064, 12160, 12288]
    # the boundary that cannot move (span starts in the chunk's first 128 rows) is kept
    starts = mv.plan_chunk_starts_outside_vision_spans(0, 16384, 4096, 12288, ((4100, 8300),))
    assert starts == [0, 4096, 8192, 12288]


def test_the_last_chunk_start_moves_when_the_last_chunk_can_absorb_the_span():
    # last chunk planned at 8192 of a 9000-token suffix; an image at [8100, 8400) crosses it
    starts = mv.plan_chunk_starts_outside_vision_spans(0, 9000, 4096, 8192, ((8100, 8400),))
    assert starts == [0, 4096, 8064]
    # a last chunk that would exceed chunk_size keeps its start
    starts = mv.plan_chunk_starts_outside_vision_spans(0, 12200, 4096, 8192, ((8100, 8400),))
    assert starts == [0, 4096, 8192]


def test_vision_row_tiles_name_the_span_end_for_vision_rows_only():
    # chunk [640, 1664) of the 1200-token prompt: image [700, 960) and video [1060, 1100)
    spans = ((700, 960), (1060, 1100))
    tiles = mv.vision_row_tiles(640, 1024, spans)
    starts = [t0 for t0, _, _ in tiles]
    assert starts == list(range(32, 320, 32)) + [416, 448]  # rows 60..319 (image) and 420..459 (video)
    t0, t1, cur = tiles[0]
    assert (t0, t1) == (32, 64) and cur[:28] == [640 + 32 + j for j in range(28)] and cur[28:] == [959] * 4
    t0, t1, cur = tiles[-1]
    assert (t0, t1) == (448, 480) and cur[:12] == [1099] * 12 and cur[12:] == [640 + 460 + j for j in range(20)]
    assert mv.vision_row_tiles(2048, 1024, spans) == []


def test_the_call_object_reports_the_chunk_and_caches_its_masks(monkeypatch):
    import ttnn

    made = []

    def from_torch(host, **kwargs):
        made.append(tuple(host.shape))
        return host

    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    vision = mv.MMPrefillVision(((700, 960),), 640, 1024, mesh_device=object())
    assert vision.in_chunk()
    assert not mv.MMPrefillVision(((700, 960),), 1024, 1024, mesh_device=object()).in_chunk()
    a = vision.mask(640, 1024, None)
    assert vision.mask(640, 1024, None) is a and vision.mask(640, 1024, 512) is not a
    assert made == [(1, 1, 1024, 1024), (1, 1, 1024, 1024)]
    assert vision.vision_row_tiles()[0][0] == 32
