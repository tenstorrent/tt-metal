# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only regression test of the vision crash "image span of N tokens at S is not contained in the E prompt tokens given" (server died: request A with an image, request B with an image, B again).
The repeat finds the image's encoder output in vLLM's encoder cache, so the scheduler does not roll the piece back before the image (``disable_chunked_mm_input`` only acts while the image still has to
be encoded) and the first-step cap / budget cuts the piece at 1536 inside the span [1360, 1566). ``cut_end`` shortens such a piece to the chunk boundary below the span; ``locate_spans`` (what crashed)
stays strict."""

import torch

from models.demos.blackhole.deepseek_v41_flash.tt import vision_state as VSN

IMG = 129264


def prompt(text_before, span, text_after):
    return torch.cat(
        [torch.randint(1, 1000, (text_before,)), torch.full((span,), IMG), torch.randint(1, 1000, (text_after,))]
    )


def test_locate_spans_still_refuses_a_cut_span():
    t = prompt(1360, 206, 14)
    try:
        VSN.locate_spans(t[:1536], [206], IMG)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_piece_ending_inside_the_span_is_cut_before_it():
    t = prompt(1360, 206, 14)  # 1580 tokens, span [1360, 1566)
    assert VSN.cut_end(t, [206], IMG, 1536, align=512) == 1024  # chunk boundary below the span start
    assert VSN.cut_end(t, [206], IMG, 1536, align=1) == 1360
    assert VSN.locate_spans(t[: VSN.cut_end(t, [206], IMG, 1536, align=512)], [206], IMG) == []


def test_pieces_that_do_not_cut_a_span_are_unchanged():
    t = prompt(1360, 206, 14)
    assert VSN.cut_end(t, [206], IMG, 1580, align=512) == 1580  # the final piece holds the whole span
    assert VSN.cut_end(t, [206], IMG, 1566, align=512) == 1566  # ends exactly at the span end
    assert VSN.cut_end(t, [206], IMG, 1024, align=512) == 1024  # before the span
    assert VSN.cut_end(t, [206], IMG, 1360, align=512) == 1360


def test_text_only_and_second_image():
    t = torch.randint(1, 1000, (700,))
    assert VSN.cut_end(t, [], IMG, 512, align=512) == 512
    t2 = torch.cat(
        [prompt(100, 50, 100), torch.full((60,), IMG), torch.randint(1, 1000, (30,))]
    )  # spans [100,150) and [250,310)
    assert VSN.cut_end(t2, [50, 60], IMG, 300, align=1) == 250
    assert VSN.cut_end(t2, [50, 60], IMG, 300, align=128) == 128
    assert VSN.cut_end(t2, [50, 60], IMG, 140, align=1) == 100
