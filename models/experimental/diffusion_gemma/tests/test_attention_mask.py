# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the canvas denoise attention-mask geometry (#47462).

Pure torch — pins the [C, P+C] mask the device SDPA path bakes in.
"""

import pytest
import torch

from models.experimental.diffusion_gemma.reference.attention_mask import (
    build_canvas_denoise_mask,
    canvas_positions,
)


def _attend(mask):
    """Boolean attend matrix from an additive (0 / -inf) mask."""
    return mask == 0


def test_canvas_positions_offset_by_prompt_len():
    pos = canvas_positions(prompt_len=100, canvas_len=8)
    assert torch.equal(pos, torch.arange(100, 108))


def test_full_attention_is_fully_visible():
    mask = build_canvas_denoise_mask(prompt_len=20, canvas_len=8, is_sliding=False)
    assert mask.shape == (8, 28)
    assert torch.all(mask == 0)  # canvas sees all prompt + all canvas (bidirectional)


def test_sliding_requires_window_half():
    with pytest.raises(ValueError):
        build_canvas_denoise_mask(prompt_len=4, canvas_len=4, is_sliding=True)


def test_sliding_window_is_symmetric_and_centered():
    prompt_len, canvas_len, w = 10, 12, 3
    mask = build_canvas_denoise_mask(prompt_len, canvas_len, is_sliding=True, window_half=w)
    attend = _attend(mask)
    assert mask.shape == (canvas_len, prompt_len + canvas_len)

    q_abs = canvas_positions(prompt_len, canvas_len)
    for i in range(canvas_len):
        keys = attend[i].nonzero(as_tuple=True)[0]
        lo, hi = int(keys.min()), int(keys.max())
        # window centered on the query's absolute position, half-width w (inclusive)
        assert lo == max(0, int(q_abs[i]) - w)
        assert hi == min(prompt_len + canvas_len - 1, int(q_abs[i]) + w)
        # symmetric around the query where not clipped by sequence bounds
        assert bool(attend[i, int(q_abs[i])])  # attends to itself


def test_sliding_window_covers_prompt_tail_for_early_canvas():
    # default (prompt not forced visible): the symmetric window over absolute
    # positions naturally reaches back into the prompt tail for early canvas pos.
    prompt_len, canvas_len, w = 10, 12, 3
    attend = _attend(build_canvas_denoise_mask(prompt_len, canvas_len, is_sliding=True, window_half=w))
    # canvas pos 0 (abs=10) attends prompt keys 7,8,9 (and canvas 0..3 -> abs 10..13)
    assert attend[0, 7] and attend[0, 8] and attend[0, 9]
    assert not attend[0, 6]
    # a canvas pos beyond the window from the prompt sees no prompt key
    deep = w + 1  # abs = prompt_len + w + 1, distance to last prompt key (9) = w+2 > w
    assert not attend[deep, prompt_len - 1]


def test_prompt_fully_visible_variant():
    prompt_len, canvas_len, w = 10, 12, 2
    attend = _attend(
        build_canvas_denoise_mask(prompt_len, canvas_len, is_sliding=True, window_half=w, prompt_fully_visible=True)
    )
    # every canvas query sees every prompt key...
    assert torch.all(attend[:, :prompt_len])
    # ...but canvas<->canvas is still windowed: the last query is far from canvas key 0
    deep = canvas_len - 1
    assert not attend[deep, prompt_len + 0]
    # and it still attends within its own canvas window
    assert attend[deep, prompt_len + deep]  # itself
    assert attend[deep, prompt_len + deep - w]  # w to the left, inside the window


def test_inclusive_vs_exclusive_boundary():
    prompt_len, canvas_len, w = 5, 6, 2
    inc = _attend(build_canvas_denoise_mask(prompt_len, canvas_len, is_sliding=True, window_half=w, inclusive=True))
    exc = _attend(build_canvas_denoise_mask(prompt_len, canvas_len, is_sliding=True, window_half=w, inclusive=False))
    # exclusive attends to strictly fewer keys (drops the |dist|==w boundary)
    assert exc.sum() < inc.sum()
    assert torch.all(inc | ~exc)  # exclusive set is a subset of inclusive


def test_additive_mask_values_are_zero_or_neg_inf():
    mask = build_canvas_denoise_mask(8, 8, is_sliding=True, window_half=2)
    vals = torch.unique(mask)
    assert torch.all((vals == 0) | torch.isinf(vals) & (vals < 0))
