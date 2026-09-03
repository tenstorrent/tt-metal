# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the canvas denoise attention-mask geometry (#47462).

Pure torch. The canonical DiffusionGemma decoder is **fully bidirectional for
every layer type** (modeling_diffusion_gemma.py:1399-1438) — the local window
shapes offsets, not visibility — so the denoise mask is all-attend. The
``local_window=True`` bake is non-canonical (exercises the ttnn SDPA windowed-mask
path only) and is tested as such.
"""

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


def test_denoise_mask_is_fully_bidirectional_by_default():
    # Canonical: BOTH full and sliding layers attend fully to prompt + canvas.
    mask = build_canvas_denoise_mask(prompt_len=20, canvas_len=8)
    assert mask.shape == (8, 28)
    assert torch.all(mask == 0)  # all-attend — no window gates visibility in the decoder


# --- below: the NON-canonical local_window bake (ttnn SDPA windowed-mask path only) ---


def test_local_window_requires_window_half(expect_error):
    with expect_error(ValueError):
        build_canvas_denoise_mask(prompt_len=4, canvas_len=4, local_window=True)


def test_local_window_is_symmetric_and_centered():
    prompt_len, canvas_len, w = 10, 12, 3
    mask = build_canvas_denoise_mask(prompt_len, canvas_len, local_window=True, window_half=w)
    attend = _attend(mask)
    assert mask.shape == (canvas_len, prompt_len + canvas_len)

    q_abs = canvas_positions(prompt_len, canvas_len)
    for i in range(canvas_len):
        keys = attend[i].nonzero(as_tuple=True)[0]
        lo, hi = int(keys.min()), int(keys.max())
        assert lo == max(0, int(q_abs[i]) - w)
        assert hi == min(prompt_len + canvas_len - 1, int(q_abs[i]) + w)
        assert bool(attend[i, int(q_abs[i])])  # attends to itself


def test_local_window_covers_prompt_tail_for_early_canvas():
    prompt_len, canvas_len, w = 10, 12, 3
    attend = _attend(build_canvas_denoise_mask(prompt_len, canvas_len, local_window=True, window_half=w))
    assert attend[0, 7] and attend[0, 8] and attend[0, 9]
    assert not attend[0, 6]
    deep = w + 1
    assert not attend[deep, prompt_len - 1]


def test_local_window_prompt_fully_visible_variant():
    prompt_len, canvas_len, w = 10, 12, 2
    attend = _attend(
        build_canvas_denoise_mask(prompt_len, canvas_len, local_window=True, window_half=w, prompt_fully_visible=True)
    )
    assert torch.all(attend[:, :prompt_len])
    deep = canvas_len - 1
    assert not attend[deep, prompt_len + 0]
    assert attend[deep, prompt_len + deep]
    assert attend[deep, prompt_len + deep - w]


def test_local_window_inclusive_vs_exclusive_boundary():
    prompt_len, canvas_len, w = 5, 6, 2
    inc = _attend(build_canvas_denoise_mask(prompt_len, canvas_len, local_window=True, window_half=w, inclusive=True))
    exc = _attend(build_canvas_denoise_mask(prompt_len, canvas_len, local_window=True, window_half=w, inclusive=False))
    assert exc.sum() < inc.sum()
    assert torch.all(inc | ~exc)


def test_additive_mask_values_are_zero_or_neg_inf():
    mask = build_canvas_denoise_mask(8, 8, local_window=True, window_half=2)
    vals = torch.unique(mask)
    assert torch.all((vals == 0) | torch.isinf(vals) & (vals < 0))
