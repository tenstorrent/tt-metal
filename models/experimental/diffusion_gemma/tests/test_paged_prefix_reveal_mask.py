# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host (no-device) tests for the paged-prefix Phase-1 reveal mask.

Guards the two load-bearing invariants of ``build_canvas_reveal_denoise_mask`` before
any device work (see ``doc/optimize_perf/paged_prefix_denoise_design.md``):

  (1) NO LEAK — uncommitted prefix slots ``[prompt_len:p_max]`` are ALWAYS masked, at
      every ``prompt_len`` and every ``p_max``.
  (2) BIT-EXACT-TO-GOLDEN — the Phase-1 reveal mask, restricted to the committed key
      columns ``[0:prompt_len] ++ canvas``, equals the current all-attend golden
      (``build_canvas_denoise_mask``), so Phase-1 does not change any committed decision.
  (3) FIXED SHAPE — the mask shape is ``[C, p_max+C]`` independent of ``prompt_len``
      (the property that makes the trace capture-once/replay-many).
"""

from __future__ import annotations


import pytest
import torch

from models.experimental.diffusion_gemma.reference.attention_mask import (
    build_canvas_denoise_mask,
    build_canvas_reveal_denoise_mask,
)

NEG = float("-inf")
CANVAS = 256  # DG output block granularity


def _committed_columns(prompt_len: int, p_max: int, canvas_len: int) -> torch.Tensor:
    """Indices into the [p_max+C] key axis that correspond to committed keys."""
    prefix = torch.arange(prompt_len)  # committed prefix slots 0..prompt_len-1
    canvas = p_max + torch.arange(canvas_len)  # canvas columns live at [p_max:p_max+C]
    return torch.cat([prefix, canvas])


@pytest.mark.parametrize("prompt_len", [0, 32, 256, 288, 1024, 4096])
@pytest.mark.parametrize("p_max", [4096, 8192])
def test_fixed_shape_independent_of_prompt_len(prompt_len, p_max):
    mask = build_canvas_reveal_denoise_mask(prompt_len, CANVAS, p_max, layer_type="full_attention")
    assert tuple(mask.shape) == (CANVAS, p_max + CANVAS)


@pytest.mark.parametrize("prompt_len", [0, 32, 256, 288, 544, 1024, 4096])
@pytest.mark.parametrize("layer_type", ["full_attention", "sliding_attention"])
@pytest.mark.parametrize("enforce_window", [False, True])
def test_no_leak_uncommitted_prefix_always_masked(prompt_len, layer_type, enforce_window):
    p_max = 8192
    mask = build_canvas_reveal_denoise_mask(
        prompt_len,
        CANVAS,
        p_max,
        layer_type=layer_type,
        sliding_window=1024,
        enforce_sliding_window=enforce_window,
    )
    # Every uncommitted prefix column [prompt_len:p_max] must be -inf for every canvas row.
    uncommitted = mask[:, prompt_len:p_max]
    assert (
        torch.isinf(uncommitted).all() and (uncommitted < 0).all()
    ), f"uncommitted prefix leaked at prompt_len={prompt_len} {layer_type} window={enforce_window}"


@pytest.mark.parametrize("prompt_len", [32, 256, 288, 544, 1024, 2048])
def test_phase1_full_attn_bit_exact_to_allattend_golden(prompt_len):
    """Phase-1 full-attn: committed columns must be exactly the all-attend golden (zeros)."""
    p_max = 8192
    reveal = build_canvas_reveal_denoise_mask(prompt_len, CANVAS, p_max, layer_type="full_attention")
    golden = build_canvas_denoise_mask(prompt_len, CANVAS, layer_type="full_attention")  # [C, prompt_len+C], all 0
    cols = _committed_columns(prompt_len, p_max, CANVAS)
    got = reveal[:, cols]
    assert got.shape == golden.shape
    assert torch.equal(got, golden), "Phase-1 full-attn reveal diverges from all-attend golden on committed span"


@pytest.mark.parametrize("prompt_len", [1024, 1281, 2048, 4096])
def test_phase2_sliding_matches_golden_on_committed_span(prompt_len):
    """Phase-2 sliding: committed columns must match the HF bidirectional-window golden."""
    p_max = 8192
    W = 1024
    reveal = build_canvas_reveal_denoise_mask(
        prompt_len, CANVAS, p_max, layer_type="sliding_attention", sliding_window=W, enforce_sliding_window=True
    )
    golden = build_canvas_denoise_mask(prompt_len, CANVAS, layer_type="sliding_attention", sliding_window=W)
    cols = _committed_columns(prompt_len, p_max, CANVAS)
    got = reveal[:, cols]
    # Compare mask topology (attend vs masked) rather than raw -inf bit patterns.
    assert torch.equal(
        torch.isfinite(got), torch.isfinite(golden)
    ), f"Phase-2 sliding visibility diverges from HF golden at prompt_len={prompt_len}"


def test_softmax_invariance_masked_tail_is_noop():
    """The -inf tail must contribute exactly 0 to softmax (bit-exact no-op vs the committed-only mask)."""
    torch.manual_seed(0)
    prompt_len, p_max = 288, 4096
    H, C, hd = 2, CANVAS, 64
    total = p_max + C
    scores = torch.randn(H, C, total, dtype=torch.float64)
    mask = build_canvas_reveal_denoise_mask(prompt_len, C, p_max, layer_type="full_attention", dtype=torch.float64)
    # Full masked softmax over the fixed span.
    full = torch.softmax(scores + mask.unsqueeze(0), dim=-1)
    # Reference: softmax over ONLY the committed columns.
    cols = _committed_columns(prompt_len, p_max, C)
    ref = torch.zeros_like(full)
    ref[:, :, cols] = torch.softmax(scores[:, :, cols], dim=-1)
    assert torch.allclose(full, ref, atol=1e-12), "masked tail is not a softmax no-op"


def test_p_max_must_not_shrink_below_prompt_len(expect_error):
    with expect_error(ValueError):
        build_canvas_reveal_denoise_mask(4096, CANVAS, 2048, layer_type="full_attention")
