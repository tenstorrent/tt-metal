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
    build_canvas_reveal_denoise_window_mask,
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


# --------------------------------------------------------------------------------------------
# hidden prefix span: the prefill pad slots
#
# Prefill right-pads the prompt to a tile multiple and writes K/V for the pad tokens, while the
# reveal predicate uses the PADDED length -- so those garbage keys are revealed, and they sit
# immediately before the canvas. Injecting that geometry into the reference (seeded canvas,
# otherwise identical) took q096 to the 48-step cap and q106/q095 to 35 steps; hiding the pads
# restored 20/12/11, i.e. baseline. See doc/decision_fidelity/device_gumbel_restored.md section 16.

PAD_P_MAX = 4096  # a reveal span wide enough that the pad slots sit well inside it
PAD_W = 1024  # the real sliding window, for the composition test


def test_hidden_span_hides_exactly_those_slots():
    """270 real tokens padded to 288 inside a 320-slot span: only 270..287 change."""
    today = build_canvas_reveal_denoise_mask(288, CANVAS, 320)
    fixed = build_canvas_reveal_denoise_mask(288, CANVAS, 320, hidden_prefix_span=(270, 288))
    assert (fixed[:, :270] == 0).all(), "real prompt keys must stay revealed"
    assert (fixed[:, 270:288] == NEG).all(), "pad slots must be hidden"
    assert torch.equal(fixed[:, 288:], today[:, 288:]), "uncommitted tail and canvas must be untouched"
    assert (fixed[:, 320:] == 0).all(), "canvas columns are always revealed"


def test_hidden_span_is_inert_when_absent():
    """The mechanism must not change any existing caller until one passes a span."""
    for prompt_len in (0, 32, 288, PAD_P_MAX):
        assert torch.equal(
            build_canvas_reveal_denoise_mask(prompt_len, CANVAS, PAD_P_MAX),
            build_canvas_reveal_denoise_mask(prompt_len, CANVAS, PAD_P_MAX, hidden_prefix_span=None),
        ), f"prompt_len={prompt_len}"


def test_empty_hidden_span_hides_nothing():
    """An aligned prompt has no pad slots, so lo == hi, which must be a no-op rather than an error."""
    assert torch.equal(
        build_canvas_reveal_denoise_mask(32, CANVAS, PAD_P_MAX),
        build_canvas_reveal_denoise_mask(32, CANVAS, PAD_P_MAX, hidden_prefix_span=(32, 32)),
    )


@pytest.mark.parametrize("span", [(-1, 8), (8, 4), (0, PAD_P_MAX + 1)])
def test_hidden_span_bounds_are_validated(span, expect_error):
    with expect_error(ValueError, match="hidden_prefix_span"):
        build_canvas_reveal_denoise_mask(PAD_P_MAX, CANVAS, PAD_P_MAX, hidden_prefix_span=span)


def test_hidden_span_composes_with_the_retention_window():
    """Both predicates are per-key, so hiding pads and enforcing retention must intersect.

    A key is attended only if it is committed AND not a pad AND still retained, so the pad slots
    stay hidden even when they fall inside the retained window.
    """
    prompt_len = PAD_P_MAX
    lo, hi = prompt_len - 4, prompt_len  # pads at the very end, inside any retained window
    windowed = build_canvas_reveal_denoise_mask(
        prompt_len, CANVAS, PAD_P_MAX, layer_type="sliding_attention", sliding_window=PAD_W, enforce_sliding_window=True
    )
    both = build_canvas_reveal_denoise_mask(
        prompt_len,
        CANVAS,
        PAD_P_MAX,
        layer_type="sliding_attention",
        sliding_window=PAD_W,
        enforce_sliding_window=True,
        hidden_prefix_span=(lo, hi),
    )
    assert (windowed[:, lo:hi] == 0).all(), "precondition: those keys are retained without the span"
    assert (both[:, lo:hi] == NEG).all(), "pads must be hidden even inside the retained window"
    assert torch.equal(both[:, :lo], windowed[:, :lo]), "keys outside the span must be unchanged"


# ---------------------------------------------------------------------------------------------
# Bounded sliding span + hidden pads. This combination used to raise NotImplementedError; the
# bounded builder now takes the same ABSOLUTE-position span, because its key axis already carries
# absolute positions (prefix column r -> lo + r) and needs no column arithmetic.
# ---------------------------------------------------------------------------------------------

SPAN = 1024  # tile-aligned read span for a sliding layer, sliding_read_span(1024, p_max)


def test_window_hidden_span_hides_exactly_the_pad_columns():
    """Pads at absolute 270..287 with the window still anchored at lo=0."""
    prompt_len, lo = 288, 0
    plain = build_canvas_reveal_denoise_window_mask(prompt_len, CANVAS, SPAN, lo, sliding_window=PAD_W)
    fixed = build_canvas_reveal_denoise_window_mask(
        prompt_len, CANVAS, SPAN, lo, sliding_window=PAD_W, hidden_prefix_span=(270, 288)
    )
    assert (plain[:, 270:288] == 0).all(), "precondition: those columns are attended without the span"
    assert (fixed[:, 270:288] == NEG).all(), "pad columns must be hidden"
    assert torch.equal(fixed[:, :270], plain[:, :270]), "real prompt keys must be unchanged"
    assert torch.equal(fixed[:, 288:], plain[:, 288:]), "everything past the pads must be unchanged"


def test_window_hidden_span_is_a_noop_once_the_window_scrolls_past_the_pads():
    """The self-retiring property: pads sit at the START, so a scrolled window cannot see them.

    This is what keeps the steady-state mask prompt_len-independent — the reason the bounded read
    is worth having in the first place.
    """
    pads = (270, 288)
    prompt_len = 8192
    lo = prompt_len - SPAN  # 7168, far past the pads
    assert lo > pads[1], "precondition: the window starts after the pad span"
    assert torch.equal(
        build_canvas_reveal_denoise_window_mask(prompt_len, CANVAS, SPAN, lo, sliding_window=PAD_W),
        build_canvas_reveal_denoise_window_mask(
            prompt_len, CANVAS, SPAN, lo, sliding_window=PAD_W, hidden_prefix_span=pads
        ),
    )


def test_window_hidden_span_is_inert_when_absent_or_empty():
    for pads in (None, (288, 288)):
        assert torch.equal(
            build_canvas_reveal_denoise_window_mask(288, CANVAS, SPAN, 0, sliding_window=PAD_W),
            build_canvas_reveal_denoise_window_mask(288, CANVAS, SPAN, 0, sliding_window=PAD_W, hidden_prefix_span=pads),
        ), f"pads={pads}"


def test_window_hidden_span_partially_overlapping_the_window_hides_only_the_overlap():
    """A window that has scrolled INTO the pad span must hide the part it can see, and only that."""
    pads = (270, 288)
    lo = 280  # window covers 280.. ; pads 280..287 are visible, 270..279 are not in this window
    prompt_len = lo + SPAN
    plain = build_canvas_reveal_denoise_window_mask(prompt_len, CANVAS, SPAN, lo, sliding_window=PAD_W)
    fixed = build_canvas_reveal_denoise_window_mask(
        prompt_len, CANVAS, SPAN, lo, sliding_window=PAD_W, hidden_prefix_span=pads
    )
    hidden_cols = slice(0, 288 - lo)  # absolute 280..287 -> columns 0..7
    assert (fixed[:, hidden_cols] == NEG).all(), "the visible part of the pad span must be hidden"
    assert torch.equal(fixed[:, 288 - lo :], plain[:, 288 - lo :]), "nothing past the pads may change"


def test_window_hidden_span_is_not_bounded_to_the_window():
    """lo <= hi is the only requirement; a span outside [lo, lo+span) is a no-op, not an error.

    Bounding it to the window would reject the normal scrolled case above.
    """
    build_canvas_reveal_denoise_window_mask(8192, CANVAS, SPAN, 7168, sliding_window=PAD_W, hidden_prefix_span=(0, 32))


@pytest.mark.parametrize("pads", [(-1, 8), (8, 4)])
def test_window_hidden_span_bounds_are_validated(pads, expect_error):
    with expect_error(ValueError, match="hidden_prefix_span"):
        build_canvas_reveal_denoise_window_mask(288, CANVAS, SPAN, 0, sliding_window=PAD_W, hidden_prefix_span=pads)
