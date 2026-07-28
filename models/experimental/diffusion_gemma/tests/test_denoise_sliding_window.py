# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer sliding-window reveal masks for denoise (#51080 roadmap item 3).

HF's sliding layers retain only the last ``sliding_window - 1`` committed tokens, so TT's
all-attend denoise currently attends keys HF does not have on 25 of 30 layers. Enforcing the
retention gives each layer TYPE its own reveal-mask content while keeping ONE shape, so every
captured trace stays valid.

The regime split is what makes this gateable:

* ``prompt_len <= sliding_window - 1`` — the window cannot bind, so the sliding mask is
  IDENTICAL to the full mask. Enabling the flag there is bit-exact.
* ``prompt_len > sliding_window - 1`` — the masks differ; that is the decision-changing regime
  whose gate is a decision-agreement run against fp32 HF.

CPU-only; the mask geometry is pure torch.
"""

import pytest
import torch
from types import SimpleNamespace

from models.experimental.diffusion_gemma.reference.attention_mask import build_canvas_reveal_denoise_mask
from models.experimental.diffusion_gemma.tt import denoise_forward as DF

W = 8
CANVAS = 4
P_MAX = 32


def _attend(mask):
    return mask == 0


def _reveal(prompt_len, *, layer_type, enforce):
    return build_canvas_reveal_denoise_mask(
        prompt_len,
        CANVAS,
        P_MAX,
        layer_type=layer_type,
        sliding_window=W,
        enforce_sliding_window=enforce,
    )


def test_flag_defaults_on_and_zero_opts_out(monkeypatch):
    """Gated by the GPQA decision-agreement run, so retention is now the default.

    56 of 64 shipped-config collapses were at or after the block whose committed prefix crosses
    W-1; below that the mask is bit-identical, so the flip only changes the regime that was wrong.
    """
    monkeypatch.delenv("DG_DENOISE_SLIDING_WINDOW", raising=False)
    assert DF.denoise_sliding_window_enabled() is True
    monkeypatch.setenv("DG_DENOISE_SLIDING_WINDOW", "0")
    assert DF.denoise_sliding_window_enabled() is False, "the maskless path must stay reachable"
    monkeypatch.setenv("DG_DENOISE_SLIDING_WINDOW", "1")
    assert DF.denoise_sliding_window_enabled() is True


@pytest.mark.parametrize("prompt_len", [0, W // 2, W - 1])
def test_below_the_window_the_sliding_mask_equals_the_full_mask(prompt_len):
    """Bit-exact regime: nothing has been evicted yet, so enforcing changes nothing."""
    full = _reveal(prompt_len, layer_type="full_attention", enforce=False)
    sliding = _reveal(prompt_len, layer_type="sliding_attention", enforce=True)
    assert torch.equal(full, sliding), f"prompt_len={prompt_len} must be unchanged by the window"


@pytest.mark.parametrize("prompt_len", [W, W + 1, 3 * W])
def test_above_the_window_only_the_retained_committed_tail_is_attended(prompt_len):
    sliding = _attend(_reveal(prompt_len, layer_type="sliding_attention", enforce=True))
    full = _attend(_reveal(prompt_len, layer_type="full_attention", enforce=False))
    assert not torch.equal(sliding, full), "the window must bind above W-1"

    keep_from = prompt_len - (W - 1)
    for j in range(P_MAX):
        expected = keep_from <= j < prompt_len
        assert bool(sliding[0, j]) is expected, f"prefix col {j} (prompt_len={prompt_len})"
    # Canvas is always fully visible, and there is no query dependence.
    assert bool(sliding[:, P_MAX:].all())
    for row in range(1, CANVAS):
        assert torch.equal(sliding[row], sliding[0])


def test_full_attention_layers_are_unaffected_by_the_window():
    prompt_len = 3 * W
    a = _reveal(prompt_len, layer_type="full_attention", enforce=True)
    b = _reveal(prompt_len, layer_type="full_attention", enforce=False)
    assert torch.equal(a, b), "enforce_sliding_window must be inert on full_attention layers"


# --------------------------------------------------------------------------------------------
# adapter-level per-layer dispatch


class _FakeBuf:
    def __init__(self, tag):
        self.tag = tag
        self.freed = False

    def deallocate(self, force=True):
        self.freed = True


def _adapter(layer_types, *, enforce):
    """Minimal adapter shell exercising only the reveal-mask buffer machinery."""
    from types import SimpleNamespace

    adapter = object.__new__(DF.DenoiseLogitsAdapter)
    adapter.tt_model = SimpleNamespace(
        layers=[
            SimpleNamespace(self_attn=SimpleNamespace(config=SimpleNamespace(sliding_window=W))) for _ in layer_types
        ],
        hf_config=SimpleNamespace(layer_types=list(layer_types), sliding_window=W),
        mesh_device=None,
    )
    adapter._reveal_canvas_len = CANVAS
    adapter._reveal_p_max = P_MAX
    adapter._reveal_enforce_window = enforce
    adapter._reveal_mask_bufs = {}
    adapter._reveal_mask_buf = None
    adapter.use_reveal_mask = False
    return adapter


DG_LAYER_TYPES = ["sliding_attention"] * 5 + ["full_attention"]


def test_one_mask_when_the_window_is_disabled():
    adapter = _adapter(DG_LAYER_TYPES, enforce=False)
    assert adapter._reveal_mask_layer_types() == ("full_attention",)


def test_two_masks_when_enabled_and_dispatch_is_per_layer_type():
    adapter = _adapter(DG_LAYER_TYPES, enforce=True)
    assert set(adapter._reveal_mask_layer_types()) == {"sliding_attention", "full_attention"}

    adapter._reveal_mask_bufs = {"sliding_attention": _FakeBuf("slide"), "full_attention": _FakeBuf("full")}
    for layer_idx, layer_type in enumerate(DG_LAYER_TYPES):
        expected = "slide" if layer_type == "sliding_attention" else "full"
        assert adapter._reveal_mask_provider(layer_idx).tag == expected, f"layer {layer_idx}"


def test_single_mask_is_shared_by_every_layer_when_disabled():
    adapter = _adapter(DG_LAYER_TYPES, enforce=False)
    only = _FakeBuf("full")
    adapter._reveal_mask_bufs = {"full_attention": only}
    for layer_idx in range(len(DG_LAYER_TYPES)):
        assert adapter._reveal_mask_provider(layer_idx) is only


def test_per_block_update_rebuilds_every_mask_at_the_new_committed_len(monkeypatch):
    """Both buffers must be refreshed per block, since the retained window slides with the prefix.

    A frozen sliding buffer would also suppress the late-block collapse, by over-restricting
    attention rather than matching HF, and an end-to-end run could not distinguish the two.
    """
    adapter = _adapter(DG_LAYER_TYPES, enforce=True)
    adapter.use_reveal_mask = True
    adapter._reveal_mask_bufs = {
        "sliding_attention": _FakeBuf("slide"),
        "full_attention": _FakeBuf("full"),
    }

    built = []

    def recording_build(self, prompt_len, layer_type="full_attention"):
        built.append((int(prompt_len), layer_type))
        return _FakeBuf(f"fresh-{layer_type}")

    monkeypatch.setattr(DF.DenoiseLogitsAdapter, "_build_reveal_mask_device", recording_build)
    monkeypatch.setattr(DF.ttnn, "copy", lambda fresh, buf: None)

    # update_reveal_mask_buffer enforces the product's 32-tile alignment on the committed length,
    # so this uses real tile multiples rather than the toy CANVAS used for mask geometry above.
    committed = [32 * n for n in (1, 2, 3, 4)]  # one tile-aligned commit per block
    for prompt_len in committed:
        adapter.update_reveal_mask_buffer(prompt_len)

    for prompt_len in committed:
        for layer_type in ("sliding_attention", "full_attention"):
            assert (prompt_len, layer_type) in built, f"{layer_type} not rebuilt at prompt_len={prompt_len}"
    assert len(built) == 2 * len(committed), f"expected one rebuild per buffer per block, got {built}"


def test_release_frees_every_mask_and_clears_state():
    adapter = _adapter(DG_LAYER_TYPES, enforce=True)
    bufs = {"sliding_attention": _FakeBuf("slide"), "full_attention": _FakeBuf("full")}
    adapter._reveal_mask_bufs = dict(bufs)
    adapter.use_reveal_mask = True

    adapter.release_reveal_mask_buffers()

    assert all(b.freed for b in bufs.values())
    assert adapter._reveal_mask_bufs == {}
    assert adapter._reveal_mask_buf is None
    assert adapter.use_reveal_mask is False


def test_sliding_layer_needs_mask_threshold_is_window_minus_one():
    """The old threshold came from the staircase and included canvas_len; it must not."""
    assert DF._sliding_layer_needs_denoise_mask(W - 1, CANVAS, W) is False
    assert DF._sliding_layer_needs_denoise_mask(W, CANVAS, W) is True
    # canvas_len must not influence the predicate
    for canvas in (1, 256, 4096):
        assert DF._sliding_layer_needs_denoise_mask(W - 1, canvas, W) is False


# --------------------------------------------------------------------------------------------
# bounded per-layer sliding span (#51080 item 3, perf half)


def test_span_gate_requires_the_retention_mask(monkeypatch):
    """A bounded read without the retention mask would CHANGE visibility, not implement it."""
    monkeypatch.setenv("DG_DENOISE_SLIDING_SPAN", "1")
    # Explicit "0", not delenv: the retention mask defaults ON now, so unsetting would satisfy the
    # dependency instead of removing it, and the test would stop testing anything.
    monkeypatch.setenv("DG_DENOISE_SLIDING_WINDOW", "0")
    assert DF.denoise_sliding_span_enabled() is False, "span must not engage without the window mask"
    monkeypatch.setenv("DG_DENOISE_SLIDING_WINDOW", "1")
    assert DF.denoise_sliding_span_enabled() is True


def test_read_span_is_tile_aligned_and_capped_by_pmax():
    assert DF.sliding_read_span(1024, 4096) == 1024
    assert DF.sliding_read_span(1000, 4096) == 1024  # rounded up to a tile
    assert DF.sliding_read_span(1024, 512) == 512  # never exceed the reveal span
    assert DF.sliding_read_span(8, 4096) % 32 == 0


def test_read_offset_tracks_the_committed_tail_and_stays_tile_aligned():
    span, p_max = 1024, 4096
    assert DF.sliding_read_offset(0, span, p_max) == 0
    assert DF.sliding_read_offset(1024, span, p_max) == 0
    assert DF.sliding_read_offset(1056, span, p_max) == 32
    # Clamped so the window never runs past the reveal span.
    assert DF.sliding_read_offset(4096, span, p_max) == p_max - span
    for P in range(0, 4096, 32):
        assert DF.sliding_read_offset(P, span, p_max) % 32 == 0


@pytest.mark.parametrize("prompt_len", [0, W, 2 * W, 3 * W, P_MAX])  # all <= P_MAX
def test_bounded_window_mask_matches_the_full_span_mask_on_shared_columns(prompt_len):
    """The bounded read + its mask must expose EXACTLY the keys the full-span path exposes.

    This is the equivalence that makes the perf half safe once the fidelity half is in: the
    bounded read simply omits columns the full-span mask had already set to NEG.
    """
    from models.experimental.diffusion_gemma.reference.attention_mask import (
        build_canvas_reveal_denoise_window_mask,
    )

    span = W  # tile-aligned stand-in for 1024
    lo = max(0, prompt_len - span)
    full = _attend(_reveal(prompt_len, layer_type="sliding_attention", enforce=True))
    bounded = build_canvas_reveal_denoise_window_mask(prompt_len, CANVAS, span, lo, sliding_window=W) == 0

    # Absolute positions visible under each construction must be identical.
    full_visible = {j for j in range(P_MAX) if bool(full[0, j])}
    bounded_visible = {lo + r for r in range(span) if bool(bounded[0, r])}
    assert bounded_visible == full_visible, f"prompt_len={prompt_len}"
    # Canvas fully visible, no query dependence.
    assert bool(bounded[:, span:].all())
    for row in range(1, CANVAS):
        assert torch.equal(bounded[row], bounded[0])


def test_bounded_mask_is_prompt_independent_in_steady_state():
    """Once the window binds, the bounded mask stops changing between blocks."""
    from models.experimental.diffusion_gemma.reference.attention_mask import (
        build_canvas_reveal_denoise_window_mask,
    )

    span = W
    masks = []
    for prompt_len in (3 * W, 4 * W, 5 * W):
        lo = max(0, prompt_len - span)
        masks.append(build_canvas_reveal_denoise_window_mask(prompt_len, CANVAS, span, lo, sliding_window=W))
    assert all(torch.equal(masks[0], m) for m in masks[1:])


def test_per_layer_ownership_never_frees_a_window_buffer():
    """A windowed layer hands back a persistent buffer; freeing it corrupts later blocks."""
    reader = object.__new__(DF.MutablePrefixKVReader)
    reader.borrow_full_span = False  # full layers would be OWNED clones
    reader.seq_len_start = 0
    reader.prompt_len = 64
    reader.read_span = P_MAX
    reader.tt_model = None
    reader._window_bufs = {0: ("k", "v"), 2: ("k", "v")}
    reader._window_lo = {0: 0, 2: 0}
    reader.window_layers = {0: W, 2: W}

    # Windowed layers: never owned, even though the reader's global flag says owned.
    assert reader.owns_result_for(0) is False
    assert reader.owns_result_for(2) is False
    # Non-windowed layer keeps the global contract (here: owned, because borrow is off).
    assert reader.owns_result_for(1) is True
    assert DF._prompt_source_is_owned(reader, 0) is False
    assert DF._prompt_source_is_owned(reader, 1) is True


def test_prompt_source_is_owned_falls_back_for_plain_sources():
    """Sources without per-layer resolution keep the historic contract."""
    assert DF._prompt_source_is_owned(lambda i: None, 3) is True
    assert DF._prompt_source_is_owned(SimpleNamespace(owns_result=False), 3) is False
    assert DF._prompt_source_is_owned(SimpleNamespace(owns_result=True), 3) is True


# --------------------------------------------------------------------------------------------
# canvas-tail workspace (#51080 item 4)


def test_workspace_layers_cover_every_layer_with_its_own_span():
    """Item 4 applies to all layers: bounded span for sliding, p_max for full."""
    p_max, sliding_span = 4096, 1024
    layer_types = ["sliding_attention"] * 25 + ["full_attention"] * 5
    window_layers = {i: sliding_span for i, t in enumerate(layer_types) if t == "sliding_attention"}
    # mirrors the traced-path wiring
    for i, t in enumerate(layer_types):
        window_layers.setdefault(i, sliding_span)
        if t != "sliding_attention":
            window_layers[i] = p_max

    assert len(window_layers) == 30
    assert sum(1 for v in window_layers.values() if v == sliding_span) == 25
    assert sum(1 for v in window_layers.values() if v == p_max) == 5
