# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Critical full-canvas norm, reveal-mask, and bounded-prefix-read regressions."""

from types import SimpleNamespace

import torch

from models.experimental.diffusion_gemma.reference.attention_mask import (
    build_canvas_reveal_denoise_mask,
    build_canvas_reveal_denoise_window_mask,
)
from models.experimental.diffusion_gemma.tt import denoise_forward as DF
from models.experimental.diffusion_gemma.tt.denoise_forward import _chunked_norm_forward


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def device(self):
        return "fake-device"


def test_chunked_norm_forward_norms_the_whole_canvas_in_one_op(monkeypatch):
    seen = []

    def fake_sharded(norm, hidden, rows):
        seen.append(rows)
        return _FakeTensor(hidden.shape)

    monkeypatch.setattr(DF, "_sharded_rms_norm_rows", fake_sharded)
    norm = SimpleNamespace(with_scale=True, tt_weight=object(), eps=1e-6)
    out = _chunked_norm_forward(norm, _FakeTensor([1, 1, 96, 2816]))
    assert out.shape == [1, 1, 96, 2816]
    assert seen == [96]


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


def test_above_the_window_only_the_retained_committed_tail_is_attended():
    prompt_len = 3 * W
    sliding = _attend(_reveal(prompt_len, layer_type="sliding_attention", enforce=True))
    full = _attend(_reveal(prompt_len, layer_type="full_attention", enforce=False))
    assert not torch.equal(sliding, full)

    keep_from = prompt_len - (W - 1)
    for column in range(P_MAX):
        assert bool(sliding[0, column]) is (keep_from <= column < prompt_len)
    assert bool(sliding[:, P_MAX:].all())
    for row in range(1, CANVAS):
        assert torch.equal(sliding[row], sliding[0])


def test_full_attention_layers_are_unaffected_by_the_window():
    prompt_len = 3 * W
    enforced = _reveal(prompt_len, layer_type="full_attention", enforce=True)
    unbounded = _reveal(prompt_len, layer_type="full_attention", enforce=False)
    assert torch.equal(enforced, unbounded)


class _FakeBuf:
    def __init__(self, tag):
        self.tag = tag

    def deallocate(self, force=True):
        pass


DG_LAYER_TYPES = ["sliding_attention"] * 5 + ["full_attention"]


def _adapter(layer_types, *, enforce):
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


def test_per_block_update_rebuilds_every_mask_at_the_new_committed_len(monkeypatch):
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
    committed = [32, 64]
    for prompt_len in committed:
        adapter.update_reveal_mask_buffer(prompt_len)
    assert built == [
        (32, "sliding_attention"),
        (32, "full_attention"),
        (64, "sliding_attention"),
        (64, "full_attention"),
    ]


W_REAL, P_MAX_REAL, CANVAS_REAL, PROMPT_REAL = 1024, 4096, 256, 2048


def _drive_fixed_reveal(monkeypatch, *, retention):
    from models.experimental.diffusion_gemma.tt import traced_denoise as TD

    captured = {}

    class _Reader:
        owns_result = True
        borrow_full_span = False

        def set_read_span(self, p_max):
            captured["read_span"] = p_max

        def prepare_window_buffers(self, layers):
            captured["window_layers"] = dict(layers)

        def refresh_windows(self, prompt_len):
            captured["refreshed"] = prompt_len

    layer_types = ["sliding_attention"] * 5 + ["full_attention"]
    adapter = SimpleNamespace(
        prompt_len=PROMPT_REAL,
        prompt_hidden_by_layer=_Reader(),
        tt_model=SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=SimpleNamespace(config=SimpleNamespace(sliding_window=W_REAL)))
                for _ in layer_types
            ],
            hf_config=SimpleNamespace(layer_types=list(layer_types), sliding_window=W_REAL),
            mesh_device=None,
        ),
        use_reveal_mask=True,
        _reveal_mask_bufs={},
        prepare_reveal_mask_buffers=lambda **kwargs: captured.update(kwargs),
        update_reveal_mask_buffer=lambda prompt_len: None,
    )

    monkeypatch.setenv("DG_DENOISE_SLIDING_WINDOW", "1" if retention else "0")
    monkeypatch.setattr(TD, "_resolve_reveal_pmax", lambda value: P_MAX_REAL)
    monkeypatch.setattr(TD, "prefix_borrow_enabled", lambda: False)
    TD._prepare_fixed_reveal(adapter, canvas_len=CANVAS_REAL)
    return captured


def test_bounded_read_is_on_by_default_when_retention_is_enforced(monkeypatch):
    captured = _drive_fixed_reveal(monkeypatch, retention=True)
    assert captured["enforce_window"] is True
    assert captured["sliding_span"] == W_REAL
    assert set(captured["window_layers"]) == {0, 1, 2, 3, 4}
    assert all(value == W_REAL for value in captured["window_layers"].values())
    assert captured["refreshed"] == PROMPT_REAL


def test_bounded_read_does_not_engage_without_the_retention_mask(monkeypatch):
    captured = _drive_fixed_reveal(monkeypatch, retention=False)
    assert captured["enforce_window"] is False
    assert captured["sliding_span"] is None
    assert "window_layers" not in captured


def test_bounded_window_mask_matches_the_full_span_mask_on_shared_columns():
    prompt_len = 3 * W
    span = W
    lo = prompt_len - span
    full = _attend(_reveal(prompt_len, layer_type="sliding_attention", enforce=True))
    bounded = build_canvas_reveal_denoise_window_mask(prompt_len, CANVAS, span, lo, sliding_window=W) == 0

    full_visible = {column for column in range(P_MAX) if bool(full[0, column])}
    bounded_visible = {lo + relative for relative in range(span) if bool(bounded[0, relative])}
    assert bounded_visible == full_visible
    assert bool(bounded[:, span:].all())
    for row in range(1, CANVAS):
        assert torch.equal(bounded[row], bounded[0])
