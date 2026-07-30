# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host (no-device) tests for the paged-prefix Phase-1 PLUMBING logic.

Covers the control-flow the recapture fix depends on, without a device:
  - `MutablePrefixKVReader` read-span decoupling: once a fixed `read_span` (p_max) is set,
    `__call__` always reads p_max rows regardless of the growing committed `prompt_len`
    (this constant read shape is what makes the trace capture-once/replay-many).
  - `set_prompt_len` still enforces monotonic + tile-aligned + `<= read_span`.
  - the up-front controller's fixed reveal-buffer preparation and p_max validation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.experimental.diffusion_gemma.tt import denoise_forward as DF
from models.experimental.diffusion_gemma.tt import traced_denoise as TD

TILE = 32


def _reader_with_recording_read_fn(prompt_len=256):
    seen = []

    def read_fn(tt_model, *, prompt_len, seq_len_start, layer_idx):
        seen.append(prompt_len)
        return ("K", "V")

    reader = DF.MutablePrefixKVReader(tt_model=object(), prompt_len=prompt_len, read_fn=read_fn)
    return reader, seen


def test_read_span_decouples_read_from_committed_len():
    reader, seen = _reader_with_recording_read_fn(prompt_len=256)
    reader.set_read_span(8192)
    reader(0)  # committed 256, but read span 8192
    reader.set_prompt_len(512)  # a block committed -> committed grows
    reader(1)
    reader.set_prompt_len(768)
    reader(2)
    assert seen == [8192, 8192, 8192], "read must be the fixed p_max span, not the growing committed len"


def test_read_span_defaults_to_prompt_len_when_unset():
    reader, seen = _reader_with_recording_read_fn(prompt_len=256)
    reader(0)
    assert seen == [256], "without a read_span the reader reads the committed prompt_len (legacy behavior)"


def test_set_prompt_len_guards_still_hold_under_read_span(expect_error):
    reader, _ = _reader_with_recording_read_fn(prompt_len=256)
    reader.set_read_span(1024)
    with expect_error(ValueError):
        reader.set_prompt_len(128)  # shrink
    with expect_error(ValueError):
        reader.set_prompt_len(300)  # not tile-aligned
    with expect_error(ValueError):
        reader.set_prompt_len(2048)  # exceeds read_span
    reader.set_prompt_len(768)  # valid
    assert reader.prompt_len == 768


def test_set_read_span_requires_tile_alignment_and_not_below_committed(expect_error):
    reader, _ = _reader_with_recording_read_fn(prompt_len=256)
    with expect_error(ValueError):
        reader.set_read_span(300)  # not tile aligned
    with expect_error(ValueError):
        reader.set_read_span(128)  # below committed 256


class _FakeKCache:
    def __init__(self, seq):
        self.shape = [1, 8, seq, 128]


class _FakeModel:
    # `layers` + `hf_config.layer_types` are required, not decoration: the bounded sliding read
    # inspects each layer's TYPE, and since DG_DENOISE_SLIDING_SPAN was deleted (2026-07-29) that
    # path runs whenever retention is enforced -- which is the default. A fake without layers only
    # passed while the span was gated off behind its own flag.
    SLIDING_WINDOW = 1024

    def __init__(self, seq):
        self.tt_kv_cache = [(_FakeKCache(seq), _FakeKCache(seq))]
        layer_types = ["sliding_attention"] * 5 + ["full_attention"]
        self.layers = [
            SimpleNamespace(self_attn=SimpleNamespace(config=SimpleNamespace(sliding_window=self.SLIDING_WINDOW)))
            for _ in layer_types
        ]
        self.hf_config = SimpleNamespace(layer_types=layer_types, sliding_window=self.SLIDING_WINDOW)


class _FakeAdapter:
    """Records the reveal plumbing calls the controller makes before capture."""

    def __init__(self, cache_seq, prompt_len):
        self.tt_model = _FakeModel(cache_seq)
        self.prompt_len = prompt_len
        self.calls = []
        self.use_reveal_mask = False
        self.prompt_hidden_by_layer = self  # acts as the reader too
        # The real adapter keeps one reveal buffer per layer type; the controller logs their keys
        # on the retention-window path, so the double has to carry them too.
        self._reveal_mask_bufs = {}

    # reader surface
    def set_read_span(self, p_max):
        self.calls.append(("set_read_span", p_max))

    def prepare_window_buffers(self, window_layers):
        self.calls.append(("prepare_window_buffers", dict(window_layers)))

    def refresh_windows(self, prompt_len):
        self.calls.append(("refresh_windows", prompt_len))

    # adapter reveal surface
    def prepare_reveal_mask_buffers(self, *, canvas_len, p_max, prompt_len, enforce_window=False, sliding_span=None):
        self.calls.append(("prepare", canvas_len, p_max, prompt_len))
        self.use_reveal_mask = True
        layer_types = ("full_attention", "sliding_attention") if enforce_window else ("full_attention",)
        self._reveal_mask_bufs = {layer_type: object() for layer_type in layer_types}

    def update_reveal_mask_buffer(self, prompt_len):
        self.calls.append(("update", prompt_len))


def test_resolve_pmax_requires_explicit_aligned_value(monkeypatch, expect_error):
    a = _FakeAdapter(cache_seq=8192, prompt_len=256)
    monkeypatch.delenv("DG_DENOISE_REVEAL_PMAX", raising=False)
    with expect_error(RuntimeError, match="explicit bounded DG_DENOISE_REVEAL_PMAX"):
        TD._resolve_reveal_pmax(a)

    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "4096")
    assert TD._resolve_reveal_pmax(a) == 4096

    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "4097")
    with expect_error(RuntimeError, match="positive 32-token multiple"):
        TD._resolve_reveal_pmax(a)


def test_prepare_fixed_reveal_wires_read_span_and_mask(monkeypatch):
    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "4096")
    a = _FakeAdapter(cache_seq=8192, prompt_len=256)
    assert TD._prepare_fixed_reveal(a, canvas_len=256) == 4096
    assert ("set_read_span", 4096) in a.calls
    assert ("prepare", 256, 4096, 256) in a.calls
    assert ("update", 256) in a.calls


@pytest.mark.parametrize(
    "flag, expect_window, expect_masks, expect_span",
    [
        ("0", False, ["full_attention"], None),
        ("1", True, ["full_attention", "sliding_attention"], _FakeModel.SLIDING_WINDOW),
    ],
)
def test_prepare_fixed_reveal_forwards_the_retention_flag(monkeypatch, flag, expect_window, expect_masks, expect_span):
    """The env gate has to reach `prepare_reveal_mask_buffers`, and add the sliding mask when on.

    HF's sliding layers retain only `sliding_window - 1` committed keys (#51080). With the flag off
    one shared full-attention mask serves all 30 layers; with it on the sliding layers need their
    own mask, so the buffer set is what tells the two regimes apart.

    It also carries the BOUNDED READ now. This used to assert `sliding_span is None` because the perf
    half was its own gate (DG_DENOISE_SLIDING_SPAN); that flag was deleted on 2026-07-29 because the
    bounded read is bit-identical whenever retention is enforced, so the span follows this flag and
    nothing else. Retention off must still mean no bounded read -- that is the one part of the
    bounded read which is NOT bit-identical.
    """
    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "4096")
    monkeypatch.setenv("DG_DENOISE_SLIDING_WINDOW", flag)
    seen = {}

    adapter = _FakeAdapter(cache_seq=8192, prompt_len=256)
    real_prepare = adapter.prepare_reveal_mask_buffers

    def recording_prepare(*, canvas_len, p_max, prompt_len, enforce_window=False, sliding_span=None):
        seen["enforce_window"] = enforce_window
        seen["sliding_span"] = sliding_span
        return real_prepare(
            canvas_len=canvas_len,
            p_max=p_max,
            prompt_len=prompt_len,
            enforce_window=enforce_window,
            sliding_span=sliding_span,
        )

    adapter.prepare_reveal_mask_buffers = recording_prepare
    TD._prepare_fixed_reveal(adapter, canvas_len=256)

    assert seen["enforce_window"] is expect_window
    assert seen["sliding_span"] == expect_span, "the bounded read must follow the retention flag"
    assert sorted(adapter._reveal_mask_bufs) == expect_masks
    allocated = [c for c in adapter.calls if c[0] == "prepare_window_buffers"]
    if expect_span is None:
        assert not allocated, "no block-resident window buffers without the retention mask"
    else:
        assert allocated and set(allocated[0][1]) == {0, 1, 2, 3, 4}, "only the sliding layers bounded"
