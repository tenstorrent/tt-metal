# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host (no-device) tests for the paged-prefix Phase-1 PLUMBING logic.

Covers the control-flow the recapture fix depends on, without a device:
  - `MutablePrefixKVReader` read-span decoupling: once a fixed `read_span` (p_max) is set,
    `__call__` always reads p_max rows regardless of the growing committed `prompt_len`
    (this constant read shape is what makes the trace capture-once/replay-many).
  - `set_prompt_len` still enforces monotonic + tile-aligned + `<= read_span`.
  - `reveal_mask_enabled` / `_resolve_reveal_pmax` / `_prepare_reveal_if_enabled` control flow.
"""

from __future__ import annotations


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


def test_reveal_mask_enabled_env(monkeypatch):
    monkeypatch.delenv("DG_DENOISE_REVEAL_MASK", raising=False)
    assert TD.reveal_mask_enabled() is False
    monkeypatch.setenv("DG_DENOISE_REVEAL_MASK", "1")
    assert TD.reveal_mask_enabled() is True
    # denoise_forward's copy reads the same env
    assert DF.reveal_mask_enabled() is True


class _FakeKCache:
    def __init__(self, seq):
        self.shape = [1, 8, seq, 128]


class _FakeModel:
    def __init__(self, seq):
        self.tt_kv_cache = [(_FakeKCache(seq), _FakeKCache(seq))]


class _FakeAdapter:
    """Records the reveal plumbing calls the controller makes before capture."""

    def __init__(self, cache_seq, prompt_len):
        self.tt_model = _FakeModel(cache_seq)
        self.prompt_len = prompt_len
        self.calls = []
        self.prompt_hidden_by_layer = self  # acts as the reader too

    # reader surface
    def set_read_span(self, p_max):
        self.calls.append(("set_read_span", p_max))

    # adapter reveal surface
    def prepare_reveal_mask_buffers(self, *, canvas_len, p_max, prompt_len, enforce_window=False):
        self.calls.append(("prepare", canvas_len, p_max, prompt_len))

    def update_reveal_mask_buffer(self, prompt_len):
        self.calls.append(("update", prompt_len))


def test_resolve_pmax_from_cache_and_override(monkeypatch):
    a = _FakeAdapter(cache_seq=8192, prompt_len=256)
    monkeypatch.delenv("DG_DENOISE_REVEAL_PMAX", raising=False)
    assert TD._resolve_reveal_pmax(a) == 8192
    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "4096")
    assert TD._resolve_reveal_pmax(a) == 4096
    # non-tile-aligned override rounds up
    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "4097")
    assert TD._resolve_reveal_pmax(a) == 4096 + TILE


def test_prepare_reveal_if_enabled_wires_read_span_and_mask(monkeypatch):
    monkeypatch.setenv("DG_DENOISE_REVEAL_MASK", "1")
    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "4096")
    a = _FakeAdapter(cache_seq=8192, prompt_len=256)
    TD._prepare_reveal_if_enabled(a, canvas_len=256, start_pos=256)
    assert ("set_read_span", 4096) in a.calls
    assert ("prepare", 256, 4096, 256) in a.calls
    assert ("update", 256) in a.calls


def test_prepare_reveal_if_enabled_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("DG_DENOISE_REVEAL_MASK", raising=False)
    a = _FakeAdapter(cache_seq=8192, prompt_len=256)
    TD._prepare_reveal_if_enabled(a, canvas_len=256, start_pos=256)
    assert a.calls == []
