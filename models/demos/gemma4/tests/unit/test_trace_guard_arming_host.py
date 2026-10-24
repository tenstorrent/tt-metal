# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only: when the decode-trace allocation guard walks the allocator (``ChunkedPrefillPageTableGuardMixin``).
GEMMA4_TRACE_ALLOC_GUARD_ARM=prefill (default): every prefill arms the next replay's check. ``alloc``: only the
call's allocation events arm it (page-table buffers allocated / grown, a prefill trace captured, a chunked or
continuation prefill, a multimodal prefill) plus a periodic check every GEMMA4_TRACE_ALLOC_GUARD_PERIOD replays.
The decision on the levels (baseline, budget) is the same in both modes. No device, no conftest."""

from types import SimpleNamespace

from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin as Mixin


class _Guard(Mixin):
    """The mixin over a stub generator: counted allocator walks, no device."""

    def __init__(self, levels=(100, 10)):
        self.model = [SimpleNamespace(_g4_persistent_alloc_event=None)]
        self.trace_id_prefill = {}
        self._g4_trace_alloc_baseline = {}
        self._g4_trace_guard_recaptures = 0
        self.levels = tuple(levels)
        self.walks = 0

    def _g4_alloc_levels(self):
        self.walks += 1
        return self.levels


def _replays(g, n):
    return [g._g4_trace_guard_should_recapture("k") for _ in range(n)]


def test_default_mode_arms_on_every_prefill(monkeypatch):
    monkeypatch.delenv("GEMMA4_TRACE_ALLOC_GUARD_ARM", raising=False)
    g = _Guard()
    assert _replays(g, 1) == [False] and g.walks == 1  # the first replay takes the baseline
    assert _replays(g, 3) == [False] * 3 and g.walks == 2  # armed by default on the first replay after capture
    assert _replays(g, 3) == [False] * 3 and g.walks == 2  # replays alone never walk
    assert g._g4_arm_alloc_check("prefill") is True
    assert _replays(g, 2) == [False] * 2 and g.walks == 3
    assert g._g4_arm_after_prefill(prefill_trace_ids_before={}, chunked=False, multimodal=False) == []


def test_alloc_mode_arms_on_events_and_periodically(monkeypatch):
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_ARM", "alloc")
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_PERIOD", "4")
    g = _Guard()
    _replays(g, 2)  # baseline + the initial pending check
    assert g.walks == 2
    assert g._g4_arm_alloc_check("prefill") is False  # a plain prefill does not arm
    assert _replays(g, 3) == [False] * 3 and g.walks == 2
    assert _replays(g, 1) == [False] and g.walks == 3  # the 4th replay since the last check: periodic
    assert g._g4_alloc_check_reason == "periodic"
    assert _replays(g, 3) == [False] * 3 and g.walks == 3
    # events arm in both modes
    g.model[0]._g4_persistent_alloc_event = "grown"
    reasons = g._g4_arm_after_prefill(prefill_trace_ids_before={}, chunked=False, multimodal=False)
    assert reasons == ["page tables grown"] and g.model[0]._g4_persistent_alloc_event is None
    assert _replays(g, 1) == [False] and g.walks == 4
    g.trace_id_prefill = {(128, 1): 7}
    assert g._g4_arm_after_prefill(prefill_trace_ids_before={}, chunked=False, multimodal=False) == [
        "prefill trace capture"
    ]
    assert _replays(g, 1) == [False] and g.walks == 5
    assert g._g4_arm_after_prefill(prefill_trace_ids_before={(128, 1): 7}, chunked=True, multimodal=True) == [
        "chunked prefill",
        "multimodal prefill",
    ]
    assert _replays(g, 1) == [False] and g.walks == 6
    assert g._g4_arm_after_prefill(prefill_trace_ids_before={(128, 1): 7}, chunked=False, multimodal=False) == []
    assert _replays(g, 1) == [False] and g.walks == 6


def test_alloc_mode_period_zero_is_events_only(monkeypatch):
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_ARM", "alloc")
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_PERIOD", "0")
    g = _Guard()
    _replays(g, 2)
    assert _replays(g, 200) == [False] * 200 and g.walks == 2


def test_decision_is_the_same_in_both_modes(monkeypatch):
    for mode in ("prefill", "alloc"):
        monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_ARM", mode)
        monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_MAX", "8")
        g = _Guard(levels=(100, 10))
        _replays(g, 2)
        g.levels = (150, 10)  # a persistent allocation grew the bottom-up DRAM level
        g._g4_arm_alloc_check("page tables grown")
        assert _replays(g, 1) == [True], mode  # re-capture
        assert g._g4_trace_guard_recaptures == 1
        g._g4_trace_alloc_baseline["k"] = g.levels  # what the re-capture does
        g.levels = (90, 10)  # lower than the baseline: tracked downwards, no re-capture
        g._g4_arm_alloc_check("page tables grown")
        assert _replays(g, 1) == [False] and g._g4_trace_alloc_baseline["k"] == (90, 10)


def test_unknown_mode_is_the_default():
    import os

    os.environ["GEMMA4_TRACE_ALLOC_GUARD_ARM"] = "sometimes"
    try:
        assert _Guard()._g4_guard_arm_mode() == "prefill"
    finally:
        del os.environ["GEMMA4_TRACE_ALLOC_GUARD_ARM"]
