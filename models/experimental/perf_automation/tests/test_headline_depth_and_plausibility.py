# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The eager headline must state the RIGHT depth and refuse an impossible delta.

llama3_1_8b_p150 rendered, for a run profiled at TT_PERF_LAYERS=16:

    eager per-op device time (all layers):  0.06 ms  ->  648.17 ms   (-1062476.1%, 0.00x)

Three defects in one line: a foreign anchor (fixed by keying the baseline file), a depth label read
from the RENDERER's env rather than the profile -- the depth is exported into the profiling
subprocess, so it read empty and claimed "all layers" -- and arithmetic performed faithfully on a
pair that cannot describe the same work.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _summary():
    spec = importlib.util.spec_from_file_location("summary_depth_ut", _ROOT / "cc_optimize" / "summary.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["summary_depth_ut"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_depth_label_prefers_the_profile_stamp(monkeypatch):
    m = _summary()
    monkeypatch.delenv("TT_PERF_LAYERS", raising=False)
    assert m._depth_label({"perf_layers": "16"}) == "16 layers"


def test_depth_label_does_not_claim_all_layers_for_a_capped_profile(monkeypatch):
    """THE REGRESSION: renderer env is empty, so the label lied about a 16-layer profile."""
    m = _summary()
    monkeypatch.delenv("TT_PERF_LAYERS", raising=False)
    assert m._depth_label({"perf_layers": "16"}) != "all layers"


def test_depth_label_says_all_layers_for_an_uncapped_profile(monkeypatch):
    m = _summary()
    monkeypatch.delenv("TT_PERF_LAYERS", raising=False)
    assert m._depth_label({"perf_layers": "all"}) == "all layers"
    assert m._depth_label({}) == "all layers"


def test_depth_label_falls_back_to_env_for_unstamped_profiles(monkeypatch):
    m = _summary()
    monkeypatch.setenv("TT_PERF_LAYERS", "8")
    assert m._depth_label(None) == "8 layers"
    assert m._depth_label({}) == "8 layers"


def test_a_pair_two_orders_of_magnitude_apart_is_implausible():
    m = _summary()
    assert m._implausible_pair(0.06, 648.17) is True


def test_a_real_speedup_is_plausible():
    m = _summary()
    assert m._implausible_pair(2464.18, 648.17) is False
    assert m._implausible_pair(648.17, 2464.18) is False


def test_missing_or_zero_readings_are_not_flagged_as_implausible():
    m = _summary()
    for a, b in ((0.0, 5.0), (5.0, 0.0), (None, 5.0), (5.0, None)):
        assert m._implausible_pair(a, b) is False
