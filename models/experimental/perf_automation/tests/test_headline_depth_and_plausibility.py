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
    # A partial window is disclosed as a coverage SAMPLE with NO count: the count was read from the
    # env default (usually 2) and did not track the depth actually profiled, so "N layers" printed a
    # wrong number (a 16-layer slice showed "2"). The profile stamp still decides partial-vs-full --
    # an "all" stamp is trusted over a capped env.
    monkeypatch.delenv("TT_PERF_LAYERS", raising=False)
    assert m._depth_label({"perf_layers": "16"}) == "a coverage sample (not the full model)"
    monkeypatch.setenv("TT_PERF_LAYERS", "8")
    assert m._depth_label({"perf_layers": "all"}) == "all layers"


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
    assert m._depth_label(None) == "a coverage sample (not the full model)"
    assert m._depth_label({}) == "a coverage sample (not the full model)"


def _render(tmp_path, monkeypatch, baseline_device_ms, final_ms, perf_layers):
    """Drive the REAL renderer, not its helpers. The helper unit tests above all passed while the
    rendered line was still wrong -- the defects lived in how render_summary combined them."""
    import json

    m = _summary()
    monkeypatch.delenv("TT_PERF_LAYERS", raising=False)
    kl = tmp_path / "kernel_log.json"
    kl.write_text(
        json.dumps(
            [
                {
                    "op_signature": "MatmulDeviceOperation",
                    "kernel_kind": "dtype",
                    "measured_ms": final_ms,
                    "beat_baseline": True,
                }
            ]
        )
    )
    prof = {"device_ms": baseline_device_ms, "perf_layers": perf_layers, "buckets": []}
    out = m.render_summary(
        kl,
        baseline_ms=baseline_device_ms,
        model="llama3_1_8b_p150",
        task="main",
        before_ms=48.38,
        after_ms=22.77,
        baseline_profile=prof,
        before_mode="trace+1cq",
        after_mode="trace+1cq",
        finalized=True,
    )
    return next(ln for ln in str(out).splitlines() if "eager per-op" in ln)
