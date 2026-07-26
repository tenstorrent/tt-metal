# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The summary must never present two incomparable numbers as a speedup.

Two real headlines from llama3_1_8b_p150 (2026-07-26), both reporting regressions that did not happen:

    baseline 832.93 ms  ->  final 1088.15 ms   (-30.6%, 0.77x)
      a 2-layer tracy profile cached in /tmp from the previous DAY, paired with a 16-layer one.
      The run was actually 2149.71 -> 1088.15.

    before 47.10 ms  ->  after 100.00 ms   (-112.3% SLOWER)
      an eager wall-clock over the whole forward, paired with a trace+1cq per-token step.
      _establish_fullpipe_baseline RE-BASELINES the stored value when the mode changes, but the
      BEFORE bookend is captured once and never re-taken, so the pair drifts apart mid-run.

Both are the same mistake: subtracting numbers that measure different things. These tests pin the two
guards -- depth for the tracy pair, mode for the full-pipeline pair.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from models.experimental.perf_automation.cc_optimize import run as R
from models.experimental.perf_automation.cc_optimize import summary as S

_KL = None


def _kl():
    global _KL
    if _KL is None:
        p = Path(tempfile.mkdtemp()) / "kl.json"
        p.write_text("[]")
        _KL = str(p)
    return _KL


def _headline(**kw):
    base = dict(
        kernel_log_path=_kl(),
        model="m",
        task="main",
        metric="device_ms",
        baseline_ms=2149.71,
        finalized=True,
    )
    base.update(kw)
    out = S.render_summary(**base)
    return out if isinstance(out, str) else "\n".join(out)


# --- the full-pipeline pair: mode must match --------------------------------------------------


def test_same_mode_reports_a_delta():
    txt = _headline(before_ms=200.0, after_ms=100.0, before_mode="trace+1cq", after_mode="trace+1cq")
    assert "before 200.00 ms  ->  after 100.00 ms" in txt
    assert "+50.0% faster" in txt
    assert "NOT COMPARABLE" not in txt


def test_different_modes_refuse_the_delta():
    """THE 47.10 -> 100.00 CASE."""
    txt = _headline(before_ms=47.10, after_ms=100.00, before_mode="eager", after_mode="trace+1cq")
    assert "NOT COMPARABLE" in txt
    assert "-112.3%" not in txt and "SLOWER" not in txt
    assert "[eager]" in txt and "[trace+1cq]" in txt  # both sides named


def test_mode_known_on_one_side_only_is_not_assumed_to_match():
    for b, a in (("trace+1cq", ""), ("", "trace+1cq")):
        txt = _headline(before_ms=200.0, after_ms=100.0, before_mode=b, after_mode=a)
        assert "NOT COMPARABLE" in txt, "assumed a match with mode %r vs %r" % (b, a)


def test_legacy_pair_with_no_modes_still_reports():
    """Older runs recorded no mode; refusing every one of them would be noise."""
    txt = _headline(before_ms=200.0, after_ms=100.0, before_mode="", after_mode="")
    assert "+50.0% faster" in txt and "NOT COMPARABLE" not in txt


def test_mode_comparison_is_case_and_space_insensitive():
    assert S._same_measurement(" Trace+1CQ ", "trace+1cq") is True
    assert S._same_measurement("eager", "trace+1cq") is False


# --- the tracy pair: depth must match ----------------------------------------------------------


def _write_orig(tmp_path, monkeypatch, device_ms, perf_layers=None):
    d = {"device_ms": device_ms}
    if perf_layers is not None:
        d["perf_layers"] = perf_layers
    p = Path(tempfile.gettempdir()) / "perf_mcp_orig_baseline_STRESSMODEL_main.json"
    p.write_text(json.dumps(d))
    return p


def test_original_baseline_used_when_depth_matches(monkeypatch):
    p = _write_orig(None, monkeypatch, 832.93, "16")
    try:
        monkeypatch.setenv("TT_PERF_LAYERS", "16")
        assert R._original_baseline_ms("STRESSMODEL", "main") == 832.93
    finally:
        p.unlink(missing_ok=True)


def test_original_baseline_refused_on_depth_mismatch(monkeypatch):
    """THE 832.93 CASE: a 2-layer number must never be the 'before' for a 16-layer run."""
    p = _write_orig(None, monkeypatch, 832.93, "2")
    try:
        monkeypatch.setenv("TT_PERF_LAYERS", "16")
        assert R._original_baseline_ms("STRESSMODEL", "main") is None
    finally:
        p.unlink(missing_ok=True)


def test_original_baseline_refused_when_unstamped(monkeypatch):
    """Files written before the stamp existed carry no depth -- cannot be shown to be comparable."""
    p = _write_orig(None, monkeypatch, 832.93, None)
    try:
        monkeypatch.setenv("TT_PERF_LAYERS", "16")
        assert R._original_baseline_ms("STRESSMODEL", "main") is None
    finally:
        p.unlink(missing_ok=True)


def test_all_layers_matches_an_absent_cap(monkeypatch):
    """The gate runs uncapped; 'all' must equal an absent TT_PERF_LAYERS, not mismatch it."""
    p = _write_orig(None, monkeypatch, 500.0, "all")
    try:
        monkeypatch.delenv("TT_PERF_LAYERS", raising=False)
        assert R._original_baseline_ms("STRESSMODEL", "main") == 500.0
    finally:
        p.unlink(missing_ok=True)


def test_headline_falls_back_to_this_runs_baseline_when_original_refused():
    """Refusing the stale number must not blank the headline -- it uses this run's own baseline."""
    txt = _headline(original_baseline_ms=None, final_override_ms=1088.15)
    assert "2149.71 ms  ->  1088.15 ms" in txt and "1.98x" in txt


# --- labelling: a ms figure without its depth is what caused the confusion ----------------------


def test_every_reported_number_names_its_depth(monkeypatch):
    monkeypatch.setenv("TT_PERF_LAYERS", "16")
    monkeypatch.setenv("PERF_MCP_MODEL_LAYERS", "32")
    txt = _headline(
        final_override_ms=1088.15,
        before_ms=200.0,
        after_ms=100.0,
        before_mode="trace+1cq",
        after_mode="trace+1cq",
        baseline_profile={"per_token_ms": 24.98},
    )
    assert "eager per-op device time (16 layers)" in txt
    assert "tracy trace pass, same window (16 layers):  24.98 ms" in txt
    assert "(all 32 layers)" in txt
    assert "baseline " not in txt.split("Roofline")[0]  # the ambiguous word is gone


def test_depth_label_says_all_layers_when_uncapped(monkeypatch):
    monkeypatch.delenv("TT_PERF_LAYERS", raising=False)
    assert S._depth_label() == "all layers"
    monkeypatch.setenv("TT_PERF_LAYERS", "0")
    assert S._depth_label() == "all layers"
    monkeypatch.setenv("TT_PERF_LAYERS", "8")
    assert S._depth_label() == "8 layers"
