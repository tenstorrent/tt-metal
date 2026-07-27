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


# --- the tracy pair: depth must match ----------------------------------------------------------


def _write_orig(tmp_path, monkeypatch, device_ms, perf_layers=None):
    d = {"device_ms": device_ms}
    if perf_layers is not None:
        d["perf_layers"] = perf_layers
    p = Path(tempfile.gettempdir()) / "perf_mcp_orig_baseline_STRESSMODEL_main.json"
    p.write_text(json.dumps(d))
    return p


# --- labelling: a ms figure without its depth is what caused the confusion ----------------------


def test_depth_label_says_all_layers_when_uncapped(monkeypatch):
    monkeypatch.delenv("TT_PERF_LAYERS", raising=False)
    assert S._depth_label() == "all layers"
    monkeypatch.setenv("TT_PERF_LAYERS", "0")
    assert S._depth_label() == "all layers"
    monkeypatch.setenv("TT_PERF_LAYERS", "8")
    assert S._depth_label() == "8 layers"


# --- the anchor must be this run's own starting point, never its current value -----------------


def test_sections_say_which_profile_they_came_from():
    """The op table and the trace line both read the BASELINE profile. Labelling the table 'latest'
    put a 2464 ms breakdown directly above a 714 ms 'measured' line."""
    txt = _headline(
        baseline_ms=714.94,
        final_override_ms=714.94,
        baseline_profile={
            "device_ms": 2464.18,
            "per_token_ms": 33.89,
            "buckets": [{"id": "matmul", "device_ms": 1010.23, "count": 5600}],
        },
    )
    assert "BASELINE profile" in txt
    assert "tracy trace pass, BASELINE" in txt
    assert "latest profile" not in txt
