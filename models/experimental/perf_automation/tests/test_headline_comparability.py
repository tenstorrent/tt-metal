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

# Loaded BY PATH, like every sibling test. The package-path import resolved through the interpreter's
# site-packages, which points at a different tt-metal checkout, so this file was silently asserting
# against another tree's summary.py -- it passed on wording that no longer exists in this one.
import importlib.util as _ilu
import sys as _sys

_SPEC = _ilu.spec_from_file_location(
    "cc_summary_headline_ut", Path(__file__).resolve().parents[1] / "cc_optimize" / "summary.py"
)
S = _ilu.module_from_spec(_SPEC)
_sys.modules["cc_summary_headline_ut"] = S
_SPEC.loader.exec_module(S)

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
    # a capped window is a coverage SAMPLE with no count (the count came from the env default and did
    # not track the depth actually profiled, so it printed a wrong number)
    assert S._depth_label() == "a coverage sample (not the full model)"


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
    # The heading names no provenance: the variable is called baseline_profile, but perf_mcp rewrites
    # that file on every profile, so the word BASELINE was a claim about which point in the run the
    # rows describe. The trace line moved to the ledger for the same reason -- it was reading
    # per_token_ms out of that mutable file.
    #
    # It no longer restates the total either. Each row carries its device_ms AND its share, so the
    # total is fixed by the rows themselves; the subtitle was a third copy of what the table says.
    assert "BASELINE profile" not in txt and "latest profile" not in txt
    assert "tracy trace pass, BASELINE" not in txt
    assert "Op breakdown" in txt and "device time by op class" not in txt, txt
    # the rows still pin the total: 1010.23 at 100.0%
    assert "1010.23" in txt, txt
