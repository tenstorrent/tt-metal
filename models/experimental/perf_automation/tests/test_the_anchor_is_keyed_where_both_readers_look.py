# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The byte anchor was filed under a key nothing looks up, so the gate and the report disagreed.

    depth=str(facts.get("unit") or "unit")

The fallback is the WORD "unit" where a unit VALUE ("token", "step", "image") belongs. It fires
often, because it fires whenever no unit is known -- and the unit is unknown exactly when this code
runs. _emit_perf_target_inputs executes ONCE AT SETUP, before any trace, and the unit is set only
from an observation (PERF_MCP_LAST_HEADLINE_UNIT) because an HF tag names the TASK and cannot say
whether a model loops: `text-to-speech` covers XTTS, which emits tokens, and Kokoro, which produces
a whole waveform in one pass. So the one moment the anchor is written is the one moment the thing it
must be keyed by does not exist.

MEASURED ON VOXTRAL RUN 18, from the ledger the run actually wrote:

    {"kind":"active_bytes","phase":"before","value_ms":7222.9663,
     "depth":"unit","mode":"bytes_mb","source":"checkpoint bytes + HF config"}

    REPORT  asks for depth="token"      -> MISS -> falls back to the snapshot -> 4.777 GB
    GATE    scans rows, takes the first -> HIT  -> the placeholder row        -> 7.223 GB
                                                                    1.51x apart

That is the exact failure the anchor was introduced to prevent, and a 3175-test suite passed
throughout -- including a test NAMED test_the_gate_and_the_report_divide_by_the_same_bytes --
because the one test that read this anchor read it under `facts.get("unit") or "unit"`, the same
placeholder. Both sides agreed, and agreeing on a wrong key is indistinguishable from being right.

Worse than wrong bytes: the depth-agnostic scan reads a row's depth AS the unit, so a placeholder row
also reports the model's unit of work as "unit", which the band, the at-floor verdict and the
headline rate all inherit.

THE FIX IS TO DECLINE, not to guess a better default. Defaulting to "token" is the bug that once
labelled every diffusion and classifier model per-token. Nothing is lost by waiting: before the first
trace there is no measurement, so there is no ceiling for two readers to disagree about.
"""
import json
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))


def _mods():
    import importlib.util as ilu

    out = []
    for name, rel in (
        ("run_kw", "cc_optimize/run.py"),
        ("led_kw", "cc_optimize/measurements.py"),
        ("mcp_kw", "cc_optimize/perf_mcp.py"),
    ):
        spec = ilu.spec_from_file_location(name, _PA / rel)
        m = ilu.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        out.append(m)
    return out


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    monkeypatch.delenv("PERF_MCP_LAST_HEADLINE_UNIT", raising=False)
    run, led, mcp = _mods()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None, *_a, **_k: 16_060_556_376)
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None, *_a, **_k: "meta-llama/Llama-3.1-8B")
    monkeypatch.setattr(run, "_hf_snapshots", lambda mid: [])
    root = tmp_path / "m"
    root.mkdir()
    return run, led, mcp, root, monkeypatch


def _emit(run, root):
    run._emit_perf_target_inputs(root, root, None, {})
    return json.loads((root / "perf_target_inputs.json").read_text())


def test_no_unit_writes_no_anchor(wired):
    """THE FIX. Declining is what the rest of the chain does: no recoverable unit means no ceiling."""
    run, led, _mcp, root, _mp = wired
    _emit(run, root)
    rows = list(led.rows(led.KIND_ACTIVE_BYTES, led.PHASE_BEFORE, "m", "main"))
    assert rows == [], "an anchor was written with no unit to key it by: %s" % rows


def test_the_placeholder_key_is_never_written(wired):
    """Specifically: not a row under the literal string "unit"."""
    run, led, _mcp, root, _mp = wired
    _emit(run, root)
    assert led.anchor_value(led.KIND_ACTIVE_BYTES, depth="unit", model="m", task="main") is None


def test_a_known_unit_anchors_under_that_unit(wired):
    """And the anchor still happens -- this must not become "never anchor"."""
    run, led, _mcp, root, mp = wired
    mp.setenv("PERF_MCP_LAST_HEADLINE_UNIT", "token")
    facts = _emit(run, root)
    assert facts.get("unit") == "token"
    assert led.anchor_value(led.KIND_ACTIVE_BYTES, depth="token", model="m", task="main")


def test_a_non_token_unit_is_carried_through(wired):
    """The whole reason the unit is observed rather than guessed. A denoise step is not a token."""
    run, led, _mcp, root, mp = wired
    mp.setenv("PERF_MCP_LAST_HEADLINE_UNIT", "step")
    _emit(run, root)
    assert led.anchor_value(led.KIND_ACTIVE_BYTES, depth="step", model="m", task="main")
    assert led.anchor_value(led.KIND_ACTIVE_BYTES, depth="token", model="m", task="main") is None


def test_the_gate_refuses_a_legacy_placeholder_row(wired):
    """Ledgers already on disk carry the bad rows, and the anchor is write-once -- so the read side
    has to refuse them too, or run 18's 7.223 GB keeps being served to the gate forever."""
    _run, led, mcp, _root, _mp = wired
    led.anchor(led.KIND_ACTIVE_BYTES, 7222.9663, depth="unit", mode="bytes_mb", source="legacy", model="m")
    assert mcp._is_real_unit("token") and mcp._is_real_unit("step")
    assert not mcp._is_real_unit("unit")
    assert not mcp._is_real_unit("")
    assert not mcp._is_real_unit("unknown")


def test_a_placeholder_row_cannot_masquerade_as_the_unit(wired):
    """The depth-agnostic scan reads depth AS the unit, so a placeholder row would report the model's
    unit of work as "unit" -- a per-"unit" ceiling scored against a per-token measurement."""
    _run, led, mcp, _root, _mp = wired
    led.anchor(led.KIND_ACTIVE_BYTES, 7222.9663, depth="unit", mode="bytes_mb", source="legacy", model="m")
    assert not mcp._is_real_unit("unit")


def test_both_readers_resolve_the_same_row(wired):
    """The guarantee the suite already had a NAME for and did not reach. One anchor, one value, from
    the keyed lookup the report uses and the scan the gate falls back to."""
    run, led, mcp, root, mp = wired
    mp.setenv("PERF_MCP_LAST_HEADLINE_UNIT", "token")
    _emit(run, root)
    report_side = led.anchor_value(led.KIND_ACTIVE_BYTES, depth="token", model="m", task="main")
    gate_side = None
    for r in led.rows(led.KIND_ACTIVE_BYTES, led.PHASE_BEFORE, "m", "main"):
        if mcp._is_real_unit(r.get("depth")) and float(r.get("value_ms") or 0) > 0:
            gate_side = float(r["value_ms"])
            break
    assert report_side is not None and gate_side is not None
    assert abs(report_side - gate_side) < 1e-9, (report_side, gate_side)
