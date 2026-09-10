# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ledger is the single source for reported numbers, and it OUTLIVES the run that wrote it."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("measurements_ut", _ROOT / "cc_optimize" / "measurements.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["measurements_ut"] = m
    spec.loader.exec_module(m)
    return m


def _led(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "led.jsonl"))
    return m


def test_a_reading_is_stored_with_its_provenance(tmp_path, monkeypatch):
    m = _led(tmp_path, monkeypatch)
    assert m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.18, depth="16", mode="eager", source="profile_model")
    r = m.first(m.KIND_EAGER, m.PHASE_BEFORE)
    assert r["value_ms"] == 2464.18 and r["depth"] == "16" and r["mode"] == "eager"


def test_an_unlabelled_reading_is_refused(tmp_path, monkeypatch):
    """Storing a number with no depth/mode would rebuild the guessing problem inside the ledger."""
    m = _led(tmp_path, monkeypatch)
    assert m.record(m.KIND_EAGER, m.PHASE_BEFORE, 100.0, depth="", mode="eager") is False
    assert m.record(m.KIND_EAGER, m.PHASE_BEFORE, 100.0, depth="16", mode="") is False
    assert m.first(m.KIND_EAGER, m.PHASE_BEFORE) is None


def test_the_original_baseline_survives_repeated_reruns(tmp_path, monkeypatch):
    """THE POINT OF DURABILITY. Run 1 measures 2464 on the untouched model; runs 2 and 3 start from
    an already-optimized model and measure far less. The reported original must stay 2464 -- without
    this, restarting optimize silently re-anchors on the optimized state and the real 3.8x becomes
    unreportable."""
    m = _led(tmp_path, monkeypatch)
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.18, depth="16", mode="eager", source="run1")
    m.record(m.KIND_EAGER, m.PHASE_AFTER, 648.17, depth="16", mode="eager", source="run1")
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 648.17, depth="16", mode="eager", source="run2")
    m.record(m.KIND_EAGER, m.PHASE_AFTER, 601.00, depth="16", mode="eager", source="run2")
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 601.00, depth="16", mode="eager", source="run3")
    m.record(m.KIND_EAGER, m.PHASE_AFTER, 590.00, depth="16", mode="eager", source="run3")
    assert m.first(m.KIND_EAGER, m.PHASE_BEFORE)["value_ms"] == 2464.18
    assert m.last(m.KIND_EAGER, m.PHASE_AFTER)["value_ms"] == 590.00
    assert round(m.delta_pct(m.first(m.KIND_EAGER), m.last(m.KIND_EAGER)), 1) == 76.1


def test_nothing_truncates_the_ledger(tmp_path, monkeypatch):
    """A fresh ladder, a cleared kernel log and a new worktree must not erase measurement history."""
    m = _led(tmp_path, monkeypatch)
    for i in range(5):
        m.record(m.KIND_EAGER, m.PHASE_AFTER, 100.0 - i, depth="16", mode="eager")
    assert len(m.rows(m.KIND_EAGER, m.PHASE_AFTER)) == 5


def test_readings_of_different_depth_are_not_subtracted(tmp_path, monkeypatch):
    """The 832.93 -> 1088.15 (-30.6%) report: a 2-layer profile paired with a 16-layer one."""
    m = _led(tmp_path, monkeypatch)
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 832.93, depth="2", mode="eager")
    m.record(m.KIND_EAGER, m.PHASE_AFTER, 1088.15, depth="16", mode="eager")
    ok, why = m.comparable(m.first(m.KIND_EAGER), m.last(m.KIND_EAGER))
    assert not ok and "depth differs" in why
    assert m.delta_pct(m.first(m.KIND_EAGER), m.last(m.KIND_EAGER)) is None


def test_readings_of_different_mode_are_not_subtracted(tmp_path, monkeypatch):
    """The 47.10 [eager] -> 100.00 [trace+1cq] report: two different units."""
    m = _led(tmp_path, monkeypatch)
    m.record(m.KIND_FULLPIPE, m.PHASE_BEFORE, 47.10, depth="all", mode="eager")
    m.record(m.KIND_FULLPIPE, m.PHASE_AFTER, 100.00, depth="all", mode="trace+1cq")
    ok, why = m.comparable(m.first(m.KIND_FULLPIPE), m.last(m.KIND_FULLPIPE))
    assert not ok and "mode differs" in why


def test_a_missing_reading_is_not_measured_never_a_substitute(tmp_path, monkeypatch):
    m = _led(tmp_path, monkeypatch)
    m.record(m.KIND_EAGER, m.PHASE_AFTER, 648.17, depth="16", mode="eager")
    assert m.first(m.KIND_EAGER, m.PHASE_BEFORE) is None
    ok, why = m.comparable(None, m.last(m.KIND_EAGER))
    assert not ok and why == "not measured"


def test_two_models_never_share_a_ledger(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setattr(m.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    a = m.ledger_path("model_a")
    b = m.ledger_path("model_b")
    assert a != b


def test_a_corrupt_line_degrades_to_not_measured(tmp_path, monkeypatch):
    m = _led(tmp_path, monkeypatch)
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.18, depth="16", mode="eager")
    with open(tmp_path / "led.jsonl", "a") as fh:
        fh.write("{not json\n")
    assert m.first(m.KIND_EAGER, m.PHASE_BEFORE)["value_ms"] == 2464.18


def _render_with_ledger(tmp_path, monkeypatch, entries):
    """Drive the REAL renderer with a populated ledger."""
    import json

    spec = importlib.util.spec_from_file_location("summary_led_ut", _ROOT / "cc_optimize" / "summary.py")
    sm = importlib.util.module_from_spec(spec)
    sys.modules["summary_led_ut"] = sm
    spec.loader.exec_module(sm)
    m = _led(tmp_path, monkeypatch)
    for e in entries:
        m.record(*e[:3], depth=e[3], mode=e[4], source="test")
    monkeypatch.delenv("TT_PERF_LAYERS", raising=False)
    kl = tmp_path / "kl.json"
    kl.write_text(
        json.dumps([{"op_signature": "Matmul", "kernel_kind": "dtype", "measured_ms": 648.17, "beat_baseline": True}])
    )
    out = sm.render_summary(
        kl,
        baseline_ms=0.06,
        model="llama3_1_8b_p150",
        task="main",
        baseline_profile={"device_ms": 0.06, "perf_layers": "16", "buckets": []},
        finalized=True,
    )
    return next((ln for ln in str(out).splitlines() if "eager per-op" in ln), "")


def test_report_uses_the_ledger_and_ignores_the_foreign_anchor(tmp_path, monkeypatch):
    """THE WHOLE POINT: a stale 0.06 sits in the old baseline file, but the ledger holds the real
    pair, so the report states the true 2464 -> 648 instead of searching and finding 0.06."""
    m = _mod()
    line = _render_with_ledger(
        tmp_path,
        monkeypatch,
        [(m.KIND_EAGER, m.PHASE_BEFORE, 2464.18, "16", "eager"), (m.KIND_EAGER, m.PHASE_AFTER, 648.17, "16", "eager")],
    )
    assert "2464.18 ms" in line and "648.17 ms" in line, line
    assert "0.06" not in line, line
    assert "16 layers" in line and "+73.7%" in line, line


def test_report_drops_the_arrow_on_a_mismatched_ledger_pair(tmp_path, monkeypatch):
    m = _mod()
    line = _render_with_ledger(
        tmp_path,
        monkeypatch,
        [(m.KIND_EAGER, m.PHASE_BEFORE, 832.93, "2", "eager"), (m.KIND_EAGER, m.PHASE_AFTER, 1088.15, "16", "eager")],
    )
    # NO ARROW, no disclaimer. The disclaimer shipped in two real reports and was read past -- see
    # test_report_omits_uncomparable_pairs.py. What survives is the latest reading as a bare number.
    assert "1088.15" in line and "16 layers" in line, line
    assert "832.93" not in line and "->" not in line, line


def test_a_garbage_profile_is_never_recorded_as_the_original(tmp_path, monkeypatch):
    """The credibility guard now protects the LEDGER's before-row instead of a separate orig file.

    A partial or garbage capture must not become the permanent anchor: it is the reading every
    future rerun is compared against, so once wrong it stays wrong.
    """
    import importlib.util as ilu

    spec = ilu.spec_from_file_location("pm_led_ut", _ROOT / "cc_optimize" / "perf_mcp.py")
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(tmp_path / "m.json"))
    (tmp_path / "m.json").write_text('{"config": {}, "perf_test_resolved": {"path": "t.py"}}')
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "led.jsonl"))
    pm = ilu.module_from_spec(spec)
    sys.modules["pm_led_ut"] = pm
    spec.loader.exec_module(pm)

    pm._ledger_record({"device_ms": 0.0001, "perf_layers": "16", "buckets": []})
    m = _mod()
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "led.jsonl"))
    assert m.first(m.KIND_EAGER, m.PHASE_BEFORE) is None, "a garbage profile became the permanent anchor"

    pm._ledger_record({"device_ms": 2464.18, "perf_layers": "16", "buckets": []})
    assert m.first(m.KIND_EAGER, m.PHASE_BEFORE)["value_ms"] == 2464.18


def test_the_reader_finds_what_the_writer_wrote_without_any_env(tmp_path, monkeypatch):
    """THE SPLIT THIS ALMOST SHIPPED WITH.

    perf_mcp named artifacts from _MODEL_ROOT while the ledger and run.py read PERF_MCP_MODEL_NAME,
    which nothing exported. Those fell back to the literal "model": the writer wrote
    perf_measurements_llama..._main.jsonl, the reader looked for perf_measurements_model_main.jsonl,
    and every model on the box would have shared one "model" ledger -- the unkeyed bug renamed. An
    env setdefault could not fix it either, since the writer and the renderer are different
    PROCESSES. The name is now passed explicitly by whoever knows it.
    """
    import importlib.util as ilu

    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    monkeypatch.delenv("PERF_MCP_MODEL_NAME", raising=False)
    monkeypatch.delenv("PERF_MCP_MODEL_ROOT", raising=False)
    m = _mod()
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setattr(m.tempfile, "gettempdir", lambda: str(tmp_path))

    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.18, depth="16", mode="eager", model="llama3_1_8b_p150")

    spec = ilu.spec_from_file_location("summary_key_ut", _ROOT / "cc_optimize" / "summary.py")
    sm = ilu.module_from_spec(spec)
    sys.modules["summary_key_ut"] = sm
    spec.loader.exec_module(sm)
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setattr(sm._ledger().tempfile, "gettempdir", lambda: str(tmp_path))

    m.record(m.KIND_EAGER, m.PHASE_AFTER, 648.17, depth="16", mode="eager", model="llama3_1_8b_p150")

    import json as _json

    kl = tmp_path / "kl.json"
    kl.write_text(
        _json.dumps([{"op_signature": "M", "kernel_kind": "dtype", "measured_ms": 648.17, "beat_baseline": True}])
    )
    out = sm.render_summary(kl, model="llama3_1_8b_p150", task="main")
    line = next((ln for ln in str(out).splitlines() if "eager per-op" in ln), "")
    assert "2464.18" in line and "648.17" in line, line
    assert "not measured" not in line, "the renderer could not find the row it was given the name for"
    assert not (tmp_path / "perf_measurements_model_main.jsonl").exists(), "collapsed to the shared key"


def test_the_ledger_key_is_never_the_literal_model_placeholder(tmp_path, monkeypatch):
    """THE SPLIT THIS ALMOST SHIPPED WITH.

    perf_mcp named artifacts from _MODEL_ROOT while run.py and the ledger read PERF_MCP_MODEL_NAME,
    which nothing exported -- so those two fell back to the literal "model". The writer wrote
    perf_measurements_llama..._main.jsonl, the reader looked for perf_measurements_model_main.jsonl,
    and every model on the box would have shared one "model" ledger: the unkeyed bug renamed.
    perf_mcp now publishes the key on import, so all three processes agree.
    """
    import importlib.util as ilu

    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    monkeypatch.delenv("PERF_MCP_MODEL_NAME", raising=False)
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(tmp_path / "models" / "demos" / "llama3_1_8b_p150"))
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(tmp_path / "m.json"))
    (tmp_path / "m.json").write_text('{"config": {}, "perf_test_resolved": {"path": "t.py"}}')

    spec = ilu.spec_from_file_location("pm_key_led_ut", _ROOT / "cc_optimize" / "perf_mcp.py")
    pm = ilu.module_from_spec(spec)
    sys.modules["pm_key_led_ut"] = pm
    spec.loader.exec_module(pm)

    m = _mod()
    name = m.ledger_path().name
    assert "llama3_1_8b_p150" in name, name
    assert name != "perf_measurements_model_main.jsonl", "fell back to the shared placeholder key"


def test_a_before_with_no_after_yet_is_still_reported(tmp_path, monkeypatch):
    """For most of a run only the BEFORE exists. Requiring both rows printed "not measured" over a
    reading the ledger actually held, hiding the anchor until the first after landed."""
    import importlib.util as ilu
    import json as _json

    m = _led(tmp_path, monkeypatch)
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 2464.18, depth="16", mode="eager", model="mdl")
    spec = ilu.spec_from_file_location("summary_partial_ut", _ROOT / "cc_optimize" / "summary.py")
    sm = ilu.module_from_spec(spec)
    sys.modules["summary_partial_ut"] = sm
    spec.loader.exec_module(sm)
    kl = tmp_path / "kl.json"
    kl.write_text(
        _json.dumps([{"op_signature": "M", "kernel_kind": "dtype", "measured_ms": 1.0, "beat_baseline": True}])
    )
    line = next(l for l in sm.render_summary(kl, model="mdl", task="main").splitlines() if "eager per-op" in l)
    assert "2464.18" in line, line
    # A half-drawn arrow into "(after not measured yet)" became a bare number: the anchor is still
    # shown, which is the point this test defends, without implying a comparison that has no second
    # term. See test_report_omits_uncomparable_pairs.py.
    assert "->" not in line, line
    assert line.strip() != "eager per-op device time: not measured (no ledger reading for this run)"
