# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The modeled floor is an ANCHOR: pinned where it is produced, only read where it is reported.

The defect: the floor was recomputed from whatever profile the current round produced, so optimizing
the model lowered its own target (537.23 -> 331.86 ms while the measurement got FASTER), and at-floor
fell from 83% to 54% during a run that improved. A target that retreats ahead of the measurement is
never reached.

Two owners, deliberately split:

    measurements.anchor()      WRITES, once, called by the producer of the number
    summary._floor_anchor()    READS only

because the first version pinned inside the renderer, which made producing a report a side effect --
the first report written pinned a value every later report then inherited, whatever its own input
said. Four unrelated tests in test_roofline_report.py started reading 341.47 ms out of a shared file.
"""
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(_ROOT / rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sm = _load("summary_anchor_ut", "cc_optimize/summary.py")
led = _load("led_anchor_ut", "cc_optimize/measurements.py")


def _ledger_env(tmp_path, monkeypatch):
    """The anchor lives in the ledger, so there is only ONE store to point at."""
    p = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("PERF_MCP_LEDGER", str(p))
    return p


def _pin(value, depth=16):
    return led.anchor(led.KIND_FLOOR, value, depth=str(depth), source="test")


def test_first_floor_is_pinned_and_returned(tmp_path, monkeypatch):
    p = _ledger_env(tmp_path, monkeypatch)
    assert _pin(537.2288) == 537.2288
    row = json.loads(p.read_text().splitlines()[0])
    assert (row["kind"], row["phase"], row["value_ms"], row["depth"]) == ("modeled_floor", "before", 537.2288, "16")


def test_later_smaller_floor_does_not_move_the_anchor(tmp_path, monkeypatch):
    """THE REGRESSION: the optimized build's lower floor must not become the new target."""
    _ledger_env(tmp_path, monkeypatch)
    _pin(537.2288)
    assert _pin(331.8585) == 537.2288
    assert _pin(341.47) == 537.2288


def test_anchor_survives_many_reprofiles(tmp_path, monkeypatch):
    _ledger_env(tmp_path, monkeypatch)
    _pin(537.2288)
    for i in range(25):
        assert _pin(500.0 - i * 10) == 537.2288


def test_depth_is_part_of_the_key(tmp_path, monkeypatch):
    """A 16-layer floor must not anchor a 2-layer profile -- that pairing produced the old
    'measured beats the floor' impossibility."""
    _ledger_env(tmp_path, monkeypatch)
    assert _pin(537.2288, 16) == 537.2288
    assert _pin(70.0, 2) == 70.0
    assert _pin(999.0, 16) == 537.2288


def test_unusable_floor_never_becomes_the_anchor(tmp_path, monkeypatch):
    """`record` owns this rule; the anchor inherits it rather than re-checking. inf slipped past an
    earlier hand-written `> 0` copy and would have been a PERMANENT floor."""
    p = _ledger_env(tmp_path, monkeypatch)
    for bad in (0, -5.0, None, float("nan"), float("inf"), "abc", [1]):
        assert _pin(bad) is None, bad
    assert not p.exists()
    assert _pin(537.2288) == 537.2288


def test_corrupt_ledger_line_degrades_to_reseeding(tmp_path, monkeypatch):
    _ledger_env(tmp_path, monkeypatch)
    (tmp_path / "ledger.jsonl").write_text("{not json\n")
    assert _pin(537.2288) == 537.2288


def test_an_unkeyed_ledger_is_never_anchored_on(tmp_path, monkeypatch):
    """An anchor is permanent, so taking one from the shared unkeyed file is how ANOTHER run's number
    becomes this run's target -- the same defect that made a foreign 0.06 ms an anchor."""
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    monkeypatch.delenv("PERF_MCP_MODEL_NAME", raising=False)
    monkeypatch.delenv("PERF_MCP_MODEL_ROOT", raising=False)
    assert led.anchor(led.KIND_FLOOR, 537.23, depth="16") is None
    assert led.anchor_value(led.KIND_FLOOR, depth="16") is None
    assert led.anchor(led.KIND_FLOOR, 537.23, depth="16", model="named_model") == 537.23


def test_anchor_shares_the_ledger_with_measured_rows_without_colliding(tmp_path, monkeypatch):
    """A floor row must not be mistaken for a timing row, or the headline would report the floor as
    a measurement."""
    _ledger_env(tmp_path, monkeypatch)
    led.record(led.KIND_EAGER, led.PHASE_BEFORE, 2464.18, depth="16", mode="eager")
    _pin(537.2288)
    led.record(led.KIND_EAGER, led.PHASE_AFTER, 615.69, depth="16", mode="eager")
    a, b = sm._ledger_pair(led.KIND_EAGER, "", "")
    assert (a["value_ms"], b["value_ms"]) == (2464.18, 615.69)
    assert _pin(331.86) == 537.2288


def test_rendering_reads_the_anchor_and_writes_NOTHING(tmp_path, monkeypatch):
    """THE DESIGN RULE: producing a report must not change what the next report says."""
    p = _ledger_env(tmp_path, monkeypatch)
    for floor in (331.86, 200.0, 999.0):
        sm._roofline_lines({"modeled_floor_ms": floor, "perf_layers": 16}, 615.69, None, "m", "main")
    assert not p.exists(), "the renderer wrote to the ledger: %s" % (p.read_text() if p.exists() else "")

    _pin(537.2288)
    before = p.read_text()
    for floor in (331.86, 200.0):
        sm._roofline_lines({"modeled_floor_ms": floor, "perf_layers": 16}, 615.69, None, "m", "main")
    assert p.read_text() == before, "the renderer appended to the ledger"


def test_report_shows_only_the_pinned_baseline_floor(tmp_path, monkeypatch):
    """RUN_REPORT.md has a FIXED structure and reports the baseline. One floor line, the pinned one:
    a second "this build's floor" line was noise in a document that goes out for confirmation, and
    printing two floors invites exactly the confusion the pin exists to remove."""
    _ledger_env(tmp_path, monkeypatch)
    _pin(537.2288)
    txt = "\n".join(sm._roofline_lines({"modeled_floor_ms": 331.8585, "perf_layers": 16}, 615.69, None, "m", "main"))
    assert "modeled floor       : 537.23 ms" in txt
    assert "331.86" not in txt, txt
    assert "this build's floor" not in txt, txt
    assert len([l for l in txt.splitlines() if "floor" in l and "at-floor" not in l]) == 1, txt


def test_at_floor_pct_is_computed_against_the_anchor(tmp_path, monkeypatch):
    _ledger_env(tmp_path, monkeypatch)
    _pin(537.2288)
    txt = "\n".join(sm._roofline_lines({"modeled_floor_ms": 331.8585, "perf_layers": 16}, 615.69, None, "m", "main"))
    assert "at-floor            : 87%" in txt, txt


def test_with_no_anchor_yet_the_current_floor_is_reported(tmp_path, monkeypatch):
    """First render of a fresh model: nothing is pinned yet, so the floor just computed IS the
    baseline -- and pinning it is what makes every later render report this same number."""
    _ledger_env(tmp_path, monkeypatch)
    txt = "\n".join(sm._roofline_lines({"modeled_floor_ms": 331.8585, "perf_layers": 16}, 615.69, None, "m", "main"))
    assert "modeled floor       : 331.86 ms" in txt
    assert "this build's floor" not in txt


def test_a_later_build_floor_never_appears_in_the_report(tmp_path, monkeypatch):
    """Whatever the current build recomputes, the report shows the pinned baseline only."""
    _ledger_env(tmp_path, monkeypatch)
    _pin(537.2288)
    for current in (540.0, 331.86, 120.0, 900.0):
        txt = "\n".join(sm._roofline_lines({"modeled_floor_ms": current, "perf_layers": 16}, 615.69, None, "m", "main"))
        assert "modeled floor       : 537.23 ms" in txt, current
        assert "this build's floor" not in txt, current


def test_beating_the_pinned_baseline_floor_is_reported_as_a_real_win(tmp_path, monkeypatch):
    """The floor is pinned to the BASELINE, so an optimized build that moves fewer bytes can legally
    beat it -- that is the goal. The report called it "impossible ... never bank this as a win" while
    llama3_1_8b_p150 sat at 534.44 ms against a 537.23 ms baseline floor with its own floor at 331.86.
    """
    _ledger_env(tmp_path, monkeypatch)
    _pin(537.2288)
    txt = "\n".join(sm._roofline_lines({"modeled_floor_ms": 331.8585, "perf_layers": 16}, 534.44, None, "m", "main"))
    assert "PAST BASELINE FLOOR" in txt, txt
    assert "impossible" not in txt and "never bank" not in txt, txt


def test_beating_the_builds_own_floor_is_still_called_suspect(tmp_path, monkeypatch):
    """No kernel beats the bound of the build it is running, so this really is a stale pairing --
    a 2-layer measurement against a 16-layer floor produced exactly this."""
    _ledger_env(tmp_path, monkeypatch)
    _pin(537.2288)
    txt = "\n".join(sm._roofline_lines({"modeled_floor_ms": 331.8585, "perf_layers": 16}, 100.0, None, "m", "main"))
    assert "stale/suspect" in txt and "never bank this as a win" in txt, txt
    assert "ABOVE_BAND" in txt and "PAST BASELINE FLOOR" not in txt


def test_with_no_current_floor_the_conservative_reading_is_kept(tmp_path, monkeypatch):
    """Nothing to compare against, so do not claim the win is real."""
    _ledger_env(tmp_path, monkeypatch)
    _pin(537.2288)
    txt = "\n".join(sm._roofline_lines({"modeled_floor_ms": None, "perf_layers": 16}, 300.0, None, "m", "main"))
    assert "PAST BASELINE FLOOR" in txt or "stale/suspect" in txt, txt


def test_the_report_still_carries_exactly_one_floor_line_in_every_case(tmp_path, monkeypatch):
    _ledger_env(tmp_path, monkeypatch)
    _pin(537.2288)
    for fm in (900.0, 700.0, 615.69, 534.44, 100.0):
        txt = "\n".join(sm._roofline_lines({"modeled_floor_ms": 331.8585, "perf_layers": 16}, fm, None, "m", "main"))
        floors = [l for l in txt.splitlines() if l.strip().startswith("modeled floor")]
        assert len(floors) == 1 and "537.23" in floors[0], (fm, txt)
