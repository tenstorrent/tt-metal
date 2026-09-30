"""A ✓ in the report is a commit, and a candidate reverted for accuracy says so -- with its PCC.

Qwen-Image-Edit on a WH Galaxy (2026-09-29): the report showed 5 ✓win rows for 2 commits. The
skill's loop is measure_candidate -> git_commit -> record_kernel_attempt, but the target learned its
measurement only in record_kernel_attempt -- one call after the commit -- so _record_committed_win
found no measured_ms, wrote no commit row, and with none anywhere the report fell back to ticking
every faster try. Two of them had been reverted for PCC (0.654 and 0.939 against 0.95) and one was a
0.9 ms blip; no row recorded its accuracy, so nothing in the table could tell them apart.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

_PA = Path(__file__).resolve().parents[1]


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


@pytest.fixture()
def summary():
    spec = importlib.util.spec_from_file_location("_summary_ticks", _PA / "cc_optimize" / "summary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- the commit row gets written whichever order the agent calls things in --------------------------


def test_measure_candidate_gives_the_target_its_measurement():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("def measure_candidate(")
    body = src[i : src.index("\n@mcp.tool()", i)]
    assert body.index("_stamp_target_measurement(dev)") < body.index('"verdict": "valid"')


def test_a_commit_after_a_measurement_is_banked(mcp):
    m = mcp
    m._persist_target({"op": "SomeOp", "rung": "fold"})
    m._stamp_target_measurement(12.5)  # what measure_candidate now does
    m._record_committed_win("lever", sha="abc1234")
    rows = [a for a in m._load_attempts() if a.get("commit_record")]
    assert len(rows) == 1 and rows[0]["commit"] == "abc1234" and rows[0]["beat_baseline"] is True


def test_a_commit_with_no_measurement_is_still_not_a_win(mcp):
    m = mcp
    m._persist_target({"op": "SomeOp", "rung": "fold"})
    m._record_committed_win("housekeeping", sha="abc1234")
    assert not [a for a in m._load_attempts() if a.get("commit_record")]


# -- every attempt owns exactly one PCC reading -------------------------------------------------------


def test_an_attempt_owns_the_pcc_reading_it_ran_and_only_once(mcp):
    m = mcp
    m.record_gate_verdict("pcc", "pcc_low", pcc=0.939, threshold=0.95, measurement_id="p-1")
    assert m._attempt_pcc_verdict() == {"pcc_status": "pcc_low", "pcc": 0.939, "pcc_threshold": 0.95}
    assert m._attempt_pcc_verdict() == {}, "a later attempt that ran no check borrows nothing"
    m.record_gate_verdict("pcc", "ok", pcc=0.9625, threshold=0.95, measurement_id="p-2")
    assert m._attempt_pcc_verdict()["pcc"] == 0.9625


def test_a_pcc_reading_without_an_id_is_not_ownable(mcp):
    m = mcp
    m.record_gate_verdict("pcc", "ok", pcc=0.99)
    assert m._attempt_pcc_verdict() == {}


def test_the_two_claims_do_not_share_a_marker(mcp):
    m = mcp
    assert m._consumed_verdict_path() != m._consumed_verdict_path("pcc")
    assert m._consumed_verdict_path().name.startswith("perf_mcp_fullpipe_consumed_"), "same file as before"


def test_the_attempt_row_carries_it():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("def record_kernel_attempt(")
    body = src[i : src.index("\n@mcp.tool()", i)]
    assert "**_attempt_pcc_verdict()," in body
    assert body.index("owns no end-to-end measurement") < body.index(
        "**_attempt_pcc_verdict(),"
    ), "a refused record must not consume the PCC reading the retry will need"


def test_a_fresh_start_clears_the_pcc_marker():
    src = (_PA / "agent" / "fresh_start.py").read_text()
    assert '"perf_mcp_pcc_consumed_*.json"' in src


# -- the one win rule ---------------------------------------------------------------------------------


def _row(delta, **kw):
    r = {"op_signature": "Op", "kernel_kind": "fold", "fullpipe_delta_ms": delta, "fullpipe_ms": 100.0 + delta}
    r.update(kw)
    return r


def test_a_pcc_failure_is_never_a_win(summary):
    led = summary._ledger()
    rows = [_row(-50.0), _row(-40.0, pcc_status="pcc_low", pcc=0.939), _row(-30.0, pcc_status="ok", pcc=0.96)]
    assert led.winning_indices(rows) == {0, 2}
    assert led.pcc_failed(rows[1]) and not led.pcc_failed(rows[0]) and not led.pcc_failed(rows[2])


# -- the result cell ----------------------------------------------------------------------------------


def test_a_banked_win_names_its_commit(summary):
    win = _row(-50.0)
    rows = [win, {"commit_record": True, "commit": "e8696829b05aa", "fullpipe_ms": win["fullpipe_ms"]}]
    assert summary._attempt_result(win, True, rows) == "✓ win (e8696829b05)"
    assert summary._banking_commit(win, rows) is rows[1]
    assert summary._banking_commit(rows[1], rows) is None, "a commit row banks nothing itself"


def test_a_reverted_candidate_says_why(summary):
    rows = [_row(-708.7, pcc_status="pcc_low", pcc=0.654, pcc_threshold=0.95), {"commit_record": True}]
    assert summary._attempt_result(rows[0], False, rows) == "✗ PCC 0.654 < 0.95"
    assert summary._attempt_result(_row(-1.0, pcc_status="crash"), False, rows) == "✗ PCC crash"
    assert summary._attempt_result(_row(-0.9), False, rows) == "· not kept"
    assert summary._attempt_result(_row(+7.0), False, rows) == "· no gain"


def test_a_log_from_before_commit_rows_reads_as_it_did(summary):
    rows = [_row(-0.9), _row(+7.0), _row(-3.0, wedged=True)]
    assert summary._attempt_result(rows[0], True, rows) == "✓ win"
    assert summary._attempt_result(rows[1], False, rows) == "· no gain"
    assert summary._attempt_result(rows[2], False, rows) == "· wedged"


def test_the_table_prints_the_pcc_and_skips_commit_rows():
    src = (_PA / "cc_optimize" / "summary.py").read_text()
    assert '"1CQ \\u0394 vs current", "PCC", "result")' in src
    assert 'if not isinstance(a, dict) or a.get("commit_record"):' in src
    assert "res = _attempt_result(a, _i in _wins, attempts)" in src


def test_two_wins_on_one_rung_name_their_own_commits(summary):
    first, second = _row(-1681.7), _row(-11.1)
    rows = [
        first,
        second,
        {
            "commit_record": True,
            "commit": "aaaaaaaaaaa1",
            "fullpipe_ms": first["fullpipe_ms"],
            "op_signature": "Op",
            "kernel_kind": "fold",
        },
        {
            "commit_record": True,
            "commit": "bbbbbbbbbbb2",
            "fullpipe_ms": second["fullpipe_ms"],
            "op_signature": "Op",
            "kernel_kind": "fold",
        },
    ]
    assert summary._banking_commit(first, rows)["commit"] == "aaaaaaaaaaa1"
    assert summary._banking_commit(second, rows)["commit"] == "bbbbbbbbbbb2", "not the first commit on the rung"
