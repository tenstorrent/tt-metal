# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""`eager` is a CONCLUSION. The count it was drawn from was printed and thrown away.

trace_replay decides a stage's path by COUNTING op dispatches during the last warmup: a traced
stage issues one, an eager stage issues hundreds. It prints both --

    TRACE_STAGE_OPS[prefill]=0 path=trace+1cq

-- and perf_mcp parsed the label off the TRACE_STAGE_MS line while discarding the count entirely.
So a reader who doubted the label had nothing to check it against, and gemma-3's prefill was
diagnosed as eager twice from the label alone, once wrongly -- a diagnosis that cost a session and
was only settled by re-running the workload by hand to read a number the tool had already measured.

The count is now parsed, persisted beside the timing it belongs to, and printed next to the label
it justifies. Same file, same run stamp, same single reader: evidence and conclusion do not get to
come from different runs.
"""

import importlib.util
import json
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"
sys.path.insert(0, str(_PA))
sys.path.insert(0, str(_PA.parent.parent.parent))
sys.path.insert(0, str(_CC))


def _pm(monkeypatch, tmp_path):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(tmp_path / "gemma3"))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-under-test")
    (tmp_path / "gemma3").mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("pmcp_ops_evidence", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_count_survives_the_round_trip(monkeypatch, tmp_path):
    m = _pm(monkeypatch, tmp_path)
    m._persist_stage_ms({"prefill": 54.81, "decode": 32.22}, {"prefill": "trace+1cq"}, {"prefill": 128}, {"prefill": 0})
    assert m.read_stage_ops() == {"prefill": 0}
    assert m.read_stage_paths() == {"prefill": "trace+1cq"}


def test_zero_is_a_reading_not_an_absence(monkeypatch, tmp_path):
    """A traced stage dispatches ZERO ops in the counted window -- the strongest evidence there is,
    and the one a truthiness filter drops."""
    m = _pm(monkeypatch, tmp_path)
    m._persist_stage_ms({"prefill": 54.81}, None, None, {"prefill": 0})
    assert m.read_stage_ops() == {"prefill": 0}


def test_a_file_from_another_run_yields_no_evidence_either(monkeypatch, tmp_path):
    """The count goes through the same single reader as the timings, so it obeys the same freshness
    rule. Evidence from one run beside a conclusion from another is the failure this fixes."""
    m = _pm(monkeypatch, tmp_path)
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-A")
    m._persist_stage_ms({"prefill": 54.81}, {"prefill": "trace+1cq"}, None, {"prefill": 0})
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-B")
    assert m.read_stage_ops() == {} and m.read_stage_ms() == {}


def test_a_stamped_file_without_the_ops_key_degrades_rather_than_crashing(monkeypatch, tmp_path):
    """Absent evidence is not a crash and not a zero -- it is simply no evidence, while the timings
    it was written beside stay readable."""
    m = _pm(monkeypatch, tmp_path)
    (tmp_path / "perf_mcp_stage_ms_gemma3_main.json").write_text(
        json.dumps({"run": "run-under-test", "stages": {"decode": 32.22}})
    )
    assert m.read_stage_ops() == {}
    assert m.read_stage_ms() == {"decode": 32.22}


def test_a_pre_stamping_file_yields_nothing_at_all(monkeypatch, tmp_path):
    """The gemma-3 case: a file predating both `run` and `paths` carried a pre-fix EAGER prefill of
    100.46 ms, and with no paths key the roofline could not even mark it eager. Refused whole."""
    m = _pm(monkeypatch, tmp_path)
    (tmp_path / "perf_mcp_stage_ms_gemma3_main.json").write_text(
        json.dumps({"stages": {"prefill": 100.4624, "decode": 50.8251}})
    )
    assert m.read_stage_ms() == {} and m.read_stage_ops() == {} and m.read_stage_paths() == {}


def test_the_marker_is_parsed_from_the_workloads_own_line(monkeypatch, tmp_path):
    """Pinned against trace_replay's format string, so a change on either side fails here rather
    than silently emptying the evidence."""
    emitter = (_PA / "agent" / "trace_replay.py").read_text()
    assert "TRACE_STAGE_OPS[%s]=%d path=%s" in emitter
    parser = (_CC / "perf_mcp.py").read_text()
    assert '"TRACE_STAGE_OPS[" in line' in parser


def test_the_report_prints_the_count_beside_the_label(monkeypatch, tmp_path):
    """The label alone reads as a verdict handed down. With the count it is a measurement."""
    import summary as S

    rows = S._roofline_tables(
        unit="token",
        theo=31.0,
        band=(18.6, 24.8),
        measured=31.0,
        bw_gbps=400.0,
        peak_bw_gbps=512.0,
        active_bytes=int(11e9),
        per_unit_ms=32.22,
        profile=None,
        stage_ms={"prefill": 91.33, "decode": 32.22},
        stage_paths={"prefill": "eager"},
        stage_ops={"prefill": 512},
    )
    text = "\n".join(rows or [])
    assert "512 op dispatches" in text, text
    assert "not comparable to a traced band" in text


def test_no_count_leaves_the_label_alone(monkeypatch, tmp_path):
    """Absent evidence must not become a fabricated `0 op dispatches` -- which would read as PROOF
    the stage was traced, the exact opposite of the truth."""
    import summary as S

    rows = S._roofline_tables(
        unit="token",
        theo=31.0,
        band=(18.6, 24.8),
        measured=31.0,
        bw_gbps=400.0,
        peak_bw_gbps=512.0,
        active_bytes=int(11e9),
        per_unit_ms=32.22,
        profile=None,
        stage_ms={"prefill": 91.33, "decode": 32.22},
        stage_paths={"prefill": "eager"},
    )
    text = "\n".join(rows or [])
    assert "op dispatches" not in text
    assert "[eager — not comparable to a traced band]" in text
