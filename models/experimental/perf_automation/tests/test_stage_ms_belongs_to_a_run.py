# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A stage measurement states which run made it, and one reader enforces that for all three fields.

perf_mcp_stage_ms_<model>_<task>.json is keyed by (model, task) -- a key that OUTLIVES the run. It
held the prefill/decode split, and later the trace path and the observed prompt length, and nothing
recorded which run wrote any of it. So a report read whatever was there last:

    prefill 91.33 ms   from a run hours earlier, rendered beside a fresh decode
    decode  39.14 ms   in Block-level timing, while the Roofline said 32.22 for the same stage

Two numbers for one stage in one report, because the headline came from this run and the split from
another. Same lifetime defect the device-recovery counters had, and the same fix: stamp the file with
PERF_MCP_RUN_ID and refuse a document belonging to a different run.

The three fields are read through ONE function. They were three separate lookups that each opened the
file and dug out their own key, which is how a freshness rule gets applied in one place and forgotten
in the other two.
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
_spec = _ilu.spec_from_file_location("pm_stage_ut", str(_PA / "cc_optimize" / "perf_mcp.py"))
PM = _ilu.module_from_spec(_spec)
sys.modules["pm_stage_ut"] = PM
_spec.loader.exec_module(PM)

_DOC = {"stages": {"prefill": 92.27, "decode": 32.21}, "paths": {"prefill": "trace+1cq"}, "isl": {"prefill": 128}}


def _write(tmp_path, run=None):
    d = dict(_DOC)
    if run is not None:
        d["run"] = run
    (tmp_path / "perf_mcp_stage_ms_m_main.json").write_text(json.dumps(d))


def test_the_writer_stamps_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-A")
    monkeypatch.setattr(PM, "_MODEL_ROOT", Path("m"), raising=False)
    PM._persist_stage_ms({"prefill": 1.0}, {"prefill": "trace+1cq"}, {"prefill": 128})
    doc = json.loads((tmp_path / "perf_mcp_stage_ms_m_main.json").read_text())
    assert doc["run"] == "run-A"


def test_a_document_from_another_run_is_refused(tmp_path, monkeypatch):
    """The case that produced a stale prefill beside a fresh decode."""
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    _write(tmp_path, run="run-A")
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-B")
    assert PM.read_stage_ms(model="m", task="main") == {}
    assert PM.read_stage_paths(model="m", task="main") == {}
    assert PM.read_stage_isl(model="m", task="main") == 0


def test_the_same_run_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    _write(tmp_path, run="run-A")
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-A")
    assert PM.read_stage_ms(model="m", task="main")["prefill"] == 92.27
    assert PM.read_stage_paths(model="m", task="main")["prefill"] == "trace+1cq"
    assert PM.read_stage_isl(model="m", task="main") == 128


def test_an_unstamped_document_is_refused(tmp_path, monkeypatch):
    """A MEASUREMENT WITH NO PROVENANCE IS NOT A MEASUREMENT.

    This was accepted, on the reasoning that an unstamped file predates stamping and refusing it
    would blank the report for anyone who had not re-run. The cost showed up immediately: gemma-3's
    report rendered `prefill 100.46 ms` from a file written 2026-08-07T00:07 -- forty hours BEFORE
    the fix that let prefill trace at all -- beside a headline from a run that had measured nothing.
    That file also predated `paths`, so the roofline could not mark it eager, and a pre-fix eager
    number rendered wearing a traced one's clothes. It was read as "prefill is still eager today".

    A blank says `not measured`, which is true and prompts a measurement. This said a number, which
    was false."""
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    _write(tmp_path)  # no run key
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-B")
    assert PM.read_stage_ms(model="m", task="main") == {}
    assert PM.read_stage_paths(model="m", task="main") == {}


def test_no_run_id_still_reads_a_stamped_document(tmp_path, monkeypatch):
    """A reader that cannot name its own run can still trust a document that names ITS run: the
    stamp is what makes the file attributable. Only the unstamped case is refused."""
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("PERF_MCP_RUN_ID", raising=False)
    _write(tmp_path, run="run-A")
    assert PM.read_stage_ms(model="m", task="main")["prefill"] == 92.27


def test_all_three_fields_share_one_reader():
    """Three copies of the lookup is how a freshness rule gets enforced once and forgotten twice."""
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    assert src.count("_read_stage_doc(") >= 4  # the definition plus all three call sites
    i = src.index("def read_stage_ms")
    j = src.index("def read_stage_isl")
    assert 'json.loads((base / ("perf_mcp_stage_ms' not in src[i:j], "a reader is still opening the file itself"
