"""A full-pipeline check starts at the trace region this model last fit in on this board.

The grow-and-retry fixed env for ONE run, so every check started at the DRAM-derived default again and
ran the model twice: once to overflow, once for real. Qwen-Image-Edit on a WH Galaxy (2026-10-02):
192 MB -> overflow -> 1.43 GB on every check, ~30 min each against ~10 for a single run. The size that
worked is remembered per model and per board (arch, per-chip DRAM, chip count, as Step 1 detected them)
and is never carried past the current board's own ceiling.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]


@pytest.fixture
def pm(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location("pm_trace_region_ut", str(_PA / "cc_optimize" / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "_model_key", lambda: "m")
    monkeypatch.setattr(m, "_ENV", {"arch": "board_a", "dram_capacity_bytes": 1000, "device_count": 8})
    monkeypatch.setattr(m, "_TRACE_REGION_DEFAULT", 100)
    monkeypatch.setattr(m, "_TRACE_REGION_MAX", 10_000)
    return m


def test_nothing_is_remembered_at_first(pm):
    assert pm.remembered_trace_region() == 0


def test_a_size_that_fit_is_remembered_and_only_grows(pm):
    pm._remember_trace_region(1430)
    assert pm.remembered_trace_region() == 1430
    pm._remember_trace_region(900)  # a later run that fit in less does not shrink it
    assert pm.remembered_trace_region() == 1430
    pm._remember_trace_region(2000)
    assert pm.remembered_trace_region() == 2000


def test_the_default_is_not_worth_remembering(pm):
    pm._remember_trace_region(100)
    pm._remember_trace_region(50)
    assert pm.remembered_trace_region() == 0
    assert not pm._trace_region_memory_path().exists()


def test_another_board_does_not_inherit_it(pm, monkeypatch):
    pm._remember_trace_region(1430)
    monkeypatch.setattr(pm, "_ENV", {"arch": "board_b", "dram_capacity_bytes": 4000, "device_count": 4})
    assert pm.remembered_trace_region() == 0
    pm._remember_trace_region(3000)
    doc = json.loads(pm._trace_region_memory_path().read_text())
    assert sorted(doc.values()) == [1430, 3000], "each board keeps its own"


def test_an_undetected_board_remembers_nothing(pm, monkeypatch):
    monkeypatch.setattr(pm, "_ENV", {"arch": "board_a"})
    pm._remember_trace_region(1430)
    assert pm.remembered_trace_region() == 0


def test_an_unreadable_memory_reads_as_nothing(pm):
    pm._trace_region_memory_path().write_text("{not json")
    assert pm.remembered_trace_region() == 0
    pm._remember_trace_region(1430)  # and is replaced, not crashed on
    assert pm.remembered_trace_region() == 1430


def test_the_check_starts_there_within_the_boards_ceiling_and_remembers_only_a_measured_run():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("_mem_reg = remembered_trace_region()")
    seg = src[i : i + 300]
    assert "_mem_reg <= _TRACE_REGION_MAX" in seg and 'env["TT_PERF_TRACE_REGION"] = str(_mem_reg)' in seg
    j = src.index("out, r = _grow_trace_region_and_retry(cmd, repo, env, out, r)\n")
    seg = src[j : j + 500]
    assert "any(v > 0 for v in _ptr(out" in seg and "_remember_trace_region(" in seg


def test_fresh_forgets_it():
    from agent import fresh_start

    assert "perf_mcp_trace_region_*.json" in fresh_start._STATE_GLOBS
