"""The report's peaks come from the hardware the run detected -- never from a default machine.

WH Galaxy, 2026-09-29: the run's manifest recorded arch=wormhole, 64 worker cores, 1.0 TFLOPS/core
at HiFi4, 288 GB/s, and its ops were ranked with exactly that. But six report/peak sites read an
environment variable nothing sets and fell back to a typed arch name, so the pinned peak and every
compute roof in RUN_REPORT.md used Blackhole's 175.5 TFLOPS -- 2.7x the 64 the chip has.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from agent import roofline
from agent.perf_target import chip_peak_flops

_PA = Path(__file__).resolve().parents[1]

# A detected env the way a manifest records one (values are this fixture's, not a table's).
_ENV = {
    "arch": "wormhole",
    "worker_cores": 64,
    "grid_x": 8,
    "grid_y": 8,
    "dram_bw_gbps": 288.0,
    "dram_capacity_bytes": 12 * 1024**3,
    "peak_tflops_per_core": {"lofi": 4.0, "hifi2": 2.0, "hifi3": 1.33, "hifi4": 1.0},
}


@pytest.fixture
def summary(monkeypatch):
    monkeypatch.delenv("PERF_MCP_MANIFEST", raising=False)
    spec = importlib.util.spec_from_file_location("_summary_hw", _PA / "cc_optimize" / "summary.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_facts_are_the_detected_env(summary):
    summary.set_run_env(_ENV)
    assert summary._hw_facts() == roofline._facts(_ENV)
    assert chip_peak_flops(summary._hw_facts(), "hifi4") == 64e12, "64 cores x 1.0 TFLOPS on this chip"


def test_the_manifest_path_is_read_when_no_env_was_handed_over(summary, tmp_path, monkeypatch):
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"env": _ENV}))
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(m))
    assert summary._hw_facts()["worker_cores"] == 64


def test_no_machine_named_prices_no_compute_roof(summary):
    assert summary._hw_facts() == {}
    assert summary._fidelity_breakdown({"buckets": []}) == (None, None)


def test_a_different_machine_prices_its_own_peak(summary):
    other = dict(_ENV, arch="other", worker_cores=10, peak_tflops_per_core={"hifi4": 3.0})
    summary.set_run_env(other)
    assert chip_peak_flops(summary._hw_facts(), "hifi4") == 30e12


def test_no_typed_machine_default_remains():
    for rel in ("cc_optimize/summary.py", "cc_optimize/perf_mcp.py"):
        src = (_PA / rel).read_text()
        assert "PERF_MCP_ARCH" not in src, rel
        for arch in ("blackhole", "wormhole"):
            code = [ln for ln in src.splitlines() if ('"%s"' % arch) in ln and not ln.lstrip().startswith("#")]
            assert code == [], (rel, code)


def test_both_renderers_hand_over_the_run_env():
    mcp = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = mcp.index("def _summary_mod(")
    assert "mod.set_run_env(_ENV)" in mcp[i : mcp.index("\ndef ", i + 1)]
    assert "roofline._facts(_ENV)" in mcp, "the pinned peak is the detected chip's"
    run = (_PA / "cc_optimize" / "run.py").read_text()
    j = run.index("def _emit_summary(")
    body = run[j : run.index("\ndef ", j + 1)]
    assert "mod.set_run_env(_run_env)" in body and "_latest_manifest(" in body
    assert "(manifest or {})" not in body, "no name this function does not have"
