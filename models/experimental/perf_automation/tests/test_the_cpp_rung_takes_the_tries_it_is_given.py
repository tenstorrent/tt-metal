"""The C++ rung takes as many clean tries per op as the operator grants, and can be the whole ladder.

The rung was hard-wired to one clean try. On minicpm5_2b (2026-10-19) one C++ try on the prefill gate/up
kernel -- about 55 ms of a stage 120 ms over its band -- closed it, and the run spent its last rounds
with nothing to work. --max-cpp-attempts (PERF_MCP_MAX_CPP_ATTEMPTS) sets the cap, and --cpp-only
(PERF_MCP_CPP_ONLY=1) offers that rung alone. Both default to the old behaviour. The ladder that hands
the rung out and the recorder that accepts it read the same cap, and the server -- which inherits no
environment -- is handed both settings. Op codes and shapes here are this file's own.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]
_CC = _PA / "cc_optimize"
_REPO = _PA.parents[2]

_OP = "CustomKernelOp 512x256 @ 256x1024"


@pytest.fixture
def pm(monkeypatch, tmp_path):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_KERNEL_LOG", str(tmp_path / "kl.json"))
    for k in ("PERF_MCP_MAX_CPP_ATTEMPTS", "PERF_MCP_MAX_TTL_ATTEMPTS", "PERF_MCP_CPP_ONLY"):
        monkeypatch.delenv(k, raising=False)
    spec = importlib.util.spec_from_file_location("pm_cpp_rung_ut", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "_ttl_available", lambda: False)
    monkeypatch.setattr(m, "_load_attempts_all", lambda: [])
    monkeypatch.setattr(m, "_load_attempts", lambda *a, **k: [])
    return m


def _open_op():
    return {"op_code": "CustomKernelOp", "shape": "512x256 @ 256x1024", "grid": "full", "bound_by": "compute"}


_N = {"n": 0}


def _row(kind, **kw):
    _N["n"] += 1
    r = {
        "op_signature": _OP,
        "kernel_kind": kind,
        "measured_ms": 10.0 + _N["n"] / 100.0,
        "note": "try %d" % _N["n"],
        "kernel_detected_in_source": True,
    }
    r.update(kw)
    return r


# Every rung below C++ spent once, so the ordinary ladder has reached it.
_BELOW_CPP = [_row(k) for k in ("grid", "fidelity", "shard", "structural")]


def _ladder(pm, attempts):
    return pm._op_ladder_status(_open_op(), _OP, list(attempts))


def test_by_default_one_clean_cpp_try_closes_the_rung(pm):
    assert _ladder(pm, _BELOW_CPP)[1] == "cpp"
    done, rung, _ = _ladder(pm, _BELOW_CPP + [_row("cpp")])
    assert done and rung == "done"
    assert pm._rung_allowance(_OP, "cpp", _BELOW_CPP + [_row("cpp")]) == (1, 1)


def test_a_raised_cap_keeps_offering_cpp_and_names_the_try(pm, monkeypatch):
    monkeypatch.setenv("PERF_MCP_MAX_CPP_ATTEMPTS", "3")
    tries = []
    for n in (1, 2):
        tries.append(_row("cpp"))
        done, rung, reason = _ladder(pm, _BELOW_CPP + tries)
        assert not done and rung == "cpp"
        assert "attempt %d of 3" % (n + 1) in reason
        assert "build on the previous kernel" in reason
    tries.append(_row("cpp"))
    assert _ladder(pm, _BELOW_CPP + tries)[0] is True, "the third clean try spends the cap"


def test_the_recorder_accepts_exactly_as_many_as_the_ladder_hands_out(pm, monkeypatch):
    monkeypatch.setenv("PERF_MCP_MAX_CPP_ATTEMPTS", "3")
    tries = [_row("cpp"), _row("cpp")]
    assert pm._rung_allowance(_OP, "cpp", _BELOW_CPP + tries) == (2, 3)
    assert pm._rung_allowance(_OP, "grid", _BELOW_CPP + tries)[1] == 1, "knob allowances are untouched"


def test_wedges_still_close_the_rung_at_their_own_cap(pm, monkeypatch):
    monkeypatch.setenv("PERF_MCP_MAX_CPP_ATTEMPTS", "10")
    wedges = [_row("cpp", wedged=True, measured_ms=None) for _ in range(pm._MAX_KERNEL_WEDGES)]
    assert _ladder(pm, _BELOW_CPP + wedges)[0] is True


def test_an_unusable_cap_falls_back_to_one(pm, monkeypatch):
    for bad in ("0", "-2", "many", ""):
        monkeypatch.setenv("PERF_MCP_MAX_CPP_ATTEMPTS", bad)
        assert pm._kernel_cap("cpp") == 1, bad
    assert pm._kernel_cap("grid") == 1


def test_cpp_only_offers_cpp_first_with_no_cheaper_rung_tried(pm, monkeypatch):
    assert _ladder(pm, [])[1] != "cpp", "the ordinary ladder starts at a knob"
    monkeypatch.setenv("PERF_MCP_CPP_ONLY", "1")
    done, rung, reason = _ladder(pm, [])
    assert not done and rung == "cpp"
    assert "C++-only" in reason


def test_cpp_only_clears_an_op_once_its_cpp_tries_are_spent(pm, monkeypatch):
    monkeypatch.setenv("PERF_MCP_CPP_ONLY", "1")
    monkeypatch.setenv("PERF_MCP_MAX_CPP_ATTEMPTS", "2")
    assert _ladder(pm, [_row("cpp")])[1] == "cpp"
    done, rung, _ = _ladder(pm, [_row("cpp"), _row("cpp")])
    assert done and rung == "done"


def test_cpp_only_clears_what_has_no_kernel_to_write(pm, monkeypatch):
    monkeypatch.setenv("PERF_MCP_CPP_ONLY", "1")
    gap = {"op_code": "host_overhead", "bucket": "host_fallback", "bound_by": "host"}
    assert pm._op_ladder_status(gap, "host_overhead", [])[0] is True


def _gate(pm, monkeypatch):
    op = dict(_open_op(), gap_ms=1.0, eff_gap_ms=1.0, device_ms=2.0, count=1, bucket="other")
    prof = {
        "device_ms": 40.0,
        "stage_buckets": {"pass": [{"id": "b", "device_ms": 10.0, "top_ops": [dict(op)]}]},
        "open_ops": [dict(op)],
    }
    monkeypatch.setattr(pm, "_model_key", lambda: "m")
    monkeypatch.setattr(pm, "read_stage_ms", lambda **k: {})
    monkeypatch.setattr(pm, "read_stage_spread", lambda **k: {})
    monkeypatch.setattr(pm, "_profile_with_zero_row_retry", lambda *a, **k: dict(prof))
    monkeypatch.setattr(pm.roofline, "residual_report", lambda p, env: {"open_ops": [dict(op, at_floor=False)]})
    monkeypatch.setattr(pm, "_persist_throughput", lambda *a, **k: None)
    monkeypatch.setattr(pm, "_stages_short_of_achievable", lambda: [])
    called = []
    for g in ("_host_gate", "_decode_gate", "_conv_gate", "_fold_gate"):
        monkeypatch.setattr(pm, g, lambda *a, _g=g, **k: called.append(_g))
    return pm.termination_check(), called


def test_cpp_only_consults_no_structural_gate(pm, monkeypatch):
    out, called = _gate(pm, monkeypatch)
    assert called, "an ordinary run consults the structural gates"
    monkeypatch.setenv("PERF_MCP_CPP_ONLY", "1")
    out, called = _gate(pm, monkeypatch)
    assert called == []
    assert out["next_target"]["rung"] == "cpp", out["next_target"]


def test_the_server_is_handed_both_settings(monkeypatch, tmp_path):
    sys.path.insert(0, str(_PA))
    spec = importlib.util.spec_from_file_location("_run_cpp_flags", str(_CC / "run.py"))
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    pipe = {"perf_test": "t.py::test_it", "pcc_test": "p::c"}

    def env():
        return run._mcp_config(tmp_path, "/m.json", pipe, "0", "/k.json")["mcpServers"]["perf-mcp"]["env"]

    for k in ("PERF_MCP_MAX_CPP_ATTEMPTS", "PERF_MCP_CPP_ONLY"):
        monkeypatch.delenv(k, raising=False)
    assert "PERF_MCP_MAX_CPP_ATTEMPTS" not in env() and "PERF_MCP_CPP_ONLY" not in env()
    monkeypatch.setenv("PERF_MCP_MAX_CPP_ATTEMPTS", "4")
    monkeypatch.setenv("PERF_MCP_CPP_ONLY", "1")
    assert env()["PERF_MCP_MAX_CPP_ATTEMPTS"] == "4"
    assert env()["PERF_MCP_CPP_ONLY"] == "1"


def test_the_optimize_flags_parse_and_reach_the_environment(monkeypatch):
    sys.path.insert(0, str(_REPO / "scripts"))
    from tt_hw_planner import cli
    from tt_hw_planner.commands import optimize

    seen = []
    monkeypatch.setattr(cli, "_autocommit_wrap", lambda fn, name: fn)
    monkeypatch.setattr(cli, "cmd_optimize", lambda a: seen.append(a) or 0)
    cli.main(["optimize", "some/model", "--max-cpp-attempts", "5", "--cpp-only"])
    cli.main(["optimize", "some/model"])
    assert (seen[0].max_cpp_attempts, seen[0].cpp_only) == (5, True)
    assert (seen[1].max_cpp_attempts, seen[1].cpp_only) == (None, False)

    for k in ("PERF_MCP_MAX_CPP_ATTEMPTS", "PERF_MCP_CPP_ONLY"):
        monkeypatch.setenv(k, "x")  # registers the restore, so what the call sets does not leak
        monkeypatch.delenv(k)
    assert optimize._apply_cpp_flags(seen[1]) is None
    assert "PERF_MCP_MAX_CPP_ATTEMPTS" not in os.environ and "PERF_MCP_CPP_ONLY" not in os.environ
    assert optimize._apply_cpp_flags(seen[0]) is None
    assert os.environ["PERF_MCP_MAX_CPP_ATTEMPTS"] == "5" and os.environ["PERF_MCP_CPP_ONLY"] == "1"
    assert optimize._apply_cpp_flags(argparse.Namespace(max_cpp_attempts=0, cpp_only=False))
