"""Unit tests for the memory-capped scope decision in commands/optimize.py.

A runaway build in an optimize run can exhaust host RAM and let the kernel's GLOBAL OOM-killer take
down the whole session. `_memory_scope_argv` decides whether to re-run the process inside a
systemd-run cgroup scope with a `memory.max`, so a runaway is killed in-scope and the supervisor
survives. These tests exercise the decision only (never the exec), with no device and no real scope.
"""
import importlib

OPT = importlib.import_module("scripts.tt_hw_planner.commands.optimize")


def test_scope_argv_wraps_cmdline_with_a_memory_cap(monkeypatch):
    monkeypatch.delenv("PERF_MCP_MEM_SCOPE", raising=False)
    monkeypatch.delenv("PERF_MCP_DISABLE_MEM_CAP", raising=False)
    monkeypatch.setattr(OPT.shutil, "which", lambda _n: "/usr/bin/systemd-run")
    argv = OPT._memory_scope_argv(["python", "-m", "scripts.tt_hw_planner", "optimize", "x"])
    assert argv is not None
    assert argv[0].endswith("systemd-run") and "--scope" in argv
    caps = [a for a in argv if a.startswith("MemoryMax=")]
    assert caps and int(caps[0].split("=")[1]) > 0, "must set a positive MemoryMax byte count"
    assert argv[-4:] == ["python", "-m", "scripts.tt_hw_planner", "optimize", "x"][-4:]


def test_scope_is_skipped_when_already_scoped(monkeypatch):
    monkeypatch.setenv("PERF_MCP_MEM_SCOPE", "1")
    monkeypatch.setattr(OPT.shutil, "which", lambda _n: "/usr/bin/systemd-run")
    assert OPT._memory_scope_argv(["python", "x"]) is None


def test_scope_is_skipped_by_the_shared_escape_hatch(monkeypatch):
    monkeypatch.delenv("PERF_MCP_MEM_SCOPE", raising=False)
    monkeypatch.setenv("PERF_MCP_DISABLE_MEM_CAP", "1")
    monkeypatch.setattr(OPT.shutil, "which", lambda _n: "/usr/bin/systemd-run")
    assert OPT._memory_scope_argv(["python", "x"]) is None


def test_scope_is_skipped_when_systemd_run_is_absent(monkeypatch):
    monkeypatch.delenv("PERF_MCP_MEM_SCOPE", raising=False)
    monkeypatch.delenv("PERF_MCP_DISABLE_MEM_CAP", raising=False)
    monkeypatch.setattr(OPT.shutil, "which", lambda _n: None)
    assert OPT._memory_scope_argv(["python", "x"]) is None


def test_cap_fraction_is_a_ratio_of_memtotal_not_a_fixed_number(monkeypatch):
    monkeypatch.delenv("PERF_MCP_MEM_SCOPE", raising=False)
    monkeypatch.delenv("PERF_MCP_DISABLE_MEM_CAP", raising=False)
    monkeypatch.setattr(OPT.shutil, "which", lambda _n: "/usr/bin/systemd-run")
    monkeypatch.setenv("PERF_MCP_MEM_CAP_FRACTION", "0.5")
    lo = int([a for a in OPT._memory_scope_argv(["p"]) if a.startswith("MemoryMax=")][0].split("=")[1])
    monkeypatch.setenv("PERF_MCP_MEM_CAP_FRACTION", "0.95")
    hi = int([a for a in OPT._memory_scope_argv(["p"]) if a.startswith("MemoryMax=")][0].split("=")[1])
    assert hi > lo, "a larger fraction must allow a larger cap; the cap scales with the box, not a constant"
