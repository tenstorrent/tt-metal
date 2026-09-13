# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""--matmul-sweep must run AFTER discovery, so it uses the perf test the tool just generated.

The sweep was a literal pre-pass: called at optimize.py:553, before run_cc, and therefore before
the engine's discover() generates a perf test. Its only possible node was an operator-supplied
--perf-test, so `--matmul-sweep` alone printed "no node to sweep" and silently did nothing, and an
operator who wanted the sweep had to hand over a perf test the tool was about to generate anyway.

Nothing required that ordering. The sweep's output is a warm-start table consumed much later, when
next_target is a matmul on the knob:fidelity/knob:dtype rung -- long after discovery. Running it
just after discover() means:

  * no --perf-test requirement: the generated node exists by then;
  * the swept node is the SAME node optimize goes on to measure, rather than a possibly-different
    hand-passed one;
  * one test, produced once, used by both.

The per-module path is untouched: it passes its own module PCC node explicitly and never depended
on this ordering.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"
sys.path.insert(0, str(_PA))


def _run():
    spec = importlib.util.spec_from_file_location("cc_run_defer", str(_CC / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PIPES = [{"task": "main", "perf_test": "models/demos/x/tests/e2e/test_main_perf.py", "case": "perf-1"}]


def _hook():
    m = _run()
    fn = getattr(m, "_matmul_sweep_after_discovery", None)
    if fn is None:
        pytest.fail(
            "run.py has no _matmul_sweep_after_discovery: the sweep still runs as a pre-pass before "
            "discover() generates a perf test, so --matmul-sweep alone does nothing and the operator "
            "must supply a perf test the tool is about to generate itself."
        )
    return m, fn


def test_hook_exists():
    assert _hook()[1] is not None


def test_flag_off_runs_nothing(monkeypatch, tmp_path):
    m, fn = _hook()
    monkeypatch.delenv("PERF_MCP_MATMUL_SWEEP", raising=False)
    called = []
    monkeypatch.setattr(m, "_invoke_matmul_sweep", lambda *a, **k: called.append(a))
    fn(tmp_path, tmp_path, PIPES, "0")
    assert not called, "the sweep ran without the flag"


def test_flag_on_uses_the_generated_node(monkeypatch, tmp_path):
    m, fn = _hook()
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    seen = {}
    monkeypatch.setattr(m, "_invoke_matmul_sweep", lambda **kw: seen.update(kw))
    fn(tmp_path, tmp_path, PIPES, "0")
    assert seen.get("node", "").endswith("test_main_perf.py"), f"wrong node: {seen}"
    assert seen.get("case") == "perf-1", f"the generated pipeline's case was dropped: {seen}"


def test_no_pipes_is_a_clean_skip(monkeypatch, tmp_path, capsys):
    m, fn = _hook()
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    called = []
    monkeypatch.setattr(m, "_invoke_matmul_sweep", lambda *a, **k: called.append(a))
    fn(tmp_path, tmp_path, [], "0")
    assert not called


def test_pipe_without_perf_test_is_skipped(monkeypatch, tmp_path):
    m, fn = _hook()
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    called = []
    monkeypatch.setattr(m, "_invoke_matmul_sweep", lambda *a, **k: called.append(a))
    fn(tmp_path, tmp_path, [{"task": "main"}], "0")
    assert not called


def test_sweep_failure_never_fails_the_run(monkeypatch, tmp_path, capsys):
    """The sweep is an optimisation, not a prerequisite -- a crash must not abort optimize."""
    m, fn = _hook()
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")

    def _boom(**_kw):
        raise RuntimeError("device exploded")

    monkeypatch.setattr(m, "_invoke_matmul_sweep", _boom)
    fn(tmp_path, tmp_path, PIPES, "0")  # must not raise
    assert "matmul-sweep" in capsys.readouterr().out.lower()


def test_tuning_params_are_read_from_env(monkeypatch, tmp_path):
    m, fn = _hook()
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP_PCC", "0.95")
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP_ITERS", "9")
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP_MAX_SHAPES", "7")
    seen = {}
    monkeypatch.setattr(m, "_invoke_matmul_sweep", lambda **kw: seen.update(kw))
    fn(tmp_path, tmp_path, PIPES, "0")
    assert seen["pcc_threshold"] == 0.95 and seen["iters"] == 9 and seen["max_shapes"] == 7


def test_defaults_when_env_absent(monkeypatch, tmp_path):
    m, fn = _hook()
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    for k in ("PERF_MCP_MATMUL_SWEEP_PCC", "PERF_MCP_MATMUL_SWEEP_ITERS", "PERF_MCP_MATMUL_SWEEP_MAX_SHAPES"):
        monkeypatch.delenv(k, raising=False)
    seen = {}
    monkeypatch.setattr(m, "_invoke_matmul_sweep", lambda **kw: seen.update(kw))
    fn(tmp_path, tmp_path, PIPES, "0")
    assert seen["pcc_threshold"] == 0.99 and seen["iters"] == 5 and seen["max_shapes"] == 0


@pytest.mark.parametrize("junk", ["", "abc", "-1", "1e9"])
def test_malformed_tuning_env_falls_back_to_defaults(monkeypatch, tmp_path, junk):
    m, fn = _hook()
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP_ITERS", junk)
    seen = {}
    monkeypatch.setattr(m, "_invoke_matmul_sweep", lambda **kw: seen.update(kw))
    fn(tmp_path, tmp_path, PIPES, "0")
    assert isinstance(seen["iters"], int) and seen["iters"] > 0


def test_wired_into_run_cc_optimize_after_discovery():
    """Ordering is the whole point: the call must sit AFTER pipelines_from_manifest."""
    src = (_CC / "run.py").read_text()
    i_disc = src.index("pipes = pipelines_from_manifest")
    i_sweep = src.index("_matmul_sweep_after_discovery(", i_disc)
    assert i_sweep > i_disc, "the sweep still runs before the perf test exists"


def test_optimize_no_longer_runs_the_prepass_before_the_engine():
    src = (_PA.parents[2] / "scripts" / "tt_hw_planner" / "commands" / "optimize.py").read_text()
    i_run = src.index("result = run_cc(")
    head = src[:i_run]
    assert "_run_matmul_sweep_prepass(args, run_root, run_demo)" not in head, (
        "cmd_optimize still calls the pre-pass before run_cc, so it still cannot see the generated " "perf test"
    )
    assert "PERF_MCP_MATMUL_SWEEP" in head, "the flag is not handed to the engine"
