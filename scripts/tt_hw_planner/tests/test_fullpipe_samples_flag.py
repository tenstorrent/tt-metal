# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""--fullpipe-samples: an operator-facing knob on how many readings the full-pipeline gate takes
per verdict (reported as their median). The BEFORE bookend and every check_full_pipeline_latency
call both read PERF_MCP_FULLPIPE_SAMPLES (default 1, i.e. no median) -- favouring speed, since a
cold kernel cache alone can cost minutes per reading (see perf_mcp.py's board_over_abort_limit
docstring: "with a cold kernel cache it runs for twelve minutes instead of two"), and that cost is
paid once per sample. Raising it (e.g. 3) filters random per-sample noise from being mistaken for a
real win/regression, at that many times the wall-clock per call -- this lets an operator make that
tradeoff explicitly, per model, from the CLI, rather than only via the PERF_MCP_FULLPIPE_SAMPLES env
var directly.

Mirrors the existing --matmul-sweep-iters -> PERF_MCP_MATMUL_SWEEP_ITERS wiring (cli.py add_argument
-> commands/optimize.py cmd_optimize -> os.environ), so BEFORE and AFTER stay comparable at
whatever count the operator picks: run.py's own note records that a MISMATCHED sample count between
the two "manufactured the full noise range as a gain on every run."
"""

import os
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _parse(monkeypatch, argv):
    from tt_hw_planner import cli

    seen = {}

    def _capture(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(cli, "cmd_optimize", _capture, raising=True)
    try:
        seen["rc"] = cli.main(argv)
    except SystemExit as exc:  # argparse error path
        seen["exit"] = exc.code
    return seen


def test_the_flag_defaults_to_1_matching_the_engines_own_default(monkeypatch):
    ns = _parse(monkeypatch, ["optimize", "m"])["args"]
    assert ns.fullpipe_samples == 1


def test_the_flag_parses_a_higher_value(monkeypatch):
    ns = _parse(monkeypatch, ["optimize", "m", "--fullpipe-samples", "3"])["args"]
    assert ns.fullpipe_samples == 3


def _run_cmd_optimize_up_to_the_disk_gate(monkeypatch, optimize_mod, args):
    """Let cmd_optimize's own body run for real past the env-var line, then stop it at the disk
    gate -- BEFORE _sweep_stale_perf_mcp/_prune_runs would touch any real process or the box's
    disk, and well before anything that opens a device. PERF_MCP_SUPERVISED=1 additionally skips
    the stale-process sweep outright, since a real optimize run may be live on this same box."""
    monkeypatch.setenv("PERF_MCP_SUPERVISED", "1")
    monkeypatch.setattr(optimize_mod, "invalid_trace_flag_error", lambda: None)
    monkeypatch.setattr(optimize_mod, "_disk_gate", lambda: (False, 0, 0))
    monkeypatch.setattr(optimize_mod, "_out_of_disk_msg", lambda low: "stop here -- test boundary")
    return optimize_mod.cmd_optimize(args)


def test_cmd_optimize_sets_the_env_var_the_engine_actually_reads(monkeypatch):
    from tt_hw_planner.commands import optimize as optimize_mod

    monkeypatch.delenv("PERF_MCP_FULLPIPE_SAMPLES", raising=False)

    class _Args:
        devices = ""
        fullpipe_samples = 1

    rc = _run_cmd_optimize_up_to_the_disk_gate(monkeypatch, optimize_mod, _Args())
    assert rc == 1  # confirms we actually reached and returned from the guarded body
    assert os.environ.get("PERF_MCP_FULLPIPE_SAMPLES") == "1"


def test_cmd_optimize_passes_through_the_default_too(monkeypatch):
    from tt_hw_planner.commands import optimize as optimize_mod

    monkeypatch.delenv("PERF_MCP_FULLPIPE_SAMPLES", raising=False)

    class _Args:
        devices = ""
        fullpipe_samples = 3

    _run_cmd_optimize_up_to_the_disk_gate(monkeypatch, optimize_mod, _Args())
    assert os.environ.get("PERF_MCP_FULLPIPE_SAMPLES") == "3"


def test_a_caller_with_no_such_attribute_leaves_the_engines_own_default_untouched(monkeypatch):
    # Anything constructing args without this field (an older test double, e.g.) must not crash
    # cmd_optimize, and must not clobber whatever PERF_MCP_FULLPIPE_SAMPLES was already set to.
    from tt_hw_planner.commands import optimize as optimize_mod

    monkeypatch.setenv("PERF_MCP_FULLPIPE_SAMPLES", "3")

    class _Args:
        devices = ""

    _run_cmd_optimize_up_to_the_disk_gate(monkeypatch, optimize_mod, _Args())
    assert os.environ.get("PERF_MCP_FULLPIPE_SAMPLES") == "3"


def test_perf_mcp_actually_reads_the_env_var_this_flag_sets(monkeypatch):
    import importlib.util as _u

    _REPO = _SCRIPTS.parent
    spec = _u.spec_from_file_location(
        "pmcp_fullpipe_flag",
        str(_REPO / "models" / "experimental" / "perf_automation" / "cc_optimize" / "perf_mcp.py"),
    )
    m = _u.module_from_spec(spec)
    monkeypatch.setenv("PERF_MCP_FULLPIPE_SAMPLES", "1")
    spec.loader.exec_module(m)
    assert m._FULLPIPE_SAMPLES == 1
