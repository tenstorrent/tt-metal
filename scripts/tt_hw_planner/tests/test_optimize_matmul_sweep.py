# SPDX-License-Identifier: Apache-2.0
"""tt_hw_planner `optimize --matmul-sweep` wiring: the sweep pre-pass runs on BOTH the full-pipeline
path and the per-module (--module-level) path, using the right enumeration node in each case. These
checks are device-free (they exercise the node-resolution + skip guards, not the on-device sweep)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_hw_planner.commands import module_optimize, optimize


class _Args:
    def __init__(self, **kw):
        self.perf_test = None
        self.case = None
        self.matmul_sweep = True
        self.matmul_sweep_pcc = 0.99
        self.matmul_sweep_iters = 5
        self.matmul_sweep_max_shapes = 0
        self.__dict__.update(kw)


def test_prepass_uses_explicit_node_over_perf_test(capsys):
    # per-module path passes the module's PCC node explicitly (no --perf-test); the helper must use it
    # and NOT bail with "no node". A bogus repo_root makes it skip at module-not-found, proving it got
    # past node resolution.
    optimize._run_matmul_sweep_prepass(
        _Args(perf_test=None),
        Path("/nonexistent_xyz"),
        Path("/tmp"),
        node="models/demos/x/tests/pcc/test_mod.py::test_x",
    )
    out = capsys.readouterr().out
    assert "no node to sweep" not in out
    assert "sweep module not found" in out


def test_prepass_skips_when_no_node_anywhere(capsys):
    optimize._run_matmul_sweep_prepass(_Args(perf_test=None), Path("/nonexistent_xyz"), Path("/tmp"))
    assert "no node to sweep" in capsys.readouterr().out


def test_prepass_falls_back_to_perf_test_when_no_explicit_node(capsys):
    # no explicit node -> resolves from --perf-test; a bogus root then skips at module-not-found,
    # which proves node resolution succeeded (it did NOT bail with "no node to sweep").
    optimize._run_matmul_sweep_prepass(
        _Args(perf_test="models/demos/y/tests/test_perf.py::test_p"), Path("/nonexistent_xyz"), Path("/tmp")
    )
    out = capsys.readouterr().out
    assert "no node to sweep" not in out and "sweep module not found" in out


def test_module_level_path_wires_the_sweep():
    # the per-module loop must invoke the sweep with the module's node (node=node), gated on the flag.
    import inspect

    src = inspect.getsource(module_optimize.run_module_level_optimize)
    assert "matmul_sweep" in src and "_run_matmul_sweep_prepass" in src and "node=node" in src


def test_full_pipeline_path_hands_the_flag_to_the_engine():
    """The full-pipeline sweep is no longer run here. It used to be a pre-pass called before
    run_cc, which is BEFORE the engine's discover() generates a perf test -- so its only possible
    node was an operator-supplied --perf-test and `--matmul-sweep` alone did nothing. cmd_optimize
    now hands the flag to the engine, which sweeps straight after discovery."""
    import inspect

    src = inspect.getsource(optimize.cmd_optimize)
    assert "PERF_MCP_MATMUL_SWEEP" in src, "the flag is not handed to the engine"
    assert (
        "_run_matmul_sweep_prepass(args, run_root, run_demo)" not in src
    ), "cmd_optimize still sweeps before run_cc, so it still cannot see the generated perf test"


# --------------------------------------------------------------------------- issue #14
# `--matmul-sweep` ignored `-k/--case` on the full-pipeline path, so the sweep enumerated the whole
# perf test instead of the selected case and came back with 0 matmul shapes.
#
#     node = node or getattr(args, "perf_test", None)     # node is now TRUTHY
#     case = case if node else getattr(args, "case", None)  # ...so args.case is unreachable
#
# The `if node` was meant to ask "did the CALLER supply a node?", but line 1 already overwrote the
# answer. The per-module path (node=..., no case) must keep behaving as before.


def _stub_root(tmp_path):
    """A run_root whose cc_optimize/matmul_sweep.py records the kwargs it was called with, so these
    tests assert the value that actually REACHES the sweep rather than a log string (the log line
    prints after the module-not-found guard, so a bogus root never reaches it)."""
    d = tmp_path / "models" / "experimental" / "perf_automation" / "cc_optimize"
    d.mkdir(parents=True)
    (d / "matmul_sweep.py").write_text(
        "import json, os\n"
        "def run_prepass(node, **kw):\n"
        "    kw['node'] = node\n"
        "    kw.pop('repo_root', None)\n"
        "    with open(os.environ['SWEEP_CALL_LOG'], 'w') as f:\n"
        "        json.dump(kw, f)\n"
        "    return {'shapes': 0, 'seeded': 0, 'improved': 0}\n"
    )
    return tmp_path


def _call(tmp_path, monkeypatch, args, **kw):
    log = tmp_path / "call.json"
    monkeypatch.setenv("SWEEP_CALL_LOG", str(log))
    optimize._run_matmul_sweep_prepass(args, _stub_root(tmp_path), tmp_path, **kw)
    assert log.is_file(), "the sweep was never invoked; node resolution bailed out"
    import json

    return json.loads(log.read_text())


def test_full_pipeline_passes_case_through_to_the_sweep(tmp_path, monkeypatch):
    got = _call(
        tmp_path,
        monkeypatch,
        _Args(perf_test="models/demos/y/tests/test_perf.py::test_p", case="decode_bsz1"),
    )
    assert got["node"] == "models/demos/y/tests/test_perf.py::test_p"
    assert got["case"] == "decode_bsz1", (
        "issue #14: --case was dropped on the full-pipeline path, so the sweep enumerates the whole "
        f"perf test and finds 0 matmul shapes. Got case={got['case']!r}"
    )


def test_per_module_path_keeps_its_own_case_semantics(tmp_path, monkeypatch):
    """The per-module caller passes a node and deliberately NO case (the module's PCC node already
    runs only that module). args.case must NOT leak in and narrow it to nothing."""
    got = _call(
        tmp_path,
        monkeypatch,
        _Args(perf_test="models/demos/y/tests/test_perf.py::test_p", case="decode_bsz1"),
        node="models/demos/x/tests/pcc/test_mod.py::test_x",
    )
    assert got["node"] == "models/demos/x/tests/pcc/test_mod.py::test_x"
    assert got["case"] is None, f"args.case leaked into the per-module sweep: {got['case']!r}"


def test_caller_supplied_case_wins(tmp_path, monkeypatch):
    got = _call(
        tmp_path,
        monkeypatch,
        _Args(perf_test="models/demos/y/tests/test_perf.py::test_p", case="from_args"),
        node="models/demos/x/tests/pcc/test_mod.py::test_x",
        case="from_caller",
    )
    assert got["case"] == "from_caller"
