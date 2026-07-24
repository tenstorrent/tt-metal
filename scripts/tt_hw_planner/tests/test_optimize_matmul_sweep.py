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


def test_full_pipeline_path_still_wires_the_sweep():
    import inspect

    src = inspect.getsource(optimize.cmd_optimize)
    assert "_run_matmul_sweep_prepass" in src
