# SPDX-License-Identifier: Apache-2.0
"""STRESS for issue #14: node/case resolution in the --matmul-sweep pre-pass.

The defect was a one-line ordering mistake -- `case = case if node else args.case` read `node`
AFTER the `node = node or args.perf_test` fallback had already overwritten it, so args.case was
unreachable on the full-pipeline path and the sweep enumerated the entire perf test (0 shapes).

The whole input surface is four values, so it is enumerable rather than sampled:

    s1  full 2x2x2x2 cross-product of (caller node, caller case, args.perf_test, args.case)
        checked against an independent oracle
    s2  the two callers that actually exist in the tree, exercised end to end
    s3  falsy-but-present values ("", 0) must not be confused with "absent"
    s4  hostile arg objects (missing attributes entirely) must not raise
    s5  determinism / purity: repeated calls agree and args is never mutated
"""

import itertools
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_hw_planner.commands import optimize


class _Args:
    def __init__(self, **kw):
        self.perf_test = None
        self.case = None
        self.matmul_sweep = True
        self.matmul_sweep_pcc = 0.99
        self.matmul_sweep_iters = 5
        self.matmul_sweep_max_shapes = 0
        self.__dict__.update(kw)


def _stub_root(tmp_path):
    d = tmp_path / "models" / "experimental" / "perf_automation" / "cc_optimize"
    d.mkdir(parents=True, exist_ok=True)
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
    """Return the kwargs that reached run_prepass, or None if the pre-pass skipped."""
    log = tmp_path / "call.json"
    if log.exists():
        log.unlink()
    monkeypatch.setenv("SWEEP_CALL_LOG", str(log))
    optimize._run_matmul_sweep_prepass(args, _stub_root(tmp_path), tmp_path, **kw)
    return json.loads(log.read_text()) if log.is_file() else None


# --------------------------------------------------------------------------- s1
_N_CALLER = "models/demos/x/tests/pcc/test_mod.py::test_x"
_N_ARGS = "models/demos/y/tests/test_perf.py::test_p"


def _oracle(caller_node, caller_case, args_perf_test, args_case):
    """Independent restatement of the intended rule, written from the docstring not the code:

    * the caller's node wins; otherwise fall back to --perf-test. No node anywhere -> skip.
    * an EXPLICIT caller case always wins -- an argument the caller passed must never lose to a
      namespace default (this is the tightening s5 pins down).
    * a caller that supplied a node but no case means "this node needs no -k"; --case must not
      leak in and narrow a per-module sweep to nothing.
    * only when the caller supplied neither half does --case apply.
    """
    node = caller_node or args_perf_test
    if not node:
        return None
    if caller_case is not None:
        case = caller_case
    elif caller_node is not None:
        case = None
    else:
        case = args_case
    return {"node": node, "case": case}


@pytest.mark.parametrize(
    "caller_node,caller_case,args_perf_test,args_case",
    list(itertools.product([None, _N_CALLER], [None, "caller_case"], [None, _N_ARGS], [None, "args_case"])),
)
def test_s1_full_cross_product_matches_oracle(
    tmp_path, monkeypatch, caller_node, caller_case, args_perf_test, args_case
):
    kw = {}
    if caller_node is not None:
        kw["node"] = caller_node
    if caller_case is not None:
        kw["case"] = caller_case
    got = _call(tmp_path, monkeypatch, _Args(perf_test=args_perf_test, case=args_case), **kw)
    want = _oracle(caller_node, caller_case, args_perf_test, args_case)
    if want is None:
        assert got is None, f"expected a skip, sweep ran with {got}"
    else:
        assert got is not None, f"expected sweep to run with {want}, but it skipped"
        assert got["node"] == want["node"]
        assert got["case"] == want["case"], (
            f"case mismatch for caller_node={caller_node!r} caller_case={caller_case!r} "
            f"perf_test={args_perf_test!r} args.case={args_case!r}"
        )


def test_s1b_the_regressing_cell_specifically(tmp_path, monkeypatch):
    """The exact cell issue #14 reported: full-pipeline (no caller node) with -k set."""
    got = _call(tmp_path, monkeypatch, _Args(perf_test=_N_ARGS, case="decode_bsz1"))
    assert got == {
        "node": _N_ARGS,
        "case": "decode_bsz1",
        "out_path": str(tmp_path / "matmul_sweep.json"),
        "pcc_threshold": 0.99,
        "iters": 5,
        "max_shapes": 0,
    }


# --------------------------------------------------------------------------- s2
def test_s2_real_callers_are_shaped_as_this_test_assumes():
    """Guard against this stress test drifting from the call sites that actually exist.

    There is now exactly ONE direct caller: the per-module path. The full-pipeline sweep moved into
    the engine (run.py::_matmul_sweep_after_discovery) so it can use the generated perf test, and
    cmd_optimize only sets the flag. s1's oracle still describes _run_matmul_sweep_prepass itself,
    which the per-module path and the engine both ultimately rely on for node/case resolution.
    """
    import inspect

    from tt_hw_planner.commands import module_optimize

    full = inspect.getsource(optimize.cmd_optimize)
    assert "PERF_MCP_MATMUL_SWEEP" in full, "cmd_optimize no longer hands the sweep flag to the engine"
    per_mod = inspect.getsource(module_optimize.run_module_level_optimize)
    assert (
        "node=node" in per_mod and "case=" not in per_mod.split("_run_matmul_sweep_prepass")[1][:80]
    ), "the per-module caller now passes a case; s1's oracle assumes it does not"


# --------------------------------------------------------------------------- s3
def test_s3_empty_string_case_is_not_confused_with_absent(tmp_path, monkeypatch):
    got = _call(tmp_path, monkeypatch, _Args(perf_test=_N_ARGS, case=""))
    assert got["case"] == "", "an explicitly empty --case must pass through as empty, not vanish"


def test_s3_empty_caller_node_still_counts_as_supplied(tmp_path, monkeypatch):
    """`node=""` is falsy but PRESENT. The old code used truthiness; the fix uses `is not None`, so
    an empty caller node must fall back for the NODE yet still suppress args.case."""
    got = _call(tmp_path, monkeypatch, _Args(perf_test=_N_ARGS, case="args_case"), node="")
    assert got is not None
    assert got["node"] == _N_ARGS, "an empty node must still fall back to --perf-test"
    assert got["case"] is None, "an explicitly-passed node (even empty) means the caller owns the case"


# --------------------------------------------------------------------------- s4
def test_s4_args_missing_attributes_entirely(tmp_path, monkeypatch):
    class _Bare:
        pass

    got = _call(tmp_path, monkeypatch, _Bare())
    assert got is None, "no node anywhere -> skip, not crash"


def test_s4_args_missing_case_attribute(tmp_path, monkeypatch):
    class _NoCase:
        perf_test = _N_ARGS

    got = _call(tmp_path, monkeypatch, _NoCase())
    assert got is not None and got["case"] is None


# --------------------------------------------------------------------------- s5
def test_s5_repeatable_and_does_not_mutate_args(tmp_path, monkeypatch):
    args = _Args(perf_test=_N_ARGS, case="decode_bsz1")
    before = dict(args.__dict__)
    first = _call(tmp_path, monkeypatch, args)
    second = _call(tmp_path, monkeypatch, args)
    assert first == second, "resolution must be deterministic"
    assert args.__dict__ == before, "the pre-pass mutated the caller's args"


def test_s5_caller_case_without_caller_node_is_honoured(tmp_path, monkeypatch):
    """case= passed alone (no node=) is an odd but legal call. It must not be silently dropped in
    favour of args.case -- an explicit argument always beats a namespace default."""
    got = _call(tmp_path, monkeypatch, _Args(perf_test=_N_ARGS, case="args_case"), case="explicit")
    assert got["node"] == _N_ARGS
    assert got["case"] == "explicit"
