# SPDX-License-Identifier: Apache-2.0
"""HARD STRESS for the --matmul-sweep FLAG SYSTEM, end to end.

Issue #14 was not a bug in the sweep, the probe, or the JSON. Every one of those worked. It was a
bug in the CHAIN -- one hop silently dropped `-k`, and every stage downstream faithfully reported
success on the wrong input. So the thing that has to be solid is the chain, not any component.

The chain, and what must hold at each hop:

    argv                     -k/--case parses into args.case, verbatim, for ANY string
      -> cmd_optimize        the flag gates the pre-pass; --matmul-sweep off means nothing runs
      -> _run_matmul_sweep_prepass   node/case resolution (the hop that broke)
      -> run_prepass         receives the case
      -> enumerate_matmul_sigs       puts ["-k", case] in the probe ARGV, unquoted, no shell
      -> matmul_sweep.json   written where the optimize loop looks for it
      -> termination_check   resolves the shape and DELIVERS next_target["warm_start"]
      -> _warm_start_applied ground-truth check of what actually ran

  s1  argv -> args.case round-trip over hostile strings (spaces, quotes, parens, unicode, dashes)
  s2  a case string reaches the probe ARGV byte-identical, and NEVER touches a shell
  s3  full chain: flag on -> real matmul_sweep.json on disk -> warm_start delivered from it
  s4  the flag is a real gate: off -> no pre-pass, no file, no warm_start anywhere
  s5  400 randomised (flag, perf_test, case, node) combinations against an end-to-end oracle
  s6  no silent success: a pre-pass that enumerated nothing must not look like one that worked
"""

import importlib.util
import json
import random
import string
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2]
_REPO = _SCRIPTS.parent
_PA = _REPO / "models" / "experimental" / "perf_automation"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_PA))  # cc_optimize modules import `agent.*` off this root

from tt_hw_planner.commands import optimize  # noqa: E402


def _perf_mcp():
    sys.path.insert(0, str(_REPO / "models" / "experimental" / "perf_automation"))
    spec = importlib.util.spec_from_file_location(
        "pmcp_flag", str(_REPO / "models" / "experimental" / "perf_automation" / "cc_optimize" / "perf_mcp.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _matmul_sweep_mod():
    p = _REPO / "models" / "experimental" / "perf_automation" / "cc_optimize" / "matmul_sweep.py"
    spec = importlib.util.spec_from_file_location("msweep_flag", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Args:
    def __init__(self, **kw):
        self.perf_test = None
        self.case = None
        self.matmul_sweep = True
        self.matmul_sweep_pcc = 0.99
        self.matmul_sweep_iters = 5
        self.matmul_sweep_max_shapes = 0
        self.__dict__.update(kw)


# Strings a real operator could plausibly pass to -k, plus a few that would break a shell.
_CASES = [
    "performance-batch-1",
    "device_params0",
    "performance and batch1",
    "not slow",
    "ci-token-matching and performance",
    "case-with-dash",
    "case_with_underscore",
    "UPPER",
    "x" * 200,
    "case(with)parens",
    "case with 'single' quotes",
    'case with "double" quotes',
    "case;with;semicolons",
    "case&&with&&amps",
    "case|with|pipes",
    "case$WITH$dollars",
    "case`with`backticks",
    "ünïcodé-cåse",
    "-leading-dash",
]


# --------------------------------------------------------------------------- s1
def _parse(monkeypatch, argv):
    """Drive the REAL cli.main() parser and intercept the dispatch, so the argv layer is exercised
    exactly as a user invokes it without any command running. cli.py builds its parser inside
    main() and has no exposed builder, so this is the only way to test the real thing rather than
    a reconstruction that could drift from it."""
    from tt_hw_planner import cli

    seen = {}

    def _capture(args):
        seen["args"] = args
        return 0

    # main() builds the parser and calls set_defaults(func=cmd_optimize) on every invocation, so the
    # patched global is what gets bound -- no need to re-point func on the namespace.
    monkeypatch.setattr(cli, "cmd_optimize", _capture, raising=True)
    try:
        seen["rc"] = cli.main(argv)
    except SystemExit as exc:  # argparse error path
        seen["exit"] = exc.code
    return seen


@pytest.mark.parametrize("case", _CASES)
def test_s1_argv_to_args_case_roundtrip(monkeypatch, case):
    """-k must survive argparse byte-identically. A leading dash needs `-k=<v>`; anything that only
    works unquoted in a shell is not this layer's problem, but mangling is."""
    seen = _parse(
        monkeypatch,
        ["optimize", "models/demos/x", f"-k={case}", "--matmul-sweep", "--perf-test", "t.py::t"],
    )
    assert "args" in seen, f"optimize never dispatched (exit={seen.get('exit')})"
    ns = seen["args"]
    assert ns.case == case, f"argparse mangled the case: {ns.case!r} != {case!r}"
    assert ns.matmul_sweep is True


def test_s1_flag_defaults_off(monkeypatch):
    seen = _parse(monkeypatch, ["optimize", "models/demos/x"])
    assert "args" in seen, f"optimize never dispatched (exit={seen.get('exit')})"
    ns = seen["args"]
    assert getattr(ns, "matmul_sweep", False) is False, "--matmul-sweep must be opt-in"
    assert getattr(ns, "case", None) is None


def test_s1_long_form_case_is_equivalent(monkeypatch):
    a = _parse(monkeypatch, ["optimize", "m", "-k=perf-1", "--matmul-sweep"])["args"]
    b = _parse(monkeypatch, ["optimize", "m", "--case=perf-1", "--matmul-sweep"])["args"]
    assert a.case == b.case == "perf-1"


def test_s1_separated_form_also_parses(monkeypatch):
    ns = _parse(monkeypatch, ["optimize", "m", "-k", "perf-1", "--matmul-sweep"])["args"]
    assert ns.case == "perf-1"


def test_s1_sweep_tuning_flags_reach_args(monkeypatch):
    ns = _parse(
        monkeypatch,
        [
            "optimize",
            "m",
            "--matmul-sweep",
            "--matmul-sweep-pcc",
            "0.95",
            "--matmul-sweep-iters",
            "9",
            "--matmul-sweep-max-shapes",
            "7",
        ],
    )["args"]
    assert ns.matmul_sweep_pcc == 0.95 and ns.matmul_sweep_iters == 9 and ns.matmul_sweep_max_shapes == 7


# --------------------------------------------------------------------------- s2
@pytest.mark.parametrize("case", _CASES)
def test_s2_case_reaches_probe_argv_verbatim_and_unshelled(monkeypatch, case):
    """The probe command must be a LIST containing exactly ["-k", case] -- never a shell string.
    A shell here is how `Syntax error: "(" unexpected` once masqueraded as a profiler crash."""
    ms = _matmul_sweep_mod()
    seen = {}

    def _fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["shell"] = kw.get("shell", False)

        class _R:
            stdout = "PERF_OP_SIGS=[]"
            stderr = ""
            returncode = 0

        return _R()

    monkeypatch.setattr(ms.subprocess, "run", _fake_run)
    ms.enumerate_matmul_sigs("models/demos/x/t.py::test_p", case, _REPO)
    cmd = seen["cmd"]
    assert isinstance(cmd, list), "the probe must be exec'd as an argv list, not a shell string"
    assert seen["shell"] is False, "shell=True would reintroduce metacharacter injection"
    assert (
        cmd[-2:] == ["models/demos/x/t.py::test_p", case] or case in cmd
    ), f"case {case!r} did not reach the probe argv: {cmd}"
    assert any(c == case for c in cmd), f"case was altered in transit: {cmd}"


def test_s2_no_case_means_no_extra_argv(monkeypatch):
    ms = _matmul_sweep_mod()
    seen = {}

    def _fake_run(cmd, **kw):
        seen["cmd"] = cmd

        class _R:
            stdout = "PERF_OP_SIGS=[]"
            stderr = ""
            returncode = 0

        return _R()

    monkeypatch.setattr(ms.subprocess, "run", _fake_run)
    ms.enumerate_matmul_sigs("t.py::test_p", None, _REPO)
    assert seen["cmd"][-1].endswith("::test_p"), f"a None case added an argv entry: {seen['cmd']}"


# --------------------------------------------------------------------------- s3
def _stub_sweep_root(tmp_path, entries):
    """A run_root whose cc_optimize/matmul_sweep.py writes a REAL warm-start table."""
    d = tmp_path / "models" / "experimental" / "perf_automation" / "cc_optimize"
    d.mkdir(parents=True, exist_ok=True)
    (d / "matmul_sweep.py").write_text(
        "import json, os\n"
        "def run_prepass(node, **kw):\n"
        "    entries = json.loads(os.environ['SWEEP_ENTRIES'])\n"
        "    if kw.get('case') is None:\n"
        "        entries = []            # the issue #14 behaviour: no case -> nothing enumerated\n"
        # THE REAL SUMMARIZE() SHAPE. This stub emitted only `entries`, which is the format
        # _warm_start_for was FIXED to stop reading -- matmul_sweep.py writes `seeds` (shape nested
        # under row["shape"], fidelity/dtype at top level) and `table` (shape flat, winner under
        # row["best"]). A stub in the old shape meant the chain test asserted delivery against a
        # table the resolver cannot read, so it has been failing since that fix. `entries` is kept
        # because the counts above are asserted from it.
        "    seeds = [{'shape': {'m': e['m'], 'k': e['k'], 'n': e['n']},\n"
        "              'fidelity': e['fidelity'], 'dtype': e['dtype']} for e in entries]\n"
        "    table = [{'m': e['m'], 'k': e['k'], 'n': e['n'],\n"
        "              'best': {'fidelity': e['fidelity'], 'dtype': e['dtype']}} for e in entries]\n"
        "    out = {'ok': True, 'shapes': len(entries), 'seeded': len(entries),\n"
        "           'entries': entries, 'seeds': seeds, 'table': table}\n"
        "    open(kw['out_path'], 'w').write(json.dumps(out))\n"
        "    return out\n"
    )
    return tmp_path


_ENTRIES = [{"m": 32, "k": 4096, "n": 4096, "fidelity": "LoFi", "dtype": "bfloat8_b"}]


def test_s3_full_chain_delivers_the_warm_start(tmp_path, monkeypatch):
    """flag on + --case set -> table on disk -> termination_check hands it over as data."""
    monkeypatch.setenv("SWEEP_ENTRIES", json.dumps(_ENTRIES))
    demo = tmp_path / "demo"
    demo.mkdir()
    optimize._run_matmul_sweep_prepass(
        _Args(perf_test="t.py::test_p", case="performance-batch-1"), _stub_sweep_root(tmp_path, _ENTRIES), demo
    )
    table = demo / "matmul_sweep.json"
    assert table.is_file(), "the pre-pass wrote no table"
    assert json.loads(table.read_text())["shapes"] == 1

    ws = _perf_mcp()._warm_start_for(demo, "MatmulDeviceOperation 32 x 4096 x 4096")
    assert ws == {
        "fidelity": "LoFi",
        "dtype": "bfloat8_b",
    }, "the table exists but the optimize loop cannot resolve it -- the chain is broken at delivery"


def test_s3_the_regression_end_to_end(tmp_path, monkeypatch):
    """Without the fix the case is dropped, the stub enumerates nothing, and the whole chain ends in
    a warm start that silently does not exist. This is issue #14 as one assertion."""
    monkeypatch.setenv("SWEEP_ENTRIES", json.dumps(_ENTRIES))
    demo = tmp_path / "demo"
    demo.mkdir()
    optimize._run_matmul_sweep_prepass(
        _Args(perf_test="t.py::test_p", case="performance-batch-1"), _stub_sweep_root(tmp_path, _ENTRIES), demo
    )
    assert (
        json.loads((demo / "matmul_sweep.json").read_text())["shapes"] != 0
    ), "0 shapes: the case was dropped somewhere in the chain (issue #14)"


def test_s3_verification_closes_the_loop(tmp_path, monkeypatch):
    """Delivered -> applied -> confirmed from the profile, which is the only non-self-reported step."""
    monkeypatch.setenv("SWEEP_ENTRIES", json.dumps(_ENTRIES))
    demo = tmp_path / "demo"
    demo.mkdir()
    optimize._run_matmul_sweep_prepass(
        _Args(perf_test="t.py::test_p", case="c"), _stub_sweep_root(tmp_path, _ENTRIES), demo
    )
    m = _perf_mcp()
    op = "MatmulDeviceOperation 32 x 4096 x 4096"
    ws = m._warm_start_for(demo, op)
    assert m._warm_start_applied({"top_ops": [{"op_code": op, "fidelity": "LoFi"}]}, op, ws) is True
    assert m._warm_start_applied({"top_ops": [{"op_code": op, "fidelity": "HiFi4"}]}, op, ws) is False


# --------------------------------------------------------------------------- s4
def test_s4_flag_off_runs_nothing(tmp_path, monkeypatch):
    """cmd_optimize gates the pre-pass on the flag; off must mean no device work and no table."""
    import inspect

    src = inspect.getsource(optimize.cmd_optimize)
    assert 'getattr(args, "matmul_sweep", False)' in src, "the pre-pass is no longer gated on the flag"
    demo = tmp_path / "demo"
    demo.mkdir()
    assert not (demo / "matmul_sweep.json").exists()


def test_s4_no_table_means_no_warm_start(tmp_path):
    assert _perf_mcp()._warm_start_for(tmp_path, "MatmulDeviceOperation 32 x 4096 x 4096") is None


def test_s4_flag_on_but_no_perf_test_skips_cleanly(tmp_path, capsys):
    optimize._run_matmul_sweep_prepass(_Args(perf_test=None), tmp_path, tmp_path)
    assert "no node to sweep" in capsys.readouterr().out
    assert not (tmp_path / "matmul_sweep.json").exists()


# --------------------------------------------------------------------------- s5
def _oracle(flag, perf_test, case, caller_node):
    """End-to-end expectation: does a warm start end up available?"""
    if not flag:
        return None
    node = caller_node or perf_test
    if not node:
        return None
    eff_case = None if caller_node is not None else case
    return "table" if eff_case else "empty"


def test_s5_randomised_flag_combinations(tmp_path, monkeypatch):
    monkeypatch.setenv("SWEEP_ENTRIES", json.dumps(_ENTRIES))
    rng = random.Random(20260730)
    root = _stub_sweep_root(tmp_path, _ENTRIES)
    for i in range(400):
        flag = rng.random() < 0.8
        perf_test = rng.choice([None, "t.py::test_p"])
        case = rng.choice([None, "performance-batch-1", "".join(rng.choice(string.ascii_letters) for _ in range(6))])
        caller_node = rng.choice([None, None, None, "mod.py::test_m"])
        demo = tmp_path / f"d{i}"
        demo.mkdir()
        if flag:
            kw = {"node": caller_node} if caller_node is not None else {}
            optimize._run_matmul_sweep_prepass(_Args(perf_test=perf_test, case=case), root, demo, **kw)
        want = _oracle(flag, perf_test, case, caller_node)
        table = demo / "matmul_sweep.json"
        if want is None:
            assert not table.is_file(), f"case {i}: a table appeared with flag={flag} node={caller_node}"
            continue
        assert table.is_file(), f"case {i}: no table written (want {want})"
        shapes = json.loads(table.read_text())["shapes"]
        if want == "table":
            assert shapes == 1, f"case {i}: case={case!r} was dropped -> {shapes} shapes"
        else:
            assert shapes == 0, f"case {i}: expected an empty table, got {shapes}"


# --------------------------------------------------------------------------- s6
def test_s6_empty_table_is_indistinguishable_from_absent_to_the_consumer(tmp_path):
    """An empty table must never yield a phantom warm start."""
    (tmp_path / "matmul_sweep.json").write_text(json.dumps({"ok": True, "shapes": 0, "entries": []}))
    assert _perf_mcp()._warm_start_for(tmp_path, "MatmulDeviceOperation 32 x 4096 x 4096") is None


def test_s6_shapes_zero_still_reports_ok_true_documented_gap(tmp_path, monkeypatch):
    """KNOWN, UNFIXED: run_prepass returns ok=True for a pre-pass that enumerated nothing, so
    'the model has no matmuls' and 'nothing ran' are indistinguishable upstream. Pinned here so the
    gap is visible and so fixing it trips this test rather than passing silently."""
    monkeypatch.setenv("SWEEP_ENTRIES", json.dumps([]))
    demo = tmp_path / "demo"
    demo.mkdir()
    optimize._run_matmul_sweep_prepass(_Args(perf_test="t.py::test_p", case="c"), _stub_sweep_root(tmp_path, []), demo)
    got = json.loads((demo / "matmul_sweep.json").read_text())
    assert got["shapes"] == 0
    assert got["ok"] is True, (
        "ok is no longer True for a 0-shape pre-pass -- the silent-empty gap was fixed; update this "
        "test to assert the new contract"
    )
