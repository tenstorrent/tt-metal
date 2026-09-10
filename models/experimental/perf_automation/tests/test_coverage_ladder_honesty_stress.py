# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""STRESS for issue 7: the coverage search must never launder a failure into a measurement.

The contract, in three mutually exclusive outcomes:

    covered            -> (depth, [],       "measured")            depth actually covers everything
    exhausted          -> (depth, missing,  "measured-incomplete") no rung covered it; say so
    inert knob         -> None                                     missing ops at FULL depth is
                                                                   impossible; the cap does nothing

  s1  randomised search space: 600 (ladder, coverage-profile) combinations against an oracle
  s2  the label can never be "measured" while anything is missing -- the core honesty property
  s3  monotone coverage (a deeper rung never reveals fewer ops) is respected and exploited
  s4  the inert-knob verdict fires only when a depth is actually declared
  s5  the search stops at the first covering rung (no wasted device probes)
  s6  degenerate inputs: empty want-set, empty ladder, single rung, probe failures
"""

import importlib.util
import json
import random
import sys
from pathlib import Path


_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"


def _mod():
    sys.path.insert(0, str(_PA))
    spec = importlib.util.spec_from_file_location("cc_run_cov_stress", str(_CC / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


OPS = ["matmul", "softmax", "rmsnorm", "rare_op", "deep_op"]


def _run(monkeypatch, tmp_path, sigs_by_depth, want, declared=None, ladder=None):
    m = _mod()
    probed = []
    if declared is not None:
        (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": declared}))
    if ladder is not None:
        monkeypatch.setenv("PERF_MCP_COV_LADDER", ladder)

    def _probe(_r, _e, _d, _n, _c, d):
        probed.append(d)
        return sorted(sigs_by_depth.get(d, set())), "", []

    monkeypatch.setattr(m, "_run_op_sigs", _probe)
    out = m._measure_cov(Path("/repo"), {}, "0", "n.py::t", None, set(want), tmp_path, base_knob={"CAP": "1"})
    return out, probed, m


def _oracle(rungs, sigs_by_depth, want, declared):
    """Independent restatement of the three outcomes."""
    want = set(want)
    got = set()
    for d in rungs:
        s = set(sigs_by_depth.get(d, set()))
        if not s:
            return "inconclusive"
        got = s
        if want <= got:
            return ("measured", d, [])
    missing = sorted(want - got)
    last = rungs[-1] if rungs else 16
    if missing and declared is not None and last >= declared:
        return "inert"
    return ("measured-incomplete", last, missing)


# --------------------------------------------------------------------------- s1
def test_s1_randomised_search_matches_the_oracle(tmp_path, monkeypatch):
    rng = random.Random(20260730)
    for i in range(600):
        declared = rng.choice([None, 4, 8, 16, 32, 64])
        rungs_env = rng.choice([None, "2,4,8,16", "1,2,4", "2,4,8,16,32,64"])
        want = set(rng.sample(OPS, rng.randint(1, len(OPS))))
        # monotone coverage: deeper rungs reveal a superset
        sigs, acc = {}, set()
        for d in sorted({1, 2, 4, 8, 16, 32, 64} | ({declared} if declared else set())):
            if rng.random() < 0.15:
                sigs[d] = set()  # probe failure
            else:
                acc = acc | set(rng.sample(OPS, rng.randint(0, len(OPS))))
                sigs[d] = set(acc)
        d = tmp_path / f"c{i}"
        d.mkdir()
        out, _probed, m = _run(monkeypatch, d, sigs, want, declared, rungs_env)
        rungs = m._cov_ladder(d)
        exp = _oracle(rungs, sigs, want, declared)
        if exp == "inconclusive" or exp == "inert":
            assert out is None, f"case {i}: expected {exp}, got {out}"
        else:
            assert out is not None, f"case {i}: expected {exp}, got None"
            assert out[2] == exp[0] and out[0] == exp[1] and out[1] == exp[2], f"case {i}: {out} != {exp}"


# --------------------------------------------------------------------------- s2
def test_s2_measured_never_coexists_with_missing_ops(tmp_path, monkeypatch):
    """The single property this issue is about."""
    rng = random.Random(7)
    for i in range(300):
        declared = rng.choice([None, 8, 32])
        want = set(rng.sample(OPS, rng.randint(1, len(OPS))))
        sigs = {d: set(rng.sample(OPS, rng.randint(1, len(OPS)))) for d in (1, 2, 4, 8, 16, 32)}
        d = tmp_path / f"p{i}"
        d.mkdir()
        out, _probed, _m = _run(monkeypatch, d, sigs, want, declared)
        if out is None:
            continue
        _depth, missing, source = out
        assert not (source == "measured" and missing), f"case {i}: 'measured' with {missing} missing"
        assert (source == "measured") == (not missing), f"case {i}: label/missing disagree: {out}"


# --------------------------------------------------------------------------- s3
def test_s3_deeper_rung_never_loses_a_hit(tmp_path, monkeypatch):
    sigs = {2: {"matmul"}, 4: {"matmul", "softmax"}, 8: set(OPS), 16: set(OPS), 32: set(OPS)}
    out, probed, _m = _run(monkeypatch, tmp_path, sigs, OPS, declared=32)
    assert out == (8, [], "measured")
    assert probed == [2, 4, 8], f"probed past the covering rung: {probed}"


# --------------------------------------------------------------------------- s4
def test_s4_inert_verdict_requires_a_declared_depth(tmp_path, monkeypatch):
    partial = {"matmul"}
    sigs = {d: partial for d in (1, 2, 4, 8, 16, 32, 64)}
    a = tmp_path / "declared"
    a.mkdir()
    out_declared, _p, _m = _run(monkeypatch, a, sigs, OPS, declared=32)
    assert out_declared is None, "missing ops at full depth must be an inert-knob verdict"
    b = tmp_path / "undeclared"
    b.mkdir()
    out_undeclared, _p, _m = _run(monkeypatch, b, sigs, OPS, declared=None)
    assert (
        out_undeclared is not None and out_undeclared[2] == "measured-incomplete"
    ), "with no declared depth we cannot claim the knob is inert -- the ladder merely ran out"


def test_s4_inert_message_names_the_missing_ops(tmp_path, monkeypatch, capsys):
    sigs = {d: {"matmul"} for d in (1, 2, 4, 8, 16, 32)}
    _run(monkeypatch, tmp_path, sigs, ["matmul", "rare_op"], declared=32)
    out = capsys.readouterr().out
    assert "rare_op" in out and "INERT" in out.upper()


# --------------------------------------------------------------------------- s5
def test_s5_no_probe_after_success(tmp_path, monkeypatch):
    sigs = {d: set(OPS) for d in (1, 2, 4, 8, 16, 32)}
    _out, probed, _m = _run(monkeypatch, tmp_path, sigs, OPS, declared=32)
    assert probed == [2], f"kept probing after the first rung already covered everything: {probed}"


def test_s5_probe_count_bounded_by_ladder(tmp_path, monkeypatch):
    sigs = {d: {"matmul"} for d in (1, 2, 4, 8, 16, 64)}
    _out, probed, m = _run(monkeypatch, tmp_path, sigs, ["matmul", "rare_op"], declared=64)
    assert len(probed) <= len(m._cov_ladder(tmp_path))


# --------------------------------------------------------------------------- s6
def test_s6_empty_want_set_is_inconclusive(tmp_path, monkeypatch):
    out, probed, _m = _run(monkeypatch, tmp_path, {2: set(OPS)}, [], declared=32)
    assert out is None and not probed, "nothing to cover -> no device probes at all"


def test_s6_probe_failure_is_inconclusive_not_incomplete(tmp_path, monkeypatch):
    """An empty probe means the measurement did not happen; that is not the same as 'this depth
    does not cover the model', and must not be reported as a coverage result."""
    out, _p, _m = _run(monkeypatch, tmp_path, {2: set()}, OPS, declared=32)
    assert out is None


def test_s6_single_rung_ladder(tmp_path, monkeypatch):
    # declared=1 -> every default rung is deeper than the model, so the ladder collapses to [1].
    out, probed, _m = _run(monkeypatch, tmp_path, {1: set(OPS)}, OPS, declared=1)
    assert out == (1, [], "measured")
    assert probed == [1]


def test_s6_no_depth_knob_is_skipped(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "_llm_depth_env", lambda *_a, **_k: {})
    out = m._measure_cov(Path("/repo"), {}, "0", "n.py::t", None, set(OPS), tmp_path, base_knob=None)
    assert out is None
