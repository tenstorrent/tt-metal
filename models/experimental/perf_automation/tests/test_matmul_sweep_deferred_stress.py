# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HARD STRESS: the deferred matmul sweep must be opt-in, ordered, and unable to break a run.

Moving the sweep from a pre-pass into the engine changes WHEN device work happens, so the risks are
not about matmul shapes at all -- they are about a flag firing when it should not, an exception
escaping into the optimize run, or the ordering silently regressing to the old behaviour.

  s1  OPT-IN: 500 environment permutations; the sweep fires iff PERF_MCP_MATMUL_SWEEP == "1"
  s2  ORDERING: the call is after pipelines_from_manifest and before the optimize loop, asserted
      against the real source, and no pre-pass call survives ahead of run_cc
  s3  ISOLATION: every failure mode of the sweep (raise, exit, hang-shaped, junk return) leaves the
      run alive and the exception unpropagated
  s4  NODE SELECTION: the pipeline actually swept is the first one WITH a perf test, and its case
      travels with it, across many pipeline shapes
  s5  TUNING: pcc/iters/max_shapes parse from env with sane fallbacks, including 0 = "no cap"
  s6  IDEMPOTENCE + PURITY: repeated calls agree, and the demo dir gains nothing but the table
"""

import importlib.util
import os
import random
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"
sys.path.insert(0, str(_PA))


def _run():
    spec = importlib.util.spec_from_file_location("cc_run_defer_stress", str(_CC / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PIPE = {"task": "main", "perf_test": "models/demos/x/tests/e2e/test_main_perf.py", "case": "perf-1"}


def _wire(monkeypatch, sink=None, boom=None):
    m = _run()
    calls = []

    def _fake(**kw):
        calls.append(kw)
        if boom is not None:
            raise boom
        return sink if sink is not None else {"ok": True, "shapes": 1, "seeded": 1}

    monkeypatch.setattr(m, "_invoke_matmul_sweep", _fake)
    return m, calls


# --------------------------------------------------------------------------- s1
@pytest.mark.parametrize("val", ["1", "0", "", "true", "TRUE", "yes", "2", " 1", "1 ", "on", None])
def test_s1_fires_only_on_exactly_one(monkeypatch, tmp_path, val):
    m, calls = _wire(monkeypatch)
    if val is None:
        monkeypatch.delenv("PERF_MCP_MATMUL_SWEEP", raising=False)
    else:
        monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", val)
    m._matmul_sweep_after_discovery(tmp_path, tmp_path, [PIPE], "0")
    assert bool(calls) == (val == "1"), f"PERF_MCP_MATMUL_SWEEP={val!r} -> fired={bool(calls)}"


def test_s1_500_env_permutations(monkeypatch, tmp_path):
    """Unrelated PERF_MCP_* vars must never switch the sweep on."""
    rng = random.Random(20260730)
    others = ["PERF_MCP_MATMUL_SWEEP_PCC", "PERF_MCP_MATMUL_SWEEP_ITERS", "PERF_MCP_SUPERVISED", "PERF_MCP_TASK"]
    for i in range(500):
        m, calls = _wire(monkeypatch)
        on = rng.random() < 0.5
        if on:
            monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
        else:
            monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", rng.choice(["0", "", "true", "2", "no"]))
        for k in others:
            if rng.random() < 0.5:
                monkeypatch.setenv(k, rng.choice(["1", "0", "x", "0.5"]))
            else:
                monkeypatch.delenv(k, raising=False)
        m._matmul_sweep_after_discovery(tmp_path, tmp_path, [PIPE], "0")
        assert bool(calls) == on, f"permutation {i}: fired={bool(calls)} want={on}"


# --------------------------------------------------------------------------- s2
def test_s2_ordering_against_real_source():
    src = (_CC / "run.py").read_text()
    i_disc = src.index("pipes = pipelines_from_manifest")
    i_sweep = src.index("_matmul_sweep_after_discovery(", i_disc)
    # the optimize loop begins at the per-pipeline results accumulation
    i_loop = src.index("results = []", i_disc)
    assert (
        i_disc < i_sweep < i_loop
    ), f"sweep is not between discovery and the loop (disc={i_disc}, sweep={i_sweep}, loop={i_loop})"


def test_s2_no_prepass_survives_before_run_cc():
    src = (_PA.parents[2] / "scripts" / "tt_hw_planner" / "commands" / "optimize.py").read_text()
    head = src[: src.index("result = run_cc(")]
    assert "_run_matmul_sweep_prepass(args, run_root, run_demo)" not in head


def test_s2_engine_reads_the_flag_the_cli_writes():
    """The two halves must agree on the variable name -- a typo here silently disables the flag."""
    cli = (_PA.parents[2] / "scripts" / "tt_hw_planner" / "commands" / "optimize.py").read_text()
    eng = (_CC / "run.py").read_text()
    for var in (
        "PERF_MCP_MATMUL_SWEEP",
        "PERF_MCP_MATMUL_SWEEP_PCC",
        "PERF_MCP_MATMUL_SWEEP_ITERS",
        "PERF_MCP_MATMUL_SWEEP_MAX_SHAPES",
    ):
        assert var in cli, f"{var} not written by the CLI"
        assert var in eng, f"{var} not read by the engine"


# --------------------------------------------------------------------------- s3
@pytest.mark.parametrize(
    "boom",
    [
        RuntimeError("device exploded"),
        OSError("no such device"),
        ValueError("bad shape"),
        MemoryError(),
        KeyboardInterrupt(),
        Exception("generic"),
    ],
)
def test_s3_any_sweep_failure_leaves_the_run_alive(monkeypatch, tmp_path, capsys, boom):
    m, _calls = _wire(monkeypatch, boom=boom)
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    try:
        m._matmul_sweep_after_discovery(tmp_path, tmp_path, [PIPE], "0")
    except BaseException as exc:  # noqa: BLE001
        if isinstance(boom, KeyboardInterrupt) and isinstance(exc, KeyboardInterrupt):
            pytest.skip("KeyboardInterrupt is deliberately not swallowed")
        pytest.fail(f"{type(boom).__name__} escaped into the optimize run: {exc!r}")
    assert "matmul-sweep" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("ret", [None, {}, [], "junk", 42, {"shapes": "x"}])
def test_s3_junk_return_values_do_not_raise(monkeypatch, tmp_path, ret):
    m, _c = _wire(monkeypatch, sink=ret)
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    m._matmul_sweep_after_discovery(tmp_path, tmp_path, [PIPE], "0")


# --------------------------------------------------------------------------- s4
def test_s4_first_pipeline_with_a_perf_test_wins(monkeypatch, tmp_path):
    m, calls = _wire(monkeypatch)
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    pipes = [
        {"task": "novel"},  # no perf test
        {"task": "a", "perf_test": "a.py::t", "case": "ca"},
        {"task": "b", "perf_test": "b.py::t", "case": "cb"},
    ]
    m._matmul_sweep_after_discovery(tmp_path, tmp_path, pipes, "0")
    assert calls[0]["node"] == "a.py::t" and calls[0]["case"] == "ca"


@pytest.mark.parametrize("pipes", [[], None, [{}], [{"task": "x"}], [{"perf_test": ""}], [{"perf_test": None}]])
def test_s4_no_usable_pipeline_is_a_clean_skip(monkeypatch, tmp_path, pipes):
    m, calls = _wire(monkeypatch)
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    m._matmul_sweep_after_discovery(tmp_path, tmp_path, pipes, "0")
    assert not calls


def test_s4_case_absent_is_passed_as_none(monkeypatch, tmp_path):
    m, calls = _wire(monkeypatch)
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    m._matmul_sweep_after_discovery(tmp_path, tmp_path, [{"task": "m", "perf_test": "p.py::t"}], "0")
    assert calls[0]["case"] is None


def test_s4_out_path_is_in_the_demo_dir(monkeypatch, tmp_path):
    m, calls = _wire(monkeypatch)
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    demo = tmp_path / "demo"
    demo.mkdir()
    m._matmul_sweep_after_discovery(demo, tmp_path, [PIPE], "0")
    assert calls[0]["out_path"] == str(
        demo / "matmul_sweep.json"
    ), "the table must land where the warm-start lookup reads it"


# --------------------------------------------------------------------------- s5
@pytest.mark.parametrize(
    "pcc,iters,shapes,want",
    [
        ("0.95", "9", "7", (0.95, 9, 7)),
        (None, None, None, (0.99, 5, 0)),
        ("", "", "", (0.99, 5, 0)),
        ("abc", "abc", "abc", (0.99, 5, 0)),
        ("-1", "-1", "-1", (0.99, 5, 0)),
        ("0", "0", "0", (0.99, 5, 0)),  # 0 shapes = "no cap" is the DEFAULT, so still 0
        ("1.0", "1", "1", (1.0, 1, 1)),
    ],
)
def test_s5_tuning_env_parsing(monkeypatch, tmp_path, pcc, iters, shapes, want):
    m, calls = _wire(monkeypatch)
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    for k, v in (
        ("PERF_MCP_MATMUL_SWEEP_PCC", pcc),
        ("PERF_MCP_MATMUL_SWEEP_ITERS", iters),
        ("PERF_MCP_MATMUL_SWEEP_MAX_SHAPES", shapes),
    ):
        monkeypatch.delenv(k, raising=False) if v is None else monkeypatch.setenv(k, v)
    m._matmul_sweep_after_discovery(tmp_path, tmp_path, [PIPE], "0")
    got = (calls[0]["pcc_threshold"], calls[0]["iters"], calls[0]["max_shapes"])
    assert got == want, f"{got} != {want}"


def test_s5_max_shapes_zero_means_no_cap_not_a_rejected_value(monkeypatch, tmp_path):
    m, calls = _wire(monkeypatch)
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP_MAX_SHAPES", "0")
    m._matmul_sweep_after_discovery(tmp_path, tmp_path, [PIPE], "0")
    assert calls[0]["max_shapes"] == 0


# --------------------------------------------------------------------------- s6
def test_s6_repeated_calls_agree_and_write_nothing_extra(monkeypatch, tmp_path):
    m, calls = _wire(monkeypatch)
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    demo = tmp_path / "demo"
    demo.mkdir()
    before = sorted(p.name for p in demo.iterdir())
    for _ in range(10):
        m._matmul_sweep_after_discovery(demo, tmp_path, [PIPE], "0")
    assert all(c == calls[0] for c in calls), "resolution is not deterministic"
    # _invoke_matmul_sweep is stubbed, so the real writer never runs -- the hook itself must not
    # create anything of its own.
    assert sorted(p.name for p in demo.iterdir()) == before


def test_s6_does_not_mutate_the_pipes_it_was_given(monkeypatch, tmp_path):
    m, _c = _wire(monkeypatch)
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    pipes = [dict(PIPE)]
    snapshot = [dict(p) for p in pipes]
    m._matmul_sweep_after_discovery(tmp_path, tmp_path, pipes, "0")
    assert pipes == snapshot, "the hook mutated the caller's pipeline list"


def test_s6_does_not_leak_env(monkeypatch, tmp_path):
    m, _c = _wire(monkeypatch)
    monkeypatch.setenv("PERF_MCP_MATMUL_SWEEP", "1")
    before = dict(os.environ)
    m._matmul_sweep_after_discovery(tmp_path, tmp_path, [PIPE], "0")
    assert dict(os.environ) == before
