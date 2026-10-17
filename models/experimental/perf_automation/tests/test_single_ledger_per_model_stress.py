# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HARD STRESS: one model, one run, ONE ledger file.

The ledger decides which reading is the BEFORE, and that decision is `first(...)` -- a question
asked OF A FILE. Split the run across two files and the question is asked twice with different
answers, which is how gemma-3-12b-it reported its OPTIMIZED 40.13 ms as the baseline and called a
240.86 ms measurement it was holding "not measured".

So the property under test is not "the name is right" -- it is that every writer in a run agrees on
ONE path, no matter which of them runs first or what the environment looks like at that moment.

  s1  CONVERGENCE: every (explicit / NAME / ROOT) combination a run can present must resolve to the
      same file, in any order -- this is the actual bug, replayed
  s2  ISOLATION: different models must never share, however similar their names
  s3  STRICT: an unkeyed call is loud when asked to be, silent otherwise, and never for a keyed call
  s4  600 randomised environments: total, deterministic, never a path outside the ledger dir
  s5  the real writers: before_loop and run.py's fullpipe bookend agree with perf_mcp
  s6  END TO END: the two writers that disagreed now produce a single before/after pair
"""

import importlib.util
import random
import string
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_KEYS = ("PERF_MCP_LEDGER", "PERF_MCP_MODEL_NAME", "PERF_MCP_MODEL_ROOT", "PERF_MCP_TASK", "PERF_MCP_STRICT_LEDGER_KEY")


def _led():
    spec = importlib.util.spec_from_file_location("meas_ledger_stress", str(_PA / "cc_optimize" / "measurements.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_M = _led()


def _clean(mp):
    for k in _KEYS:
        mp.delenv(k, raising=False)


GEMMA_ROOT = "/home/ttuser/tt-metal-gemma3/models/demos/multimodal/gemma3"
LLAMA_ROOT = "/home/ttuser/tt-metal-llama/models/demos/llama3_1_8b_p150"


# --------------------------------------------------------------------------- s1
def test_s1_the_gemma3_split_cannot_recur(monkeypatch):
    """Replay the exact sequence: before_loop keys from the directory while PERF_MCP_MODEL_NAME is
    still unset, then run.py sets the name and perf_mcp writes. Both must hit one file."""
    _clean(monkeypatch)
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", GEMMA_ROOT)
    early = _M.ledger_path()  # before_loop, name not yet set
    monkeypatch.setenv("PERF_MCP_MODEL_NAME", "gemma3")
    late = _M.ledger_path()  # perf_mcp, after run.py sets it
    explicit = _M.ledger_path("gemma3", "main")  # a caller passing it directly
    assert early == late == explicit, f"{early.name} / {late.name} / {explicit.name}"
    assert "model_main" not in early.name


@pytest.mark.parametrize("order", [("root", "name"), ("name", "root"), ("explicit", "root"), ("root", "explicit")])
def test_s1_convergence_is_order_independent(monkeypatch, order):
    _clean(monkeypatch)
    seen = set()
    for step in order:
        if step == "root":
            monkeypatch.setenv("PERF_MCP_MODEL_ROOT", GEMMA_ROOT)
            seen.add(_M.ledger_path())
        elif step == "name":
            monkeypatch.setenv("PERF_MCP_MODEL_NAME", "gemma3")
            seen.add(_M.ledger_path())
        else:
            seen.add(_M.ledger_path("gemma3", "main"))
    assert len(seen) == 1, f"{order} produced {len(seen)} ledgers: {[p.name for p in seen]}"


def test_s1_task_participates_in_the_key(monkeypatch):
    """Two pipelines of one model are separate histories -- that split is intended."""
    _clean(monkeypatch)
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", GEMMA_ROOT)
    main = _M.ledger_path()
    monkeypatch.setenv("PERF_MCP_TASK", "vision")
    assert _M.ledger_path() != main


# --------------------------------------------------------------------------- s2
def test_s2_models_never_share(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", GEMMA_ROOT)
    g = _M.ledger_path()
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", LLAMA_ROOT)
    assert _M.ledger_path() != g, "llama's readings could anchor gemma3's baseline"


@pytest.mark.parametrize(
    "a,b",
    [
        ("gemma3", "gemma3_v2"),
        ("gemma-3-4b", "gemma-3-27b"),
        ("m", "m_"),
        ("A" * 200, "A" * 199),
    ],
)
def test_s2_similar_names_stay_distinct(monkeypatch, a, b):
    _clean(monkeypatch)
    assert _M.ledger_path(a, "main") != _M.ledger_path(b, "main"), f"{a!r} collided with {b!r}"


# --------------------------------------------------------------------------- s3
def test_s3_strict_rejects_only_the_unkeyed(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("PERF_MCP_STRICT_LEDGER_KEY", "1")
    with pytest.raises(ValueError):  # allow-pytest.raises: repo-root conftest bypassed
        _M.ledger_path()
    assert _M.ledger_path("gemma3")  # explicit is fine
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", GEMMA_ROOT)
    assert _M.ledger_path()  # ROOT is fine


@pytest.mark.parametrize("val", ["0", "", "true", "yes", "2", None])
def test_s3_strict_is_opt_in_only(monkeypatch, val):
    """Anything but exactly "1" leaves production degrading rather than crashing a long run."""
    _clean(monkeypatch)
    if val is not None:
        monkeypatch.setenv("PERF_MCP_STRICT_LEDGER_KEY", val)
    assert _M.ledger_path()


def test_s3_task_alone_does_not_count_as_keyed(monkeypatch):
    """A task with no model is still shared across every model -- the thing we are preventing."""
    _clean(monkeypatch)
    monkeypatch.setenv("PERF_MCP_STRICT_LEDGER_KEY", "1")
    with pytest.raises(ValueError):  # allow-pytest.raises: repo-root conftest bypassed
        _M.ledger_path(task="main")


# --------------------------------------------------------------------------- s4
def test_s4_600_random_environments(monkeypatch):
    rng = random.Random(20260731)
    _clean(monkeypatch)
    base = _M.ledger_path("x", "y").parent
    for i in range(600):
        _clean(monkeypatch)
        if rng.random() < 0.6:
            monkeypatch.setenv(
                "PERF_MCP_MODEL_ROOT",
                "/a/b/" + "".join(rng.choice(string.printable[:70]) for _ in range(rng.randint(0, 20))),
            )
        if rng.random() < 0.5:
            monkeypatch.setenv(
                "PERF_MCP_MODEL_NAME", "".join(rng.choice(string.printable[:70]) for _ in range(rng.randint(0, 20)))
            )
        if rng.random() < 0.4:
            monkeypatch.setenv(
                "PERF_MCP_TASK", "".join(rng.choice(string.printable[:70]) for _ in range(rng.randint(0, 10)))
            )
        try:
            p = _M.ledger_path()
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"case {i} raised {exc!r}")
        assert p.parent == base, f"case {i}: escaped the ledger dir -> {p}"
        assert p.name.startswith("perf_measurements_") and p.name.endswith(".jsonl")
        assert "/" not in p.name and ".." not in p.name
        assert _M.ledger_path() == p, f"case {i}: not deterministic"


def test_s4_long_keys_stay_within_the_filename_limit(monkeypatch):
    _clean(monkeypatch)
    p = _M.ledger_path("org/" + "n" * 400, "t" * 200)
    assert len(p.name.encode()) < 255


# --------------------------------------------------------------------------- s5
def test_s5_real_writers_all_key_their_calls():
    """Source check on the two that did not. A regression here is invisible at runtime -- it just
    quietly starts a second ledger again."""
    run = (_PA / "cc_optimize" / "run.py").read_text()
    i = run.index("KIND_FULLPIPE, led.PHASE_BEFORE")
    assert "model=" in run[i - 300 : i + 700], "run.py fullpipe bookend is unkeyed again"

    bl = (_PA / "agent" / "before_loop.py").read_text()
    j = bl.index("_record_baseline_anchor(profile, ")  # the CALL; the def is "(profile: dict"
    assert "model=" in bl[j : j + 120], "before_loop's baseline anchor no longer passes the model"


def test_s5_before_loop_does_not_depend_on_the_late_env_var():
    """PERF_MCP_MODEL_NAME is set by run.py AFTER discover() -- reading it at baseline time is what
    produced model_main.jsonl."""
    bl = (_PA / "agent" / "before_loop.py").read_text()
    j = bl.index("_record_baseline_anchor(profile, ")
    call = bl[j : j + 120]
    assert "PERF_MCP_MODEL_NAME" not in call, "the anchor is keyed off the late env var again"


# --------------------------------------------------------------------------- s6
def test_s6_one_before_one_after_end_to_end(monkeypatch, tmp_path):
    """The whole point: two writers, one ledger, a real before/after pair -- not two befores."""
    _clean(monkeypatch)
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    led = _led()

    # writer 1: the genuine baseline (what run.py's bookend records)
    led.record(
        led.KIND_FULLPIPE,
        led.PHASE_BEFORE,
        240.86,
        depth="all",
        mode="trace+1cq",
        source="fullpipe-gate:baseline",
        model="gemma3",
        task="main",
    )
    # writer 2: the committed best, later -- must NOT claim the before slot
    seen = led.first(led.KIND_FULLPIPE, led.PHASE_BEFORE, model="gemma3", task="main")
    phase = led.PHASE_AFTER if seen else led.PHASE_BEFORE
    led.record(
        led.KIND_FULLPIPE,
        phase,
        40.13,
        depth="all",
        mode="trace+1cq",
        source="fullpipe-gate:committed-best",
        model="gemma3",
        task="main",
    )

    b = led.first(led.KIND_FULLPIPE, led.PHASE_BEFORE, model="gemma3", task="main")
    a = led.last(led.KIND_FULLPIPE, led.PHASE_AFTER, model="gemma3", task="main")
    assert b["value_ms"] == 240.86, "the baseline is no longer the before"
    assert a and a["value_ms"] == 40.13, "the optimized result did not land as the after"
