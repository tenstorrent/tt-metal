# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""emit-e2e stamps its run, so the gate it launches shares that identity.

The stamp was copied into the gate's environment only `if` the environment already held one -- and
nothing in emit-e2e's process ever set it, so the condition was always false and every driver-side
gate ran unstamped. Two mechanisms key on it and both were silently disabled:

  * device_recovery's recovery counters. stamp_run's own docstring records the cost of an empty
    stamp: "an emit-e2e run's three failed resets left reset_fails=3 there, and every later
    emit-e2e refused to reset at all."
  * the correctness cache. `_correctness_key` returns None without a stamp, so the driver-side gate
    could neither record a pass nor reuse one -- while the agent's gate, which runs under a process
    that IS stamped, used a different key space entirely. Measured on a live run: emit-e2e pid
    214027 had PERF_MCP_RUN_ID unset while its agent had 1790773851_214027, the cache recorded
    exactly once in eight hours and hit zero times, and the ~70 minute correctness run was paid
    twice per round -- once driver-side, once when the agent asked the same question a minute later.

The fix is that emit-e2e takes an identity before its first device work, which is what stamp_run
exists for and what every other entry point already does.
"""

from __future__ import annotations

import ast
import inspect
import os
import textwrap

from pathlib import Path

from models.experimental.perf_automation.agent import device_recovery as DR
from scripts.tt_hw_planner.commands import emit_e2e as E

_ENV = "PERF_MCP_RUN_ID"


# --- the stamp reaches the gate -------------------------------------------------------------------


def test_the_gate_env_is_stamped_even_when_the_parent_had_none(monkeypatch):
    """THE BUG: with nothing preset, the old code copied nothing and the gate ran unstamped."""
    monkeypatch.delenv(_ENV, raising=False)
    # ast.unparse normalises quoting, so the needles must be quote-agnostic -- asserting on `"` here
    # made the guard check pass vacuously, which is its own small lesson about source-shape tests.
    body = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(E._run_emit_e2e_cc))).body[0]).replace('"', "'")
    assert "stamp_run()" in body, "emit-e2e must take an identity, not wait to be handed one"
    assert "if _os.environ.get('PERF_MCP_RUN_ID')" not in body, "the always-false guard must be gone"
    assert "mcp_env['PERF_MCP_RUN_ID'] = _run_id" in body


def test_stamp_run_gives_an_identity_when_there_is_none(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    got = DR.stamp_run()
    assert got and got == os.environ[_ENV]


def test_it_never_overwrites_an_existing_identity(monkeypatch):
    """A supervisor restart must not silently get a fresh recovery budget."""
    monkeypatch.setenv(_ENV, "operator-value")
    assert DR.stamp_run() == "operator-value"


def test_it_is_idempotent(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert DR.stamp_run() == DR.stamp_run()


# --- what the stamp unlocks -----------------------------------------------------------------------


def test_without_a_stamp_the_correctness_cache_is_entirely_disabled(tmp_path, monkeypatch):
    """This is why it never hit: no stamp -> no key -> it can neither record nor reuse."""
    monkeypatch.setenv(_ENV, "")
    (tmp_path / "scripts").mkdir()
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tt").mkdir(parents=True)
    (demo / "tt" / "pipeline.py").write_text("X = 1\n")
    assert E._correctness_key(demo, 0.99, 32) is None
    assert E._cached_correctness_pass(demo, None) is None
    E._record_correctness_pass(demo, None, ["PCC=1.0"])
    assert not (demo / E._GATE_CACHE_FILE).exists(), "an unstamped run must record nothing"


def test_with_a_stamp_it_records_and_then_reuses(tmp_path, monkeypatch):
    """The whole point: two gates in ONE run, no edit between, second one reuses the first."""
    monkeypatch.setenv(_ENV, "run-A")
    (tmp_path / "scripts").mkdir()
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tt").mkdir(parents=True)
    (demo / "tt" / "pipeline.py").write_text("X = 1\n")

    k1 = E._correctness_key(demo, 0.99, 32)
    assert k1 is not None
    E._record_correctness_pass(demo, k1, ["PERF_BATCH_STREAMS=32", "e2e PCC=0.997"])
    # a second gate in the same run, nothing edited
    assert E._correctness_key(demo, 0.99, 32) == k1
    assert E._cached_correctness_pass(demo, k1) == ["PERF_BATCH_STREAMS=32", "e2e PCC=0.997"]


def test_the_two_gates_of_one_round_share_a_key_space(tmp_path, monkeypatch):
    """The live failure: emit-e2e unstamped, its agent stamped -> different key spaces."""
    (tmp_path / "scripts").mkdir()
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tt").mkdir(parents=True)
    (demo / "tt" / "pipeline.py").write_text("X = 1\n")

    monkeypatch.setenv(_ENV, "")  # the driver gate, as it actually ran
    unstamped = E._correctness_key(demo, 0.99, 32)
    monkeypatch.setenv(_ENV, "run-A")  # the agent's gate, as it actually ran
    stamped = E._correctness_key(demo, 0.99, 32)
    assert unstamped is None and stamped is not None, "this mismatch is what cost 70 min a round"


def test_a_different_run_still_re_verifies(tmp_path, monkeypatch):
    """The stamp must still scope a pass to one run; this is not a way to cache forever."""
    (tmp_path / "scripts").mkdir()
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tt").mkdir(parents=True)
    (demo / "tt" / "pipeline.py").write_text("X = 1\n")
    monkeypatch.setenv(_ENV, "run-A")
    a = E._correctness_key(demo, 0.99, 32)
    monkeypatch.setenv(_ENV, "run-B")
    assert E._correctness_key(demo, 0.99, 32) != a


# --- constraints -----------------------------------------------------------------------------------


def test_it_asks_device_recovery_rather_than_minting_its_own_id():
    """Constraint: one owner for the run identity -- not a second id format invented here."""
    body = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(E._run_emit_e2e_cc))).body[0]).replace('"', "'")
    assert "stamp_run" in body
    assert "time.time()" not in body and "uuid" not in body, "the id format has one owner"


def test_a_missing_device_recovery_does_not_break_the_run():
    body = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(E._run_emit_e2e_cc))).body[0])
    assert "except Exception" in body, "no identity must be survivable"


def test_it_names_no_model_or_stage():
    body = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(E._run_emit_e2e_cc))).body[0]).lower()
    for name in ("qwen", "denoise", "prefill", "vision", "vae"):
        assert name not in body, f"{name!r} assumes the model"
