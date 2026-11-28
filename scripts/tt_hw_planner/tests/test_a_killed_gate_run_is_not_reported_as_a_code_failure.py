# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A run that was KILLED must not reach the agent as a run that FAILED.

A step reports through two channels: its output and its exit status. A signal death says nothing in
the first -- SIGKILL has no handler, so there is no traceback and the output stops mid-line -- and
this branch read only the output, quoting the raw returncode and the literal last 15 lines. On a
Qwen-Image-Edit bring-up that meant a run terminated from outside arrived as:

    G2/G3: tests/e2e did not pass (pytest rc=-9); tail:
      <15 lines of "ttnn.all_gather args are deprecated">

The agent then spent a round inferring the kill from `ps` output before concluding -- correctly --
that nothing in the model was wrong. "You were killed" and "your code is wrong" call for opposite
responses, and the exit status already holds the whole fact: a negative returncode IS the signal.

Both halves already existed and are reused rather than re-spelled: `signal_note` names the signal,
and `_extract_error` picks a log's real failure lines (its whitelist keeps the stage markers, so how
far a hang got survives even though it has no exception to anchor on).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from models.experimental.perf_automation.agent import perf_adapter as PA
from models.experimental.perf_automation.agent import probes as _PR
from models.experimental.perf_automation.agent.perf_test_gen import signal_note
from scripts.tt_hw_planner.commands import emit_e2e as E

_NOISE = "\n".join(
    "2026-09-29 13:00:0%d.000 | warning | Op | The following ttnn.all_gather args are deprecated" % (i % 10)
    for i in range(20)
)


def _demo(tmp_path):
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tests" / "e2e").mkdir(parents=True)
    (demo / "demo").mkdir()
    (demo / "tt").mkdir()
    (demo / "tests" / "e2e" / "test_e2e_m.py").write_text("def test_e2e():\n    pass\n")
    (demo / "demo" / "demo_m.py").write_text("if __name__ == '__main__':\n    pass\n")
    (demo / "README.md").write_text("# m\n")
    return demo


def _gate_reasons(monkeypatch, tmp_path, rc, output):
    """Run the real correctness gate with a step that exits `rc` having printed `output`."""
    monkeypatch.setenv("E2E_REQUIRE_ON_DEVICE", "0")
    monkeypatch.delenv(PA.BATCH_ENV, raising=False)
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-1")

    def _exec(cmd, cwd, env, timeout_s, log_path, **k):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(output)
        return rc

    class _NoProbe:
        returncode = 1

        def __init__(self, *a, **k):
            import os

            self.pid = os.getpid()

        def communicate(self, timeout=None):
            return "", ""

        def poll(self):
            return 1

    monkeypatch.setattr(_PR, "_execute", _exec)
    monkeypatch.setattr(E.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", ""))
    monkeypatch.setattr(E.subprocess, "Popen", _NoProbe)
    _, reasons = E._run_deterministic_gates(_demo(tmp_path), 0.99, 60)
    return [r for r in reasons if r.startswith("G2/G3")]


# --- a kill is reported as a kill -----------------------------------------------------------------


def test_a_killed_run_says_it_was_killed_and_names_the_signal(monkeypatch, tmp_path):
    reasons = _gate_reasons(monkeypatch, tmp_path, -9, _NOISE)
    said = " ".join(reasons)
    assert "KILLED" in said, said
    assert "SIGKILL" in said and "rc=-9" in said
    assert "did not pass (pytest rc=" not in said, "a kill must not be worded as an ordinary failure"


def test_it_tells_the_agent_not_to_rewrite_working_code(monkeypatch, tmp_path):
    """The whole cost of this bug was edits (or four hours of forensics) aimed at innocent code."""
    said = " ".join(_gate_reasons(monkeypatch, tmp_path, -9, _NOISE))
    assert "do not rewrite working code" in said.lower()
    assert "no traceback" in said.lower(), "it must explain WHY there is nothing to read"


def test_any_signal_is_named_not_just_sigkill(monkeypatch, tmp_path):
    said = " ".join(_gate_reasons(monkeypatch, tmp_path, -15, _NOISE))
    assert "SIGTERM" in said and "rc=-15" in said


# --- an ordinary failure is unchanged -------------------------------------------------------------


def test_an_ordinary_failure_keeps_its_existing_wording(monkeypatch, tmp_path):
    reasons = _gate_reasons(monkeypatch, tmp_path, 1, _NOISE + "\nE   assert 0.96 >= 0.99\n")
    said = " ".join(reasons)
    assert "did not pass (pytest rc=1)" in said
    assert "KILLED" not in said


def test_a_passing_run_reports_nothing_here(monkeypatch, tmp_path):
    assert _gate_reasons(monkeypatch, tmp_path, 0, "1 passed") == []


# --- what it quotes -------------------------------------------------------------------------------


def test_the_real_failure_line_is_quoted_not_the_literal_tail(monkeypatch, tmp_path):
    """The last 15 lines of a long device log are routine chatter; the failure is further up."""
    out = _NOISE + "\nE   assert 0.96 >= 0.99\n" + _NOISE
    said = " ".join(_gate_reasons(monkeypatch, tmp_path, 1, out))
    assert "assert 0.96 >= 0.99" in said, "the anchored failure line was dropped"


def test_a_hangs_progress_markers_survive(monkeypatch, tmp_path):
    """A kill has no exception, so HOW FAR IT GOT is the only evidence there is."""
    out = _NOISE + "\nTRACE_STAGE_BYTES[some_stage]=123 ops=45\n" + _NOISE
    said = " ".join(_gate_reasons(monkeypatch, tmp_path, -9, out))
    assert "TRACE_STAGE_BYTES[some_stage]=123" in said


def test_a_log_with_nothing_anchorable_still_quotes_something(monkeypatch, tmp_path):
    said = " ".join(_gate_reasons(monkeypatch, tmp_path, -9, _NOISE))
    assert "deprecated" in said, "with no anchor it must fall back, not go silent"


# --- constraints ----------------------------------------------------------------------------------


def test_the_gate_reuses_the_existing_helpers_instead_of_its_own():
    """Constraint: no parallel implementation of "what does this returncode mean".

    The EXECUTABLE body is scanned -- `ast.unparse` drops comments, which may name a signal while
    explaining the case history, exactly as this suite's other guards allow prose to."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(E._run_deterministic_gates)))
    node = tree.body[0]
    if ast.get_docstring(node) is not None:
        node.body = node.body[1:]
    body = ast.unparse(node)
    assert "signal_note" in body and "_extract_error" in body
    # It may SEND a signal (the G6 probe kills its own process group); what it must not do is work
    # out what a returncode means, which is signal_note's job and must have exactly one owner.
    assert "signal.Signals" not in body, "the gate must not re-derive a signal name"
    assert "Signals(" not in body


def test_signal_note_is_the_single_authority_and_is_unchanged():
    assert signal_note(-9) == "terminated by SIGKILL (rc=-9)"
    assert signal_note(0) == "" and signal_note(1) == "" and signal_note(None) == ""
    assert signal_note("x") == ""


def test_it_names_no_model_or_stage(monkeypatch, tmp_path):
    """The stage name in the quoted markers comes from the log, never from a literal here."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(E._run_deterministic_gates)))
    node = tree.body[0]
    if ast.get_docstring(node) is not None:
        node.body = node.body[1:]
    lowered = ast.unparse(node).lower()
    for name in ("qwen", "denoise", "prefill", "vision_", "audio_", "vae", "some_stage"):
        assert name not in lowered, f"{name!r} would assume the model's vocabulary"
