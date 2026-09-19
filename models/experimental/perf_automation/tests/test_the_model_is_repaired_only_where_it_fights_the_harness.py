# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Repair what the model gets WRONG about the harness; refer the rest to emit-e2e.

Sweeping 87 models split the findings cleanly. The PORTING gaps -- PIPELINE_STAGES, the per-stage
hooks, the self-tests -- fire on essentially every hand-written model because they are emit-e2e's
OUTPUT shape; generating them needs the model's stage decomposition and reference outputs, which is
a port, not a patch. The COMPATIBILITY gaps are small, local, identical every time, and mean the
model actively fights the harness. Only those are repaired here.

The one that matters today is trace-authority: a trace gate the harness cannot reach. Two models
still carry it -- gpt_oss and llama3_1_8b_p150, the latter already optimized by this tool -- and it
is what produced 194 fatals and a dead baseline.
"""
from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

from agent.model_contract import check  # noqa: E402
from agent.model_repair import apply, plan, report  # noqa: E402

_GATE = '''
    class Args:
        def get_trace_prefill_supported_seq_lens(self):
            """Docstring, followed by a blank line -- the shape that broke the first version."""

            default = {"P150": [128]}
            return default["P150"]

        def can_enable_trace(self, seq_len, cached=0):
            allowed = self.trace_prefill_supported_seq_lens
            return seq_len in allowed
'''


def _model(tmp_path, body=_GATE, name="model_config.py"):
    (tmp_path / name).write_text(textwrap.dedent(body))
    return tmp_path


# ---------------------------------------------------------------- it repairs, and proves it


def test_the_repair_clears_the_clause_it_targets(tmp_path):
    root = _model(tmp_path)
    assert [f.clause for f in check(root) if f.blocking] == ["trace-authority"]
    res = apply(root)
    assert res["cleared"] is True, res["remaining"]
    assert res["written"]


def test_the_guard_returns_the_right_EMPTY_for_each_gate(tmp_path):
    """A list-returning gate must return [], a predicate must return False. Returning [] from a
    predicate is falsy and would appear to work, until something compares it to False."""
    e = plan(_model(tmp_path))[0]
    seq = e.after.split("def get_trace_prefill_supported_seq_lens")[1].split("def ")[0]
    pred = e.after.split("def can_enable_trace")[1].split("def ")[0]
    assert "return []" in seq and "return False" not in seq
    assert "return False" in pred and "return []" not in pred


def test_a_docstring_followed_by_a_blank_line_still_parses(tmp_path):
    """THE BUG THAT SHIPPED IN THE FIRST VERSION. The insert point was 'the line after the
    docstring' and the indent was that line's leading whitespace -- on a blank line that is '\\n',
    so the guard was inserted into the gap at nonsense indentation and the file stopped parsing.
    The SyntaxError was then swallowed by `except SyntaxError: continue`, so the repair reported
    'nothing mechanically repairable' for precisely the two models it exists for.

    Both come from the AST now: the first real statement's lineno and its col_offset."""
    e = plan(_model(tmp_path))[0]
    ast.parse(e.after)  # the assertion
    assert '"""Docstring, followed by a blank line' in e.after, "the docstring was destroyed"


def test_the_guard_precedes_the_body_it_guards(tmp_path):
    """A guard after the first statement lets that statement run -- which for a gate is the thing
    being prevented."""
    e = plan(_model(tmp_path))[0]
    body = e.after.split("def can_enable_trace")[1]
    assert body.index("TT_METAL_DEVICE_PROFILER") < body.index("allowed = self.")


def test_nothing_else_in_the_file_is_touched(tmp_path):
    e = plan(_model(tmp_path))[0]
    removed = [l for l in e.before.splitlines() if l.strip() and l not in e.after.splitlines()]
    assert removed == [], removed


# ---------------------------------------------------------------- it declines, correctly


def test_a_gate_that_already_consults_the_harness_is_left_alone(tmp_path):
    root = _model(
        tmp_path,
        body="""
        import os
        class Args:
            def can_enable_trace(self, seq_len, cached=0):
                if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
                    return False
                if os.environ.get("TT_PERF_TRACE") == "0":
                    return False
                return seq_len in (128,)
        """,
    )
    assert plan(root) == []


def test_a_model_with_no_gate_gets_no_edits(tmp_path):
    assert plan(_model(tmp_path, body="X = 1\n")) == []


def test_porting_gaps_are_never_patched(tmp_path):
    """PIPELINE_STAGES and the self-tests need the model's stage decomposition and reference
    outputs. That is emit-e2e's job; a second generator would duplicate it and diverge."""
    root = _model(tmp_path, body="X = 1\n")
    gaps = [f.clause for f in check(root)]
    assert "stages" in gaps and "selftests" in gaps
    assert plan(root) == [], "repair attempted a porting gap"


# ---------------------------------------------------------------- it does not write unasked


def test_plan_writes_nothing(tmp_path):
    root = _model(tmp_path)
    before = (root / "model_config.py").read_text()
    plan(root)
    assert (root / "model_config.py").read_text() == before


def test_a_repair_that_does_not_clear_its_clause_reports_failure(tmp_path, monkeypatch):
    """A repair is only a repair if the contract agrees afterwards. Reporting success and leaving
    the run to discover otherwise is the failure mode this exists to prevent."""
    import agent.model_repair as MR

    root = _model(tmp_path)
    monkeypatch.setattr(MR, "plan", lambda r: [])  # apply nothing
    res = MR.apply(root, [])
    assert res["cleared"] is False and res["remaining"]


def test_an_unpatchable_file_is_reported_not_silently_skipped(tmp_path, monkeypatch):
    """The swallowed SyntaxError is why the first version claimed there was nothing to do."""
    import agent.model_repair as MR

    root = _model(tmp_path)
    monkeypatch.setattr(MR, "_GUARD_SRC", "this is not python(\n")
    assert plan(root) == []
    assert "could not patch" in report([], root)
