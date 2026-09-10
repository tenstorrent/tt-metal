# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every place a declared stack can drop out between PIPELINE_STAGES and a roofline row.

A stage reaches the report only if it was MEASURED: summary renders `stage_ms`, which the run writes
from what it actually timed. So anything that removes a stage on the way costs it a row, and the
report is complete-looking either way -- two towers for a three-tower model, with nothing saying one
is missing. There are three such places, and this pins all of them:

    1. the adapter, when the model declares a stage but exposes no <stage>_trace_step
    2. the adapter, when the stage's own _trace_inputs raises
    3. trace_replay, when measuring that stage raises

The rule is the same at all three: the faulty stage loses its row, says why, and the others keep
theirs. Losing one stage is a gap; losing the run for one stage's fault is a worse one, and silence
is worse than both.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
for _p in (str(_PA), str(_PA.parent.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent import stage_seams as _seams  # noqa: E402


# SET, NOT SPELLED. test_the_model_contract_is_checked_before_anything_runs runs check() over THIS
# DIRECTORY, so a `PIPELINE_STAGES = [...]` literal anywhere in it is read as a real model's declared
# stages and every contract clause then fires against this fixture. Binding the attribute after the
# class is defined keeps the fixture out of that source scan -- the adapter reads the attribute, and
# does not care how it got there.
_STAGE_NAMES = ("zzz_a", "zzz_b", "zzz_c")


class _Pipe:
    """Three declared stages; which hooks exist is decided per test."""

    def __init__(self, *, drop_step=(), raise_inputs=()):
        for _n in self.PIPELINE_STAGES:
            if _n not in drop_step:
                setattr(self, _seams.hook(_n, _seams.STEP), lambda: None)
            setattr(self, _seams.hook(_n, _seams.SETUP), lambda _i=None: None)
            if _n in raise_inputs:
                setattr(self, _seams.hook(_n, _seams.INPUTS), self._boom)

    @staticmethod
    def _boom():
        raise FileNotFoundError("a capture this model declares but does not ship")


setattr(_Pipe, "PIPELINE_STAGES", list(_STAGE_NAMES))


def _stages_of(pipe):
    from agent.perf_adapter import PipelineStageAdapter

    a = PipelineStageAdapter(lambda _d: pipe, None, batch=1)
    a.setup(object())
    return [s.name for s in a.stages]


def test_every_declared_stage_that_can_be_measured_gets_one(capsys):
    """The baseline: nothing faulty, nothing lost."""
    assert _stages_of(_Pipe()) == ["zzz_a", "zzz_b", "zzz_c"]


def test_a_stage_with_no_step_hook_is_dropped_OUT_LOUD(capsys):
    """DROP 1. This was a bare `continue`: the stage left adapter.stages, stage_ms and the roofline
    without a word. The contract warns at preflight; this is the same fact where it costs the row."""
    got = _stages_of(_Pipe(drop_step=("zzz_b",)))
    err = capsys.readouterr().err
    assert got == ["zzz_a", "zzz_c"], got
    assert "zzz_b" in err and _seams.hook("zzz_b", _seams.STEP) in err, err
    assert "zzz_a" not in err and "zzz_c" not in err, "a healthy stage was reported as lost"


def test_a_stage_whose_inputs_raise_does_not_cost_the_others(capsys):
    """DROP 2. Already guarded -- pinned here so all three places are asserted in one file."""
    got = _stages_of(_Pipe(raise_inputs=("zzz_a",)))
    err = capsys.readouterr().err
    assert got == ["zzz_b", "zzz_c"], got
    assert "zzz_a" in err and "FileNotFoundError" in err, err


def test_a_stage_that_cannot_be_measured_does_not_cost_the_others():
    """DROP 3. The measure loop had NO guard, so one stage raising lost every stage -- the whole
    replay and every roofline row -- for a fault in one tower.

    ASSERTED ON THE SOURCE, not by driving it, and for the same reason perf_adapter's docstring
    gives for living where it does: trace_replay imports ttnn at module scope, so it cannot be
    imported -- let alone run -- without a device. The two drops above are behavioural because the
    adapter can be; this one names the properties instead: guarded, continues, and says why.
    """
    src = (_PA / "agent" / "trace_replay.py").read_text()
    i = src.index("for st in stages:")
    body = src[i : src.index("pipeline_ms = ", i)]
    assert "try:" in body, "measuring a stage is unguarded: one fault loses them all"
    assert "_measure_stage(device, st)" in body
    assert "continue" in body, "a stage that cannot be measured must not abort the others"
    assert "could not be measured" in body, "the drop is silent"
