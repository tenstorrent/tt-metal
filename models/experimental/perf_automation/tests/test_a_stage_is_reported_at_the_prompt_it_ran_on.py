# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A stage's time is reported, and its ceiling priced, at the prompt length the stage actually ran on.

MiniCPM5-2B, 2026-11: its <stage>_trace_inputs() hooks took no argument and rebuilt the request the
bring-up captured -- 20 tokens -- so decode was timed at a 20-token context (10.07 ms, 99.3 tok/s/u).
The scorecard quoted ISL=128, the roofline priced KV at 128 tokens, and the same pipeline run at 128
measured 10.70 ms (93.5). Three things fix it, all pinned here:

    1. a hook that names the INPUTS_PROMPT_ARG keyword is handed the run's declared prompt;
    2. a zero-arg hook keeps its own inputs, and the length they carry is read off them and stated;
    3. perf_mcp takes the length the stages report over the declared one, and says when they differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_PA = Path(__file__).resolve().parents[1]
for _p in (str(_PA), str(_PA.parent.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent import perf_adapter as PA  # noqa: E402
from agent import stage_seams as _seams  # noqa: E402

# Bound after the class, never spelled as a literal: the contract check scans this directory.
_OWN, _OFFERED, _WRAPPED, _IGNORES = "zzz_own", "zzz_offered", "zzz_wrapped", "zzz_ignores"
_CAPTURED_LEN, _DECLARED_LEN, _BATCH = 20, 128, 4


class _Pipe:
    def __init__(self):
        self.offered = {}
        for name in self.PIPELINE_STAGES:
            setattr(self, _seams.hook(name, _seams.STEP), lambda: None)
            setattr(self, _seams.hook(name, _seams.SETUP), lambda _i=None: None)
        captured = {
            "input_ids": torch.zeros(_BATCH, _CAPTURED_LEN, dtype=torch.long),
            "noise": torch.zeros(_BATCH, 7, dtype=torch.float32),
        }
        setattr(self, _seams.hook(_OWN, _seams.INPUTS), lambda: captured)

        def offered(prompt_ids=None):
            self.offered[_OFFERED] = prompt_ids
            return {"input_ids": prompt_ids}

        setattr(self, _seams.hook(_OFFERED, _seams.INPUTS), offered)

        def wrapped(*args, **kwargs):
            if args or kwargs:
                raise TypeError("the wrapped hook takes no arguments")
            return captured

        setattr(self, _seams.hook(_WRAPPED, _seams.INPUTS), wrapped)

        def ignores(prompt_ids=None):
            return captured

        setattr(self, _seams.hook(_IGNORES, _seams.INPUTS), ignores)


setattr(_Pipe, "PIPELINE_STAGES", [_OWN, _OFFERED, _WRAPPED, _IGNORES])


def _adapter(pipe, prompt=tuple(range(_DECLARED_LEN))):
    a = PA.PipelineStageAdapter(lambda _d: pipe, list(prompt) if prompt is not None else None, batch=_BATCH)
    a.setup(object())
    return {s.name: s for s in a.stages}


def test_a_hook_that_takes_the_prompt_is_given_the_declared_one():
    pipe = _Pipe()
    stages = _adapter(pipe)
    got = pipe.offered[_OFFERED]
    assert tuple(got.shape) == (_BATCH, _DECLARED_LEN), "one row per user, the whole declared prompt"
    assert got[0].tolist() == list(range(_DECLARED_LEN))
    assert stages[_OFFERED].prompt_tokens == _DECLARED_LEN


def test_a_zero_arg_hook_keeps_its_own_request_and_its_length_is_stated(capsys):
    stages = _adapter(_Pipe())
    assert stages[_OWN].prompt_tokens == _CAPTURED_LEN
    err = capsys.readouterr().err
    assert _OWN in err and "own %d-token request" % _CAPTURED_LEN in err, err
    assert _OFFERED not in err, "a stage given the declared prompt is not reported as off it"


def test_a_hook_that_takes_the_prompt_but_ignores_it_is_reported_at_what_it_built(capsys):
    stages = _adapter(_Pipe())
    assert stages[_IGNORES].prompt_tokens == _CAPTURED_LEN, "what the inputs carry beats what was offered"
    assert "%s ignores the offered" % _seams.hook(_IGNORES, _seams.INPUTS) in capsys.readouterr().err


def test_a_kwargs_wrapper_is_not_offered_a_keyword_it_would_raise_on():
    stages = _adapter(_Pipe())
    assert _WRAPPED in stages, "the stage was lost to an offered keyword"
    assert stages[_WRAPPED].prompt_tokens == _CAPTURED_LEN


def test_with_no_declared_prompt_nothing_is_offered_and_nothing_is_lost():
    pipe = _Pipe()
    stages = _adapter(pipe, prompt=None)
    assert set(stages) == {_OWN, _OFFERED, _WRAPPED, _IGNORES}
    assert pipe.offered[_OFFERED] is None, "the hook falls back to its own request"
    assert stages[_OWN].prompt_tokens == _CAPTURED_LEN


@pytest.mark.parametrize(
    "inputs,want",
    [
        ({"ids": torch.zeros(2, 20, dtype=torch.long), "noise": torch.zeros(2, 9)}, 20),
        ((torch.zeros(1, 33, dtype=torch.int32), [torch.zeros(1, 33, dtype=torch.long)]), 33),
        ({"ids": torch.zeros(2, 20, dtype=torch.long), "other": torch.zeros(2, 21, dtype=torch.long)}, 0),
        ({"pos": torch.zeros(32, dtype=torch.long)}, 0),
        ({"mask": torch.zeros(2, 20, dtype=torch.bool)}, 0),
        ({"x": torch.zeros(2, 20)}, 0),
        (None, 0),
        ("not a tensor", 0),
    ],
)
def test_the_length_is_read_only_when_the_inputs_say_it_unambiguously(inputs, want):
    assert PA.request_tokens(inputs) == want


def test_the_marker_the_replay_prints_is_the_one_perf_mcp_reads():
    line = _seams.prompt_tokens_line("anystage", 20)
    marker = _seams.PROMPT_TOKENS_MARK + "["
    assert line.split(marker, 1)[1].split("]", 1)[0] == "anystage"
    assert int(line.split("]=", 1)[1].split()[0]) == 20
    assert line.startswith("TRACE_"), "trace_replay keeps only TRACE_ lines across a retried step"
    replay = (_PA / "agent" / "trace_replay.py").read_text()
    assert "_prompt_tokens_line(st.name, _pt)" in replay
    mcp = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    assert '(_PROMPT_TOKENS_MARK + "[", stage_prompt_tokens)' in mcp
    assert "prompt_tokens_seen = _measured_prompt_tokens(stage_prompt_tokens, prompt_tokens_seen)" in mcp
    assert "isl = prompt_tokens_seen or " in mcp


def test_perf_mcp_reports_the_length_the_stages_ran_on(capsys):
    from cc_optimize import perf_mcp as pm

    assert pm._measured_prompt_tokens({"a": 20, "b": 20}, 128) == 20
    assert "ran on a 20-token prompt, not the declared 128" in capsys.readouterr().err
    assert pm._measured_prompt_tokens({"a": 128}, 128) == 128
    assert capsys.readouterr().err == "", "agreement is not a warning"
    assert pm._measured_prompt_tokens({}, 128) == 128, "nothing stated: the declared length stands"
    assert pm._measured_prompt_tokens({}, 0) == 0
    assert pm._measured_prompt_tokens({"a": 20}, 0) == 20, "observed with nothing declared"
    assert pm._measured_prompt_tokens({"a": 20, "b": 128}, 128) == 128, "disagreement keeps the declared"
    assert "different prompt lengths" in capsys.readouterr().err
