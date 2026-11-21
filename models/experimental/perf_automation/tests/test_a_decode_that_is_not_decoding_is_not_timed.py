# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The decode loop that gets TIMED is the one nothing else checks.

check_pcc runs the model's own e2e test, which drives the forward while advancing the position ITSELF
and teacher-forcing reference tokens. So the wrapper measured by trace_replay -- its token pick, its
position advance, its feeding of its own output back in -- never reaches the correctness gate. A
wrapper that forgot to advance would re-run one position forever: plausibly FASTER, since the cache
never grows, and completely wrong. These pin the check that catches it, and -- just as important --
the cases it must NOT convict.
"""

import pytest

from models.experimental.perf_automation.agent.perf_adapter import advance_verdict


class _Tensor:
    """The smallest thing that answers the questions the walk asks of a tensor."""

    def __init__(self, values):
        self._v = list(values)

    def numel(self):
        return len(self._v)

    def item(self):
        return self._v[0]

    def sum(self):
        return sum(self._v)


def _state(pos, logits):
    """A decode state shaped like the ones pipelines actually hand back."""
    return {"current_pos": _Tensor([pos]), "iteration": pos, "out_tok": _Tensor(logits)}


def test_a_counter_that_climbs_reads_as_progress() -> None:
    out = advance_verdict([_state(4, [1.0, 2.0]), _state(5, [1.0, 3.0]), _state(6, [1.0, 4.0])])
    assert out["status"] == "holds", out


def test_a_step_that_repeats_itself_exactly_is_refused() -> None:
    """THE case: position never advances, so every step re-runs the same token and looks fast."""
    stuck = [_state(4, [1.0, 2.0]) for _ in range(3)]
    out = advance_verdict(stuck)
    assert out["status"] == "stuck", out
    assert "one position" in out["reason"], out


def test_a_position_kept_on_the_device_is_not_mistaken_for_a_stuck_loop() -> None:
    """A pipeline may hold its position in a device buffer this cannot read.

    Its host-visible numbers then stand still while it decodes perfectly well. Convicting on the
    counter alone would fail a correct model, so a changing output has to count as movement.
    """
    moving = [
        {"out_tok": _Tensor([1.0, 2.0])},
        {"out_tok": _Tensor([1.0, 9.0])},
        {"out_tok": _Tensor([1.0, 5.0])},
    ]
    out = advance_verdict(moving)
    assert out["status"] == "holds", out
    assert "output differs" in out["reason"], out


def test_a_step_that_hands_back_nothing_readable_says_so() -> None:
    """Absence of evidence is not evidence: unreadable is `unverified`, never `stuck`."""
    assert advance_verdict([None, None, None])["status"] == "unverified"
    assert advance_verdict([object(), object()])["status"] == "unverified"


def test_one_sample_is_not_enough_to_judge() -> None:
    assert advance_verdict([_state(1, [1.0])])["status"] == "unverified"
    assert advance_verdict([])["status"] == "unverified"


def test_the_check_reads_the_state_rather_than_field_names() -> None:
    """Every model names its own state, so a check keyed on `current_pos` would stop checking.

    Same states as the first test with every key renamed to something no name table would carry.
    """
    renamed = [
        {"zzz_offset": _Tensor([4]), "blorp": 4},
        {"zzz_offset": _Tensor([5]), "blorp": 5},
        {"zzz_offset": _Tensor([6]), "blorp": 6},
    ]
    assert advance_verdict(renamed)["status"] == "holds"


def test_a_flag_flipping_is_not_progress() -> None:
    """A bool that toggles is not a position advancing, and must not be read as one."""
    toggling = [{"first": True, "out": _Tensor([1.0, 1.0])}, {"first": False, "out": _Tensor([1.0, 1.0])}]
    assert advance_verdict(toggling)["status"] == "stuck"


def test_it_works_on_the_tensors_a_real_pipeline_returns() -> None:
    torch = pytest.importorskip("torch")

    climbing = [
        {"pos": torch.tensor([4]), "logits": torch.tensor([1.0, 2.0])},
        {"pos": torch.tensor([5]), "logits": torch.tensor([1.0, 3.0])},
    ]
    assert advance_verdict(climbing)["status"] == "holds"
    frozen = [{"pos": torch.tensor([4]), "logits": torch.tensor([1.0, 2.0])} for _ in range(2)]
    assert advance_verdict(frozen)["status"] == "stuck"


def test_both_measurement_paths_run_the_same_warmup() -> None:
    """The capture path and the self-traced path each have their own warmup loop.

    gemma3 declares self_traced, voxtral does not, so a check wired into one of them would silently
    not run for half the models the tool produces. Both must go through the shared helper.
    """
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "agent" / "trace_replay.py"
    body = src.read_text()
    for fn in ("def _capture_step_trace", "def _measure_native"):
        start = body.index(fn)
        window = body[start : body.index("\ndef ", start + 1)]
        assert "_warm(" in window, "%s no longer runs the shared warmup" % fn
    assert body.count("_check_advance(_warm(") == 2, "both paths must check, and only via the warmup"


def test_the_marker_the_gate_reads_is_the_marker_the_run_prints() -> None:
    """The refusal travels between two processes as a printed line, so both sides must agree.

    It has to be a marker rather than the exception: the generated perf test wraps the traced pass in
    `except Exception`, so the raise alone lands as TRACE_REPLAY_SKIPPED and the gate falls back to
    the eager wall -- still a number, still banked, still measured on a step going nowhere.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    printed = (root / "agent" / "trace_replay.py").read_text()
    parsed = (root / "cc_optimize" / "perf_mcp.py").read_text()
    assert 'print("TRACE_DECODE_ADVANCE=%s"' in printed, "the run no longer prints the marker"
    assert '_DECODE_STUCK_MARKER = "TRACE_DECODE_ADVANCE=stuck"' in parsed, "the gate reads a different name"
    assert "if decode_stuck:" in parsed, "the gate reads the marker but does not act on it"


def test_the_eager_fallback_cannot_rescue_a_stuck_decode() -> None:
    """The refusal must land BEFORE the eager wall is considered, or it changes nothing.

    A stuck decode fails the trace, and the eager path then times the same non-advancing step. The
    check therefore has to sit ahead of both readings, not merely ahead of the trace one.
    """
    from pathlib import Path

    body = (Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py").read_text()
    assert body.index("if decode_stuck:") < body.index('if walls:\n        return statistics.median(walls), "eager"')


def test_a_stuck_decode_stops_the_measurement() -> None:
    """The verdict has to REFUSE, not just print. A reported number nobody blocks is banked."""
    pytest.importorskip("ttnn")
    from models.experimental.perf_automation.agent.perf_adapter import DecodeNotAdvancing
    from models.experimental.perf_automation.agent.trace_replay import _check_advance

    with pytest.raises(DecodeNotAdvancing):  # allow-pytest.raises: no expect_error fixture
        _check_advance([_state(4, [1.0, 2.0]) for _ in range(3)])
    _check_advance([_state(4, [1.0, 2.0]), _state(5, [1.0, 3.0])])  # a moving decode passes through
