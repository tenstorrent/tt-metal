"""Accuracy buys speed only for a stage that still needs it.

check_pcc asks one question -- is the reading still above the model's own floor -- so every reading
above that floor is spendable by whichever stage happens to be measured next. Accuracy does not come
back: the only way to recover it is to revert the lever that spent it. So the budget is finite and
one-way, and the stages that need it are the ones the ceiling reports as short of achievable.

On voxtral_mini_3b_2507 (2026-09-07) a round spent 0.9601 -> 0.9576 of a 0.95 floor on the recurring
stage, which was already PAST its band, while the prompt stage sat 72 ms above its own and did not
move. A quarter of the remaining budget went to the one stack that had nothing left to buy, and the
stack that had to buy its own fix was left with less than it started the round with.

The floor gate is unchanged and still end-to-end. This adds the second question: whose stage is this,
and does that stage still need the money.

No stage is named here. The improved stages come from the verdict and the short ones from the
ceiling, so the rule reads whatever this model calls its stacks.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_KERNEL_LOG", str(tmp_path / "kernel_attempts.json"))
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


# Stacks this model happens to call these things. The rule must never recognise the words.
_RECURRING = "the-stack-that-repeats"
_PROMPT = "the-stack-that-is-short"


def _verdicts(m, pcc, improved, banked=None):
    stages = {
        _RECURRING: {"ms": 10.0, "best": 10.4, "improved": _RECURRING in improved, "regressed": False},
        _PROMPT: {"ms": 106.0, "best": 106.2, "improved": _PROMPT in improved, "regressed": False},
    }
    m.record_gate_verdict("full_pipeline", "ok", full_pipeline_ms=10.0, best_ms=10.4, stages=stages)
    m.record_gate_verdict("pcc", "ok", pcc=pcc)
    if banked is not None:
        m.record_gate_verdict(m._PCC_BANKED, "banked", pcc=banked)


def _short(m, monkeypatch, *names):
    monkeypatch.setattr(m, "_stages_short_of_achievable", lambda: [{"stage": n} for n in names])


# ---------------------------------------------------------------- the rule


def test_accuracy_bought_for_a_stage_in_its_band_is_refused(mcp, monkeypatch):
    """The voxtral case: paid for the stack that was already past its band."""
    _verdicts(mcp, pcc=0.9576, improved=(_RECURRING,), banked=0.9601)
    _short(mcp, monkeypatch, _PROMPT)

    allowed, why = mcp.gates_allow_banking()

    assert allowed is False, why
    assert _PROMPT in why and "0.9601" in why and "0.9576" in why


def test_accuracy_bought_for_a_stage_that_is_short_is_allowed(mcp, monkeypatch):
    """The budget exists to be spent -- on the stack that still has something to buy."""
    _verdicts(mcp, pcc=0.9576, improved=(_PROMPT,), banked=0.9601)
    _short(mcp, monkeypatch, _PROMPT)

    allowed, why = mcp.gates_allow_banking()

    assert allowed is True, why


def test_a_win_that_costs_no_accuracy_is_allowed(mcp, monkeypatch):
    """Only a FALL is a purchase. Holding the reading buys nothing and owes nothing."""
    _verdicts(mcp, pcc=0.9601, improved=(_RECURRING,), banked=0.9601)
    _short(mcp, monkeypatch, _PROMPT)

    allowed, why = mcp.gates_allow_banking()

    assert allowed is True, why


def test_with_nothing_banked_yet_the_gate_is_unchanged(mcp, monkeypatch):
    """Before the first win there is nothing to compare against, so the rule stays silent."""
    _verdicts(mcp, pcc=0.9576, improved=(_RECURRING,), banked=None)
    _short(mcp, monkeypatch, _PROMPT)

    allowed, why = mcp.gates_allow_banking()

    assert allowed is True, why


def test_with_no_stage_short_the_gate_is_unchanged(mcp, monkeypatch):
    """Every stack inside its band: there is no one the budget was taken from."""
    _verdicts(mcp, pcc=0.9576, improved=(_RECURRING,), banked=0.9601)
    _short(mcp, monkeypatch)

    allowed, why = mcp.gates_allow_banking()

    assert allowed is True, why


def test_a_reading_that_only_wobbled_is_not_a_purchase(mcp, monkeypatch):
    """Accuracy is measured, so it moves on its own. A wobble must not read as a spend."""
    _verdicts(mcp, pcc=0.9601 - mcp._PCC_SPEND_EPS / 2.0, improved=(_RECURRING,), banked=0.9601)
    _short(mcp, monkeypatch, _PROMPT)

    allowed, why = mcp.gates_allow_banking()

    assert allowed is True, why


def test_banking_a_win_records_the_accuracy_it_was_banked_at(mcp):
    """Nothing to compare against unless the win writes down what it cost."""
    mcp.record_gate_verdict("pcc", "ok", pcc=0.9612)

    mcp._bank_pcc()

    assert (mcp.gate_verdicts().get(mcp._PCC_BANKED) or {}).get("pcc") == 0.9612


# ---------------------------------------------------------------- what a win is worth


def test_a_stage_delta_states_the_time_it_moved_not_only_the_fraction(mcp):
    """A percent is measured against its own stage, so the same percent is a different amount.

    Read as one column the fractions rank the small stack first: 1% of the repeating stack is a
    tenth of a millisecond, 1% of the prompt stack is a whole one.
    """
    now = {_RECURRING: 9.9, _PROMPT: 104.94}
    bar = {_RECURRING: 10.0, _PROMPT: 106.0}

    rows = mcp._stage_deltas(now, bar, {_RECURRING: 0.0005, _PROMPT: 0.0335})

    assert rows[_RECURRING]["delta_pct"] == rows[_PROMPT]["delta_pct"] == -1.0
    assert rows[_RECURRING]["delta_ms"] == -0.1
    assert rows[_PROMPT]["delta_ms"] == -1.06
    # and the bar each had to clear, in the same unit
    assert rows[_RECURRING]["tol_ms"] < rows[_PROMPT]["tol_ms"]
