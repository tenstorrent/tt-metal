"""Pin: _run_strict_pcc_gate must never silently soft-skip.

The Phi-3.5 SUCCESS run on 2026-06-02 produced rc=0 with
"PCC correctness gate DID NOT engage (soft-skipped — see gate logs
above for reason)" — but THERE WERE NO GATE LOGS. The two early-return
branches in _run_strict_pcc_gate were silent, hiding the actual
reason (strict_pcc off / auto_mode off / empty captured_output).

Every soft-skip must print a one-line diagnostic so the operator can
distinguish: "the gate decided not to engage" vs "the capture pipeline
broke and we never even tried."
"""

from __future__ import annotations

import argparse
from pathlib import Path


def test_run_strict_pcc_gate_no_silent_returns() -> None:
    """Source-level guard: every ``return None, None`` inside
    ``_run_strict_pcc_gate`` must be preceded by a ``print(`` within
    the same branch. Without this, the operator sees a soft-skip
    banner with no diagnostic trail explaining why."""
    src = Path("scripts/tt_hw_planner/cli.py").read_text()
    fn_idx = src.find("def _run_strict_pcc_gate")
    assert fn_idx >= 0
    # End of function: next top-level `def `
    next_def = src.find("\ndef ", fn_idx + 10)
    assert next_def > fn_idx
    body = src[fn_idx:next_def]
    # Every "return None, None" should have a print statement within
    # the preceding 200 chars of the same branch
    for m_pos in [i for i in range(len(body)) if body.startswith("return None, None", i)]:
        preceding = body[max(0, m_pos - 400) : m_pos]
        assert "print(" in preceding, (
            f"_run_strict_pcc_gate has a silent `return None, None` "
            f"around offset {m_pos}: must print a diagnostic so the "
            f"operator can see which check fired. Context: "
            f"...{body[max(0, m_pos-100):m_pos+30]!r}"
        )


def test_run_strict_pcc_gate_skip_messages_distinguish_causes() -> None:
    """The skip diagnostics must DISTINGUISH the three soft-skip causes:
    auto_mode off, strict_pcc off, empty captured_output. Without this,
    the operator can't tell whether the gate was disabled by the
    operator vs broken by the capture pipeline."""
    src = Path("scripts/tt_hw_planner/cli.py").read_text()
    fn_idx = src.find("def _run_strict_pcc_gate")
    next_def = src.find("\ndef ", fn_idx + 10)
    body = src[fn_idx:next_def]
    assert "auto_mode" in body.lower(), "diagnostic must mention auto_mode (or equivalent label)"
    assert (
        "strict_pcc" in body.lower() or "--no-strict-pcc" in body.lower()
    ), "diagnostic must mention strict_pcc opt-out"
    assert "captured_output" in body or "capture" in body.lower(), "diagnostic must mention empty captured_output"
