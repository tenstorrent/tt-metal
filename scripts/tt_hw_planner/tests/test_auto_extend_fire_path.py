"""Integration test for AUTO-EXTEND fire path.

We don't run the full auto_iterate (it needs hardware + HF + LLM). But we
DO run the EXACT while-loop structure from auto_iterate against the REAL
brain (should_extend_budget), with carefully constructed state that
forces:
  1. Budget exhaustion (it > max_iters)
  2. Pending residue (≤ 2 components)
  3. Graduations this run (momentum proxy)

The test asserts:
  - The loop reaches the budget-check branch
  - The brain is invoked
  - The brain says extend=True
  - max_iters is bumped per verdict.bump
  - Loop re-enters (additional iters happen)
  - On second exhaustion, brain refuses (cap=1)
  - Final budget_extensions_used == 1

This is the closest we can get to "the AUTO-EXTEND banner fires in
production" without hardware."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Dict, List, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.agentic.convergence import should_extend_budget


def _simulate_loop(
    *,
    max_iters: int,
    initial_pending: List[str],
    pending_after_budget_exhaust: List[str],
    pcc_history_per_component: Dict[str, Sequence[float]],
    graduated_this_run: List[str],
    permanently_skipped: List[str],
) -> dict:
    """Mirror the auto_iterate while-loop EXACTLY. The body is a no-op
    here (real body needs hardware), but the while-condition, brain call,
    and max_iters mutation are byte-for-byte identical to production."""

    # State the real loop maintains
    budget_extensions_used = 0
    it = 0
    iters_run = 0
    brain_consultations = []
    banners_emitted: List[str] = []
    declines_emitted: List[str] = []

    def banner(msg: str) -> None:
        banners_emitted.append(msg)

    while True:
        it += 1
        if it > max_iters:
            # Production: _ext_ungrad, _ext_smoke = _auto_iteration_blockers(MODEL)
            # Here: we inject what those would have returned.
            _ext_pending = sorted(set(pending_after_budget_exhaust) - set(permanently_skipped))

            # === IDENTICAL TO PRODUCTION FROM HERE ===
            _verdict = should_extend_budget(
                pending_components=_ext_pending,
                pcc_history_per_component=pcc_history_per_component,
                graduated_this_run=graduated_this_run,
                max_iters=max_iters,
                extensions_used=budget_extensions_used,
            )
            brain_consultations.append(
                {
                    "called_at_it": it,
                    "max_iters_when_called": max_iters,
                    "extensions_used_when_called": budget_extensions_used,
                    "verdict_extend": _verdict.extend,
                    "verdict_bump": _verdict.bump,
                    "verdict_reason": _verdict.reason,
                }
            )
            if _verdict.extend:
                banner(
                    f"AUTO-EXTEND (brain G8): max_iters {max_iters} → "
                    f"{max_iters + _verdict.bump} — {_verdict.reason}"
                )
                max_iters += _verdict.bump
                budget_extensions_used += 1
                # Mirrors production: decrement so the bump = real extra
                # body iters, not bump-1.
                it -= 1
                continue
            if _ext_pending:
                declines_emitted.append(f"[brain G8] declined to extend budget: {_verdict.reason}")
            break
        # === body would run here in production ===
        iters_run += 1

    return {
        "iters_run": iters_run,
        "final_max_iters": max_iters,
        "budget_extensions_used": budget_extensions_used,
        "brain_consultations": brain_consultations,
        "banners_emitted": banners_emitted,
        "declines_emitted": declines_emitted,
    }


# ---------------------------------------------------------------------------
# THE MONEY TEST: positive fire on realistic state
# ---------------------------------------------------------------------------


def test_fire_path_extends_then_caps_on_second_exhaustion() -> None:
    """Realistic SAM2-style scenario:
      - max_iters=4
      - 4 components graduated this run (momentum strong)
      - 1 component still pending (decomposition child added late, no PCC history)
    Expected:
      - First exhaustion: brain extends (bump=2 since max_iters//2 < 2 floors)
        Actually max_iters//2 = 2 so bump=max(2,2)=2.
        max_iters: 4 → 6.
      - 2 additional iters run.
      - Second exhaustion: brain refuses (cap=1).
      - Loop exits cleanly.
    """
    result = _simulate_loop(
        max_iters=4,
        initial_pending=["leftover"],
        pending_after_budget_exhaust=["leftover"],
        pcc_history_per_component={},  # late-added, no history
        graduated_this_run=["a", "b", "c", "d"],
        permanently_skipped=[],
    )

    # The loop ran 4 iters, exhausted, brain extended, ran 2 more,
    # exhausted again, brain refused.
    assert result["iters_run"] == 6, f"expected 6 iters total (4+2), got {result['iters_run']}"
    assert result["final_max_iters"] == 6
    assert result["budget_extensions_used"] == 1

    # Brain was consulted TWICE
    assert len(result["brain_consultations"]) == 2

    # First consultation: extension granted
    first = result["brain_consultations"][0]
    assert first["called_at_it"] == 5  # 1 over max_iters=4
    assert first["max_iters_when_called"] == 4
    assert first["extensions_used_when_called"] == 0
    assert first["verdict_extend"] is True
    assert first["verdict_bump"] == 2
    assert "graduations" in first["verdict_reason"] or "momentum" in first["verdict_reason"]

    # Second consultation: cap hit
    second = result["brain_consultations"][1]
    assert second["called_at_it"] == 7  # 1 over max_iters=6
    assert second["max_iters_when_called"] == 6
    assert second["extensions_used_when_called"] == 1
    assert second["verdict_extend"] is False
    assert "already extended" in second["verdict_reason"] or "cap" in second["verdict_reason"]

    # AUTO-EXTEND banner fired ONCE
    assert len(result["banners_emitted"]) == 1
    banner = result["banners_emitted"][0]
    assert "AUTO-EXTEND" in banner
    assert "brain G8" in banner
    assert "max_iters 4 → 6" in banner
    print(f"\n  ✓ BANNER FIRED: {banner}")

    # Decline trace fired ONCE
    assert len(result["declines_emitted"]) == 1
    print(f"  ✓ DECLINE TRACE: {result['declines_emitted'][0]}")


def test_fire_path_skips_extension_when_no_residue() -> None:
    """Scenario: budget exhausts but nothing is pending (everything either
    graduated or routed to CPU). Brain must say no-pending-components and
    skip extension cleanly. No banner, no decline trace."""
    result = _simulate_loop(
        max_iters=4,
        initial_pending=[],
        pending_after_budget_exhaust=[],  # nothing pending
        pcc_history_per_component={},
        graduated_this_run=["a", "b"],
        permanently_skipped=[],
    )
    assert result["iters_run"] == 4
    assert result["budget_extensions_used"] == 0
    assert len(result["banners_emitted"]) == 0
    assert len(result["declines_emitted"]) == 0  # no pending → silent skip
    assert result["brain_consultations"][0]["verdict_extend"] is False
    assert "no-pending" in result["brain_consultations"][0]["verdict_reason"]


def test_fire_path_skips_extension_when_residue_too_large() -> None:
    """Scenario: budget exhausts but 5 components still pending (broadly-
    failing run). Brain refuses; banner does NOT fire; decline trace
    DOES fire so the user sees why."""
    result = _simulate_loop(
        max_iters=4,
        initial_pending=["a", "b", "c", "d", "e"],
        pending_after_budget_exhaust=["a", "b", "c", "d", "e"],
        pcc_history_per_component={c: [0.5, 0.4, 0.3] for c in "abcde"},
        graduated_this_run=["graduated"],
        permanently_skipped=[],
    )
    assert len(result["banners_emitted"]) == 0  # no fire
    assert len(result["declines_emitted"]) == 1  # decline trace
    assert "residue too large" in result["declines_emitted"][0]
    print(f"\n  ✓ DECLINE: {result['declines_emitted'][0]}")


def test_fire_path_extends_on_favorable_trajectory_without_graduations() -> None:
    """Scenario: no graduations this run, but the brain IS making progress
    on the pending component (descending PCC history). Trajectory alone
    is enough for the brain to extend."""
    result = _simulate_loop(
        max_iters=4,
        initial_pending=["progressing"],
        pending_after_budget_exhaust=["progressing"],
        pcc_history_per_component={"progressing": [0.50, 0.30, 0.15, 0.05]},
        graduated_this_run=[],  # no graduations yet
        permanently_skipped=[],
    )
    assert result["budget_extensions_used"] == 1
    assert len(result["banners_emitted"]) == 1
    banner = result["banners_emitted"][0]
    assert "favorable trajectory on progressing" in banner
    print(f"\n  ✓ BANNER FIRED (trajectory alone): {banner}")
