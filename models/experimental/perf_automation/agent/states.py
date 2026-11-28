"""Canonical state names + loop constants for the Agent Loop (PLAN section 8).

Single source of truth so handlers never typo a transition. Every handler
returns one of these strings; the engine (agent/engine.py) dispatches on it.
"""

from __future__ import annotations

# Entry (written by the Before Loop) ----------------------------------------
BEFORE_LOOP_DONE = "BEFORE_LOOP_DONE"

# Outer loop: decide & act (Member 1) ----------------------------------------
ROUTE = "ROUTE"
SELECT = "SELECT"
PLAN = "PLAN"
APPLY = "APPLY"
VERIFY = "VERIFY"
REPAIR_CODE = "REPAIR_CODE"
REPAIR_PCC = "REPAIR_PCC"

# Evaluate & record (Member 2) -----------------------------------------------
GATE_PCC = "GATE_PCC"
REMEASURE = "REMEASURE"
DECIDE = "DECIDE"
COMMIT = "COMMIT"
REVERT = "REVERT"
LOG = "LOG"
CHECK_EXIT = "CHECK_EXIT"

# Terminals ------------------------------------------------------------------
DONE = "DONE"
STOPPED = "STOPPED"
FAILED = "FAILED"
TERMINAL = frozenset({DONE, STOPPED, FAILED})

# Repair budgets (decided 2026-06-11) ----------------------------------------
MAX_CODE_FIX = 5  # parse / import / run-crash repairs before ABANDON
MAX_CODE_FIX_PRINCIPLES = 8
MAX_CODE_FIX_KERNEL = 12  # tt-lang kernel authoring: writing a correct kernel needs more repair rounds
MAX_PCC_FIX = 2  # PCC-below-threshold repairs before DISCARD
MAX_INERT_RETRY = 6
JUDGE_STREAK_THRESHOLD = 3
MAX_STRUCT_FIX = 3

FROM_PRINCIPLES = "auto-principles"
KERNEL_LEVER = "tt-lang-kernel"  # GUIDELINES anchor for the tt-lang custom-kernel lever


def code_fix_budget(lever: str | None) -> int:
    """Repair budget for the selected lever. Authoring a tt-lang kernel is the hardest edit
    (compile + correctness), so it gets the largest budget; off-menu from-principles next."""
    if lever == KERNEL_LEVER:
        return MAX_CODE_FIX_KERNEL
    return MAX_CODE_FIX_PRINCIPLES if lever == FROM_PRINCIPLES else MAX_CODE_FIX


# Reference transition map (documentation + a test can assert handlers conform).
# Conditional edges (verdicts, counters) are decided INSIDE the handler; this
# lists every state a handler may legally return.
TRANSITIONS = {
    BEFORE_LOOP_DONE: [ROUTE],
    ROUTE: [SELECT],
    SELECT: [PLAN],
    PLAN: [APPLY, REVERT],
    APPLY: [VERIFY],
    VERIFY: [GATE_PCC, REPAIR_CODE, REVERT],
    REPAIR_CODE: [VERIFY],
    REPAIR_PCC: [VERIFY],
    GATE_PCC: [REMEASURE, REPAIR_PCC, REPAIR_CODE, REVERT],
    REMEASURE: [DECIDE, REVERT, REPAIR_CODE],
    DECIDE: [COMMIT, REVERT, APPLY],
    COMMIT: [LOG],
    REVERT: [LOG],
    LOG: [CHECK_EXIT],
    CHECK_EXIT: [ROUTE, DONE, STOPPED],
}
