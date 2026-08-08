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
MAX_PCC_FIX = 2  # PCC-below-threshold repairs before DISCARD

# Reference transition map (documentation + a test can assert handlers conform).
# Conditional edges (verdicts, counters) are decided INSIDE the handler; this
# lists every state a handler may legally return.
TRANSITIONS = {
    BEFORE_LOOP_DONE: [ROUTE],
    ROUTE: [SELECT],
    SELECT: [APPLY],
    APPLY: [VERIFY],
    VERIFY: [GATE_PCC, REPAIR_CODE, REVERT],
    REPAIR_CODE: [VERIFY],
    REPAIR_PCC: [VERIFY],
    GATE_PCC: [REMEASURE, REPAIR_PCC, REPAIR_CODE, REVERT],
    REMEASURE: [DECIDE, REVERT],
    DECIDE: [COMMIT, REVERT],
    COMMIT: [LOG],
    REVERT: [LOG],
    LOG: [CHECK_EXIT],
    CHECK_EXIT: [ROUTE, DONE, STOPPED],
}
