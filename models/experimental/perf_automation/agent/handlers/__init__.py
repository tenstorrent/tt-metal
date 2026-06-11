"""Handler registry — the ONE shared file where members wire in their work.

To integrate: replace a `mocks.X` entry with your real module's handler, ONE
line at a time, and re-run `tests/test_engine.py`. The engine and every other
handler stay untouched.

  REAL today:  BEFORE_LOOP_DONE, ROUTE (M1), LOG + CHECK_EXIT (M2)
  MOCK (TODO): SELECT, APPLY, VERIFY, REPAIR_CODE, REPAIR_PCC  (M1)
               GATE_PCC, REMEASURE, DECIDE, COMMIT, REVERT     (M2)
"""

from __future__ import annotations

from .. import states
from . import log_exit, mocks, route


def build_handlers() -> dict:
    return {
        states.BEFORE_LOOP_DONE: lambda ctx: states.ROUTE,
        # --- Member 1: decide & act ---
        states.ROUTE: route.route,  # REAL
        states.SELECT: mocks.select,
        states.APPLY: mocks.apply,
        states.VERIFY: mocks.verify,
        states.REPAIR_CODE: mocks.repair_code,
        states.REPAIR_PCC: mocks.repair_pcc,
        # --- Member 2: evaluate & record ---
        states.GATE_PCC: mocks.gate_pcc,
        states.REMEASURE: mocks.remeasure,
        states.DECIDE: mocks.decide,
        states.COMMIT: mocks.commit,
        states.REVERT: mocks.revert,
        states.LOG: log_exit.log,  # REAL
        states.CHECK_EXIT: log_exit.check_exit,  # REAL
    }
