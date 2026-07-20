"""Handler registry — the ONE shared file where members wire in their work.

To integrate: replace a `mocks.X` entry with your real module's handler, ONE
line at a time, and re-run `tests/test_engine.py`. The engine and every other
handler stay untouched.

  REAL today:  BEFORE_LOOP_DONE, ROUTE, SELECT, PLAN, APPLY, VERIFY,
               REPAIR_CODE, REPAIR_PCC (M1); GATE_PCC, REMEASURE, DECIDE,
               COMMIT, REVERT, LOG + CHECK_EXIT (M2 — COMMIT/REVERT git ops
               ported from apande/perf_automation_handlers)
"""

from __future__ import annotations

from .. import states
from . import apply as _apply
from . import commit as _commit
from . import revert as _revert
from . import log_exit, mocks, plan as _plan, route
from . import select as _select
from . import gate_pcc as _gate_pcc
from . import decide as _decide
from . import remeasure as _remeasure
from . import verify as _verify
from . import repair_code as _repair_code
from . import repair_pcc as _repair_pcc


def build_handlers() -> dict:
    return {
        states.BEFORE_LOOP_DONE: lambda ctx: states.ROUTE,
        # --- Member 1: decide & act ---
        states.ROUTE: route.route,  # REAL
        states.SELECT: _select.select,  # REAL
        states.PLAN: _plan.plan,  # REAL
        states.APPLY: _apply.apply,  # REAL
        states.VERIFY: _verify.verify,  # REAL
        states.REPAIR_CODE: _repair_code.repair_code,  # REAL
        states.REPAIR_PCC: _repair_pcc.repair_pcc,  # REAL
        # --- Member 2: evaluate & record ---
        states.GATE_PCC: _gate_pcc.gate_pcc,  # REAL
        states.REMEASURE: _remeasure.remeasure,  # REAL
        states.DECIDE: _decide.decide,  # REAL
        states.COMMIT: _commit.commit,  # REAL (apande/perf_automation_handlers)
        states.REVERT: _revert.revert,  # REAL (apande/perf_automation_handlers)
        states.LOG: log_exit.log,  # REAL
        states.CHECK_EXIT: log_exit.check_exit,  # REAL
    }
