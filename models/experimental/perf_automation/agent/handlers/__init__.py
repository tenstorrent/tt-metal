"""The one surviving handler: REMEASURE.

This package was the state-handler registry for the FSM engine (`agent/loop.py`), which was retired
when cc became the only engine. Every other handler -- route, select, plan, apply, verify,
repair_code, repair_pcc, gate_pcc, decide, commit, revert, log_exit and the mocks -- was reachable
only from `build_handlers()`, which only that engine called, so they went with it.

`remeasure` outlived the engine because `cc_optimize/perf_mcp.py` imports it directly for the
re-measure-and-compare step, independent of any state machine.
"""
from __future__ import annotations

from . import remeasure

__all__ = ["remeasure"]
