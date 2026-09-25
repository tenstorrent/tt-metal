# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One import of the state directory, for the four modules that need it.

WHY A LOADER AT ALL. cc_optimize/tmpstate.py owns where durable temp state lives, and `agent` cannot
import it the ordinary way: the two are sibling packages, and these modules run both as
`agent.<mod>` and by path (spec_from_file_location), where a relative import has no parent package.
So each of them loaded tmpstate by file path -- and each wrote out the same six lines to do it:

    agent/before_loop.py        agent/perf_test_agent.py
    agent/device_recovery.py    agent/integrity.py

Identical in all four, down to the `_ilu_ts` alias. Four copies of a loader is four places to fix
when the path moves, and nothing to make them agree in the meantime.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path

_spec = _ilu.spec_from_file_location(
    "_tmpstate", str(Path(__file__).resolve().parent.parent / "cc_optimize" / "tmpstate.py")
)
_tmpstate = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tmpstate)

state_dir = _tmpstate.state_dir
state_path = getattr(_tmpstate, "state_path", None)
