"""Import a cc_optimize sibling module under EVERY load style this tool uses.

THE PROBLEM THIS OWNS. Modules in this package are loaded three different ways, and a plain import
works under only some of them:

  * as a package  -- `cc_optimize.run`, where `from .perf_mcp import x` resolves;
  * BY PATH       -- `spec_from_file_location("cc_optimize_run", ...)` in tt_hw_planner's
                     _load_cc_runner, giving NO package and often no sys.path entry, where both
                     `from .perf_mcp import x` and `from cc_optimize.perf_mcp import x` raise;
  * as a sibling top-level package -- `agent.probes` with perf_automation itself on sys.path, where
                     `from ..cc_optimize.x import y` raises "beyond top-level package".

Each of those failures is an ImportError raised deep inside a `try`, and this tool's callers
generally swallow import failures so a missing optional path never kills a run. That combination is
expensive: on 2026-08-29 the thermal gates were unreachable under the by-path load, every gate
silently did nothing, the board held 99-103C for an hour and two chips stopped answering. summary.py
had already hit the same wall -- its readers returned empty and the report printed "not measured"
over data that was sitting on disk.

So the resolution order lives HERE, once, instead of being rewritten per caller.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_CACHE: dict = {}


def load(name: str):
    """Return the cc_optimize sibling module `name`, or None if it cannot be reached.

    Never raises: a caller that cannot reach an optional sibling must be able to carry on, and one
    that cannot carry on can raise its own error with its own message.
    """
    if name in _CACHE:
        return _CACHE[name] or None
    mod = _resolve(name)
    _CACHE[name] = mod or False
    return mod


def _resolve(name: str):
    package = (__package__ or "").strip()
    for candidate in ([package + "." + name] if package else []) + ["cc_optimize." + name, name]:
        try:
            return importlib.import_module(candidate)
        except ImportError:
            continue
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        return importlib.import_module("cc_optimize." + name)
    except ImportError:
        pass
    try:
        import importlib.util as _ilu

        path = Path(__file__).resolve().parent / (name + ".py")
        alias = "cc_sibling_" + name
        spec = _ilu.spec_from_file_location(alias, str(path))
        mod = _ilu.module_from_spec(spec)
        sys.modules.setdefault(alias, mod)
        spec.loader.exec_module(mod)
        return mod
    except Exception:  # noqa: BLE001 -- exhausted every route; the caller decides what that means
        return None
