# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""ttpoly — gold-standard math-approximation library core.

The bit-exact evaluator (``ttpoly.precision.eval.eval_segments``) plus the single
Goldberg ULP definition (``ttpoly.spec.units``), routed through
``ttpoly.stages.s40_eval``, are now the **canonical and only** accuracy path. The
legacy raw-ULP / IEEE-default evaluation has been removed from the hot metric
paths (csv2error.py and bf16_grid.py).

``USE_NEW_EVAL`` therefore defaults **ON**. The toggle is retained only as a
bring-up/test escape hatch; production code no longer branches on it. Override
via the environment variable ``TTPOLY_USE_NEW_EVAL=0`` or programmatically
through ``ttpoly.set_use_new_eval(False)``.
"""

import os

__all__ = ["USE_NEW_EVAL", "use_new_eval", "set_use_new_eval"]


def _env_flag(name, default):
    """Read a boolean env flag, defaulting to ``default`` when unset/blank."""
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    return raw in ("1", "true", "yes", "on")


# Default ON: the bit-exact evaluator + Goldberg ULP are the only metric path.
# The legacy raw-ULP / IEEE-default route has been removed from the hot paths,
# so this flag is now effectively always-True in production; it stays as a
# bring-up/test escape hatch (set TTPOLY_USE_NEW_EVAL=0 to flip).
USE_NEW_EVAL = _env_flag("TTPOLY_USE_NEW_EVAL", True)


def use_new_eval():
    """Return whether the new bit-exact evaluator is enabled."""
    return USE_NEW_EVAL


def set_use_new_eval(value):
    """Programmatically toggle the new evaluator (test/bring-up helper)."""
    global USE_NEW_EVAL
    USE_NEW_EVAL = bool(value)
    return USE_NEW_EVAL
