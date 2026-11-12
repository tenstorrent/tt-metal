# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Who a warning blames.

Imports nothing from this package, so any unit can use it. Its own module
because :mod:`.fidelity` imports :mod:`.pack` and both need this.
"""

import inspect
import os

#: Frames inside here are plumbing. A warning raised in one is about something
#: the caller outside asked for.
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def caller_stacklevel() -> int:
    """The ``stacklevel`` that blames the first frame outside this package.

    Call it from the function that calls :func:`warnings.warn`::

        warnings.warn(..., UnmodelledHardwareWarning, stacklevel=caller_stacklevel())

    Counted rather than hard-coded, because a hard-coded level is a claim about
    how many wrappers sit between a unit and whoever asked. Any new layer makes
    it wrong, and a wrong one does not fail. It blames the wrong file.

    This package had six such constants and they did not agree: two warnings
    raised from the same function at the same depth used 2 and 3, so one of
    them had to be wrong. See `README.md`.
    """
    frame = inspect.currentframe()
    if frame is None:
        # No frame support. Blame our caller, inside the unit that raised.
        return 2
    # f_back is the frame that will call warnings.warn, which is level 1.
    frame, level = frame.f_back, 1
    while frame is not None:
        if not os.path.realpath(frame.f_code.co_filename).startswith(_PACKAGE_ROOT):
            return level
        frame, level = frame.f_back, level + 1
    return level


__all__ = ["caller_stacklevel"]
