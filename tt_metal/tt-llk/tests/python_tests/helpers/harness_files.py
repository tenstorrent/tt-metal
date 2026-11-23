# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Where the harness finds the tt-llk files its own Python reads.

A few modules read a file that lives outside ``helpers``: ``counters.py`` parses
``tests/helpers/include/counters.h``, and the perf split reads the cost tables
beside ``tests/``. In a checkout those sit at fixed places relative to this
package. In an installed harness (the wheel ``tests/harness_wheel/make_wheel.py``
builds) there is no tt-llk tree around the package, so the wheel carries them,
in the same tt-llk layout, under ``tt_llk_harness/llk_home``.

``harness_files_root()`` returns whichever of the two applies, so a caller
writes one path for both.
"""

from __future__ import annotations

from pathlib import Path

# A file every harness has; its presence is what tells a root apart from a directory that merely exists.
_MARKER = Path("tests") / "helpers" / "include" / "counters.h"

_PYTHON_ROOT = Path(__file__).resolve().parents[1]
# helpers -> python_tests -> tests -> <tt-llk root>
_CHECKOUT_ROOT = Path(__file__).resolve().parents[3]
# tt_llk_harness is installed beside helpers.
_INSTALLED_ROOT = _PYTHON_ROOT / "tt_llk_harness" / "llk_home"


def harness_files_root() -> Path:
    """Root, in tt-llk layout, of the tt-llk files the harness's Python reads.

    The tt-llk checkout when the harness runs from one, the copy the wheel
    carries when it is installed.
    """
    for root in (_CHECKOUT_ROOT, _INSTALLED_ROOT):
        if (root / _MARKER).is_file():
            return root
    raise RuntimeError(
        f"Cannot find the tt-llk files the harness reads ({_MARKER}): looked in "
        f"{_CHECKOUT_ROOT} (a tt-llk checkout) and {_INSTALLED_ROOT} (an installed "
        "harness). The installation is incomplete."
    )
