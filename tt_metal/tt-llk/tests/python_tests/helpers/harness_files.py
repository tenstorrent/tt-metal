# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Where the harness finds the tt-llk files its own Python reads.

In a checkout they sit at fixed places relative to this package. An installed
harness (the wheel of ``tests/harness_wheel/make_wheel.py``) has no tt-llk tree
around it and carries them, in tt-llk layout, under ``tt_llk_harness/llk_home``.
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
    """The tt-llk checkout the harness runs from, or the copy its wheel carries."""
    for root in (_CHECKOUT_ROOT, _INSTALLED_ROOT):
        if (root / _MARKER).is_file():
            return root
    raise RuntimeError(
        f"Cannot find the tt-llk files the harness reads ({_MARKER}): looked in "
        f"{_CHECKOUT_ROOT} (a tt-llk checkout) and {_INSTALLED_ROOT} (an installed "
        "harness). The installation is incomplete."
    )


def harness_is_installed() -> bool:
    """Whether the harness runs from its wheel, not from a tt-llk checkout."""
    return harness_files_root() == _INSTALLED_ROOT


def build_helpers_tree(llk_home: Path) -> Path:
    """The ``tests/helpers`` tree variants are built with: ``LLK_HOME``'s from a checkout (the SFPU report
    tool runs one harness on two trees), the wheel's own copy when installed."""
    root = harness_files_root() if harness_is_installed() else Path(llk_home)
    return root / "tests" / "helpers"
