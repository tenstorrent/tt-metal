# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Path resolution for the out-of-tree contract fixture.

This module is deliberately the only place the fixture knows about its own
layout. It is the in-repo stand-in for a consumer's ``harness.py``: an external
suite writes the same handful of helpers against its own tree.
"""

from __future__ import annotations

import os
from pathlib import Path

FIXTURE_ROOT = Path(__file__).resolve().parent
# tests/out_of_tree -> tests -> <tt-llk root>
LLK_TESTS = FIXTURE_ROOT.parent
DEFAULT_LLK_HOME = LLK_TESTS.parent


def llk_home() -> Path:
    """tt-llk root. ``LLK_HOME`` wins so the fixture can point at a checkout."""
    return Path(os.environ.get("LLK_HOME", DEFAULT_LLK_HOME)).resolve()


def llk_python_tests() -> Path:
    return llk_home() / "tests" / "python_tests"


def cpp_source(name: str) -> str:
    """Absolute driver path, so it is never resolved from ``tests/sources/``."""
    path = FIXTURE_ROOT / "sources" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def helpers_tree() -> Path:
    """``tests/helpers``-layout tree: ``include/`` + ``src/``."""
    return FIXTURE_ROOT / "helpers"


def shadowed_include_dirs() -> tuple[Path, Path]:
    """Two dirs providing the same ``oot_probe.h``, low priority first.

    Registered in this order, ``add_include_dirs`` must leave the *second* one
    winning, because it prepends. The driver ``#error``s if it does not.
    """
    return FIXTURE_ROOT / "include_low", FIXTURE_ROOT / "include_high"
