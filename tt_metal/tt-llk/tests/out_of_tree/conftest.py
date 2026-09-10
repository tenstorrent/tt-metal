# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Rootdir conftest for the out-of-tree contract fixture.

This file is the executable specification of the contract documented in
``docs/tests/getting_started.md`` §9. It does exactly what an external consumer
does — nothing more, and nothing that reaches into harness internals — so that
a harness change which breaks consumers breaks this suite first.

Run it as its own rootdir (``pytest`` from this directory); nested
``pytest_plugins`` is rejected by pytest, which is itself part of the contract.
"""

from __future__ import annotations

import os
import sys

from fixture_paths import (
    helpers_tree,
    llk_home,
    llk_python_tests,
    shadowed_include_dirs,
)

os.environ.setdefault("LLK_HOME", str(llk_home()))

_llk_python = llk_python_tests()
if not _llk_python.is_dir():
    raise RuntimeError(
        f"tt-llk python_tests not found at {_llk_python}. "
        "Set LLK_HOME to the tt-llk root (contains tests/ and tt_llk_*)."
    )
sys.path.insert(0, str(_llk_python))

pytest_plugins = ["tt_llk_harness.plugin"]

import tt_llk_harness  # noqa: E402
from tt_llk_harness import TestConfig  # noqa: E402

tt_llk_harness.require_version(1, 0)

# Registered low-priority first: add_include_dirs prepends, so include_high
# must end up winning. The driver #errors if that stops being true.
_low, _high = shadowed_include_dirs()
TestConfig.add_include_dirs(_low)
TestConfig.add_include_dirs(_high)
TestConfig.add_helpers_tree(helpers_tree())
