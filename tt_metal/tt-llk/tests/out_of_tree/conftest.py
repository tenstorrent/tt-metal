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
    harness_python_root,
    helpers_tree,
    llk_home,
    shadowed_include_dirs,
)

os.environ.setdefault("LLK_HOME", str(llk_home()))

_harness_python = harness_python_root()
if not (_harness_python / "tt_llk_harness").is_dir():
    raise RuntimeError(
        f"tt-llk harness not found at {_harness_python}. Set LLK_HOME to the "
        "tt-llk root (contains tests/ and tt_llk_*), or LLK_HARNESS_SITE to the "
        "directory an installed harness lives in."
    )
# An installed harness sits in site-packages, behind the suite's own directories; a checkout goes first.
if os.environ.get("LLK_HARNESS_SITE"):
    sys.path.append(str(_harness_python))
else:
    sys.path.insert(0, str(_harness_python))

pytest_plugins = ["tt_llk_harness.plugin"]

import tt_llk_harness  # noqa: E402
from tt_llk_harness import TestConfig  # noqa: E402

tt_llk_harness.require_version(1, 1)

# Registered low-priority first: add_include_dirs prepends, so include_high
# must end up winning. The driver #errors if that stops being true.
_low, _high = shadowed_include_dirs()
TestConfig.add_include_dirs(_low)
TestConfig.add_include_dirs(_high)
TestConfig.add_helpers_tree(helpers_tree())
