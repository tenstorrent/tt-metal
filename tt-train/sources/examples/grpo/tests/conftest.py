# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration for the grpo_speedup update-method tests.

Two responsibilities:

1. Put ``HERE`` (``tests/``), ``GRPO_SPEEDUP`` (``grpo_speedup/``), and
   ``REPO_ROOT`` (``tt-metal/``) on ``sys.path`` so the tests can import
   ``utils.llama_completer_ttt``, ``models.tt_transformers.*``, and the
   sibling ``_completer_utils`` helpers regardless of where pytest was
   invoked from.

2. Configure ttnn's fabric exactly once per pytest session, before any
   test fixture opens a device. We do this in an ``autouse=True``
   session-scoped fixture so individual tests don't have to remember to
   declare the dependency.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

HERE = Path(__file__).resolve().parent
GRPO_SPEEDUP = HERE.parent
REPO_ROOT = HERE.parents[4]

for _p in (str(HERE), str(GRPO_SPEEDUP), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="session", autouse=True)
def _set_fabric_2d():
    """Configure ttnn fabric exactly once per pytest session."""
    import ttnn

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    yield
