# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest config: put tests/, grpo_speedup/, and repo root on sys.path,
and set ttnn fabric once per session before any device opens."""

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
