# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest config: put this test dir, the grpo example dir, and the repo
root on sys.path, and set ttnn fabric once per session before any device opens."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
EXAMPLE_DIR = REPO_ROOT / "tt-train" / "sources" / "examples" / "grpo_remote_rollout"

for _p in (str(HERE), str(EXAMPLE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="session", autouse=True)
def _set_fabric_2d():
    """Configure ttnn fabric exactly once per pytest session."""
    import ttnn

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    yield
