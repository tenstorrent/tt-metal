# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest config: put this test dir, the examples root, and the repo
root on sys.path, and set ttnn fabric once per session before any device opens."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
EXAMPLES_DIR = REPO_ROOT / "tt-train" / "sources" / "examples"

for _p in (str(HERE), str(EXAMPLES_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


_WORLD_SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))


@pytest.fixture(scope="session", autouse=True)
def _set_fabric_2d():
    """Configure ttnn fabric exactly once per pytest session, for 2-rank runs only."""
    if _WORLD_SIZE != 2:
        return

    import ttnn

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
