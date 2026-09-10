# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest config: put this test dir, the examples root, and the repo root
on sys.path, and offer an opt-in fixture that sets ttnn fabric once per session
before any device opens."""

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


@pytest.fixture(scope="session")
def _set_fabric_2d():
    """Configure ttnn fabric to 2-rank.

    Request this fixture explicitly, e.g. through ``pytest.mark.usefixtures``,
    from a module that already self-skips unless ``OMPI_COMM_WORLD_SIZE == 2``.
    Both ranks must agree on the fabric config before either one opens a device.
    See description in ``_completer_utils.open_device`` for more details.
    """
    import ttnn

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
