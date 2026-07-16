# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(autouse=True)
def _enable_parallel_sequential(monkeypatch):
    """Opt this suite in to Sequential/Parallel fusion.

    Sequential/Parallel are gated behind ``TT_METAL_ENABLE_PARALLEL_SEQUENTIAL`` until
    ProgramSpec is exposed to Python (see fusion.py). This autouse fixture enables
    them only for the duration of each test in this directory and reverts after,
    so the guard stays active for every other test sharing the same process.
    """
    monkeypatch.setenv("TT_METAL_ENABLE_PARALLEL_SEQUENTIAL", "1")
