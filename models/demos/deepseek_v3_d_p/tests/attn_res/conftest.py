# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared AttnRes test fixtures."""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def kimi_k3_checkpoint_dir() -> Path | None:
    """The pinned Kimi K3 checkpoint subset, or `None` for random weights.

    `None` rather than a skip: the gates are meant to hold on weights the model never saw,
    so the random arm is the one CI runs and the checkpoint is the opt-in. Point
    `KIMI_K3_CKPT` at a directory an index can be read from — `fetch_query_weights.py`
    writes one holding only the query weights.
    """
    value = os.getenv("KIMI_K3_CKPT")
    return Path(value) if value else None
