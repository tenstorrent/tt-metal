# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared KDA test fixtures."""

import os
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "perf: mark explicit KDA performance tests")
    config.addinivalue_line("markers", "long_running: mark opt-in KDA tests excluded from routine runs")


@pytest.fixture(scope="session")
def kimi_k3_checkpoint_dir() -> Path:
    """Return the explicitly selected pinned Kimi-K3 checkpoint subset."""
    value = os.getenv("KIMI_K3_CKPT")
    if value is None:
        pytest.skip("set KIMI_K3_CKPT to the pinned Kimi-K3 checkpoint subset")
    return Path(value)
