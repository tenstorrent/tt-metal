# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared KDA test fixtures."""

import os
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "perf: mark explicit KDA performance tests")


@pytest.fixture
def isolated_program_cache(device):
    """Give one cache-sensitive test an enabled, empty program cache."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()
    device.enable_program_cache()


@pytest.fixture(scope="session")
def kimi_k3_checkpoint_dir() -> Path:
    """Return the explicitly selected pinned Kimi-K3 checkpoint subset."""
    value = os.getenv("KIMI_K3_CKPT")
    if value is None:
        pytest.skip("set KIMI_K3_CKPT to the pinned Kimi-K3 checkpoint subset")
    return Path(value)
