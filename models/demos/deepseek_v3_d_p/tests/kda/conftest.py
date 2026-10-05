# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared KDA test fixtures."""

import os
from pathlib import Path

import pytest

_KDA_TESTS = Path(__file__).resolve().parent


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    """Run each KDA test's setup, body and teardown, including its CPU oracles, on one Torch thread.

    Pytest calls this hook for every item in the session, not only those under this conftest, so it acts
    only on items in this directory.
    """
    if _KDA_TESTS not in item.path.resolve().parents:
        return (yield)
    from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import single_threaded_torch

    with single_threaded_torch():
        return (yield)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "perf: mark explicit KDA performance tests")
    config.addinivalue_line("markers", "extended_actual_start: exhaustive aligned-start local acceptance")


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


@pytest.fixture(scope="session")
def glm_5_3_flash_checkpoint_dir() -> Path:
    """Return the explicitly selected pinned GLM-5.3-Flash checkpoint subset (index + layer-0 shard)."""
    value = os.getenv("GLM_5_3_FLASH_CKPT")
    if value is None:
        pytest.skip("set GLM_5_3_FLASH_CKPT to the pinned GLM-5.3-Flash checkpoint subset")
    return Path(value)
