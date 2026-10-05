# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

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


@pytest.fixture
def isolated_program_cache(device):
    # Setup: give the test an enabled, empty program cache.
    device.disable_and_clear_program_cache()
    device.enable_program_cache()

    # Yield control to pytest to execute the test.
    yield

    # Teardown: remove programs created by the test and restore the enabled, empty state.
    device.disable_and_clear_program_cache()
    device.enable_program_cache()


@pytest.fixture
def zero_actual_start(device):
    from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

    return make_actual_start(device, 0)
