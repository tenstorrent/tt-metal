# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest


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
