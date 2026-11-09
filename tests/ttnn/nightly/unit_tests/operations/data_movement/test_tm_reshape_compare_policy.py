# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""test_tm_reshape.py again under TT_METAL_EXECUTION_POLICY_RESHAPE=compare: every reshape the suite makes also
runs on legacy and must match it bit for bit."""

import pytest

from ttnn._ttnn.operations.data_movement import set_reshape_execution_policy

from tests.ttnn.nightly.unit_tests.operations.data_movement.test_tm_reshape import *  # noqa: F401,F403


@pytest.fixture(autouse=True)
def reshape_compare_policy():
    set_reshape_execution_policy("compare")
    yield
    set_reshape_execution_policy(None)
