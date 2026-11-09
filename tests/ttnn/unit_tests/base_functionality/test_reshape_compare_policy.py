# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""test_reshape.py again under TT_METAL_EXECUTION_POLICY_RESHAPE=compare: every reshape the suite makes also runs
on legacy and must match it bit for bit (cases legacy rejects are logged and checked against torch only)."""

import pytest

from ttnn._ttnn.operations.data_movement import set_reshape_execution_policy

from tests.ttnn.unit_tests.base_functionality.test_reshape import *  # noqa: F401,F403


@pytest.fixture(autouse=True)
def reshape_compare_policy():
    set_reshape_execution_policy("compare")
    yield
    set_reshape_execution_policy(None)
