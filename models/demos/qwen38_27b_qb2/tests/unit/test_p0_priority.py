# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A priority controller must never signal a replacement process."""

import pytest

from models.demos.qwen38_27b_qb2.demo.run_p0_priority import owned


@pytest.mark.parametrize(
    "props,expected",
    [
        ({"MainPID": "123", "InvocationID": "original", "ActiveState": "active"}, True),
        ({"MainPID": "124", "InvocationID": "original", "ActiveState": "active"}, False),
        ({"MainPID": "123", "InvocationID": "replacement", "ActiveState": "active"}, False),
        ({"MainPID": "123", "InvocationID": "original", "ActiveState": "deactivating"}, False),
        ({"MainPID": "0", "InvocationID": "", "ActiveState": "inactive"}, False),
        ({}, False),
    ],
)
def test_exact_controller_ownership(props, expected):
    assert owned(props, 123, "original") is expected
