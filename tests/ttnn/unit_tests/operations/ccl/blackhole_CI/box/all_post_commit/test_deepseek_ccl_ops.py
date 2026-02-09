# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Deepseek CCL Operations Tests for APC

This module includes the deepseek CCL operation tests as part of the CCL APC test suite.
It imports and re-exports tests from:
- models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_broadcast.py
- models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py
- models/demos/deepseek_v3_b1/tests/unit_tests/test_reduce_to_one_b1.py

To run all CCL APC tests including these:
    pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/all_post_commit

To run only these deepseek tests:
    pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/all_post_commit/test_deepseek_ccl_ops.py
"""

# Import tests from deepseek_v3_b1 test modules
# pytest will automatically discover and run these imported test functions

from models.demos.deepseek_v3_b1.tests.unit_tests.test_reduce_to_one_b1 import (
    test_reduce_to_one_2d,
)

from models.demos.deepseek_v3_b1.tests.unit_tests.test_ccl_broadcast import (
    test_ccl_broadcast_dual_axis,
)

from models.demos.deepseek_v3_b1.tests.unit_tests.test_ccl_all_reduce import (
    test_ccl_all_reduce,
)

# Re-export for pytest discovery
__all__ = [
    "test_reduce_to_one_2d",
    "test_ccl_broadcast_dual_axis",
    "test_ccl_all_reduce",
]
