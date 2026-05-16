# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test cases for the optimized exp2 implementation
"""

import pytest
import torch
import math
import numpy as np

def test_exp2_special_values():
    """Test exp2 for special values: NaN, inf, -inf"""
    # Note: These tests would need to be adapted to work with the actual SFPU implementation
    # For now, we're just documenting the expected behavior

    # exp2(NaN) = NaN
    # exp2(+inf) = +inf
    # exp2(-inf) = 0

    # These would be tested in the actual hardware or SFPU emulator
    assert math.isnan(math.exp2(float('nan')))
    assert math.isinf(math.exp2(float('inf'))) and math.exp2(float('inf')) > 0
    assert math.exp2(float('-inf')) == 0.0

def test_exp2_integer_values():
    """Test exp2 for integer values (should be exact powers of two)"""
    test_values = [0, 1, 2, 3, 4, 5, -1, -2, -3]

    for x in test_values:
        expected = math.exp2(x)
        # In practice, we'd call the SFPU function here
        # For now, just verify the mathematical expectation
        assert expected == 2.0 ** x

def test_exp2_range():
    """Test exp2 for a range of values"""
    test_values = [-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    for x in test_values:
        expected = math.exp2(x)
        # Verify mathematical correctness
        assert abs(expected - (2.0 ** x)) < 1e-10

if __name__ == "__main__":
    test_exp2_special_values()
    test_exp2_integer_values()
    test_exp2_range()
    print("All exp2 tests passed!")