# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest


@pytest.mark.parametrize(
    "base_ranges, subtract_ranges, expected_result",
    [
        # Test Case 1: Basic subtraction - remove a portion of a range
        (
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 9))]),
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 9))]),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 9)),
                ]
            ),
        ),
        # Test Case 2: Subtraction with no overlap - should return the original
        (
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 9))]),
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 9))]),
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 9))]),
        ),
        # Test Case 3: Subtraction with complete overlap - should return empty set
        (
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 2))]),
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 2))]),
            ttnn.CoreRangeSet([]),
        ),
        # Test Case 4: Multiple ranges in both sets
        (
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(4, 9)),
                ]
            ),
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 5))]),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 6), ttnn.CoreCoord(1, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 6), ttnn.CoreCoord(3, 9)),
                ]
            ),
        ),
        # Test Case 5: Partial overlaps across multiple ranges
        (
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(6, 9)),
                ]
            ),
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 5), ttnn.CoreCoord(5, 5))]),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 4)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 6), ttnn.CoreCoord(2, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(6, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 4)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 6), ttnn.CoreCoord(5, 9)),
                ]
            ),
        ),
        # Test Case 6: Empty base set
        (
            ttnn.CoreRangeSet([]),
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 9))]),
            ttnn.CoreRangeSet([]),
        ),
        # Test Case 7: Empty subtract set
        (
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 9))]),
            ttnn.CoreRangeSet([]),
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 9))]),
        ),
    ],
)
def test_corerangeset_subtract(base_ranges, subtract_ranges, expected_result):
    """
    Test the CoreRangeSet::subtract method which removes the common ranges (intersection)
    from the current CoreRangeSet.

    Args:
        base_ranges: The original CoreRangeSet
        subtract_ranges: The CoreRangeSet to subtract from the original
        expected_result: The expected result after subtraction
    """
    result = base_ranges.subtract(subtract_ranges)
    assert result.to_json() == expected_result.to_json()
