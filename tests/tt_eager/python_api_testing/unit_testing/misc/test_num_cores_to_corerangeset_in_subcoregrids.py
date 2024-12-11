# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest


@pytest.mark.parametrize(
    "start_core, num_cores, sub_core_grids, row_wise, expected_core_range_set",
    [
        # Test Case 1: Basic row-wise scenario with enough cores in sub_core_grids
        (
            ttnn.CoreCoord(1, 0),
            32,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
                ]
            ),
            True,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 0)),
                ]
            ),
        ),
        # Test Case 2: Basic Column-wise processing
        (
            ttnn.CoreCoord(1, 0),
            32,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
                ]
            ),
            False,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, 1)),
                ]
            ),
        ),
        # Test Case 3: row-wise scenario with small target cores and start offset
        (
            ttnn.CoreCoord(3, 2),
            8,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
                ]
            ),
            True,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(3, 2), ttnn.CoreCoord(3, 2)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 3), ttnn.CoreCoord(3, 4)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 5), ttnn.CoreCoord(1, 5)),
                ]
            ),
        ),
        # Test Case 4: col-wise scenario with small target cores and start offset
        (
            ttnn.CoreCoord(1, 8),
            8,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
                ]
            ),
            False,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 8), ttnn.CoreCoord(1, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 5)),
                ]
            ),
        ),
    ],
)
def test_numcores_to_corerangeset_in_subcoregrids(
    start_core, num_cores, sub_core_grids, row_wise, expected_core_range_set
):
    output_corerangeset = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        start_core, num_cores, sub_core_grids, row_wise=row_wise
    )
    assert output_corerangeset.to_json() == expected_core_range_set.to_json()
