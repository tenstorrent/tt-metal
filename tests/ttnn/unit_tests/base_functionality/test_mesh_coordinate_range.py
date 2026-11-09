# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn


def test_mesh_coordinate_range_rejects_zero_mesh_dimension(expect_error):
    # shape_back builds the inclusive end as extent - 1 in uint32, so an empty dimension wrapped to
    # 4294967295 and the range claimed to span four billion coordinates.
    with expect_error(RuntimeError, "greater than zero"):
        ttnn.MeshCoordinateRange(ttnn.MeshShape(0, 2))


def test_mesh_coordinate_range_covers_every_coordinate():
    # Control: a non-empty mesh shape still yields every coordinate.
    coords = list(ttnn.MeshCoordinateRange(ttnn.MeshShape(2, 3)))
    assert len(coords) == 6
