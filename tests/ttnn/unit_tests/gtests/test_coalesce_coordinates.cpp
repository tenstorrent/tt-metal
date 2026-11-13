// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// mesh_device_operation_utils::coalesce_coordinates, which mesh workload factories use to emit one program per
// rectangle of coordinates that run the same program (paged_cache's mesh_coords selections).

#include <gtest/gtest.h>

#include <set>
#include <vector>

#include "ttnn/mesh_device_operation_utils.hpp"

namespace {

using ttnn::MeshCoordinate;
using ttnn::MeshCoordinateRange;
using ttnn::device_operation::mesh_device_operation_utils::coalesce_coordinates;

// Every coordinate of `range` is in exactly one returned rectangle iff it is selected.
void expect_exact_cover(
    const MeshCoordinateRange& range,
    const std::set<MeshCoordinate>& selected,
    const std::vector<MeshCoordinateRange>& rectangles) {
    for (const auto& coord : range) {
        int covering = 0;
        for (const auto& rectangle : rectangles) {
            covering += rectangle.contains(coord) ? 1 : 0;
        }
        EXPECT_EQ(covering, selected.contains(coord) ? 1 : 0) << coord;
    }
    for (const auto& rectangle : rectangles) {
        for (const auto& coord : rectangle) {
            EXPECT_TRUE(range.contains(coord)) << coord;
        }
    }
}

std::vector<MeshCoordinateRange> coalesce(const MeshCoordinateRange& range, const std::set<MeshCoordinate>& selected) {
    return coalesce_coordinates(range, [&](const MeshCoordinate& coord) { return selected.contains(coord); });
}

TEST(CoalesceCoordinates, AllSelectedIsOneRange) {
    const MeshCoordinateRange range(ttnn::MeshShape(8, 4));
    std::set<MeshCoordinate> all(range.begin(), range.end());
    const auto rectangles = coalesce(range, all);
    ASSERT_EQ(rectangles.size(), 1u);
    EXPECT_EQ(rectangles.front(), range);
}

TEST(CoalesceCoordinates, NoneSelectedIsEmpty) {
    EXPECT_TRUE(coalesce(MeshCoordinateRange(ttnn::MeshShape(2, 4)), {}).empty());
}

TEST(CoalesceCoordinates, RowAndBlock) {
    const MeshCoordinateRange range(ttnn::MeshShape(8, 4));
    std::set<MeshCoordinate> row;
    for (uint32_t col = 0; col < 4; col++) {
        row.insert(MeshCoordinate(2, col));
    }
    auto rectangles = coalesce(range, row);
    ASSERT_EQ(rectangles.size(), 1u);
    EXPECT_EQ(rectangles.front(), MeshCoordinateRange(MeshCoordinate(2, 0), MeshCoordinate(2, 3)));

    std::set<MeshCoordinate> block;
    for (uint32_t r = 1; r <= 4; r++) {
        for (uint32_t c = 1; c <= 2; c++) {
            block.insert(MeshCoordinate(r, c));
        }
    }
    rectangles = coalesce(range, block);
    ASSERT_EQ(rectangles.size(), 1u);
    EXPECT_EQ(rectangles.front(), MeshCoordinateRange(MeshCoordinate(1, 1), MeshCoordinate(4, 2)));
}

TEST(CoalesceCoordinates, IrregularSelectionsAreCoveredExactly) {
    const MeshCoordinateRange range(ttnn::MeshShape(8, 4));
    std::set<MeshCoordinate> holes(range.begin(), range.end());
    holes.erase(MeshCoordinate(0, 1));
    holes.erase(MeshCoordinate(7, 0));
    holes.erase(MeshCoordinate(3, 2));
    const auto hole_rectangles = coalesce(range, holes);
    expect_exact_cover(range, holes, hole_rectangles);
    EXPECT_LT(hole_rectangles.size(), holes.size());

    std::set<MeshCoordinate> checker;
    for (const auto& coord : range) {
        if ((coord[0] + coord[1]) % 2 == 0) {
            checker.insert(coord);
        }
    }
    expect_exact_cover(range, checker, coalesce(range, checker));
}

TEST(CoalesceCoordinates, SubrangeNotAtOrigin) {
    const MeshCoordinateRange range(MeshCoordinate(2, 1), MeshCoordinate(5, 3));
    std::set<MeshCoordinate> selected = {
        MeshCoordinate(2, 1), MeshCoordinate(3, 1), MeshCoordinate(5, 3), MeshCoordinate(0, 0) /* outside */};
    const auto rectangles = coalesce(range, selected);
    selected.erase(MeshCoordinate(0, 0));
    expect_exact_cover(range, selected, rectangles);
}

}  // namespace
