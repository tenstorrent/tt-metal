// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <gtest/gtest.h>
#include <stdint.h>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/shape_base.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal::distributed {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Optional;
using ::testing::UnorderedElementsAre;

TEST(MeshShapeTest, Construction) {
    MeshShape shape_1d(3);
    EXPECT_EQ(shape_1d.dims(), 1);
    EXPECT_EQ(shape_1d[0], 3);
    EXPECT_EQ(shape_1d.mesh_size(), 3);

    MeshShape shape_2d(3, 4);
    EXPECT_EQ(shape_2d.dims(), 2);
    EXPECT_EQ(shape_2d[0], 3);
    EXPECT_EQ(shape_2d[1], 4);
    EXPECT_EQ(shape_2d.mesh_size(), 12);

    MeshShape shape_3d(2, 3, 4);
    EXPECT_EQ(shape_3d.dims(), 3);
    EXPECT_EQ(shape_3d[0], 2);
    EXPECT_EQ(shape_3d[1], 3);
    EXPECT_EQ(shape_3d[2], 4);
    EXPECT_EQ(shape_3d.mesh_size(), 24);

    MeshShape shape_5d({2, 3, 4, 5, 6});
    EXPECT_EQ(shape_5d.dims(), 5);
    EXPECT_EQ(shape_5d[0], 2);
    EXPECT_EQ(shape_5d[1], 3);
    EXPECT_EQ(shape_5d[2], 4);
    EXPECT_EQ(shape_5d[3], 5);
    EXPECT_EQ(shape_5d[4], 6);
    EXPECT_EQ(shape_5d.mesh_size(), 720);
}

TEST(MeshShapeTest, ZeroShape) {
    // MeshShapes are not allowed to be empty.
    EXPECT_ANY_THROW(MeshShape shape({}));
}

TEST(MeshShapeTest, Strides) {
    MeshShape shape(2, 3, 4);
    EXPECT_EQ(shape.get_stride(0), 12);  // 3 * 4
    EXPECT_EQ(shape.get_stride(1), 4);   // 4
    EXPECT_EQ(shape.get_stride(2), 1);   // 1
}

TEST(MeshShapeTest, Equality) {
    MeshShape shape(2, 3);

    EXPECT_EQ(shape, MeshShape(2, 3));
    EXPECT_NE(shape, MeshShape(3, 2));
    EXPECT_NE(shape, MeshShape(1, 2, 3));
}

TEST(MeshShapeTest, LinearTopology) {
    EXPECT_TRUE(MeshShape(1).is_line_topology());
    EXPECT_TRUE(MeshShape(3).is_line_topology());
    EXPECT_TRUE(MeshShape(1, 1).is_line_topology());
    EXPECT_TRUE(MeshShape(1, 3).is_line_topology());
    EXPECT_TRUE(MeshShape(3, 1).is_line_topology());
    EXPECT_FALSE(MeshShape(3, 3).is_line_topology());
    EXPECT_TRUE(MeshShape(1, 1, 1).is_line_topology());
    EXPECT_TRUE(MeshShape(1, 1, 3).is_line_topology());
    EXPECT_TRUE(MeshShape(1, 3, 1).is_line_topology());
    EXPECT_TRUE(MeshShape(3, 1, 1).is_line_topology());
    EXPECT_FALSE(MeshShape(1, 3, 3).is_line_topology());
    EXPECT_FALSE(MeshShape(3, 1, 3).is_line_topology());
    EXPECT_FALSE(MeshShape(3, 3, 3).is_line_topology());
}

TEST(MeshCoordinateTest, Construction) {
    MeshCoordinate coord_1d(1);
    EXPECT_EQ(coord_1d.dims(), 1);
    EXPECT_THAT(coord_1d.coords(), ElementsAre(1));
    EXPECT_EQ(coord_1d[0], 1);

    MeshCoordinate coord_2d(1, 2);
    EXPECT_EQ(coord_2d.dims(), 2);
    EXPECT_THAT(coord_2d.coords(), ElementsAre(1, 2));
    EXPECT_EQ(coord_2d[0], 1);
    EXPECT_EQ(coord_2d[1], 2);

    MeshCoordinate coord_3d(1, 2, 3);
    EXPECT_EQ(coord_3d.dims(), 3);
    EXPECT_THAT(coord_3d.coords(), ElementsAre(1, 2, 3));
    EXPECT_EQ(coord_3d[0], 1);
    EXPECT_EQ(coord_3d[1], 2);
    EXPECT_EQ(coord_3d[2], 3);

    std::vector<uint32_t> values = {1, 2, 3, 4, 5};
    MeshCoordinate coord_span(values);
    EXPECT_EQ(coord_span.dims(), 5);
    EXPECT_THAT(coord_span.coords(), ElementsAre(1, 2, 3, 4, 5));
    EXPECT_EQ(coord_span[0], 1);
    EXPECT_EQ(coord_span[1], 2);
    EXPECT_EQ(coord_span[2], 3);
    EXPECT_EQ(coord_span[3], 4);
    EXPECT_EQ(coord_span[4], 5);
}

TEST(MeshCoordinateTest, Equality) {
    MeshCoordinate coord1(1, 2);

    EXPECT_EQ(coord1, MeshCoordinate(1, 2));
    EXPECT_NE(coord1, MeshCoordinate(2, 1));
    EXPECT_NE(coord1, MeshCoordinate(1, 2, 1));
}

TEST(MeshCoordinateTest, ComparisonMismatchedDimensions) {
    MeshCoordinate coord1(1);
    EXPECT_ANY_THROW(void(coord1 < MeshCoordinate(1, 2)));
}

TEST(MeshCoordinateTest, Comparison) {
    MeshCoordinate coord1(1);
    MeshCoordinate coord2(2);
    EXPECT_TRUE(coord1 < coord2);
    EXPECT_FALSE(coord2 < coord1);

    MeshCoordinate coord3(2, 0, 3);
    MeshCoordinate coord4(2, 1, 3);
    EXPECT_TRUE(coord3 < coord4);
    EXPECT_FALSE(coord4 < coord3);
}

TEST(MeshCoordinateTest, UnorderedSet) {
    std::unordered_set<MeshCoordinate> set;
    set.insert(MeshCoordinate(0, 0, 0));
    set.insert(MeshCoordinate(0, 0, 1));
    set.insert(MeshCoordinate(0, 0, 2));

    EXPECT_FALSE(set.insert(MeshCoordinate(0, 0, 2)).second);
    EXPECT_THAT(
        set,
        UnorderedElementsAre(
            MeshCoordinate(0, 0, 0),  //
            MeshCoordinate(0, 0, 1),
            MeshCoordinate(0, 0, 2)));
}

TEST(MeshCoordinateTest, ZeroCoordinate) {
    EXPECT_EQ(MeshCoordinate::zero_coordinate(1), MeshCoordinate(0));
    EXPECT_EQ(MeshCoordinate::zero_coordinate(2), MeshCoordinate(0, 0));
    EXPECT_EQ(MeshCoordinate::zero_coordinate(3), MeshCoordinate(0, 0, 0));
}

TEST(MeshCoordinateTest, NegativeIndexing) {
    MeshCoordinate coord(10, 20, 30);

    // Positive indexing
    EXPECT_EQ(coord[0], 10);
    EXPECT_EQ(coord[1], 20);
    EXPECT_EQ(coord[2], 30);

    // Negative indexing
    EXPECT_EQ(coord[-1], 30);  // Last element
    EXPECT_EQ(coord[-2], 20);  // Second to last
    EXPECT_EQ(coord[-3], 10);  // Third to last (first element)

    // Out of bounds negative indexing
    EXPECT_ANY_THROW(coord[-4]);
}

TEST(MeshCoordinateTest, MutableIndexAccess) {
    MeshCoordinate coord(10, 20, 30);

    // Modify through positive index
    coord[0] = 15;
    EXPECT_EQ(coord[0], 15);

    // Modify through negative index
    coord[-1] = 35;
    EXPECT_EQ(coord[2], 35);
    EXPECT_EQ(coord[-1], 35);

    // Verify final state
    EXPECT_THAT(coord.coords(), ElementsAre(15, 20, 35));
}

TEST(MeshCoordinateRangeTest, FromShape) {
    MeshShape shape(2, 3);
    MeshCoordinateRange range(shape);

    std::vector<MeshCoordinate> coords;
    for (const auto& coord : range) {
        coords.push_back(coord);
    }

    EXPECT_THAT(
        coords,
        ElementsAre(
            MeshCoordinate(0, 0),
            MeshCoordinate(0, 1),
            MeshCoordinate(0, 2),
            MeshCoordinate(1, 0),
            MeshCoordinate(1, 1),
            MeshCoordinate(1, 2)));
}

TEST(MeshCoordinateRangeTest, Subrange) {
    MeshCoordinate start(1, 1, 1);
    MeshCoordinate end(2, 1, 4);
    MeshCoordinateRange range(start, end);

    std::vector<MeshCoordinate> coords;
    for (const auto& coord : range) {
        coords.push_back(coord);
    }

    EXPECT_THAT(
        coords,
        ElementsAre(
            MeshCoordinate(1, 1, 1),
            MeshCoordinate(1, 1, 2),
            MeshCoordinate(1, 1, 3),
            MeshCoordinate(1, 1, 4),
            MeshCoordinate(2, 1, 1),
            MeshCoordinate(2, 1, 2),
            MeshCoordinate(2, 1, 3),
            MeshCoordinate(2, 1, 4)));
}

TEST(MeshCoordinateRangeTest, SubrangeOneElement) {
    MeshCoordinate start(1, 1, 1);
    MeshCoordinate end(1, 1, 1);
    MeshCoordinateRange range(start, end);

    std::vector<MeshCoordinate> coords;
    for (const auto& coord : range) {
        coords.push_back(coord);
    }

    EXPECT_THAT(coords, ElementsAre(MeshCoordinate(1, 1, 1)));
}

TEST(MeshCoordinateRangeTest, ContainsInvalidDimensions) {
    MeshCoordinateRange range(MeshCoordinate(1, 1, 3), MeshCoordinate(1, 1, 3));
    EXPECT_ANY_THROW(range.contains(MeshCoordinate(1, 1)));
    EXPECT_ANY_THROW(range.contains(MeshCoordinateRange(MeshCoordinate(1, 1), MeshCoordinate(1, 1))));
}

TEST(MeshCoordinateRangeTest, Contains) {
    MeshCoordinateRange range(MeshCoordinate(1, 1, 3), MeshCoordinate(1, 1, 3));
    EXPECT_TRUE(range.contains(MeshCoordinate(1, 1, 3)));

    range = MeshCoordinateRange(MeshCoordinate(0, 2), MeshCoordinate(1, 2));
    EXPECT_TRUE(range.contains(MeshCoordinate(0, 2)));
    EXPECT_TRUE(range.contains(MeshCoordinate(1, 2)));
    EXPECT_FALSE(range.contains(MeshCoordinate(0, 1)));
    EXPECT_FALSE(range.contains(MeshCoordinate(2, 1)));
    EXPECT_FALSE(range.contains(MeshCoordinate(2, 2)));
}

TEST(MeshCoordinateRangeTest, ContainsRange) {
    MeshCoordinateRange range(MeshCoordinate(1, 1, 3), MeshCoordinate(1, 1, 3));
    EXPECT_TRUE(range.contains(range));

    EXPECT_FALSE(range.contains(MeshCoordinateRange(MeshCoordinate(1, 1, 2), MeshCoordinate(1, 1, 3))));
    EXPECT_FALSE(range.contains(MeshCoordinateRange(MeshCoordinate(1, 1, 3), MeshCoordinate(1, 1, 4))));

    range = MeshCoordinateRange(MeshCoordinate(1, 1), MeshCoordinate(2, 2));
    EXPECT_FALSE(range.contains(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0))));
    EXPECT_FALSE(range.contains(MeshCoordinateRange(MeshCoordinate(0, 3), MeshCoordinate(0, 3))));
    EXPECT_FALSE(range.contains(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 1))));
    EXPECT_FALSE(range.contains(MeshCoordinateRange(MeshCoordinate(0, 2), MeshCoordinate(1, 2))));
    EXPECT_TRUE(range.contains(MeshCoordinateRange(MeshCoordinate(1, 1), MeshCoordinate(1, 2))));
}

TEST(MeshCoordinateRangeTest, Intersection) {
    MeshCoordinateRange range(MeshCoordinate(1, 1), MeshCoordinate(3, 3));
    auto intersection = range.intersection(MeshCoordinateRange(MeshCoordinate(2, 2), MeshCoordinate(4, 4)));
    ASSERT_TRUE(intersection.has_value());
    EXPECT_EQ(intersection->start_coord(), MeshCoordinate(2, 2));
    EXPECT_EQ(intersection->end_coord(), MeshCoordinate(3, 3));

    intersection = range.intersection(MeshCoordinateRange(MeshCoordinate(1, 1), MeshCoordinate(1, 1)));
    ASSERT_TRUE(intersection.has_value());
    EXPECT_EQ(intersection->start_coord(), MeshCoordinate(1, 1));
    EXPECT_EQ(intersection->end_coord(), MeshCoordinate(1, 1));

    intersection = range.intersection(MeshCoordinateRange(MeshCoordinate(3, 3), MeshCoordinate(3, 3)));
    ASSERT_TRUE(intersection.has_value());
    EXPECT_EQ(intersection->start_coord(), MeshCoordinate(3, 3));
    EXPECT_EQ(intersection->end_coord(), MeshCoordinate(3, 3));

    intersection = range.intersection(MeshCoordinateRange(MeshCoordinate(2, 2), MeshCoordinate(2, 2)));
    ASSERT_TRUE(intersection.has_value());
    EXPECT_EQ(intersection->start_coord(), MeshCoordinate(2, 2));
    EXPECT_EQ(intersection->end_coord(), MeshCoordinate(2, 2));

    intersection = range.intersection(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(5, 5)));
    ASSERT_TRUE(intersection.has_value());
    EXPECT_EQ(intersection->start_coord(), MeshCoordinate(1, 1));
    EXPECT_EQ(intersection->end_coord(), MeshCoordinate(3, 3));

    intersection = range.intersection(MeshCoordinateRange(MeshCoordinate(5, 5), MeshCoordinate(6, 6)));
    EXPECT_FALSE(intersection.has_value());
}

TEST(MeshCoordinateRangeTest, Dimensionality) {
    EXPECT_EQ(MeshCoordinateRange(MeshCoordinate(0), MeshCoordinate(5)).dims(), 1);
    EXPECT_EQ(MeshCoordinateRange(MeshCoordinate(0, 1), MeshCoordinate(5, 1)).dims(), 2);
    EXPECT_EQ(MeshCoordinateRange(MeshCoordinate(0, 1, 2), MeshCoordinate(5, 1, 2)).dims(), 3);
}

TEST(MeshCoordinateRangeTest, ContainsMismatchedDimensions) {
    MeshCoordinateRange range(MeshCoordinate(1, 1, 3), MeshCoordinate(1, 1, 3));

    EXPECT_EQ(range.dims(), 3);
    EXPECT_ANY_THROW(range.contains(MeshCoordinate(1, 1)));
}

TEST(MeshCoordinateRangeTest, MismatchedDimensions) {
    MeshCoordinate start(1, 0);
    MeshCoordinate end(2, 3, 1);
    EXPECT_ANY_THROW(MeshCoordinateRange(start, end));
}

TEST(MeshCoordinateRangeTest, InvalidRange) {
    MeshCoordinate start(1, 2, 0);
    MeshCoordinate end(1, 1, 1);
    EXPECT_ANY_THROW(MeshCoordinateRange(start, end));
}

TEST(MeshCoordinateRangeSetTest, MergeInvalidDimensions) {
    MeshCoordinateRangeSet range_set;
    range_set.merge(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));

    EXPECT_ANY_THROW(range_set.merge(MeshCoordinateRange(MeshCoordinate(0, 0, 0), MeshCoordinate(1, 1, 1))));
}

TEST(MeshCoordinateRangeSetTest, Merge1D) {
    MeshCoordinateRangeSet set;

    // Merge first range: [0, 3].
    set.merge(MeshCoordinateRange(MeshCoordinate(0), MeshCoordinate(3)));
    EXPECT_THAT(set.ranges(), ElementsAre(MeshCoordinateRange(MeshCoordinate(0), MeshCoordinate(3))));

    // Merge an adjacent range: [4, 6] (adjacent to r1, since 3 and 4 touch).
    set.merge(MeshCoordinateRange(MeshCoordinate(4), MeshCoordinate(6)));
    EXPECT_THAT(set.ranges(), ElementsAre(MeshCoordinateRange(MeshCoordinate(0), MeshCoordinate(6))));

    // Merge a separate range: [8, 10].
    set.merge(MeshCoordinateRange(MeshCoordinate(8), MeshCoordinate(10)));
    ASSERT_EQ(set.size(), 2);
    EXPECT_THAT(
        set.ranges(),
        ElementsAre(
            Eq(MeshCoordinateRange(MeshCoordinate(0), MeshCoordinate(6))),
            Eq(MeshCoordinateRange(MeshCoordinate(8), MeshCoordinate(10)))));

    // Merge a range bridging the gap: [7, 7] should merge all into one [0, 10].
    set.merge(MeshCoordinateRange(MeshCoordinate(7), MeshCoordinate(7)));

    EXPECT_THAT(set.ranges(), ElementsAre(Eq(MeshCoordinateRange(MeshCoordinate(0), MeshCoordinate(10)))));
}

TEST(MeshCoordinateRangeSetTest, MergeOrder) {
    MeshCoordinateRangeSet set;
    set.merge(MeshCoordinateRange(MeshCoordinate(5, 5), MeshCoordinate(6, 6)));

    EXPECT_THAT(set.ranges(), ElementsAre(Eq(MeshCoordinateRange(MeshCoordinate(5, 5), MeshCoordinate(6, 6)))));

    set.merge(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));

    EXPECT_THAT(
        set.ranges(),
        ElementsAre(
            Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1))),
            Eq(MeshCoordinateRange(MeshCoordinate(5, 5), MeshCoordinate(6, 6)))));

    set.merge(MeshCoordinateRange(MeshCoordinate(0, 2), MeshCoordinate(1, 3)));

    EXPECT_THAT(
        set.ranges(),
        ElementsAre(
            Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 3))),
            Eq(MeshCoordinateRange(MeshCoordinate(5, 5), MeshCoordinate(6, 6)))));

    set.merge(MeshCoordinateRange(MeshCoordinate(2, 0), MeshCoordinate(3, 1)));

    EXPECT_THAT(
        set.ranges(),
        ElementsAre(
            Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 3))),
            Eq(MeshCoordinateRange(MeshCoordinate(2, 0), MeshCoordinate(3, 1))),
            Eq(MeshCoordinateRange(MeshCoordinate(5, 5), MeshCoordinate(6, 6)))));

    set.merge(MeshCoordinateRange(MeshCoordinate(2, 2), MeshCoordinate(3, 3)));

    EXPECT_THAT(
        set.ranges(),
        ElementsAre(
            Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(3, 3))),
            Eq(MeshCoordinateRange(MeshCoordinate(5, 5), MeshCoordinate(6, 6)))));

    set.merge(MeshCoordinateRange(MeshCoordinate(4, 0), MeshCoordinate(4, 6)));

    EXPECT_THAT(
        set.ranges(),
        ElementsAre(
            Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(3, 3))),
            Eq(MeshCoordinateRange(MeshCoordinate(4, 0), MeshCoordinate(4, 6))),
            Eq(MeshCoordinateRange(MeshCoordinate(5, 5), MeshCoordinate(6, 6)))));

    set.merge(MeshCoordinateRange(MeshCoordinate(5, 0), MeshCoordinate(6, 4)));

    EXPECT_THAT(
        set.ranges(),
        ElementsAre(
            Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(3, 3))),
            Eq(MeshCoordinateRange(MeshCoordinate(4, 0), MeshCoordinate(6, 6)))));

    set.merge(MeshCoordinateRange(MeshCoordinate(0, 4), MeshCoordinate(3, 6)));

    EXPECT_THAT(set.ranges(), ElementsAre(Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(6, 6)))));
}

TEST(MeshCoordinateRangeSetTest, MergeWithOverlaps) {
    MeshCoordinateRangeSet set;
    set.merge(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));
    EXPECT_THAT(set.ranges(), ElementsAre(Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)))));

    set.merge(MeshCoordinateRange(MeshCoordinate(1, 1), MeshCoordinate(2, 2)));
    EXPECT_THAT(
        set.ranges(),
        ElementsAre(
            Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 1))),
            Eq(MeshCoordinateRange(MeshCoordinate(1, 0), MeshCoordinate(1, 0))),
            Eq(MeshCoordinateRange(MeshCoordinate(1, 1), MeshCoordinate(2, 2)))));

    set.merge(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(3, 3)));
    EXPECT_THAT(set.ranges(), ElementsAre(Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(3, 3)))));

    set.merge(MeshCoordinateRange(MeshCoordinate(0, 4), MeshCoordinate(2, 6)));
    EXPECT_THAT(
        set.ranges(),
        ElementsAre(
            Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(3, 3))),
            Eq(MeshCoordinateRange(MeshCoordinate(0, 4), MeshCoordinate(2, 6)))));

    set.merge(MeshCoordinateRange(MeshCoordinate(2, 2), MeshCoordinate(4, 4)));
    EXPECT_THAT(
        set.ranges(),
        ElementsAre(
            Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 6))),
            Eq(MeshCoordinateRange(MeshCoordinate(2, 0), MeshCoordinate(3, 1))),
            Eq(MeshCoordinateRange(MeshCoordinate(2, 2), MeshCoordinate(4, 4))),
            Eq(MeshCoordinateRange(MeshCoordinate(2, 5), MeshCoordinate(2, 6)))));

    set.merge(MeshCoordinateRange(MeshCoordinate(0, 3), MeshCoordinate(1, 3)));
    EXPECT_THAT(
        set.ranges(),
        ElementsAre(
            Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 6))),
            Eq(MeshCoordinateRange(MeshCoordinate(2, 0), MeshCoordinate(3, 1))),
            Eq(MeshCoordinateRange(MeshCoordinate(2, 2), MeshCoordinate(4, 4))),
            Eq(MeshCoordinateRange(MeshCoordinate(2, 5), MeshCoordinate(2, 6)))));

    set.merge(MeshCoordinateRange(MeshCoordinate(3, 0), MeshCoordinate(4, 2)));
    EXPECT_THAT(
        set.ranges(),
        ElementsAre(
            Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 6))),
            Eq(MeshCoordinateRange(MeshCoordinate(2, 0), MeshCoordinate(4, 2))),
            Eq(MeshCoordinateRange(MeshCoordinate(2, 3), MeshCoordinate(2, 6))),
            Eq(MeshCoordinateRange(MeshCoordinate(3, 3), MeshCoordinate(4, 4)))));

    set.merge(MeshCoordinateRange(MeshCoordinate(1, 3), MeshCoordinate(4, 6)));
    EXPECT_THAT(set.ranges(), ElementsAre(Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(4, 6)))));

    set.merge(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(5, 7)));
    EXPECT_THAT(set.ranges(), ElementsAre(Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(5, 7)))));
}

TEST(MeshCoordinateRangeSetTest, SubtractInvalidDimensions) {
    EXPECT_ANY_THROW(subtract(
        MeshCoordinateRange(MeshCoordinate(0, 0, 0), MeshCoordinate(1, 1, 1)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1))));
}

TEST(MeshCoordinateRangeSetTest, SubtractNoIntersection) {
    MeshCoordinateRange parent(MeshCoordinate(0, 0), MeshCoordinate(4, 10));
    MeshCoordinateRange intersection(MeshCoordinate(5, 5), MeshCoordinate(12, 12));
    EXPECT_THAT(subtract(parent, intersection).ranges(), ElementsAre(Eq(parent)));
}

TEST(MeshCoordinateRangeSetTest, SubtractParentEqualsIntersection) {
    MeshCoordinateRange parent(MeshCoordinate(0, 0), MeshCoordinate(4, 10));
    MeshCoordinateRange intersection(MeshCoordinate(0, 0), MeshCoordinate(4, 10));
    EXPECT_THAT(subtract(parent, intersection).ranges(), IsEmpty());
}

TEST(MeshCoordinateRangeSetTest, Subtract1DAdjacentIntersection) {
    // Parent [0, 10] and intersection [3, 7] should yield [0,2] and [8,10].
    MeshCoordinateRange parent(MeshCoordinate(0), MeshCoordinate(10));
    MeshCoordinateRange intersection(MeshCoordinate(3), MeshCoordinate(7));

    EXPECT_THAT(
        subtract(parent, intersection).ranges(),
        ElementsAre(
            Eq(MeshCoordinateRange(MeshCoordinate(0), MeshCoordinate(2))),
            Eq(MeshCoordinateRange(MeshCoordinate(8), MeshCoordinate(10)))));
}

TEST(MeshCoordinateRangeSetTest, Subtract2DNonAdjacentIntersection) {
    MeshCoordinateRange parent(MeshCoordinate(0, 0), MeshCoordinate(2, 2));
    MeshCoordinateRange intersection(MeshCoordinate(1, 1), MeshCoordinate(1, 1));

    EXPECT_THAT(
        subtract(parent, intersection).ranges(),
        ElementsAre(
            Eq(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 2))),
            Eq(MeshCoordinateRange(MeshCoordinate(1, 0), MeshCoordinate(2, 0))),
            Eq(MeshCoordinateRange(MeshCoordinate(1, 2), MeshCoordinate(2, 2))),
            Eq(MeshCoordinateRange(MeshCoordinate(2, 1), MeshCoordinate(2, 1)))));
}

TEST(MeshCoordinateRangeSetTest, Equality) {
    MeshCoordinateRangeSet set1;
    set1.merge(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));
    set1.merge(MeshCoordinateRange(MeshCoordinate(3, 3), MeshCoordinate(4, 4)));

    MeshCoordinateRangeSet set2;
    // Add in different order, should still be equal due to sorting
    set2.merge(MeshCoordinateRange(MeshCoordinate(3, 3), MeshCoordinate(4, 4)));
    set2.merge(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));

    MeshCoordinateRangeSet set3;
    set3.merge(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));
    set3.merge(MeshCoordinateRange(MeshCoordinate(3, 3), MeshCoordinate(5, 5)));

    MeshCoordinateRangeSet set4;
    set4.merge(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));

    EXPECT_EQ(set1, set2);
    EXPECT_NE(set1, set3);
    EXPECT_NE(set1, set4);
    EXPECT_NE(set2, set3);
    EXPECT_NE(set2, set4);
    EXPECT_NE(set3, set4);
}

TEST(MeshCoordinateRangeSetTest, UnorderedSet) {
    MeshCoordinateRangeSet set1;
    set1.merge(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));
    set1.merge(MeshCoordinateRange(MeshCoordinate(3, 3), MeshCoordinate(4, 4)));

    MeshCoordinateRangeSet set2;  // Same ranges as set1, added in different order
    set2.merge(MeshCoordinateRange(MeshCoordinate(3, 3), MeshCoordinate(4, 4)));
    set2.merge(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));

    MeshCoordinateRangeSet set3;  // Different set
    set3.merge(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(2, 2)));

    std::unordered_set<MeshCoordinateRangeSet> set;
    EXPECT_TRUE(set.insert(set1).second);
    EXPECT_FALSE(set.insert(set2).second);
    EXPECT_TRUE(set.insert(set3).second);

    EXPECT_THAT(set, UnorderedElementsAre(Eq(set1), Eq(set3)));
}

TEST(MeshCoordinateRangeSetTest, Coords) {
    MeshCoordinateRangeSet set;

    EXPECT_THAT(set.coords(), IsEmpty());

    set.merge(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 1)));
    set.merge(MeshCoordinateRange(MeshCoordinate(1, 1), MeshCoordinate(1, 1)));
    set.merge(MeshCoordinateRange(MeshCoordinate(0, 3), MeshCoordinate(0, 3)));

    EXPECT_THAT(
        set.coords(),
        ElementsAre(
            MeshCoordinate(0, 0),  //
            MeshCoordinate(0, 1),
            MeshCoordinate(0, 3),
            MeshCoordinate(1, 1)));

    set.merge(MeshCoordinateRange(MeshCoordinate(0, 2), MeshCoordinate(1, 2)));
    set.merge(MeshCoordinateRange(MeshCoordinate(1, 3), MeshCoordinate(1, 3)));

    EXPECT_THAT(
        set.coords(),
        ElementsAre(
            MeshCoordinate(0, 0),
            MeshCoordinate(0, 1),
            MeshCoordinate(0, 2),
            MeshCoordinate(0, 3),
            MeshCoordinate(1, 1),
            MeshCoordinate(1, 2),
            MeshCoordinate(1, 3)));
}

TEST(ToLinearIndexTest, Basic) {
    MeshShape shape(2, 2, 3);

    EXPECT_EQ(MeshCoordinate(0, 0, 0).to_linear_index(shape), 0);
    EXPECT_EQ(MeshCoordinate(0, 0, 1).to_linear_index(shape), 1);
    EXPECT_EQ(MeshCoordinate(0, 0, 2).to_linear_index(shape), 2);
    EXPECT_EQ(MeshCoordinate(0, 1, 0).to_linear_index(shape), 3);
    EXPECT_EQ(MeshCoordinate(0, 1, 1).to_linear_index(shape), 4);
    EXPECT_EQ(MeshCoordinate(0, 1, 2).to_linear_index(shape), 5);
    EXPECT_EQ(MeshCoordinate(1, 0, 0).to_linear_index(shape), 6);
    EXPECT_EQ(MeshCoordinate(1, 0, 1).to_linear_index(shape), 7);
    EXPECT_EQ(MeshCoordinate(1, 0, 2).to_linear_index(shape), 8);
    EXPECT_EQ(MeshCoordinate(1, 1, 0).to_linear_index(shape), 9);
    EXPECT_EQ(MeshCoordinate(1, 1, 1).to_linear_index(shape), 10);
    EXPECT_EQ(MeshCoordinate(1, 1, 2).to_linear_index(shape), 11);
}

TEST(ToLinearIndexTest, MismatchedDimensions) {
    EXPECT_ANY_THROW(MeshCoordinate(0, 0).to_linear_index(MeshShape(1, 2, 3)));
}

TEST(ToLinearIndexTest, OutOfBounds) {
    EXPECT_ANY_THROW(MeshCoordinate(2, 0).to_linear_index(MeshShape(2, 3)));
    EXPECT_ANY_THROW(MeshCoordinate(0, 3).to_linear_index(MeshShape(2, 3)));
}

TEST(MeshContainerTest, InitialValues) {
    MeshShape shape(2, 3);
    MeshContainer<int> container(shape, 3);

    std::vector<int> initial_values;
    for (const auto& [_, value] : container) {
        initial_values.push_back(value);
    }
    EXPECT_THAT(initial_values, ElementsAre(3, 3, 3, 3, 3, 3));
}

TEST(MeshContainerTest, FromVector) {
    MeshShape shape(2, 3);
    MeshContainer<int> container(shape, std::vector<int>{0, 1, 2, 3, 4, 5});

    std::vector<int> initial_values;
    for (const auto& [_, value] : container) {
        initial_values.push_back(value);
    }
    EXPECT_THAT(initial_values, ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST(MeshContainerTest, FromVectorInvalidSize) {
    MeshShape shape(2, 3);
    EXPECT_ANY_THROW(MeshContainer<int>(shape, std::vector<int>{0, 1, 2, 3, 4}));
}

TEST(MeshContainerTest, ElementAccessRowMajor) {
    MeshShape shape(2, 3);
    MeshContainer<int> container(shape, 0);

    container.at(MeshCoordinate(0, 0)) = 0;
    container.at(MeshCoordinate(0, 1)) = 1;
    container.at(MeshCoordinate(0, 2)) = 2;
    container.at(MeshCoordinate(1, 0)) = 3;
    container.at(MeshCoordinate(1, 1)) = 4;
    container.at(MeshCoordinate(1, 2)) = 5;

    std::vector<MeshCoordinate> coords;
    std::vector<int> values;
    for (const auto& [coord, value] : container) {
        coords.push_back(coord);
        values.push_back(value);
    }
    EXPECT_THAT(
        coords,
        ElementsAre(
            MeshCoordinate(0, 0),
            MeshCoordinate(0, 1),
            MeshCoordinate(0, 2),
            MeshCoordinate(1, 0),
            MeshCoordinate(1, 1),
            MeshCoordinate(1, 2)));
    EXPECT_THAT(values, ElementsAre(0, 1, 2, 3, 4, 5));
    EXPECT_THAT(container.values(), ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST(MeshContainerTest, ConstContainer) {
    MeshShape shape(2, 3);
    const MeshContainer<int> container(shape, 0);

    std::vector<MeshCoordinate> coords;
    std::vector<int> values;
    for (const auto& [coord, value] : container) {
        coords.push_back(coord);
        values.push_back(value);
    }
    EXPECT_THAT(
        coords,
        ElementsAre(
            MeshCoordinate(0, 0),
            MeshCoordinate(0, 1),
            MeshCoordinate(0, 2),
            MeshCoordinate(1, 0),
            MeshCoordinate(1, 1),
            MeshCoordinate(1, 2)));
    EXPECT_THAT(values, ElementsAre(0, 0, 0, 0, 0, 0));
    EXPECT_THAT(container.values(), ElementsAre(0, 0, 0, 0, 0, 0));
}

TEST(MeshContainerTest, MutateThroughProxy) {
    MeshShape shape(2, 3);
    MeshContainer<int> container(shape, 0);

    // Proxy class provides access to the container value through the mutable reference.
    int updated_value = 0;
    for (auto& [_, value] : container) {
        value = updated_value++;
    }

    // `auto` makes a copy of the value, verify this loop is a no-op.
    for (auto [_, value] : container) {
        value = updated_value++;
    }

    std::vector<int> values;
    for (const auto& [_, value] : container) {
        values.push_back(value);
    }
    EXPECT_THAT(values, ElementsAre(0, 1, 2, 3, 4, 5));
    EXPECT_THAT(container.values(), ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST(MeshContainerTest, OutOfBounds) {
    MeshShape shape(2, 3);
    MeshContainer<int> container(shape, 0);

    EXPECT_ANY_THROW(container.at(MeshCoordinate(2, 0)));
    EXPECT_ANY_THROW(container.at(MeshCoordinate(0, 0, 0)));
}

// Test that MeshContainer iterators have the correct iterator traits
TEST(MeshContainerIteratorTraitsTest, IteratorTraits) {
    using Container = MeshContainer<int>;
    using Iterator = Container::Iterator;
    using ConstIterator = Container::ConstIterator;

    // Check Iterator traits
    EXPECT_TRUE((std::is_same_v<std::iterator_traits<Iterator>::iterator_category, std::forward_iterator_tag>));
    EXPECT_TRUE((std::is_same_v<std::iterator_traits<Iterator>::value_type, Iterator::ValueProxy>));
    EXPECT_TRUE((std::is_same_v<std::iterator_traits<Iterator>::difference_type, std::ptrdiff_t>));
    EXPECT_TRUE((std::is_same_v<std::iterator_traits<Iterator>::pointer, Iterator::ValueProxy*>));
    EXPECT_TRUE((std::is_same_v<std::iterator_traits<Iterator>::reference, Iterator::ValueProxy&>));

    // Check ConstIterator traits
    EXPECT_TRUE((std::is_same_v<std::iterator_traits<ConstIterator>::iterator_category, std::forward_iterator_tag>));
    EXPECT_TRUE((std::is_same_v<std::iterator_traits<ConstIterator>::value_type, ConstIterator::ValueProxy>));
    EXPECT_TRUE((std::is_same_v<std::iterator_traits<ConstIterator>::difference_type, std::ptrdiff_t>));
    EXPECT_TRUE((std::is_same_v<std::iterator_traits<ConstIterator>::pointer, const ConstIterator::ValueProxy*>));
    EXPECT_TRUE((std::is_same_v<std::iterator_traits<ConstIterator>::reference, const ConstIterator::ValueProxy&>));
}

// Test that iterators work with STL algorithms
TEST(MeshContainerIteratorTraitsTest, STLAlgorithmsCompatibility) {
    MeshContainer<int> container(MeshShape(2, 3), std::vector<int>{0, 1, 2, 3, 4, 5});

    // Test std::for_each
    std::vector<int> values_from_foreach;
    std::for_each(container.begin(), container.end(), [&values_from_foreach](const auto& proxy) {
        values_from_foreach.push_back(proxy.value());
    });
    EXPECT_THAT(values_from_foreach, ElementsAre(0, 1, 2, 3, 4, 5));

    // Test std::find_if
    auto it = std::find_if(container.begin(), container.end(), [](const auto& proxy) {
        return proxy.value() == 3;
    });
    ASSERT_NE(it, container.end());
    EXPECT_EQ(it->value(), 3);
    EXPECT_EQ(it->coord(), MeshCoordinate(1, 0));

    // Test std::count_if
    auto count = std::count_if(container.begin(), container.end(), [](const auto& proxy) {
        return proxy.value() > 2;
    });
    EXPECT_EQ(count, 3);

    // Test std::any_of
    bool has_value_4 = std::any_of(container.begin(), container.end(), [](const auto& proxy) {
        return proxy.value() == 4;
    });
    EXPECT_TRUE(has_value_4);

    // Test std::all_of
    bool all_non_negative = std::all_of(container.begin(), container.end(), [](const auto& proxy) {
        return proxy.value() >= 0;
    });
    EXPECT_TRUE(all_non_negative);

    // Test std::none_of
    bool none_greater_than_10 = std::none_of(container.begin(), container.end(), [](const auto& proxy) {
        return proxy.value() > 10;
    });
    EXPECT_TRUE(none_greater_than_10);
}

TEST(GetNeighborTest, Basic1D) {
    MeshShape shape(5);
    MeshCoordinate coord(2);

    // Move forward
    EXPECT_THAT(coord.get_neighbor(shape, 1, 0, MeshCoordinate::BoundaryMode::WRAP), Optional(MeshCoordinate(3)));

    // Move backward
    EXPECT_THAT(coord.get_neighbor(shape, -1, 0, MeshCoordinate::BoundaryMode::WRAP), Optional(MeshCoordinate(1)));

    // Move multiple steps
    EXPECT_THAT(coord.get_neighbor(shape, 2, 0, MeshCoordinate::BoundaryMode::WRAP), Optional(MeshCoordinate(4)));
}

TEST(GetNeighborTest, Basic2D) {
    MeshShape shape(3, 4);
    MeshCoordinate coord(1, 2);

    // Move along dimension 0 (row)
    EXPECT_THAT(coord.get_neighbor(shape, 1, 0, MeshCoordinate::BoundaryMode::WRAP), Optional(MeshCoordinate(2, 2)));
    EXPECT_THAT(coord.get_neighbor(shape, -1, 0, MeshCoordinate::BoundaryMode::WRAP), Optional(MeshCoordinate(0, 2)));

    // Move along dimension 1 (column)
    EXPECT_THAT(coord.get_neighbor(shape, 1, 1, MeshCoordinate::BoundaryMode::WRAP), Optional(MeshCoordinate(1, 3)));
    EXPECT_THAT(coord.get_neighbor(shape, -1, 1, MeshCoordinate::BoundaryMode::WRAP), Optional(MeshCoordinate(1, 1)));
}

TEST(GetNeighborTest, BoundaryModeWrap) {
    MeshShape shape(3, 4);

    // Wrap around at edges
    EXPECT_THAT(
        MeshCoordinate(0, 0).get_neighbor(shape, -1, 0, MeshCoordinate::BoundaryMode::WRAP),
        Optional(MeshCoordinate(2, 0)));  // Wraps to last row

    EXPECT_THAT(
        MeshCoordinate(2, 3).get_neighbor(shape, 1, 1, MeshCoordinate::BoundaryMode::WRAP),
        Optional(MeshCoordinate(2, 0)));  // Wraps to first column

    // Wrap with larger offsets
    EXPECT_THAT(
        MeshCoordinate(1, 1).get_neighbor(shape, 5, 1, MeshCoordinate::BoundaryMode::WRAP),
        Optional(MeshCoordinate(1, 2)));  // (1 + 5) % 4 = 2

    // Negative wrap with larger offsets
    EXPECT_THAT(
        MeshCoordinate(1, 1).get_neighbor(shape, -6, 1, MeshCoordinate::BoundaryMode::WRAP),
        Optional(MeshCoordinate(1, 3)));  // (1 - 6) wrapped = 3
}

TEST(GetNeighborTest, BoundaryModeClamp) {
    MeshShape shape(3, 4);

    // Clamp at boundaries
    EXPECT_THAT(
        MeshCoordinate(0, 0).get_neighbor(shape, -1, 0, MeshCoordinate::BoundaryMode::CLAMP),
        Optional(MeshCoordinate(0, 0)));  // Clamped to boundary

    EXPECT_THAT(
        MeshCoordinate(2, 3).get_neighbor(shape, 1, 1, MeshCoordinate::BoundaryMode::CLAMP),
        Optional(MeshCoordinate(2, 3)));  // Clamped to boundary

    // Clamp with larger offsets
    EXPECT_THAT(
        MeshCoordinate(1, 1).get_neighbor(shape, 10, 0, MeshCoordinate::BoundaryMode::CLAMP),
        Optional(MeshCoordinate(2, 1)));  // Clamped to max

    EXPECT_THAT(
        MeshCoordinate(1, 1).get_neighbor(shape, -10, 1, MeshCoordinate::BoundaryMode::CLAMP),
        Optional(MeshCoordinate(1, 0)));  // Clamped to min
}

TEST(GetNeighborTest, BoundaryModeNone) {
    MeshShape shape(3, 4);

    // Valid neighbors
    EXPECT_THAT(
        MeshCoordinate(1, 1).get_neighbor(shape, 1, 0, MeshCoordinate::BoundaryMode::NONE),
        Optional(MeshCoordinate(2, 1)));

    // Out of bounds returns nullopt
    EXPECT_THAT(MeshCoordinate(0, 0).get_neighbor(shape, -1, 0, MeshCoordinate::BoundaryMode::NONE), Eq(std::nullopt));
    EXPECT_THAT(MeshCoordinate(2, 3).get_neighbor(shape, 1, 1, MeshCoordinate::BoundaryMode::NONE), Eq(std::nullopt));

    // Larger offsets that go out of bounds
    EXPECT_THAT(MeshCoordinate(1, 1).get_neighbor(shape, 2, 0, MeshCoordinate::BoundaryMode::NONE), Eq(std::nullopt));
}

TEST(GetNeighborTest, NegativeDimensionIndex) {
    MeshShape shape(3, 4, 5);
    MeshCoordinate coord(1, 2, 3);

    // Negative dimension indexing (Python-style)
    EXPECT_THAT(
        coord.get_neighbor(shape, 1, -1, MeshCoordinate::BoundaryMode::WRAP),  // Last dimension
        Optional(MeshCoordinate(1, 2, 4)));

    EXPECT_THAT(
        coord.get_neighbor(shape, 1, -2, MeshCoordinate::BoundaryMode::WRAP),  // Second to last dimension
        Optional(MeshCoordinate(1, 3, 3)));

    EXPECT_THAT(
        coord.get_neighbor(shape, 1, -3, MeshCoordinate::BoundaryMode::WRAP),  // Third to last (first) dimension
        Optional(MeshCoordinate(2, 2, 3)));
}

TEST(GetNeighborTest, DirectionalMovement2D) {
    // Test N/E/S/W movement as in the user's example
    MeshShape shape(3, 4);
    MeshCoordinate src(1, 2);

    // North (negative along dimension 0)
    EXPECT_THAT(src.get_neighbor(shape, -1, 0, MeshCoordinate::BoundaryMode::WRAP), Optional(MeshCoordinate(0, 2)));

    // East (positive along dimension 1)
    EXPECT_THAT(src.get_neighbor(shape, 1, 1, MeshCoordinate::BoundaryMode::WRAP), Optional(MeshCoordinate(1, 3)));

    // South (positive along dimension 0)
    EXPECT_THAT(src.get_neighbor(shape, 1, 0, MeshCoordinate::BoundaryMode::WRAP), Optional(MeshCoordinate(2, 2)));

    // West (negative along dimension 1)
    EXPECT_THAT(src.get_neighbor(shape, -1, 1, MeshCoordinate::BoundaryMode::WRAP), Optional(MeshCoordinate(1, 1)));
}

TEST(GetNeighborTest, TorusTopology) {
    // Test torus wrapping behavior
    MeshShape shape(3, 3);

    // Corner wrapping
    MeshCoordinate corner(0, 0);

    EXPECT_THAT(
        corner.get_neighbor(shape, -1, 0, MeshCoordinate::BoundaryMode::WRAP),
        Optional(MeshCoordinate(2, 0)));  // Wraps to bottom

    EXPECT_THAT(
        corner.get_neighbor(shape, -1, 1, MeshCoordinate::BoundaryMode::WRAP),
        Optional(MeshCoordinate(0, 2)));  // Wraps to right

    // Opposite corner
    corner = MeshCoordinate(2, 2);

    EXPECT_THAT(
        corner.get_neighbor(shape, 1, 0, MeshCoordinate::BoundaryMode::WRAP),
        Optional(MeshCoordinate(0, 2)));  // Wraps to top

    EXPECT_THAT(
        corner.get_neighbor(shape, 1, 1, MeshCoordinate::BoundaryMode::WRAP),
        Optional(MeshCoordinate(2, 0)));  // Wraps to left
}

TEST(GetNeighborTest, DimensionMismatch) {
    MeshShape shape(3, 4);
    MeshCoordinate coord(1, 2, 3);  // 3D coordinate

    // Should throw due to dimension mismatch
    EXPECT_ANY_THROW(coord.get_neighbor(shape, 1, 0, MeshCoordinate::BoundaryMode::WRAP));
}

TEST(GetNeighborTest, InvalidDimension) {
    MeshShape shape(3, 4);
    MeshCoordinate coord(1, 2);

    // Out of range dimension
    EXPECT_ANY_THROW(coord.get_neighbor(shape, 1, 2, MeshCoordinate::BoundaryMode::WRAP));
    EXPECT_ANY_THROW(coord.get_neighbor(shape, 1, -3, MeshCoordinate::BoundaryMode::WRAP));
}

TEST(GetNeighborTest, OutOfBoundsInputCoordinate) {
    MeshShape shape(3, 4);

    // Coordinate with first dimension out of bounds
    MeshCoordinate out_of_bounds1(3, 2);  // First dim is 3, but shape is only 3 (valid: 0-2)
    EXPECT_ANY_THROW(out_of_bounds1.get_neighbor(shape, 1, 0, MeshCoordinate::BoundaryMode::WRAP));
    EXPECT_ANY_THROW(out_of_bounds1.get_neighbor(shape, 1, 0, MeshCoordinate::BoundaryMode::CLAMP));
    EXPECT_ANY_THROW(out_of_bounds1.get_neighbor(shape, 1, 0, MeshCoordinate::BoundaryMode::NONE));

    // Coordinate with second dimension out of bounds
    MeshCoordinate out_of_bounds2(1, 4);  // Second dim is 4, but shape is only 4 (valid: 0-3)
    EXPECT_ANY_THROW(out_of_bounds2.get_neighbor(shape, 1, 1, MeshCoordinate::BoundaryMode::WRAP));
    EXPECT_ANY_THROW(out_of_bounds2.get_neighbor(shape, 1, 1, MeshCoordinate::BoundaryMode::CLAMP));
    EXPECT_ANY_THROW(out_of_bounds2.get_neighbor(shape, 1, 1, MeshCoordinate::BoundaryMode::NONE));

    // Coordinate with both dimensions out of bounds
    MeshCoordinate out_of_bounds3(5, 10);
    EXPECT_ANY_THROW(out_of_bounds3.get_neighbor(shape, 0, 0, MeshCoordinate::BoundaryMode::WRAP));
}

}  // namespace
}  // namespace tt::tt_metal::distributed
