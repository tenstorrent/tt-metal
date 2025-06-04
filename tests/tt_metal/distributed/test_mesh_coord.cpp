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
    EXPECT_TRUE(is_line_topology(MeshShape(1)));
    EXPECT_TRUE(is_line_topology(MeshShape(3)));
    EXPECT_TRUE(is_line_topology(MeshShape(1, 1)));
    EXPECT_TRUE(is_line_topology(MeshShape(1, 3)));
    EXPECT_TRUE(is_line_topology(MeshShape(3, 1)));
    EXPECT_FALSE(is_line_topology(MeshShape(3, 3)));
    EXPECT_TRUE(is_line_topology(MeshShape(1, 1, 1)));
    EXPECT_TRUE(is_line_topology(MeshShape(1, 1, 3)));
    EXPECT_TRUE(is_line_topology(MeshShape(1, 3, 1)));
    EXPECT_TRUE(is_line_topology(MeshShape(3, 1, 1)));
    EXPECT_FALSE(is_line_topology(MeshShape(1, 3, 3)));
    EXPECT_FALSE(is_line_topology(MeshShape(3, 1, 3)));
    EXPECT_FALSE(is_line_topology(MeshShape(3, 3, 3)));
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

    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(0, 0, 0)), 0);
    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(0, 0, 1)), 1);
    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(0, 0, 2)), 2);
    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(0, 1, 0)), 3);
    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(0, 1, 1)), 4);
    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(0, 1, 2)), 5);
    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(1, 0, 0)), 6);
    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(1, 0, 1)), 7);
    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(1, 0, 2)), 8);
    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(1, 1, 0)), 9);
    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(1, 1, 1)), 10);
    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(1, 1, 2)), 11);
}

TEST(ToLinearIndexTest, MismatchedDimensions) {
    EXPECT_ANY_THROW(to_linear_index(MeshShape(1, 2, 3), MeshCoordinate(0, 0)));
}

TEST(ToLinearIndexTest, OutOfBounds) {
    EXPECT_ANY_THROW(to_linear_index(MeshShape(2, 3), MeshCoordinate(2, 0)));
    EXPECT_ANY_THROW(to_linear_index(MeshShape(2, 3), MeshCoordinate(0, 3)));
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

}  // namespace
}  // namespace tt::tt_metal::distributed
