// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <unordered_set>

#include "mesh_coord.hpp"

namespace tt::tt_metal::distributed {
namespace {

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

TEST(SimpleMeshShapeTest, Construction) {
    SimpleMeshShape shape_1d(3);
    EXPECT_EQ(shape_1d.dims(), 1);
    EXPECT_EQ(shape_1d[0], 3);
    EXPECT_EQ(shape_1d.mesh_size(), 3);

    SimpleMeshShape shape_2d(3, 4);
    EXPECT_EQ(shape_2d.dims(), 2);
    EXPECT_EQ(shape_2d[0], 3);
    EXPECT_EQ(shape_2d[1], 4);
    EXPECT_EQ(shape_2d.mesh_size(), 12);

    SimpleMeshShape shape_3d(2, 3, 4);
    EXPECT_EQ(shape_3d.dims(), 3);
    EXPECT_EQ(shape_3d[0], 2);
    EXPECT_EQ(shape_3d[1], 3);
    EXPECT_EQ(shape_3d[2], 4);
    EXPECT_EQ(shape_3d.mesh_size(), 24);

    SimpleMeshShape shape_5d({2, 3, 4, 5, 6});
    EXPECT_EQ(shape_5d.dims(), 5);
    EXPECT_EQ(shape_5d[0], 2);
    EXPECT_EQ(shape_5d[1], 3);
    EXPECT_EQ(shape_5d[2], 4);
    EXPECT_EQ(shape_5d[3], 5);
    EXPECT_EQ(shape_5d[4], 6);
    EXPECT_EQ(shape_5d.mesh_size(), 720);
}

TEST(SimpleMeshShapeTest, ZeroShape) {
    SimpleMeshShape shape({});
    EXPECT_EQ(shape.dims(), 0);
    EXPECT_EQ(shape.mesh_size(), 0);
}

TEST(SimpleMeshShapeTest, Strides) {
    SimpleMeshShape shape(2, 3, 4);
    EXPECT_EQ(shape.get_stride(0), 12);  // 3 * 4
    EXPECT_EQ(shape.get_stride(1), 4);   // 4
    EXPECT_EQ(shape.get_stride(2), 1);   // 1
}

TEST(SimpleMeshShapeTest, Comparison) {
    SimpleMeshShape shape(2, 3);

    EXPECT_EQ(shape, SimpleMeshShape(2, 3));
    EXPECT_NE(shape, SimpleMeshShape(3, 2));
    EXPECT_NE(shape, SimpleMeshShape(1, 2, 3));
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

TEST(MeshCoordinateTest, Comparison) {
    MeshCoordinate coord1(1, 2);

    EXPECT_EQ(coord1, MeshCoordinate(1, 2));
    EXPECT_NE(coord1, MeshCoordinate(2, 1));
    EXPECT_NE(coord1, MeshCoordinate(1, 2, 1));
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

TEST(MeshCoordinateRangeTest, FromShape) {
    SimpleMeshShape shape(2, 3);
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

TEST(ToLinearIndexTest, Basic) {
    SimpleMeshShape shape(2, 2, 3);

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
    EXPECT_ANY_THROW(to_linear_index(SimpleMeshShape(1, 2, 3), MeshCoordinate(0, 0)));
}

TEST(ToLinearIndexTest, OutOfBounds) {
    EXPECT_ANY_THROW(to_linear_index(SimpleMeshShape(2, 3), MeshCoordinate(2, 0)));
    EXPECT_ANY_THROW(to_linear_index(SimpleMeshShape(2, 3), MeshCoordinate(0, 3)));
}

TEST(MeshContainerTest, InitialValues) {
    SimpleMeshShape shape(2, 3);
    MeshContainer<int> container(shape, 3);

    std::vector<int> initial_values;
    for (const auto& [_, value] : container) {
        initial_values.push_back(value);
    }
    EXPECT_THAT(initial_values, ElementsAre(3, 3, 3, 3, 3, 3));
}

TEST(MeshContainerTest, FromVector) {
    SimpleMeshShape shape(2, 3);
    MeshContainer<int> container(shape, std::vector<int>{0, 1, 2, 3, 4, 5});

    std::vector<int> initial_values;
    for (const auto& [_, value] : container) {
        initial_values.push_back(value);
    }
    EXPECT_THAT(initial_values, ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST(MeshContainerTest, FromVectorInvalidSize) {
    SimpleMeshShape shape(2, 3);
    EXPECT_ANY_THROW(MeshContainer<int>(shape, std::vector<int>{0, 1, 2, 3, 4}));
}

TEST(MeshContainerTest, ElementAccessRowMajor) {
    SimpleMeshShape shape(2, 3);
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
    SimpleMeshShape shape(2, 3);
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
    SimpleMeshShape shape(2, 3);
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
    SimpleMeshShape shape(2, 3);
    MeshContainer<int> container(shape, 0);

    EXPECT_ANY_THROW(container.at(MeshCoordinate(2, 0)));
    EXPECT_ANY_THROW(container.at(MeshCoordinate(0, 0, 0)));
}

}  // namespace
}  // namespace tt::tt_metal::distributed
