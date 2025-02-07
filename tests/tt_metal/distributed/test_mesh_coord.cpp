// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "mesh_coord.hpp"

namespace tt::tt_metal::distributed {
namespace {

using ::testing::ElementsAre;

TEST(SimpleMeshShapeTest, Construction) {
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

TEST(SimpleMeshShapeTest, Strides) {
    SimpleMeshShape shape(2, 3, 4);
    EXPECT_EQ(shape.get_stride(0), 12);  // 3 * 4
    EXPECT_EQ(shape.get_stride(1), 4);   // 4
    EXPECT_EQ(shape.get_stride(2), 1);   // 1
}

TEST(SimpleMeshShapeTest, Comparison) {
    SimpleMeshShape shape1(2, 3);
    SimpleMeshShape shape2(2, 3);
    SimpleMeshShape shape3(3, 2);

    EXPECT_EQ(shape1, shape2);
    EXPECT_NE(shape1, shape3);
}

TEST(MeshCoordinateTest, Construction) {
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
    MeshCoordinate coord2(1, 2);
    MeshCoordinate coord3(2, 1);

    EXPECT_EQ(coord1, coord2);
    EXPECT_NE(coord1, coord3);
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
    MeshCoordinate start(1, 0);
    MeshCoordinate end(2, 2);
    MeshCoordinateRange range(start, end);

    std::vector<MeshCoordinate> coords;
    for (const auto& coord : range) {
        coords.push_back(coord);
    }

    EXPECT_THAT(
        coords,
        ElementsAre(
            MeshCoordinate(1, 0),
            MeshCoordinate(1, 1),
            MeshCoordinate(1, 2),
            MeshCoordinate(2, 0),
            MeshCoordinate(2, 1),
            MeshCoordinate(2, 2)));
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
    SimpleMeshShape shape(2, 3);

    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(0, 0)), 0);
    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(0, 1)), 1);
    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(0, 2)), 2);
    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(1, 0)), 3);
    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(1, 1)), 4);
    EXPECT_EQ(to_linear_index(shape, MeshCoordinate(1, 2)), 5);
}

TEST(ToLinearIndexTest, MismatchedDimensions) {
    EXPECT_ANY_THROW(to_linear_index(SimpleMeshShape(2, 3), MeshCoordinate(2, 0)));
}

TEST(ToLinearIndexTest, OutOfBounds) {
    EXPECT_ANY_THROW(to_linear_index(SimpleMeshShape(1, 2, 3), MeshCoordinate(0, 0)));
}

TEST(MeshContainerTest, InitialValues) {
    SimpleMeshShape shape(2, 3);
    MeshContainer<int> container(shape, 3);

    std::vector<int> initial_values;
    for (const auto& [coord, value] : container) {
        initial_values.push_back(value);
    }
    EXPECT_THAT(initial_values, ElementsAre(3, 3, 3, 3, 3, 3));
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

    std::vector<int> values;
    for (const auto& [coord, value] : container) {
        values.push_back(value);
    }
    EXPECT_THAT(values, ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST(MeshContainerTest, OutOfBounds) {
    SimpleMeshShape shape(2, 3);
    MeshContainer<int> container(shape, 0);

    EXPECT_ANY_THROW(container.at(MeshCoordinate(2, 0)));
    EXPECT_ANY_THROW(container.at(MeshCoordinate(0, 0, 0)));
}

}  // namespace
}  // namespace tt::tt_metal::distributed
