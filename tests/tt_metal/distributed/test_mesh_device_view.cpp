// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gmock/gmock.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/mesh_device_view.hpp>

namespace tt::tt_metal::distributed {
namespace {

using ::testing::IsEmpty;
using ::testing::SizeIs;

TEST(MeshDeviceViewTest, GetRingCoordinatesRingShapeEmpty) {
    EXPECT_ANY_THROW(
        (void)MeshDeviceView::get_ring_coordinates(/*ring_shape*/ Shape2D(1, 0), /*mesh_shape*/ Shape2D(2, 4)));
    EXPECT_ANY_THROW(
        (void)MeshDeviceView::get_ring_coordinates(/*ring_shape*/ Shape2D(0, 1), /*mesh_shape*/ Shape2D(2, 4)));
}

TEST(MeshDeviceViewTest, GetRingCoordinatesRingShapeTooBig) {
    EXPECT_ANY_THROW(
        (void)MeshDeviceView::get_ring_coordinates(/*ring_shape*/ Shape2D(2, 4), /*mesh_shape*/ Shape2D(2, 2)));
    EXPECT_ANY_THROW(
        (void)MeshDeviceView::get_ring_coordinates(/*ring_shape*/ Shape2D(4, 2), /*mesh_shape*/ Shape2D(2, 2)));
}

TEST(MeshDeviceViewTest, GetRingCoordinates) {
    auto ring_coords = MeshDeviceView::get_ring_coordinates(Shape2D(2, 2), Shape2D(2, 2));
    ASSERT_THAT(ring_coords, SizeIs(4));
    EXPECT_EQ(ring_coords[0], MeshCoordinate(0, 0));
    EXPECT_EQ(ring_coords[1], MeshCoordinate(0, 1));
    EXPECT_EQ(ring_coords[2], MeshCoordinate(1, 1));
    EXPECT_EQ(ring_coords[3], MeshCoordinate(1, 0));
}

TEST(MeshDeviceViewTest, GetRingCoordinatesDonut) {
    auto ring_coords = MeshDeviceView::get_ring_coordinates(Shape2D(3, 3), Shape2D(4, 4));
    ASSERT_THAT(ring_coords, SizeIs(8));
    EXPECT_EQ(ring_coords[0], MeshCoordinate(0, 0));
    EXPECT_EQ(ring_coords[1], MeshCoordinate(0, 1));
    EXPECT_EQ(ring_coords[2], MeshCoordinate(0, 2));
    EXPECT_EQ(ring_coords[3], MeshCoordinate(1, 2));
    EXPECT_EQ(ring_coords[4], MeshCoordinate(2, 2));
    EXPECT_EQ(ring_coords[5], MeshCoordinate(2, 1));
    EXPECT_EQ(ring_coords[6], MeshCoordinate(2, 0));
    EXPECT_EQ(ring_coords[7], MeshCoordinate(1, 0));
}

TEST(MeshDeviceViewTest, GetLineCoordinatesLineTooBig) {
    EXPECT_ANY_THROW((void)MeshDeviceView::get_line_coordinates(
        /*length*/ 10, /*mesh_shape*/ Shape2D(2, 2), /*mesh_offset*/ Shape2D(0, 0)));
}

TEST(MeshDeviceViewTest, GetLineCoordinatesWithShorterLine) {
    auto line_coords =
        MeshDeviceView::get_line_coordinates(/*length*/ 3, /*mesh_shape*/ Shape2D(2, 2), /*mesh_offset*/ Shape2D(0, 0));
    ASSERT_THAT(line_coords, SizeIs(3));
    EXPECT_EQ(line_coords[0], MeshCoordinate(0, 0));
    EXPECT_EQ(line_coords[1], MeshCoordinate(0, 1));
    EXPECT_EQ(line_coords[2], MeshCoordinate(1, 1));
}

TEST(MeshDeviceViewTest, GetLineCoordinates2x2) {
    auto line_coords =
        MeshDeviceView::get_line_coordinates(/*length*/ 4, /*mesh_shape*/ Shape2D(2, 2), /*mesh_offset*/ Shape2D(0, 0));
    ASSERT_THAT(line_coords, SizeIs(4));
    EXPECT_EQ(line_coords[0], MeshCoordinate(0, 0));
    EXPECT_EQ(line_coords[1], MeshCoordinate(0, 1));
    EXPECT_EQ(line_coords[2], MeshCoordinate(1, 1));
    EXPECT_EQ(line_coords[3], MeshCoordinate(1, 0));
}

TEST(MeshDeviceViewTest, GetLineCoordinates2x2WithOffset) {
    auto line_coords =
        MeshDeviceView::get_line_coordinates(/*length*/ 2, /*mesh_shape*/ Shape2D(2, 2), /*mesh_offset*/ Shape2D(1, 0));
    ASSERT_THAT(line_coords, SizeIs(2));
    EXPECT_EQ(line_coords[0], MeshCoordinate(1, 0));
    EXPECT_EQ(line_coords[1], MeshCoordinate(1, 1));
}

TEST(MeshDeviceViewTest, GetLineCoordinates3x3) {
    auto line_coords =
        MeshDeviceView::get_line_coordinates(/*length*/ 9, /*mesh_shape*/ Shape2D(3, 3), /*mesh_offset*/ Shape2D(0, 0));
    ASSERT_THAT(line_coords, SizeIs(9));
    EXPECT_EQ(line_coords[0], MeshCoordinate(0, 0));
    EXPECT_EQ(line_coords[1], MeshCoordinate(0, 1));
    EXPECT_EQ(line_coords[2], MeshCoordinate(0, 2));
    EXPECT_EQ(line_coords[3], MeshCoordinate(1, 2));
    EXPECT_EQ(line_coords[4], MeshCoordinate(1, 1));
    EXPECT_EQ(line_coords[5], MeshCoordinate(1, 0));
    EXPECT_EQ(line_coords[6], MeshCoordinate(2, 0));
    EXPECT_EQ(line_coords[7], MeshCoordinate(2, 1));
    EXPECT_EQ(line_coords[8], MeshCoordinate(2, 2));
}

TEST(MeshDeviceViewTest, GetLineCoordinates3x3WithOffset) {
    auto line_coords =
        MeshDeviceView::get_line_coordinates(/*length*/ 5, /*mesh_shape*/ Shape2D(3, 3), /*mesh_offset*/ Shape2D(1, 1));
    ASSERT_THAT(line_coords, SizeIs(5));
    EXPECT_EQ(line_coords[0], MeshCoordinate(1, 1));
    EXPECT_EQ(line_coords[1], MeshCoordinate(1, 2));
    EXPECT_EQ(line_coords[2], MeshCoordinate(2, 2));
    EXPECT_EQ(line_coords[3], MeshCoordinate(2, 1));
    EXPECT_EQ(line_coords[4], MeshCoordinate(2, 0));
}

}  // namespace
}  // namespace tt::tt_metal::distributed
