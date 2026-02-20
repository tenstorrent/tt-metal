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
#include <distributed/mesh_device_view_impl.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed {
namespace {

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
    // Actual path produced by DFS algorithm (prefers ring but falls back if not possible)
    EXPECT_EQ(line_coords[0], MeshCoordinate(0, 0));
    EXPECT_EQ(line_coords[1], MeshCoordinate(0, 1));
    EXPECT_EQ(line_coords[2], MeshCoordinate(0, 2));
    EXPECT_EQ(line_coords[3], MeshCoordinate(1, 2));
    EXPECT_EQ(line_coords[4], MeshCoordinate(2, 2));
    EXPECT_EQ(line_coords[5], MeshCoordinate(2, 1));
    EXPECT_EQ(line_coords[6], MeshCoordinate(2, 0));
    EXPECT_EQ(line_coords[7], MeshCoordinate(1, 0));
    EXPECT_EQ(line_coords[8], MeshCoordinate(1, 1));
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

TEST(MeshDeviceViewTest, GetLineCoordinatesRingFormation) {
    // Test successful ring formation with various lengths and starting positions
    // Helper lambda to verify ring formation
    auto verify_ring = [](const std::vector<MeshCoordinate>& line_coords) {
        ASSERT_GT(line_coords.size(), 1);
        const MeshCoordinate& start = line_coords[0];
        const MeshCoordinate& last = line_coords.back();
        const size_t row_diff = (last[0] > start[0]) ? (last[0] - start[0]) : (start[0] - last[0]);
        const size_t col_diff = (last[1] > start[1]) ? (last[1] - start[1]) : (start[1] - last[1]);
        EXPECT_TRUE((row_diff == 1 && col_diff == 0) || (row_diff == 0 && col_diff == 1))
            << "Last coordinate " << last << " must be adjacent to start " << start << " to form a ring";
    };

    // Test small length (2 nodes)
    auto line_coords_2 = MeshDeviceView::get_line_coordinates(
        /*length*/ 2, /*mesh_shape*/ Shape2D(3, 3), /*mesh_offset*/ Shape2D(1, 1));
    ASSERT_THAT(line_coords_2, SizeIs(2));
    verify_ring(line_coords_2);

    // Test visiting all nodes in 2x2 mesh
    auto line_coords_4 = MeshDeviceView::get_line_coordinates(
        /*length*/ 4, /*mesh_shape*/ Shape2D(2, 2), /*mesh_offset*/ Shape2D(0, 0));
    ASSERT_THAT(line_coords_4, SizeIs(4));
    std::set<MeshCoordinate> unique_coords_4(line_coords_4.begin(), line_coords_4.end());
    EXPECT_EQ(unique_coords_4.size(), 4);
    verify_ring(line_coords_4);
}

TEST(MeshDeviceViewTest, GetLineCoordinatesRingPreferredButNotRequired) {
    // Test cases where ring formation may not be possible, but function should still succeed
    // The function prefers forming a ring but will return a valid path even if ring is impossible

    // Helper lambda to check if path forms a ring
    // Requesting more nodes than exist in the mesh should still fail
    EXPECT_ANY_THROW((void)MeshDeviceView::get_line_coordinates(
        /*length*/ 10, /*mesh_shape*/ Shape2D(2, 2), /*mesh_offset*/ Shape2D(0, 0)));

    // Cases where ring may not be possible - function should still succeed
    // Visiting 3 nodes from corner (0,0) in 3x3 mesh - may or may not form a ring
    auto line_coords_3_corner = MeshDeviceView::get_line_coordinates(
        /*length*/ 3, /*mesh_shape*/ Shape2D(3, 3), /*mesh_offset*/ Shape2D(0, 0));
    ASSERT_THAT(line_coords_3_corner, SizeIs(3));
    // Function succeeds, ring formation is preferred but not required

    // Visiting 3 nodes from center (1,1) in 3x3 mesh - may or may not form a ring
    auto line_coords_3_center = MeshDeviceView::get_line_coordinates(
        /*length*/ 3, /*mesh_shape*/ Shape2D(3, 3), /*mesh_offset*/ Shape2D(1, 1));
    ASSERT_THAT(line_coords_3_center, SizeIs(3));
    // Function succeeds, ring formation is preferred but not required

    // Visiting all nodes in 3x3 mesh from corner - may or may not form a ring
    auto line_coords_9 = MeshDeviceView::get_line_coordinates(
        /*length*/ 9, /*mesh_shape*/ Shape2D(3, 3), /*mesh_offset*/ Shape2D(0, 0));
    ASSERT_THAT(line_coords_9, SizeIs(9));
    std::set<MeshCoordinate> unique_coords_9(line_coords_9.begin(), line_coords_9.end());
    EXPECT_EQ(unique_coords_9.size(), 9);  // All nodes should be unique
    // Function succeeds, ring formation is preferred but not required
}

using MeshDeviceView2x4Test = MeshDevice2x4Fixture;

TEST_F(MeshDeviceView2x4Test, MeshId) {
    const auto& view = mesh_device_->get_view();
    EXPECT_EQ(view.shape(), MeshShape(2, 4));
    EXPECT_EQ(view.mesh_id(), tt::tt_fabric::MeshId(0));
}

TEST_F(MeshDeviceView2x4Test, ViewBasicProperties) {
    const auto& view = mesh_device_->get_view();

    EXPECT_FALSE(view.empty());
    EXPECT_EQ(view.size(), 8);
    EXPECT_EQ(view.num_devices(), 8);
    EXPECT_EQ(view.shape(), MeshShape(2, 4));
    EXPECT_TRUE(view.is_mesh_2d());
    EXPECT_EQ(view.num_rows(), 2);
    EXPECT_EQ(view.num_cols(), 4);
}

TEST_F(MeshDeviceView2x4Test, ViewContains) {
    const auto& view = mesh_device_->get_view();

    EXPECT_TRUE(view.contains(MeshCoordinate{0, 0}));
    EXPECT_TRUE(view.contains(MeshCoordinate{1, 3}));
    EXPECT_TRUE(view.contains(MeshCoordinate{0, 2}));

    EXPECT_FALSE(view.contains(MeshCoordinate{2, 0}));
    EXPECT_FALSE(view.contains(MeshCoordinate{0, 4}));
    EXPECT_FALSE(view.contains(MeshCoordinate{3, 3}));
}

TEST_F(MeshDeviceView2x4Test, ViewGetDevice) {
    const auto& view = mesh_device_->get_view();

    auto* device_00 = view.impl().get_device(MeshCoordinate{0, 0});
    auto* device_13 = view.impl().get_device(MeshCoordinate{1, 3});

    EXPECT_NE(device_00, nullptr);
    EXPECT_NE(device_13, nullptr);
    EXPECT_NE(device_00->id(), device_13->id());

    // Out of bounds returns nullptr
    EXPECT_EQ(view.impl().get_device(MeshCoordinate{2, 0}), nullptr);
}

TEST_F(MeshDeviceView2x4Test, ViewGetFabricNodeId) {
    const auto& view = mesh_device_->get_view();

    auto fabric_id_00 = view.get_fabric_node_id(MeshCoordinate{0, 0});
    auto fabric_id_13 = view.get_fabric_node_id(MeshCoordinate{1, 3});

    EXPECT_NE(fabric_id_00, fabric_id_13);

    // Out of bounds throws
    EXPECT_ANY_THROW((void)view.get_fabric_node_id(MeshCoordinate{2, 0}));
}

TEST_F(MeshDeviceView2x4Test, ViewGetDevices) {
    const auto& view = mesh_device_->get_view();

    auto all_devices = view.get_devices();
    EXPECT_THAT(all_devices, SizeIs(8));

    // Verify all devices are unique
    std::set<ChipId> device_ids;
    for (auto* device : all_devices) {
        device_ids.insert(device->id());
    }
    EXPECT_EQ(device_ids.size(), 8);
}

TEST_F(MeshDeviceView2x4Test, ViewGetDevicesInRange) {
    const auto& view = mesh_device_->get_view();

    // Get 2x2 subregion
    MeshCoordinateRange range(MeshCoordinate{0, 0}, MeshCoordinate(1, 1));
    auto devices = view.get_devices(range);
    EXPECT_THAT(devices, SizeIs(4));

    // Get 1x4 row
    MeshCoordinateRange row_range(MeshCoordinate{0, 0}, MeshCoordinate(0, 3));
    auto row_devices = view.get_devices(row_range);
    EXPECT_THAT(row_devices, SizeIs(4));
}

TEST_F(MeshDeviceView2x4Test, ViewGetFabricNodeIds) {
    const auto& view = mesh_device_->get_view();

    auto all_fabric_ids = view.get_fabric_node_ids();
    EXPECT_THAT(all_fabric_ids, SizeIs(8));

    // Verify all fabric node IDs are unique
    std::set<tt::tt_fabric::FabricNodeId> fabric_ids_set;
    for (const auto& fabric_id : all_fabric_ids) {
        fabric_ids_set.insert(fabric_id);
    }
    EXPECT_EQ(fabric_ids_set.size(), 8);
}

TEST_F(MeshDeviceView2x4Test, ViewGetFabricNodeIdsInRange) {
    const auto& view = mesh_device_->get_view();

    MeshCoordinateRange range(MeshCoordinate{0, 0}, MeshCoordinate(1, 1));
    auto fabric_ids = view.get_fabric_node_ids(range);
    EXPECT_THAT(fabric_ids, SizeIs(4));
}

TEST_F(MeshDeviceView2x4Test, ViewGetDevicesOnRow) {
    const auto& view = mesh_device_->get_view();

    auto row_0_devices = view.get_devices_on_row(0);
    EXPECT_THAT(row_0_devices, SizeIs(4));

    auto row_1_devices = view.get_devices_on_row(1);
    EXPECT_THAT(row_1_devices, SizeIs(4));

    // Out of bounds throws
    EXPECT_ANY_THROW((void)view.get_devices_on_row(2));
}

TEST_F(MeshDeviceView2x4Test, ViewGetDevicesOnColumn) {
    const auto& view = mesh_device_->get_view();

    auto col_0_devices = view.get_devices_on_column(0);
    EXPECT_THAT(col_0_devices, SizeIs(2));

    auto col_3_devices = view.get_devices_on_column(3);
    EXPECT_THAT(col_3_devices, SizeIs(2));

    // Out of bounds throws
    EXPECT_ANY_THROW((void)view.get_devices_on_column(4));
}

TEST_F(MeshDeviceView2x4Test, ViewGetFabricNodeIdsOnRow) {
    const auto& view = mesh_device_->get_view();

    auto row_0_fabric_ids = view.get_fabric_node_ids_on_row(0);
    EXPECT_THAT(row_0_fabric_ids, SizeIs(4));

    auto row_1_fabric_ids = view.get_fabric_node_ids_on_row(1);
    EXPECT_THAT(row_1_fabric_ids, SizeIs(4));

    // Out of bounds throws
    EXPECT_ANY_THROW((void)view.get_fabric_node_ids_on_row(2));
}

TEST_F(MeshDeviceView2x4Test, ViewGetFabricNodeIdsOnColumn) {
    const auto& view = mesh_device_->get_view();

    auto col_0_fabric_ids = view.get_fabric_node_ids_on_column(0);
    EXPECT_THAT(col_0_fabric_ids, SizeIs(2));

    auto col_3_fabric_ids = view.get_fabric_node_ids_on_column(3);
    EXPECT_THAT(col_3_fabric_ids, SizeIs(2));

    // Out of bounds throws
    EXPECT_ANY_THROW((void)view.get_fabric_node_ids_on_column(4));
}

TEST_F(MeshDeviceView2x4Test, ViewFindDevice) {
    const auto& view = mesh_device_->get_view();

    auto* device = view.impl().get_device(MeshCoordinate{1, 2});
    ASSERT_NE(device, nullptr);

    auto coord = view.find_device(device->id());
    EXPECT_EQ(coord, MeshCoordinate(1, 2));

    // Non-existent device throws
    EXPECT_ANY_THROW((void)view.find_device(9999));
}

TEST_F(MeshDeviceView2x4Test, ViewLineCoordinates) {
    const auto& view = mesh_device_->get_view();

    auto line_coords = view.get_line_coordinates();
    EXPECT_THAT(line_coords, SizeIs(8));

    // Verify zigzag pattern: row 0 left-to-right, row 1 right-to-left
    EXPECT_EQ(line_coords[0], MeshCoordinate(0, 0));
    EXPECT_EQ(line_coords[1], MeshCoordinate(0, 1));
    EXPECT_EQ(line_coords[2], MeshCoordinate(0, 2));
    EXPECT_EQ(line_coords[3], MeshCoordinate(0, 3));
    EXPECT_EQ(line_coords[4], MeshCoordinate(1, 3));
    EXPECT_EQ(line_coords[5], MeshCoordinate(1, 2));
    EXPECT_EQ(line_coords[6], MeshCoordinate(1, 1));
    EXPECT_EQ(line_coords[7], MeshCoordinate(1, 0));
}

TEST_F(MeshDeviceView2x4Test, ViewRingCoordinates) {
    const auto& view = mesh_device_->get_view();

    auto ring_coords = view.get_ring_coordinates();
    EXPECT_THAT(ring_coords, SizeIs(8));

    // Verify ring traversal (clockwise from top-left)
    EXPECT_EQ(ring_coords[0], MeshCoordinate(0, 0));
    EXPECT_EQ(ring_coords[1], MeshCoordinate(0, 1));
    EXPECT_EQ(ring_coords[2], MeshCoordinate(0, 2));
    EXPECT_EQ(ring_coords[3], MeshCoordinate(0, 3));
    EXPECT_EQ(ring_coords[4], MeshCoordinate(1, 3));
    EXPECT_EQ(ring_coords[5], MeshCoordinate(1, 2));
    EXPECT_EQ(ring_coords[6], MeshCoordinate(1, 1));
    EXPECT_EQ(ring_coords[7], MeshCoordinate(1, 0));
}

TEST_F(MeshDeviceView2x4Test, ViewIsLocal) {
    const auto& view = mesh_device_->get_view();

    // All devices should be local in single-host tests
    EXPECT_TRUE(view.impl().is_local(MeshCoordinate{0, 0}));
    EXPECT_TRUE(view.impl().is_local(MeshCoordinate{1, 3}));

    // Out of bounds throws
    EXPECT_ANY_THROW((void)view.impl().is_local(MeshCoordinate{2, 0}));
}

TEST_F(MeshDeviceView2x4Test, ViewIterator) {
    const auto& view = mesh_device_->get_view();

    std::vector<IDevice*> iterated_devices;
    for (auto device : view) {
        iterated_devices.push_back(*device);
    }

    EXPECT_THAT(iterated_devices, SizeIs(8));

    // Should match get_devices()
    auto all_devices = view.get_devices();
    EXPECT_EQ(iterated_devices, all_devices);
}

TEST_F(MeshDeviceView2x4Test, ViewMeshId) {
    const auto& view = mesh_device_->get_view();

    auto mesh_id = view.mesh_id();

    // All fabric node IDs should have the same mesh ID
    for (const auto& fabric_id : view.get_fabric_node_ids()) {
        EXPECT_EQ(fabric_id.mesh_id, mesh_id);
    }
}

TEST_F(MeshDeviceView2x4Test, View2DMethodsThrowOnNon2DMesh) {
    std::vector<IDevice*> devices;
    std::vector<tt::tt_fabric::FabricNodeId> fabric_node_ids;
    for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
        devices.push_back(mesh_device_->get_view().impl().get_device(coord));
        fabric_node_ids.push_back(mesh_device_->get_view().get_fabric_node_id(coord));
    }

    MeshDeviceView view_1d(MeshShape(8), devices, fabric_node_ids);

    EXPECT_ANY_THROW((void)view_1d.num_rows());
    EXPECT_ANY_THROW((void)view_1d.num_cols());
    EXPECT_ANY_THROW((void)view_1d.get_devices_on_row(0));
    EXPECT_ANY_THROW((void)view_1d.get_devices_on_column(0));
    EXPECT_ANY_THROW((void)view_1d.get_fabric_node_ids_on_row(0));
    EXPECT_ANY_THROW((void)view_1d.get_fabric_node_ids_on_column(0));
    EXPECT_ANY_THROW((void)view_1d.get_line_coordinates());
    EXPECT_ANY_THROW((void)view_1d.get_ring_coordinates());
    EXPECT_ANY_THROW((void)view_1d.get_line_devices());
    EXPECT_ANY_THROW((void)view_1d.get_ring_devices());
    EXPECT_ANY_THROW((void)view_1d.get_line_fabric_node_ids());
    EXPECT_ANY_THROW((void)view_1d.get_ring_fabric_node_ids());
}

}  // namespace
}  // namespace tt::tt_metal::distributed
