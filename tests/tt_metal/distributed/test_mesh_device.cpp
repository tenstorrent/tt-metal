// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <tt-metalium/allocator.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/allocator_types.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include "gmock/gmock.h"
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/shape_base.hpp>
#include <tt-metalium/system_mesh.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed {
namespace {

using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::tt::tt_metal::distributed::MeshContainer;

TEST(MeshDeviceInitTest, Init1x1Mesh) {
    auto& sys = SystemMesh::instance();

    MeshDeviceConfig config(MeshShape(1, 1));

    EXPECT_NO_THROW({
        auto mesh = tt::tt_metal::distributed::MeshDevice::create(
            config, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER);
        mesh->close();
    });
}

using MeshDeviceT3000Test = T3000MeshDeviceFixture;
using MeshDeviceTest = GenericMeshDeviceFixture;

TEST_F(MeshDeviceT3000Test, SystemMeshTearDownWithoutClose) {
    auto& sys = SystemMesh::instance();

    const auto system_shape = sys.shape();
    ASSERT_EQ(system_shape.dims(), 2);
    EXPECT_EQ(system_shape[0], 2);
    EXPECT_EQ(system_shape[1], 4);
}

TEST_F(MeshDeviceT3000Test, MemoryAllocationStatistics) {
    auto stats = mesh_device_->allocator()->get_statistics(tt::tt_metal::BufferType::DRAM);
    for (auto* device : mesh_device_->get_devices()) {
        auto device_stats = device->allocator()->get_statistics(tt::tt_metal::BufferType::DRAM);
        EXPECT_EQ(stats.total_allocatable_size_bytes, device_stats.total_allocatable_size_bytes);
    }
}

TEST_F(MeshDeviceT3000Test, ViewIs2D) {
    std::vector<IDevice*> devices = mesh_device_->get_devices();

    MeshContainer<IDevice*> container_1d(MeshShape(8), devices);
    MeshDeviceView view_1d(container_1d);
    EXPECT_FALSE(view_1d.is_mesh_2d());

    MeshContainer<IDevice*> container_2d(MeshShape(2, 4), devices);
    MeshDeviceView view_2d(container_2d);
    EXPECT_TRUE(view_2d.is_mesh_2d());

    MeshContainer<IDevice*> container_3d(MeshShape(2, 2, 2), devices);
    MeshDeviceView view_3d(container_3d);
    EXPECT_FALSE(view_3d.is_mesh_2d());
}

TEST_F(MeshDeviceT3000Test, CreateSubmeshInvalidConfig) {
    EXPECT_EQ(mesh_device_->shape(), MeshShape(2, 4));

    EXPECT_ANY_THROW(mesh_device_->create_submesh(MeshShape{1, 3}, MeshCoordinate{1}));
    EXPECT_ANY_THROW(mesh_device_->create_submesh(MeshShape{0, 3}, MeshCoordinate{0, 0}));
    EXPECT_ANY_THROW(mesh_device_->create_submesh(MeshShape{2, 4}, MeshCoordinate{1, 1}));
    EXPECT_ANY_THROW(mesh_device_->create_submesh(MeshShape{2, 4, 1}, MeshCoordinate{0, 0}));
}

TEST_F(MeshDeviceT3000Test, CreateSubmesh) {
    EXPECT_EQ(mesh_device_->shape(), MeshShape(2, 4));
    EXPECT_THAT(mesh_device_->get_devices(), SizeIs(8));
    EXPECT_TRUE(mesh_device_->is_parent_mesh());
    EXPECT_THAT(mesh_device_->get_submeshes(), IsEmpty());

    auto submesh = mesh_device_->create_submesh(MeshShape{1, 2}, MeshCoordinate{1, 1});
    EXPECT_THAT(mesh_device_->get_submeshes(), SizeIs(1));
    EXPECT_EQ(submesh->shape(), MeshShape(1, 2));
    EXPECT_THAT(submesh->get_devices(), SizeIs(2));
    EXPECT_FALSE(submesh->is_parent_mesh());
    EXPECT_THAT(submesh->get_submeshes(), IsEmpty());

    // Verify coordinates are correct.
    EXPECT_EQ(mesh_device_->get_device(MeshCoordinate{1, 1})->id(), submesh->get_device(MeshCoordinate{0, 0})->id());
    EXPECT_EQ(mesh_device_->get_device(MeshCoordinate{1, 2})->id(), submesh->get_device(MeshCoordinate{0, 1})->id());
    EXPECT_EQ(submesh->get_device(MeshCoordinate{1, 1}), nullptr);
}

TEST_F(MeshDeviceT3000Test, CreateSubmeshesNonDivisibleSubshape) {
    EXPECT_EQ(mesh_device_->shape(), MeshShape(2, 4));
    EXPECT_ANY_THROW(mesh_device_->create_submeshes(MeshShape{1, 3}));
}

TEST_F(MeshDeviceT3000Test, CreateSubmeshes) {
    EXPECT_EQ(mesh_device_->shape(), MeshShape(2, 4));

    auto submeshes = mesh_device_->create_submeshes(MeshShape{1, 2});
    EXPECT_THAT(submeshes, SizeIs(4));
    for (const auto& submesh : submeshes) {
        EXPECT_EQ(submesh->shape(), MeshShape(1, 2));
        EXPECT_THAT(submesh->get_devices(), SizeIs(2));
    }

    EXPECT_EQ(mesh_device_->get_submeshes(), submeshes);
}

TEST(GetOptimalDramBankToLogicalWorkerAssignmentAPI, UnitMeshes) {
    auto device_ids_set = tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids();
    std::vector<int> device_ids(device_ids_set.begin(), device_ids_set.end());
    auto devs = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(device_ids);
    for (auto& [_, dev] : devs) {
        EXPECT_NO_THROW(dev->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0));
        EXPECT_NO_THROW(dev->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_1));
    }
}

TEST_F(MeshDeviceTest, CheckFabricNodeIds) {
    // Check that the fabric node IDs are correctly assigned to the devices in the mesh. Only works for 2D meshes
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    EXPECT_EQ(mesh_device_->shape().dims(), 2);
    for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
        tt_fabric::FabricNodeId fabric_node_id = mesh_device_->get_device_fabric_node_id(coord);
        EXPECT_EQ(
            control_plane.get_fabric_node_id_from_physical_chip_id(mesh_device_->get_device(coord)->id()),
            fabric_node_id);
    }
}
}  // namespace
}  // namespace tt::tt_metal::distributed
