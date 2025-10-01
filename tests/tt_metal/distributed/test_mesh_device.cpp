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
    MeshDeviceConfig config(MeshShape(1, 1));

    EXPECT_NO_THROW({
        auto mesh = tt::tt_metal::distributed::MeshDevice::create(
            config, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER);
        mesh->close();
    });
}

using MeshDevice2x4Test = MeshDevice2x4Fixture;
using MeshDeviceTest = GenericMeshDeviceFixture;

TEST_F(MeshDevice2x4Test, SystemMeshTearDownWithoutClose) {
    auto& sys = SystemMesh::instance();

    const auto system_shape = sys.shape();
    ASSERT_EQ(system_shape.dims(), 2);
    EXPECT_EQ(system_shape[0], 2);
    EXPECT_EQ(system_shape[1], 4);
}

TEST_F(MeshDevice2x4Test, MemoryAllocationStatistics) {
    auto stats = mesh_device_->allocator()->get_statistics(tt::tt_metal::BufferType::DRAM);
    for (auto* device : mesh_device_->get_devices()) {
        auto device_stats = device->allocator()->get_statistics(tt::tt_metal::BufferType::DRAM);
        EXPECT_EQ(stats.total_allocatable_size_bytes, device_stats.total_allocatable_size_bytes);
    }
}

TEST_F(MeshDevice2x4Test, ViewIs2D) {
    std::vector<IDevice*> devices;
    std::vector<tt::tt_fabric::FabricNodeId> fabric_node_ids;
    for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
        devices.push_back(mesh_device_->get_view().get_device(coord));
        fabric_node_ids.push_back(mesh_device_->get_view().get_fabric_node_id(coord));
    }

    MeshDeviceView view_1d(MeshShape(8), devices, fabric_node_ids);
    EXPECT_FALSE(view_1d.is_mesh_2d());

    MeshDeviceView view_2d(MeshShape(2, 4), devices, fabric_node_ids);
    EXPECT_TRUE(view_2d.is_mesh_2d());

    MeshDeviceView view_3d(MeshShape(2, 2, 2), devices, fabric_node_ids);
    EXPECT_FALSE(view_3d.is_mesh_2d());
}

TEST_F(MeshDevice2x4Test, CreateSubmeshInvalidConfig) {
    EXPECT_EQ(mesh_device_->shape(), MeshShape(2, 4));

    EXPECT_ANY_THROW(mesh_device_->create_submesh(MeshShape{1, 3}, MeshCoordinate{1}));
    EXPECT_ANY_THROW(mesh_device_->create_submesh(MeshShape{0, 3}, MeshCoordinate{0, 0}));
    EXPECT_ANY_THROW(mesh_device_->create_submesh(MeshShape{2, 4}, MeshCoordinate{1, 1}));
    EXPECT_ANY_THROW(mesh_device_->create_submesh(MeshShape{2, 4, 1}, MeshCoordinate{0, 0}));
}

TEST_F(MeshDevice2x4Test, CreateSubmesh) {
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

TEST_F(MeshDevice2x4Test, CreateSubmeshesNonDivisibleSubshape) {
    EXPECT_EQ(mesh_device_->shape(), MeshShape(2, 4));
    EXPECT_ANY_THROW(mesh_device_->create_submeshes(MeshShape{1, 3}));
}

TEST_F(MeshDevice2x4Test, CreateSubmeshes) {
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

TEST(ThrowOnMultipleMeshDeviceInitialization, UnitMeshes) {
    auto device_ids_set = tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids();
    std::vector<int> device_ids(device_ids_set.begin(), device_ids_set.end());
    auto unit_meshes = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(device_ids);
    for (auto& [_, unit_mesh] : unit_meshes) {
        EXPECT_EQ(unit_mesh->is_initialized(), true);
        EXPECT_ANY_THROW(unit_mesh->initialize(
            /*num_hw_cqs=*/1,
            /*l1_small_size=*/DEFAULT_L1_SMALL_SIZE,
            /*trace_region_size=*/DEFAULT_TRACE_REGION_SIZE,
            /*worker_l1_size=*/DEFAULT_WORKER_L1_SIZE,
            /*l1_bank_remap=*/{},
            /*minimal=*/false)
        );
    }
}

TEST_F(MeshDeviceTest, CheckFabricNodeIds) {
    // Check that the fabric node IDs are correctly assigned to the devices in the mesh. Only works for 2D meshes
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    EXPECT_EQ(mesh_device_->shape().dims(), 2);
    for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
        tt_fabric::FabricNodeId fabric_node_id = mesh_device_->get_fabric_node_id(coord);
        EXPECT_EQ(
            control_plane.get_fabric_node_id_from_physical_chip_id(mesh_device_->get_device(coord)->id()),
            fabric_node_id);
    }
}

TEST_F(MeshDevice2x4Test, OverlappedSubmeshes) {
    const auto submesh_range_1 = MeshCoordinateRange(MeshShape{2, 2});
    const auto submesh_range_2 = MeshCoordinateRange(MeshCoordinate{0, 2}, MeshCoordinate{1, 3});
    const auto submesh_range_3 = MeshCoordinateRange(MeshShape{1, 4});
    auto submeshes = mesh_device_->create_overlapped_submeshes({submesh_range_1, submesh_range_2, submesh_range_3});
    ASSERT_EQ(submeshes.size(), 3);
    EXPECT_EQ(submeshes[0]->shape(), (MeshShape{2, 2}));
    EXPECT_EQ(submeshes[1]->shape(), (MeshShape{2, 2}));
    EXPECT_EQ(submeshes[2]->shape(), (MeshShape{1, 4}));

    // Test buffer allocation behavior with overlapped submeshes
    // Submesh dependencies: submesh1 and submesh2 don't overlap directly but both overlap with submesh3
    // Expected: submesh1 and submesh2 should have independent allocations, but submesh3 should respect both

    // Create buffer configuration for testing
    const DeviceAddr buffer_size = 4096;  // 4KB buffer
    const DeviceAddr page_size = 4096;    // 1KB page size
    using namespace tt::tt_metal::distributed;

    // Buffer configuration for testing
    ReplicatedBufferConfig replicated_config{buffer_size};
    // DRAM issues:
    // - BankManager doesn't take in Alloc Deps
    // - APIs don't take in Alloc id
    // - Everything is using default, so everything independent
    DeviceLocalBufferConfig l1_device_config{
        .page_size = page_size, .buffer_type = BufferType::L1, .sharding_args = std::nullopt, .bottom_up = false};

    // Allocate a buffer in submesh1
    auto buffer1 = MeshBuffer::create(replicated_config, l1_device_config, submeshes[0].get());
    EXPECT_TRUE(buffer1->is_allocated());
    DeviceAddr addr1 = buffer1->address();

    // Allocate a buffer in submesh2 and check that address is same as submesh1
    // NOTE: This should be the same because they don't overlap and should use independent allocators
    auto buffer2 = MeshBuffer::create(replicated_config, l1_device_config, submeshes[1].get());
    EXPECT_TRUE(buffer2->is_allocated());
    DeviceAddr addr2 = buffer2->address();

    // Since submesh1 and submesh2 don't overlap, they should have independent allocations
    // and should get the same starting address (both start from beginning of their allocators)
    EXPECT_EQ(addr2, addr1) << "Submesh1 and submesh2 should have same starting address (independent allocators)";

    // Allocate another buffer in submesh2 and check that it continues after the previous buffer
    auto buffer2_next = MeshBuffer::create(replicated_config, l1_device_config, submeshes[1].get());
    EXPECT_TRUE(buffer2_next->is_allocated());
    DeviceAddr addr2_next = buffer2_next->address();
    EXPECT_EQ(addr2_next, addr2 - buffer_size) << "Second buffer in submesh2 should continue after first buffer";

    // Allocate a buffer in submesh3 and check that it's in the same address as after allocations in submesh2
    // Submesh3 overlaps with both submesh1 and submesh2, so it should respect both allocators
    auto buffer3 = MeshBuffer::create(replicated_config, l1_device_config, submeshes[2].get());
    EXPECT_TRUE(buffer3->is_allocated());
    DeviceAddr addr3 = buffer3->address();

    // Submesh3 should respect both submesh1 and submesh2 allocations
    // Since both submesh1 and submesh2 have allocated buffers, submesh3 should start after the maximum
    EXPECT_EQ(addr3, addr2_next - buffer_size) << "Submesh3 should respect both submesh1 and submesh2 allocations";

    // Do the same in DRAM buffer
    DeviceLocalBufferConfig dram_device_config{
        .page_size = page_size, .buffer_type = BufferType::DRAM, .sharding_args = std::nullopt, .bottom_up = true};

    auto buffer1_dram = MeshBuffer::create(replicated_config, dram_device_config, submeshes[0].get());
    EXPECT_TRUE(buffer1_dram->is_allocated());
    DeviceAddr addr1_dram = buffer1_dram->address();

    auto buffer2_dram = MeshBuffer::create(replicated_config, dram_device_config, submeshes[1].get());
    EXPECT_TRUE(buffer2_dram->is_allocated());
    DeviceAddr addr2_dram = buffer2_dram->address();

    EXPECT_EQ(addr2_dram, addr1_dram)
        << "Submesh1 and submesh2 should have same starting address (independent allocators)";

    auto buffer2_next_dram = MeshBuffer::create(replicated_config, dram_device_config, submeshes[1].get());
    EXPECT_TRUE(buffer2_next_dram->is_allocated());
    DeviceAddr addr2_next_dram = buffer2_next_dram->address();
    EXPECT_EQ(addr2_next_dram, addr2_dram + buffer_size)
        << "Second buffer in submesh2 should continue after first buffer";

    auto buffer3_dram = MeshBuffer::create(replicated_config, dram_device_config, submeshes[2].get());
    EXPECT_TRUE(buffer3_dram->is_allocated());
    DeviceAddr addr3_dram = buffer3_dram->address();

    EXPECT_EQ(addr3_dram, addr2_next_dram + buffer_size)
        << "Submesh3 should respect both submesh1 and submesh2 allocations";
}

}  // namespace
}  // namespace tt::tt_metal::distributed
