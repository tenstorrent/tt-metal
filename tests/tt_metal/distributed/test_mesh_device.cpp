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
#include <tt-metalium/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include "context/metal_context.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
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

// Test to ensure writing from 16B aligned L1 address to 16B aligned pinned memory works using MeshDevice
TEST_F(MeshDevice2x4Test, MeshL1ToPinnedMemoryAt16BAlignedAddress) {
    // Use first device from the mesh for this test
    MeshCoordinate target_coord(0, 0);
    IDevice* device = mesh_device_->get_device(target_coord);
    EXPECT_TRUE(device->is_mmio_capable());

    CoreCoord logical_core(0, 0);

    uint32_t base_l1_src_address = device->allocator()->get_base_allocator_addr(HalMemType::L1) +
                                   MetalContext::instance().hal().get_alignment(HalMemType::L1);

    uint32_t size_bytes = 2048 * 128;
    std::vector<uint32_t> src =
        tt::test_utils::generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, size_bytes / sizeof(uint32_t));
    EXPECT_EQ(MetalContext::instance().hal().get_alignment(HalMemType::L1), 16);
    uint32_t num_16b_writes = size_bytes / MetalContext::instance().hal().get_alignment(HalMemType::L1);

    // Allocate and pin host memory
    auto host_buffer = std::make_shared<std::vector<uint32_t>>(size_bytes / sizeof(uint32_t), 0);
    tt::tt_metal::HostBuffer host_buffer_view(host_buffer);
    auto coordinate_range_set = MeshCoordinateRangeSet(MeshCoordinateRange(target_coord, target_coord));
    auto pinned_memory = mesh_device_->pin_memory(
        coordinate_range_set,
        host_buffer_view,
        true  // map_to_noc
    );

    // Get the pinned memory address that the device can write to
    uint64_t pinned_memory_device_addr = pinned_memory->get_device_addr(device->id());

    // Write source data to L1
    tt_metal::detail::WriteToDeviceL1(device, logical_core, base_l1_src_address, src);

    // Create program and kernel for mesh workload
    tt_metal::Program program = tt_metal::CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/pcie_write_16b.cpp",
        logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_0_default,
            .compile_args = {base_l1_src_address, (uint32_t)pinned_memory_device_addr, num_16b_writes}});

    // Create mesh workload and add program
    MeshWorkload mesh_workload = CreateMeshWorkload();
    MeshCoordinateRange device_range(target_coord, target_coord);
    AddProgramToMeshWorkload(mesh_workload, std::move(program), device_range);

    // Launch workload using mesh command queue
    auto& mesh_cq = mesh_device_->mesh_command_queue();
    EnqueueMeshWorkload(mesh_cq, mesh_workload, true);  // blocking = true

    // Verify the data was written correctly to pinned memory
    EXPECT_EQ(src, *host_buffer);
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

}  // namespace
}  // namespace tt::tt_metal::distributed
