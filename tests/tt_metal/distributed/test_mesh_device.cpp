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

using MeshDeviceTest = T3000MeshDeviceFixture;

TEST_F(MeshDeviceTest, SystemMeshTearDownWithoutClose) {
    auto& sys = SystemMesh::instance();

    const auto system_shape = sys.get_shape();
    ASSERT_EQ(system_shape.dims(), 2);
    EXPECT_EQ(system_shape[0], 2);
    EXPECT_EQ(system_shape[1], 4);
}

TEST_F(MeshDeviceTest, MemoryAllocationStatistics) {
    auto stats = mesh_device_->allocator()->get_statistics(tt::tt_metal::BufferType::DRAM);
    for (auto* device : mesh_device_->get_devices()) {
        auto device_stats = device->allocator()->get_statistics(tt::tt_metal::BufferType::DRAM);
        EXPECT_EQ(stats.total_allocatable_size_bytes, device_stats.total_allocatable_size_bytes);
    }
}

TEST_F(MeshDeviceTest, ViewIs2D) {
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

TEST_F(MeshDeviceTest, CreateSubmeshInvalidConfig) {
    EXPECT_EQ(mesh_device_->shape(), MeshShape(2, 4));

    EXPECT_ANY_THROW(mesh_device_->create_submesh(MeshShape{1, 3}, MeshCoordinate{1}));
    EXPECT_ANY_THROW(mesh_device_->create_submesh(MeshShape{0, 3}, MeshCoordinate{0, 0}));
    EXPECT_ANY_THROW(mesh_device_->create_submesh(MeshShape{2, 4}, MeshCoordinate{1, 1}));
    EXPECT_ANY_THROW(mesh_device_->create_submesh(MeshShape{2, 4, 1}, MeshCoordinate{0, 0}));
}

TEST_F(MeshDeviceTest, CreateSubmesh) {
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

TEST_F(MeshDeviceTest, CreateSubmeshesNonDivisibleSubshape) {
    EXPECT_EQ(mesh_device_->shape(), MeshShape(2, 4));
    EXPECT_ANY_THROW(mesh_device_->create_submeshes(MeshShape{1, 3}));
}

TEST_F(MeshDeviceTest, CreateSubmeshes) {
    EXPECT_EQ(mesh_device_->shape(), MeshShape(2, 4));

    auto submeshes = mesh_device_->create_submeshes(MeshShape{1, 2});
    EXPECT_THAT(submeshes, SizeIs(4));
    for (const auto& submesh : submeshes) {
        EXPECT_EQ(submesh->shape(), MeshShape(1, 2));
        EXPECT_THAT(submesh->get_devices(), SizeIs(2));
    }

    EXPECT_EQ(mesh_device_->get_submeshes(), submeshes);
}

TEST_F(MeshDeviceTest, MeshSockets) {
    auto md0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    socket_connection_t socket_connection = {
        .sender_core = {MeshCoordinate(0, 0), CoreCoord(0, 0)},
        .receiver_core = {MeshCoordinate(0, 0), CoreCoord(0, 1)},
    };

    socket_memory_config_t socket_mem_config = {
        .socket_type = BufferType::L1,
        .fifo_size = 1024,
    };

    socket_config_t socket_config = {
        .socket_connection_config = {socket_connection},
        .socket_mem_config = socket_mem_config,
    };
    auto [send_socket, recv_socket] = create_sockets(md0, md0, socket_config);

    auto socket_program = CreateProgram();
    auto socket_sender_kernel = CreateKernel(
        socket_program,
        "tests/tt_metal/distributed/socket_validation.cpp",
        {CoreCoord(0, 0)},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {send_socket.config_buffer->address()}});
    auto socket_receiver_kernel = CreateKernel(
        socket_program,
        "tests/tt_metal/distributed/recv_socket_validation.cpp",
        {CoreCoord(0, 1)},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {recv_socket.config_buffer->address()}});
    auto socket_mesh_workload = CreateMeshWorkload();
    AddProgramToMeshWorkload(
        socket_mesh_workload,
        std::move(socket_program),
        MeshCoordinateRange(MeshCoordinate{0, 0}, MeshCoordinate{0, 0}));
    EnqueueMeshWorkload(md0->mesh_command_queue(), socket_mesh_workload, true);
}

}  // namespace
}  // namespace tt::tt_metal::distributed
