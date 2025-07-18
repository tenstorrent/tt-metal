// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <impl/context/metal_context.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/host_buffer.hpp>

#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_metal::distributed {

using tt_fabric::HostRankId;
using tt_fabric::MeshId;
using tt_fabric::MeshScope;

TEST(BigMeshDualRankTestT3K, DistributedContext) {
    auto& dctx = MetalContext::instance().get_distributed_context();
    auto world_size = dctx.size();
    EXPECT_EQ(*world_size, 2);
}

TEST(BigMeshDualRankTestT3K, LocalRankBinding) {
    auto& dctx = MetalContext::instance().get_distributed_context();
    auto& control_plane = MetalContext::instance().get_control_plane();

    tt_fabric::HostRankId local_rank_binding = control_plane.get_local_host_rank_id_binding();
    if (*dctx.rank() == 0) {
        EXPECT_EQ(*local_rank_binding, 0);
    } else {
        EXPECT_EQ(*local_rank_binding, 1);
    }
}

TEST(BigMeshDualRankTestT3K, SystemMeshValidation) {
    EXPECT_NO_THROW({
        const auto& system_mesh = SystemMesh::instance();
        EXPECT_EQ(system_mesh.shape(), MeshShape(2,4));
        EXPECT_EQ(system_mesh.local_shape(), MeshShape(2,2));
    });
}

TEST(BigMeshDualRankTestT3K, MeshDevice2x4Validation) {
    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2,4)), DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER);
    EXPECT_EQ(mesh_device->shape(), MeshShape(2,4));
}

TEST(BigMeshDualRankTestT3K, SystemMeshShape) {
    const auto& system_mesh = SystemMesh::instance();
    EXPECT_EQ(system_mesh.local_shape(), MeshShape(2, 2));

    auto& control_plane = MetalContext::instance().get_control_plane();
    auto rank = control_plane.get_local_host_rank_id_binding();

    if (rank == HostRankId{0}) {
        EXPECT_NO_THROW(system_mesh.get_physical_device_id(MeshCoordinate(0, 0)));
        EXPECT_NO_THROW(system_mesh.get_physical_device_id(MeshCoordinate(0, 1)));
        EXPECT_NO_THROW(system_mesh.get_physical_device_id(MeshCoordinate(1, 0)));
        EXPECT_NO_THROW(system_mesh.get_physical_device_id(MeshCoordinate(1, 1)));
    } else {
        EXPECT_NO_THROW(system_mesh.get_physical_device_id(MeshCoordinate(0, 2)));
        EXPECT_NO_THROW(system_mesh.get_physical_device_id(MeshCoordinate(0, 3)));
        EXPECT_NO_THROW(system_mesh.get_physical_device_id(MeshCoordinate(1, 2)));
        EXPECT_NO_THROW(system_mesh.get_physical_device_id(MeshCoordinate(1, 3)));
    }
}

}  // namespace tt::tt_metal::distributed
