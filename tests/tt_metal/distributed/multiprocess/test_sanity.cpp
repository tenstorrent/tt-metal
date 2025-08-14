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

using ::testing::ElementsAre;
using ::tt::tt_fabric::MeshHostRankId;
using ::tt::tt_fabric::MeshId;
using ::tt::tt_fabric::MeshScope;

TEST(BigMeshDualRankTest2x4, DistributedContext) {
    auto& dctx = MetalContext::instance().global_distributed_context();
    EXPECT_EQ(dctx.size(), multihost::Size(2));
}

TEST(BigMeshDualRankTest2x4, LocalRankBinding) {
    auto& global_context = MetalContext::instance().global_distributed_context();
    auto& control_plane = MetalContext::instance().get_control_plane();

    tt_fabric::MeshHostRankId local_rank_binding = control_plane.get_local_host_rank_id_binding();
    if (global_context.rank() == multihost::Rank(0)) {
        EXPECT_EQ(local_rank_binding, MeshHostRankId(0));
    } else {
        EXPECT_EQ(local_rank_binding, MeshHostRankId(1));
    }

    const auto local_mesh_ids = control_plane.get_local_mesh_id_bindings();
    ASSERT_THAT(local_mesh_ids, ElementsAre(MeshId(0)));

    if (*global_context.rank() == 0) {
        EXPECT_EQ(control_plane.get_local_mesh_offset(), MeshCoordinate(0, 0));
    } else {
        EXPECT_EQ(control_plane.get_local_mesh_offset(), MeshCoordinate(0, 2));
    }

    const auto mesh_subcontext = control_plane.get_distributed_context(MeshId(0));
    ASSERT_NE(mesh_subcontext, nullptr);
    EXPECT_EQ(mesh_subcontext->size(), multihost::Size(2));

    std::array original_ranks = {0, 1};
    std::array translated_ranks = {-1, -1};
    mesh_subcontext->translate_ranks_to_other_ctx(
        original_ranks,
        tt::tt_metal::distributed::multihost::DistributedContext::get_current_world(),
        translated_ranks);

    EXPECT_THAT(translated_ranks, ElementsAre(0, 1));
    EXPECT_NE(MetalContext::instance().global_distributed_context().id(), mesh_subcontext->id());
}

TEST(BigMeshDualRankTest2x4, SystemMeshValidation) {
    ASSERT_NO_THROW({ SystemMesh::instance(); });

    const auto& system_mesh = SystemMesh::instance();
    EXPECT_EQ(system_mesh.local_shape(), MeshShape(2, 2));

    auto& control_plane = MetalContext::instance().get_control_plane();
    auto rank = control_plane.get_local_host_rank_id_binding();

    auto mapped_devices = system_mesh.get_mapped_devices(MeshShape(2, 4));
    const MeshContainer<MaybeRemote<int>> physical_device_ids(MeshShape(2, 4), std::move(mapped_devices.device_ids));
    const MeshContainer<tt::tt_fabric::FabricNodeId> fabric_node_ids(
        MeshShape(2, 4), std::move(mapped_devices.fabric_node_ids));
    if (rank == MeshHostRankId{0}) {
        EXPECT_TRUE(physical_device_ids.at(MeshCoordinate(0, 0)).is_local());
        EXPECT_TRUE(physical_device_ids.at(MeshCoordinate(0, 1)).is_local());
        EXPECT_TRUE(physical_device_ids.at(MeshCoordinate(1, 0)).is_local());
        EXPECT_TRUE(physical_device_ids.at(MeshCoordinate(1, 1)).is_local());
        EXPECT_TRUE(physical_device_ids.at(MeshCoordinate(0, 2)).is_remote());
        EXPECT_TRUE(physical_device_ids.at(MeshCoordinate(0, 3)).is_remote());
        EXPECT_TRUE(physical_device_ids.at(MeshCoordinate(1, 2)).is_remote());
        EXPECT_TRUE(physical_device_ids.at(MeshCoordinate(1, 3)).is_remote());
    } else {
        EXPECT_TRUE(physical_device_ids.at(MeshCoordinate(0, 0)).is_remote());
        EXPECT_TRUE(physical_device_ids.at(MeshCoordinate(0, 1)).is_remote());
        EXPECT_TRUE(physical_device_ids.at(MeshCoordinate(1, 0)).is_remote());
        EXPECT_TRUE(physical_device_ids.at(MeshCoordinate(1, 1)).is_remote());
        EXPECT_TRUE(physical_device_ids.at(MeshCoordinate(0, 2)).is_local());
        EXPECT_TRUE(physical_device_ids.at(MeshCoordinate(0, 3)).is_local());
        EXPECT_TRUE(physical_device_ids.at(MeshCoordinate(1, 2)).is_local());
        EXPECT_TRUE(physical_device_ids.at(MeshCoordinate(1, 3)).is_local());
    }

    // Check fabric node IDs are set for all devices, globally.
    EXPECT_EQ(fabric_node_ids.at(MeshCoordinate(0, 0)).chip_id, 0);
    EXPECT_EQ(fabric_node_ids.at(MeshCoordinate(0, 1)).chip_id, 1);
    EXPECT_EQ(fabric_node_ids.at(MeshCoordinate(0, 2)).chip_id, 2);
    EXPECT_EQ(fabric_node_ids.at(MeshCoordinate(0, 3)).chip_id, 3);
    EXPECT_EQ(fabric_node_ids.at(MeshCoordinate(1, 0)).chip_id, 4);
    EXPECT_EQ(fabric_node_ids.at(MeshCoordinate(1, 1)).chip_id, 5);
    EXPECT_EQ(fabric_node_ids.at(MeshCoordinate(1, 2)).chip_id, 6);
    EXPECT_EQ(fabric_node_ids.at(MeshCoordinate(1, 3)).chip_id, 7);
}

TEST(BigMeshDualRankTest2x4, MeshDevice2x4Validation) {
    auto mesh_device = MeshDevice::create(
        MeshDeviceConfig(MeshShape(2, 4)),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        tt::tt_metal::DispatchCoreType::WORKER);
    EXPECT_EQ(mesh_device->shape(), MeshShape(2, 4));
    EXPECT_EQ(mesh_device->get_view().mesh_id(), MeshId(0));
}

}  // namespace tt::tt_metal::distributed
