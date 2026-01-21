// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <impl/context/metal_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/distributed/mesh_device_impl.hpp"

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed {

using ::testing::ElementsAre;
using ::tt::tt_fabric::MeshHostRankId;
using ::tt::tt_fabric::MeshId;

// Parameterized test fixture for mesh device validation
class BigMeshDualRankMeshShapeSweepFixture : public MeshDeviceFixtureBase,
                                             public testing::WithParamInterface<MeshShape> {
public:
    BigMeshDualRankMeshShapeSweepFixture() :
        MeshDeviceFixtureBase(Config{
            .mesh_shape = GetParam(),
        }) {}
};

INSTANTIATE_TEST_SUITE_P(
    BigMeshDualRankMeshShapeSweep,
    BigMeshDualRankMeshShapeSweepFixture,
    ::testing::Values(
        MeshShape(2, 4),
        MeshShape(1, 1),
        MeshShape(1, 2),
        MeshShape(2, 1),
        MeshShape(2, 2),
        MeshShape(1, 8),
        MeshShape(8, 1)));

using BigMeshDualRankTest2x4 = MeshDevice2x4Fixture;

TEST(BigMeshDualRankTest, DistributedContext) {
    const auto& dctx = MetalContext::instance().global_distributed_context();
    EXPECT_EQ(dctx.size(), multihost::Size(2));
}

TEST(BigMeshDualRankTest, LocalRankBinding) {
    const auto& global_context = MetalContext::instance().global_distributed_context();
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

    const auto local_subcontext = control_plane.get_host_local_context();
    ASSERT_NE(local_subcontext, nullptr);
    EXPECT_EQ(local_subcontext->rank(), multihost::Rank(0));
    EXPECT_EQ(local_subcontext->size(), multihost::Size(1));
    EXPECT_NE(MetalContext::instance().global_distributed_context().id(), local_subcontext->id());
}

TEST_P(BigMeshDualRankMeshShapeSweepFixture, MeshDeviceValidation) { EXPECT_EQ(mesh_device_->shape(), GetParam()); }

TEST_F(BigMeshDualRankTest2x4, SystemMeshValidation) {
    ASSERT_NO_THROW({ SystemMesh::instance(); });

    const auto& system_mesh = SystemMesh::instance();
    EXPECT_EQ(system_mesh.local_shape(), MeshShape(2, 2));

    auto mapped_devices = system_mesh.get_mapped_devices(MeshShape(2, 4));
    const MeshContainer<MaybeRemote<int>> physical_device_ids(MeshShape(2, 4), std::move(mapped_devices.device_ids));
    const MeshContainer<tt::tt_fabric::FabricNodeId> fabric_node_ids(
        MeshShape(2, 4), std::move(mapped_devices.fabric_node_ids));

    // Generic local/remote validation
    int local_count = 0;
    int remote_count = 0;
    int col_counts[4] = {0, 0, 0, 0};

    for (uint32_t row = 0; row < 2; ++row) {
        for (uint32_t col = 0; col < 4; ++col) {
            MeshCoordinate coord(row, col);
            const auto& dev = physical_device_ids.at(coord);
            if (dev.is_local()) {
                local_count++;
                col_counts[col]++;
            } else if (dev.is_remote()) {
                remote_count++;
            } else {
                FAIL() << "Device at " << coord << " is neither local nor remote";
            }
        }
    }

    EXPECT_EQ(local_count, 4);
    EXPECT_EQ(remote_count, 4);

    EXPECT_THAT(col_counts, ::testing::AnyOf(ElementsAre(2, 2, 0, 0), ElementsAre(0, 0, 2, 2)));

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

TEST_F(BigMeshDualRankTest2x4, DistributedHostBuffer) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    DistributedHostBuffer host_buffer = DistributedHostBuffer::create(mesh_device_->get_view());
    auto rank = control_plane.get_local_host_rank_id_binding();

    for (const auto& coord : ::tt::tt_metal::distributed::MeshCoordinateRange(mesh_device_->get_view().shape())) {
        host_buffer.emplace_shard(coord, [rank]() { return HostBuffer(std::vector<int>(3, *rank)); });
    }

    auto validate_local_shards = [rank](const HostBuffer& buffer) {
        fmt::print(
            "Rank {}: {}\n", *rank, std::vector<int>(buffer.view_as<int>().begin(), buffer.view_as<int>().end()));
        auto span = buffer.view_as<int>();
        for (const auto& value : span) {
            EXPECT_EQ(value, *rank);
        }
    };

    host_buffer.apply(validate_local_shards);
}

TEST_F(BigMeshDualRankTest2x4, SimpleShardedBufferTest) {
    // Simple test with a 2x4 mesh, 64x128 buffer, 32x32 shards
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    Shape2D global_buffer_shape = {64, 128};
    Shape2D shard_shape = {32, 32};

    uint32_t global_buffer_size = global_buffer_shape.height() * global_buffer_shape.width() * sizeof(uint32_t);

    ShardedBufferConfig sharded_config{
        .global_size = global_buffer_size,
        .global_buffer_shape = global_buffer_shape,
        .shard_shape = shard_shape,
        .shard_orientation = ShardOrientation::ROW_MAJOR,
    };

    auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());

    // Create input data
    std::vector<uint32_t> src_vec(global_buffer_shape.height() * global_buffer_shape.width(), 0);
    std::iota(src_vec.begin(), src_vec.end(), 0);

    // Write and read back
    EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), mesh_buffer, src_vec);
    std::vector<uint32_t> dst_vec;
    EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), dst_vec, mesh_buffer);

    // The expectation is that EnqueueWriteMeshBuffer/EnqueueReadMeshBuffer
    // should handle sharding/unsharding transparently, so dst should equal src
    for (int i = 0; i < dst_vec.size(); i++) {
        auto shard_row = i / global_buffer_shape.width();
        auto shard_col = i % global_buffer_shape.width();
        auto device_row = shard_row / shard_shape.height();
        auto device_col = shard_col / shard_shape.width();
        if (mesh_device_->impl().is_local(MeshCoordinate(device_row, device_col))) {
            EXPECT_EQ(dst_vec[i], src_vec[i]) << "Mismatch at index: " << i;
        }
    }
}

TEST_F(BigMeshDualRankTest2x4, SubmeshCreationSingleSubmesh) {
    auto submesh = mesh_device_->create_submesh(MeshShape(2, 2));
    ASSERT_NE(submesh, nullptr);
    EXPECT_EQ(submesh->shape(), MeshShape(2, 2));
}

TEST_F(BigMeshDualRankTest2x4, SubmeshCreationMultipleSubmeshes) {
    auto submeshes = mesh_device_->create_submeshes(MeshShape(2, 2));
    EXPECT_EQ(submeshes.size(), 2);
    for (const auto& submesh : submeshes) {
        ASSERT_NE(submesh, nullptr);
        EXPECT_EQ(submesh->shape(), MeshShape(2, 2));
    }
}

}  // namespace tt::tt_metal::distributed
