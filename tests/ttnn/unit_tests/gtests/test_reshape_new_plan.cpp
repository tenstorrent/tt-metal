// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only checks of the new reshape's planner (which kind of program a reshape needs), its static program
// parameters, and the mesh topology relabeling (#64322). No device is opened.

#include <gtest/gtest.h>

#include <cstdint>
#include <variant>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/tensor_layout_apis_with_custom_alignment.hpp>

#include "ttnn/operations/data_movement/reshape_new/reshape_new_plan.hpp"
#include "ttnn/operations/data_movement/reshape_new/reshape_new_topology.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::reshape_new::test {
namespace {

using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::ShardOrientation;
using ::tt::tt_metal::ShardSpec;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorMemoryLayout;
using ::tt::tt_metal::TensorSpec;
using ::tt::tt_metal::TensorTopology;
using ::tt::tt_metal::distributed::MeshCoordinate;
using ::tt::tt_metal::distributed::MeshMapperConfig;
using ::tt::tt_metal::distributed::MeshShape;

TensorSpec spec(
    const ttnn::Shape& shape,
    Layout layout,
    const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG,
    DataType dtype = DataType::BFLOAT16) {
    return TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), memory_config));
}

MemoryConfig height_sharded_l1(uint32_t cores, uint32_t shard_height, uint32_t shard_width) {
    return MemoryConfig(
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{cores - 1, 0})),
            {shard_height, shard_width},
            ShardOrientation::ROW_MAJOR));
}

TEST(ReshapeNewPlan, SameSpecIsNoOp) {
    const auto s = spec(ttnn::Shape{1, 2, 64, 64}, Layout::TILE);
    EXPECT_EQ(plan_reshape(s, s), ReshapeKind::NoOp);
}

TEST(ReshapeNewPlan, TileOuterDimsChangeIsView) {
    EXPECT_EQ(
        plan_reshape(spec(ttnn::Shape{1, 1, 64, 64}, Layout::TILE), spec(ttnn::Shape{64, 64}, Layout::TILE)),
        ReshapeKind::View);
    // Rows padded inside a tile stay in place when dim -2 is unchanged.
    EXPECT_EQ(
        plan_reshape(spec(ttnn::Shape{1, 4, 30, 64}, Layout::TILE), spec(ttnn::Shape{2, 2, 30, 64}, Layout::TILE)),
        ReshapeKind::View);
    // Merging dim -3 into dim -2 needs no data movement when no row is padded.
    EXPECT_EQ(
        plan_reshape(spec(ttnn::Shape{1, 2, 32, 64}, Layout::TILE), spec(ttnn::Shape{1, 1, 64, 64}, Layout::TILE)),
        ReshapeKind::View);
}

TEST(ReshapeNewPlan, TilePaddedRowsMergeIsGather) {
    EXPECT_EQ(
        plan_reshape(spec(ttnn::Shape{1, 2, 30, 64}, Layout::TILE), spec(ttnn::Shape{1, 1, 60, 64}, Layout::TILE)),
        ReshapeKind::Gather);
}

TEST(ReshapeNewPlan, LastDimChangeIsGather) {
    EXPECT_EQ(
        plan_reshape(spec(ttnn::Shape{64, 64}, Layout::TILE), spec(ttnn::Shape{32, 128}, Layout::TILE)),
        ReshapeKind::Gather);
    EXPECT_EQ(
        plan_reshape(spec(ttnn::Shape{4, 6}, Layout::ROW_MAJOR), spec(ttnn::Shape{2, 12}, Layout::ROW_MAJOR)),
        ReshapeKind::Gather);
}

TEST(ReshapeNewPlan, RowMajorOuterDimsChangeIsView) {
    EXPECT_EQ(
        plan_reshape(spec(ttnn::Shape{2, 3, 8}, Layout::ROW_MAJOR), spec(ttnn::Shape{6, 8}, Layout::ROW_MAJOR)),
        ReshapeKind::View);
}

TEST(ReshapeNewPlan, DistributionChangeIsPageCopy) {
    EXPECT_EQ(
        plan_reshape(
            spec(ttnn::Shape{1, 1, 64, 64}, Layout::TILE, ttnn::DRAM_MEMORY_CONFIG),
            spec(ttnn::Shape{64, 64}, Layout::TILE, ttnn::L1_MEMORY_CONFIG)),
        ReshapeKind::PageCopy);
    EXPECT_EQ(
        plan_reshape(
            spec(ttnn::Shape{1, 2, 32, 64}, Layout::TILE),
            spec(ttnn::Shape{1, 1, 64, 64}, Layout::TILE, height_sharded_l1(2, 32, 64))),
        ReshapeKind::PageCopy);
}

TEST(ReshapeNewPlan, SideGeometry) {
    const auto tiled = side_geometry(spec(ttnn::Shape{3, 2, 30, 70}, Layout::TILE));
    EXPECT_EQ(tiled.batch, 6u);
    EXPECT_EQ(tiled.height, 30u);
    EXPECT_EQ(tiled.padded_height, 32u);
    EXPECT_EQ(tiled.width, 70u);
    EXPECT_EQ(tiled.page_width, 32u);
    EXPECT_EQ(tiled.pages_per_row, 3u);
    EXPECT_EQ(tiled.num_pages, 6u * 3u);

    const auto row_major = side_geometry(spec(ttnn::Shape{3, 5, 7}, Layout::ROW_MAJOR));
    EXPECT_EQ(row_major.batch, 3u);
    EXPECT_EQ(row_major.width, 7u);
    EXPECT_EQ(row_major.page_width, 7u);
    EXPECT_EQ(row_major.pages_per_row, 1u);
    EXPECT_EQ(row_major.num_pages, 15u);
}

TEST(ReshapeNewPlan, PageCopyParams) {
    const auto in = spec(ttnn::Shape{1, 1, 64, 64}, Layout::TILE);
    const auto out = spec(ttnn::Shape{64, 64}, Layout::TILE, ttnn::L1_MEMORY_CONFIG);
    const auto params = make_page_copy_params(in, out, 32, 16);
    EXPECT_EQ(params.num_pages, 4u);
    EXPECT_EQ(params.page_bytes, 2048u);
    EXPECT_EQ(params.chunks_per_page, 1u);
    EXPECT_EQ(params.num_units, 4u);
    EXPECT_EQ(params.entry_bytes, 2048u);
    EXPECT_EQ(params.units_per_batch, 8u);
    EXPECT_EQ(params.num_entries, 16u);
}

TEST(ReshapeNewPlan, PageCopyParamsChunkWidePages) {
    // 20000 bf16 elements per row: 40000-byte pages, moved in 16 KiB chunks.
    const auto in = spec(ttnn::Shape{1, 3, 20000}, Layout::ROW_MAJOR);
    const auto out = spec(ttnn::Shape{3, 20000}, Layout::ROW_MAJOR, ttnn::L1_MEMORY_CONFIG);
    const auto params = make_page_copy_params(in, out, 64, 16);
    EXPECT_EQ(params.page_bytes, 40000u);
    EXPECT_EQ(params.chunk_bytes, 16384u);
    EXPECT_EQ(params.chunks_per_page, 3u);
    EXPECT_EQ(params.num_units, 9u);
    EXPECT_EQ(params.units_per_batch, 1u);
    EXPECT_EQ(params.num_entries, 2u);
}

TEST(ReshapeNewPlan, GatherParamsChunkWideRows) {
    const auto in = spec(ttnn::Shape{4, 4096}, Layout::ROW_MAJOR);
    const auto out = spec(ttnn::Shape{2, 8192}, Layout::ROW_MAJOR);
    const auto params = make_gather_params(in, out, 32);
    EXPECT_FALSE(params.tiled);
    EXPECT_EQ(params.element_size, 2u);
    EXPECT_EQ(params.page_bytes, 16384u);
    EXPECT_EQ(params.unit_bytes, 4096u);
    EXPECT_EQ(params.chunks_per_page, 4u);
    EXPECT_EQ(params.num_units, 8u);
    EXPECT_EQ(params.header_bytes % 16, 0u);
    EXPECT_GE(params.stage_data_bytes, params.unit_bytes);
    EXPECT_EQ(params.stage_entry_bytes, params.header_bytes + params.stage_data_bytes);
}

TEST(ReshapeNewPlan, GatherParamsTile) {
    const auto in = spec(ttnn::Shape{2, 30, 64}, Layout::TILE, ttnn::DRAM_MEMORY_CONFIG, DataType::FLOAT32);
    const auto out = spec(ttnn::Shape{60, 64}, Layout::TILE, ttnn::DRAM_MEMORY_CONFIG, DataType::FLOAT32);
    const auto params = make_gather_params(in, out, 64);
    EXPECT_TRUE(params.tiled);
    EXPECT_EQ(params.element_size, 4u);
    EXPECT_EQ(params.face_width, 16u);
    EXPECT_EQ(params.unit_bytes, 4096u);
    EXPECT_EQ(params.num_units, 4u);
    EXPECT_EQ(params.source_alignment, 64u);
}

TEST(ReshapeNewPlan, GatherRejectsBlockFloat) {
    const auto in = spec(ttnn::Shape{64, 64}, Layout::TILE, ttnn::DRAM_MEMORY_CONFIG, DataType::BFLOAT8_B);
    const auto out = spec(ttnn::Shape{32, 128}, Layout::TILE, ttnn::DRAM_MEMORY_CONFIG, DataType::BFLOAT8_B);
    EXPECT_ANY_THROW(make_gather_params(in, out, 32));
}

TEST(ReshapeNewPlan, PaddedOuterDimIsRejected) {
    const auto padded = TensorSpec(
        ttnn::Shape{2, 32, 32},
        ::tt::tt_metal::experimental::tensor_layout_from_padded_shape(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            ttnn::DRAM_MEMORY_CONFIG,
            ttnn::Shape{2, 32, 32},
            ttnn::Shape{3, 32, 32}));
    EXPECT_ANY_THROW(side_geometry(padded));
}

TensorTopology sharded_topology(int dim) {
    return TensorTopology(MeshShape{2}, {MeshMapperConfig::Shard{dim}}, {MeshCoordinate{0, 0}, MeshCoordinate{0, 1}});
}

int shard_dim(const TensorTopology& topology) {
    const auto& placements = topology.placements();
    if (placements.empty()) {
        return -1000;
    }
    const auto* shard = std::get_if<MeshMapperConfig::Shard>(&placements[0]);
    return shard == nullptr ? -1000 : shard->dim;
}

TEST(ReshapeNewTopology, ShardOnLastDimFollowsTheDim) {
    EXPECT_EQ(shard_dim(remap_topology(sharded_topology(3), ttnn::Shape{1, 1, 64, 64}, ttnn::Shape{64, 64})), 1);
    EXPECT_EQ(shard_dim(remap_topology(sharded_topology(-1), ttnn::Shape{1, 1, 64, 64}, ttnn::Shape{64, 64})), -1);
}

TEST(ReshapeNewTopology, ShardOnOuterDimFollowsTheDim) {
    // Issue #64322: [B, S, H] sharded on S, reshaped to [B, S, heads, head_dim], stays sharded on S.
    EXPECT_EQ(shard_dim(remap_topology(sharded_topology(1), ttnn::Shape{1, 128, 256}, ttnn::Shape{1, 128, 8, 32})), 1);
    EXPECT_EQ(shard_dim(remap_topology(sharded_topology(0), ttnn::Shape{64, 64}, ttnn::Shape{1, 1, 64, 64})), 2);
}

TEST(ReshapeNewTopology, ShardWithoutMatchingDimKeepsLabel) {
    // [2, 3, 4] -> [6, 4]: no output dim starts after the 2 x ... prefix of dim 1.
    EXPECT_EQ(shard_dim(remap_topology(sharded_topology(1), ttnn::Shape{2, 3, 4}, ttnn::Shape{6, 4})), 1);
}

TEST(ReshapeNewTopology, ReplicateIsUnchanged) {
    const auto replicated = TensorTopology::create_fully_replicated_tensor_topology(MeshShape{2});
    EXPECT_EQ(remap_topology(replicated, ttnn::Shape{4, 8}, ttnn::Shape{32}), replicated);
}

}  // namespace
}  // namespace ttnn::operations::data_movement::reshape_new::test
