// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <optional>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/vocabulary/node_coord.hpp>
#include "tt_metal/impl/allocator/allocation_request.hpp"

namespace tt::tt_metal::allocator {
namespace {

TEST(AllocationRequestTest, ComputeBankRequest) {
    const CoreRangeSet compute_grid{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}};  // 64 nodes
    // Interleaved spreads pages over every bank: 10 pages of 1000 B over 4 banks are 3 pages per
    // bank, each rounded up to 1024 B
    const auto interleaved =
        compute_bank_request(BufferType::DRAM, 10'000, 1000, 4, 1024, 0, compute_grid, std::nullopt);
    EXPECT_EQ(interleaved.num_banks, 4);
    EXPECT_EQ(interleaved.size_per_bank, 3 * 1024);
    EXPECT_EQ(interleaved.address_limit, 0);
    // Interleaved L1 is limited to the interleaved address limit, which must be set
    const auto interleaved_l1 =
        compute_bank_request(BufferType::L1, 4096, 1024, 64, 16, 0x20000, compute_grid, std::nullopt);
    EXPECT_EQ(interleaved_l1.address_limit, 0x20000);
    EXPECT_ANY_THROW(compute_bank_request(BufferType::L1, 4096, 1024, 64, 16, 0, compute_grid, std::nullopt));
    // Sharded spreads over the shards and is not limited
    const auto sharded = compute_bank_request(BufferType::L1, 8 * 2048, 2048, 64, 16, 0x20000, compute_grid, 8);
    EXPECT_EQ(sharded.num_banks, 8);
    EXPECT_EQ(sharded.size_per_bank, 2048);
    EXPECT_EQ(sharded.address_limit, 0);
    // More shards than compute grid nodes throw
    EXPECT_ANY_THROW(compute_bank_request(BufferType::L1, 65 * 2048, 2048, 64, 16, 0x20000, compute_grid, 65));
}

using NodeCoords = std::vector<NodeCoord>;

const CoreRangeSet square_grid{CoreRange{CoreCoord{0, 0}, CoreCoord{1, 1}}};

ShardSpecBuffer square_shard_spec(ShardOrientation orientation) {
    return ShardSpecBuffer{square_grid, {32, 32}, orientation, {32, 32}, {2, 2}};
}

TEST(AllocationRequestTest, PerNodeAllocationNodes) {
    // Every shard grid node, in shard orientation order
    EXPECT_EQ(
        per_node_allocation_nodes(square_shard_spec(ShardOrientation::ROW_MAJOR)),
        (NodeCoords{{0, 0}, {1, 0}, {0, 1}, {1, 1}}));
    EXPECT_EQ(
        per_node_allocation_nodes(square_shard_spec(ShardOrientation::COL_MAJOR)),
        (NodeCoords{{0, 0}, {0, 1}, {1, 0}, {1, 1}}));
    // No shard spec, or an empty grid, throws
    EXPECT_ANY_THROW(per_node_allocation_nodes(std::nullopt));
    EXPECT_ANY_THROW(per_node_allocation_nodes(
        ShardSpecBuffer{CoreRangeSet{}, {32, 32}, ShardOrientation::ROW_MAJOR, {32, 32}, {2, 2}}));
}

TEST(AllocationRequestTest, RangeLockstepAllocationNodes) {
    // 2 shards over the 4 nodes of the grid only put data on the first 2
    const BufferDistributionSpec half_full{Shape{1, 2}, Shape{1, 1}, square_grid, ShardOrientation::COL_MAJOR};
    EXPECT_EQ(range_lockstep_allocation_nodes(half_full, std::nullopt), (NodeCoords{{0, 0}, {0, 1}}));
    // The distribution spec wins over the shard spec
    EXPECT_EQ(
        range_lockstep_allocation_nodes(half_full, square_shard_spec(ShardOrientation::ROW_MAJOR)),
        (NodeCoords{{0, 0}, {0, 1}}));
    // Without a distribution spec, every shard grid node
    EXPECT_EQ(
        range_lockstep_allocation_nodes(std::nullopt, square_shard_spec(ShardOrientation::ROW_MAJOR)),
        (NodeCoords{{0, 0}, {1, 0}, {0, 1}, {1, 1}}));
    // Neither spec throws
    EXPECT_ANY_THROW(range_lockstep_allocation_nodes(std::nullopt, std::nullopt));
}

}  // namespace
}  // namespace tt::tt_metal::allocator
