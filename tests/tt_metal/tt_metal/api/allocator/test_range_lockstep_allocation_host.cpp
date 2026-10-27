// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side tests for experimental::range_lockstep_allocation: the guards on the two opt-in
// surfaces, and whether the flag survives the conversions between a MemoryConfig and the
// sharding args the allocator sees. Nothing here opens a device.
//
// The allocation behaviour itself lives in test_range_lockstep_allocation.cpp, which needs one.

#include <optional>
#include <vector>
#include <gtest/gtest.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/memory_config.hpp>
#include <tt-metalium/experimental/range_lockstep_allocation/buffer.hpp>
#include <tt-metalium/experimental/range_lockstep_allocation/memory_config.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>

namespace tt::tt_metal {

namespace per_core = experimental::per_core_allocation;
namespace range_lockstep = experimental::range_lockstep_allocation;

namespace {
BufferShardingArgs shard_spec_args(const CoreCoord& core) {
    return BufferShardingArgs(
        ShardSpecBuffer(CoreRangeSet(core), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1}),
        TensorMemoryLayout::HEIGHT_SHARDED);
}

}  // namespace

namespace {
MemoryConfig sharded_l1_memory_config() {
    return MemoryConfig(
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec(CoreRangeSet(CoreCoord(0, 0)), {32, 32}, ShardOrientation::ROW_MAJOR));
}
}  // namespace

TEST(RangeLockstepAllocationTest, RangeLockstepRejectsPerCoreAllocation) {
    auto args = shard_spec_args(CoreCoord(0, 0));
    per_core::set_per_core_allocation(args, true);
    EXPECT_ANY_THROW(range_lockstep::set_range_lockstep_allocation(args, true))
        << "A buffer cannot both take one address across its cores and an independent address on each";
}

TEST(RangeLockstepAllocationTest, RangeLockstepRejectsInterleaved) {
    BufferShardingArgs interleaved;
    EXPECT_ANY_THROW(range_lockstep::set_range_lockstep_allocation(interleaved, true))
        << "An interleaved buffer spans every bank, so there is no narrower core set to scope to";
}

// Both setters have to reject the other flag, or the pair is only mutually exclusive in one call
// order. Setting per-core last is the dangerous direction: allocate_buffer tests per_core_allocation
// first and returns from that branch, so range lockstep would be dropped without a word.
TEST(RangeLockstepAllocationTest, RangeLockstepAndPerCoreExcludeEachOtherInBothOrders) {
    auto args_for = [](const CoreCoord& core) {
        return BufferShardingArgs(
            ShardSpecBuffer(CoreRangeSet(core), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1}),
            TensorMemoryLayout::HEIGHT_SHARDED);
    };
    const CoreCoord core(0, 0);

    auto per_core_first = args_for(core);
    per_core::set_per_core_allocation(per_core_first, true);
    EXPECT_ANY_THROW(range_lockstep::set_range_lockstep_allocation(per_core_first, true));

    auto range_lockstep_first = args_for(core);
    range_lockstep::set_range_lockstep_allocation(range_lockstep_first, true);
    EXPECT_ANY_THROW(per_core::set_per_core_allocation(range_lockstep_first, true));
}

TEST(RangeLockstepAllocationTest, RangeLockstepMemoryConfigReachesShardingArgs) {
    auto config = sharded_l1_memory_config();
    ASSERT_FALSE(range_lockstep::is_range_lockstep_allocation(config));
    range_lockstep::set_range_lockstep_allocation(config, true);
    EXPECT_TRUE(range_lockstep::is_range_lockstep_allocation(config));

    // TensorSpec is what turns a MemoryConfig into the BufferShardingArgs the allocator sees.
    const TensorSpec spec(Shape({32, 32}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), config));
    const auto sharding_args = spec.compute_buffer_sharding_args();
    EXPECT_TRUE(range_lockstep::is_range_lockstep_allocation(sharding_args))
        << "range lockstep set on the MemoryConfig did not reach the buffer's sharding args";
}

TEST(RangeLockstepAllocationTest, RangeLockstepSurvivesNdShardSpecConversion) {
    // TensorSpec rebuilds the MemoryConfig from named fields when it converts an nd shard spec to
    // a legacy one, and a rebuild drops the experimental flags unless they are explicitly restored.
    // Nothing on that path reports the loss: the buffer just reverts to the chip-wide scan. Whether
    // the conversion runs at all is shape-dependent, so this uses a shape it succeeds for.
    //
    // per_core_allocation cannot reach this path -- it refuses an NdShardSpec outright -- so this
    // flag is the only one for which the restore is live.
    const CoreRangeSet grid(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));
    MemoryConfig config(BufferType::L1, NdShardSpec{.shard_shape = Shape({1, 32}), .grid = grid});
    range_lockstep::set_range_lockstep_allocation(config, true);

    const TensorSpec spec(Shape({2, 32}), TensorLayout(DataType::UINT8, PageConfig(Layout::ROW_MAJOR), config));
    EXPECT_TRUE(range_lockstep::is_range_lockstep_allocation(spec.memory_config()))
        << "the nd -> legacy shard spec conversion dropped range lockstep";
    EXPECT_TRUE(range_lockstep::is_range_lockstep_allocation(spec.compute_buffer_sharding_args()))
        << "range lockstep did not reach the buffer's sharding args from an nd shard spec";
}

TEST(RangeLockstepAllocationTest, RangeLockstepMemoryConfigRejectsPerCoreAndInterleaved) {
    auto per_core_config = sharded_l1_memory_config();
    per_core::set_per_core_allocation(per_core_config, true);
    EXPECT_ANY_THROW(range_lockstep::set_range_lockstep_allocation(per_core_config, true));

    MemoryConfig interleaved(TensorMemoryLayout::INTERLEAVED, BufferType::L1);
    EXPECT_ANY_THROW(range_lockstep::set_range_lockstep_allocation(interleaved, true));
}

}  // namespace tt::tt_metal
