// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/moe_compute_l1_plan.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/moe_compute.hpp"

namespace ttnn::experimental::prim::detail {
namespace {

TEST(MoEComputeL1Plan, LargeIntermediateUsesCompactWeightPipeline) {
    constexpr auto plan = plan_moe_compute_l1(
        /*intermediate_tiles=*/14336 / 32, /*ring_cores=*/12, /*weight_tile_bytes=*/576, false);

    EXPECT_EQ(plan.a2a_tiles_per_step, 38);
    EXPECT_EQ(plan.a2a_buffer_slots, 11);
    EXPECT_EQ(plan.a2a_tiles, 418);
    EXPECT_EQ(plan.weight_tiles_per_block, 4);
    EXPECT_EQ(plan.weight_pipeline_slots, 1);
}

TEST(MoEComputeL1Plan, RingBufferOmitsOnlyTheRedundantReturnSlot) {
    for (const uint32_t ring_cores : {8u, 12u, 16u}) {
        const auto plan = plan_moe_compute_l1(
            /*intermediate_tiles=*/14336 / 32, ring_cores, /*weight_tile_bytes=*/576, false);
        const uint32_t expected_tiles_per_step = ((14336 / 32 + ring_cores - 1) / ring_cores + 1) & ~1u;

        EXPECT_EQ(plan.a2a_tiles_per_step, expected_tiles_per_step);
        EXPECT_EQ(plan.a2a_buffer_slots, ring_cores - 1);
        EXPECT_EQ(plan.a2a_tiles, expected_tiles_per_step * (ring_cores - 1));
    }
}

TEST(MoEComputeL1Plan, LargeIntermediateStaticAllocationStaysBelowTensorRange) {
    // The live tokens=32 integrated matrix at l1_small_size=0 placed the
    // lowest L1 tensor at 1,029,760. Reusing slot zero for the final incoming
    // shard leaves ample space even at the current worker CB base.
    constexpr uint32_t circular_buffer_base = 111'360;
    constexpr uint32_t lowest_l1_tensor_address = 1'029'760;
    constexpr auto compact = plan_moe_compute_l1(
        /*intermediate_tiles=*/14336 / 32, /*ring_cores=*/12, /*weight_tile_bytes=*/576, false);
    constexpr uint32_t old_full_ring_bytes = compact.a2a_tiles_per_step * /*ring slots=*/12 * 2048 +
                                             compact.weight_tiles_per_block * compact.weight_pipeline_slots * 576 +
                                             /*bookkeeping=*/64;

    EXPECT_LT(circular_buffer_base + compact.matmul_static_bytes, lowest_l1_tensor_address);
    EXPECT_GE(circular_buffer_base + old_full_ring_bytes, lowest_l1_tensor_address);
}

TEST(MoEComputeL1Plan, NormalIntermediatePreservesTripleBuffering) {
    constexpr auto plan = plan_moe_compute_l1(
        /*intermediate_tiles=*/2048 / 32, /*ring_cores=*/12, /*weight_tile_bytes=*/576, false);

    EXPECT_EQ(plan.weight_tiles_per_block, 28);
    EXPECT_EQ(plan.weight_pipeline_slots, 3);
}

TEST(MoEComputeL1Plan, LargeIntermediateAccountsForBfloat16WeightTiles) {
    constexpr uint32_t circular_buffer_base = 111'360;
    constexpr uint32_t lowest_live_l1_tensor_address = 1'029'760;
    constexpr auto bfp4 = plan_moe_compute_l1(
        /*intermediate_tiles=*/14336 / 32, /*ring_cores=*/12, /*weight_tile_bytes=*/576, false);
    constexpr auto bfp8 = plan_moe_compute_l1(
        /*intermediate_tiles=*/14336 / 32, /*ring_cores=*/12, /*weight_tile_bytes=*/1088, false);
    constexpr auto bf16 = plan_moe_compute_l1(
        /*intermediate_tiles=*/14336 / 32, /*ring_cores=*/12, /*weight_tile_bytes=*/2048, false);

    EXPECT_EQ(bf16.matmul_static_bytes - bfp4.matmul_static_bytes, 4 * (2048 - 576));
    EXPECT_EQ(bf16.weight_tiles_per_block, 4);
    EXPECT_EQ(bf16.weight_pipeline_slots, 1);
    EXPECT_LT(circular_buffer_base + bf16.matmul_static_bytes, lowest_live_l1_tensor_address);
    EXPECT_EQ(moe_compute_l1_deficit(bfp4, circular_buffer_base, lowest_live_l1_tensor_address), 0);
    EXPECT_EQ(moe_compute_l1_deficit(bfp8, circular_buffer_base, lowest_live_l1_tensor_address), 0);
    EXPECT_EQ(moe_compute_l1_deficit(bf16, circular_buffer_base, lowest_live_l1_tensor_address), 0);
    EXPECT_EQ(lowest_live_l1_tensor_address - (circular_buffer_base + bf16.matmul_static_bytes), 54'080);
}

TEST(MoEComputeCapability, ReportsEveryValidatedWeightDtype) {
    const auto dtypes = ttnn::experimental::moe_compute_supported_weight_dtypes();

    ASSERT_EQ(dtypes.size(), 3);
    EXPECT_EQ(dtypes[0], ttnn::DataType::BFLOAT4_B);
    EXPECT_EQ(dtypes[1], ttnn::DataType::BFLOAT8_B);
    EXPECT_EQ(dtypes[2], ttnn::DataType::BFLOAT16);
}

}  // namespace
}  // namespace ttnn::experimental::prim::detail
