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

    EXPECT_EQ(plan.a2a_tiles, 456);
    EXPECT_EQ(plan.weight_tiles_per_block, 4);
    EXPECT_EQ(plan.weight_pipeline_slots, 1);
}

TEST(MoEComputeL1Plan, LargeIntermediateStaticAllocationStaysBelowTensorRange) {
    // Wormhole's current default CB allocation starts at 95,072. The lowest
    // sharded tensor in this primitive begins at 1,031,552 with the campaign's
    // l1_small_size, so the ranges must remain disjoint.
    constexpr uint32_t circular_buffer_base = 95'072;
    constexpr uint32_t lowest_l1_tensor_address = 1'031'552;
    constexpr auto compact = plan_moe_compute_l1(
        /*intermediate_tiles=*/14336 / 32, /*ring_cores=*/12, /*weight_tile_bytes=*/576, false);
    constexpr uint32_t old_fast_path_bytes =
        compact.a2a_tiles * 2048 + /*weight tiles=*/28 * /*pipeline slots=*/3 * 576 + /*bookkeeping=*/64;

    EXPECT_LT(circular_buffer_base + compact.matmul_static_bytes, lowest_l1_tensor_address);
    EXPECT_GE(circular_buffer_base + old_fast_path_bytes, lowest_l1_tensor_address);
}

TEST(MoEComputeL1Plan, NormalIntermediatePreservesTripleBuffering) {
    constexpr auto plan = plan_moe_compute_l1(
        /*intermediate_tiles=*/2048 / 32, /*ring_cores=*/12, /*weight_tile_bytes=*/576, false);

    EXPECT_EQ(plan.weight_tiles_per_block, 28);
    EXPECT_EQ(plan.weight_pipeline_slots, 3);
}

TEST(MoEComputeL1Plan, LargeIntermediateAccountsForBfloat16WeightTiles) {
    constexpr uint32_t circular_buffer_base = 95'072;
    constexpr uint32_t lowest_l1_tensor_with_default_small_bank = 1'031'552;
    constexpr uint32_t lowest_l1_tensor_without_small_bank = 1'047'936;
    constexpr auto bfp4 = plan_moe_compute_l1(
        /*intermediate_tiles=*/14336 / 32, /*ring_cores=*/12, /*weight_tile_bytes=*/576, false);
    constexpr auto bf16 = plan_moe_compute_l1(
        /*intermediate_tiles=*/14336 / 32, /*ring_cores=*/12, /*weight_tile_bytes=*/2048, false);

    EXPECT_EQ(bf16.matmul_static_bytes - bfp4.matmul_static_bytes, 4 * (2048 - 576));
    EXPECT_EQ(bf16.weight_tiles_per_block, 4);
    EXPECT_EQ(bf16.weight_pipeline_slots, 1);
    EXPECT_GE(circular_buffer_base + bf16.matmul_static_bytes, lowest_l1_tensor_with_default_small_bank);
    EXPECT_LT(circular_buffer_base + bf16.matmul_static_bytes, lowest_l1_tensor_without_small_bank);
    EXPECT_EQ(moe_compute_l1_deficit(bf16, circular_buffer_base, lowest_l1_tensor_with_default_small_bank), 5'664);
    EXPECT_EQ(moe_compute_l1_deficit(bf16, circular_buffer_base, lowest_l1_tensor_without_small_bank), 0);
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
