// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Tests for TensorSpec validation of sharding configurations.
//
// These tests verify that TensorSpec correctly rejects illegal shard specifications
// with appropriate error messages. Specifically, it validates:
// - HEIGHT_SHARDED: Shard width must match tensor physical width, and enough cores must be available
// - WIDTH_SHARDED: Shard height must match tensor physical height, and enough cores must be available
//
// These are pure validation tests that don't require device access.

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <functional>
#include <string>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>

using namespace tt::tt_metal;

const CoreCoord grid_size{8, 7};

struct IllegalShardSpecParams {
    Shape shape;
    PageConfig page_config;
    MemoryConfig memory_config;
    std::string expected_err_msg;
};

class IllegalTensorSpecCreationTests : public ::testing::TestWithParam<IllegalShardSpecParams> {};

TEST_P(IllegalTensorSpecCreationTests, ExpectFailAndCheckErrMsg) {
    const auto& params = GetParam();

    auto tensor_layout = TensorLayout(DataType::BFLOAT16, params.page_config, params.memory_config);
    EXPECT_THAT(
        std::function<void()>(
            [&params, &tensor_layout]() { auto tensor_spec = TensorSpec(params.shape, tensor_layout); }),
        ThrowsMessage<std::runtime_error>(::testing::HasSubstr(params.expected_err_msg)));
}

INSTANTIATE_TEST_SUITE_P(
    TensorShardingTests,
    IllegalTensorSpecCreationTests,
    // clang-format off
    ::testing::Values(
        // HEIGHT sharded: Not enough cores
        IllegalShardSpecParams{
            .shape = Shape{100, 16},
            .page_config = PageConfig(Layout::TILE, Tile({32, 16})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::HEIGHT_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            num_cores_to_corerangeset(3, grid_size, /*row_wise=*/true),
                            {32, 16},
                            ShardOrientation::ROW_MAJOR,
                        }
                },
            .expected_err_msg = "Number of shards along height 4 must not exceed number of cores 3"
        },
        // HEIGHT sharded: Shard width does not match
        IllegalShardSpecParams{
            .shape = Shape{100, 20},
            .page_config = PageConfig(Layout::TILE, Tile({32, 16})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::HEIGHT_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            num_cores_to_corerangeset(10, grid_size, /*row_wise=*/true),
                            {32, 16},
                            ShardOrientation::ROW_MAJOR,
                        }
                },
            .expected_err_msg = "Shard width 16 must match physical width 32 for height sharded"
        },
        // WIDTH sharded: Not enough cores
        IllegalShardSpecParams{
            .shape = Shape{16, 100},
            .page_config = PageConfig(Layout::TILE, Tile({16, 32})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::WIDTH_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            num_cores_to_corerangeset(3, grid_size, /*row_wise=*/true),
                            {16, 32},
                            ShardOrientation::ROW_MAJOR,
                        }
                },
            .expected_err_msg = "Number of shards along width 4 must not exceed number of cores 3"
        },
        // WIDTH sharded: Shard height does not match
        IllegalShardSpecParams{
            .shape = Shape{20, 100},
            .page_config = PageConfig(Layout::TILE, Tile({16, 32})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::WIDTH_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            num_cores_to_corerangeset(10, grid_size, /*row_wise=*/true),
                            {16, 32},
                            ShardOrientation::ROW_MAJOR,
                        }
                },
            .expected_err_msg = "Shard height 16 must match physical height 32 for width sharded"
        }
    )  // Values
    // clang-format on
);

class MemoryConfigEqualityTest : public ::testing::Test {
protected:
    CoreRangeSet core_grid_ = num_cores_to_corerangeset(4, grid_size, true);
    ShardSpec shard_spec_a_{core_grid_, {32, 32}, ShardOrientation::ROW_MAJOR};
    ShardSpec shard_spec_b_{core_grid_, {64, 64}, ShardOrientation::ROW_MAJOR};
    NdShardSpec nd_shard_spec_a_{Shape{32, 32}, core_grid_, ShardOrientation::ROW_MAJOR};
    NdShardSpec nd_shard_spec_b_{Shape{64, 64}, core_grid_, ShardOrientation::ROW_MAJOR};
};

TEST_F(MemoryConfigEqualityTest, DefaultInterleaved) {
    MemoryConfig a;
    MemoryConfig b;
    EXPECT_EQ(a, b);
}

TEST_F(MemoryConfigEqualityTest, DifferentBufferType) {
    MemoryConfig dram(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    MemoryConfig l1(TensorMemoryLayout::INTERLEAVED, BufferType::L1);
    EXPECT_NE(dram, l1);
}

TEST_F(MemoryConfigEqualityTest, DifferentMemoryLayout) {
    MemoryConfig interleaved(TensorMemoryLayout::INTERLEAVED, BufferType::L1);
    MemoryConfig height_sharded(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec_a_);
    EXPECT_NE(interleaved, height_sharded);
}

TEST_F(MemoryConfigEqualityTest, BothLegacyShardSpec_Equal) {
    MemoryConfig a(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec_a_);
    MemoryConfig b(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec_a_);
    EXPECT_EQ(a, b);
}

TEST_F(MemoryConfigEqualityTest, BothLegacyShardSpec_Different) {
    MemoryConfig a(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec_a_);
    MemoryConfig b(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec_b_);
    EXPECT_NE(a, b);
}

TEST_F(MemoryConfigEqualityTest, BothNdShardSpec_Equal) {
    MemoryConfig a(BufferType::L1, nd_shard_spec_a_);
    MemoryConfig b(BufferType::L1, nd_shard_spec_a_);
    EXPECT_EQ(a, b);
}

TEST_F(MemoryConfigEqualityTest, BothNdShardSpec_Different) {
    MemoryConfig a(BufferType::L1, nd_shard_spec_a_);
    MemoryConfig b(BufferType::L1, nd_shard_spec_b_);
    EXPECT_NE(a, b);
}

TEST_F(MemoryConfigEqualityTest, BothNoShardSpec) {
    MemoryConfig a(TensorMemoryLayout::INTERLEAVED, BufferType::L1);
    MemoryConfig b(TensorMemoryLayout::INTERLEAVED, BufferType::L1);
    EXPECT_EQ(a, b);
}

TEST_F(MemoryConfigEqualityTest, NdShardCreatedIgnoresLegacyField) {
    // Two configs both created with nd_shard_spec should compare equal on nd_shard_spec alone,
    // even if one has a legacy shard_spec prepopulated and the other does not.
    MemoryConfig user_config(BufferType::L1, nd_shard_spec_a_);
    MemoryConfig tensor_config = MemoryConfig::create_with_prepopulated_shard_specs(
        TensorMemoryLayout::ND_SHARDED,
        BufferType::L1,
        shard_spec_a_,
        nd_shard_spec_a_,
        /*created_with_nd_shard_spec=*/true);
    EXPECT_EQ(user_config, tensor_config);
}

TEST_F(MemoryConfigEqualityTest, LegacyShardCreatedIgnoresNdField) {
    // Two configs both created with legacy shard_spec should compare equal on shard_spec alone,
    // even if one has an nd_shard_spec prepopulated and the other does not.
    MemoryConfig user_config(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec_a_);
    MemoryConfig tensor_config = MemoryConfig::create_with_prepopulated_shard_specs(
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        shard_spec_a_,
        nd_shard_spec_a_,
        /*created_with_nd_shard_spec=*/false);
    EXPECT_EQ(user_config, tensor_config);
}

TEST_F(MemoryConfigEqualityTest, MixedCreationPath_BothFieldsMatch) {
    // One created with nd_shard_spec, the other with legacy shard_spec.
    // Equality falls through to comparing both fields.
    MemoryConfig nd_config = MemoryConfig::create_with_prepopulated_shard_specs(
        TensorMemoryLayout::ND_SHARDED,
        BufferType::L1,
        shard_spec_a_,
        nd_shard_spec_a_,
        /*created_with_nd_shard_spec=*/true);
    MemoryConfig legacy_config = MemoryConfig::create_with_prepopulated_shard_specs(
        TensorMemoryLayout::ND_SHARDED,
        BufferType::L1,
        shard_spec_a_,
        nd_shard_spec_a_,
        /*created_with_nd_shard_spec=*/false);
    EXPECT_EQ(nd_config, legacy_config);
}

TEST_F(MemoryConfigEqualityTest, MixedCreationPath_NdFieldDiffers) {
    MemoryConfig nd_config = MemoryConfig::create_with_prepopulated_shard_specs(
        TensorMemoryLayout::ND_SHARDED,
        BufferType::L1,
        shard_spec_a_,
        nd_shard_spec_a_,
        /*created_with_nd_shard_spec=*/true);
    MemoryConfig legacy_config = MemoryConfig::create_with_prepopulated_shard_specs(
        TensorMemoryLayout::ND_SHARDED,
        BufferType::L1,
        shard_spec_a_,
        nd_shard_spec_b_,
        /*created_with_nd_shard_spec=*/false);  // This is not a realistic scenario, but it's here to test the equality
                                                // operator.
    EXPECT_NE(nd_config, legacy_config);
}

TEST_F(MemoryConfigEqualityTest, MixedCreationPath_LegacyFieldDiffers) {
    MemoryConfig nd_config = MemoryConfig::create_with_prepopulated_shard_specs(
        TensorMemoryLayout::ND_SHARDED,
        BufferType::L1,
        shard_spec_a_,
        nd_shard_spec_a_,
        /*created_with_nd_shard_spec=*/true);
    MemoryConfig legacy_config = MemoryConfig::create_with_prepopulated_shard_specs(
        TensorMemoryLayout::ND_SHARDED,
        BufferType::L1,
        shard_spec_b_,
        nd_shard_spec_a_,
        /*created_with_nd_shard_spec=*/false);  // This is not a realistic scenario, but it's here to test the equality
                                                // operator.
    EXPECT_NE(nd_config, legacy_config);
}

TEST_F(MemoryConfigEqualityTest, NotEqualOperator) {
    MemoryConfig a(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec_a_);
    MemoryConfig b(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec_b_);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a != a);
}

TEST_F(MemoryConfigEqualityTest, NotEqualOperator_NdShardSpec) {
    MemoryConfig a(BufferType::L1, nd_shard_spec_a_);
    MemoryConfig b(BufferType::L1, nd_shard_spec_b_);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a != a);
}
