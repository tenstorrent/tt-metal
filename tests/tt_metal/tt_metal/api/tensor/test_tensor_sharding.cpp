// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
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
