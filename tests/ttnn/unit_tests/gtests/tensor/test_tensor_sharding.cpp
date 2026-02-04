// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <fmt/base.h>
#include <cstddef>
#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt_stl/span.hpp>
#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "common_tensor_test_utils.hpp"
#include <tt-metalium/core_coord.hpp>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <tt-metalium/math.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/tile.hpp>
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace {
const CoreCoord grid_size{8, 7};

struct IllegalShardSpecParams {
    Shape shape;
    PageConfig page_config;
    MemoryConfig memory_config;
    std::string expected_err_msg;
};
}  // namespace

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
