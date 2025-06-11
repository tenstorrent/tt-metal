// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <set>
#include <variant>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include "gtest/gtest.h"
#include <tt_stl/reflection.hpp>
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

struct TestMemoryConfigParams {
    ttnn::MemoryConfig memory_config;
};

class TEST_MEMORY_CONFIG : public ::testing::TestWithParam<TestMemoryConfigParams> {};

TEST_P(TEST_MEMORY_CONFIG, SerializeDeserialize) {
    const auto& memory_config = GetParam().memory_config;
    auto json_object = tt::stl::json::to_json(memory_config);

    auto deserialized_memory_config = tt::stl::json::from_json<ttnn::MemoryConfig>(json_object);

    ASSERT_EQ(memory_config, deserialized_memory_config);
}

INSTANTIATE_TEST_SUITE_P(
    TEST_JSON_CONVERSION,
    TEST_MEMORY_CONFIG,
    // clang-format off
    ::testing::Values(
        // Interleaved
        TestMemoryConfigParams{
            ttnn::MemoryConfig{
                ttnn::TensorMemoryLayout::INTERLEAVED,
                ttnn::BufferType::DRAM
            }
        },
        // Physical shard mode
        TestMemoryConfigParams{
            ttnn::MemoryConfig{
                ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                ttnn::BufferType::DRAM,
                tt::tt_metal::ShardSpec(
                    CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{1, 2}, CoreCoord{7, 4}}}},
                    {32, 128},
                    tt::tt_metal::ShardOrientation::ROW_MAJOR
                )
            }
        },
        // Logical shard mode
        TestMemoryConfigParams{
            ttnn::MemoryConfig{
                ttnn::TensorMemoryLayout::BLOCK_SHARDED,
                ttnn::BufferType::DRAM,
                tt::tt_metal::ShardSpec(
                    CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 4}}}},
                    {5, 6},
                    tt::tt_metal::ShardOrientation::ROW_MAJOR,
                    tt::tt_metal::ShardMode::LOGICAL
                )
            }
        },
        // Logical shard mode + custom physical shard shape
        TestMemoryConfigParams{
            ttnn::MemoryConfig{
                ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                ttnn::BufferType::L1,
                tt::tt_metal::ShardSpec(
                    CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                    {3, 4},
                    {32, 32},
                    tt::tt_metal::ShardOrientation::COL_MAJOR
                )
            }
        }
    )  // Values
    // clang-format on
);

TEST(TEST_JSON_CONVERSION, TEST_MATMUL_CONFIG) {
    auto matmul_multi_core_reuse_program_config =
        ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig{CoreCoord{2, 3}, 32, 64, 48, 128, 96};
    auto matmul_program_config = ttnn::operations::matmul::MatmulProgramConfig{matmul_multi_core_reuse_program_config};

    auto json_object = tt::stl::json::to_json(matmul_program_config);

    auto deserialized_matmul_program_config =
        tt::stl::json::from_json<ttnn::operations::matmul::MatmulProgramConfig>(json_object);

    ASSERT_EQ(
        matmul_multi_core_reuse_program_config.compute_with_storage_grid_size,
        std::get<ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig>(deserialized_matmul_program_config)
            .compute_with_storage_grid_size);
    ASSERT_EQ(
        matmul_multi_core_reuse_program_config.in0_block_w,
        std::get<ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig>(deserialized_matmul_program_config)
            .in0_block_w);
    ASSERT_EQ(
        matmul_multi_core_reuse_program_config.out_subblock_h,
        std::get<ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig>(deserialized_matmul_program_config)
            .out_subblock_h);
    ASSERT_EQ(
        matmul_multi_core_reuse_program_config.out_subblock_w,
        std::get<ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig>(deserialized_matmul_program_config)
            .out_subblock_w);
    ASSERT_EQ(
        matmul_multi_core_reuse_program_config.per_core_M,
        std::get<ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig>(deserialized_matmul_program_config)
            .per_core_M);
    ASSERT_EQ(
        matmul_multi_core_reuse_program_config.per_core_N,
        std::get<ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig>(deserialized_matmul_program_config)
            .per_core_N);
}
