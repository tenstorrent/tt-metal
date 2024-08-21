// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "gtest/gtest.h"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/types.hpp"

TEST(TEST_JSON_CONVERSION, TEST_MEMORY_CONFIG) {
    auto memory_config = ttnn::MemoryConfig{
        .memory_layout = ttnn::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = ttnn::BufferType::DRAM,
        .shard_spec = ShardSpec{
            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{1, 2}, CoreCoord{7, 4}}}},
            {3, 4},
            ShardOrientation::COL_MAJOR,
            true}};

    auto json_object = tt::stl::json::to_json(memory_config);

    auto deserialized_memory_config = tt::stl::json::from_json<ttnn::MemoryConfig>(json_object);

    ASSERT_EQ(memory_config, deserialized_memory_config);
}

TEST(TEST_JSON_CONVERSION, TEST_MATMUL_CONFIG) {
    auto matmul_multi_core_reuse_program_config =
        ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig{CoreCoord{2, 3}, 32, 64, 48, 128, 96};
    auto matmul_program_config = ttnn::operations::matmul::MatmulProgramConfig{matmul_multi_core_reuse_program_config};

    auto json_object = tt::stl::json::to_json(matmul_program_config);

    auto deserialized_matmul_program_config = tt::stl::json::from_json<ttnn::operations::matmul::MatmulProgramConfig>(json_object);

    ASSERT_EQ(
        matmul_multi_core_reuse_program_config.compute_with_storage_grid_size,
        std::get<ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig>(deserialized_matmul_program_config)
            .compute_with_storage_grid_size);
    ASSERT_EQ(
        matmul_multi_core_reuse_program_config.in0_block_w,
        std::get<ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig>(deserialized_matmul_program_config).in0_block_w);
    ASSERT_EQ(
        matmul_multi_core_reuse_program_config.out_subblock_h,
        std::get<ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig>(deserialized_matmul_program_config).out_subblock_h);
    ASSERT_EQ(
        matmul_multi_core_reuse_program_config.out_subblock_w,
        std::get<ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig>(deserialized_matmul_program_config).out_subblock_w);
    ASSERT_EQ(
        matmul_multi_core_reuse_program_config.per_core_M,
        std::get<ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig>(deserialized_matmul_program_config).per_core_M);
    ASSERT_EQ(
        matmul_multi_core_reuse_program_config.per_core_N,
        std::get<ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig>(deserialized_matmul_program_config).per_core_N);
}
