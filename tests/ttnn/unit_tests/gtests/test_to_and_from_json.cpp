// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "gtest/gtest.h"
#include "ttnn/cpp/ttnn/operations/matmul/matmul.hpp"
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
        ttnn::MatmulMultiCoreReuseProgramConfig{CoreCoord{2, 3}, 32, 64, 48, 128, 96};
    auto matmul_program_config = ttnn::MatmulProgramConfig{matmul_multi_core_reuse_program_config};

    auto json_object = tt::stl::json::to_json(matmul_program_config);

    auto deserialized_matmul_program_config = tt::stl::json::from_json<ttnn::MatmulProgramConfig>(json_object);

    // ASSERT_EQ(memory_config, deserialized_memory_config);
}
