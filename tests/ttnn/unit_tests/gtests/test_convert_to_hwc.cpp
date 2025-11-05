// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdint.h>

#include <tt-metalium/core_coord.hpp>
#include "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/convert_to_hwc_program_factory.hpp"

namespace ttnn {
namespace operations {
namespace experimental {
namespace cnn {
namespace detail {
namespace test {

class ConvertToHWCTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper to create core coordinate vectors
    std::vector<CoreCoord> make_cores(uint32_t num_cores) {
        std::vector<CoreCoord> cores;
        cores.reserve(num_cores);
        for (uint32_t i = 0; i < num_cores; i++) {
            cores.emplace_back(i, 0);  // Simple linear arrangement
        }
        return cores;
    }
};

class ConvertToHWCGenerateBatchRedistributionTransfersTest : public ConvertToHWCTest {};

TEST_F(ConvertToHWCGenerateBatchRedistributionTransfersTest, SimpleSingleCoreCase) {
    // Test the originally buggy case: B=2, C=2, HW=32, 1->1 core
    uint32_t batch_size = 2;
    uint32_t channels = 2;
    uint32_t hw_total = 32;
    auto input_cores = make_cores(1);
    auto output_cores = make_cores(1);
    uint32_t element_size_bytes = 2;

    uint32_t padded_shard_width = hw_total / input_cores.size();  // Even sharding for test
    auto instructions = generate_batch_redistribution_transfers(
        batch_size, channels, hw_total, input_cores, output_cores, element_size_bytes, padded_shard_width);

    // Should generate exactly 2 transfers (one per batch)
    EXPECT_EQ(instructions.size(), 2u);

    // Verify first transfer (batch 0)
    EXPECT_EQ(instructions[0].src_core_idx, 0u);
    EXPECT_EQ(instructions[0].dst_core_idx, 0u);
    EXPECT_EQ(instructions[0].src_offset, 0u);
    EXPECT_EQ(instructions[0].dst_offset, 0u);
    EXPECT_EQ(instructions[0].transfer_size, 128u);  // 32 elements * 2 channels * 2 bytes

    // Verify second transfer (batch 1)
    EXPECT_EQ(instructions[1].src_core_idx, 0u);
    EXPECT_EQ(instructions[1].dst_core_idx, 0u);
    EXPECT_EQ(instructions[1].src_offset, 128u);  // Start of second batch
    EXPECT_EQ(instructions[1].dst_offset, 128u);
    EXPECT_EQ(instructions[1].transfer_size, 128u);
}

TEST_F(ConvertToHWCGenerateBatchRedistributionTransfersTest, MultipleBatchesCase) {
    // Test B=3, C=4, HW=16, 1->1 core
    uint32_t batch_size = 3;
    uint32_t channels = 4;
    uint32_t hw_total = 16;
    auto input_cores = make_cores(1);
    auto output_cores = make_cores(1);
    uint32_t element_size_bytes = 2;

    uint32_t padded_shard_width = hw_total / input_cores.size();  // Even sharding for test
    auto instructions = generate_batch_redistribution_transfers(
        batch_size, channels, hw_total, input_cores, output_cores, element_size_bytes, padded_shard_width);

    // Should generate exactly 3 transfers (one per batch)
    EXPECT_EQ(instructions.size(), 3u);

    uint32_t batch_size_bytes = channels * hw_total * element_size_bytes;  // 4 * 16 * 2 = 128

    for (uint32_t i = 0; i < 3; i++) {
        EXPECT_EQ(instructions[i].src_core_idx, 0u);
        EXPECT_EQ(instructions[i].dst_core_idx, 0u);
        EXPECT_EQ(instructions[i].src_offset, i * batch_size_bytes);
        EXPECT_EQ(instructions[i].dst_offset, i * batch_size_bytes);
        EXPECT_EQ(instructions[i].transfer_size, batch_size_bytes);
    }
}

TEST_F(ConvertToHWCGenerateBatchRedistributionTransfersTest, MultiCoreCase) {
    // Test B=2, C=2, HW=64, 2->2 cores
    uint32_t batch_size = 2;
    uint32_t channels = 2;
    uint32_t hw_total = 64;
    auto input_cores = make_cores(2);
    auto output_cores = make_cores(2);
    uint32_t element_size_bytes = 2;

    uint32_t padded_shard_width = hw_total / input_cores.size();  // Even sharding for test
    auto instructions = generate_batch_redistribution_transfers(
        batch_size, channels, hw_total, input_cores, output_cores, element_size_bytes, padded_shard_width);

    // Should have transfers for each output core
    // Each output core gets data from both input cores
    // So we should have at least one transfer per (output_core, batch) pair
    EXPECT_GE(instructions.size(), 4u);  // At least 2 output cores * 2 batches

    // Verify we have transfers from both source cores
    bool has_src_0 = false, has_src_1 = false;
    bool has_dst_0 = false, has_dst_1 = false;

    for (const auto& instr : instructions) {
        if (instr.src_core_idx == 0) {
            has_src_0 = true;
        }
        if (instr.src_core_idx == 1) {
            has_src_1 = true;
        }
        if (instr.dst_core_idx == 0) {
            has_dst_0 = true;
        }
        if (instr.dst_core_idx == 1) {
            has_dst_1 = true;
        }
    }

    EXPECT_TRUE(has_src_0);
    EXPECT_TRUE(has_src_1);
    EXPECT_TRUE(has_dst_0);
    EXPECT_TRUE(has_dst_1);
}

TEST_F(ConvertToHWCGenerateBatchRedistributionTransfersTest, DifferentGridSizes) {
    // Test B=4, C=2, HW=32, 2->4 cores (input cores != output cores)
    uint32_t batch_size = 4;
    uint32_t channels = 2;
    uint32_t hw_total = 32;
    auto input_cores = make_cores(2);
    auto output_cores = make_cores(4);
    uint32_t element_size_bytes = 2;

    uint32_t padded_shard_width = hw_total / input_cores.size();  // Even sharding for test
    auto instructions = generate_batch_redistribution_transfers(
        batch_size, channels, hw_total, input_cores, output_cores, element_size_bytes, padded_shard_width);

    // Should have transfers for each output core
    EXPECT_GT(instructions.size(), 0u);

    // Verify all output cores are represented
    std::set<uint32_t> dst_cores_seen;
    for (const auto& instr : instructions) {
        dst_cores_seen.insert(instr.dst_core_idx);
        // Source core should be valid
        EXPECT_LT(instr.src_core_idx, 2u);
        // Destination core should be valid
        EXPECT_LT(instr.dst_core_idx, 4u);
    }

    EXPECT_EQ(dst_cores_seen.size(), 4u);  // All 4 output cores should be present
}

TEST_F(ConvertToHWCGenerateBatchRedistributionTransfersTest, SingleBatchCase) {
    // Test B=1, C=8, HW=64, 1->1 core
    uint32_t batch_size = 1;
    uint32_t channels = 8;
    uint32_t hw_total = 64;
    auto input_cores = make_cores(1);
    auto output_cores = make_cores(1);
    uint32_t element_size_bytes = 2;

    uint32_t padded_shard_width = hw_total / input_cores.size();  // Even sharding for test
    auto instructions = generate_batch_redistribution_transfers(
        batch_size, channels, hw_total, input_cores, output_cores, element_size_bytes, padded_shard_width);

    // Should generate exactly 1 transfer (single batch)
    EXPECT_EQ(instructions.size(), 1u);

    // Verify the single transfer
    EXPECT_EQ(instructions[0].src_core_idx, 0u);
    EXPECT_EQ(instructions[0].dst_core_idx, 0u);
    EXPECT_EQ(instructions[0].src_offset, 0u);
    EXPECT_EQ(instructions[0].dst_offset, 0u);
    EXPECT_EQ(instructions[0].transfer_size, 1024u);  // 64 * 8 * 2
}

}  // namespace test
}  // namespace detail
}  // namespace cnn
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn
