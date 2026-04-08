// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <set>
#include <tt-metalium/core_coord.hpp>

#include "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/gather.hpp"

namespace ttnn::experimental::prim::test {

class GatherTransferTest : public ::testing::Test {
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

    // Verify transfer correctness by checking that all elements are accounted for
    bool verify_transfers(const std::vector<GatherTransfer>& transfers, uint32_t B, uint32_t C, uint32_t HW) {
        // Track which elements have been transferred
        std::set<std::tuple<uint32_t, uint32_t, uint32_t>> transferred_elements;

        // Calculate number of input cores
        std::set<uint32_t> input_cores;
        for (const auto& transfer : transfers) {
            input_cores.insert(transfer.src_core_idx);
        }
        uint32_t num_input_cores = input_cores.size();
        uint32_t hw_per_core = HW / num_input_cores;

        for (const auto& transfer : transfers) {
            // For each element in this transfer
            for (uint32_t i = 0; i < transfer.length; i++) {
                // Calculate the logical position this element came from
                uint32_t hw = (transfer.src_core_idx * hw_per_core) + transfer.src_offset + i;
                auto element = std::make_tuple(transfer.batch, transfer.channel, hw);

                // Check for duplicates
                if (transferred_elements.contains(element)) {
                    return false;  // Duplicate transfer
                }
                transferred_elements.insert(element);
            }
        }

        // Check that all elements were transferred
        uint32_t expected_elements = B * C * HW;
        return transferred_elements.size() == expected_elements;
    }

    // Check if transfers are properly coalesced
    bool check_coalescing(const std::vector<GatherTransfer>& transfers) {
        for (size_t i = 1; i < transfers.size(); i++) {
            const auto& prev = transfers[i - 1];
            const auto& curr = transfers[i];

            // Check if these could have been coalesced but weren't
            if (prev.src_core_idx == curr.src_core_idx && prev.dst_core_idx == curr.dst_core_idx &&
                prev.channel == curr.channel && prev.batch == curr.batch &&
                prev.src_offset + prev.length == curr.src_offset && prev.dst_offset + prev.length == curr.dst_offset) {
                return false;  // Should have been coalesced
            }
        }
        return true;
    }

    // Validate blocked transfer groups
    bool verify_blocked_groups(
        const std::vector<BlockedTransferGroup>& groups, uint32_t output_width, uint32_t block_size) {
        // Check that groups are properly organized
        std::set<std::pair<uint32_t, uint32_t>> seen_blocks;

        for (const auto& group : groups) {
            auto key = std::make_pair(group.dst_shard_idx, group.dst_block_idx);
            if (seen_blocks.contains(key)) {
                return false;  // Duplicate block
            }
            seen_blocks.insert(key);

            // Verify block index is valid
            uint32_t max_blocks = (output_width + block_size - 1) / block_size;
            if (group.dst_block_idx >= max_blocks) {
                return false;  // Invalid block index
            }
        }

        return true;
    }

    // Helper to create test input shards
    std::vector<std::vector<float>> create_test_input(uint32_t B, uint32_t C, uint32_t HW, uint32_t num_cores) {
        std::vector<std::vector<float>> shards(num_cores);
        uint32_t shard_width = HW / num_cores;
        uint32_t shard_height = B * C;

        float value = 0.0f;
        for (uint32_t core = 0; core < num_cores; core++) {
            shards[core].resize(shard_height * shard_width);
            for (uint32_t row = 0; row < shard_height; row++) {
                for (uint32_t col = 0; col < shard_width; col++) {
                    // Fill with sequential values for easy verification
                    shards[core][(row * shard_width) + col] = value++;
                }
            }
        }

        return shards;
    }

    // Helper to compute expected output value for a given position
    // Output layout: [C, B*HW/num_output_cores] per shard
    // For output shard idx, position [c, pos] corresponds to input [b, c, hw]
    float compute_expected_output_value(
        uint32_t output_shard_idx,
        uint32_t c,
        uint32_t pos,
        uint32_t B,
        uint32_t C,
        uint32_t HW,
        uint32_t num_input_cores,
        uint32_t num_output_cores) {
        uint32_t output_shard_width = B * HW / num_output_cores;
        uint32_t input_shard_width = HW / num_input_cores;

        // Determine which batch and hw this position represents
        // For output shard idx, we get elements output_shard_idx * output_shard_width to (idx+1) * output_shard_width -
        // 1
        uint32_t global_pos = (output_shard_idx * output_shard_width) + pos;
        uint32_t b = global_pos / HW;
        uint32_t hw = global_pos % HW;

        // Find which input core has this hw
        uint32_t input_core_idx = hw / input_shard_width;
        uint32_t hw_in_shard = hw % input_shard_width;

        // Input shard layout: [B*C, HW/num_input_cores]
        // Row = b*C + c
        uint32_t input_row = (b * C) + c;
        uint32_t input_col = hw_in_shard;

        // Calculate the value: each input core starts with a base value
        // Core 0: starts at 0
        // Core 1: starts at (B*C) * input_shard_width
        // Core k: starts at k * (B*C) * input_shard_width
        uint32_t base_value = input_core_idx * (B * C) * input_shard_width;
        uint32_t value = base_value + (input_row * input_shard_width) + input_col;

        return static_cast<float>(value);
    }

    // Verify all elements in output shards match expected values
    void verify_all_output_elements(
        const std::vector<std::vector<float>>& output_shards,
        uint32_t B,
        uint32_t C,
        uint32_t HW,
        uint32_t num_input_cores,
        uint32_t num_output_cores) {
        uint32_t output_shard_width = B * HW / num_output_cores;

        for (uint32_t shard_idx = 0; shard_idx < output_shards.size(); shard_idx++) {
            const auto& shard = output_shards[shard_idx];
            EXPECT_EQ(shard.size(), C * output_shard_width) << "Shard " << shard_idx << " has wrong size";

            for (uint32_t c = 0; c < C; c++) {
                for (uint32_t pos = 0; pos < output_shard_width; pos++) {
                    uint32_t idx = (c * output_shard_width) + pos;
                    float expected =
                        compute_expected_output_value(shard_idx, c, pos, B, C, HW, num_input_cores, num_output_cores);
                    float actual = shard[idx];

                    EXPECT_FLOAT_EQ(actual, expected)
                        << "Mismatch at shard " << shard_idx << ", c=" << c << ", pos=" << pos << " (idx=" << idx
                        << "): expected " << expected << ", got " << actual;
                }
            }
        }
    }
};

// CPU execution functions moved from production code for testing purposes

/**
 * @brief Reference implementation of blocked gather operation
 *
 * This is a software reference implementation that demonstrates the blocked
 * transfer approach. It's used for testing and validation, not for hardware execution.
 *
 * @param B Batch size
 * @param C Number of channels
 * @param HW Total spatial dimension
 * @param input_cores Vector of input core coordinates
 * @param output_cores Vector of output core coordinates
 * @param input_shards Input data organized as shards (vector of flattened arrays)
 * @param block_size Width of each column block (default 4)
 * @return Output shards after gather operation
 */
std::vector<std::vector<float>> gather_with_blocked_transfers(
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores,
    const std::vector<std::vector<float>>& input_shards,
    uint32_t block_size) {
    uint32_t num_input_cores = input_cores.size();
    uint32_t num_output_cores = output_cores.size();

    // Input validation
    TT_FATAL(HW % num_input_cores == 0, "HW={} must be divisible by num_input_cores={}", HW, num_input_cores);
    TT_FATAL(
        (B * HW) % num_output_cores == 0, "B*HW={} must be divisible by num_output_cores={}", B * HW, num_output_cores);

    // First, precompute the high-level transfer list
    auto transfers = precompute_gather_transfers(B, C, HW, input_cores, output_cores);

    // Calculate output dimensions
    uint32_t output_shard_height = C;
    uint32_t output_shard_width = B * HW / num_output_cores;

    // Group transfers by output column blocks
    auto blocked_result = group_transfers_by_output_column_blocks(
        transfers, B, C, HW, input_cores, num_output_cores, sizeof(float), block_size, output_shard_width);
    const auto& blocked_groups = blocked_result.blocked_transfers;

    // Flatten input shards for C-style access
    std::vector<std::vector<float>> input_shards_flat;
    input_shards_flat.reserve(input_shards.size());
    for (const auto& shard : input_shards) {
        input_shards_flat.push_back(shard);  // Already flattened in this implementation
    }

    // Initialize output shards
    std::vector<std::vector<float>> output_shards(num_output_cores);
    for (uint32_t i = 0; i < num_output_cores; i++) {
        output_shards[i].resize(output_shard_height * output_shard_width, -1.0f);
    }

    // Process each column block
    for (const auto& group : blocked_groups) {
        // Calculate column range for this block
        uint32_t col_start = group.dst_block_idx * block_size;
        uint32_t col_end = std::min(col_start + block_size, output_shard_width);
        uint32_t actual_block_width = col_end - col_start;

        // Allocate temporary buffer for this column block (all rows, block_size columns)
        // In real hardware, this could be in local memory
        std::vector<float> block_buffer(output_shard_height * actual_block_width, 0.0f);

        // Execute all transfers for this column block
        for (const auto& transfer : group.transfers) {
            // Get source data
            const auto& src_flat = input_shards_flat[transfer.src_shard_idx];

            // dst_offset is now relative to the block start (after splitting in gather.cpp)
            // We need to calculate the row from the source offset
            uint32_t input_shard_width = HW / input_cores.size();

            // For each element in the transfer
            for (uint32_t i = 0; i < transfer.length; i++) {
                uint32_t src_idx = transfer.src_offset + i;

                // Calculate which row in the input shard this element belongs to
                uint32_t src_row = src_idx / input_shard_width;

                // Input shard layout: [B*C, HW/num_cores], so row = batch*C + channel
                // Output shard layout: [C, B*HW/num_cores], so row = channel
                // Extract channel from src_row: channel = src_row % C
                uint32_t dst_row = src_row % C;

                // dst_offset is relative to block start, so use it directly
                uint32_t block_col = transfer.dst_offset + i;

                // Write to block buffer
                block_buffer[(dst_row * actual_block_width) + block_col] = src_flat[src_idx];
            }
        }

        // Write the complete column block to the output shard
        auto& output_shard = output_shards[group.dst_shard_idx];
        for (uint32_t row = 0; row < output_shard_height; row++) {
            std::memcpy(
                &output_shard[(row * output_shard_width) + col_start],
                &block_buffer[row * actual_block_width],
                actual_block_width * sizeof(float));
        }
    }

    return output_shards;
}

/**
 * @brief Generic implementation of blocked gather operation for arbitrary element types
 *
 * This function performs the gather operation on any data type by working with
 * raw bytes and element sizes, making it suitable for different precision formats.
 *
 * The implementation follows a blocked transfer approach where transfers are grouped
 * by output column blocks to improve memory access patterns and enable efficient
 * hardware implementation.
 *
 * This implementation keeps all offset calculations in elements and uses
 * element_size only for memory operations. This design is more aligned
 * with hardware DMA operations which work with byte counts.
 *
 * Memory efficiency:
 * - Block buffer size = C × block_size × element_size bytes
 * - For C=16, block_size=4:
 *   - float32 (element_size=4): 16 × 4 × 4 = 256 bytes
 *   - bfloat16 (element_size=2): 16 × 4 × 2 = 128 bytes
 */
void gather_with_blocked_transfers_generic(
    const void* input_data,
    void* output_data,
    uint32_t element_size,
    uint32_t B,
    uint32_t C,
    uint32_t HW,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores,
    uint32_t block_size) {
    // First, precompute transfers (element-based, size-agnostic)
    auto transfers = precompute_gather_transfers(B, C, HW, input_cores, output_cores);

    // Group transfers by output column blocks
    uint32_t output_shard_width = B * HW / output_cores.size();
    auto blocked_result = group_transfers_by_output_column_blocks(
        transfers, B, C, HW, input_cores, output_cores.size(), sizeof(float), block_size, output_shard_width);
    const auto& blocked_groups = blocked_result.blocked_transfers;

    uint32_t input_shard_width = HW / input_cores.size();
    uint32_t output_shard_height = C;

    // Cast to byte pointers for arithmetic
    const uint8_t* input_bytes = static_cast<const uint8_t*>(input_data);
    uint8_t* output_bytes = static_cast<uint8_t*>(output_data);

    // Process each block group
    for (const auto& group : blocked_groups) {
        uint32_t col_start = group.dst_block_idx * block_size;
        uint32_t col_end = std::min(col_start + block_size, output_shard_width);
        uint32_t actual_block_width = col_end - col_start;

        // Allocate block buffer in bytes
        uint32_t block_buffer_elements = output_shard_height * actual_block_width;
        std::vector<uint8_t> block_buffer(block_buffer_elements * element_size);

        // Execute transfers into block buffer
        for (const auto& transfer : group.transfers) {
            // Calculate source pointer for this shard
            const uint8_t* src_shard_base =
                input_bytes + (transfer.src_shard_idx * B * C * input_shard_width * element_size);

            // dst_offset is now relative to the block start (after splitting in gather.cpp)
            // Calculate the row from the source offset
            for (uint32_t i = 0; i < transfer.length; i++) {
                uint32_t src_idx = transfer.src_offset + i;

                // Calculate which row in the input shard this element belongs to
                uint32_t src_row = src_idx / input_shard_width;

                // Input shard layout: [B*C, HW/num_cores], so row = batch*C + channel
                // Output shard layout: [C, B*HW/num_cores], so row = channel
                // Extract channel from src_row: channel = src_row % C
                uint32_t dst_row = src_row % C;

                // dst_offset is relative to block start, so use it directly
                uint32_t block_col = transfer.dst_offset + i;
                uint32_t block_idx = (dst_row * actual_block_width) + block_col;

                // Calculate source pointer for this element
                const uint8_t* src_ptr = src_shard_base + (src_idx * element_size);

                // Copy element_size bytes
                std::memcpy(&block_buffer[block_idx * element_size], src_ptr, element_size);
            }
        }

        // Copy block buffer to output shard
        uint8_t* output_shard_base =
            output_bytes + (group.dst_shard_idx * output_shard_height * output_shard_width * element_size);

        for (uint32_t row = 0; row < output_shard_height; row++) {
            // Copy entire row of the block
            std::memcpy(
                &output_shard_base[(row * output_shard_width + col_start) * element_size],
                &block_buffer[row * actual_block_width * element_size],
                actual_block_width * element_size);
        }
    }
}

TEST_F(GatherTransferTest, BasicTransferGeneration) {
    // Test simple single core to single core case: B=1, C=2, HW=8
    uint32_t B = 1, C = 2, HW = 8;
    auto input_cores = make_cores(1);
    auto output_cores = make_cores(1);

    auto transfers = precompute_gather_transfers(B, C, HW, input_cores, output_cores);

    // Should have exactly C transfers (one per channel) since everything is on same core
    EXPECT_EQ(transfers.size(), C);

    // Verify all transfers
    EXPECT_TRUE(verify_transfers(transfers, B, C, HW));

    // Check first transfer details
    EXPECT_EQ(transfers[0].src_core_idx, 0u);
    EXPECT_EQ(transfers[0].dst_core_idx, 0u);
    EXPECT_EQ(transfers[0].channel, 0u);
    EXPECT_EQ(transfers[0].batch, 0u);
    EXPECT_EQ(transfers[0].length, HW);  // Should transfer entire row
}

TEST_F(GatherTransferTest, TransferCoalescing) {
    // Test that adjacent transfers are properly coalesced
    uint32_t B = 1, C = 2, HW = 16;
    auto input_cores = make_cores(2);
    auto output_cores = make_cores(2);

    auto transfers = precompute_gather_transfers(B, C, HW, input_cores, output_cores);

    // Verify coalescing worked properly
    EXPECT_TRUE(check_coalescing(transfers));

    // With 2 input cores and 2 output cores, we expect 4 transfers total
    // (2 channels × 2 source cores = 4 transfers)
    EXPECT_EQ(transfers.size(), 4u);
}

TEST_F(GatherTransferTest, MultiCoreTransfers) {
    // Test various core configurations
    struct TestCase {
        uint32_t B, C, HW;
        uint32_t num_input_cores, num_output_cores;
        std::string description;
    };

    std::vector<TestCase> test_cases = {
        {1, 2, 8, 1, 2, "1→2 cores"},
        {2, 2, 8, 2, 2, "2→2 cores"},
        {1, 4, 16, 2, 4, "2→4 cores"},
        {2, 4, 16, 4, 2, "4→2 cores"},
        {1, 3, 12, 3, 3, "3→3 cores (non-power-of-2)"},
    };

    for (const auto& tc : test_cases) {
        auto input_cores = make_cores(tc.num_input_cores);
        auto output_cores = make_cores(tc.num_output_cores);

        auto transfers = precompute_gather_transfers(tc.B, tc.C, tc.HW, input_cores, output_cores);

        // Verify correctness
        EXPECT_TRUE(verify_transfers(transfers, tc.B, tc.C, tc.HW)) << "Failed for " << tc.description;
        EXPECT_TRUE(check_coalescing(transfers)) << "Coalescing failed for " << tc.description;

        // Verify we have transfers from/to all cores
        std::set<uint32_t> src_cores, dst_cores;
        for (const auto& t : transfers) {
            src_cores.insert(t.src_core_idx);
            dst_cores.insert(t.dst_core_idx);
        }

        EXPECT_EQ(src_cores.size(), tc.num_input_cores) << "Not all input cores used in " << tc.description;
        EXPECT_EQ(dst_cores.size(), tc.num_output_cores) << "Not all output cores used in " << tc.description;
    }
}

TEST_F(GatherTransferTest, BlockedTransferGrouping) {
    // Test column block grouping
    uint32_t B = 1, C = 2, HW = 16;
    uint32_t block_size = 4;
    auto input_cores = make_cores(2);
    auto output_cores = make_cores(2);

    auto transfers = precompute_gather_transfers(B, C, HW, input_cores, output_cores);
    uint32_t output_shard_width = B * HW / output_cores.size();
    auto blocked_result = group_transfers_by_output_column_blocks(
        transfers, B, C, HW, input_cores, output_cores.size(), sizeof(float), block_size, output_shard_width);
    const auto& blocked_groups = blocked_result.blocked_transfers;

    // Calculate expected number of blocks
    uint32_t blocks_per_shard = (output_shard_width + block_size - 1) / block_size;

    // Verify blocked groups
    EXPECT_TRUE(verify_blocked_groups(blocked_groups, output_shard_width, block_size));

    // Check that all blocks are represented
    std::set<std::pair<uint32_t, uint32_t>> block_keys;
    for (const auto& group : blocked_groups) {
        block_keys.insert(std::make_pair(group.dst_shard_idx, group.dst_block_idx));
    }

    // We should have blocks for each output shard
    for (uint32_t shard = 0; shard < output_cores.size(); shard++) {
        for (uint32_t block = 0; block < blocks_per_shard; block++) {
            EXPECT_GT(block_keys.count(std::make_pair(shard, block)), 0u)
                << "Missing block " << block << " for shard " << shard;
        }
    }
}

TEST_F(GatherTransferTest, EdgeCases) {
    // Test edge cases

    // Case 1: B=1, C=1, HW=4 (minimal case)
    {
        uint32_t B = 1, C = 1, HW = 4;
        auto input_cores = make_cores(1);
        auto output_cores = make_cores(1);

        auto transfers = precompute_gather_transfers(B, C, HW, input_cores, output_cores);

        // Should have exactly 1 transfer
        EXPECT_EQ(transfers.size(), 1u);
        EXPECT_EQ(transfers[0].length, 4u);
        EXPECT_TRUE(verify_transfers(transfers, B, C, HW));
    }

    // Case 2: Single channel with multiple cores
    {
        uint32_t B = 1, C = 1, HW = 16;
        auto input_cores = make_cores(4);
        auto output_cores = make_cores(4);

        auto transfers = precompute_gather_transfers(B, C, HW, input_cores, output_cores);

        // Each input core should contribute to each output core
        EXPECT_EQ(transfers.size(), 4u);  // 4 transfers for single channel
        EXPECT_TRUE(verify_transfers(transfers, B, C, HW));
    }

    // Case 3: Large batch size
    {
        uint32_t B = 4, C = 2, HW = 8;
        auto input_cores = make_cores(2);
        auto output_cores = make_cores(2);

        auto transfers = precompute_gather_transfers(B, C, HW, input_cores, output_cores);

        EXPECT_TRUE(verify_transfers(transfers, B, C, HW));
        EXPECT_TRUE(check_coalescing(transfers));
    }
}

TEST_F(GatherTransferTest, LargeConfigurations) {
    // Stress test with larger configuration
    uint32_t B = 4, C = 16, HW = 64;
    auto input_cores = make_cores(8);
    auto output_cores = make_cores(8);

    auto transfers = precompute_gather_transfers(B, C, HW, input_cores, output_cores);

    // Verify basic properties
    EXPECT_TRUE(verify_transfers(transfers, B, C, HW));
    EXPECT_TRUE(check_coalescing(transfers));

    // Test blocking with different block sizes
    std::vector<uint32_t> block_sizes = {4, 8, 16};
    for (auto block_size : block_sizes) {
        uint32_t output_width = B * HW / output_cores.size();
        auto blocked_result = group_transfers_by_output_column_blocks(
            transfers, B, C, HW, input_cores, output_cores.size(), sizeof(float), block_size, output_width);
        const auto& blocked = blocked_result.blocked_transfers;
        EXPECT_TRUE(verify_blocked_groups(blocked, output_width, block_size)) << "Failed for block_size=" << block_size;
    }
}

TEST_F(GatherTransferTest, UnevenSharding) {
    // Test when HW is not evenly divisible by cores
    // This should fail with TT_FATAL, so we test that it throws
    uint32_t B = 1, C = 2, HW = 15;  // 15 not divisible by 2
    auto input_cores = make_cores(2);
    auto output_cores = make_cores(2);

    // This should throw due to TT_FATAL
    try {
        precompute_gather_transfers(B, C, HW, input_cores, output_cores);
        FAIL() << "Expected TT_FATAL exception";
    } catch (const std::exception& e) {
        std::string error_msg(e.what());
        EXPECT_NE(error_msg.find("HW=15 must be divisible by num_input_cores=2"), std::string::npos)
            << "Unexpected exception message: " << error_msg;
    }
}

TEST_F(GatherTransferTest, TransferLowering) {
    // Test high-level to low-level transfer conversion
    uint32_t B = 2, C = 2, HW = 8;
    auto input_cores = make_cores(2);
    auto output_cores = make_cores(2);

    auto transfers = precompute_gather_transfers(B, C, HW, input_cores, output_cores);
    uint32_t derived_output_width = B * HW / output_cores.size();
    auto low_level = lower_gather_transfers(
        transfers, B, C, HW, input_cores, output_cores.size(), sizeof(float), derived_output_width);

    // Should have same number of transfers
    EXPECT_EQ(transfers.size(), low_level.size());

    // Verify offsets are calculated correctly
    uint32_t input_shard_width = HW / input_cores.size();
    uint32_t output_shard_width = B * HW / output_cores.size();

    for (size_t i = 0; i < transfers.size(); i++) {
        const auto& hl = transfers[i];
        const auto& ll = low_level[i];

        // Verify source offset calculation
        uint32_t expected_src_row = (hl.batch * C) + hl.channel;
        uint32_t expected_src_offset = (expected_src_row * input_shard_width) + hl.src_offset;
        EXPECT_EQ(ll.src_offset, expected_src_offset) << "Source offset mismatch for transfer " << i;

        // Verify destination offset calculation
        uint32_t expected_dst_row = hl.channel;
        uint32_t expected_dst_offset = (expected_dst_row * output_shard_width) + hl.dst_offset;
        EXPECT_EQ(ll.dst_offset, expected_dst_offset) << "Destination offset mismatch for transfer " << i;

        // Length should be preserved
        EXPECT_EQ(ll.length, hl.length);
    }
}

TEST_F(GatherTransferTest, EndToEndGatherOperation) {
    // Test the full gather operation with blocked transfers
    uint32_t B = 2, C = 2, HW = 8;
    uint32_t block_size = 2;
    auto input_cores = make_cores(2);
    auto output_cores = make_cores(2);

    // Create test input
    auto input_shards = create_test_input(B, C, HW, input_cores.size());

    // Run gather operation
    auto output_shards = gather_with_blocked_transfers(B, C, HW, input_cores, output_cores, input_shards, block_size);

    // Verify output shape
    EXPECT_EQ(output_shards.size(), output_cores.size());

    uint32_t output_shard_height = C;
    uint32_t output_shard_width = B * HW / output_cores.size();
    for (const auto& shard : output_shards) {
        EXPECT_EQ(shard.size(), output_shard_height * output_shard_width);
    }

    // Verify data correctness by checking a few specific values
    // Input layout: [B*C, HW/cores] = [4, 4] per core
    // Output layout: [C, B*HW/cores] = [2, 8] per core

    // For example, element at (b=0, c=0, hw=0) should map correctly
    // Input: core 0, row 0, col 0 -> value 0
    // Output: core 0, row 0, col 0 -> should also be value 0
    EXPECT_EQ(output_shards[0][0], 0.0f);

    // Element at (b=0, c=1, hw=0)
    // Input: core 0, row 1, col 0 -> value 4
    // Output: core 0, row 1, col 0 -> should be value 4
    EXPECT_EQ(output_shards[0][output_shard_width], 4.0f);
}

TEST_F(GatherTransferTest, VerifyTransferSorting) {
    // Test that transfers are properly sorted for cache efficiency
    uint32_t B = 2, C = 4, HW = 16;
    auto input_cores = make_cores(4);
    auto output_cores = make_cores(4);

    auto transfers = precompute_gather_transfers(B, C, HW, input_cores, output_cores);

    // Verify transfers are sorted by source core first
    for (size_t i = 1; i < transfers.size(); i++) {
        EXPECT_LE(transfers[i - 1].src_core_idx, transfers[i].src_core_idx) << "Transfers not sorted by source core";

        // Within same source core, check row ordering
        if (transfers[i - 1].src_core_idx == transfers[i].src_core_idx) {
            uint32_t prev_row = (transfers[i - 1].batch * C) + transfers[i - 1].channel;
            uint32_t curr_row = (transfers[i].batch * C) + transfers[i].channel;
            EXPECT_LE(prev_row, curr_row) << "Transfers within core not sorted by row";

            // Within same row, check column ordering
            if (prev_row == curr_row) {
                EXPECT_LE(transfers[i - 1].src_offset, transfers[i].src_offset)
                    << "Transfers within row not sorted by offset";
            }
        }
    }
}

TEST_F(GatherTransferTest, GenericElementTypes) {
    // Test with different element types to verify generic implementation
    uint32_t B = 2, C = 4, HW = 16;
    auto input_cores = make_cores(2);
    auto output_cores = make_cores(2);

    // Test with float32 (host) - element_size = 4
    {
        // Create flat input data manually
        std::vector<float> input_data(B * C * HW);
        for (uint32_t i = 0; i < input_data.size(); i++) {
            input_data[i] = static_cast<float>(i);
        }
        std::vector<float> output_data(B * C * HW, 0.0f);

        gather_with_blocked_transfers_generic(
            input_data.data(), output_data.data(), sizeof(float), B, C, HW, input_cores, output_cores, 4);

        // Verify a few values
        // Input is in [B, C, HW] layout flattened as: b0c0hw0, b0c0hw1, ..., b0c1hw0, b0c1hw1, ...
        // Output should be in [C, B, HW] layout

        // First element (c=0, b=0, hw=0) should map to output position 0
        EXPECT_FLOAT_EQ(output_data[0], 0.0f);

        // For c=1, b=0, hw=1:
        // Input position: b*C*HW + c*HW + hw = 0*4*16 + 1*16 + 1 = 17
        // Output position: c*B*HW + b*HW + hw = 1*2*16 + 0*16 + 1 = 33
        uint32_t input_idx = 0 * C * HW + 1 * HW + 1;  // b=0, c=1, hw=1
        EXPECT_FLOAT_EQ(output_data[1 * B * HW + 0 * HW + 1], static_cast<float>(input_idx));
    }

    // Test with uint16_t (simulating bfloat16 on device) - element_size = 2
    {
        std::vector<uint16_t> input_data(B * C * HW);
        for (size_t i = 0; i < input_data.size(); i++) {
            input_data[i] = static_cast<uint16_t>(i);
        }
        std::vector<uint16_t> output_data(B * C * HW, 0);

        gather_with_blocked_transfers_generic(
            input_data.data(), output_data.data(), sizeof(uint16_t), B, C, HW, input_cores, output_cores, 4);

        // Verify transformation worked (same mapping as float test)
        EXPECT_EQ(output_data[0], 0);  // First element
        // For c=1, b=0, hw=1: input position = 0*4*16 + 1*16 + 1 = 17
        uint32_t input_idx_u16 = 0 * C * HW + 1 * HW + 1;
        EXPECT_EQ(output_data[1 * B * HW + 0 * HW + 1], static_cast<uint16_t>(input_idx_u16));
    }

    // Verify element size calculations
    EXPECT_EQ(sizeof(float), 4u);
    EXPECT_EQ(sizeof(uint16_t), 2u);
    EXPECT_EQ(elements_to_bytes(100, 4), 400u);  // 100 elements * 4 bytes
    EXPECT_EQ(elements_to_bytes(100, 2), 200u);  // 100 elements * 2 bytes
}

TEST_F(GatherTransferTest, VerifyInputAndOutputShards) {
    // Test the gather operation with simple configuration to verify correctness
    uint32_t B = 2, C = 3, HW = 4;
    auto input_cores = make_cores(2);
    auto output_cores = make_cores(2);

    auto input_shards = create_test_input(B, C, HW, input_cores.size());

    // Input is width-sharded: each core has [B*C, HW/num_cores] = [6, 2]
    // Core 0: values 0-11 (sequential)
    EXPECT_FLOAT_EQ(input_shards[0][0], 0.0f);    // row=0, col=0
    EXPECT_FLOAT_EQ(input_shards[0][1], 1.0f);    // row=0, col=1
    EXPECT_FLOAT_EQ(input_shards[0][2], 2.0f);    // row=1, col=0
    EXPECT_FLOAT_EQ(input_shards[0][3], 3.0f);    // row=1, col=1
    EXPECT_FLOAT_EQ(input_shards[0][4], 4.0f);    // row=2, col=0
    EXPECT_FLOAT_EQ(input_shards[0][5], 5.0f);    // row=2, col=1
    EXPECT_FLOAT_EQ(input_shards[0][6], 6.0f);    // row=3, col=0
    EXPECT_FLOAT_EQ(input_shards[0][7], 7.0f);    // row=3, col=1
    EXPECT_FLOAT_EQ(input_shards[0][8], 8.0f);    // row=4, col=0
    EXPECT_FLOAT_EQ(input_shards[0][9], 9.0f);    // row=4, col=1
    EXPECT_FLOAT_EQ(input_shards[0][10], 10.0f);  // row=5, col=0
    EXPECT_FLOAT_EQ(input_shards[0][11], 11.0f);  // row=5, col=1

    // Core 1: values 12-23 (sequential)
    EXPECT_FLOAT_EQ(input_shards[1][0], 12.0f);   // row=0, col=0
    EXPECT_FLOAT_EQ(input_shards[1][1], 13.0f);   // row=0, col=1
    EXPECT_FLOAT_EQ(input_shards[1][2], 14.0f);   // row=1, col=0
    EXPECT_FLOAT_EQ(input_shards[1][3], 15.0f);   // row=1, col=1
    EXPECT_FLOAT_EQ(input_shards[1][4], 16.0f);   // row=2, col=0
    EXPECT_FLOAT_EQ(input_shards[1][5], 17.0f);   // row=2, col=1
    EXPECT_FLOAT_EQ(input_shards[1][6], 18.0f);   // row=3, col=0
    EXPECT_FLOAT_EQ(input_shards[1][7], 19.0f);   // row=3, col=1
    EXPECT_FLOAT_EQ(input_shards[1][8], 20.0f);   // row=4, col=0
    EXPECT_FLOAT_EQ(input_shards[1][9], 21.0f);   // row=4, col=1
    EXPECT_FLOAT_EQ(input_shards[1][10], 22.0f);  // row=5, col=0
    EXPECT_FLOAT_EQ(input_shards[1][11], 23.0f);  // row=5, col=1

    // Run gather operation to get output shards
    // Output shard structure: [C, output_shard_width] where output_shard_width = B*HW/num_cores = 4
    auto output_shards = gather_with_blocked_transfers(B, C, HW, input_cores, output_cores, input_shards, 2);

    // Core 0 checks
    EXPECT_FLOAT_EQ(output_shards[0][0], 0.0f);   // c=0, pos 0
    EXPECT_FLOAT_EQ(output_shards[0][1], 1.0f);   // c=0, pos 1
    EXPECT_FLOAT_EQ(output_shards[0][2], 12.0f);  // c=0, pos 2
    EXPECT_FLOAT_EQ(output_shards[0][3], 13.0f);  // c=0, pos 3

    EXPECT_FLOAT_EQ(output_shards[0][4], 2.0f);   // c=1, pos 0
    EXPECT_FLOAT_EQ(output_shards[0][5], 3.0f);   // c=1, pos 1
    EXPECT_FLOAT_EQ(output_shards[0][6], 14.0f);  // c=1, pos 2
    EXPECT_FLOAT_EQ(output_shards[0][7], 15.0f);  // c=1, pos 3

    EXPECT_FLOAT_EQ(output_shards[0][8], 4.0f);    // c=2, pos 0
    EXPECT_FLOAT_EQ(output_shards[0][9], 5.0f);    // c=2, pos 1
    EXPECT_FLOAT_EQ(output_shards[0][10], 16.0f);  // c=2, pos 2
    EXPECT_FLOAT_EQ(output_shards[0][11], 17.0f);  // c=2, pos 3

    // Core 1 checks
    EXPECT_FLOAT_EQ(output_shards[1][0], 6.0f);   // c=0, pos 0
    EXPECT_FLOAT_EQ(output_shards[1][1], 7.0f);   // c=0, pos 1
    EXPECT_FLOAT_EQ(output_shards[1][2], 18.0f);  // c=0, pos 2
    EXPECT_FLOAT_EQ(output_shards[1][3], 19.0f);  // c=0, pos 3

    EXPECT_FLOAT_EQ(output_shards[1][4], 8.0f);   // c=1, pos 0
    EXPECT_FLOAT_EQ(output_shards[1][5], 9.0f);   // c=1, pos 1
    EXPECT_FLOAT_EQ(output_shards[1][6], 20.0f);  // c=1, pos 2
    EXPECT_FLOAT_EQ(output_shards[1][7], 21.0f);  // c=1, pos 3

    EXPECT_FLOAT_EQ(output_shards[1][8], 10.0f);   // c=2, pos 0
    EXPECT_FLOAT_EQ(output_shards[1][9], 11.0f);   // c=2, pos 1
    EXPECT_FLOAT_EQ(output_shards[1][10], 22.0f);  // c=2, pos 2
    EXPECT_FLOAT_EQ(output_shards[1][11], 23.0f);  // c=2, pos 3

    // Verify all elements were written (no -1 or uninitialized values)
    for (const auto& shard : output_shards) {
        for (float val : shard) {
            EXPECT_NE(val, -1.0f) << "Found unwritten element";
        }
    }
}

TEST_F(GatherTransferTest, OneToManyCores) {
    // Test 1 input core to 2 output cores
    uint32_t B = 2, C = 2, HW = 4;
    auto input_cores = make_cores(1);
    auto output_cores = make_cores(2);

    // Create input with 1 shard containing all data
    std::vector<std::vector<float>> input_shards(1);
    input_shards[0].resize(B * C * HW);
    for (uint32_t i = 0; i < input_shards[0].size(); i++) {
        input_shards[0][i] = static_cast<float>(i);
    }

    // Verify input values
    EXPECT_FLOAT_EQ(input_shards[0][0], 0.0f);    // b=0,c=0,hw=0
    EXPECT_FLOAT_EQ(input_shards[0][1], 1.0f);    // b=0,c=0,hw=1
    EXPECT_FLOAT_EQ(input_shards[0][2], 2.0f);    // b=0,c=0,hw=2
    EXPECT_FLOAT_EQ(input_shards[0][3], 3.0f);    // b=0,c=0,hw=3
    EXPECT_FLOAT_EQ(input_shards[0][4], 4.0f);    // b=0,c=1,hw=0
    EXPECT_FLOAT_EQ(input_shards[0][7], 7.0f);    // b=0,c=1,hw=3
    EXPECT_FLOAT_EQ(input_shards[0][8], 8.0f);    // b=1,c=0,hw=0
    EXPECT_FLOAT_EQ(input_shards[0][15], 15.0f);  // b=1,c=1,hw=3

    // Run gather operation
    auto output_shards = gather_with_blocked_transfers(B, C, HW, input_cores, output_cores, input_shards, 2);

    // Output is height-sharded: each core gets B*HW/2 = 4 elements per channel
    // Core 0 should get b=0 data, Core 1 should get b=1 data

    // Core 0: c=0[0,1,2,3], c=1[4,5,6,7]
    EXPECT_FLOAT_EQ(output_shards[0][0], 0.0f);  // c=0, b=0, hw=0
    EXPECT_FLOAT_EQ(output_shards[0][1], 1.0f);  // c=0, b=0, hw=1
    EXPECT_FLOAT_EQ(output_shards[0][2], 2.0f);  // c=0, b=0, hw=2
    EXPECT_FLOAT_EQ(output_shards[0][3], 3.0f);  // c=0, b=0, hw=3
    EXPECT_FLOAT_EQ(output_shards[0][4], 4.0f);  // c=1, b=0, hw=0
    EXPECT_FLOAT_EQ(output_shards[0][5], 5.0f);  // c=1, b=0, hw=1
    EXPECT_FLOAT_EQ(output_shards[0][6], 6.0f);  // c=1, b=0, hw=2
    EXPECT_FLOAT_EQ(output_shards[0][7], 7.0f);  // c=1, b=0, hw=3

    // Core 1: c=0[8,9,10,11], c=1[12,13,14,15]
    EXPECT_FLOAT_EQ(output_shards[1][0], 8.0f);   // c=0, b=1, hw=0
    EXPECT_FLOAT_EQ(output_shards[1][1], 9.0f);   // c=0, b=1, hw=1
    EXPECT_FLOAT_EQ(output_shards[1][2], 10.0f);  // c=0, b=1, hw=2
    EXPECT_FLOAT_EQ(output_shards[1][3], 11.0f);  // c=0, b=1, hw=3
    EXPECT_FLOAT_EQ(output_shards[1][4], 12.0f);  // c=1, b=1, hw=0
    EXPECT_FLOAT_EQ(output_shards[1][5], 13.0f);  // c=1, b=1, hw=1
    EXPECT_FLOAT_EQ(output_shards[1][6], 14.0f);  // c=1, b=1, hw=2
    EXPECT_FLOAT_EQ(output_shards[1][7], 15.0f);  // c=1, b=1, hw=3

    // Verify all elements were written (no -1 or uninitialized values)
    for (const auto& shard : output_shards) {
        for (float val : shard) {
            EXPECT_NE(val, -1.0f) << "Found unwritten element";
        }
    }
}

TEST_F(GatherTransferTest, ManyToOneCores) {
    // Test 2 input cores to 1 output core
    uint32_t B = 1, C = 2, HW = 8;
    auto input_cores = make_cores(2);
    auto output_cores = make_cores(1);

    // Create input shards - width sharded across 2 cores
    auto input_shards = create_test_input(B, C, HW, input_cores.size());

    // Verify key input values
    EXPECT_FLOAT_EQ(input_shards[0][0], 0.0f);   // c=0, hw=0
    EXPECT_FLOAT_EQ(input_shards[0][3], 3.0f);   // c=0, hw=3
    EXPECT_FLOAT_EQ(input_shards[0][4], 4.0f);   // c=1, hw=0
    EXPECT_FLOAT_EQ(input_shards[1][0], 8.0f);   // c=0, hw=4
    EXPECT_FLOAT_EQ(input_shards[1][3], 11.0f);  // c=0, hw=7
    EXPECT_FLOAT_EQ(input_shards[1][4], 12.0f);  // c=1, hw=4

    // Run gather operation
    auto output_shards = gather_with_blocked_transfers(B, C, HW, input_cores, output_cores, input_shards, 4);

    // All output on single core: [C, B*HW] = [2, 8]
    // Logical output tensor is identical to input tensor
    // Channel 0: interleaved from both cores [0,1,2,3,8,9,10,11]
    EXPECT_FLOAT_EQ(output_shards[0][0], 0.0f);
    EXPECT_FLOAT_EQ(output_shards[0][1], 1.0f);
    EXPECT_FLOAT_EQ(output_shards[0][2], 2.0f);
    EXPECT_FLOAT_EQ(output_shards[0][3], 3.0f);
    EXPECT_FLOAT_EQ(output_shards[0][4], 8.0f);
    EXPECT_FLOAT_EQ(output_shards[0][5], 9.0f);
    EXPECT_FLOAT_EQ(output_shards[0][6], 10.0f);
    EXPECT_FLOAT_EQ(output_shards[0][7], 11.0f);

    // Channel 1: interleaved from both cores [4,5,6,7,12,13,14,15]
    EXPECT_FLOAT_EQ(output_shards[0][8], 4.0f);
    EXPECT_FLOAT_EQ(output_shards[0][9], 5.0f);
    EXPECT_FLOAT_EQ(output_shards[0][10], 6.0f);
    EXPECT_FLOAT_EQ(output_shards[0][11], 7.0f);
    EXPECT_FLOAT_EQ(output_shards[0][12], 12.0f);
    EXPECT_FLOAT_EQ(output_shards[0][13], 13.0f);
    EXPECT_FLOAT_EQ(output_shards[0][14], 14.0f);
    EXPECT_FLOAT_EQ(output_shards[0][15], 15.0f);

    // Verify all elements were written (no -1 or uninitialized values)
    for (const auto& shard : output_shards) {
        for (float val : shard) {
            EXPECT_NE(val, -1.0f) << "Found unwritten element";
        }
    }
}

TEST_F(GatherTransferTest, FourCoresToFourCores) {
    // Test 4x4 core configuration with B=2, C=4, HW=8
    uint32_t B = 2, C = 4, HW = 8;
    auto input_cores = make_cores(4);
    auto output_cores = make_cores(4);

    // Create input shards
    auto input_shards = create_test_input(B, C, HW, input_cores.size());

    // Each input core has [B*C, HW/4] = [8, 2] elements
    // Verify a sample from each input core
    EXPECT_FLOAT_EQ(input_shards[0][0], 0.0f);    // core0: first element
    EXPECT_FLOAT_EQ(input_shards[0][15], 15.0f);  // core0: last element
    EXPECT_FLOAT_EQ(input_shards[1][0], 16.0f);   // core1: first element
    EXPECT_FLOAT_EQ(input_shards[2][0], 32.0f);   // core2: first element
    EXPECT_FLOAT_EQ(input_shards[3][0], 48.0f);   // core3: first element
    EXPECT_FLOAT_EQ(input_shards[3][15], 63.0f);  // core3: last element

    // Run gather operation
    auto output_shards = gather_with_blocked_transfers(B, C, HW, input_cores, output_cores, input_shards, 2);

    // Each output core has [C, B*HW/4] = [4, 4] elements
    // Output distribution follows height sharding pattern

    // Based on debug output, Core 0 has pattern: [0,1,16,17, 2,3,18,19, 4,5,20,21, 6,7,22,23]
    EXPECT_FLOAT_EQ(output_shards[0][0], 0.0f);
    EXPECT_FLOAT_EQ(output_shards[0][1], 1.0f);
    EXPECT_FLOAT_EQ(output_shards[0][2], 16.0f);
    EXPECT_FLOAT_EQ(output_shards[0][3], 17.0f);
    EXPECT_FLOAT_EQ(output_shards[0][4], 2.0f);
    EXPECT_FLOAT_EQ(output_shards[0][5], 3.0f);
    EXPECT_FLOAT_EQ(output_shards[0][6], 18.0f);
    EXPECT_FLOAT_EQ(output_shards[0][7], 19.0f);

    // Core 3 has pattern: [40,41,56,57, 42,43,58,59, 44,45,60,61, 46,47,62,63]
    EXPECT_FLOAT_EQ(output_shards[3][0], 40.0f);
    EXPECT_FLOAT_EQ(output_shards[3][1], 41.0f);
    EXPECT_FLOAT_EQ(output_shards[3][2], 56.0f);
    EXPECT_FLOAT_EQ(output_shards[3][3], 57.0f);
    EXPECT_FLOAT_EQ(output_shards[3][12], 46.0f);
    EXPECT_FLOAT_EQ(output_shards[3][13], 47.0f);
    EXPECT_FLOAT_EQ(output_shards[3][14], 62.0f);
    EXPECT_FLOAT_EQ(output_shards[3][15], 63.0f);

    // Verify all elements were written (no -1 or uninitialized values)
    for (const auto& shard : output_shards) {
        for (float val : shard) {
            EXPECT_NE(val, -1.0f) << "Found unwritten element";
        }
    }
}

TEST_F(GatherTransferTest, SingleChannelManyBatches) {
    // Test with C=1, B=4, HW=4 across 2 cores
    uint32_t B = 4, C = 1, HW = 4;
    auto input_cores = make_cores(2);
    auto output_cores = make_cores(2);

    // Create input shards
    auto input_shards = create_test_input(B, C, HW, input_cores.size());

    // Input layout: each core has [4, 2] = 8 elements
    // Core 0: b0hw0-1, b1hw0-1, b2hw0-1, b3hw0-1
    // Core 1: b0hw2-3, b1hw2-3, b2hw2-3, b3hw2-3
    EXPECT_FLOAT_EQ(input_shards[0][0], 0.0f);   // b=0,hw=0
    EXPECT_FLOAT_EQ(input_shards[0][2], 2.0f);   // b=1,hw=0
    EXPECT_FLOAT_EQ(input_shards[0][4], 4.0f);   // b=2,hw=0
    EXPECT_FLOAT_EQ(input_shards[0][6], 6.0f);   // b=3,hw=0
    EXPECT_FLOAT_EQ(input_shards[1][0], 8.0f);   // b=0,hw=2
    EXPECT_FLOAT_EQ(input_shards[1][7], 15.0f);  // b=3,hw=3

    // Run gather operation
    auto output_shards = gather_with_blocked_transfers(B, C, HW, input_cores, output_cores, input_shards, 4);

    // Output layout: each core has [1, 8] = 8 elements
    // Since C=1, output is just redistributed by B*HW
    // Core 0: b0-1 data, Core 1: b2-3 data

    // Core 0 should have b=0,1 all hw values
    EXPECT_FLOAT_EQ(output_shards[0][0], 0.0f);   // b=0,hw=0
    EXPECT_FLOAT_EQ(output_shards[0][1], 1.0f);   // b=0,hw=1
    EXPECT_FLOAT_EQ(output_shards[0][2], 8.0f);   // b=0,hw=2
    EXPECT_FLOAT_EQ(output_shards[0][3], 9.0f);   // b=0,hw=3
    EXPECT_FLOAT_EQ(output_shards[0][4], 2.0f);   // b=1,hw=0
    EXPECT_FLOAT_EQ(output_shards[0][5], 3.0f);   // b=1,hw=1
    EXPECT_FLOAT_EQ(output_shards[0][6], 10.0f);  // b=1,hw=2
    EXPECT_FLOAT_EQ(output_shards[0][7], 11.0f);  // b=1,hw=3

    // Core 1 should have b=2,3 all hw values
    EXPECT_FLOAT_EQ(output_shards[1][0], 4.0f);   // b=2,hw=0
    EXPECT_FLOAT_EQ(output_shards[1][1], 5.0f);   // b=2,hw=1
    EXPECT_FLOAT_EQ(output_shards[1][2], 12.0f);  // b=2,hw=2
    EXPECT_FLOAT_EQ(output_shards[1][3], 13.0f);  // b=2,hw=3
    EXPECT_FLOAT_EQ(output_shards[1][4], 6.0f);   // b=3,hw=0
    EXPECT_FLOAT_EQ(output_shards[1][5], 7.0f);   // b=3,hw=1
    EXPECT_FLOAT_EQ(output_shards[1][6], 14.0f);  // b=3,hw=2
    EXPECT_FLOAT_EQ(output_shards[1][7], 15.0f);  // b=3,hw=3

    // Verify all elements were written (no -1 or uninitialized values)
    for (const auto& shard : output_shards) {
        for (float val : shard) {
            EXPECT_NE(val, -1.0f) << "Found unwritten element";
        }
    }
}

TEST_F(GatherTransferTest, BlockBoundaryCorrectness) {
    // Test that values at block boundaries are handled correctly
    // Use a simpler test with wrapper function to verify block boundaries
    uint32_t B = 1, C = 2, HW = 8;
    uint32_t block_size = 3;  // Non-power-of-2 to test edge cases
    auto input_cores = make_cores(2);
    auto output_cores = make_cores(2);

    // Create input shards with clear pattern
    std::vector<std::vector<float>> input_shards(2);
    uint32_t input_shard_width = HW / 2;  // 4 elements per core

    // Fill input shards with pattern: shard0 has 0-3, 8-11; shard1 has 4-7, 12-15
    for (uint32_t core = 0; core < 2; core++) {
        input_shards[core].resize(C * input_shard_width);
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t col = 0; col < input_shard_width; col++) {
                uint32_t hw = core * input_shard_width + col;
                input_shards[core][c * input_shard_width + col] = c * 100.0f + hw;
            }
        }
    }

    // Run gather with block_size=3
    auto output_shards = gather_with_blocked_transfers(B, C, HW, input_cores, output_cores, input_shards, block_size);

    // With block_size=3 and output_shard_width=4, we have:
    // Block 0: columns 0-2
    // Block 1: column 3
    // The blocked algorithm should still produce correct results

    // Verify output values match expected pattern
    // For B=1, the gather should essentially preserve values but rearrange sharding
    uint32_t output_shard_width = B * HW / 2;  // 4 elements per core

    for (uint32_t core = 0; core < 2; core++) {
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t col = 0; col < output_shard_width; col++) {
                uint32_t hw = core * output_shard_width + col;
                float expected = c * 100.0f + hw;
                float actual = output_shards[core][c * output_shard_width + col];

                EXPECT_FLOAT_EQ(actual, expected) << "Block boundary error at core=" << core << ", c=" << c
                                                  << ", col=" << col << " (hw=" << hw << ")";
            }
        }
    }
}

TEST_F(GatherTransferTest, SmallBlockSizes) {
    // Test with small block sizes (32, 64) to verify L1 reduction works correctly
    uint32_t B = 2, C = 4, HW = 128;
    auto input_cores = make_cores(4);
    auto output_cores = make_cores(4);

    auto input_shards = create_test_input(B, C, HW, input_cores.size());
    uint32_t output_shard_width = B * HW / output_cores.size();  // 64 elements per core

    // Test with block_size = 32
    {
        uint32_t block_size = 32;
        auto output_shards =
            gather_with_blocked_transfers(B, C, HW, input_cores, output_cores, input_shards, block_size);

        // Verify output shape
        EXPECT_EQ(output_shards.size(), output_cores.size());
        for (const auto& shard : output_shards) {
            EXPECT_EQ(shard.size(), C * output_shard_width);
        }

        // Verify all elements explicitly
        verify_all_output_elements(output_shards, B, C, HW, input_cores.size(), output_cores.size());
    }

    // Test with block_size = 64 (full shard width)
    {
        uint32_t block_size = 64;
        auto output_shards =
            gather_with_blocked_transfers(B, C, HW, input_cores, output_cores, input_shards, block_size);

        // Verify output shape
        EXPECT_EQ(output_shards.size(), output_cores.size());
        for (const auto& shard : output_shards) {
            EXPECT_EQ(shard.size(), C * output_shard_width);
        }

        // Verify all elements explicitly
        verify_all_output_elements(output_shards, B, C, HW, input_cores.size(), output_cores.size());
    }
}

TEST_F(GatherTransferTest, LargeBlockSizes) {
    // Test with larger configurations and block sizes to verify scalability
    uint32_t B = 4, C = 8, HW = 256;
    auto input_cores = make_cores(8);
    auto output_cores = make_cores(8);

    auto input_shards = create_test_input(B, C, HW, input_cores.size());

    // Test with block_size = 32 (smaller than shard width)
    {
        uint32_t block_size = 32;
        auto output_shards =
            gather_with_blocked_transfers(B, C, HW, input_cores, output_cores, input_shards, block_size);

        // Verify all elements explicitly
        verify_all_output_elements(output_shards, B, C, HW, input_cores.size(), output_cores.size());
    }

    // Test with block_size = 64
    {
        uint32_t block_size = 64;
        auto output_shards =
            gather_with_blocked_transfers(B, C, HW, input_cores, output_cores, input_shards, block_size);

        // Verify all elements explicitly
        verify_all_output_elements(output_shards, B, C, HW, input_cores.size(), output_cores.size());
    }
}

TEST_F(GatherTransferTest, BlockSizeVariations) {
    // Test various block sizes to ensure the blocking logic works correctly
    uint32_t B = 2, C = 3, HW = 64;
    auto input_cores = make_cores(4);
    auto output_cores = make_cores(4);

    auto input_shards = create_test_input(B, C, HW, input_cores.size());
    uint32_t output_shard_width = B * HW / output_cores.size();  // 32 elements per core

    // Test different block sizes that divide the output shard width
    std::vector<uint32_t> block_sizes = {8, 16, 32};

    for (uint32_t block_size : block_sizes) {
        // Skip if block_size doesn't divide output_shard_width
        if (output_shard_width % block_size != 0) {
            continue;
        }

        auto output_shards =
            gather_with_blocked_transfers(B, C, HW, input_cores, output_cores, input_shards, block_size);

        // Verify output shape
        EXPECT_EQ(output_shards.size(), output_cores.size()) << "Failed for block_size=" << block_size;
        for (const auto& shard : output_shards) {
            EXPECT_EQ(shard.size(), C * output_shard_width) << "Failed for block_size=" << block_size;
        }

        // Verify all elements explicitly
        verify_all_output_elements(output_shards, B, C, HW, input_cores.size(), output_cores.size());
    }
}

TEST_F(GatherTransferTest, CrossBlockTransferSplitting) {
    // Test that transfers crossing block boundaries are correctly split
    uint32_t B = 1, C = 2, HW = 16;
    uint32_t block_size = 4;
    auto input_cores = make_cores(2);
    auto output_cores = make_cores(2);

    // Create input with known pattern
    auto input_shards = create_test_input(B, C, HW, input_cores.size());

    // Precompute transfers and verify blocking
    auto transfers = precompute_gather_transfers(B, C, HW, input_cores, output_cores);
    uint32_t output_shard_width = B * HW / output_cores.size();  // 8 elements per core

    auto blocked_result = group_transfers_by_output_column_blocks(
        transfers, B, C, HW, input_cores, output_cores.size(), sizeof(float), block_size, output_shard_width);
    const auto& blocked_groups = blocked_result.blocked_transfers;

    // Verify that transfers are properly distributed across blocks
    // With output_shard_width=8 and block_size=4, we should have 2 blocks per shard
    uint32_t expected_blocks_per_shard = (output_shard_width + block_size - 1) / block_size;
    std::map<uint32_t, uint32_t> blocks_per_shard;
    for (const auto& group : blocked_groups) {
        blocks_per_shard[group.dst_shard_idx]++;
    }

    for (uint32_t shard = 0; shard < output_cores.size(); shard++) {
        EXPECT_GE(blocks_per_shard[shard], expected_blocks_per_shard)
            << "Shard " << shard << " should have at least " << expected_blocks_per_shard << " blocks";
    }

    // Run the actual gather to verify correctness
    auto output_shards = gather_with_blocked_transfers(B, C, HW, input_cores, output_cores, input_shards, block_size);

    // Verify all elements explicitly to ensure cross-block splitting works correctly
    verify_all_output_elements(output_shards, B, C, HW, input_cores.size(), output_cores.size());
}

}  // namespace ttnn::experimental::prim::test
