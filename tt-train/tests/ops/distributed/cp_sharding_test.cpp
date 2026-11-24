// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core/distributed/cp_sharding.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <set>
#include <vector>

namespace {

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;
using ttml::core::distributed::cp_global_position;
using ttml::core::distributed::CpSharding;
using ttml::core::distributed::validate_cp_sequence_length;

// Contiguous sharding keeps every token at its own position.
TEST(CpShardingTest, ContiguousIsIdentity) {
    for (uint32_t pos = 0U; pos < 64U; ++pos) {
        EXPECT_EQ(cp_global_position(CpSharding::Contiguous, pos, 64U, 4U), pos);
    }
}

// Zig-zag gives device d global blocks d and 2N-1-d, and every position exactly once.
TEST(CpShardingTest, ZigZagGivesDeviceItsBlockAndTheMirrorBlock) {
    constexpr uint32_t seq_len = 64U;
    for (const uint32_t cp_size : {2U, 4U, 8U}) {
        const uint32_t block = seq_len / (2U * cp_size);
        std::set<uint32_t> seen;
        for (uint32_t pos = 0U; pos < seq_len; ++pos) {
            const uint32_t global = cp_global_position(CpSharding::ZigZag, pos, seq_len, cp_size);
            const uint32_t device = pos / (2U * block);
            const uint32_t global_block = global / block;
            EXPECT_TRUE(global_block == device || global_block == 2U * cp_size - 1U - device)
                << "cp_size " << cp_size << " pos " << pos << " -> " << global;
            seen.insert(global);
        }
        EXPECT_EQ(seen.size(), seq_len) << "cp_size " << cp_size;
    }
}

// Global positions in device order, written out by hand: block d first, then block 2N-1-d, each in order.
TEST(CpShardingTest, ZigZagMatchesHandWrittenLayout) {
    const auto layout = [](uint32_t seq_len, uint32_t cp_size) {
        std::vector<uint32_t> out;
        for (uint32_t pos = 0U; pos < seq_len; ++pos) {
            out.push_back(cp_global_position(CpSharding::ZigZag, pos, seq_len, cp_size));
        }
        return out;
    };
    EXPECT_EQ(layout(8U, 2U), (std::vector<uint32_t>{0, 1, 6, 7, 2, 3, 4, 5}));
    EXPECT_EQ(layout(16U, 4U), (std::vector<uint32_t>{0, 1, 14, 15, 2, 3, 12, 13, 4, 5, 10, 11, 6, 7, 8, 9}));
}

TEST(CpShardingTest, ZigZagRejectsSequenceNotSplitIntoWholeBlocks) {
    EXPECT_NO_THROW(validate_cp_sequence_length(CpSharding::ZigZag, 32U, 2U));
    EXPECT_NO_THROW(validate_cp_sequence_length(CpSharding::Contiguous, 34U, 2U));
    EXPECT_THAT(
        [] { validate_cp_sequence_length(CpSharding::ZigZag, 34U, 2U); },
        ThrowsMessage<std::runtime_error>(HasSubstr("divisible by 2 * cp_size (4)")));
}

}  // namespace
