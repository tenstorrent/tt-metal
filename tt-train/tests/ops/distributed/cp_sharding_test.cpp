// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core/distributed/cp_sharding.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <core/xtensor_utils.hpp>
#include <set>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/distributed/pipeline_parallel_llama.hpp"
#include "ops/rope_op.hpp"
#include "test_utils/mesh_utils.hpp"

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

TEST(CpShardingTest, ReadCpShardingAcceptsOnlyKnownNames) {
    EXPECT_EQ(ttml::core::distributed::read_cp_sharding("contiguous"), CpSharding::Contiguous);
    EXPECT_EQ(ttml::core::distributed::read_cp_sharding("zigzag"), CpSharding::ZigZag);
    EXPECT_THAT(
        [] { ttml::core::distributed::read_cp_sharding("zig-zag"); },
        ThrowsMessage<std::runtime_error>(HasSubstr("Supported: contiguous, zigzag")));
}

TEST(CpShardingTest, PipelineParallelRejectsZigZag) {
    using ttml::models::distributed::pipeline_parallel_llama::reject_unsupported_cp_sharding;
    EXPECT_NO_THROW(reject_unsupported_cp_sharding(CpSharding::Contiguous));
    EXPECT_THAT(
        [] { reject_unsupported_cp_sharding(CpSharding::ZigZag); },
        ThrowsMessage<std::logic_error>(HasSubstr("not supported with pipeline parallelism")));
}

TEST(CpShardingTest, ZigZagRejectsSequenceNotSplitIntoWholeBlocks) {
    EXPECT_NO_THROW(validate_cp_sequence_length(CpSharding::ZigZag, 32U, 2U));
    EXPECT_NO_THROW(validate_cp_sequence_length(CpSharding::Contiguous, 34U, 2U));
    EXPECT_THAT(
        [] { validate_cp_sequence_length(CpSharding::ZigZag, 34U, 2U); },
        ThrowsMessage<std::runtime_error>(HasSubstr("divisible by 2 * cp_size (4)")));
}

class Mesh1x2ZigZagRopeTest : public ttml::test_utils::Mesh1x2SuiteFixture {
protected:
    static void SetUpTestSuite() {
        Mesh1x2SuiteFixture::SetUpTestSuite();
        if (ttml::test_utils::system_supports_mesh(mesh_shape())) {
            ttml::autograd::ctx().initialize_parallelism_context(
                {.cp = ttml::autograd::ContextParallelConfig{.sharding = CpSharding::ZigZag}});
        }
    }
};

// Each device's RoPE rows are the full-sequence rows at the global positions of the tokens it holds.
TEST_F(Mesh1x2ZigZagRopeTest, RopeRowsFollowTheTokenPositions) {
    constexpr uint32_t seq_len = 128U;
    constexpr uint32_t head_dim = 64U;
    constexpr uint32_t cp_size = 2U;
    const auto params = ttml::ops::build_rope_params(seq_len, head_dim);
    const auto [full_sin, full_cos] = ttml::ops::gen_freqs(head_dim, seq_len, params.theta, params.rope_scaling_params);
    const auto expected_sin = ttml::core::to_xtensor<float>(full_sin, ttml::core::IdentityComposer{}).front();
    const auto expected_cos = ttml::core::to_xtensor<float>(full_cos, ttml::core::IdentityComposer{}).front();
    const auto sin_shards = ttml::core::to_xtensor<float>(params.sin_cache, ttml::core::IdentityComposer{});
    const auto cos_shards = ttml::core::to_xtensor<float>(params.cos_cache, ttml::core::IdentityComposer{});
    ASSERT_EQ(sin_shards.size(), cp_size);

    const uint32_t local_len = seq_len / cp_size;
    for (uint32_t device = 0U; device < cp_size; ++device) {
        for (uint32_t row = 0U; row < local_len; ++row) {
            const uint32_t global = cp_global_position(CpSharding::ZigZag, device * local_len + row, seq_len, cp_size);
            for (uint32_t col = 0U; col < head_dim; ++col) {
                ASSERT_EQ(sin_shards[device](0, 0, row, col), expected_sin(0, 0, global, col))
                    << "device " << device << " row " << row << " col " << col;
                ASSERT_EQ(cos_shards[device](0, 0, row, col), expected_cos(0, 0, global, col))
                    << "device " << device << " row " << row << " col " << col;
            }
        }
    }
}

TEST_F(Mesh1x2ZigZagRopeTest, RejectsSequenceNotSplitIntoWholeBlocks) {
    EXPECT_THAT(
        [] { ttml::ops::build_rope_params(/*sequence_length=*/34U, /*head_dim=*/64U); },
        ThrowsMessage<std::runtime_error>(HasSubstr("Zig-zag CP sharding needs sequence_length (34)")));
}

}  // namespace
