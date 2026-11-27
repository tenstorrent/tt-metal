// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/multihost/single_host_context.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

#include <gtest/gtest.h>

namespace tt::tt_metal::distributed::multihost {
namespace {

TEST(SingleHostContextTest, AllReduceCopiesInput) {
    const SingleHostContext single_host_context;
    const DistributedContext& distributed_context = single_host_context;

    std::array<uint32_t, 4> input = {1, 3, 5, 7};
    std::array<uint32_t, 4> output = {};
    distributed_context.all_reduce<uint32_t>(input, output, ReduceOp::MAX);

    EXPECT_EQ(output, input);

    distributed_context.all_reduce<uint32_t>(input, input, ReduceOp::MAX);
    EXPECT_EQ(input, (std::array<uint32_t, 4>{1, 3, 5, 7}));
}

TEST(SingleHostContextTest, AllReduceRejectsMismatchedBufferSizes) {
    const SingleHostContext single_host_context;
    const DistributedContext& distributed_context = single_host_context;

    std::array<uint32_t, 2> input = {1, 3};
    std::array<uint32_t, 1> output = {};
    EXPECT_THROW(distributed_context.all_reduce<uint32_t>(input, output, ReduceOp::MAX), std::runtime_error);
}

TEST(SingleHostContextTest, AllReduceRejectsBufferNotMultipleOfElementSize) {
    const SingleHostContext single_host_context;
    const DistributedContext& distributed_context = single_host_context;

    std::array<std::byte, 6> buffer = {};
    const ttsl::Span<std::byte> bytes{buffer.data(), buffer.size()};
    EXPECT_THROW(distributed_context.all_reduce(bytes, bytes, ReduceOp::MAX, DType::UINT32), std::runtime_error);
}

TEST(SingleHostContextTest, RootedCollectivesBehaveAsOneRankMpi) {
    const SingleHostContext single_host_context;
    const DistributedContext& distributed_context = single_host_context;

    std::array<std::byte, 4> input = {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    std::array<std::byte, 4> output = {};
    const ttsl::Span<std::byte> in{input.data(), input.size()};
    const ttsl::Span<std::byte> out{output.data(), output.size()};

    EXPECT_NO_THROW(distributed_context.broadcast(in, Rank{0}));
    distributed_context.gather(in, out, Rank{0});
    EXPECT_EQ(output, input);
    output = {};
    distributed_context.scatter(in, out, Rank{0});
    EXPECT_EQ(output, input);

    EXPECT_THROW(distributed_context.broadcast(in, Rank{1}), std::runtime_error);
    EXPECT_THROW(distributed_context.gather(in, out, Rank{1}), std::runtime_error);
    EXPECT_THROW(distributed_context.scatter(in, out, Rank{1}), std::runtime_error);
}

TEST(SingleHostContextTest, ReductionsReturnTheInput) {
    const SingleHostContext single_host_context;
    const DistributedContext& distributed_context = single_host_context;

    std::array<uint32_t, 3> input = {4, 5, 6};
    std::array<uint32_t, 3> output = {};
    distributed_context.reduce<uint32_t>(input, output, ReduceOp::SUM, Rank{0});
    EXPECT_EQ(output, input);
    output = {};
    distributed_context.reduce_scatter<uint32_t>(input, output, ReduceOp::SUM);
    EXPECT_EQ(output, input);
    output = {};
    distributed_context.scan<uint32_t>(input, output, ReduceOp::SUM);
    EXPECT_EQ(output, input);
    EXPECT_THROW(distributed_context.reduce<uint32_t>(input, output, ReduceOp::SUM, Rank{1}), std::runtime_error);
}

TEST(SingleHostContextTest, AllToAllCopiesAndRejectsMismatchedSizes) {
    const SingleHostContext single_host_context;
    const DistributedContext& distributed_context = single_host_context;

    std::array<std::byte, 2> input = {std::byte{7}, std::byte{8}};
    std::array<std::byte, 2> output = {};
    distributed_context.all_to_all({input.data(), input.size()}, {output.data(), output.size()});
    EXPECT_EQ(output, input);

    std::array<std::byte, 1> short_output = {};
    EXPECT_THROW(
        distributed_context.all_to_all({input.data(), input.size()}, {short_output.data(), short_output.size()}),
        std::runtime_error);
}

TEST(SingleHostContextTest, CommunicatorManagementStaysSingleRank) {
    const SingleHostContext single_host_context;

    const ContextPtr duplicate = single_host_context.duplicate();
    ASSERT_NE(duplicate, nullptr);
    EXPECT_EQ(*duplicate->size(), 1);
    EXPECT_NE(duplicate->id(), single_host_context.id());

    std::array<int, 1> self = {0};
    const ContextPtr sub_context = single_host_context.create_sub_context(self);
    ASSERT_NE(sub_context, nullptr);
    EXPECT_EQ(*sub_context->size(), 1);
    std::array<int, 2> two_ranks = {0, 1};
    EXPECT_THROW((void)single_host_context.create_sub_context(two_ranks), std::runtime_error);

    std::array<int, 1> translated = {-1};
    single_host_context.translate_ranks_to_other_ctx(self, duplicate, translated);
    EXPECT_EQ(translated[0], 0);
    std::array<int, 1> other_rank = {1};
    EXPECT_THROW(
        single_host_context.translate_ranks_to_other_ctx(other_rank, duplicate, translated), std::runtime_error);
}

TEST(SingleHostContextTest, PointToPointIsUnsupported) {
    const SingleHostContext single_host_context;
    std::array<std::byte, 1> buffer = {};
    const ttsl::Span<std::byte> bytes{buffer.data(), buffer.size()};
    EXPECT_THROW(single_host_context.send(bytes, Rank{0}, Tag{0}), std::runtime_error);
    EXPECT_THROW(single_host_context.recv(bytes, Rank{0}, Tag{0}), std::runtime_error);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::multihost
