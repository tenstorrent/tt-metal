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

}  // namespace
}  // namespace tt::tt_metal::distributed::multihost
