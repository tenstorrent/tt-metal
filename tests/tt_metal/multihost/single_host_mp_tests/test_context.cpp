// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed_context.hpp>
#include <gtest/gtest.h>

TEST(DistributedContextTest, TestSendRecv) {
    // assuming context is already initialized in main with argc, argv
    // in this case we will just get a world context
    auto context = tt::tt_metal::distributed::multihost::DistributedContext::create(0, nullptr);
    using Rank = tt::tt_metal::distributed::multihost::Rank;
    using Tag = tt::tt_metal::distributed::multihost::Tag;
    using Size = tt::tt_metal::distributed::multihost::Size;

    auto size = context->size();

    ASSERT_TRUE(*size > 1);

    std::vector<std::byte> orig_bytes(10);

    for (int i = 0; i < 10; ++i) {
        orig_bytes[i] = static_cast<std::byte>(i);
    }
    if (*context->rank() == 0) {
        tt::stl::Span<std::byte> view(orig_bytes.data(), orig_bytes.size());
        context->send(view, Rank{1}, Tag{0});
    } else if (*context->rank() == 1) {
        std::vector<std::byte> bytes(10);
        tt::stl::Span<std::byte> view(bytes.data(), bytes.size());
        context->recv(view, Rank{0}, Tag{0});
        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(orig_bytes[i], bytes[i]);
        }
    }
}

TEST(DistributedContextTest, AllReduceInt) {
    // assuming context is already initialized in main with argc, argv
    // in this case we will just get a world context
    auto context = tt::tt_metal::distributed::multihost::DistributedContext::create(0, nullptr);
    using Rank = tt::tt_metal::distributed::multihost::Rank;
    using Tag = tt::tt_metal::distributed::multihost::Tag;
    using Size = tt::tt_metal::distributed::multihost::Size;

    auto size = context->size();

    ASSERT_TRUE(*size > 1);
    std::vector<int> data(10);
    for (int i = 0; i < 10; ++i) {
        data[i] = *context->rank() + i;
    }
    std::vector<int> result(10);
    tt::stl::Span<int> view(data.data(), data.size());
    tt::stl::Span<int> result_view(result.data(), result.size());
    context->all_reduce(view, result_view, tt::tt_metal::distributed::multihost::ReduceOp::SUM);
    for (int i = 0; i < 10; ++i) {
        int expected = 0;
        for (int j = 0; j < *size; ++j) {
            expected += j + i;
        }
        EXPECT_EQ(expected, result[i]);
    }
}
