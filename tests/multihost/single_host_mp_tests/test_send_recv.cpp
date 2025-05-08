// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed_context.hpp>
#include <gtest/gtest.h>

TEST(DistributedContextTest, TestSendRecv) {
    // assuming context is already initialized in main with argc, argv
    // in this case we will jsut get a world context
    auto context = tt::tt_metal::distributed::multihost::DistributedContext::create(0, nullptr);
    using Rank = tt::tt_metal::distributed::multihost::Rank;
    using Tag = tt::tt_metal::distributed::multihost::Tag;
    using Size = tt::tt_metal::distributed::multihost::Size;

    auto size = context->size();

    ASSERT_TRUE(*size > 1, "This test requires at least 2 ranks");
    std::vector<std::byte> orig_bytes(10);
    for (int i = 0; i < 10; ++i) {
        orig_bytes[i] = static_cast<std::byte>(i);
    }
    if (*context->rank() == 0) {
        fmt::print("Sending bytes to rank 1\n");
        tt::stl::Span<std::byte> view(orig_bytes.data(), orig_bytes.size());
        context->send(view, Rank{1}, Tag{0});
    } else {
        fmt::print("Hello from rank {}\n", context->rank());
        std::vector<std::byte> bytes(10);
        fmt::print("Receiving bytes from rank 0\n");
        tt::stl::Span<std::byte> view(bytes.data(), bytes.size());
        context->recv(view, Rank{0}, Tag{0});
        fmt::print("Bytes: ");
        for (size_t i = 0; i < bytes.size(); ++i) {
            fmt::print("{}{}", static_cast<int>(bytes[i]), (i < bytes.size() - 1) ? ", " : "\n");
        }
    }
}
