// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed_context.hpp>
#include <gtest/gtest.h>
#include <numeric>
#include <thread>

namespace {
using Rank = tt::tt_metal::distributed::multihost::Rank;
using Tag = tt::tt_metal::distributed::multihost::Tag;
using Size = tt::tt_metal::distributed::multihost::Size;
using DistributedContext = tt::tt_metal::distributed::multihost::DistributedContext;
using ReduceOp = tt::tt_metal::distributed::multihost::ReduceOp;
using Color = tt::tt_metal::distributed::multihost::Color;
using Key = tt::tt_metal::distributed::multihost::Key;
using RequestPtr = std::shared_ptr<tt::tt_metal::distributed::multihost::Request>;
using ContextPtr = std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>;

tt::stl::Span<std::byte> int_to_byte_span(int* val_ptr) {
    return tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(val_ptr), sizeof(int));
}
}  // namespace

TEST(DistributedContextTest, TestSendRecv) {
    // assuming context is already initialized in main with argc, argv
    // in this case we will just get a world context
    auto context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();

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
    auto context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
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

// -----------------------------------------------------------------------------
// Basic synchronisation --------------------------------------------------------
// -----------------------------------------------------------------------------
TEST(DistributedContextExtraTest, BarrierSynchronisation) {
    auto ctx = DistributedContext::get_current_world();
    ASSERT_TRUE(*ctx->size() > 1);
    // Rank 0 sleeps for a short time; others hit barrier immediately.
    if (*ctx->rank() == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ctx->barrier();  // should return roughly at the same time for all ranks.
    SUCCEED();       // absence of dead‑lock == pass.
}

// -----------------------------------------------------------------------------
// Broadcast --------------------------------------------------------------------
// -----------------------------------------------------------------------------
TEST(DistributedContextExtraTest, BroadcastVectorInt) {
    auto ctx = DistributedContext::get_current_world();
    ASSERT_TRUE(*ctx->size() > 1);

    std::vector<int> buffer(8);
    if (*ctx->rank() == 0) {
        std::iota(buffer.begin(), buffer.end(), 42);  // 42,43,…
    }
    auto span_bytes = tt::stl::as_writable_bytes(tt::stl::Span<int>{buffer.data(), buffer.size()});
    ctx->broadcast(span_bytes, Rank{0});

    for (std::size_t i = 0; i < buffer.size(); ++i) {
        EXPECT_EQ(buffer[i], 42 + static_cast<int>(i));
    }
}

// -----------------------------------------------------------------------------
// Gather / Scatter -------------------------------------------------------------
// -----------------------------------------------------------------------------
TEST(DistributedContextExtraTest, GatherIntToRoot) {
    auto ctx = DistributedContext::get_current_world();
    auto world_size = *ctx->size();
    ASSERT_GT(world_size, 1);

    int send_val = static_cast<int>(*ctx->rank());
    std::vector<int> recv_buf(world_size, -1);

    auto send_span = int_to_byte_span(&send_val);
    auto recv_span = tt::stl::as_writable_bytes(tt::stl::Span<int>{recv_buf.data(), recv_buf.size()});

    ctx->gather(send_span, recv_span, Rank{0});

    if (*ctx->rank() == 0) {
        for (int r = 0; r < world_size; ++r) {
            EXPECT_EQ(recv_buf[r], r);
        }
    }
}

TEST(DistributedContextExtraTest, ScatterIntFromRoot) {
    auto ctx = DistributedContext::get_current_world();
    auto world_size = *ctx->size();
    ASSERT_GT(world_size, 1);

    std::vector<int> send_vec;  // valid only on root
    int recv_val = -1;
    if (*ctx->rank() == 0) {
        send_vec.resize(world_size);
        std::iota(send_vec.begin(), send_vec.end(), 100);  // 100,101,…
    }
    auto send_span = *ctx->rank() == 0
                         ? tt::stl::as_writable_bytes(tt::stl::Span<int>{send_vec.data(), send_vec.size()})
                         : tt::stl::Span<std::byte>{};
    auto recv_span = int_to_byte_span(&recv_val);

    ctx->scatter(send_span, recv_span, Rank{0});
    EXPECT_EQ(recv_val, 100 + *ctx->rank());
}

// -----------------------------------------------------------------------------
// All‑gather / All‑to‑all -------------------------------------------------------
// -----------------------------------------------------------------------------
TEST(DistributedContextExtraTest, AllGatherRankInt) {
    auto ctx = DistributedContext::get_current_world();
    int world_size = *ctx->size();
    ASSERT_GT(world_size, 1);

    int send_val = *ctx->rank();
    std::vector<int> recv_buf(world_size);

    ctx->all_gather(
        int_to_byte_span(&send_val), tt::stl::as_writable_bytes(tt::stl::Span<int>{recv_buf.data(), recv_buf.size()}));

    for (int r = 0; r < world_size; ++r) {
        EXPECT_EQ(recv_buf[r], r);
    }
}

TEST(DistributedContextExtraTest, AllToAllInt) {
    auto ctx = DistributedContext::get_current_world();
    int world_size = *ctx->size();
    ASSERT_GT(world_size, 1);

    // Each rank prepares vector [rank*100 + 0, rank*100 + 1, …]
    std::vector<int> send_buf(world_size);
    std::vector<int> recv_buf(world_size, -1);
    for (int i = 0; i < world_size; ++i) {
        send_buf[i] = *ctx->rank() * 100 + i;
    }

    ctx->all_to_all(
        tt::stl::as_writable_bytes(tt::stl::Span<int>{send_buf.data(), send_buf.size()}),
        tt::stl::as_writable_bytes(tt::stl::Span<int>{recv_buf.data(), recv_buf.size()}));

    for (int i = 0; i < world_size; ++i) {
        EXPECT_EQ(recv_buf[i], i * 100 + *ctx->rank());
    }
}

// -----------------------------------------------------------------------------
// Reduce / Reduce‑scatter / Scan (typed wrappers) ------------------------------
// -----------------------------------------------------------------------------
TEST(DistributedContextExtraTest, ReduceSumIntToRoot) {
    auto ctx = DistributedContext::get_current_world();
    int world_size = *ctx->size();
    ASSERT_GT(world_size, 1);

    int send_val = *ctx->rank() + 1;  // 1..N
    int recv_val = 0;

    ctx->reduce<int>({&send_val, 1}, {&recv_val, 1}, ReduceOp::SUM, Rank{0});

    if (*ctx->rank() == 0) {
        EXPECT_EQ(recv_val, world_size * (world_size + 1) / 2);
    }
}

TEST(DistributedContextExtraTest, ReduceScatterSumInt) {
    auto ctx = DistributedContext::get_current_world();
    int world_size = *ctx->size();
    ASSERT_GT(world_size, 1);

    // Each rank contributes vector [rank, rank, …]
    std::vector<int> send_buf(world_size, *ctx->rank());
    int recv_val = -1;

    ctx->reduce_scatter<int>({send_buf.data(), send_buf.size()}, {&recv_val, 1}, ReduceOp::SUM);

    // Expected: sum over ranks of value at index == rank
    int expected = world_size * (world_size - 1) / 2;  // 0+1+…
    EXPECT_EQ(recv_val, expected);
}

TEST(DistributedContextExtraTest, PrefixScanSumInt) {
    auto ctx = DistributedContext::get_current_world();
    int world_size = *ctx->size();
    ASSERT_GT(world_size, 1);

    int send_val = 1;  // everyone contributes 1
    int recv_val = 0;

    ctx->scan<int>({&send_val, 1}, {&recv_val, 1}, ReduceOp::SUM);

    // Rank r should get r+1 (1‑based scan)
    EXPECT_EQ(recv_val, *ctx->rank() + 1);
}

// -----------------------------------------------------------------------------
// Non‑blocking P2P -------------------------------------------------------------
// -----------------------------------------------------------------------------
TEST(DistributedContextExtraTest, IsendIrecvWait) {
    auto ctx = DistributedContext::get_current_world();
    ASSERT_TRUE(*ctx->size() > 1);

    std::array<int, 4> buf_send{};
    std::array<int, 4> buf_recv{};
    for (int i = 0; i < 4; ++i) {
        buf_send[i] = i + (*ctx->rank() * 10);
    }

    if (*ctx->rank() == 0) {
        auto req = ctx->isend(
            tt::stl::as_writable_bytes(tt::stl::Span<int>{buf_send.data(), buf_send.size()}), Rank{1}, Tag{7});
        auto status = req->wait();
        // EXPECT_EQ(status.count, buf_send.size());
    } else if (*ctx->rank() == 1) {
        auto req = ctx->irecv(
            tt::stl::as_writable_bytes(tt::stl::Span<int>{buf_recv.data(), buf_recv.size()}), Rank{0}, Tag{7});
        auto status = req->wait();
        // EXPECT_EQ(status.count, buf_send.size());
        for (int i = 0; i < 4; ++i) {
            EXPECT_EQ(buf_recv[i], i);
        }
    }
}

TEST(DistributedContextExtraTest, IsendIrecvTestPolling) {
    auto ctx = DistributedContext::get_current_world();
    ASSERT_TRUE(*ctx->size() > 1);

    int send_val = 1234;
    int recv_val = 0;

    if (*ctx->rank() == 0) {
        auto req = ctx->isend(int_to_byte_span(&send_val), Rank{1}, Tag{8});
        // Busy‑poll until completed.
        while (!req->test()) {
            std::this_thread::yield();
        }
    } else if (*ctx->rank() == 1) {
        auto req = ctx->irecv(int_to_byte_span(&recv_val), Rank{0}, Tag{8});
        while (!req->test()) {
            std::this_thread::yield();
        }
        EXPECT_EQ(recv_val, 1234);
    }
}

// -----------------------------------------------------------------------------
// Communicator management ------------------------------------------------------
// -----------------------------------------------------------------------------

TEST(DistributedContextExtraTest, DuplicateCommunicator) {
    auto world = DistributedContext::get_current_world();
    auto dup = world->duplicate();

    EXPECT_EQ(*world->size(), *dup->size());
    EXPECT_EQ(*world->rank(), *dup->rank());

    // Simple send/recv inside duplicate to ensure it is functional.
    int send_val = *dup->rank();
    int recv_val = 0;
    if (*dup->rank() == 0) {
        dup->send(int_to_byte_span(&send_val), Rank{1}, Tag{9});
    } else if (*dup->rank() == 1) {
        dup->recv(int_to_byte_span(&recv_val), Rank{0}, Tag{9});
        EXPECT_EQ(recv_val, 0);
    }
}

TEST(DistributedContextExtraTest, SplitCommunicatorEvenOdd) {
    auto world = DistributedContext::get_current_world();
    int color = *world->rank() % 2;  // 0 = even, 1 = odd
    auto sub = world->split(Color{color}, Key{0});

    // Each sub‑comm should have roughly half the size (±1).
    int expected = (*world->size() + 1 - color) / 2;  // ceil or floor
    EXPECT_EQ(*sub->size(), expected);
}

TEST(DistributedContextExtraTest, CreateSubContextTranslateRanks) {
    auto world = DistributedContext::get_current_world();
    int world_size = *world->size();
    int world_rank = *world->rank();
    ASSERT_GT(world_size, 2);

    // Sub‑context containing ranks 0,1,2 (or all if smaller)
    std::array<int, 3> group = {0, 1, 2};
    if (world_rank > 2) {
        // skiping this test for ranks which are not in the group
        SUCCEED();
        return;
    }
    auto sub = world->create_sub_context({group.data(), 3});

    // Translate sub ranks back to world.
    std::array<int, 3> translated = {-1, -1, -1};
    sub->translate_ranks_to_other_ctx({group.data(), 3}, world, {translated.data(), 3});

    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(translated[i], group[i]);
    }
}
