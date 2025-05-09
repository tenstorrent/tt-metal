// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed_context.hpp>
#include <gtest/gtest.h>
#include "common/multihost_asserts.hpp"

TEST(FaultTolerance, shrink_after_rank_failure) {
    // -------- initial communicator wrapped by DistributedContext ----------
    auto ctx = tt::tt_metal::distributed::multihost::DistributedContext::create(0, nullptr);
    using DistributedException = tt::tt_metal::distributed::multihost::DistributedException;
    using Rank = tt::tt_metal::distributed::multihost::Rank;
    using Tag = tt::tt_metal::distributed::multihost::Tag;
    using Size = tt::tt_metal::distributed::multihost::Size;
    const int world = *ctx->size();
    const int me = *ctx->rank();
    constexpr int kill_rank = 1;

    // -------- induce failure on 'kill_rank' ------------------------------
    if (me == kill_rank) {
        // Give everybody else time to reach the barrier → clean fail pattern
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        ctx->abort(322);  // never returns
    }

    try {
        ctx->barrier();  // ULFM barrier handshake
    } catch (const DistributedException& e) {
        ctx->revoke_and_shrink();

        const int new_world = *ctx->size();
        const int lost = world - new_world;

        // -------- assertions --------------------------------------------------
        EXPECT_EQ(lost, 1);                   // exactly one failed
        EXPECT_EQ_ALL_RANKS(new_world, ctx);  // survivors agree on size
    }

    // Ranks < kill_rank survive; others died.
    // First collective after the death must detect the revoke.

    // Barrier on new communicator to ensure test exits cleanly
    ctx->barrier();
}
