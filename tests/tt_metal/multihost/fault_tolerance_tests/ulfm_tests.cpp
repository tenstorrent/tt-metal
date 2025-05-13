// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed_context.hpp>
#include <gtest/gtest.h>
#include "common/multihost_asserts.hpp"

TEST(FaultTolerance, shrink_after_rank_failure) {
    using tt::tt_metal::distributed::multihost::DistributedContext;
    using tt::tt_metal::distributed::multihost::DistributedException;
    using tt::tt_metal::distributed::multihost::Rank;

    //----------------------------------------------------------------------
    // 0 · Create world communicator and install MPI_ERRORS_RETURN
    //----------------------------------------------------------------------
    auto ctx = DistributedContext::create(0, nullptr);

    const int world = *ctx->size();
    const int me = *ctx->rank();
    const int victim_rank = 1;  // rank to kill (if exists)

    //----------------------------------------------------------------------
    // 1 · Simulate a hard failure on one rank
    //----------------------------------------------------------------------
    if (world > 1 && me == victim_rank) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        // Use SIGKILL so only *this* process dies; MPI_Abort would kill all
        raise(SIGKILL);  // never returns
    }

    //----------------------------------------------------------------------
    // 2 · First collective detects the failure (throws DistributedException)
    //----------------------------------------------------------------------
    try {
        ctx->barrier();  // may throw or return error
    } catch (const DistributedException& e) {
        // 2a · Repair the communicator for every *surviving* rank
        ctx->revoke_and_shrink();
    }

    //----------------------------------------------------------------------
    // 3 · All survivors now run on the shrunken communicator
    //----------------------------------------------------------------------
    const int new_world = *ctx->size();
    const int lost = world - new_world;

    EXPECT_EQ(lost, world > 1 ? 1 : 0);   // exactly one failed if >1 ranks
    EXPECT_EQ_ALL_RANKS(new_world, ctx);  // survivors agree
                                          // (macro from sync_helpers.hpp)

    //----------------------------------------------------------------------
    // 4 · Final barrier on the repaired communicator
    //----------------------------------------------------------------------
    ctx->barrier();
}
