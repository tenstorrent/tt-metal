// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed_context.hpp>
#include <gtest/gtest.h>
#include "common/multihost_test_tools.hpp"

TEST(FaultTolerance, shrink_after_rank_failure) {
    using tt::tt_metal::distributed::multihost::DistributedContext;
    using tt::tt_metal::distributed::multihost::DistributedException;
    using tt::tt_metal::distributed::multihost::Rank;

    //----------------------------------------------------------------------
    // 0 · Create world communicator and install MPI_ERRORS_RETURN
    //----------------------------------------------------------------------
    auto ctx = DistributedContext::get_current_world();

    const int world = *ctx->size();
    const int self_rank = *ctx->rank();
    const int victim_rank = 1;  // rank to kill (if exists)

    //----------------------------------------------------------------------
    // 1 · Simulate a hard failure on one rank
    //----------------------------------------------------------------------
    if (world > 1 && self_rank == victim_rank) {
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

        // need to make sure other threads don't use the same context,
        // good idea to make a duplication of the context so each thread has its own comm.
        ctx->revoke_and_shrink();

        // we know that our ctx is okay. Now need to check the world one.
        // If world is revoked we need to set a new world communicator
        // In our case ctx is the world communicator
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
