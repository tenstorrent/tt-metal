// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <tt-metalium/distributed_context.hpp>
#include <gtest/gtest.h>
#include "common/multihost_test_tools.hpp"
#include <thread>

using tt::tt_metal::distributed::multihost::Color;
using tt::tt_metal::distributed::multihost::DistributedContext;
using tt::tt_metal::distributed::multihost::DistributedException;
using tt::tt_metal::distributed::multihost::Key;
using tt::tt_metal::distributed::multihost::Rank;

TEST(FaultTolerance, ShrinkAfterRankFailure) {
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

TEST(FaultTolerance, DisableBrokenBlock) {
    // ‑‑ configuration ------------------------------------------------------
    constexpr int ranks_per_block = 2;  // two ranks share one machine / block
    constexpr int victim_block = 1;     // second block (ranks 2 & 3) – adjust at will

    auto ctx = DistributedContext::get_current_world();
    const int world = *ctx->size();
    const int rank = *ctx->rank();

    if (world < ranks_per_block * 2) {
        GTEST_SKIP() << "Need at least " << ranks_per_block * 2 << " processes";
    }

    const int victim_rank = victim_block * ranks_per_block;  // first rank in that block

    // ‑‑ 1 · Simulate a hard failure on one rank ---------------------------
    if (rank == victim_rank) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        raise(SIGKILL);  // never returns
    }

    // ‑‑ 2 · First collective detects failure ------------------------------
    try {
        ctx->barrier();  // will throw / error out on surviving ranks
    } catch (const DistributedException&) {
        ctx->revoke_and_shrink();
    }

    // After shrink the failed rank is gone but its partner (same block)
    // may still be alive. We want to disable the entire block.

    // ‑‑ 3 · Build sub‑communicator for our logical block ------------------
    const int block_id = rank / ranks_per_block;
    auto block_ctx = ctx->split(Color{block_id}, Key{0});

    const bool is_block_healthy = (*block_ctx->size() == ranks_per_block);

    // ‑‑ 4 · Split off unhealthy blocks (color 0 = healthy, 1 = unhealthy) -
    auto filtered_ctx = ctx->split(Color{is_block_healthy ? 0 : 1}, Key{0});

    if (!is_block_healthy) {
        // This rank is in a disabled block – no further collectives.
        SUCCEED();
        return;
    }

    ctx = filtered_ctx;  // continue with communicator containing only healthy blocks

    // ‑‑ 5 · Verify global properties -------------------------------------
    const int expected_blocks = world / ranks_per_block - 1;  // one block lost
    const int expected_ranks = expected_blocks * ranks_per_block;
    const int new_world = *ctx->size();

    EXPECT_EQ(new_world, expected_ranks);

    // All surviving ranks should agree on the new size
    ctx->barrier();

    int size_val = new_world;
    std::vector<int> gathered(expected_ranks);
    ctx->all_gather(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&size_val), sizeof(int)),
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(gathered.data()), gathered.size() * sizeof(int)));
    for (int v : gathered) {
        EXPECT_EQ(v, new_world);
    }

    // ‑‑ 6 · Final collective sanity check --------------------------------
    ctx->barrier();
}
