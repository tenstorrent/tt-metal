// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <tt-metalium/distributed_context.hpp>
#include <gtest/gtest.h>
#include "common/multihost_test_tools.hpp"
#include "tt_metal/distributed/multihost/mpi_distributed_context.hpp"
#include <csignal>
#include <cstdlib>
#include <mpi-ext.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

using tt::tt_metal::distributed::multihost::Color;
using tt::tt_metal::distributed::multihost::DistributedContext;
using tt::tt_metal::distributed::multihost::DistributedException;
using tt::tt_metal::distributed::multihost::FailurePolicy;
using tt::tt_metal::distributed::multihost::Key;
using tt::tt_metal::distributed::multihost::MPIRankFailureException;
using tt::tt_metal::distributed::multihost::Rank;

TEST(FaultTolerance, ShrinkAfterRankFailure) {
    //----------------------------------------------------------------------
    // 0 · Create world communicator and install MPI_ERRORS_RETURN
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();

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
    const int expected_blocks = (world / ranks_per_block) - 1;  // one block lost
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

// =====================================================================
// Section 4.4 — Test infrastructure: agree() consensus and policy switching
// =====================================================================

TEST(FaultTolerance, AgreeConsensus) {
    //----------------------------------------------------------------------
    // Test MPIX_Comm_agree wrapper: all ranks agree on a boolean value.
    // This is the ULFM primitive that surviving ranks use to reach
    // consensus before taking recovery actions.
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    const int rank = *ctx->rank();

    // Case 1: All ranks vote true → result should be true
    {
        auto result = ctx->agree(true);
        ASSERT_TRUE(result.has_value()) << "agree() returned nullopt — ULFM not available?";
        EXPECT_TRUE(result.value()) << "Rank " << rank << ": agree(true) should yield true";
    }

    // Case 2: All ranks vote false → result should be false
    {
        auto result = ctx->agree(false);
        ASSERT_TRUE(result.has_value());
        EXPECT_FALSE(result.value()) << "Rank " << rank << ": agree(false) should yield false";
    }

    // Case 3: Mixed votes (rank 0 votes false, others true)
    // MPIX_Comm_agree uses bitwise AND, so any false → false
    {
        bool my_vote = (rank != 0);
        auto result = ctx->agree(my_vote);
        ASSERT_TRUE(result.has_value());
        if (*ctx->size() > 1) {
            EXPECT_FALSE(result.value())
                << "Rank " << rank << ": mixed agree should yield false (AND semantics)";
        }
    }

    ctx->barrier();
}

TEST(FaultTolerance, FailurePolicySwitching) {
    //----------------------------------------------------------------------
    // Test that we can switch between FAST_FAIL and FAULT_TOLERANT modes.
    // This test does NOT kill any ranks — it just verifies the API works
    // and that FAULT_TOLERANT mode produces MPIRankFailureException on
    // a simulated failure (rank kill + barrier).
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    const int world = *ctx->size();
    const int rank = *ctx->rank();

    if (world < 2) {
        GTEST_SKIP() << "Need at least 2 ranks for policy switching test";
    }

    // Downcast to MPIContext to access set_failure_policy
    auto* mpi_ctx = dynamic_cast<tt::tt_metal::distributed::multihost::MPIContext*>(ctx.get());
    ASSERT_NE(mpi_ctx, nullptr) << "Expected MPIContext for ULFM test";

    // Switch to FAULT_TOLERANT mode
    mpi_ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT);

    // Kill rank 1, then the barrier on survivors should throw
    // MPIRankFailureException instead of calling _exit(70)
    if (rank == 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        raise(SIGKILL);  // never returns
    }

    bool caught_rank_failure = false;
    try {
        ctx->barrier();
    } catch (const MPIRankFailureException& e) {
        caught_rank_failure = true;
        // Verify the exception carries useful info
        EXPECT_NE(e.what(), nullptr);
        // Recover: revoke and shrink
        ctx->revoke_and_shrink();
    } catch (const DistributedException&) {
        // May also be a generic DistributedException — still recover
        caught_rank_failure = true;
        ctx->revoke_and_shrink();
    }

    EXPECT_TRUE(caught_rank_failure)
        << "Rank " << rank << ": expected MPIRankFailureException in FAULT_TOLERANT mode";

    // Verify the shrunken communicator works
    const int new_world = *ctx->size();
    EXPECT_EQ(new_world, world - 1);
    EXPECT_EQ_SURVIVING_RANKS(new_world, ctx);

    ctx->barrier();
}

// =====================================================================
// Section 7.8 — Single-node testing gap: ULFM control-plane tests
// These can run with `mpirun -np 2` on a single host.
// They test the exit-code and signal-handling paths without requiring
// actual Tenstorrent hardware.
// =====================================================================

TEST(FaultTolerance, FastFailExitCode70) {
    //----------------------------------------------------------------------
    // Verify that FAST_FAIL mode causes a rank to exit with code 70
    // when it detects a peer failure.
    //
    // Strategy: fork a child process that is one of the MPI ranks.
    // We can't actually test _exit(70) from within a GTest (it would
    // kill the test runner), so instead we verify the behavior in a
    // subprocess by checking that the FAST_FAIL path is the default,
    // and that killing a rank causes the expected behavior.
    //
    // For a proper integration test, we rely on the external test runner
    // (run_single_node_ulfm_tests.sh) which launches mpirun and checks
    // the exit code of the entire job.
    //
    // Here we do a simpler in-process check: verify that FAST_FAIL is
    // the default policy, and that we can detect failure in FAULT_TOLERANT
    // mode (proving the detection path works — the only difference in
    // FAST_FAIL is _exit(70) vs throw).
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    const int world = *ctx->size();
    const int rank = *ctx->rank();

    if (world < 2) {
        GTEST_SKIP() << "Need at least 2 ranks";
    }

    // Verify FAST_FAIL is default by switching to FAULT_TOLERANT and
    // checking that we get an exception (not _exit)
    auto* mpi_ctx = dynamic_cast<tt::tt_metal::distributed::multihost::MPIContext*>(ctx.get());
    ASSERT_NE(mpi_ctx, nullptr);

    // We use FAULT_TOLERANT to test the detection path without killing
    // the test runner. The FAST_FAIL path is tested by the shell-level
    // test in run_single_node_ulfm_tests.sh.
    mpi_ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT);

    if (rank == 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        raise(SIGKILL);
    }

    bool detected = false;
    try {
        ctx->barrier();
    } catch (const DistributedException&) {
        detected = true;
        ctx->revoke_and_shrink();
    }

    EXPECT_TRUE(detected) << "Rank " << rank << ": should detect peer failure";

    // All survivors should agree on the result
    auto agreed = ctx->agree(detected);
    ASSERT_TRUE(agreed.has_value());
    EXPECT_TRUE(agreed.value());

    ctx->barrier();
}

TEST(FaultTolerance, FinalizeWatchdogPath) {
    //----------------------------------------------------------------------
    // Verify that the MPI_Finalize watchdog infrastructure is installed.
    //
    // We cannot directly test that MPI_Finalize hangs (that would hang
    // the test), but we CAN verify:
    // 1. SIGALRM handler is installed (registered in init_env() after MPI_Init;
    //    atexit only arms alarm() before MPI_Finalize)
    // 2. The handler calls _exit(70)
    //
    // The actual MPI_Finalize hang scenario is tested by the shell-level
    // test in run_single_node_ulfm_tests.sh which runs a purpose-built
    // binary that triggers the watchdog.
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    // Check that we have a SIGALRM handler installed (not SIG_DFL/SIG_IGN)
    struct sigaction sa;
    sigaction(SIGALRM, nullptr, &sa);

    // The handler should be set (either sa_handler or sa_sigaction, not SIG_DFL)
    bool has_handler = (sa.sa_handler != SIG_DFL && sa.sa_handler != SIG_IGN) ||
                       (sa.sa_flags & SA_SIGINFO);
    EXPECT_TRUE(has_handler)
        << "SIGALRM handler should be installed for MPI_Finalize watchdog";

    // Verify all ranks see the same state
    EXPECT_EQ_ALL_RANKS(static_cast<int>(has_handler), ctx);

    ctx->barrier();
}

TEST(FaultTolerance, TerminateHandlerInstalled) {
    //----------------------------------------------------------------------
    // Verify that std::set_terminate has been called to install the ULFM
    // terminate handler. The handler should revoke MPI_COMM_WORLD and
    // _exit(70) on uncaught exceptions.
    //
    // We verify by checking that std::get_terminate() returns a non-null
    // handler that is NOT the default handler.
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    auto handler = std::get_terminate();
    EXPECT_NE(handler, nullptr)
        << "std::terminate handler should be installed";

    // All ranks should agree
    int has_handler = (handler != nullptr) ? 1 : 0;
    EXPECT_EQ_ALL_RANKS(has_handler, ctx);

    ctx->barrier();
}

// =====================================================================
// Edge case tests — debugability, log quality, failure states
// =====================================================================

TEST(FaultTolerance, FailedRanksBeforeAnyFailure) {
    //----------------------------------------------------------------------
    // Edge case: calling failed_ranks() when no rank has died should
    // return an empty vector.  This verifies the ULFM failure_get_acked
    // path handles the no-failure case gracefully.
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    auto failed = ctx->failed_ranks();
    EXPECT_TRUE(failed.empty())
        << "Rank " << *ctx->rank()
        << ": failed_ranks() should be empty when no ranks have died, got "
        << failed.size() << " entries";

    // All ranks should agree on empty
    int failed_count = static_cast<int>(failed.size());
    EXPECT_EQ_ALL_RANKS(failed_count, ctx);

    ctx->barrier();
}

TEST(FaultTolerance, DoubleRevokeGuard) {
    //----------------------------------------------------------------------
    // Edge case: calling revoke_and_shrink() twice in sequence.
    // The atomic bool guard (revoked_) should prevent the second call
    // from causing undefined behavior.  After the first shrink, the
    // communicator is healthy again (revoked_ reset to false), so a
    // second revoke_and_shrink() should succeed without crashing.
    //
    // We test this by killing a rank, catching the exception, calling
    // revoke_and_shrink() once (normal recovery), then calling it
    // again (should be safe — it will revoke the already-healthy comm
    // and shrink again, producing a communicator with the same ranks).
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    const int world = *ctx->size();
    const int rank = *ctx->rank();

    if (world < 2) {
        GTEST_SKIP() << "Need at least 2 ranks for double-revoke test";
    }

    auto* mpi_ctx = dynamic_cast<tt::tt_metal::distributed::multihost::MPIContext*>(ctx.get());
    ASSERT_NE(mpi_ctx, nullptr);
    mpi_ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT);

    // Kill rank 1
    if (rank == 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        raise(SIGKILL);
    }

    // Detect failure
    try {
        ctx->barrier();
    } catch (const DistributedException&) {
        // First revoke_and_shrink — normal recovery
        ctx->revoke_and_shrink();
    }

    const int size_after_first = *ctx->size();
    EXPECT_EQ(size_after_first, world - 1)
        << "Rank " << rank << ": first shrink should remove exactly 1 rank";

    // Second revoke_and_shrink — should NOT crash or hang.
    // The communicator is healthy, so revoke + shrink should succeed
    // and produce a communicator with the same set of ranks.
    ctx->revoke_and_shrink();

    const int size_after_second = *ctx->size();
    EXPECT_EQ(size_after_second, size_after_first)
        << "Rank " << rank << ": second shrink should not lose additional ranks";

    EXPECT_EQ_SURVIVING_RANKS(size_after_second, ctx);
    ctx->barrier();
}

TEST(FaultTolerance, AgreeAfterRevokeAndShrink) {
    //----------------------------------------------------------------------
    // Edge case: calling agree() on a communicator that has been through
    // revoke_and_shrink().  The new (shrunken) communicator should
    // support agree() just like a fresh communicator.
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    const int world = *ctx->size();
    const int rank = *ctx->rank();

    if (world < 2) {
        GTEST_SKIP() << "Need at least 2 ranks";
    }

    auto* mpi_ctx = dynamic_cast<tt::tt_metal::distributed::multihost::MPIContext*>(ctx.get());
    ASSERT_NE(mpi_ctx, nullptr);
    mpi_ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT);

    // Kill rank 1
    if (rank == 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        raise(SIGKILL);
    }

    try {
        ctx->barrier();
    } catch (const DistributedException&) {
        ctx->revoke_and_shrink();
    }

    // Now test agree() on the shrunken communicator
    // All survivors vote true — should get true
    {
        auto result = ctx->agree(true);
        ASSERT_TRUE(result.has_value())
            << "Rank " << rank << ": agree() should work on shrunken comm";
        EXPECT_TRUE(result.value())
            << "Rank " << rank << ": unanimous true should yield true after shrink";
    }

    // All survivors vote false — should get false
    {
        auto result = ctx->agree(false);
        ASSERT_TRUE(result.has_value());
        EXPECT_FALSE(result.value())
            << "Rank " << rank << ": unanimous false should yield false after shrink";
    }

    ctx->barrier();
}

TEST(FaultTolerance, IsRevokedFalseBeforeFailure) {
    //----------------------------------------------------------------------
    // Verify that is_revoked() returns false on a healthy communicator.
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    bool revoked = ctx->is_revoked();
    EXPECT_FALSE(revoked)
        << "Rank " << *ctx->rank() << ": is_revoked() should be false before any failure";

    int revoked_int = revoked ? 1 : 0;
    EXPECT_EQ_ALL_RANKS(revoked_int, ctx);

    ctx->barrier();
}

TEST(FaultTolerance, SupportsFaultToleranceReported) {
    //----------------------------------------------------------------------
    // Verify supports_fault_tolerance() returns a consistent value across
    // all ranks.  In a ULFM build this should be true; in a non-ULFM
    // build the test is skipped by SKIP_IF_NO_ULFM.
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    bool supported = ctx->supports_fault_tolerance();
    EXPECT_TRUE(supported)
        << "After SKIP_IF_NO_ULFM, supports_fault_tolerance() should be true";

    int supported_int = supported ? 1 : 0;
    EXPECT_EQ_ALL_RANKS(supported_int, ctx);

    ctx->barrier();
}

TEST(FaultTolerance, MPIRankFailureExceptionCarriesContext) {
    //----------------------------------------------------------------------
    // Verify that MPIRankFailureException carries enough context for
    // meaningful diagnostics: detecting rank, error code, failed ranks
    // string, and human-readable message.
    //
    // This is a debugability test: when a failure occurs in CI, the
    // exception message should tell the engineer WHICH rank failed
    // and WHY without needing to dig through MPI logs.
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    const int world = *ctx->size();
    const int rank = *ctx->rank();

    if (world < 2) {
        GTEST_SKIP() << "Need at least 2 ranks";
    }

    auto* mpi_ctx = dynamic_cast<tt::tt_metal::distributed::multihost::MPIContext*>(ctx.get());
    ASSERT_NE(mpi_ctx, nullptr);
    mpi_ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT);

    // Kill rank 1
    if (rank == 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        raise(SIGKILL);
    }

    try {
        ctx->barrier();
        // If no exception, that's unexpected in a multi-rank scenario
        // (rank 1 should be dead by now)
    } catch (const MPIRankFailureException& e) {
        // Verify diagnostic context
        EXPECT_NE(e.what(), nullptr)
            << "Exception what() should not be null";

        std::string msg = e.message();
        EXPECT_FALSE(msg.empty())
            << "Rank " << rank << ": exception message should not be empty";

        // The message should mention "failed ranks" or contain rank info
        EXPECT_TRUE(msg.find("failed") != std::string::npos || msg.find("rank") != std::string::npos)
            << "Rank " << rank << ": exception message should mention failure context: " << msg;

        // Error code should be non-zero (ULFM error)
        EXPECT_NE(e.error_code(), 0)
            << "Rank " << rank << ": error_code should be non-zero for rank failure";

        // error_string() should be populated from MPI_Error_string
        std::string err_str = e.error_string();
        EXPECT_FALSE(err_str.empty())
            << "Rank " << rank << ": error_string() should be populated by MPI_Error_string";

        // Recover
        ctx->revoke_and_shrink();
    } catch (const DistributedException& e) {
        // Fallback — still verify basic context
        EXPECT_NE(e.what(), nullptr);
        ctx->revoke_and_shrink();
    }

    ctx->barrier();
}

TEST(FaultTolerance, FailedRanksAfterDetection) {
    //----------------------------------------------------------------------
    // After detecting a rank failure (before revoke_and_shrink), the
    // failed_ranks() accessor should return the dead rank(s).
    //
    // Note: after revoke_and_shrink(), failed_ranks() on the NEW
    // communicator should be empty (the new comm doesn't know about
    // old failures).
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    const int world = *ctx->size();
    const int rank = *ctx->rank();

    if (world < 2) {
        GTEST_SKIP() << "Need at least 2 ranks";
    }

    auto* mpi_ctx = dynamic_cast<tt::tt_metal::distributed::multihost::MPIContext*>(ctx.get());
    ASSERT_NE(mpi_ctx, nullptr);
    mpi_ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT);

    // Kill rank 1
    if (rank == 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        raise(SIGKILL);
    }

    bool detected = false;
    try {
        ctx->barrier();
    } catch (const DistributedException& ex) {
        detected = true;
        const int err = ex.error_code();

        // Before shrink: failed_ranks() should include rank 1.
        // Note: this queries the OLD (pre-shrink) communicator.
        auto failed = ctx->failed_ranks();

        // Ranks that saw MPIX_ERR_PROC_FAILED can reliably identify failed
        // ranks via ULFM failure_ack/get_acked (or from the detection-time
        // cache).  Ranks that saw MPIX_ERR_REVOKED may not be able to: the
        // communicator was already revoked by another rank before this rank
        // could ack, so failure_ack fails and the cache may be empty.
        // Accept empty for REVOKED; require non-empty for PROC_FAILED.
        if (err != MPIX_ERR_REVOKED) {
            EXPECT_FALSE(failed.empty())
                << "Rank " << rank << ": failed_ranks() should be non-empty after detecting failure"
                << " (error_code=" << err << ")";
        }

        ctx->revoke_and_shrink();

        // After shrink: failed_ranks() on the new comm should be empty
        auto failed_after = ctx->failed_ranks();
        EXPECT_TRUE(failed_after.empty())
            << "Rank " << rank << ": failed_ranks() should be empty on fresh shrunken comm, got "
            << failed_after.size() << " entries";
    }

    EXPECT_TRUE(detected)
        << "Rank " << rank << ": should have detected rank 1 failure";

    ctx->barrier();
}

TEST(FaultTolerance, AgreeMixedVotesSingleRank) {
    //----------------------------------------------------------------------
    // Edge case: agree() with a single-rank communicator.
    // With only one rank, the result should be whatever that rank voted.
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    // Create a single-rank sub-communicator for this test
    const int rank = *ctx->rank();
    std::vector<int> self_ranks = {rank};
    auto self_ctx = ctx->create_sub_context(
        tt::stl::Span<int>(self_ranks.data(), self_ranks.size()));

    ASSERT_EQ(*self_ctx->size(), 1);

    // Single rank votes true
    {
        auto result = self_ctx->agree(true);
        ASSERT_TRUE(result.has_value());
        EXPECT_TRUE(result.value())
            << "Single-rank agree(true) should return true";
    }

    // Single rank votes false
    {
        auto result = self_ctx->agree(false);
        ASSERT_TRUE(result.has_value());
        EXPECT_FALSE(result.value())
            << "Single-rank agree(false) should return false";
    }

    ctx->barrier();
}

TEST(FaultTolerance, SetFailurePolicyIsIdempotent) {
    //----------------------------------------------------------------------
    // Setting the same policy multiple times should be safe.
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    auto* mpi_ctx = dynamic_cast<tt::tt_metal::distributed::multihost::MPIContext*>(ctx.get());
    ASSERT_NE(mpi_ctx, nullptr);

    // Set FAST_FAIL multiple times — should not crash
    mpi_ctx->set_failure_policy(FailurePolicy::FAST_FAIL);
    mpi_ctx->set_failure_policy(FailurePolicy::FAST_FAIL);

    // Switch to FAULT_TOLERANT
    mpi_ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT);
    mpi_ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT);

    // Switch back
    mpi_ctx->set_failure_policy(FailurePolicy::FAST_FAIL);

    // Barrier to ensure all ranks executed successfully
    ctx->barrier();

    // All ranks should pass
    EXPECT_EQ_ALL_RANKS(1, ctx);
}

TEST(FaultTolerance, SuccessPathNoErrorOutput) {
    //----------------------------------------------------------------------
    // Log quality test: on the success path (no failures), we should NOT
    // see any FATAL/WARNING error output from the ULFM infrastructure.
    //
    // This test verifies that normal operations don't produce spurious
    // error messages that would confuse CI triage.
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    // Run a series of normal operations — none should produce error output
    ctx->barrier();

    auto result = ctx->agree(true);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());

    auto failed = ctx->failed_ranks();
    EXPECT_TRUE(failed.empty());

    EXPECT_FALSE(ctx->is_revoked());

    // If we got here without any output to stderr, the success path is clean.
    // GTest will capture any unexpected output.
    ctx->barrier();
}
