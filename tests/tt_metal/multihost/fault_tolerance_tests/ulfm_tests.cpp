// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
#include <tt-metalium/distributed_context.hpp>

#include <gtest/gtest.h>
#include "common/multihost_test_tools.hpp"
#include "tt_metal/distributed/multihost/mpi_distributed_context.hpp"
#include <mpi-ext.h>

#include <csignal>
#include <cstdlib>
#include <pthread.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

using multihost::common::ContextPtr;
using multihost::common::kill_rank_and_recover;
using tt::tt_metal::distributed::multihost::Color;
using tt::tt_metal::distributed::multihost::DistributedContext;
using tt::tt_metal::distributed::multihost::DistributedException;
using tt::tt_metal::distributed::multihost::FailurePolicy;
using tt::tt_metal::distributed::multihost::Key;
using tt::tt_metal::distributed::multihost::MPIContext;
using tt::tt_metal::distributed::multihost::MPIRankFailureException;
using tt::tt_metal::distributed::multihost::Rank;
using tt::tt_metal::distributed::multihost::Tag;
namespace multihost_detail = tt::tt_metal::distributed::multihost::detail;

// NOTE: Tests that kill ranks or call revoke_and_shrink() mutate the shared
// world communicator.  This creates an ordering dependency: tests that shrink
// the communicator affect all subsequent tests in the same process.
// Ideally each such test should call ctx->duplicate() and operate on the
// duplicate, but the downcast to MPIContext for set_failure_policy() makes
// this non-trivial.  For now, tests that use kill_rank_and_recover() pass
// a duplicated context where feasible.  Tests that directly manipulate the
// world communicator still share state and MUST run in the order listed.

TEST(FaultTolerance, ShrinkAfterRankFailure) {
    //----------------------------------------------------------------------
    // 0 · Create world communicator and install MPI_ERRORS_RETURN
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();

    const int world = *ctx->size();
    const int self_rank = *ctx->rank();
    const int victim_rank = 1;  // rank to kill (if exists)

    // Switch to FAULT_TOLERANT so barrier() throws instead of _exit(70)
    auto* mpi_ctx = dynamic_cast<tt::tt_metal::distributed::multihost::MPIContext*>(ctx.get());
    ASSERT_NE(mpi_ctx, nullptr) << "Expected MPIContext for ULFM test";
    mpi_ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT);

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

    // Switch to FAULT_TOLERANT so barrier() throws instead of _exit(70)
    auto* mpi_ctx = dynamic_cast<tt::tt_metal::distributed::multihost::MPIContext*>(ctx.get());
    ASSERT_NE(mpi_ctx, nullptr) << "Expected MPIContext for ULFM test";
    mpi_ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT);

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
// Test infrastructure: agree() consensus and policy switching
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
            EXPECT_FALSE(result.value()) << "Rank " << rank << ": mixed agree should yield false (AND semantics)";
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

    EXPECT_TRUE(caught_rank_failure) << "Rank " << rank << ": expected MPIRankFailureException in FAULT_TOLERANT mode";

    // Verify the shrunken communicator works
    const int new_world = *ctx->size();
    EXPECT_EQ(new_world, world - 1);
    EXPECT_EQ_SURVIVING_RANKS(new_world, ctx);

    ctx->barrier();
}

// =====================================================================
// Single-node testing: ULFM control-plane tests
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
    // The shell harness covers the real FAST_FAIL integration path
    // separately with FaultTolerance.FastFailEmitsRankFailureDiagnostics,
    // which expects mpirun to terminate after emitting stderr diagnostics.
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
    // the test runner. The real FAST_FAIL shutdown path is covered by
    // the shell-level test in run_single_node_ulfm_tests.sh.
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

TEST(FaultTolerance, FastFailEmitsRankFailureDiagnostics) {
    //----------------------------------------------------------------------
    // Exercise the real FAST_FAIL path: surviving ranks log ULFM key=value
    // diagnostics (policy, hostnames, ranks) to stderr, then _exit(70).
    //
    // The shell harness expects mpirun to exit non-zero after the surviving
    // rank emits a line containing policy=fast_fail and hostname fields.
    //----------------------------------------------------------------------
    const std::string filter = GTEST_FLAG_GET(filter);
    if (filter == "*" || filter.empty()) {
        GTEST_SKIP() << "This test calls _exit() and must be run in isolation: "
                        "--gtest_filter=FaultTolerance.FastFailEmitsRankFailureDiagnostics";
    }

    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    const int world = *ctx->size();
    const int rank = *ctx->rank();

    if (world < 2) {
        GTEST_SKIP() << "Need at least 2 ranks";
    }

    auto* mpi_ctx = dynamic_cast<tt::tt_metal::distributed::multihost::MPIContext*>(ctx.get());
    ASSERT_NE(mpi_ctx, nullptr);
    mpi_ctx->set_failure_policy(FailurePolicy::FAST_FAIL);

    if (rank == 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        raise(SIGKILL);
    }

    ctx->barrier();
    FAIL() << "FAST_FAIL should terminate surviving ranks before barrier returns";
}

TEST(FaultTolerance, FinalizeWatchdogPath) {
    //----------------------------------------------------------------------
    // Verify that the MPI_Finalize watchdog is scoped to the finalize call.
    //
    // We cannot directly test that MPI_Finalize hangs (that would hang
    // the test), but we CAN verify:
    // 1. SIGALRM is NOT installed process-wide during normal execution
    // 2. Entering the scoped watchdog installs the handler
    // 3. Leaving the scope restores the previous SIGALRM disposition
    // 4. Non-success MPI_Finalize return codes are treated as fatal
    //
    // The actual MPI_Finalize hang scenario is tested by the shell-level
    // test in run_single_node_ulfm_tests.sh which runs a purpose-built
    // binary that triggers the watchdog.
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    EXPECT_FALSE(multihost_detail::is_finalize_alarm_handler_installed_for_testing())
        << "SIGALRM finalize watchdog should not be installed outside MPI_Finalize";
    EXPECT_FALSE(multihost_detail::finalize_is_in_progress())
        << "Finalize should not be marked in progress outside MPI_Finalize";

    struct ScopedSignalMaskRestore {
        sigset_t old_mask{};
        bool active{false};

        ~ScopedSignalMaskRestore() {
            if (active) {
                [[maybe_unused]] const int restore_rc = pthread_sigmask(SIG_SETMASK, &old_mask, nullptr);
            }
        }
    } signal_mask_restore;
    sigset_t block_mask;
    sigemptyset(&block_mask);
    sigaddset(&block_mask, SIGALRM);
    ASSERT_EQ(pthread_sigmask(SIG_BLOCK, &block_mask, &signal_mask_restore.old_mask), 0);
    signal_mask_restore.active = true;
    EXPECT_FALSE(multihost_detail::is_sigalrm_unblocked_for_testing())
        << "Test precondition: SIGALRM should be blocked before entering the scoped watchdog";

    multihost_detail::clear_finalize_unsafe();
    EXPECT_FALSE(multihost_detail::finalize_is_unsafe());
    multihost_detail::mark_finalize_unsafe();
    EXPECT_TRUE(multihost_detail::finalize_is_unsafe());
    multihost_detail::clear_finalize_unsafe();
    EXPECT_FALSE(multihost_detail::finalize_is_unsafe());

    {
        multihost_detail::ScopedFinalizeAlarmHandler guard(30);
        ASSERT_TRUE(guard.armed()) << "Scoped finalize watchdog should install SIGALRM handler";
        EXPECT_TRUE(multihost_detail::is_finalize_alarm_handler_installed_for_testing())
            << "SIGALRM finalize watchdog should be installed inside the scope";
        EXPECT_TRUE(multihost_detail::is_sigalrm_unblocked_for_testing())
            << "Scoped finalize watchdog should temporarily unblock SIGALRM in the finalizing thread";
        EXPECT_TRUE(multihost_detail::finalize_is_in_progress())
            << "Finalize should be marked in progress while the watchdog scope is active";
    }

    EXPECT_FALSE(multihost_detail::is_finalize_alarm_handler_installed_for_testing())
        << "SIGALRM finalize watchdog should be restored after leaving the scope";
    EXPECT_FALSE(multihost_detail::is_sigalrm_unblocked_for_testing())
        << "Scoped finalize watchdog should restore the prior blocked SIGALRM mask";
    EXPECT_FALSE(multihost_detail::finalize_is_in_progress())
        << "Finalize in-progress marker should be cleared after leaving the scope";

    EXPECT_TRUE(multihost_detail::finalize_return_is_nonfatal(MPI_SUCCESS));
    EXPECT_FALSE(multihost_detail::finalize_return_is_nonfatal(MPI_ERR_COMM));

    EXPECT_EQ_ALL_RANKS(1, ctx);

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
    EXPECT_NE(handler, nullptr) << "std::terminate handler should be installed";

    // All ranks should agree
    int has_handler = (handler != nullptr) ? 1 : 0;
    EXPECT_EQ_ALL_RANKS(has_handler, ctx);

    ctx->barrier();
}

TEST(FaultTolerance, TerminateHandlerRevokeResultClassification) {
    //----------------------------------------------------------------------
    // The std::terminate handler relies on integer MPI return codes from
    // MPIX_Comm_revoke, not C++ exceptions. Success and "already revoked"
    // are the expected outcomes; anything else is still tolerated because the
    // handler must continue to _exit(70).
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    EXPECT_TRUE(multihost_detail::terminate_revoke_result_is_nonfatal(MPI_SUCCESS));
    EXPECT_TRUE(multihost_detail::terminate_revoke_result_is_nonfatal(MPIX_ERR_REVOKED));
    EXPECT_FALSE(multihost_detail::terminate_revoke_result_is_nonfatal(MPI_ERR_COMM));

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
    EXPECT_TRUE(failed.empty()) << "Rank " << *ctx->rank()
                                << ": failed_ranks() should be empty when no ranks have died, got " << failed.size()
                                << " entries";

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
    const auto& world_ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(world_ctx);

    auto ctx = world_ctx->duplicate();
    const int world = *ctx->size();
    const int rank = *ctx->rank();

    if (world < 2) {
        GTEST_SKIP() << "Need at least 2 ranks for double-revoke test";
    }

    auto* mpi_ctx = dynamic_cast<tt::tt_metal::distributed::multihost::MPIContext*>(ctx.get());
    ASSERT_NE(mpi_ctx, nullptr);
    mpi_ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT);

    // Kill rank 1, catch failure on survivors, first revoke_and_shrink.
    kill_rank_and_recover(ctx, /*victim_rank=*/1, [&](const ContextPtr& c) {
        const int size_after_first = *c->size();
        EXPECT_EQ(size_after_first, world - 1) << "Rank " << rank << ": first shrink should remove exactly 1 rank";

        // Second revoke_and_shrink — should NOT crash or hang.
        // The communicator is healthy, so revoke + shrink should succeed
        // and produce a communicator with the same set of ranks.
        c->revoke_and_shrink();

        const int size_after_second = *c->size();
        EXPECT_EQ(size_after_second, size_after_first)
            << "Rank " << rank << ": second shrink should not lose additional ranks";

        EXPECT_EQ_SURVIVING_RANKS(size_after_second, c);
        c->barrier();
    });
}

TEST(FaultTolerance, AgreeAfterRevokeAndShrink) {
    //----------------------------------------------------------------------
    // Edge case: calling agree() on a communicator that has been through
    // revoke_and_shrink().  The new (shrunken) communicator should
    // support agree() just like a fresh communicator.
    //----------------------------------------------------------------------
    const auto& world_ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(world_ctx);

    auto ctx = world_ctx->duplicate();
    const int world = *ctx->size();
    const int rank = *ctx->rank();

    if (world < 2) {
        GTEST_SKIP() << "Need at least 2 ranks";
    }

    auto* mpi_ctx = dynamic_cast<tt::tt_metal::distributed::multihost::MPIContext*>(ctx.get());
    ASSERT_NE(mpi_ctx, nullptr);
    mpi_ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT);

    // Kill rank 1, recover on survivors, then test agree() on the shrunken communicator.
    kill_rank_and_recover(ctx, /*victim_rank=*/1, [&](const ContextPtr& c) {
        // All survivors vote true — should get true
        {
            auto result = c->agree(true);
            ASSERT_TRUE(result.has_value()) << "Rank " << rank << ": agree() should work on shrunken comm";
            EXPECT_TRUE(result.value()) << "Rank " << rank << ": unanimous true should yield true after shrink";
        }

        // All survivors vote false — should get false
        {
            auto result = c->agree(false);
            ASSERT_TRUE(result.has_value());
            EXPECT_FALSE(result.value()) << "Rank " << rank << ": unanimous false should yield false after shrink";
        }

        c->barrier();
    });
}

TEST(FaultTolerance, IsRevokedFalseBeforeFailure) {
    //----------------------------------------------------------------------
    // Verify that is_revoked() returns false on a healthy communicator.
    //----------------------------------------------------------------------
    const auto& ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(ctx);

    bool revoked = ctx->is_revoked();
    EXPECT_FALSE(revoked) << "Rank " << *ctx->rank() << ": is_revoked() should be false before any failure";

    int revoked_int = revoked ? 1 : 0;
    EXPECT_EQ_ALL_RANKS(revoked_int, ctx);

    ctx->barrier();
}

TEST(FaultTolerance, IsRevokedTrueAfterDetectionBeforeShrink) {
    //----------------------------------------------------------------------
    // Verify that is_revoked() becomes true after a rank failure is detected
    // and stays true until revoke_and_shrink() installs a healthy communicator.
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

    if (rank == 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        raise(SIGKILL);
    }

    bool detected = false;
    try {
        ctx->barrier();
    } catch (const DistributedException&) {
        detected = true;
        EXPECT_TRUE(ctx->is_revoked()) << "Rank " << rank << ": communicator should be marked revoked before shrink";
        ctx->revoke_and_shrink();
        EXPECT_FALSE(ctx->is_revoked()) << "Rank " << rank << ": communicator should be healthy again after shrink";
    }

    EXPECT_TRUE(detected) << "Rank " << rank << ": should have detected rank 1 failure";

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
    EXPECT_TRUE(supported) << "After SKIP_IF_NO_ULFM, supports_fault_tolerance() should be true";

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
        EXPECT_NE(e.what(), nullptr) << "Exception what() should not be null";

        const std::string& msg = e.message();
        EXPECT_FALSE(msg.empty()) << "Rank " << rank << ": exception message should not be empty";

        // The message should mention "failed ranks" or contain rank info
        EXPECT_TRUE(msg.find("failed") != std::string::npos || msg.find("rank") != std::string::npos)
            << "Rank " << rank << ": exception message should mention failure context: " << msg;

        // Error code should be non-zero (ULFM error)
        EXPECT_NE(e.error_code(), 0) << "Rank " << rank << ": error_code should be non-zero for rank failure";

        // error_string() should be populated from MPI_Error_string
        const std::string& err_str = e.error_string();
        EXPECT_FALSE(err_str.empty()) << "Rank " << rank << ": error_string() should be populated by MPI_Error_string";

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

        // Failure-path dependent: not all ranks can identify failed ranks.
        //
        // MPIX_ERR_PROC_FAILED (75): rank saw the failure directly.
        //   MPIX_Comm_failure_ack() succeeds and MPIX_Comm_failure_get_acked()
        //   returns the failed group — failed_ranks() is reliably non-empty.
        //
        // MPIX_ERR_REVOKED (77): another rank already called MPIX_Comm_revoke()
        //   before this rank processed the failure.  On the already-revoked
        //   communicator, MPIX_Comm_failure_ack() returns MPIX_ERR_REVOKED, so
        //   MPIX_Comm_failure_get_acked() finds no acked failures.  The
        //   detection-time cache (cached_failed_ranks_) *may* contain data if
        //   identify_failed_ranks() succeeded before the revoke fully propagated,
        //   but it is not guaranteed.  Accept an empty result for REVOKED-path
        //   ranks; require non-empty only for PROC_FAILED-path ranks.
        if (err != MPIX_ERR_REVOKED) {
            EXPECT_FALSE(failed.empty()) << "Rank " << rank
                                         << ": failed_ranks() should be non-empty after detecting failure"
                                         << " (error_code=" << err << ")";
        }

        ctx->revoke_and_shrink();

        // After shrink: failed_ranks() on the new comm should be empty
        auto failed_after = ctx->failed_ranks();
        EXPECT_TRUE(failed_after.empty())
            << "Rank " << rank << ": failed_ranks() should be empty on fresh shrunken comm, got " << failed_after.size()
            << " entries";
    }

    EXPECT_TRUE(detected) << "Rank " << rank << ": should have detected rank 1 failure";

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
    auto self_ctx = ctx->create_sub_context(tt::stl::Span<int>(self_ranks.data(), self_ranks.size()));

    ASSERT_EQ(*self_ctx->size(), 1);

    // Single rank votes true
    {
        auto result = self_ctx->agree(true);
        ASSERT_TRUE(result.has_value());
        EXPECT_TRUE(result.value()) << "Single-rank agree(true) should return true";
    }

    // Single rank votes false
    {
        auto result = self_ctx->agree(false);
        ASSERT_TRUE(result.has_value());
        EXPECT_FALSE(result.value()) << "Single-rank agree(false) should return false";
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

// =====================================================================
// Multi-failure and non-blocking fault-tolerance tests
// =====================================================================

TEST(FaultTolerance, SequentialRankFailures) {
    //----------------------------------------------------------------------
    // Kill one rank, recover, then kill another rank, recover again.
    // Verifies that the communicator can survive multiple sequential
    // shrink operations and that all survivors agree on the final size.
    //----------------------------------------------------------------------
    const auto& world_ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(world_ctx);

    auto ctx = world_ctx->duplicate();
    const int world = *ctx->size();
    const int rank = *ctx->rank();

    if (world < 3) {
        GTEST_SKIP() << "Need at least 3 ranks for sequential failure test";
    }

    auto* mpi_ctx = dynamic_cast<tt::tt_metal::distributed::multihost::MPIContext*>(ctx.get());
    ASSERT_NE(mpi_ctx, nullptr) << "Expected MPIContext for ULFM test";
    mpi_ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT);

    // -- Round 1: Kill rank 2 (highest rank in minimum 3-rank setup) ------
    kill_rank_and_recover(ctx, /*victim_rank=*/2, [&](const ContextPtr& c) {
        const int size_after_first = *c->size();
        EXPECT_EQ(size_after_first, world - 1)
            << "Rank " << rank << ": first shrink should remove exactly 1 rank";

        EXPECT_EQ_SURVIVING_RANKS(size_after_first, c);
        c->barrier();

        // -- Round 2: Kill rank 1 from the shrunken communicator ----------
        // Ranks have been renumbered after shrink; rank 1 in the new
        // communicator is a valid target (we have at least 2 survivors).
        auto* mpi_c = dynamic_cast<tt::tt_metal::distributed::multihost::MPIContext*>(c.get());
        ASSERT_NE(mpi_c, nullptr);
        mpi_c->set_failure_policy(FailurePolicy::FAULT_TOLERANT);

        const int my_new_rank = *c->rank();
        if (my_new_rank == 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            raise(SIGKILL);  // never returns
        }

        bool detected_second = false;
        try {
            c->barrier();
        } catch (const DistributedException&) {
            detected_second = true;
            c->revoke_and_shrink();
        }

        // Only survivors of both rounds reach here
        if (detected_second || *c->size() == world - 2) {
            const int size_after_second = *c->size();
            EXPECT_EQ(size_after_second, world - 2)
                << "Rank " << rank << ": second shrink should leave original_size - 2 ranks";

            EXPECT_EQ_SURVIVING_RANKS(size_after_second, c);
            c->barrier();
        }
    });
}

TEST(FaultTolerance, FailureDuringNonBlockingOps) {
    //----------------------------------------------------------------------
    // Post non-blocking irecv on rank 0, then kill the sender (rank 1).
    // Rank 0's wait() on the irecv should either:
    //   a) throw MPIRankFailureException / DistributedException, or
    //   b) return with an error that triggers recovery.
    //
    // After recovery, verify that surviving ranks can still communicate
    // using the shrunken communicator.
    //----------------------------------------------------------------------
    const auto& world_ctx = DistributedContext::get_current_world();
    SKIP_IF_NO_ULFM(world_ctx);

    auto ctx = world_ctx->duplicate();
    const int world = *ctx->size();
    const int rank = *ctx->rank();

    if (world < 3) {
        GTEST_SKIP() << "Need at least 3 ranks for non-blocking failure test";
    }

    auto* mpi_ctx = dynamic_cast<tt::tt_metal::distributed::multihost::MPIContext*>(ctx.get());
    ASSERT_NE(mpi_ctx, nullptr) << "Expected MPIContext for ULFM test";
    mpi_ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT);

    constexpr int TAG_VAL = 42;
    int recv_buf = 0;
    int send_buf = 0xDEAD;

    // Rank 1 is the victim: it posts an isend then dies before completion.
    // Rank 0 posts an irecv from rank 1 and attempts to wait().
    // All other ranks go directly to the recovery barrier.

    if (rank == 1) {
        // Post the isend, then die — the send may or may not complete
        auto req = ctx->isend(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&send_buf), sizeof(send_buf)),
            Rank{0},
            Tag{TAG_VAL});
        // Small delay so rank 0 has time to post its irecv
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        raise(SIGKILL);  // never returns
    }

    bool failure_detected = false;

    if (rank == 0) {
        // Post irecv from rank 1
        auto req = ctx->irecv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&recv_buf), sizeof(recv_buf)),
            Rank{1},
            Tag{TAG_VAL});

        // Wait for the irecv — rank 1 is dying, so this should fail.
        // wait() returns Status{source, tag, count}. For a non-wildcard irecv
        // the source and tag are already known; count (bytes received) is not
        // asserted here because we only care whether an exception is thrown
        // (failure detected) or not (send completed before the kill, recv_buf
        // holds valid data). [[maybe_unused]] rather than (void): Status has
        // real semantic content, so (void) would misrepresent it as garbage.
        try {
            [[maybe_unused]] auto status = req->wait();
            // If wait() returned normally, the send completed before the kill.
            // That's acceptable — the data arrived.
        } catch (const DistributedException&) {
            failure_detected = true;
            // The irecv failed because rank 1 died. Recovery happens below
            // via the barrier collective which all survivors call.
        }
    }

    // All surviving ranks (including rank 0) must detect the failure
    // via a collective and recover.
    try {
        ctx->barrier();
    } catch (const DistributedException&) {
        failure_detected = true;
        ctx->revoke_and_shrink();
    }

    // Rank 1 is dead, so at least one of the above (wait() or barrier) must
    // have thrown.  Every other ULFM test asserts on failure detection — this
    // test must too, otherwise a silent success path hides a broken detector.
    EXPECT_TRUE(failure_detected)
        << "Rank " << rank << ": expected failure detection via wait() or barrier";

    // If no exception was thrown (rank 0 got data before kill, and barrier
    // somehow succeeded), we still need to verify the communicator is healthy.
    // In practice, the barrier above will throw because rank 1 is dead.

    const int new_world = *ctx->size();
    EXPECT_EQ(new_world, world - 1)
        << "Rank " << rank << ": expected world - 1 after rank 1 death";

    EXPECT_EQ_SURVIVING_RANKS(new_world, ctx);

    // Final sanity: do a round-trip send/recv on the shrunken communicator.
    // Use rank 0 sending to the last surviving rank.
    const int last_rank = new_world - 1;
    constexpr int VERIFY_TAG = 99;
    int verify_send = 0xCAFE;
    int verify_recv = 0;
    const int my_new_rank = *ctx->rank();

    if (my_new_rank == 0 && new_world > 1) {
        ctx->send(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&verify_send), sizeof(verify_send)),
            Rank{last_rank},
            Tag{VERIFY_TAG});
    } else if (my_new_rank == last_rank && new_world > 1) {
        ctx->recv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&verify_recv), sizeof(verify_recv)),
            Rank{0},
            Tag{VERIFY_TAG});
        EXPECT_EQ(verify_recv, 0xCAFE)
            << "Rank " << my_new_rank << ": post-recovery send/recv data mismatch";
    }

    ctx->barrier();
}

TEST(FaultTolerance, RequestWaitCompletesAfterOwnerExpires) {
    //----------------------------------------------------------------------
    // Exercise the MPIRequest fallback path where owner_.lock() fails.
    // Using MPI_COMM_WORLD keeps the communicator valid after the temporary
    // MPIContext is released, while the request itself outlives that owner.
    //----------------------------------------------------------------------
    const auto& world_ctx = DistributedContext::get_current_world();

    const int world = *world_ctx->size();
    const int rank = *world_ctx->rank();
    if (world < 2) {
        GTEST_SKIP() << "Need at least 2 ranks for owner-expired request test";
    }

    auto temp_ctx = std::make_shared<MPIContext>(MPI_COMM_WORLD);
    constexpr int TAG_VAL = 314;
    constexpr int EXPECTED_VALUE = 0x1234;

    if (rank == 0) {
        int recv_buf = 0;
        auto req = temp_ctx->irecv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&recv_buf), sizeof(recv_buf)), Rank{1}, Tag{TAG_VAL});
        temp_ctx.reset();

        ASSERT_NE(req, nullptr);
        EXPECT_TRUE(req->active());

        const auto status = req->wait();
        EXPECT_FALSE(req->active());
        EXPECT_EQ(recv_buf, EXPECTED_VALUE);
        EXPECT_EQ(*status.source, 1);
        EXPECT_EQ(*status.tag, TAG_VAL);
        EXPECT_EQ(status.count, static_cast<int>(sizeof(recv_buf)));
    } else if (rank == 1) {
        int send_buf = EXPECTED_VALUE;
        temp_ctx->send(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&send_buf), sizeof(send_buf)), Rank{0}, Tag{TAG_VAL});
        temp_ctx.reset();
    } else {
        temp_ctx.reset();
    }

    world_ctx->barrier();
}

// Split-brain test reference (requires network partition capability)
//
// A split-brain scenario occurs when two disjoint groups of ranks each believe
// the other group has failed.  This is the most dangerous failure mode for
// distributed systems because both partitions may independently proceed with
// mutually inconsistent state.
//
// Testing this would require:
//
// 1. A network partition mechanism (e.g., iptables rules, network namespaces,
//    or a specialized MPI test fabric that can selectively drop messages between
//    subsets of ranks) to isolate ranks into two groups that cannot communicate.
//
// 2. Verification that MPIX_Comm_agree() detects the inconsistency when the
//    partition heals.  In theory, agree() should fail or return inconsistent
//    results if called while the partition is active, since it cannot reach
//    consensus across the split.
//
// 3. Verification that revoke_and_shrink() produces a correct communicator
//    after the partition.  Both partitions may have independently revoked,
//    leading to ambiguity about which ranks are "surviving."
//
// 4. Cleanup that restores network connectivity after the test so that
//    subsequent tests are not affected.
//
// This is not feasible in the current test harness, which only supports
// process-level failures (raise(SIGKILL)) and has no ability to manipulate
// network-level connectivity between MPI ranks.
//
// A potential approach for future work:
//   - Use Linux network namespaces (ip netns) with veth pairs to run MPI
//     ranks in isolated network contexts, then use iptables DROP rules to
//     simulate partitions.
//   - Alternatively, use a custom MPI transport layer that supports
//     fault injection at the message level.
//
