// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression fence for host-interleaved (persistent) dispatch — the RunOutcome::HostWait path that
// lets a run quiesce back to the host mid-run so it can stream socket bytes, then resume via pump().
// See tt-emule docs/fiber-engine.md §6 and docs/socket-emulation.md §7.
//
// No canonical tt-metal suite drives a host-fed socket, so nothing upstream reaches
// run_persistent()/pump(); end-to-end coverage is tt-blaze's test_temporal_llama_decoders, which
// needs a mesh and a model. What a host-only gtest can pin is the decision logic, where a silent
// revert turns a device deadlock into a Finish() that succeeds over buffers no kernel wrote.
//
// Negative cases are death tests: a run whose only waiter is a poller the engine declines to call
// host-waiting never quiesces, so it ends at the tier-2 watchdog's abort. Death tests also keep a
// failure from leaving the process-global registry dirty for later tests.

#include <gtest/gtest.h>

#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "impl/emulation/emule_fiber_scheduler.hpp"
#include "jit_hw/internal/emule_thread_ctx.h"

namespace {

using tt::tt_metal::emule_fiber::FiberEngineStall;
using tt::tt_metal::emule_fiber::FiberIdentity;
using tt::tt_metal::emule_fiber::FiberScheduler;
using tt::tt_metal::emule_fiber::RunOutcome;

// A distinct identity per fiber so a hang dump names which one wedged.
FiberIdentity ident(uint8_t x, uint8_t y, const char* src) {
    FiberIdentity id;
    id.phys_x = x;
    id.phys_y = y;
    id.logical_x = x;
    id.logical_y = y;
    id.proc_id = 0;
    id.kernel_src = src;  // static string literal — outlives the run
    return id;
}

// The body only calls bridge ops, so a default-constructed ctx is enough: no L1, no DFBs, no device.
void spawn_fiber(std::function<void()> body, uint8_t x, const char* src) {
    FiberScheduler::instance().spawn(std::move(body), std::make_unique<DatamovementThreadCtx>(), ident(x, 0, src));
}

// The shape this feature exists for: the poll variant of socket_wait_for_pages. It cannot park (it
// must return so the kernel can re-check termination), so it tags, yields, and comes back.
std::function<void()> polling_body(const std::atomic<bool>* done, bool host_fed, std::atomic<unsigned>* entries) {
    return [done, host_fed, entries] {
        if (entries != nullptr) {
            entries->fetch_add(1);  // a double-queued fiber would run its body twice
        }
        auto& sched = FiberScheduler::instance();
        while (!done->load(std::memory_order_acquire)) {
            sched.note_socket_poll_wait(/*waiting=*/true, host_fed);
            sched.yield();
        }
        sched.note_socket_poll_wait(/*waiting=*/false, host_fed);  // bytes landed
    };
}

// Fixture: threadsafe death tests (the engine owns a 64-thread pool, so forking a live
// process is not safe), and a guarantee that each test leaves the process-global registry
// empty for the next one.
class EmuleHostWait : public ::testing::Test {
protected:
    void SetUp() override { ::testing::FLAGS_gtest_death_test_style = "threadsafe"; }

    void TearDown() override {
        // A test that leaks a fiber poisons the process-global registry (there is no public reset),
        // so every later test fails here too — when several fail at once, fix the FIRST.
        ASSERT_EQ(FiberScheduler::instance().oldest_live_spawn_generation(), UINT64_MAX)
            << "test left a live fiber in the process-global scheduler registry; if this is "
               "not the first failure in EmuleHostWait.*, fix the first one first";
    }

    // Trip the tier-2 watchdog fast. It re-reads both knobs per run, and the threadsafe death-test
    // child inherits them across the re-exec.
    static void arm_fast_watchdog() {
        ::setenv("TT_EMULE_FIBER_PROGRESS_WINDOW", "2000", 1);
        ::setenv("TT_EMULE_FIBER_WATCHDOG_SEC", "5", 1);
        ::setenv("TT_EMULE_HOST_WAIT_WATCHDOG_SEC", "5", 1);  // parked time has its own, far larger, bound
    }
    static void disarm_fast_watchdog() {
        ::unsetenv("TT_EMULE_FIBER_PROGRESS_WINDOW");
        ::unsetenv("TT_EMULE_FIBER_WATCHDOG_SEC");
        ::unsetenv("TT_EMULE_HOST_WAIT_WATCHDOG_SEC");
    }
};

// The core contract: a host-fed poll makes a stalled run resumable, and pump() resumes the SAME
// fibers rather than restarting them.
TEST_F(EmuleHostWait, HostFedPollQuiescesToHostWaitAndPumpsToCompletion) {
    std::atomic<bool> done{false};
    std::atomic<unsigned> entries{0};
    spawn_fiber(polling_body(&done, /*host_fed=*/true, &entries), 1, "h2d_receiver_poll");

    auto& sched = FiberScheduler::instance();

    // The run cannot advance without the host, and says so instead of dying.
    ASSERT_EQ(sched.run_persistent(), RunOutcome::HostWait);
    // No teardown happened: the fiber is still live, which is the entire point.
    EXPECT_NE(sched.oldest_live_spawn_generation(), UINT64_MAX);
    EXPECT_EQ(entries.load(), 1u);

    // A pump with the socket still dry returns HostWait again, and must not re-enter the
    // kernel body from the top.
    ASSERT_EQ(sched.pump(), RunOutcome::HostWait);
    EXPECT_EQ(entries.load(), 1u) << "pump re-ran a kernel body instead of resuming it";

    // The host streamed. Now the run drains.
    done.store(true, std::memory_order_release);
    ASSERT_EQ(sched.pump(), RunOutcome::Completed);
    EXPECT_EQ(entries.load(), 1u);
}

// A d2d receiver's sender is a PEER, so its wait must not read as a host wait — that would retire
// the dump naming the real culprit. Dying on this regex proves both the outcome and the attribution.
TEST_F(EmuleHostWait, PeerFedPollIsNamedAsPeerFedInTheDump) {
    arm_fast_watchdog();
    EXPECT_DEATH(
        {
            std::atomic<bool> never{false};
            spawn_fiber(polling_body(&never, /*host_fed=*/false, nullptr), 3, "d2d_receiver_poll");
            (void)FiberScheduler::instance().run_persistent();
        },
        "spin-polling a d2d socket");
    disarm_fast_watchdog();
}

// The tag is sticky (clearing it on the early-exit return would let the engine sample zero
// mid-poll), so a kernel that LEAVES the poll loop must age out or it pins the run host-waiting.
TEST_F(EmuleHostWait, StalePollTagAgesOut) {
    arm_fast_watchdog();
    EXPECT_DEATH(
        {
            // Tag once, then leave the loop and spin elsewhere. Freshness is measured against this
            // fiber's OWN resume count, so its own yields age the tag out.
            spawn_fiber(
                [] {
                    auto& sched = FiberScheduler::instance();
                    sched.note_socket_poll_wait(/*waiting=*/true, /*host_fed=*/true);
                    for (;;) {
                        sched.yield();  // never re-tags
                    }
                },
                4,
                "poller_that_moved_on");
            (void)FiberScheduler::instance().run_persistent();
        },
        "no global progress");
    disarm_fast_watchdog();
}

// Parking must retire the tag: a parked fiber never resumes, so its freshness delta would freeze
// and the tag could never go stale on its own. A clean throw, since parking reaches quiescence.
TEST_F(EmuleHostWait, ParkingRetiresThePollTag) {
    static uint32_t dead_key = 0;  // a key nobody will ever wake
    std::atomic<bool> never{false};

    spawn_fiber(
        [&never] {
            auto& sched = FiberScheduler::instance();
            // Freshly tagged as host-waiting…
            sched.note_socket_poll_wait(/*waiting=*/true, /*host_fed=*/true);
            sched.yield();
            // …then parks on a peer predicate. The loop is required: quiescence force-releases every
            // parked fiber before deciding, so a fiber that parks once would run off its body end.
            while (!never.load(std::memory_order_acquire)) {
                sched.lock();
                sched.park_locked(&dead_key);  // releases the lock; nobody wakes this
            }
        },
        5,
        "poller_then_parked");

    // If the poll tag had survived the park, its freshness delta would be frozen (a parked
    // fiber is never resumed) and this run would report a resumable HostWait forever.
    EXPECT_THROW(FiberScheduler::instance().run_persistent(), FiberEngineStall);
}

// A park tagged as a host-fed socket wait is the OTHER half of any_waiting_on_host(), and
// must still resolve to HostWait — this is the path a blocking socket_wait_for_pages takes.
TEST_F(EmuleHostWait, ParkedHostFedSocketWaitIsAHostWait) {
    static uint32_t credit_word = 0;
    std::atomic<bool> bytes_arrived{false};

    spawn_fiber(
        [&bytes_arrived] {
            auto& sched = FiberScheduler::instance();
            // Re-check on resume, as a real blocking socket_wait_for_pages does: a spurious
            // force-release must re-park, not fall through as if the bytes had landed.
            while (!bytes_arrived.load(std::memory_order_acquire)) {
                sched.lock();
                sched.park_locked_socket(&credit_word);  // the HOST owes this one
            }
        },
        6,
        "h2d_receiver_blocking");

    auto& sched = FiberScheduler::instance();
    ASSERT_EQ(sched.run_persistent(), RunOutcome::HostWait);

    // The host advances the credit word and wakes the receiver; the run drains.
    bytes_arrived.store(true, std::memory_order_release);
    sched.wake(&credit_word);
    ASSERT_EQ(sched.pump(), RunOutcome::Completed);
}

// Nothing guarantees the host ever pumps again after a HostWait return, so the watchdog must stay
// up across the gap or the job ends as a silent success over output no kernel wrote.
TEST_F(EmuleHostWait, UnpumpedHostWaitStillTripsTheWatchdog) {
    arm_fast_watchdog();
    EXPECT_DEATH(
        {
            static std::atomic<bool> never{false};
            spawn_fiber(polling_body(&never, /*host_fed=*/true, nullptr), 15, "h2d_receiver_poll");
            auto& sched = FiberScheduler::instance();
            // A legitimate HostWait — this is NOT the failure. The failure is what follows.
            if (sched.run_persistent() != RunOutcome::HostWait) {
                std::exit(2);  // distinguishable from the abort this test expects
            }
            // The host now walks away: no pump, no Finish, no read. Nothing in the engine is
            // running. Only a watchdog that outlived the return can still call this a hang.
            //
            // Bounded well past the 5s backstop: unbounded would turn a regression into a CI
            // timeout rather than a failure. Exit 3 reads distinctly against the expected abort.
            for (int i = 0; i < 150; ++i) {
                ::usleep(200 * 1000);  // 30s total
            }
            std::exit(3);
        },
        "no global progress");
    disarm_fast_watchdog();
}

// The mirror: the gap watchdog must not outlive the run it was guarding. A pump that
// completes tears the run down, and a watchdog still ticking after that would abort the
// NEXT program the moment it spent longer than the backstop between progress points.
TEST_F(EmuleHostWait, CompletedRunLeavesNoWatchdogBehind) {
    arm_fast_watchdog();
    auto& sched = FiberScheduler::instance();

    std::atomic<bool> done{false};
    spawn_fiber(polling_body(&done, /*host_fed=*/true, nullptr), 16, "h2d_receiver_poll");
    ASSERT_EQ(sched.run_persistent(), RunOutcome::HostWait);
    done.store(true, std::memory_order_release);
    ASSERT_EQ(sched.pump(), RunOutcome::Completed);

    // Idle for longer than the armed backstop (5s). A leaked watchdog aborts the process here.
    ::usleep(6500 * 1000);
    disarm_fast_watchdog();

    // And the engine is still usable: a fresh run works normally.
    std::atomic<unsigned> ran{0};
    spawn_fiber([&ran] { ran.fetch_add(1); }, 17, "post_gap_program");
    ASSERT_EQ(sched.run_persistent(), RunOutcome::Completed);
    EXPECT_EQ(ran.load(), 1u);
}

// Outcome flags are per-launch: a torn-down run must not leave host_wait_ for the next launch to
// read, or the runner frees the CURRENT dispatch's keepalives under fibers still using them.
TEST_F(EmuleHostWait, EmptyLaunchAfterAStalledRunIsNotAHostWait) {
    auto& sched = FiberScheduler::instance();

    // Wedge a run and let the engine tear it down. Park on a key nobody wakes, with no host-facing
    // wait — the tier-1 deadlock, same teardown path, without depending on the pump bound.
    static uint32_t dead_key = 0;
    std::atomic<bool> never{false};
    spawn_fiber(
        [&never] {
            auto& s = FiberScheduler::instance();
            while (!never.load(std::memory_order_acquire)) {
                s.lock();
                s.park_locked(&dead_key);
            }
        },
        18,
        "wedged_kernel");
    EXPECT_THROW(sched.run_persistent(), FiberEngineStall);
    ASSERT_EQ(sched.oldest_live_spawn_generation(), UINT64_MAX);

    // Now a dispatch that registers nothing — the empty-registry early return in the launch
    // path. It must report the end of a run, not a resumable one.
    EXPECT_EQ(sched.run_persistent(), RunOutcome::Completed);
    // And a pump against no registry stays a no-op rather than acting on a phantom run.
    EXPECT_EQ(sched.pump(), RunOutcome::Completed);
}

// One CB tag slot per fiber, kept by the CB actually starving it. Probing a wedged CB and an empty
// one each iteration must not record whichever was last, or the dump points at a healthy producer.
TEST_F(EmuleHostWait, CbPollTagNamesTheStarvingCbNotTheLastProbed) {
    arm_fast_watchdog();
    EXPECT_DEATH(
        {
            spawn_fiber(
                [] {
                    auto& sched = FiberScheduler::instance();
                    for (;;) {
                        sched.note_cb_poll_wait(/*cb_id=*/3, /*n=*/2);  // wedged producer
                        sched.note_cb_poll_wait(/*cb_id=*/9, /*n=*/1);  // healthy, momentarily empty
                        sched.yield();
                    }
                },
                19,
                "compute_two_cb_probe");
            (void)FiberScheduler::instance().run_persistent();
        },
        "spin-polling CB 3 for 2 page");
    disarm_fast_watchdog();
}

// Spawn generations gate the runner's keepalive reclaim. If either query says "dead" while a fiber
// is live, the runner frees DFB arrays and ASAN snapshots under a running kernel.
TEST_F(EmuleHostWait, OldestLiveGenerationTracksLiveFibers) {
    auto& sched = FiberScheduler::instance();

    // Nothing registered.
    ASSERT_EQ(sched.oldest_live_spawn_generation(), UINT64_MAX);

    sched.begin_spawn_generation();
    std::atomic<bool> done{false};
    spawn_fiber(polling_body(&done, /*host_fed=*/true, nullptr), 8, "gen_a");

    // Registered but not yet run: still live (not Done), so it must be reported.
    EXPECT_NE(sched.oldest_live_spawn_generation(), UINT64_MAX);

    ASSERT_EQ(sched.run_persistent(), RunOutcome::HostWait);
    const uint64_t parked_gen = sched.oldest_live_spawn_generation();
    EXPECT_NE(parked_gen, UINT64_MAX) << "a parked fiber must keep its generation alive";

    // A later dispatch opens a newer generation. The parked run's generation is older, so it
    // is what the runner must not free past.
    sched.begin_spawn_generation();
    EXPECT_EQ(sched.oldest_live_spawn_generation(), parked_gen);

    done.store(true, std::memory_order_release);
    ASSERT_EQ(sched.pump(), RunOutcome::Completed);
    EXPECT_EQ(sched.oldest_live_spawn_generation(), UINT64_MAX);
}

// The per-generation query. An age bound cannot express this: a host-interleaved run parks ONE
// generation for the whole sequence, so "older than the oldest live" frees nothing after it.
TEST_F(EmuleHostWait, GenerationLivenessIsPerGenerationNotAnAgeBound) {
    auto& sched = FiberScheduler::instance();

    // Generation A parks for the duration — the socket relay.
    std::atomic<bool> done{false};
    const uint64_t gen_a = sched.begin_spawn_generation();
    spawn_fiber(polling_body(&done, /*host_fed=*/true, nullptr), 20, "h2d_receiver_poll");
    ASSERT_EQ(sched.run_persistent(), RunOutcome::HostWait);
    EXPECT_TRUE(sched.spawn_generation_is_live(gen_a));

    // Generation B is a compute stage dispatched on top of it; it runs to completion.
    const uint64_t gen_b = sched.begin_spawn_generation();
    spawn_fiber([] {}, 21, "compute_stage");
    ASSERT_EQ(sched.run_persistent(), RunOutcome::HostWait);

    // B is finished, so B's keepalives are reclaimable NOW — even though A, which is older, is
    // still parked and still pins oldest_live_spawn_generation() to itself.
    EXPECT_FALSE(sched.spawn_generation_is_live(gen_b))
        << "a finished generation must be reclaimable while an OLDER one is still parked";
    EXPECT_TRUE(sched.spawn_generation_is_live(gen_a));
    EXPECT_EQ(sched.oldest_live_spawn_generation(), gen_a);

    done.store(true, std::memory_order_release);
    ASSERT_EQ(sched.pump(), RunOutcome::Completed);
    EXPECT_FALSE(sched.spawn_generation_is_live(gen_a));
}

// Two co-resident programs is the NORMAL host-interleaved case. launch_and_wait must integrate only
// the new fibers: never re-queue a parked one, drop a Ready one, re-home a live one, or reset active_.
TEST_F(EmuleHostWait, ResumingLaunchIntegratesNewFibersWithoutDisturbingLiveOnes) {
    auto& sched = FiberScheduler::instance();

    std::atomic<bool> done{false};
    std::atomic<unsigned> receiver_entries{0};
    sched.begin_spawn_generation();
    spawn_fiber(polling_body(&done, /*host_fed=*/true, &receiver_entries), 9, "h2d_receiver_poll");

    ASSERT_EQ(sched.run_persistent(), RunOutcome::HostWait);
    ASSERT_EQ(receiver_entries.load(), 1u);
    const uint64_t receiver_gen = sched.oldest_live_spawn_generation();
    ASSERT_NE(receiver_gen, UINT64_MAX);

    // Dispatch a second program on top of the parked one. It runs to completion while the
    // receiver stays parked, so the run quiesces to HostWait again.
    std::atomic<unsigned> compute_entries{0};
    sched.begin_spawn_generation();
    spawn_fiber([&compute_entries] { compute_entries.fetch_add(1); }, 10, "compute_stage");

    ASSERT_EQ(sched.run_persistent(), RunOutcome::HostWait);
    EXPECT_EQ(compute_entries.load(), 1u) << "the newly registered fiber never ran";
    EXPECT_EQ(receiver_entries.load(), 1u) << "the live fiber's body was re-entered";
    // The receiver is still the oldest thing alive — the second program has finished.
    EXPECT_EQ(sched.oldest_live_spawn_generation(), receiver_gen);

    // A third dispatch: the retirement pass must reclaim the finished fiber's 1 MiB stack. Teardown
    // is the only other site that frees stacks, and a HostWait return never reaches it.
    std::atomic<unsigned> compute2_entries{0};
    sched.begin_spawn_generation();
    spawn_fiber([&compute2_entries] { compute2_entries.fetch_add(1); }, 11, "compute_stage_2");

    ASSERT_EQ(sched.run_persistent(), RunOutcome::HostWait);
    EXPECT_EQ(compute2_entries.load(), 1u);
    EXPECT_EQ(receiver_entries.load(), 1u);

    done.store(true, std::memory_order_release);
    ASSERT_EQ(sched.pump(), RunOutcome::Completed);
    EXPECT_EQ(sched.oldest_live_spawn_generation(), UINT64_MAX);
}

// drain_device() must tolerate "will not drain" without swallowing kernel faults, so the stall needs
// its own catchable type and a kernel fault must not be one.
TEST_F(EmuleHostWait, EngineStallIsADistinctType) {
    // A runtime_error, so existing broad handlers still see it…
    EXPECT_TRUE((std::is_base_of<std::runtime_error, FiberEngineStall>::value));
    // …but distinguishable, which is what lets a caller catch it and nothing broader.
    try {
        throw FiberEngineStall("stall");
    } catch (const FiberEngineStall& e) {
        EXPECT_STREQ(e.what(), "stall");
    } catch (...) {
        FAIL() << "FiberEngineStall was not caught by its own type";
    }
}

TEST_F(EmuleHostWait, KernelFaultIsNotAnEngineStall) {
    spawn_fiber([] { throw std::out_of_range("DFB range"); }, 12, "faulting_kernel");

    // The kernel's own exception propagates verbatim. If this were widened to
    // FiberEngineStall, drain_device()'s narrow catch would swallow real faults.
    EXPECT_THROW(FiberScheduler::instance().run_persistent(), std::out_of_range);
}

// A captured kernel fault outranks a host wait: a "resumable" run would defer the throw to whichever
// program tears the registry down next, which is where it would then be blamed.
TEST_F(EmuleHostWait, KernelFaultOutranksHostWait) {
    std::atomic<bool> never{false};
    spawn_fiber(polling_body(&never, /*host_fed=*/true, nullptr), 13, "h2d_receiver_poll");
    spawn_fiber([] { throw std::out_of_range("DFB range"); }, 14, "faulting_kernel");

    // Even though a host-fed poller is live and would otherwise force HostWait.
    EXPECT_THROW(FiberScheduler::instance().run_persistent(), std::out_of_range);
}

// The drain backstop sizes itself above this by reading it here rather than re-parsing the env var,
// which env_size parses by different rules — an independent parse would let the two disagree.
TEST_F(EmuleHostWait, StallLimitIsPositiveAndStable) {
    const uint64_t a = FiberScheduler::host_wait_stall_limit();
    EXPECT_GT(a, 0u) << "a zero bound would make pump() escalate on its first no-progress pump";
    // Read once at first use, so it cannot drift — what lets the runner cache its floor as a static.
    EXPECT_EQ(a, FiberScheduler::host_wait_stall_limit());
}

}  // namespace
