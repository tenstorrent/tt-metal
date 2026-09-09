// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression fence for RunOutcome::HostWait: host-resolvable stalls vs device faults. Negatives are death tests.

#include <gtest/gtest.h>

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>

#include "impl/emulation/emule_fiber_scheduler.hpp"
#include "jit_hw/internal/emule_fiber_bridge.h"
#include "jit_hw/internal/emule_fiber_trace.h"
#include "jit_hw/internal/emule_thread_ctx.h"
#include "tt_emule/cb_sync_state.hpp"

namespace {

using tt::tt_metal::emule_fiber::FiberEngineStall;
using tt::tt_metal::emule_fiber::FiberIdentity;
using tt::tt_metal::emule_fiber::FiberScheduler;
using tt::tt_metal::emule_fiber::RunOutcome;

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::filesystem::path fresh_trace_dir(const char* name) {
    const char* configured_root = std::getenv("TT_EMULE_TEST_TRACE_ROOT");
    const auto root = configured_root ? std::filesystem::path(configured_root) : std::filesystem::temp_directory_path();
    auto path = root / ("emule-flight-recorder-" + std::string(name));
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    ::setenv("TT_EMULE_TRACE_DIR", path.c_str(), 1);
    ::setenv("TT_EMULE_TRACE_MB", "1", 1);
    return path;
}

void clear_trace_env() {
    ::unsetenv("TT_EMULE_TRACE_DIR");
    ::unsetenv("TT_EMULE_TRACE_MB");
}

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

// The poll variant of socket_wait_for_pages: it cannot park, so it tags, yields, and comes back.
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

// Threadsafe death tests (forking the engine's 64-thread pool is not safe) + a clean-registry check.
class EmuleHostWait : public ::testing::Test {
protected:
    void SetUp() override { ::testing::FLAGS_gtest_death_test_style = "threadsafe"; }

    void TearDown() override {
        // A leaked fiber poisons the global registry, so every later test fails too — fix the FIRST.
        ASSERT_EQ(FiberScheduler::instance().oldest_live_spawn_generation(), UINT64_MAX)
            << "test left a live fiber in the process-global scheduler registry; if this is "
               "not the first failure in EmuleHostWait.*, fix the first one first";
    }

    // Trip the tier-2 watchdog fast; re-read per run, and inherited by the death-test child.
    static void arm_fast_watchdog() {
        ::setenv("TT_EMULE_FIBER_PROGRESS_WINDOW", "2000", 1);
        ::setenv("TT_EMULE_FIBER_WATCHDOG_SEC", "3", 1);
        ::setenv("TT_EMULE_HOST_WAIT_WATCHDOG_SEC", "3", 1);  // parked time has its own, far larger, bound
    }
    static void disarm_fast_watchdog() {
        ::unsetenv("TT_EMULE_FIBER_PROGRESS_WINDOW");
        ::unsetenv("TT_EMULE_FIBER_WATCHDOG_SEC");
        ::unsetenv("TT_EMULE_HOST_WAIT_WATCHDOG_SEC");
    }
};

// The core contract: a host-fed poll is resumable, and pump() resumes the SAME fibers.
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

    // A pump with the socket still dry returns HostWait, without re-entering the body from the top.
    ASSERT_EQ(sched.pump(), RunOutcome::HostWait);
    EXPECT_EQ(entries.load(), 1u) << "pump re-ran a kernel body instead of resuming it";

    // The host streamed. Now the run drains.
    done.store(true, std::memory_order_release);
    ASSERT_EQ(sched.pump(), RunOutcome::Completed);
    EXPECT_EQ(entries.load(), 1u);
}

// A peer-fed socket can be causally downstream of the host-fed source. The external host root is
// sufficient to suspend the whole chain; requiring every poller to be host-fed would misdiagnose
// a normal pipeline as a device-only deadlock.
TEST_F(EmuleHostWait, HostFedPollWithCausalD2DPollResumes) {
    std::atomic<bool> host_ready{false};
    std::atomic<bool> downstream_ready{false};

    spawn_fiber(
        [&host_ready, &downstream_ready] {
            auto& sched = FiberScheduler::instance();
            while (!host_ready.load(std::memory_order_acquire)) {
                sched.note_socket_poll_wait(/*waiting=*/true, /*host_fed=*/true);
                sched.yield();
            }
            sched.note_socket_poll_wait(/*waiting=*/false, /*host_fed=*/true);
            downstream_ready.store(true, std::memory_order_release);
        },
        2,
        "h2d_pipeline_source");
    spawn_fiber(polling_body(&downstream_ready, /*host_fed=*/false, nullptr), 3, "d2d_pipeline_receiver");

    auto& sched = FiberScheduler::instance();
    ASSERT_EQ(sched.run_persistent(), RunOutcome::HostWait);
    ASSERT_EQ(sched.pump(), RunOutcome::HostWait) << "a dry pump must preserve the causal host wait";

    host_ready.store(true, std::memory_order_release);
    ASSERT_EQ(sched.pump(), RunOutcome::Completed);
}

// A later host page may be absent while the current page still has runnable work deferred to
// quiescence. The spin-release HostWait path must service that work before handing control back;
// otherwise every D2H pump suspends on the next-page H2D poll and strands the current-page producer.
TEST_F(EmuleHostWait, DeferredProducerRunsBeforeDownstreamHostWait) {
    std::atomic<bool> next_host_page_ready{false};
    std::atomic<bool> current_page_published{false};

    spawn_fiber(polling_body(&next_host_page_ready, /*host_fed=*/true, nullptr), 4, "next_page_h2d_receiver_poll");
    spawn_fiber(
        polling_body(&current_page_published, /*host_fed=*/false, nullptr), 5, "current_page_d2d_receiver_poll");
    spawn_fiber(
        [&current_page_published] {
            auto& sched = FiberScheduler::instance();
            sched.quiescence_park();
            current_page_published.store(true, std::memory_order_release);
        },
        6,
        "current_page_deferred_producer");

    auto& sched = FiberScheduler::instance();
    ASSERT_EQ(sched.run_persistent(), RunOutcome::HostWait);
    EXPECT_TRUE(current_page_published.load(std::memory_order_acquire))
        << "HostWait stranded runnable current-page work behind a next-page host poll";

    // Complete the future host dependency so the process-global scheduler registry is clean.
    next_host_page_ready.store(true, std::memory_order_release);
    ASSERT_EQ(sched.pump(), RunOutcome::Completed);
}

// The existential host root must not become a sticky exemption. Once it clears, an unrelated
// peer-fed poll remains a genuine d2d-only deadlock and must retain the peer diagnostic.
TEST_F(EmuleHostWait, D2DPollStillDeadlocksAfterHostPollClears) {
    arm_fast_watchdog();
    EXPECT_DEATH(
        {
            std::atomic<bool> host_ready{false};
            std::atomic<bool> peer_never_ready{false};
            spawn_fiber(polling_body(&host_ready, /*host_fed=*/true, nullptr), 4, "h2d_pipeline_source");
            spawn_fiber(polling_body(&peer_never_ready, /*host_fed=*/false, nullptr), 5, "d2d_pipeline_receiver");

            auto& sched = FiberScheduler::instance();
            if (sched.run_persistent() != RunOutcome::HostWait) {
                std::exit(2);
            }
            host_ready.store(true, std::memory_order_release);
            (void)sched.pump();
            std::exit(3);
        },
        "spin-polling a d2d socket");
    disarm_fast_watchdog();
}

// A d2d sender is a PEER, so dying on this regex proves the attribution, not just the outcome.
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

// The tag is sticky, so a kernel that LEAVES the loop must age out or it pins the run host-waiting.
TEST_F(EmuleHostWait, StalePollTagAgesOut) {
    arm_fast_watchdog();
    EXPECT_DEATH(
        {
            // Tag once, then spin elsewhere: freshness is this fiber's own resumes, so its yields age it.
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

// Parking must retire the tag: a parked fiber never resumes, so its freshness delta would freeze.
TEST_F(EmuleHostWait, ParkingRetiresThePollTag) {
    static uint32_t dead_key = 0;  // a key nobody will ever wake
    std::atomic<bool> never{false};

    spawn_fiber(
        [&never] {
            auto& sched = FiberScheduler::instance();
            // Freshly tagged as host-waiting…
            sched.note_socket_poll_wait(/*waiting=*/true, /*host_fed=*/true);
            sched.yield();
            // …then parks on a peer predicate. Loop required: quiescence force-releases before deciding.
            while (!never.load(std::memory_order_acquire)) {
                sched.lock();
                sched.park_locked(&dead_key);  // releases the lock; nobody wakes this
            }
        },
        5,
        "poller_then_parked");

    // Had the tag survived the park, its delta would be frozen and this would report HostWait forever.
    EXPECT_THROW(FiberScheduler::instance().run_persistent(), FiberEngineStall);
}

// The OTHER half of any_waiting_on_host(): the path a blocking socket_wait_for_pages takes.
TEST_F(EmuleHostWait, ParkedHostFedSocketWaitIsAHostWait) {
    static uint32_t credit_word = 0;
    std::atomic<bool> bytes_arrived{false};

    spawn_fiber(
        [&bytes_arrived] {
            auto& sched = FiberScheduler::instance();
            // Re-check on resume, as the real wait does: a force-release must re-park, not fall through.
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

// Nothing guarantees a next pump, so without the watchdog the job ends as a silent success.
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
            // The host walks away, so only a watchdog that outlived the return calls this a hang.
            for (int i = 0; i < 60; ++i) {
                ::usleep(200 * 1000);  // 12s total
            }
            std::exit(3);
        },
        "no global progress");
    disarm_fast_watchdog();
}

// The mirror: a watchdog still ticking after the completing pump would abort the NEXT program.
TEST_F(EmuleHostWait, CompletedRunLeavesNoWatchdogBehind) {
    arm_fast_watchdog();
    auto& sched = FiberScheduler::instance();

    std::atomic<bool> done{false};
    spawn_fiber(polling_body(&done, /*host_fed=*/true, nullptr), 16, "h2d_receiver_poll");
    ASSERT_EQ(sched.run_persistent(), RunOutcome::HostWait);
    done.store(true, std::memory_order_release);
    ASSERT_EQ(sched.pump(), RunOutcome::Completed);

    // Idle for longer than the armed backstop (3s). A leaked watchdog aborts the process here.
    ::usleep(4500 * 1000);
    disarm_fast_watchdog();

    // And the engine is still usable: a fresh run works normally.
    std::atomic<unsigned> ran{0};
    spawn_fiber([&ran] { ran.fetch_add(1); }, 17, "post_gap_program");
    ASSERT_EQ(sched.run_persistent(), RunOutcome::Completed);
    EXPECT_EQ(ran.load(), 1u);
}

TEST_F(EmuleHostWait, ComputeProgressKeepsLongBusyFiberAlive) {
    arm_fast_watchdog();
    std::atomic<uint64_t> result{0};
    spawn_fiber(
        [&result] {
            auto& sched = FiberScheduler::instance();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(4500);
            uint64_t value = 1;
            do {
                for (unsigned i = 0; i < 100000; ++i) {
                    value = value * 1664525u + 1013904223u;
                }
                sched.note_progress(1);
            } while (std::chrono::steady_clock::now() < deadline);
            result.store(value | 1u, std::memory_order_relaxed);
        },
        22,
        "long_compute_with_progress");

    EXPECT_EQ(FiberScheduler::instance().run_persistent(), RunOutcome::Completed);
    EXPECT_NE(result.load(std::memory_order_relaxed), 0u);
    disarm_fast_watchdog();
}

TEST_F(EmuleHostWait, WallWatchdogRepollsParkedFiberAfterRawStore) {
    arm_fast_watchdog();
    std::atomic<bool> predicate{false};
    std::atomic<bool> waiter_parked{false};
    std::atomic<bool> waiter_done{false};

    spawn_fiber(
        [&predicate, &waiter_parked, &waiter_done] {
            auto& sched = FiberScheduler::instance();
            while (!predicate.load(std::memory_order_acquire)) {
                sched.lock();
                if (predicate.load(std::memory_order_acquire)) {
                    sched.unlock();
                    break;
                }
                waiter_parked.store(true, std::memory_order_release);
                sched.park_locked(&predicate);
            }
            waiter_done.store(true, std::memory_order_release);
        },
        24,
        "raw_store_waiter");
    spawn_fiber(
        [&waiter_done] {
            auto& sched = FiberScheduler::instance();
            while (!waiter_done.load(std::memory_order_acquire)) {
                ::usleep(5 * 1000);
                sched.yield();
            }
        },
        25,
        "slow_runnable_peer");

    std::thread raw_store([&predicate, &waiter_parked] {
        while (!waiter_parked.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        auto& sched = FiberScheduler::instance();
        sched.lock();  // Acquiring after the marker proves the waiter completed its park.
        predicate.store(true, std::memory_order_release);
        sched.unlock();  // Deliberately no wake.
    });

    EXPECT_EQ(FiberScheduler::instance().run_persistent(), RunOutcome::Completed);
    raw_store.join();
    EXPECT_TRUE(waiter_done.load(std::memory_order_acquire));
    disarm_fast_watchdog();
}

TEST_F(EmuleHostWait, WallWatchdogReportsStarvationAfterOneRepoll) {
    arm_fast_watchdog();
    EXPECT_DEATH(
        {
            std::atomic<bool> predicate{false};
            std::atomic<unsigned> waiter_polls{0};
            spawn_fiber(
                [&predicate, &waiter_polls] {
                    auto& sched = FiberScheduler::instance();
                    while (!predicate.load(std::memory_order_acquire)) {
                        if (waiter_polls.fetch_add(1, std::memory_order_relaxed) >= 2) {
                            std::exit(4);  // More than one watchdog re-poll without progress.
                        }
                        sched.lock();
                        if (predicate.load(std::memory_order_acquire)) {
                            sched.unlock();
                            break;
                        }
                        sched.park_locked(&predicate);
                    }
                },
                26,
                "starved_waiter");
            spawn_fiber(
                [] {
                    auto& sched = FiberScheduler::instance();
                    for (;;) {
                        ::usleep(5 * 1000);
                        sched.yield();
                    }
                },
                27,
                "slow_runnable_peer");
            (void)FiberScheduler::instance().run_persistent();
        },
        "starvation after parked-fiber re-poll");
    disarm_fast_watchdog();
}

TEST_F(EmuleHostWait, SemaphoreWaitContextTracksAndClearsPredicate) {
    ::setenv("TT_EMULE_FIBER_WORKERS", "1", 1);
    static std::atomic<uint32_t> semaphore{0};
    semaphore.store(0, std::memory_order_relaxed);
    auto ctx = std::make_unique<DatamovementThreadCtx>();
    auto* ctx_ptr = ctx.get();
    ctx->bridge_l1 = reinterpret_cast<uint8_t*>(&semaphore);
    ctx->l1_size = sizeof(semaphore);
    std::atomic<bool> observed{false};
    std::atomic<bool> cleared{false};

    FiberScheduler::instance().spawn(
        [&cleared] {
            __emule_fiber_wait_semaphore(&semaphore, 1, EmuleSemaphoreWaitRelation::Minimum, [&] {
                return semaphore.load(std::memory_order_acquire) >= 1;
            });
            cleared.store(!__emule_self->sem_wait_active, std::memory_order_release);
        },
        std::move(ctx),
        ident(28, 0, "semaphore_waiter"));
    spawn_fiber(
        [ctx_ptr, &observed] {
            auto& sched = FiberScheduler::instance();
            for (;;) {
                sched.lock();
                const bool registered = ctx_ptr->sem_wait_active;
                if (registered) {
                    observed.store(
                        ctx_ptr->sem_wait_host_addr == &semaphore && ctx_ptr->sem_wait_l1_offset == 0 &&
                            ctx_ptr->sem_wait_want == 1 &&
                            ctx_ptr->sem_wait_relation == EmuleSemaphoreWaitRelation::Minimum,
                        std::memory_order_release);
                    semaphore.store(1, std::memory_order_release);
                    sched.unlock();
                    break;
                }
                sched.unlock();
                sched.yield();
            }
            sched.wake(&semaphore);
        },
        29,
        "semaphore_producer");

    const auto outcome = FiberScheduler::instance().run_persistent();
    ::unsetenv("TT_EMULE_FIBER_WORKERS");
    ASSERT_EQ(outcome, RunOutcome::Completed);
    EXPECT_TRUE(observed.load(std::memory_order_acquire));
    EXPECT_TRUE(cleared.load(std::memory_order_acquire));
}

TEST_F(EmuleHostWait, SemaphoreWaitDumpNamesExactAndMinimumPredicates) {
    static std::atomic<uint32_t> exact_semaphore{2};
    static std::atomic<uint32_t> minimum_semaphore{2};
    exact_semaphore.store(2, std::memory_order_relaxed);
    minimum_semaphore.store(2, std::memory_order_relaxed);

    auto exact_ctx = std::make_unique<DatamovementThreadCtx>();
    exact_ctx->bridge_l1 = reinterpret_cast<uint8_t*>(&exact_semaphore);
    exact_ctx->l1_size = sizeof(exact_semaphore);
    FiberScheduler::instance().spawn(
        [] {
            __emule_fiber_wait_semaphore(&exact_semaphore, 0, EmuleSemaphoreWaitRelation::Exact, [&] {
                return exact_semaphore.load(std::memory_order_acquire) == 0;
            });
        },
        std::move(exact_ctx),
        ident(30, 0, "exact_semaphore_waiter"));

    auto minimum_ctx = std::make_unique<DatamovementThreadCtx>();
    minimum_ctx->bridge_l1 = reinterpret_cast<uint8_t*>(&minimum_semaphore);
    minimum_ctx->l1_size = sizeof(minimum_semaphore);
    FiberScheduler::instance().spawn(
        [] {
            __emule_fiber_wait_semaphore(&minimum_semaphore, 4, EmuleSemaphoreWaitRelation::Minimum, [&] {
                return minimum_semaphore.load(std::memory_order_acquire) >= 4;
            });
        },
        std::move(minimum_ctx),
        ident(31, 0, "minimum_semaphore_waiter"));

    try {
        (void)FiberScheduler::instance().run_persistent();
        FAIL() << "unsatisfied semaphore waits unexpectedly completed";
    } catch (const FiberEngineStall& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("L1 sem @ 0x0 (cur=2 want=0 starved; exact)"), std::string::npos) << message;
        EXPECT_NE(message.find("L1 sem @ 0x0 (cur=2 want=4 starved; minimum)"), std::string::npos) << message;
    }
}

TEST_F(EmuleHostWait, FlightRecorderOrdersEventsAndReportsWraps) {
    const auto trace_dir = fresh_trace_dir("wrap");
    spawn_fiber(
        [] {
            for (uint64_t marker = 0; marker < 50000; ++marker) {
                FiberScheduler::instance().trace_event(EmuleTraceEventKind::Diagnostic, nullptr, marker);
            }
        },
        34,
        "flight_recorder_wrap");
    auto& scheduler = FiberScheduler::instance();
    ASSERT_EQ(scheduler.run_persistent(), RunOutcome::Completed);
    scheduler.flush_trace_for_test("wrap-test");
    clear_trace_env();

    std::ifstream events(trace_dir / "events.jsonl");
    std::string line;
    uint64_t previous = 0;
    uint64_t count = 0;
    while (std::getline(events, line)) {
        const auto pos = line.find("\"sequence\":");
        ASSERT_NE(pos, std::string::npos) << line;
        const uint64_t sequence = std::stoull(line.substr(pos + 11));
        EXPECT_GT(sequence, previous);
        previous = sequence;
        ++count;
    }
    EXPECT_GT(count, 0u);
    EXPECT_LT(count, 50000u);
    const std::string snapshot = read_file(trace_dir / "stall-snapshot.json");
    EXPECT_NE(snapshot.find("\"wrapped_events\":"), std::string::npos) << snapshot;
    EXPECT_NE(snapshot.find("\"dropped_events\":0"), std::string::npos) << snapshot;
}

TEST_F(EmuleHostWait, FlightRecorderFreezeDrainsRunningWritersWithoutScanningRunningStacks) {
    const auto trace_dir = fresh_trace_dir("concurrent-freeze");
    static std::atomic<uint32_t> synthetic_wait_word{0};
    std::atomic<unsigned> started{0};
    std::atomic<bool> stop{false};
    for (uint8_t fiber = 0; fiber < 4; ++fiber) {
        spawn_fiber(
            [&started, &stop] {
                started.fetch_add(1, std::memory_order_release);
                uint64_t marker = 0;
                while (!stop.load(std::memory_order_acquire)) {
                    __emule_self->sem_wait_host_addr = &synthetic_wait_word;
                    __emule_self->sem_wait_want = static_cast<uint32_t>(marker);
                    __emule_self->sem_wait_relation = EmuleSemaphoreWaitRelation::Minimum;
                    __emule_self->sem_wait_active = true;
                    FiberScheduler::instance().trace_event(EmuleTraceEventKind::Diagnostic, nullptr, marker);
                    __emule_self->sem_wait_active = false;
                    ++marker;
                }
            },
            static_cast<uint8_t>(40 + fiber),
            "concurrent_trace_writer");
    }

    auto& scheduler = FiberScheduler::instance();
    std::thread freezer([&] {
        while (started.load(std::memory_order_acquire) < 1) {
            std::this_thread::yield();
        }
        scheduler.flush_trace_for_test("concurrent-freeze-test");
        stop.store(true, std::memory_order_release);
    });
    ASSERT_EQ(scheduler.run_persistent(), RunOutcome::Completed);
    freezer.join();

    const std::string snapshot = read_file(trace_dir / "stall-snapshot.json");
    const auto running_snapshot = snapshot.find("\"state\":\"running\"");
    ASSERT_NE(running_snapshot, std::string::npos) << snapshot;
    const auto next_fiber = snapshot.find("},{\"fiber_id\":", running_snapshot);
    const std::string running_entry = snapshot.substr(running_snapshot, next_fiber - running_snapshot);
    EXPECT_EQ(running_entry.find("\"semaphore_wait\":"), std::string::npos) << running_entry;

    const std::string stacks = read_file(trace_dir / "stacks.json");
    const auto running = stacks.find("\"state\":\"running\"");
    ASSERT_NE(running, std::string::npos) << stacks;
    EXPECT_NE(stacks.find("\"candidates\":[]", running), std::string::npos) << stacks;

    std::ifstream events(trace_dir / "events.jsonl");
    std::string line;
    uint64_t previous = 0;
    while (std::getline(events, line)) {
        const auto pos = line.find("\"sequence\":");
        ASSERT_NE(pos, std::string::npos) << line;
        const uint64_t sequence = std::stoull(line.substr(pos + 11));
        EXPECT_GT(sequence, previous);
        previous = sequence;
    }
    EXPECT_GT(previous, 0u);
    events.close();

    spawn_fiber(
        [] { FiberScheduler::instance().trace_event(EmuleTraceEventKind::Diagnostic, nullptr, 12648430); },
        48,
        "same_directory_trace_reuse");
    ASSERT_EQ(scheduler.run_persistent(), RunOutcome::Completed);
    scheduler.flush_trace_for_test("same-directory-reuse-test");
    clear_trace_env();
    const std::string reused_events = read_file(trace_dir / "events.jsonl");
    EXPECT_NE(reused_events.find("\"a\":12648430"), std::string::npos) << reused_events;
}

TEST_F(EmuleHostWait, FlightRecorderSerializesNormalizedTransferSchemas) {
    const auto trace_dir = fresh_trace_dir("transfer-schema");
    static uint32_t destination = 0;
    spawn_fiber(
        [] {
            auto& scheduler = FiberScheduler::instance();
            scheduler.trace_event(EmuleTraceEventKind::NocIssue, nullptr, 11, 22, 33, 44);
            scheduler.trace_event(EmuleTraceEventKind::FabricPublish, &destination, 1, 2, 3, (uint64_t{4} << 32) | 5);
        },
        44,
        "transfer_schema");
    auto& scheduler = FiberScheduler::instance();
    ASSERT_EQ(scheduler.run_persistent(), RunOutcome::Completed);
    scheduler.flush_trace_for_test("transfer-schema-test");
    clear_trace_env();

    const std::string events = read_file(trace_dir / "events.jsonl");
    EXPECT_NE(
        events.find("\"transfer_source\":11,\"transfer_destination\":22,"
                    "\"byte_count\":33,\"transfer_metadata\":44"),
        std::string::npos)
        << events;
    EXPECT_NE(
        events.find("\"source_chip\":1,\"destination_chip\":2,"
                    "\"destination_noc_address\":3,\"byte_count\":5,\"send_type\":4"),
        std::string::npos)
        << events;
}

TEST_F(EmuleHostWait, DeadlockSnapshotUsesDirectionSpecificCbProducers) {
    const auto trace_dir = fresh_trace_dir("cb-direction");
    static tt_emule::CBSyncState cbs[__EMULE_CTX_MAX_CBS];
    cbs[4].page_size = 2048;
    cbs[4].num_pages = 1;
    cbs[4].occupied.store(0, std::memory_order_relaxed);
    cbs[5].page_size = 2048;
    cbs[5].num_pages = 1;
    cbs[5].occupied.store(1, std::memory_order_relaxed);

    spawn_fiber(
        [] {
            auto& scheduler = FiberScheduler::instance();
            scheduler.trace_event(EmuleTraceEventKind::CbPush, &cbs[4], 1, 0, 1, 1);
            scheduler.trace_event(EmuleTraceEventKind::CbPop, &cbs[4], 1, 1, 0, 1);
            scheduler.trace_event(EmuleTraceEventKind::CbPop, &cbs[5], 1, 1, 0, 1);
            scheduler.trace_event(EmuleTraceEventKind::CbPush, &cbs[5], 1, 0, 1, 1);
        },
        45,
        "cb_direction_publishers");

    auto front_ctx = std::make_unique<DatamovementThreadCtx>();
    front_ctx->cbs = cbs;
    FiberScheduler::instance().spawn(
        [] {
            __emule_fiber_wait_cb(
                &cbs[4], 4, 1, false, [] { return cbs[4].occupied.load(std::memory_order_acquire) >= 1; });
        },
        std::move(front_ctx),
        ident(46, 0, "cb_front_waiter"));

    auto reserve_ctx = std::make_unique<DatamovementThreadCtx>();
    reserve_ctx->cbs = cbs;
    FiberScheduler::instance().spawn(
        [] {
            __emule_fiber_wait_cb(
                &cbs[5], 5, 1, true, [] { return cbs[5].occupied.load(std::memory_order_acquire) == 0; });
        },
        std::move(reserve_ctx),
        ident(47, 0, "cb_reserve_waiter"));

    EXPECT_THROW((void)FiberScheduler::instance().run_persistent(), FiberEngineStall);
    clear_trace_env();

    const std::string snapshot = read_file(trace_dir / "stall-snapshot.json");
    const auto front = snapshot.find("\"cb_id\":4");
    ASSERT_NE(front, std::string::npos) << snapshot;
    EXPECT_NE(snapshot.find("\"last_producer_kind\":\"cb_push\"", front), std::string::npos) << snapshot;
    const auto reserve = snapshot.find("\"cb_id\":5");
    ASSERT_NE(reserve, std::string::npos) << snapshot;
    EXPECT_NE(snapshot.find("\"last_producer_kind\":\"cb_pop\"", reserve), std::string::npos) << snapshot;
}

TEST_F(EmuleHostWait, DeadlockSnapshotNamesSemaphoreCbWorkersAndStacks) {
    const auto trace_dir = fresh_trace_dir("deadlock");
    static std::atomic<uint32_t> semaphore{0};
    semaphore.store(0, std::memory_order_relaxed);
    static tt_emule::CBSyncState cbs[__EMULE_CTX_MAX_CBS];
    cbs[3].page_size = 2048;
    cbs[3].num_pages = 8;
    cbs[3].occupied.store(0, std::memory_order_relaxed);

    auto sem_ctx = std::make_unique<DatamovementThreadCtx>();
    sem_ctx->bridge_l1 = reinterpret_cast<uint8_t*>(&semaphore);
    sem_ctx->l1_size = sizeof(semaphore);
    FiberScheduler::instance().spawn(
        [] {
            __emule_fiber_wait_semaphore(&semaphore, 4, EmuleSemaphoreWaitRelation::Minimum, [] {
                return semaphore.load(std::memory_order_acquire) >= 4;
            });
        },
        std::move(sem_ctx),
        ident(32, 0, "synthetic_semaphore_waiter"));

    auto cb_ctx = std::make_unique<DatamovementThreadCtx>();
    cb_ctx->cbs = cbs;
    FiberScheduler::instance().spawn(
        [] {
            __emule_fiber_wait_cb(
                &cbs[3], 3, 2, false, [] { return cbs[3].occupied.load(std::memory_order_acquire) >= 2; });
        },
        std::move(cb_ctx),
        ident(33, 0, "synthetic_cb_waiter"));

    EXPECT_THROW((void)FiberScheduler::instance().run_persistent(), FiberEngineStall);
    clear_trace_env();

    const std::string snapshot = read_file(trace_dir / "stall-snapshot.json");
    EXPECT_NE(snapshot.find("synthetic_semaphore_waiter"), std::string::npos) << snapshot;
    EXPECT_NE(snapshot.find("\"wanted\":4"), std::string::npos) << snapshot;
    EXPECT_NE(snapshot.find("\"relation\":\"minimum\""), std::string::npos) << snapshot;
    EXPECT_NE(snapshot.find("\"status\":\"starved\""), std::string::npos) << snapshot;
    EXPECT_NE(snapshot.find("synthetic_cb_waiter"), std::string::npos) << snapshot;
    EXPECT_NE(snapshot.find("\"cb_id\":3"), std::string::npos) << snapshot;
    EXPECT_NE(snapshot.find("\"requested_pages\":2"), std::string::npos) << snapshot;
    EXPECT_NE(snapshot.find("\"missing_producer\":true"), std::string::npos) << snapshot;
    EXPECT_NE(snapshot.find("\"threadpool_size\":"), std::string::npos) << snapshot;
    EXPECT_NE(snapshot.find("\"home_worker\":"), std::string::npos) << snapshot;

    const std::string stacks = read_file(trace_dir / "stacks.json");
    EXPECT_NE(stacks.find("\"saved_rip\":"), std::string::npos) << stacks;
    EXPECT_NE(stacks.find("\"saved_rsp\":"), std::string::npos) << stacks;
    EXPECT_NE(stacks.find("\"candidates\":["), std::string::npos) << stacks;
    EXPECT_TRUE(std::filesystem::exists(trace_dir / "wait-for.dot"));
}

TEST_F(EmuleHostWait, InfiniteBusyFiberWithoutProgressTripsWatchdog) {
    arm_fast_watchdog();
    EXPECT_DEATH(
        {
            std::atomic<uint64_t> result{0};
            spawn_fiber(
                [&result] {
                    uint64_t value = 1;
                    for (;;) {
                        for (unsigned i = 0; i < 100000; ++i) {
                            value = value * 1664525u + 1013904223u;
                        }
                        result.store(value, std::memory_order_relaxed);
                    }
                },
                23,
                "infinite_compute_without_progress");
            (void)FiberScheduler::instance().run_persistent();
        },
        "no global progress");
    disarm_fast_watchdog();
}

// Outcome flags are per-launch: a stale host_wait_ frees the NEXT dispatch's keepalives under it.
TEST_F(EmuleHostWait, EmptyLaunchAfterAStalledRunIsNotAHostWait) {
    auto& sched = FiberScheduler::instance();

    // Wedge a run: park on a key nobody wakes, no host-facing wait — the tier-1 teardown path.
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

    // A dispatch registering nothing hits the empty-registry early return: an end, not a resume.
    EXPECT_EQ(sched.run_persistent(), RunOutcome::Completed);
    // And a pump against no registry stays a no-op rather than acting on a phantom run.
    EXPECT_EQ(sched.pump(), RunOutcome::Completed);
}

// One CB slot per fiber, kept by the starving CB: recording the LAST probed points at a healthy one.
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

// Generations gate the keepalive reclaim: a false "dead" frees DFB/ASAN state under a running kernel.
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

    // A later dispatch opens a newer generation; the parked run's older one is the floor.
    sched.begin_spawn_generation();
    EXPECT_EQ(sched.oldest_live_spawn_generation(), parked_gen);

    done.store(true, std::memory_order_release);
    ASSERT_EQ(sched.pump(), RunOutcome::Completed);
    EXPECT_EQ(sched.oldest_live_spawn_generation(), UINT64_MAX);
}

// An age bound cannot express this: a parked run pins ONE generation for the whole sequence.
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

    // B's keepalives are reclaimable NOW, even though older A is still parked and pins the oldest.
    EXPECT_FALSE(sched.spawn_generation_is_live(gen_b))
        << "a finished generation must be reclaimable while an OLDER one is still parked";
    EXPECT_TRUE(sched.spawn_generation_is_live(gen_a));
    EXPECT_EQ(sched.oldest_live_spawn_generation(), gen_a);

    done.store(true, std::memory_order_release);
    ASSERT_EQ(sched.pump(), RunOutcome::Completed);
    EXPECT_FALSE(sched.spawn_generation_is_live(gen_a));
}

// The normal host-interleaved case: integrate only the new fibers, disturbing no live one.
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

    // A second program on top of the parked one runs to completion; the run quiesces to HostWait again.
    std::atomic<unsigned> compute_entries{0};
    sched.begin_spawn_generation();
    spawn_fiber([&compute_entries] { compute_entries.fetch_add(1); }, 10, "compute_stage");

    ASSERT_EQ(sched.run_persistent(), RunOutcome::HostWait);
    EXPECT_EQ(compute_entries.load(), 1u) << "the newly registered fiber never ran";
    EXPECT_EQ(receiver_entries.load(), 1u) << "the live fiber's body was re-entered";
    // The receiver is still the oldest thing alive — the second program has finished.
    EXPECT_EQ(sched.oldest_live_spawn_generation(), receiver_gen);

    // A third dispatch: the retirement pass is the only non-teardown site that frees a 1 MiB stack.
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

// drain_device() catches "will not drain" narrowly, so a kernel fault must not be that type.
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

    // Propagates verbatim: widened to FiberEngineStall, drain_device() would swallow real faults.
    EXPECT_THROW(FiberScheduler::instance().run_persistent(), std::out_of_range);
}

// A kernel fault outranks a host wait, or the throw lands on whichever program tears down next.
TEST_F(EmuleHostWait, KernelFaultOutranksHostWait) {
    std::atomic<bool> never{false};
    spawn_fiber(polling_body(&never, /*host_fed=*/true, nullptr), 13, "h2d_receiver_poll");
    spawn_fiber([] { throw std::out_of_range("DFB range"); }, 14, "faulting_kernel");

    // Even though a host-fed poller is live and would otherwise force HostWait.
    EXPECT_THROW(FiberScheduler::instance().run_persistent(), std::out_of_range);
}

// The drain backstop reads this rather than re-parsing the env var, which env_size reads differently.
TEST_F(EmuleHostWait, StallLimitIsPositiveAndStable) {
    const uint64_t a = FiberScheduler::host_wait_stall_limit();
    EXPECT_GT(a, 0u) << "a zero bound would make pump() escalate on its first no-progress pump";
    // Read once at first use, so it cannot drift — what lets the runner cache its floor as a static.
    EXPECT_EQ(a, FiberScheduler::host_wait_stall_limit());
}

}  // namespace
