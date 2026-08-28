// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only tests for tt::tt_metal::GraphTracker focused on its multi-threading
// contract: `processors` and `hook` are thread_local, so a graph capture pushed
// on thread A only observes ops dispatched on thread A, and concurrent
// push/pop on another thread cannot race with the dispatch hot path.
//
// Background processors are the other half of that contract. They are process-wide rather
// than thread_local, registered once and never popped, so the thing that can go wrong is
// registering one twice -- see the last test.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

#include <tt-metalium/graph_tracking.hpp>
#include <internal/graph_tracking.hpp>

namespace tt::tt_metal::graph_tracking_test {

namespace {

class CountingProcessor : public IGraphProcessor {
public:
    std::atomic<int> function_starts{0};
    std::atomic<int> function_ends{0};

    void track_function_start(
        std::string_view /*function_name*/, std::span<TrackedArgument> /*input_parameters*/) override {
        function_starts.fetch_add(1);
    }

    void track_function_end() override { function_ends.fetch_add(1); }

    void track_function_end(const std::any& /*output_tensors*/) override { function_ends.fetch_add(1); }
};

// Process-wide observer for the background-registration test below. Counts buffer events and
// does nothing else: background processors are never popped, so this one outlives its test and
// keeps being notified for the rest of the binary. is_capture_processor() must be false, or it
// would make GraphTracker::is_enabled() true for every later test.
class BackgroundCountingProcessor : public IGraphProcessor {
public:
    inline static std::atomic<int> allocate_calls{0};
    inline static std::atomic<int> instances{0};

    BackgroundCountingProcessor() { instances.fetch_add(1); }

    bool is_capture_processor() const override { return false; }

    void track_allocate(const tt::tt_metal::Buffer* /*buffer*/) override { allocate_calls.fetch_add(1); }
};

}  // namespace

TEST(GraphTrackerThreading, CPU_SingleThreadCapturesEachEventOnce) {
    auto& tracker = GraphTracker::instance();
    tracker.clear();

    auto processor = std::make_shared<CountingProcessor>();
    tracker.push_processor(processor);

    constexpr int kIterations = 100;
    for (int i = 0; i < kIterations; ++i) {
        tracker.track_function_start("op");
        tracker.track_function_end();
    }

    tracker.pop_processor();

    EXPECT_EQ(processor->function_starts.load(), kIterations);
    EXPECT_EQ(processor->function_ends.load(), kIterations);
    EXPECT_TRUE(tracker.get_processors().empty());
}

// Per-thread storage means a processor pushed on thread A only sees ops
// dispatched on thread A — never on thread B firing in parallel. Each thread
// pushes its processor before either starts dispatching (via a barrier), so
// if storage were shared both threads would iterate both processors and each
// would see 2 * kIterations events.
TEST(GraphTrackerThreading, CPU_ProcessorsAreIsolatedPerThread) {
    constexpr int kIterations = 1000;
    constexpr int kNumThreads = 2;

    std::atomic<int> ready_count{0};
    std::atomic<bool> go{false};

    auto run_one_thread = [&](const std::shared_ptr<CountingProcessor>& proc) {
        auto& tracker = GraphTracker::instance();
        tracker.clear();
        tracker.push_processor(proc);

        ready_count.fetch_add(1);
        while (!go.load()) {
            std::this_thread::yield();
        }

        for (int i = 0; i < kIterations; ++i) {
            tracker.track_function_start("op");
            tracker.track_function_end();
        }
        tracker.pop_processor();
    };

    auto proc_a = std::make_shared<CountingProcessor>();
    auto proc_b = std::make_shared<CountingProcessor>();

    std::thread t_a(run_one_thread, proc_a);
    std::thread t_b(run_one_thread, proc_b);

    while (ready_count.load() < kNumThreads) {
        std::this_thread::yield();
    }
    go.store(true);

    t_a.join();
    t_b.join();

    EXPECT_EQ(proc_a->function_starts.load(), kIterations);
    EXPECT_EQ(proc_a->function_ends.load(), kIterations);
    EXPECT_EQ(proc_b->function_starts.load(), kIterations);
    EXPECT_EQ(proc_b->function_ends.load(), kIterations);
}

// Reproduces the race from tt-mlir#8302. One thread spins push/pop_processor
// while another spins track_function_start.
TEST(GraphTrackerThreading, CPU_ConcurrentPushPopAndTrackDoNotRace) {
    constexpr auto kDuration = std::chrono::milliseconds(200);

    std::atomic<bool> stop{false};
    std::atomic<bool> dispatcher_ran{false};

    std::thread mutator([&] {
        auto& tracker = GraphTracker::instance();
        tracker.clear();
        while (!stop.load()) {
            tracker.push_processor(std::make_shared<CountingProcessor>());
            tracker.pop_processor();
        }
    });

    auto dispatcher_proc = std::make_shared<CountingProcessor>();
    std::thread dispatcher([&] {
        auto& tracker = GraphTracker::instance();
        tracker.clear();
        tracker.push_processor(dispatcher_proc);
        while (!stop.load()) {
            tracker.track_function_start("op");
            tracker.track_function_end();
            dispatcher_ran.store(true);
        }
        tracker.pop_processor();
    });

    std::this_thread::sleep_for(kDuration);
    stop.store(true);

    mutator.join();
    dispatcher.join();

    if (dispatcher_ran.load()) {
        EXPECT_GT(dispatcher_proc->function_starts.load(), 0);
    }
    EXPECT_EQ(dispatcher_proc->function_starts.load(), dispatcher_proc->function_ends.load());
}

// Registering a background processor used to be check-then-act: has_background_processor_of_type()
// released its shared lock before push_background_processor() took the exclusive one, so two
// threads could both observe absence and both register. That is reachable from ordinary
// multi-device startup, where Device::initialize() runs concurrently -- and the consequence is not
// a harmless duplicate: every buffer allocation is then reported to the SHM tracker twice, so the
// device-wide totals read double.
//
// register_background_processor_once() does the lookup and the insertion under one exclusive lock.
// Asserted several ways, because "exactly one registration happened" and "the observer fires once
// per event" are different claims and only the second is what actually breaks.
//
// On what this does and does not prove. Against the old check-then-act it is decisive whenever the
// gap between releasing the shared lock and taking the exclusive one is non-trivial: ablated with
// 200us of work in that gap, all 16 threads register and a single allocation is reported 16 times.
// Ablated with nothing in the gap it is sub-microsecond and one thread usually completes the whole
// sequence before the others finish acquiring the shared lock, so the interleaving often does not
// occur at all. Which is to say the assertions below are a genuine regression guard on the
// contract -- one registration, one construction, one notification, later calls inert -- and only
// an opportunistic one on the interleaving. The single lock is what makes the interleaving
// impossible; this cannot prove that, and correct code passes it deterministically, so it fails
// only ever for a real reason.
TEST(GraphTrackerThreading, CPU_ConcurrentBackgroundRegistrationRegistersExactlyOne) {
    constexpr int kThreads = 16;

    BackgroundCountingProcessor::allocate_calls.store(0);
    BackgroundCountingProcessor::instances.store(0);

    std::atomic<int> registered{0};
    std::atomic<int> arrived{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; i++) {
        threads.emplace_back([&] {
            // Arrival barrier, spun tight on purpose. A flag set after the spawn loop only proves
            // the threads were created, and waiting on it with yield() is actively harmful: the
            // last thread to arrive never waits, so it runs alone while every other thread sits
            // descheduled -- which serialises the very calls that need to overlap. Busy-spinning
            // until all have arrived keeps them hot on their cores so they leave together, within
            // tens of nanoseconds. Bounded with a yield fallback so this still terminates on a
            // machine with fewer cores than threads.
            arrived.fetch_add(1, std::memory_order_acq_rel);
            for (int spins = 0; arrived.load(std::memory_order_acquire) < kThreads; spins++) {
                if (spins > 1'000'000) {
                    std::this_thread::yield();
                }
            }
            if (internal::register_background_processor_once(typeid(BackgroundCountingProcessor), [] {
                    return std::make_shared<BackgroundCountingProcessor>();
                })) {
                registered.fetch_add(1);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(registered.load(), 1) << "expected exactly one of " << kThreads
                                    << " concurrent callers to report that it registered the processor";
    EXPECT_EQ(BackgroundCountingProcessor::instances.load(), 1)
        << "the factory ran more than once, so a processor was constructed for a registration that "
           "should have been skipped";

    // The symptom the race actually produces. track_allocate() passes the pointer through without
    // dereferencing it and this processor ignores it, so no real buffer is needed.
    GraphTracker::instance().track_allocate(nullptr);
    EXPECT_EQ(BackgroundCountingProcessor::allocate_calls.load(), 1)
        << "one buffer allocation was reported " << BackgroundCountingProcessor::allocate_calls.load()
        << " times; a duplicate background processor doubles every figure the tracker publishes";

    // A later call must still be a no-op, which is what makes repeated device init safe.
    EXPECT_FALSE(internal::register_background_processor_once(typeid(BackgroundCountingProcessor), [] {
        return std::make_shared<BackgroundCountingProcessor>();
    })) << "re-registering an already-registered type reported a fresh registration";
}

}  // namespace tt::tt_metal::graph_tracking_test
