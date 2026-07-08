// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only tests for tt::tt_metal::GraphTracker focused on its multi-threading
// contract: `processors` and `hook` are thread_local, so a graph capture pushed
// on thread A only observes ops dispatched on thread A, and concurrent
// push/pop on another thread cannot race with the dispatch hot path.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <span>
#include <string_view>
#include <thread>

#include <tt-metalium/graph_tracking.hpp>

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

}  // namespace

TEST(GraphTrackerThreading, SingleThreadCapturesEachEventOnce) {
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
TEST(GraphTrackerThreading, ProcessorsAreIsolatedPerThread) {
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
TEST(GraphTrackerThreading, ConcurrentPushPopAndTrackDoNotRace) {
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

}  // namespace tt::tt_metal::graph_tracking_test
