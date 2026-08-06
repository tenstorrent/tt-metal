// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only tests for tt::tt_metal::GraphTracker focused on its multi-threading
// contract: `processors` and `hook` are thread_local, so a graph capture pushed
// on thread A only observes ops dispatched on thread A, and concurrent
// push/pop on another thread cannot race with the dispatch hot path. Work handed
// to a thread pool is the one exception — `wrap_with_current_context` installs the
// enqueuing thread's processors on the worker for the duration of the task.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

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

// Stands in for permanently-registered background processors such as ShmTrackingProcessor,
// which is pushed at device init and never popped.
class BackgroundProcessor : public CountingProcessor {
public:
    bool is_capture_processor() const override { return false; }
};

class SuppressingHooks : public IGraphHooks {
public:
    bool hook_allocate(const Buffer* /*buffer*/) override { return true; }
    bool hook_deallocate(Buffer* /*buffer*/) override { return true; }
    bool hook_program(Program* /*program*/) override { return true; }
    bool hook_write_to_device(const Buffer* /*buffer*/) override { return true; }
    bool hook_read_from_device(Buffer* /*buffer*/) override { return true; }
    bool hook_read_from_device(const distributed::MeshBuffer* /*mesh_buffer*/) override { return true; }
    bool hook_write_to_device(const distributed::MeshBuffer* /*mesh_buffer*/) override { return true; }
};

}  // namespace

// These tests need a clean per-thread capture stack, but calling GraphTracker::clear() outright
// would drop processors registered process-wide by whatever else shares this binary. In particular
// ShmTrackingProcessor is pushed behind a once-flag at device init (tt_metal/impl/device/device.cpp),
// so clearing it after a device test has run disables SHM tracking for every later test with no way
// to re-register. Snapshot the main thread's state and put it back instead.
class GraphTrackerThreading : public ::testing::Test {
protected:
    void SetUp() override {
        auto& tracker = GraphTracker::instance();
        saved_processors_ = tracker.get_processors();
        saved_hook_ = tracker.get_hook();
        tracker.clear();
    }

    void TearDown() override {
        auto& tracker = GraphTracker::instance();
        tracker.clear();
        for (const auto& processor : saved_processors_) {
            tracker.push_processor(processor);
        }
        if (saved_hook_) {
            tracker.add_hook(saved_hook_);
        }
    }

private:
    std::vector<std::shared_ptr<IGraphProcessor>> saved_processors_;
    std::shared_ptr<IGraphHooks> saved_hook_;
};

TEST_F(GraphTrackerThreading, SingleThreadCapturesEachEventOnce) {
    auto& tracker = GraphTracker::instance();

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
TEST_F(GraphTrackerThreading, ProcessorsAreIsolatedPerThread) {
    constexpr int kIterations = 1000;
    constexpr int kNumThreads = 2;

    std::atomic<int> ready_count{0};
    std::atomic<bool> go{false};

    auto run_one_thread = [&](const std::shared_ptr<CountingProcessor>& proc) {
        auto& tracker = GraphTracker::instance();
        // A freshly spawned thread starts with an empty thread_local stack; no clear() needed.
        ASSERT_TRUE(tracker.get_processors().empty());
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
TEST_F(GraphTrackerThreading, ConcurrentPushPopAndTrackDoNotRace) {
    constexpr auto kDuration = std::chrono::milliseconds(200);

    std::atomic<bool> stop{false};
    std::atomic<bool> dispatcher_ran{false};

    std::thread mutator([&] {
        auto& tracker = GraphTracker::instance();
        while (!stop.load()) {
            tracker.push_processor(std::make_shared<CountingProcessor>());
            tracker.pop_processor();
        }
    });

    auto dispatcher_proc = std::make_shared<CountingProcessor>();
    std::thread dispatcher([&] {
        auto& tracker = GraphTracker::instance();
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

// Work offloaded to a worker thread while a capture is active must observe the
// same processor as the enqueuing thread. This reproduces the CCL/collective
// case (ttnn-visualizer #1684): a function_start recorded on the main thread
// whose matching function_end runs on a worker thread. With context
// propagation the single processor sees both events (balanced), instead of the
// worker's empty thread-local list silently dropping the end.
TEST_F(GraphTrackerThreading, WrapPropagatesContextToWorkerThread) {
    auto& tracker = GraphTracker::instance();

    auto processor = std::make_shared<CountingProcessor>();
    tracker.push_processor(processor);

    // Begin an op on the main thread.
    tracker.track_function_start("op");

    // Snapshot the current capture context and hand the matching end off to a
    // worker thread (as the dispatch thread pool would).
    auto task = tracker.wrap_with_current_context([]() { GraphTracker::instance().track_function_end(); });

    std::thread worker([task = std::move(task)]() mutable {
        auto& worker_tracker = GraphTracker::instance();
        // Worker starts with an empty thread-local capture.
        EXPECT_FALSE(worker_tracker.is_enabled());
        task();
        // After the wrapped task completes, the worker's own (empty) context is restored.
        EXPECT_FALSE(worker_tracker.is_enabled());
    });
    worker.join();

    tracker.pop_processor();

    EXPECT_EQ(processor->function_starts.load(), 1);
    EXPECT_EQ(processor->function_ends.load(), 1);
}

// When no capture is active on the enqueuing thread, wrap_with_current_context
// must return a transparent wrapper (the task still runs, nothing installed).
TEST_F(GraphTrackerThreading, WrapIsTransparentWhenNoCaptureActive) {
    auto& tracker = GraphTracker::instance();

    std::atomic<int> ran{0};
    auto task = tracker.wrap_with_current_context([&ran]() { ran.fetch_add(1); });
    task();

    EXPECT_EQ(ran.load(), 1);
    EXPECT_TRUE(tracker.get_processors().empty());
}

// Background processors (ShmTrackingProcessor is registered at device init and never
// popped) leave `processors` non-empty on an ordinary run. The no-capture fast path must
// key off is_enabled(), not emptiness, or every dispatch would copy the processor stack
// onto the worker — adding allocations to the hot path and letting a background processor
// observe worker-thread events it never saw before.
TEST_F(GraphTrackerThreading, WrapIsTransparentWithOnlyBackgroundProcessors) {
    auto& tracker = GraphTracker::instance();

    auto background = std::make_shared<BackgroundProcessor>();
    tracker.push_processor(background);
    ASSERT_FALSE(tracker.is_enabled());
    ASSERT_FALSE(tracker.get_processors().empty());

    auto task = tracker.wrap_with_current_context([]() { GraphTracker::instance().track_function_end(); });

    std::atomic<bool> worker_saw_processors{true};
    std::thread worker([task = std::move(task), &worker_saw_processors]() mutable {
        auto& worker_tracker = GraphTracker::instance();
        task();
        worker_saw_processors.store(!worker_tracker.get_processors().empty());
    });
    worker.join();

    tracker.pop_processor();

    EXPECT_FALSE(worker_saw_processors.load());
    EXPECT_EQ(background->function_ends.load(), 0);
}

// Hooks are intentionally left behind: they change behaviour (under RunMode::NO_DISPATCH
// they suppress writes and program dispatch) rather than just observing it, so propagating
// them would alter what offloaded work does. Propagation stays purely additive.
TEST_F(GraphTrackerThreading, WrapDoesNotPropagateHook) {
    auto& tracker = GraphTracker::instance();

    auto processor = std::make_shared<CountingProcessor>();
    tracker.push_processor(processor);
    ASSERT_TRUE(tracker.add_hook(std::make_shared<SuppressingHooks>()));

    auto task = tracker.wrap_with_current_context([]() {});

    std::atomic<bool> worker_had_hook{true};
    std::thread worker([task = std::move(task), &worker_had_hook]() mutable {
        auto& worker_tracker = GraphTracker::instance();
        task();
        worker_had_hook.store(worker_tracker.get_hook() != nullptr);
    });
    worker.join();

    tracker.pop_processor();

    EXPECT_FALSE(worker_had_hook.load());
}

// The wrapper moves its snapshot into the tracker on entry and takes it back on exit, which
// avoids copying the processor vector per task. std::function makes no single-invocation
// promise, so guard that the hand-back actually happens: a second call must still propagate
// rather than silently install an empty (moved-from) stack.
TEST_F(GraphTrackerThreading, WrapCanRunMoreThanOnce) {
    auto& tracker = GraphTracker::instance();

    auto processor = std::make_shared<CountingProcessor>();
    tracker.push_processor(processor);
    auto task = tracker.wrap_with_current_context([]() { GraphTracker::instance().track_function_start("op"); });

    std::thread worker([task = std::move(task)]() mutable {
        task();
        task();
    });
    worker.join();

    tracker.pop_processor();

    EXPECT_EQ(processor->function_starts.load(), 2);
}

// The wrapper must restore whatever capture state the worker thread already had,
// even if that thread was itself driving an unrelated capture.
TEST_F(GraphTrackerThreading, WrapRestoresPreexistingWorkerContext) {
    auto& outer = GraphTracker::instance();

    auto propagated = std::make_shared<CountingProcessor>();
    outer.push_processor(propagated);
    auto task = outer.wrap_with_current_context([]() { GraphTracker::instance().track_function_end(); });
    outer.pop_processor();

    std::thread worker([task = std::move(task)]() mutable {
        auto& tracker = GraphTracker::instance();

        // The worker owns its own capture before running the propagated task.
        auto local_proc = std::make_shared<CountingProcessor>();
        tracker.push_processor(local_proc);

        task();  // installs `propagated`, runs, then restores `local_proc`

        // The worker's own processor is back on top and unaffected by the task.
        EXPECT_EQ(tracker.get_processors().size(), 1u);
        tracker.track_function_end();
        tracker.pop_processor();

        // Only the local end was seen by the worker's own processor.
        EXPECT_EQ(local_proc->function_ends.load(), 1);
    });
    worker.join();

    // The propagated processor saw exactly the one end fired inside the task.
    EXPECT_EQ(propagated->function_ends.load(), 1);
}

}  // namespace tt::tt_metal::graph_tracking_test
