// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only tests for tt::tt_metal::GraphTracker. Two areas are covered: its
// multi-threading contract (`processors` and `hook` are thread_local, so a graph
// capture pushed on thread A only observes ops dispatched on thread A, and
// concurrent push/pop on another thread cannot race with the dispatch hot path),
// and ScopedTrackedFunction's guarantee that a tracked scope is always closed,
// including when it is left by an exception.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
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
    std::atomic<int> function_aborts{0};
    std::vector<std::string> abort_reasons;
    std::atomic<int> unwinds{0};
    std::vector<std::string> unwind_reasons;

    void track_function_start(
        std::string_view /*function_name*/, std::span<TrackedArgument> /*input_parameters*/) override {
        function_starts.fetch_add(1);
    }

    void track_function_end() override { function_ends.fetch_add(1); }

    void track_function_end(const std::any& /*output_tensors*/) override { function_ends.fetch_add(1); }

    void track_function_abort(std::string_view reason) override {
        function_aborts.fetch_add(1);
        abort_reasons.emplace_back(reason);
    }

    void unwind_open_functions(std::string_view reason) override {
        unwinds.fetch_add(1);
        unwind_reasons.emplace_back(reason);
    }
};

// Scoped push/pop so a failing expectation cannot leave a processor registered for later tests.
class ScopedProcessor {
public:
    explicit ScopedProcessor(std::shared_ptr<CountingProcessor> processor) : processor_(std::move(processor)) {
        GraphTracker::instance().clear();
        GraphTracker::instance().push_processor(processor_);
    }
    ~ScopedProcessor() { GraphTracker::instance().clear(); }

    ScopedProcessor(const ScopedProcessor&) = delete;
    ScopedProcessor& operator=(const ScopedProcessor&) = delete;

private:
    std::shared_ptr<CountingProcessor> processor_;
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

TEST(ScopedTrackedFunction, ReportsOutputOnTheSuccessPath) {
    auto processor = std::make_shared<CountingProcessor>();
    const ScopedProcessor registration(processor);

    int output = 0;
    {
        ScopedTrackedFunction tracked("op");
        tracked.end(output);
    }

    EXPECT_EQ(processor->function_starts.load(), 1);
    EXPECT_EQ(processor->function_ends.load(), 1);
    EXPECT_EQ(processor->function_aborts.load(), 0);
}

TEST(ScopedTrackedFunction, ClosesTheScopeWhenLeftWithoutAnExplicitEnd) {
    auto processor = std::make_shared<CountingProcessor>();
    const ScopedProcessor registration(processor);

    {
        ScopedTrackedFunction tracked("op");
    }

    EXPECT_EQ(processor->function_starts.load(), 1);
    EXPECT_EQ(processor->function_ends.load(), 1);
    EXPECT_EQ(processor->function_aborts.load(), 0);
}

// The regression behind #28836: a scope left by an exception used to emit no end at all.
TEST(ScopedTrackedFunction, AbortsWhenUnwinding) {
    auto processor = std::make_shared<CountingProcessor>();
    const ScopedProcessor registration(processor);

    EXPECT_THROW(
        {
            ScopedTrackedFunction tracked("op");
            throw std::runtime_error("circular buffers clash with L1 buffers");
        },
        std::runtime_error);

    EXPECT_EQ(processor->function_starts.load(), 1);
    EXPECT_EQ(processor->function_ends.load(), 0);
    ASSERT_EQ(processor->function_aborts.load(), 1);
    // The destructor runs during unwinding, before any handler, so the message is out of reach
    // there. Pinned so the reason stays a documented gap rather than a surprise.
    EXPECT_EQ(processor->abort_reasons.front(), "");
}

// A catch block does have the message, and abort() is how it reaches the trace.
TEST(ScopedTrackedFunction, ExplicitAbortCarriesTheReason) {
    auto processor = std::make_shared<CountingProcessor>();
    const ScopedProcessor registration(processor);

    try {
        ScopedTrackedFunction tracked("op");
        try {
            throw std::runtime_error("circular buffers clash with L1 buffers");
        } catch (const std::exception& e) {
            tracked.abort(e.what());
            throw;
        }
    } catch (const std::runtime_error&) {
    }

    EXPECT_EQ(processor->function_ends.load(), 0);
    ASSERT_EQ(processor->function_aborts.load(), 1);
    EXPECT_EQ(processor->abort_reasons.front(), "circular buffers clash with L1 buffers");
}

// An explicit abort must not be doubled by the destructor.
TEST(ScopedTrackedFunction, ExplicitAbortClosesTheScopeOnlyOnce) {
    auto processor = std::make_shared<CountingProcessor>();
    const ScopedProcessor registration(processor);

    EXPECT_THROW(
        {
            ScopedTrackedFunction tracked("op");
            tracked.abort("already reported");
            throw std::runtime_error("boom");
        },
        std::runtime_error);

    EXPECT_EQ(processor->function_ends.load(), 0);
    EXPECT_EQ(processor->function_aborts.load(), 1);
}

// Nested scopes must each close exactly once, innermost first, so the processor's scope stack
// unwinds to the same depth it started at.
TEST(ScopedTrackedFunction, AbortsEveryOpenScopeWhileUnwinding) {
    auto processor = std::make_shared<CountingProcessor>();
    const ScopedProcessor registration(processor);

    EXPECT_THROW(
        {
            ScopedTrackedFunction outer("outer");
            ScopedTrackedFunction inner("inner");
            throw std::runtime_error("boom");
        },
        std::runtime_error);

    EXPECT_EQ(processor->function_starts.load(), 2);
    EXPECT_EQ(processor->function_ends.load(), 0);
    EXPECT_EQ(processor->function_aborts.load(), 2);
}

// A scope that returns normally while an unrelated exception is already unwinding elsewhere on the
// stack must still be reported as a normal end, which is why the guard compares uncaught-exception
// counts rather than merely testing std::uncaught_exceptions() != 0.
TEST(ScopedTrackedFunction, EndsNormallyWhenAnUnrelatedExceptionIsAlreadyInFlight) {
    auto processor = std::make_shared<CountingProcessor>();
    const ScopedProcessor registration(processor);

    struct Unwinder {
        ~Unwinder() { ScopedTrackedFunction tracked("cleanup op"); }
    };

    EXPECT_THROW(
        {
            Unwinder unwinder;
            throw std::runtime_error("unrelated");
        },
        std::runtime_error);

    EXPECT_EQ(processor->function_starts.load(), 1);
    EXPECT_EQ(processor->function_ends.load(), 1);
    EXPECT_EQ(processor->function_aborts.load(), 0);
}

// A capture pushed inside an already-open scope never saw the start, so it must not be handed the
// end either: that would pop a scope it does not own.
TEST(ScopedTrackedFunction, DoesNotEndAProcessorThatMissedTheStart) {
    auto processor = std::make_shared<CountingProcessor>();
    auto& tracker = GraphTracker::instance();
    tracker.clear();

    {
        ScopedTrackedFunction tracked("op");
        tracker.push_processor(processor);
    }
    tracker.clear();

    EXPECT_EQ(processor->function_starts.load(), 0);
    EXPECT_EQ(processor->function_ends.load(), 0);
    EXPECT_EQ(processor->function_aborts.load(), 0);
}

// The safety net for call sites that are not guarded: the caller that knows nothing can still be
// open (a new top-level operation) asks every processor of its thread to drop what it holds.
TEST(UnwindOpenFunctions, ReachesEveryProcessorWithTheReason) {
    auto processor = std::make_shared<CountingProcessor>();
    const ScopedProcessor registration(processor);

    GraphTracker::instance().track_function_start("op that never ends");
    GraphTracker::instance().unwind_open_functions("closed when 'next op' started");

    EXPECT_EQ(processor->function_ends.load(), 0);
    ASSERT_EQ(processor->unwinds.load(), 1);
    EXPECT_EQ(processor->unwind_reasons.front(), "closed when 'next op' started");
}

// Processors are thread_local, so an unwind on one thread must leave another thread's capture
// alone: its scopes are open for a good reason.
TEST(UnwindOpenFunctions, LeavesAnotherThreadsProcessorAlone) {
    auto processor = std::make_shared<CountingProcessor>();
    const ScopedProcessor registration(processor);

    std::thread([] { GraphTracker::instance().unwind_open_functions("from another thread"); }).join();

    EXPECT_EQ(processor->unwinds.load(), 0);
}

}  // namespace tt::tt_metal::graph_tracking_test
