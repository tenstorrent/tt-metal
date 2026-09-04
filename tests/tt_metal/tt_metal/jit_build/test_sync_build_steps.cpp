// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression tests for tt::tt_metal::sync_build_steps (tt_metal/jit_build/build.cpp).
//
// Background
// ----------
// ProgramImpl::compile() launches each kernel's build work as an async task
// (launch_build_step() -> detail::async()), collects the futures, and waits on
// them with sync_build_steps(). Each task's lambda captures compile()'s locals
// BY REFERENCE -- notably `IDevice* device`, dereferenced later as
// device->build_id(). A naive wait would be:
//
//     for (auto& event : events) { event.get(); }
//
// If one build step throws (a kernel fails to compile), the FIRST event.get()
// that observes the exception rethrows and the loop exits, ABANDONING the
// remaining in-flight async tasks. Those tasks still reference compile()'s stack
// frame, which is now unwinding -> use-after-free -> SIGSEGV on a worker thread
// (device->build_id() on a dangling pointer), masking the real compile error.
//
// sync_build_steps guards against this by draining ALL events -- catching each,
// remembering the first exception -- and only rethrowing after every task has
// finished, so no task outlives the compile() frame (aa85a788b33).
//
// These tests pin that invariant with no device and no timing dependence on the
// tasks themselves: a "throwing" step is observed first while sibling steps are
// provably still in flight (parked on a gate the test controls), and we check
// whether sync_build_steps returns before those siblings finish.
//   - naive wait -> returns on the first throw, siblings abandoned          -> FAIL
//   - draining    -> cannot return until the siblings finish (aa85a788b33)  -> PASS

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <functional>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

#include "common/executor.hpp"
#include "jit_build/build.hpp"

namespace tt::tt_metal {
namespace {

// How a task functor is turned into an event appended to `events`. Two variants
// exercise the property: (1) via std::async (guaranteed threads, zero executor
// dependence -- the bulletproof guard) and (2) via the real production helper
// launch_build_step()/detail::async() (the exact path the compile() bug rides).
using Launcher = std::function<void(std::function<void()>, std::vector<std::shared_future<void>>&)>;

void via_std_async(std::function<void()> fn, std::vector<std::shared_future<void>>& events) {
    events.emplace_back(std::async(std::launch::async, std::move(fn)).share());
}

void via_launch_build_step(const std::function<void()>& fn, std::vector<std::shared_future<void>>& events) {
    launch_build_step(fn, events);
}

// Core property: sync_build_steps must not return (throw) while any launched
// build step is still in flight. `uses_executor` guards the launch_build_step
// variant on executors too small to keep every sibling concurrently in flight
// (detail::async runs inline once the executor is saturated, which would
// deadlock a gated sibling at launch rather than test anything).
void expect_drains_all_in_flight_steps(const Launcher& launch, bool uses_executor) {
    constexpr int kSiblings = 3;  // in-flight steps that must not be abandoned

    if (uses_executor && detail::GetExecutor().num_workers() < static_cast<size_t>(kSiblings) + 2) {
        GTEST_SKIP() << "executor has too few workers to keep sibling build steps concurrently in flight";
    }

    std::atomic<int> parked{0};      // siblings that reached the (closed) gate
    std::atomic<int> completed{0};   // siblings that ran to completion
    std::atomic<bool> threw{false};  // the throwing step has thrown
    std::promise<void> gate_promise;
    std::shared_future<void> gate = gate_promise.get_future().share();

    std::vector<std::shared_future<void>> events;

    // events[0]: throws as soon as it runs. Placed FIRST so a naive loop's very
    // first event.get() observes it and exits, abandoning everything after it.
    launch(
        [&threw] {
            threw.store(true);
            throw std::runtime_error("build failed (sync_build_steps regression probe)");
        },
        events);

    // events[1..kSiblings]: block on the gate so they are provably still in flight
    // when the throwing step is observed. They only touch `completed` post-gate.
    for (int i = 0; i < kSiblings; ++i) {
        launch(
            [&parked, &completed, gate] {
                parked.fetch_add(1);
                gate.wait();
                completed.fetch_add(1);
            },
            events);
    }

    // Wait until the throwing step has thrown AND every sibling is parked at the
    // closed gate. At this instant completed == 0 is guaranteed: no sibling can
    // advance past gate.wait().
    while (!threw.load() || parked.load() != kSiblings) {
        std::this_thread::yield();
    }
    ASSERT_EQ(completed.load(), 0);

    // Run sync_build_steps on a helper thread so THIS thread stays free to observe
    // whether it returns early and to open the gate. (Calling it inline would let
    // the draining implementation block here on the parked siblings, with no way left
    // to open the gate -> deadlock.)
    std::atomic<bool> sync_returned{false};
    std::exception_ptr sync_exc;
    std::thread sync_thread([&] {
        try {
            sync_build_steps(events);
        } catch (...) {
            sync_exc = std::current_exception();
        }
        sync_returned.store(true);
    });

    // Discriminator: with the gate still CLOSED, does sync_build_steps return?
    //   naive wait -> yes: rethrows on events[0], abandons the parked siblings (completed == 0)
    //   draining   -> no:  blocks on events[1].get() until we open the gate
    // We cannot prove "blocks forever" without a bounded wait; the window is far
    // longer than the naive near-instant return and irrelevant to the draining path
    // (which genuinely cannot finish until the gate opens).
    constexpr auto kObserveWindow = std::chrono::milliseconds(2000);
    const auto deadline = std::chrono::steady_clock::now() + kObserveWindow;
    bool returned_before_gate = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (sync_returned.load()) {
            returned_before_gate = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    EXPECT_FALSE(returned_before_gate)
        << "sync_build_steps returned before draining all build steps; "
        << (kSiblings - completed.load()) << " sibling step(s) were still in flight (abandoned). "
        << "This is the abandon-on-throw use-after-free that draining all steps before rethrowing prevents.";

    // Release the parked siblings so the drain can finish and the thread joins.
    gate_promise.set_value();
    sync_thread.join();

    // A naive early return would not have waited for the siblings; make sure every step
    // has actually finished before asserting on `completed`. wait() (unlike get()) does
    // not rethrow events[0]'s stored build failure -- already accounted for below.
    for (auto& event : events) {
        event.wait();
    }

    EXPECT_TRUE(static_cast<bool>(sync_exc)) << "sync_build_steps must rethrow the build failure";
    EXPECT_EQ(completed.load(), kSiblings) << "every launched build step must run to completion (none abandoned)";
}

}  // namespace

// The regression guard, via std::async -- guaranteed threads, no executor dependence.
TEST(SyncBuildStepsTest, DrainsAllStepsBeforeRethrowStdAsync) {
    expect_drains_all_in_flight_steps(via_std_async, /*uses_executor=*/false);
}

// The same guard via the real production helper launch_build_step()/detail::async()
// -- the exact path ProgramImpl::compile() uses when the crash was hit.
TEST(SyncBuildStepsTest, DrainsAllStepsBeforeRethrowLaunchBuildStep) {
    expect_drains_all_in_flight_steps(via_launch_build_step, /*uses_executor=*/true);
}

// Basic contract: a failing step surfaces as a rethrown exception.
TEST(SyncBuildStepsTest, RethrowsBuildFailure) {
    std::vector<std::shared_future<void>> events;
    events.emplace_back(std::async(std::launch::async, [] {
                            throw std::runtime_error("build failed");
                        }).share());
    EXPECT_THROW(sync_build_steps(events), std::exception);
}

// Basic contract: when all steps succeed, sync_build_steps returns cleanly and
// every step ran.
TEST(SyncBuildStepsTest, NoThrowWhenAllStepsSucceed) {
    constexpr int kSteps = 4;
    std::atomic<int> counter{0};
    std::vector<std::shared_future<void>> events;
    events.reserve(kSteps);
    for (int i = 0; i < kSteps; ++i) {
        events.emplace_back(std::async(std::launch::async, [&counter] { counter.fetch_add(1); }).share());
    }
    EXPECT_NO_THROW(sync_build_steps(events));
    EXPECT_EQ(counter.load(), kSteps);
}

}  // namespace tt::tt_metal
