// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only tests for the impl-side buffer allocation observer registry, which is how SHM
// memory tracking learns about buffer allocations.
//
// Observers are process-wide and registered once during device init, so the thing that can go
// wrong is registering one twice: every buffer event is then recorded twice and every figure the
// tracker publishes reads double.

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "impl/memory_tracking/buffer_allocation_observer.hpp"

namespace tt::tt_metal::buffer_allocation_observer_test {

namespace {

// Observers are never unregistered, so this one outlives its test and keeps being notified for
// the rest of the binary. It counts and does nothing else.
class CountingObserver : public BufferAllocationObserver {
public:
    inline static std::atomic<int> allocate_calls{0};
    inline static std::atomic<int> instances{0};

    CountingObserver() { instances.fetch_add(1); }

    void track_allocate(const Buffer* /*buffer*/) override { allocate_calls.fetch_add(1); }
    void track_deallocate(Buffer* /*buffer*/) override {}
};

}  // namespace

// Registration has to be atomic, not check-then-act. Looking first and inserting after lets two
// devices initializing concurrently both observe absence and both register -- reachable from
// ordinary multi-device startup, where Device::initialize() runs in parallel.
//
// On what this proves. Against a check-then-act implementation it is decisive whenever the gap
// between the lookup and the insertion is non-trivial: ablated with 200us in that gap, all 16
// threads register and one allocation is reported 16 times. With nothing in the gap the window is
// sub-microsecond and one thread usually finishes before the others have finished looking, so the
// interleaving often does not occur at all. The assertions below are therefore a firm guard on the
// contract -- one registration, one construction, one notification, later calls inert -- and only
// an opportunistic one on the interleaving. The single lock is what makes the race impossible;
// a test cannot prove absence, and correct code passes this deterministically.
TEST(BufferAllocationObserver, CPU_ConcurrentRegistrationRegistersExactlyOne) {
    constexpr int kThreads = 16;

    CountingObserver::allocate_calls.store(0);
    CountingObserver::instances.store(0);

    std::atomic<int> registered{0};
    std::atomic<int> arrived{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; i++) {
        threads.emplace_back([&] {
            // Arrival barrier, spun tight on purpose. A flag set after the spawn loop only proves
            // the threads were created, and waiting on it with yield() is actively harmful: the
            // last thread to arrive never waits, so it runs alone while every other thread sits
            // descheduled -- which serialises the very calls that need to overlap. Bounded with a
            // yield fallback so this still terminates with fewer cores than threads.
            arrived.fetch_add(1, std::memory_order_acq_rel);
            for (int spins = 0; arrived.load(std::memory_order_acquire) < kThreads; spins++) {
                if (spins > 1'000'000) {
                    std::this_thread::yield();
                }
            }
            if (register_buffer_allocation_observer_once(
                    typeid(CountingObserver), [] { return std::make_shared<CountingObserver>(); })) {
                registered.fetch_add(1);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(registered.load(), 1) << "expected exactly one of " << kThreads
                                    << " concurrent callers to report that it registered the observer";
    EXPECT_EQ(CountingObserver::instances.load(), 1)
        << "the factory ran more than once, so an observer was constructed for a registration that "
           "should have been skipped";

    // The symptom the race actually produces. notify_buffer_allocated() passes the pointer through
    // without dereferencing it and this observer ignores it, so no real buffer is needed.
    notify_buffer_allocated(nullptr);
    EXPECT_EQ(CountingObserver::allocate_calls.load(), 1)
        << "one buffer allocation was reported " << CountingObserver::allocate_calls.load()
        << " times; a duplicate observer doubles every figure the tracker publishes";

    // A later call must still be inert, which is what makes repeated device init safe.
    EXPECT_FALSE(register_buffer_allocation_observer_once(typeid(CountingObserver), [] {
        return std::make_shared<CountingObserver>();
    })) << "re-registering an already-registered type reported a fresh registration";
}

}  // namespace tt::tt_metal::buffer_allocation_observer_test
