// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "tt_metal/impl/threading/thread_pool.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/context_types.hpp"
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal::distributed::test {
namespace {

// Stress test for thread pool used by TT-Mesh
TEST(ThreadPoolTest, StressDeviceBound) {
    // Enqueue enough tasks to saturate the thread pool.
    uint64_t NUM_ITERS = 1 << 18;
    uint32_t num_threads = MetalContext::instance().get_cluster().number_of_user_devices();
    auto thread_pool = create_device_bound_thread_pool(
        DEFAULT_CONTEXT_ID, MetalContext::instance(DEFAULT_CONTEXT_ID).get_cluster().number_of_user_devices());
    // Increment this once for each task in the thread pool.
    // Use this to verify that tasks actually executed.
    std::atomic<uint64_t> counter = 0;
    auto incrementer_fn = [&counter]() {
        counter++;
        // Sleep every 10 iterations to slow down the workers - allows
        // the thread pool to get saturated
        if (counter.load() % 10 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }
    };
    // Rely on thread-pool to automatically distribute tasks across workers
    for (std::size_t iter = 0; iter < NUM_ITERS; iter++) {
        thread_pool->enqueue([&incrementer_fn]() mutable { incrementer_fn(); });
    }

    // Explicitly specify the thread each task will go to.
    for (std::size_t iter = 0; iter < NUM_ITERS; iter++) {
        thread_pool->enqueue([&incrementer_fn]() mutable { incrementer_fn(); }, iter % num_threads);
    }

    thread_pool->wait();
    EXPECT_EQ(counter.load(), 2 * NUM_ITERS);
}

// Test that an exception generated in the thread pool is propagated to the main thread
TEST(ThreadPoolTest, Exception) {
    auto thread_pool = create_device_bound_thread_pool(
        DEFAULT_CONTEXT_ID, MetalContext::instance(DEFAULT_CONTEXT_ID).get_cluster().number_of_user_devices());
    auto exception_fn = []() { TT_THROW("Failed"); };
    thread_pool->enqueue(exception_fn);
    EXPECT_THROW(thread_pool->wait(), std::exception);
}

// Two threads can wait on the pool at the same time; both return once the tasks have finished.
TEST(ThreadPoolTest, ConcurrentWait) {
    const uint32_t num_threads = MetalContext::instance().get_cluster().number_of_user_devices();
    auto thread_pool = create_device_bound_thread_pool(DEFAULT_CONTEXT_ID, num_threads);
    std::atomic<uint64_t> counter = 0;
    for (int iter = 0; iter < 200; iter++) {
        // Long enough that both waiters park.
        for (uint32_t i = 0; i < num_threads; i++) {
            thread_pool->enqueue([&counter]() {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                counter++;
            });
        }
        std::thread other([&] { thread_pool->wait(); });
        thread_pool->wait();
        other.join();
        EXPECT_EQ(counter.load(), (iter + 1) * num_threads);
    }
}

std::shared_ptr<ThreadPool> create_pool_per_device() {
    return create_device_bound_thread_pool(
        DEFAULT_CONTEXT_ID, MetalContext::instance(DEFAULT_CONTEXT_ID).get_cluster().number_of_user_devices());
}

std::vector<uint32_t> one_call_per_device(uint32_t calls_per_device) {
    uint32_t num_devices = MetalContext::instance(DEFAULT_CONTEXT_ID).get_cluster().number_of_user_devices();
    std::vector<uint32_t> device_ids;
    for (uint32_t round = 0; round < calls_per_device; round++) {
        for (uint32_t device = 0; device < num_devices; device++) {
            device_ids.push_back(device);
        }
    }
    return device_ids;
}

// Every call runs exactly once, including when several calls map to the same device, and across many
// back-to-back fan-outs whose tasks may still be in flight on the workers when the next one starts.
TEST(ThreadPoolTest, ParallelForRunsEachCallOnce) {
    auto thread_pool = create_pool_per_device();
    for (uint32_t calls_per_device : {1u, 3u}) {
        auto device_ids = one_call_per_device(calls_per_device);
        std::vector<std::atomic<uint32_t>> runs(device_ids.size());
        constexpr uint32_t NUM_FAN_OUTS = 10000;
        for (uint32_t iter = 0; iter < NUM_FAN_OUTS; iter++) {
            thread_pool->parallel_for(device_ids, [&runs](size_t call) { runs[call]++; });
        }
        for (const auto& count : runs) {
            EXPECT_EQ(count.load(), NUM_FAN_OUTS);
        }
    }
}

// A call that takes long enough for the workers to wake up runs on them, not only on the caller.
TEST(ThreadPoolTest, ParallelForUsesWorkers) {
    // Two workers on any host: the caller can run all calls of a single worker before it wakes.
    auto thread_pool = create_device_bound_thread_pool(DEFAULT_CONTEXT_ID, 2);
    std::vector<uint32_t> device_ids = {0, 1};
    std::vector<std::thread::id> ran_on(device_ids.size());
    thread_pool->parallel_for(device_ids, [&ran_on](size_t call) {
        ran_on[call] = std::this_thread::get_id();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    });
    auto on_caller = std::count(ran_on.begin(), ran_on.end(), std::this_thread::get_id());
    EXPECT_LT(on_caller, static_cast<int64_t>(ran_on.size()));
}

// Calls for the same device never overlap and run in index order.
TEST(ThreadPoolTest, ParallelForSameDeviceInOrder) {
    auto thread_pool = create_pool_per_device();
    auto device_ids = one_call_per_device(4);
    uint32_t num_devices = device_ids.size() / 4;
    std::vector<std::atomic<bool>> busy(num_devices);
    std::vector<int64_t> last_call(num_devices);
    std::atomic<uint32_t> violations = 0;
    for (uint32_t iter = 0; iter < 1000; iter++) {
        std::fill(last_call.begin(), last_call.end(), -1);
        thread_pool->parallel_for(device_ids, [&](size_t call) {
            uint32_t device = device_ids[call];
            if (busy[device].exchange(true) || last_call[device] >= static_cast<int64_t>(call)) {
                violations++;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            last_call[device] = static_cast<int64_t>(call);
            busy[device] = false;
        });
    }
    EXPECT_EQ(violations.load(), 0u);
}

TEST(ThreadPoolTest, ParallelForException) {
    auto thread_pool = create_pool_per_device();
    auto device_ids = one_call_per_device(2);
    std::atomic<uint32_t> runs = 0;
    EXPECT_THROW(
        thread_pool->parallel_for(
            device_ids,
            [&runs](size_t call) {
                runs++;
                if (call == 1) {
                    TT_THROW("Failed");
                }
            }),
        std::exception);
    // The other calls still ran, and the pool is usable afterwards.
    EXPECT_EQ(runs.load(), device_ids.size());
    thread_pool->parallel_for(device_ids, [](size_t) {});
}

// An unknown device id throws before any call runs, and the pool stays usable.
TEST(ThreadPoolTest, ParallelForUnknownDevice) {
    auto thread_pool = create_pool_per_device();
    std::atomic<uint32_t> runs = 0;
    std::vector<uint32_t> unknown = {0, std::numeric_limits<uint32_t>::max()};
    EXPECT_THROW(thread_pool->parallel_for(unknown, [&runs](size_t) { runs++; }), std::out_of_range);
    EXPECT_EQ(runs.load(), 0u);
    auto device_ids = one_call_per_device(1);
    thread_pool->parallel_for(device_ids, [&runs](size_t) { runs++; });
    EXPECT_EQ(runs.load(), device_ids.size());
}

TEST(ThreadPoolTest, ParallelForEmpty) {
    auto thread_pool = create_pool_per_device();
    thread_pool->parallel_for({}, [](size_t) { FAIL(); });
}

// parallel_for waits only for its own calls, and tasks from enqueue() are still joined by wait().
TEST(ThreadPoolTest, ParallelForWithEnqueue) {
    auto thread_pool = create_pool_per_device();
    auto device_ids = one_call_per_device(1);
    std::atomic<uint32_t> enqueued_runs = 0, parallel_runs = 0;
    for (uint32_t iter = 0; iter < 1000; iter++) {
        thread_pool->enqueue([&enqueued_runs] { enqueued_runs++; }, iter % device_ids.size());
        thread_pool->parallel_for(device_ids, [&parallel_runs](size_t) { parallel_runs++; });
        EXPECT_EQ(parallel_runs.load(), (iter + 1) * device_ids.size());
    }
    thread_pool->wait();
    EXPECT_EQ(enqueued_runs.load(), 1000u);
}

// Destroying the pool right after a fan-out, while workers may still hold the job, is safe.
TEST(ThreadPoolTest, ParallelForThenDestroy) {
    auto device_ids = one_call_per_device(1);
    for (uint32_t iter = 0; iter < 50; iter++) {
        auto thread_pool = create_pool_per_device();
        std::atomic<uint32_t> runs = 0;
        thread_pool->parallel_for(device_ids, [&runs](size_t) { runs++; });
        EXPECT_EQ(runs.load(), device_ids.size());
    }
}

TEST(ThreadPoolTest, ParallelForPassThrough) {
    auto thread_pool = create_passthrough_thread_pool(DEFAULT_CONTEXT_ID);
    std::vector<uint32_t> device_ids = {0, 0, 0};
    std::vector<size_t> calls;
    thread_pool->parallel_for(device_ids, [&calls](size_t call) { calls.push_back(call); });
    EXPECT_EQ(calls, (std::vector<size_t>{0, 1, 2}));
    // Like the other pools, runs every call before rethrowing the first exception.
    calls.clear();
    EXPECT_THROW(
        thread_pool->parallel_for(
            device_ids,
            [&calls](size_t call) {
                calls.push_back(call);
                TT_THROW("Failed");
            }),
        std::exception);
    EXPECT_EQ(calls, (std::vector<size_t>{0, 1, 2}));
}

}  // namespace

}  // namespace tt::tt_metal::distributed::test
