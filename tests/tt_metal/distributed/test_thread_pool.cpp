// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/common/thread_pool.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

// Stress test for thread pool used by TT-Mesh
TEST(ThreadPoolTest, StressDeviceBound) {
    // Enqueue enough tasks to saturate the thread pool.
    uint64_t NUM_ITERS = 1 << 18;
    uint32_t num_threads = MetalContext::instance().get_cluster().number_of_user_devices();
    auto thread_pool = create_device_bound_thread_pool(MetalContext::instance().get_cluster().number_of_user_devices());
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
    uint32_t num_threads = MetalContext::instance().get_cluster().number_of_user_devices();
    auto thread_pool = create_device_bound_thread_pool(MetalContext::instance().get_cluster().number_of_user_devices());
    auto exception_fn = []() { TT_THROW("Failed"); };
    thread_pool->enqueue(exception_fn);
    EXPECT_THROW(thread_pool->wait(), std::exception);
}

}  // namespace

}  // namespace tt::tt_metal::distributed::test
