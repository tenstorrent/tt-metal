// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/thread_pool.hpp>

namespace tt::tt_metal::distributed {
namespace {

// Stress test for thread pool used by TT-Mesh
TEST(ThreadPoolTest, Stress) {
    // Enqueue enough tasks to saturate the thread pool.
    uint64_t NUM_ITERS = 1 << 18;
    ThreadPool thread_pool = ThreadPool();
    // Increment this once for each task in the thread pool.
    // Use this to verify that tasks actually executed.
    std::atomic<uint64_t> counter = 0;
    auto incrementer_fn = std::make_shared<std::function<void()>>([&counter]() {
        counter++;
        // Sleep every 10 iterations to slow down the workers - allows
        // the thread pool to get saturated
        if (counter.load() % 10 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });
    for (std::size_t iter = 0; iter < NUM_ITERS; iter++) {
        thread_pool.enqueue([incrementer_fn]() mutable { (*incrementer_fn)(); });
    }

    thread_pool.barrier();
    EXPECT_EQ(counter.load(), NUM_ITERS);
}

}  // namespace
}  // namespace tt::tt_metal::distributed
