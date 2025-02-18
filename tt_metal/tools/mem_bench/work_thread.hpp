// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>
#include <chrono>
#include <vector>
#include <atomic>
#include <thread>
#include "host_utils.hpp"

namespace tt::tt_metal::tools::mem_bench {

// Execute work_fn on num_threads threads and do intermediate_fn on the calling thread
// Returns time taken in seconds for all work_fn to complete. Time is calculated by latest thread end - earliest thread
// start.
template <typename F, typename IntermediateF, typename... Args>
double execute_work_synced_start(int num_threads, F&& work_fn, IntermediateF&& intermediate_fn, Args&&... args) {
    std::vector<double> thread_start_times(num_threads);
    std::vector<double> thread_end_times(num_threads);
    std::vector<std::thread> threads(num_threads);
    std::atomic<bool> start_flag{false};
    std::atomic<int> threads_ready{0};

    for (int i = 0; i < num_threads; ++i) {
        threads[i] = std::thread([i,
                                  &threads_ready,
                                  &start_flag,
                                  &thread_start_times,
                                  &thread_end_times,
                                  work_fn = std::forward<F>(work_fn),
                                  ... args = std::forward<Args>(args)]() mutable {
            threads_ready++;

            // Wait for start signal
            while (!start_flag.load()) {
                std::this_thread::yield();
            }

            thread_start_times[i] = get_current_time_seconds();
            work_fn(i, std::forward<Args>(args)...);
            thread_end_times[i] = get_current_time_seconds();
        });
    }

    while (threads_ready.load() < num_threads) {
        std::this_thread::yield();
    }

    // Start threads and do other work at once
    start_flag.store(true);
    intermediate_fn();

    for (auto& thread : threads) {
        thread.join();
    }

    // Calculate work time based on earliest start and latest end
    double earliest_start = *std::min_element(thread_start_times.begin(), thread_start_times.end());
    double latest_end = *std::max_element(thread_end_times.begin(), thread_end_times.end());

    return latest_end - earliest_start;
}

};  // namespace tt::tt_metal::tools::mem_bench
