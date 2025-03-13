// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>
#include "host_utils.hpp"

namespace tt::tt_metal::tools::mem_bench {

// Execute work_fn on num_threads threads and also do intermediate_fn on the side.
// Returns time taken in seconds for all work_fn to complete. Time is calculated by latest thread end - earliest thread
// start.
template <typename F, typename IntermediateF, typename... Args>
double execute_work_synced_start(int num_threads, F&& work_fn, IntermediateF&& intermediate_fn, Args&&... args) {
    std::mutex m;
    int threads_ready{0};
    std::condition_variable go_cv;         // Signal to all threads to go
    auto total_threads = num_threads + 1;  // Including intermediate
    std::vector<double> thread_start_times(num_threads);
    std::vector<double> thread_end_times(num_threads);
    std::vector<std::thread> threads(total_threads);

    for (int i = 0; i < num_threads; ++i) {
        threads[i] = std::thread([i,
                                  &m,
                                  &go_cv,
                                  &threads_ready,
                                  &thread_start_times,
                                  &thread_end_times,
                                  total_threads,
                                  work_fn = std::forward<F>(work_fn),
                                  ... args = std::forward<Args>(args)]() mutable {
            {
                std::unique_lock lk{m};
                threads_ready++;
                if (threads_ready == total_threads) {
                    go_cv.notify_all();
                }
                go_cv.wait(lk, [&] { return threads_ready == total_threads; });
            }

            thread_start_times[i] = get_current_time_seconds();
            work_fn(i, std::forward<Args>(args)...);
            thread_end_times[i] = get_current_time_seconds();
        });
    }

    threads[num_threads] = std::thread([&]() mutable {
        std::unique_lock lk{m};
        threads_ready++;
        if (threads_ready == total_threads) {
            go_cv.notify_all();
        }
        go_cv.wait(lk, [&] { return threads_ready == total_threads; });

        intermediate_fn();
    });

    for (auto& thread : threads) {
        thread.join();
    }

    // Calculate work time based on earliest start and latest end
    double earliest_start = *std::min_element(thread_start_times.begin(), thread_start_times.end());
    double latest_end = *std::max_element(thread_end_times.begin(), thread_end_times.end());

    return latest_end - earliest_start;
}

};  // namespace tt::tt_metal::tools::mem_bench
