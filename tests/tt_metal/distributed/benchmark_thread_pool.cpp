// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>
#include "tt_metal/common/env_lib.hpp"
#include "tt_metal/common/thread_pool.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

template <typename ThreadPoolCreator>
static void BM_ThreadPool(benchmark::State& state, ThreadPoolCreator create_thread_pool) {
    uint32_t num_threads = tt::parse_env("TT_METAL_NUM_BENCHMARK_THREADS", 8);

    auto thread_pool = create_thread_pool(num_threads);

    for ([[maybe_unused]] auto _ : state) {
        uint64_t NUM_ITERS = state.range(0);

        auto work = []() { std::this_thread::sleep_for(std::chrono::microseconds(20)); };

        std::chrono::high_resolution_clock::time_point enqueue_start, enqueue_end;
        std::chrono::high_resolution_clock::time_point wait_start, wait_end;

        enqueue_start = std::chrono::high_resolution_clock::now();
        for (std::size_t iter = 0; iter < NUM_ITERS; iter++) {
            thread_pool->enqueue([&work]() mutable { work(); }, iter % num_threads);
        }
        enqueue_end = std::chrono::high_resolution_clock::now();

        wait_start = std::chrono::high_resolution_clock::now();
        thread_pool->wait();
        wait_end = std::chrono::high_resolution_clock::now();

        auto enqueue_time = std::chrono::duration_cast<std::chrono::microseconds>(enqueue_end - enqueue_start).count();
        auto wait_time = std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count();

        state.counters["enqueue_time_us"] = enqueue_time;
        state.counters["wait_time_us"] = wait_time;
        state.counters["enqueue_time_per_task_us"] = enqueue_time / static_cast<double>(NUM_ITERS);
        state.counters["wait_time_per_task_us"] = wait_time / static_cast<double>(NUM_ITERS);
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
    state.SetComplexityN(state.range(0));
}

static void BM_DeviceBoundThreadPool(benchmark::State& state) {
    BM_ThreadPool(
        state, [](uint32_t num_threads) { return tt::tt_metal::create_device_bound_thread_pool(num_threads); });
}

BENCHMARK(BM_DeviceBoundThreadPool)->RangeMultiplier(2)->Range(1, 1 << 18)->Complexity(benchmark::oN)->UseRealTime();
