// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fan-out latency of the host thread pools: per-task hand-off cost on the caller, submit->start latency
// after an idle gap, join cost, and CPU spent by the workers.
//
// Each iteration keeps the caller busy for `gap_us` (host work between ops), then submits
// `tasks_per_worker` tasks to each of `workers` workers, each busy for `work_ns`, and joins.
// Only the submit and join are timed. The pass-through pool runs the same tasks inline and is the
// serial reference.
//
// The default grid is sized for a regression run; TT_POOL_BENCH_FULL=1 registers the full grid.
// TT_POOL_BENCH_WORKERS sets the device-bound pool size (default 32) and TT_POOL_BENCH_CALLER_CPU pins
// the calling thread.

#include <benchmark/benchmark.h>

#include <sched.h>
#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "tt_metal/common/env_lib.hpp"
#include "tt_metal/impl/threading/thread_pool.hpp"
#include "impl/context/context_types.hpp"

namespace {

using tt::tt_metal::ThreadPool;

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void spin_for_ns(int64_t ns) {
    if (ns <= 0) {
        return;
    }
    const int64_t end = now_ns() + ns;
    while (now_ns() < end) {
    }
}

struct Usage {
    double cpu_s = 0;
    int64_t voluntary_switches = 0;
};

Usage usage(int who) {
    rusage ru{};
    getrusage(who, &ru);
    return {
        ru.ru_utime.tv_sec + (ru.ru_utime.tv_usec * 1e-6) + ru.ru_stime.tv_sec + (ru.ru_stime.tv_usec * 1e-6),
        ru.ru_nvcsw};
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) {
        return 0.0;
    }
    auto idx = std::min(v.size() - 1, static_cast<size_t>(std::lround(p * (v.size() - 1))));
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}

uint32_t max_workers() { return tt::parse_env("TT_POOL_BENCH_WORKERS", 32); }

void pin_caller() {
    static const bool pinned = [] {
        int cpu = tt::parse_env("TT_POOL_BENCH_CALLER_CPU", -1);
        if (cpu >= 0) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(cpu, &cpuset);
            sched_setaffinity(0, sizeof(cpuset), &cpuset);
        }
        return true;
    }();
    (void)pinned;
}

// Pools are created once per process: core assignment advances a process-wide counter on every pool
// creation, so recreating them per benchmark would move workers onto different cores.
struct DeviceBoundPool {
    static constexpr const char* name = "DeviceBound";
    static constexpr bool runs_on_caller = false;
    static ThreadPool& get() {
        static auto pool =
            tt::tt_metal::create_device_bound_thread_pool(tt::tt_metal::DEFAULT_CONTEXT_ID, max_workers());
        return *pool;
    }
};

struct PassThroughPool {
    static constexpr const char* name = "PassThrough";
    static constexpr bool runs_on_caller = true;
    static ThreadPool& get() {
        static auto pool = tt::tt_metal::create_passthrough_thread_pool(tt::tt_metal::DEFAULT_CONTEXT_ID);
        return *pool;
    }
};

struct alignas(64) TaskTimes {
    int64_t start = 0;
    int64_t end = 0;
};

// Submits one fan-out. Batched submit APIs get their own variant of this function.
template <typename MakeTask>
void submit_fan_out(ThreadPool& pool, uint32_t workers, uint32_t tasks_per_worker, const MakeTask& make_task) {
    for (uint32_t t = 0; t < tasks_per_worker; t++) {
        for (uint32_t w = 0; w < workers; w++) {
            pool.enqueue(make_task((t * workers) + w), w);
        }
    }
}

template <typename Pool, size_t PadBytes>
void BM_FanOut(benchmark::State& state) {
    const auto workers = static_cast<uint32_t>(state.range(0));
    const auto tasks_per_worker = static_cast<uint32_t>(state.range(1));
    const int64_t work_ns = state.range(2);
    const int64_t gap_ns = state.range(3) * 1000;
    if (!Pool::runs_on_caller && workers > max_workers()) {
        state.SkipWithError("more workers requested than TT_POOL_BENCH_WORKERS");
        return;
    }
    pin_caller();
    auto& pool = Pool::get();
    const uint32_t num_tasks = workers * tasks_per_worker;
    std::vector<TaskTimes> times(num_tasks);
    std::vector<double> wall_us, submit_us, first_start_us, last_start_us, join_us;
    // Padding makes the capture exceed the callable's small buffer, as the dispatch captures do.
    std::array<char, PadBytes> pad{};
    auto make_task = [&times, work_ns, &pad](uint32_t i) {
        return [slot = &times[i], work_ns, pad]() {
            slot->start = now_ns();
            benchmark::DoNotOptimize(pad);
            spin_for_ns(work_ns);
            slot->end = now_ns();
        };
    };

    const Usage self_before = usage(RUSAGE_SELF);
    const Usage caller_before = usage(RUSAGE_THREAD);
    const int64_t run_begin = now_ns();
    for ([[maybe_unused]] auto _ : state) {
        spin_for_ns(gap_ns);
        const int64_t t0 = now_ns();
        submit_fan_out(pool, workers, tasks_per_worker, make_task);
        const int64_t t1 = now_ns();
        pool.wait();
        const int64_t t2 = now_ns();

        int64_t first_start = INT64_MAX, last_start = 0, last_end = 0;
        for (const auto& t : times) {
            first_start = std::min(first_start, t.start);
            last_start = std::max(last_start, t.start);
            last_end = std::max(last_end, t.end);
        }
        state.SetIterationTime((t2 - t0) * 1e-9);
        wall_us.push_back((t2 - t0) * 1e-3);
        submit_us.push_back((t1 - t0) * 1e-3 / num_tasks);
        first_start_us.push_back((first_start - t0) * 1e-3);
        last_start_us.push_back((last_start - t0) * 1e-3);
        join_us.push_back((t2 - last_end) * 1e-3);
    }
    const double run_s = (now_ns() - run_begin) * 1e-9;
    const Usage self_after = usage(RUSAGE_SELF);
    const Usage caller_after = usage(RUSAGE_THREAD);
    const auto iters = static_cast<double>(state.iterations());

    state.counters["wall_p50_us"] = percentile(wall_us, 0.5);
    state.counters["wall_p99_us"] = percentile(wall_us, 0.99);
    // For the pass-through pool the submit time includes running the tasks.
    state.counters["submit_per_task_us"] = percentile(submit_us, 0.5);
    state.counters["first_start_p50_us"] = percentile(first_start_us, 0.5);
    state.counters["first_start_p99_us"] = percentile(first_start_us, 0.99);
    state.counters["last_start_p50_us"] = percentile(last_start_us, 0.5);
    state.counters["last_start_p99_us"] = percentile(last_start_us, 0.99);
    state.counters["join_p50_us"] = percentile(join_us, 0.5);
    state.counters["caller_parks"] = (caller_after.voluntary_switches - caller_before.voluntary_switches) / iters;
    if (!Pool::runs_on_caller) {
        const double worker_cpu_s = (self_after.cpu_s - self_before.cpu_s) - (caller_after.cpu_s - caller_before.cpu_s);
        const double task_cpu_s = iters * num_tasks * work_ns * 1e-9;
        // Cores kept busy by the workers beyond the task bodies (spinning, wake-ups, queue operations),
        // averaged over the run including the idle gaps.
        state.counters["worker_overhead_cores"] = (worker_cpu_s - task_cpu_s) / run_s;
        state.counters["worker_parks"] = ((self_after.voluntary_switches - self_before.voluntary_switches) -
                                          (caller_after.voluntary_switches - caller_before.voluntary_switches)) /
                                         iters;
    }
}

// {workers, tasks_per_worker, work_ns, gap_us}
using Grid = std::vector<std::vector<int64_t>>;

template <typename Pool, size_t PadBytes>
void register_fan_out(const Grid& grid, int64_t iterations) {
    const std::string name = std::string("BM_ThreadPoolFanOut/") + Pool::name + "/pad:" + std::to_string(PadBytes);
    benchmark::RegisterBenchmark(name, BM_FanOut<Pool, PadBytes>)
        ->ArgNames({"workers", "tasks_per_worker", "work_ns", "gap_us"})
        ->ArgsProduct(grid)
        ->Iterations(iterations)
        ->UseManualTime()
        ->Unit(benchmark::kMicrosecond);
}

template <typename Pool>
void register_pool() {
    if (tt::parse_env("TT_POOL_BENCH_FULL", false)) {
        register_fan_out<Pool, 0>({{1, 8, 32}, {1, 4}, {0, 600, 1300, 2000, 10000}, {0, 5, 63, 144, 500}}, 2000);
        register_fan_out<Pool, 64>({{32}, {1}, {600}, {0, 63}}, 2000);
        return;
    }
    // Measured on GLM-5.2 and Kimi K2.7 chunked prefill on a Blackhole Galaxy (#57586): a per-device command write
    // takes 0.6 us at the median and 1.3 us at p90, and the host spends 63 us between enqueues at the median and
    // 144 us at p90.
    register_fan_out<Pool, 0>({{8, 32}, {1}, {600, 1300}, {0, 63, 144}}, 5000);
    // Chunked fan-out.
    register_fan_out<Pool, 0>({{32}, {4}, {600}, {0}}, 5000);
    // Capture larger than the callable's small buffer.
    register_fan_out<Pool, 64>({{32}, {1}, {600}, {0}}, 5000);
    // Parked workers: the dispatch pool's fan-outs in those models are ~10 ms apart.
    register_fan_out<Pool, 0>({{32}, {1}, {600}, {10000}}, 300);
}

const bool registered = [] {
    register_pool<DeviceBoundPool>();
    register_pool<PassThroughPool>();
    return true;
}();

}  // namespace

// Bulk enqueue throughput, kept from the earlier benchmark: N tasks of a 20 us sleep round-robin over the workers,
// then one wait(). Each run makes its own pool, so it is registered after the fan-out benchmarks above, whose pools
// would otherwise move to other cores.
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
    BM_ThreadPool(state, [](uint32_t num_threads) {
        return tt::tt_metal::create_device_bound_thread_pool(tt::tt_metal::DEFAULT_CONTEXT_ID, num_threads);
    });
}

BENCHMARK(BM_DeviceBoundThreadPool)->RangeMultiplier(2)->Range(1, 1 << 18)->Complexity(benchmark::oN)->UseRealTime();
