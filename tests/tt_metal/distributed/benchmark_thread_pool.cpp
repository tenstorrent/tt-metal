// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

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

// Fan-out latency of the host thread pools. Each iteration keeps the caller busy for gap_us, then times only the
// submit and the join, through enqueue() and wait() (BM_ThreadPoolFanOut) or one parallel_for()
// (BM_ThreadPoolParallelFor). Supported env variables:
//  * TT_POOL_BENCH_FULL=1 runs the full grid (false),
//  * TT_POOL_BENCH_WORKERS sets the pool size (32),
//  * TT_POOL_BENCH_CALLER_CPU pins the caller thread (unset).
namespace fan_out {

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

struct DeviceBoundPool {
    static constexpr const char* name = "DeviceBound";
    static constexpr bool runs_on_caller = false;
    static ThreadPool& get() {
        // Created once: every pool creation moves later pools to other cores.
        static auto pool =
            tt::tt_metal::create_device_bound_thread_pool(tt::tt_metal::DEFAULT_CONTEXT_ID, max_workers());
        return *pool;
    }
};

// PassThrough: serial reference.
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
    bool on_caller = false;
};

// Task i goes to worker i % workers.
std::vector<uint32_t> worker_of_task(uint32_t workers, uint32_t tasks_per_worker) {
    std::vector<uint32_t> ids(workers * tasks_per_worker);
    for (uint32_t i = 0; i < ids.size(); i++) {
        ids[i] = i % workers;
    }
    return ids;
}

// One task per enqueue(), joined with wait().
struct EnqueueWait {
    static constexpr const char* name = "BM_ThreadPoolFanOut";
    static constexpr bool separate_submit = true;
    // Returns when the last task was submitted.
    template <typename MakeTask>
    static int64_t fan_out(ThreadPool& pool, const std::vector<uint32_t>& worker_ids, const MakeTask& make_task) {
        for (uint32_t i = 0; i < worker_ids.size(); i++) {
            pool.enqueue(make_task(i), worker_ids[i]);
        }
        const int64_t submitted = now_ns();
        pool.wait();
        return submitted;
    }
};

// All tasks in one parallel_for(), which submits and joins.
struct ParallelFor {
    static constexpr const char* name = "BM_ThreadPoolParallelFor";
    static constexpr bool separate_submit = false;
    template <typename MakeTask>
    static int64_t fan_out(ThreadPool& pool, const std::vector<uint32_t>& worker_ids, const MakeTask& make_task) {
        pool.parallel_for(worker_ids, [&make_task](size_t i) { make_task(i)(); });
        return 0;
    }
};

template <typename Pool, typename Api, size_t PadBytes>
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
    const auto worker_ids = worker_of_task(workers, tasks_per_worker);
    const auto caller = std::this_thread::get_id();
    std::vector<TaskTimes> times(num_tasks);
    std::vector<double> wall_us, submit_us, first_start_us, last_start_us, join_us;
    std::array<char, PadBytes> pad{};  // Optionally exceeds the callable's small buffer, like the dispatch captures.
    auto make_task = [&times, work_ns, &pad, caller](uint32_t i) {
        return [slot = &times[i], work_ns, pad, caller]() {
            slot->start = now_ns();
            slot->on_caller = std::this_thread::get_id() == caller;
            benchmark::DoNotOptimize(pad);
            spin_for_ns(work_ns);
            slot->end = now_ns();
        };
    };

    const Usage self_before = usage(RUSAGE_SELF);
    const Usage caller_before = usage(RUSAGE_THREAD);
    int64_t tasks_on_caller = 0;
    const int64_t run_begin = now_ns();
    for ([[maybe_unused]] auto _ : state) {
        spin_for_ns(gap_ns);
        const int64_t t0 = now_ns();
        const int64_t t1 = Api::fan_out(pool, worker_ids, make_task);
        const int64_t t2 = now_ns();

        int64_t first_start = INT64_MAX, last_start = 0, last_end = 0;
        for (const auto& t : times) {
            first_start = std::min(first_start, t.start);
            last_start = std::max(last_start, t.start);
            last_end = std::max(last_end, t.end);
            tasks_on_caller += t.on_caller ? 1 : 0;
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
    if (Api::separate_submit) {
        // For the pass-through pool the submit time includes running the tasks.
        state.counters["submit_per_task_us"] = percentile(submit_us, 0.5);
    }
    state.counters["first_start_p50_us"] = percentile(first_start_us, 0.5);
    state.counters["first_start_p99_us"] = percentile(first_start_us, 0.99);
    state.counters["last_start_p50_us"] = percentile(last_start_us, 0.5);
    state.counters["last_start_p99_us"] = percentile(last_start_us, 0.99);
    state.counters["join_p50_us"] = percentile(join_us, 0.5);
    state.counters["caller_parks"] = (caller_after.voluntary_switches - caller_before.voluntary_switches) / iters;
    state.counters["caller_task_share"] = tasks_on_caller / (iters * num_tasks);
    if (!Pool::runs_on_caller) {
        const double worker_cpu_s = (self_after.cpu_s - self_before.cpu_s) - (caller_after.cpu_s - caller_before.cpu_s);
        const double task_cpu_s = ((iters * num_tasks) - tasks_on_caller) * work_ns * 1e-9;
        // Worker CPU beyond the task bodies, averaged over the run.
        state.counters["worker_overhead_cores"] = (worker_cpu_s - task_cpu_s) / run_s;
        state.counters["worker_parks"] = ((self_after.voluntary_switches - self_before.voluntary_switches) -
                                          (caller_after.voluntary_switches - caller_before.voluntary_switches)) /
                                         iters;
    }
}

// {workers, tasks_per_worker, work_ns, gap_us}
using Grid = std::vector<std::vector<int64_t>>;

template <typename Pool, size_t PadBytes, typename Api = EnqueueWait>
void register_fan_out(const Grid& grid, int64_t iterations) {
    const std::string name = std::string(Api::name) + "/" + Pool::name + "/pad:" + std::to_string(PadBytes);
    benchmark::RegisterBenchmark(name, BM_FanOut<Pool, Api, PadBytes>)
        ->ArgNames({"workers", "tasks_per_worker", "work_ns", "gap_us"})
        ->ArgsProduct(grid)
        ->Iterations(iterations)
        ->UseManualTime()
        ->Unit(benchmark::kMicrosecond);
}

template <typename Pool>
void register_pool() {
    if (tt::parse_env("TT_POOL_BENCH_FULL", false)) {
        const Grid full = {{1, 8, 32}, {1, 4}, {0, 600, 1300, 2000, 10000}, {0, 5, 63, 144, 500}};
        register_fan_out<Pool, 0>(full, 2000);
        register_fan_out<Pool, 64>({{32}, {1}, {600}, {0, 63}}, 2000);
        register_fan_out<Pool, 0, ParallelFor>(full, 2000);
        return;
    }
    // p50/p90 per-device write and host gap in GLM-5.2 and Kimi K2.7 prefill (#57586).
    const Grid model_sized = {{8, 32}, {1}, {600, 1300}, {0, 63, 144}};
    register_fan_out<Pool, 0>(model_sized, 5000);
    // Chunked fan-out.
    register_fan_out<Pool, 0>({{32}, {4}, {600}, {0}}, 5000);
    // Capture larger than the callable's small buffer.
    register_fan_out<Pool, 64>({{32}, {1}, {600}, {0}}, 5000);
    // Parked workers: the dispatch pool's fan-outs are ~10 ms apart in those models.
    const Grid parked = {{32}, {1}, {600}, {10000}};
    register_fan_out<Pool, 0>(parked, 300);
    // The same fan-outs through parallel_for; the capture size does not apply.
    register_fan_out<Pool, 0, ParallelFor>(model_sized, 5000);
    register_fan_out<Pool, 0, ParallelFor>({{32}, {4}, {600}, {0}}, 5000);
    register_fan_out<Pool, 0, ParallelFor>(parked, 300);
}

const bool registered = [] {
    register_pool<DeviceBoundPool>();
    register_pool<PassThroughPool>();
    return true;
}();

}  // namespace fan_out

// Bulk enqueue throughput. Must stay after fan_out: each run creates a pool. Supported env variables:
//  * TT_METAL_NUM_BENCHMARK_THREADS sets the pool size (8).
namespace bulk {

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

}  // namespace bulk
