// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Evaluation harness for GitHub issue #57444: pick the implementation to back
// ttsl::move_only_function. Compares zoo and fu2 against std::function at a common inline
// capacity. See README.md for how to run and how to read the numbers.

#include "inline_capacity.hpp"

#include <benchmark/benchmark.h>

#include <function2/function2.hpp>
#include <zoo/FunctionPolicy.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <cstdio>
#include <new>
#include <utility>

namespace {

// Allocation counter. Every benchmark reports allocs/iteration, which is what proves a "small"
// case really stayed inline and a "large" case really spilled to the heap, rather than us
// inferring it from capture size.
std::atomic<std::size_t> g_allocations{0};

std::size_t take_allocations() { return g_allocations.exchange(0, std::memory_order_relaxed); }

using bench_config::kInlineBytes;
using bench_config::kInlinePointers;

using StdFn = std::function<void()>;
using Fu2Fn = fu2::function_base<
    /*IsOwning=*/true,
    /*IsCopyable=*/false,
    fu2::capacity_fixed<kInlineBytes>,
    /*IsThrowing=*/true,
    /*HasStrongExceptGuarantee=*/false,
    void()>;
using ZooFn = zoo::VTableFunction<kInlinePointers, void()>;

// Captures sized to sit either side of kInlineBytes.
struct SmallCapture {
    std::uint64_t a = 1;
    std::uint64_t b = 2;
};
static_assert(sizeof(SmallCapture) <= kInlineBytes);

// 24 bytes: inline under libc++, heap under libstdc++. Probes exactly the band where the two
// standard libraries disagree.
struct BoundaryCapture {
    std::uint64_t a = 1;
    std::uint64_t b = 2;
    std::uint64_t c = 3;
};
static_assert(sizeof(BoundaryCapture) == 24);

struct LargeCapture {
    std::uint64_t data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
};
static_assert(sizeof(LargeCapture) > kInlineBytes);

std::uint64_t g_sink = 0;

// --- 1 / 2: construct, invoke, destroy ---------------------------------------------------------

template <typename Fn, typename Capture>
void BM_ConstructInvokeDestroy(benchmark::State& state) {
    take_allocations();
    std::size_t iterations = 0;
    for (auto _ : state) {
        Capture cap{};
        benchmark::DoNotOptimize(cap);
        Fn f{[cap]() mutable { benchmark::DoNotOptimize(cap); }};
        benchmark::DoNotOptimize(f);
        f();
        benchmark::ClobberMemory();
        ++iterations;
    }
    state.counters["allocs/iter"] = static_cast<double>(take_allocations()) / static_cast<double>(iterations);
}

// --- 3: move-only capture ----------------------------------------------------------------------

// std::function cannot store a move-only callable, so the baseline is the shared_ptr wrapper the
// codebase uses today. That workaround is precisely what #57444 wants to remove, so it is the
// honest baseline rather than an omission.
void BM_MoveOnlyCapture_Std(benchmark::State& state) {
    take_allocations();
    std::size_t iterations = 0;
    for (auto _ : state) {
        auto owned = std::make_shared<std::unique_ptr<std::uint64_t>>(std::make_unique<std::uint64_t>(7));
        StdFn f{[owned]() { benchmark::DoNotOptimize(**owned); }};
        benchmark::DoNotOptimize(f);
        f();
        benchmark::ClobberMemory();
        ++iterations;
    }
    state.counters["allocs/iter"] = static_cast<double>(take_allocations()) / static_cast<double>(iterations);
}

template <typename Fn>
void BM_MoveOnlyCapture(benchmark::State& state) {
    take_allocations();
    std::size_t iterations = 0;
    for (auto _ : state) {
        auto owned = std::make_unique<std::uint64_t>(7);
        Fn f{[owned = std::move(owned)]() { benchmark::DoNotOptimize(*owned); }};
        benchmark::DoNotOptimize(f);
        f();
        benchmark::ClobberMemory();
        ++iterations;
    }
    state.counters["allocs/iter"] = static_cast<double>(take_allocations()) / static_cast<double>(iterations);
}

// --- 4: move construction / move assignment ----------------------------------------------------

template <typename Fn, typename Capture>
void BM_MoveConstruct(benchmark::State& state) {
    take_allocations();
    std::size_t iterations = 0;
    for (auto _ : state) {
        Capture cap{};
        Fn src{[cap]() mutable { benchmark::DoNotOptimize(cap); }};
        benchmark::DoNotOptimize(src);
        Fn dst{std::move(src)};
        benchmark::DoNotOptimize(dst);
        benchmark::ClobberMemory();
        ++iterations;
    }
    state.counters["allocs/iter"] = static_cast<double>(take_allocations()) / static_cast<double>(iterations);
}

template <typename Fn, typename Capture>
void BM_MoveAssign(benchmark::State& state) {
    take_allocations();
    std::size_t iterations = 0;
    for (auto _ : state) {
        Capture cap{};
        Fn src{[cap]() mutable { benchmark::DoNotOptimize(cap); }};
        Fn dst{[]() {}};
        benchmark::DoNotOptimize(src);
        dst = std::move(src);
        benchmark::DoNotOptimize(dst);
        benchmark::ClobberMemory();
        ++iterations;
    }
    state.counters["allocs/iter"] = static_cast<double>(take_allocations()) / static_cast<double>(iterations);
}

// --- 5: queue throughput, mirroring the ThreadPool job path ------------------------------------

// tt::tt_metal::ThreadPool::enqueue takes std::function<void()>&& today; this models that hot path
// as enqueue-many then drain-many through a container.
template <typename Fn, typename Capture>
void BM_QueueThroughput(benchmark::State& state) {
    const std::size_t batch = static_cast<std::size_t>(state.range(0));
    take_allocations();
    std::size_t iterations = 0;
    std::deque<Fn> queue;
    for (auto _ : state) {
        for (std::size_t i = 0; i < batch; ++i) {
            Capture cap{};
            queue.emplace_back([cap]() mutable { g_sink += cap.a; });
        }
        while (!queue.empty()) {
            queue.front()();
            queue.pop_front();
        }
        benchmark::ClobberMemory();
        ++iterations;
    }
    state.counters["allocs/iter"] = static_cast<double>(take_allocations()) / static_cast<double>(iterations);
    state.SetItemsProcessed(static_cast<std::int64_t>(iterations * batch));
}

// Reported once at startup: object size drives how many entries fit in a deque block, which is
// visible in the queue-throughput allocation counts.
struct SizeReport {
    SizeReport() {
        std::printf(
            "inline capacity in use: %zu B (%s)\n"
            "sizes: std::function=%zu/%zu zoo=%zu/%zu fu2=%zu/%zu (bytes: sizeof/alignof)\n",
            kInlineBytes,
            bench_config::kStdlibName,
            sizeof(StdFn),
            alignof(StdFn),
            sizeof(ZooFn),
            alignof(ZooFn),
            sizeof(Fu2Fn),
            alignof(Fu2Fn));
    }
};
const SizeReport g_size_report{};

}  // namespace

// Counted global allocation. Defined outside the anonymous namespace so it replaces the real one.
void* operator new(std::size_t size) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(size == 0 ? 1 : size)) {
        return p;
    }
    throw std::bad_alloc{};
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

#define BENCH_ALL_CAPTURES(bm)                      \
    BENCHMARK_TEMPLATE(bm, StdFn, SmallCapture);    \
    BENCHMARK_TEMPLATE(bm, ZooFn, SmallCapture);    \
    BENCHMARK_TEMPLATE(bm, Fu2Fn, SmallCapture);    \
    BENCHMARK_TEMPLATE(bm, StdFn, BoundaryCapture); \
    BENCHMARK_TEMPLATE(bm, ZooFn, BoundaryCapture); \
    BENCHMARK_TEMPLATE(bm, Fu2Fn, BoundaryCapture); \
    BENCHMARK_TEMPLATE(bm, StdFn, LargeCapture);    \
    BENCHMARK_TEMPLATE(bm, ZooFn, LargeCapture);    \
    BENCHMARK_TEMPLATE(bm, Fu2Fn, LargeCapture)

BENCH_ALL_CAPTURES(BM_ConstructInvokeDestroy);
BENCH_ALL_CAPTURES(BM_MoveConstruct);
BENCH_ALL_CAPTURES(BM_MoveAssign);

BENCHMARK(BM_MoveOnlyCapture_Std);
BENCHMARK_TEMPLATE(BM_MoveOnlyCapture, ZooFn);
BENCHMARK_TEMPLATE(BM_MoveOnlyCapture, Fu2Fn);

BENCHMARK_TEMPLATE(BM_QueueThroughput, StdFn, SmallCapture)->Arg(1024);
BENCHMARK_TEMPLATE(BM_QueueThroughput, ZooFn, SmallCapture)->Arg(1024);
BENCHMARK_TEMPLATE(BM_QueueThroughput, Fu2Fn, SmallCapture)->Arg(1024);

BENCHMARK_MAIN();
