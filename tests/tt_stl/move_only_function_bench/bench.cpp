// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Evaluation harness for GitHub issue #57444: pick the implementation to back
// ttsl::move_only_function. Compares zoo and fu2 against std::function at a common inline
// capacity. See README.md for how to run and how to read the numbers.

#include "candidates.hpp"

#include <benchmark/benchmark.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <vector>
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

using bench::kInlineBytes;
using bench::kInlinePointers;

using StdFn = bench::StdFunction<void()>;
using ZooFn = bench::ZooFunction<void()>;
using Fu2Fn = bench::Fu2Function<void()>;

// Captures sized to sit either side of kInlineBytes.
struct SmallCapture {
    std::uint64_t a = 1;
    std::uint64_t b = 2;
    std::uint64_t value() const { return a; }
};
static_assert(sizeof(SmallCapture) <= kInlineBytes);

// Three 64-bit words. On LP64 that is 24 bytes: inline under libc++ (3-pointer buffer), heap under
// libstdc++ (2-pointer buffer), so it probes exactly the band where the two disagree.
struct BoundaryCapture {
    std::uint64_t a = 1;
    std::uint64_t b = 2;
    std::uint64_t c = 3;
    std::uint64_t value() const { return a; }
};
static_assert(sizeof(BoundaryCapture) == 24);

struct LargeCapture {
    std::uint64_t data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::uint64_t value() const { return data[0]; }
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

// --- 5: ring-buffer job path, as tt::tt_metal::ThreadPool actually implements it ---------------
//
// INFORMATIONAL, not decisive. Read the heap rows; treat the inline rows as indicative only.
//
// Even with the barriers below, the compiler can still see the callable type here, which it cannot
// do behind ThreadPool::push in another TU. A cross-TU probe measured std::function at 9.75 ns/job
// against ~5 ns here, so the inline numbers understate the cost of a genuinely erased call, and
// they understate it unevenly -- whichever type the optimiser sees through benefits most. The heap
// rows are sturdier, since an allocation cannot be optimised away.

// thread_pool.cpp:129-186 is a statically allocated ring buffer of 65536 slots, each permanently
// holding a std::function<void()>. A job therefore costs two move-assignments into already-live
// objects -- into the slot on push, out of it on pop -- and allocates nothing per job.
//
// An earlier version of this scenario used std::deque<Fn> with emplace_back/pop_front. That
// measured construct-in-place plus deque block allocation, an operation the ThreadPool never
// performs, and never exercised move-assignment at all.
template <typename Fn, typename Capture>
void BM_RingBufferJobPath(benchmark::State& state) {
    const std::size_t batch = static_cast<std::size_t>(state.range(0));

    // Pre-allocated slots, live for the whole run, exactly like the ring buffer.
    constexpr std::size_t kSlots = 2048;
    std::vector<Fn> slots(kSlots);

    take_allocations();
    std::size_t iterations = 0;
    for (auto _ : state) {
        for (std::size_t i = 0; i < batch; ++i) {
            Capture cap{};
            benchmark::DoNotOptimize(cap);
            // push: move-assign the job into its slot.
            Fn* slot = &slots[i % kSlots];
            benchmark::DoNotOptimize(slot);
            *slot = Fn{[cap]() mutable { g_sink += cap.value(); }};
            // Keep the stored callable opaque. Without a memory clobber the compiler devirtualises
            // the invocation and deletes the whole push/pop/call -- std::function collapsed to
            // three instructions and measured ~3x faster than it really is. ThreadPool::push takes
            // its argument across a TU boundary with the type erased, so that optimisation is not
            // available to it. DoNotOptimize on the pointer alone is not enough; the clobber is.
            benchmark::ClobberMemory();
            // pop: move-assign out into the worker's local, then invoke.
            Fn task = std::move(*slot);
            benchmark::DoNotOptimize(task);
            benchmark::ClobberMemory();
            task();
        }
        benchmark::ClobberMemory();
        ++iterations;
    }
    state.counters["allocs/iter"] = static_cast<double>(take_allocations()) / static_cast<double>(iterations);
    state.SetItemsProcessed(static_cast<std::int64_t>(iterations * batch));
}

// Reported once at startup.
struct SizeReport {
    SizeReport() {
        std::printf(
            "inline capacity in use: %zu B (%s)\n"
            "sizes: std::function=%zu/%zu zoo=%zu/%zu fu2=%zu/%zu (bytes: sizeof/alignof)\n",
            kInlineBytes,
            bench::kStdlibName,
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

// Real enqueue sites in tt_metal/impl/profiler capture from [this] (8 B) up to
// [&a, &b, &c, &d, i] (~36 B), so the heap path is the common case at the shipped capacity, not
// the exception. Both are measured.
BENCHMARK_TEMPLATE(BM_RingBufferJobPath, StdFn, SmallCapture)->Arg(1024);
BENCHMARK_TEMPLATE(BM_RingBufferJobPath, ZooFn, SmallCapture)->Arg(1024);
BENCHMARK_TEMPLATE(BM_RingBufferJobPath, Fu2Fn, SmallCapture)->Arg(1024);
BENCHMARK_TEMPLATE(BM_RingBufferJobPath, StdFn, LargeCapture)->Arg(1024);
BENCHMARK_TEMPLATE(BM_RingBufferJobPath, ZooFn, LargeCapture)->Arg(1024);
BENCHMARK_TEMPLATE(BM_RingBufferJobPath, Fu2Fn, LargeCapture)->Arg(1024);

BENCHMARK_MAIN();
