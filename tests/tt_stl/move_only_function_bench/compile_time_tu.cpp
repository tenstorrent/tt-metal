// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Item 7 of #57444: compile-time impact. Instantiates the chosen candidate over many distinct
// signatures and capture types; time this TU per candidate, best-of-N.

// This TU exists to measure one library's parse and instantiation cost, so it must not pay to
// include the others.
#if defined(CANDIDATE_STD)
#define BENCH_SKIP_ZOO
#define BENCH_SKIP_FU2
#elif defined(CANDIDATE_ZOO)
#define BENCH_SKIP_FU2
#elif defined(CANDIDATE_FU2)
#define BENCH_SKIP_ZOO
#endif

#include "candidates.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#if defined(CANDIDATE_STD)
template <typename Sig>
using Fn = bench::StdFunction<Sig>;
#elif defined(CANDIDATE_FU2)
template <typename Sig>
using Fn = bench::Fu2Function<Sig>;
#elif defined(CANDIDATE_ZOO)
template <typename Sig>
using Fn = bench::ZooFunction<Sig>;
#else
#error "define one of CANDIDATE_STD / CANDIDATE_ZOO / CANDIDATE_FU2"
#endif

template <int N>
struct Tag {
    std::uint64_t v[(N % 5) + 1] = {};
};

template <int N>
[[gnu::noinline]] void instantiate() {
    Tag<N> cap{};
    Fn<void()> a{[cap]() mutable { (void)cap; }};
    Fn<std::uint64_t()> b{[cap]() mutable -> std::uint64_t { return cap.v[0] + N; }};
    Fn<void(std::uint64_t)> c{[cap](std::uint64_t x) mutable { (void)(cap.v[0] + x); }};
    a();
    (void)b();
    c(1);
}

#define INSTANTIATE_10(base) \
    instantiate<base + 0>(); \
    instantiate<base + 1>(); \
    instantiate<base + 2>(); \
    instantiate<base + 3>(); \
    instantiate<base + 4>(); \
    instantiate<base + 5>(); \
    instantiate<base + 6>(); \
    instantiate<base + 7>(); \
    instantiate<base + 8>(); \
    instantiate<base + 9>()

void exercise_compile_time() {
    INSTANTIATE_10(0);
    INSTANTIATE_10(10);
    INSTANTIATE_10(20);
    INSTANTIATE_10(30);
    INSTANTIATE_10(40);
    INSTANTIATE_10(50);
    INSTANTIATE_10(60);
    INSTANTIATE_10(70);
    INSTANTIATE_10(80);
    INSTANTIATE_10(90);
}
