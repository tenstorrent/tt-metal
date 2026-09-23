// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Item 6 of #57444: object-size comparison. One translation unit, compiled once per candidate via
// -DCANDIDATE_{STD,ZOO,FU2}, so the resulting .o sizes are directly comparable.

#include "inline_capacity.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#if defined(CANDIDATE_FU2)
#include <function2/function2.hpp>
#elif defined(CANDIDATE_ZOO)
#include <zoo/FunctionPolicy.h>
#endif

namespace {
using bench_config::kInlineBytes;
using bench_config::kInlinePointers;
}  // namespace

#if defined(CANDIDATE_STD)
using Fn = std::function<void()>;
#elif defined(CANDIDATE_FU2)
using Fn = fu2::function_base<true, false, fu2::capacity_fixed<kInlineBytes>, true, true, void()>;
#elif defined(CANDIDATE_ZOO)
using Fn =
    zoo::Function<zoo::AnyContainer<zoo::Policy<void* [kInlinePointers], zoo::Destroy, zoo::Move, zoo::RTTI>>, void()>;
#else
#error "define one of CANDIDATE_STD / CANDIDATE_ZOO / CANDIDATE_FU2"
#endif

std::uint64_t g_sink = 0;

// A spread of distinct capture types, so the measurement reflects per-instantiation cost rather
// than a single specialization.
template <int N>
struct Capture {
    std::uint64_t data[N] = {};
};

template <int N>
[[gnu::noinline]] Fn make() {
    Capture<N> cap{};
    return Fn{[cap]() mutable { g_sink += cap.data[0] + N; }};
}

template <int N>
[[gnu::noinline]] void run() {
    Fn f = make<N>();
    f();
}

void exercise() {
    run<1>();
    run<2>();
    run<3>();
    run<4>();
    run<6>();
    run<8>();
    run<12>();
    run<16>();
    run<24>();
    run<32>();
}
