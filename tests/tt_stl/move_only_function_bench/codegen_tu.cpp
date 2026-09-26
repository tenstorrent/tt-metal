// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Item 6 of #57444: object-size comparison. One translation unit, compiled once per candidate via
// -DCANDIDATE_{STD,ZOO,FU2}, so the resulting .o sizes are directly comparable.

#include "selected_candidate.hpp"

#include <cstdint>

std::uint64_t g_sink = 0;

// A spread of distinct capture types, so the measurement reflects per-instantiation cost rather
// than a single specialization.
template <int N>
struct Capture {
    std::uint64_t data[N] = {};
};

template <int N>
[[gnu::noinline]] Fn<void()> make() {
    Capture<N> cap{};
    return Fn<void()>{[cap]() mutable { g_sink += cap.data[0] + N; }};
}

template <int N>
[[gnu::noinline]] void run() {
    Fn<void()> f = make<N>();
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
