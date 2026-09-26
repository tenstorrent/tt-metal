// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Measures std::function's real small-object buffer for the active standard library and checks it
// against the constant the benchmark compiles in (candidates.hpp).
//
// The benchmark's fairness rests on those two agreeing: the candidates are pinned to exactly the
// buffer std::function gets. Exits non-zero on a mismatch so a standard library that moves the
// boundary fails loudly rather than silently skewing the comparison.

#include "candidates.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <new>
#include <utility>

namespace {

std::size_t g_allocs = 0;

template <std::size_t N>
struct Cap {
    char data[N]{};
};

// Reported buffer is the largest capture that still avoids an allocation. A char array has
// alignment 1, the most permissive case, so this finds the true maximum.
template <std::size_t N>
void probe(std::size_t& largest_inline, bool verbose) {
    Cap<N> c{};
    g_allocs = 0;
    {
        std::function<void()> f{[c]() { (void)c; }};
        f();
    }
    if (g_allocs == 0) {
        largest_inline = N;
    }
    if (verbose) {
        std::printf("  capture %2zu B -> %s\n", N, g_allocs ? "HEAP" : "inline");
    }
}

template <std::size_t... Ns>
void probe_all(std::size_t& largest, bool verbose, std::index_sequence<Ns...>) {
    (probe<Ns + 1>(largest, verbose), ...);
}

}  // namespace

void* operator new(std::size_t n) {
    ++g_allocs;
    void* p = std::malloc(n ? n : 1);
    if (!p) {
        throw std::bad_alloc{};
    }
    return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

int main(int argc, char** argv) {
    const bool verbose = argc > 1;

    std::printf("stdlib: %s\n", bench::kStdlibName);
    std::printf("sizeof(std::function<void()>) = %zu\n", sizeof(std::function<void()>));

    std::size_t largest_inline = 0;
    probe_all(largest_inline, verbose, std::make_index_sequence<64>{});

    std::printf("measured inline buffer: %zu B\n", largest_inline);
    std::printf("compiled-in kInlineBytes: %zu B\n", bench::kInlineBytes);

    if (largest_inline != bench::kInlineBytes) {
        std::printf(
            "MISMATCH: std::function holds %zu B inline but the benchmark pins candidates to %zu B.\n"
            "The comparison would not be apples-to-apples. Update kInlineBytes in "
            "candidates.hpp.\n",
            largest_inline,
            bench::kInlineBytes);
        return 1;
    }

    std::printf("OK: candidates are pinned to the same inline capacity as std::function.\n");
    return 0;
}
