// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per-op cycle cost of integer divide and modulus on the DM (RISC-V).
//
// Two sweeps:
//   1. Static-divisor sweep. Numerator is `0xDEADBEEF + i * 0x9E3779B9` (varies
//      per iter so the compiler can't constant-fold). Divisor is a literal.
//   2. Runtime (numerator, divisor) matrix. Both n and d are loaded from a
//      volatile, then in the loop the numerator is `(n ^ i)` so it varies
//      near n's magnitude (defeats LICM). Densely samples small numerators so
//      the `n < d` anomaly and the bit-width scaling are easy to plot.
//
// Results land in a small in-L1 records buffer at kResultsBase:
//   word 0    : record count
//   words 1.. : 4-word records {kind, n, d, cycles}
// where kind is one of {Baseline, StaticDiv, StaticMod, RuntimeDiv, RuntimeMod}.
// The host reads the buffer back after the program completes and writes a
// CSV.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/dev_mem_map.h"
#include <cstdint>

namespace {

inline uint32_t rdcycle() {
    uint32_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}

// L1 anchor so the optimizer can't drop the measurement loops as dead code.
constexpr uint32_t kSinkAddr = 0x10000;
inline volatile uint32_t* sink_ptr() { return reinterpret_cast<volatile uint32_t*>(kSinkAddr); }

// Records buffer. Layout matches the host-side reader.
// Note: we write to MEM_L1_UNCACHED_BASE + 0x30000 so the host's NOC readback
// (which bypasses the DM's L1 cache) sees the writes immediately. The host
// passes 0x30000 to ReadFromDeviceL1; that's the L1 *offset*, not the
// kernel-visible address, so the uncached alias on the kernel side and the
// raw offset on the host side hit the same physical L1 cells.
constexpr uint32_t kResultsL1Offset = 0x30000;
constexpr uint32_t kResultsAddr = MEM_L1_UNCACHED_BASE + kResultsL1Offset;

constexpr uint32_t kKindBaseline = 0;
constexpr uint32_t kKindStaticDiv = 1;
constexpr uint32_t kKindStaticMod = 2;
constexpr uint32_t kKindRuntimeDiv = 3;
constexpr uint32_t kKindRuntimeMod = 4;

inline volatile uint32_t* results_word(uint32_t idx) {
    return reinterpret_cast<volatile uint32_t*>(kResultsAddr + idx * sizeof(uint32_t));
}

uint32_t g_next_slot = 1;  // word 0 is the record count

inline void append_result(uint32_t kind, uint32_t n, uint32_t d, uint32_t cycles) {
    *results_word(g_next_slot + 0) = kind;
    *results_word(g_next_slot + 1) = n;
    *results_word(g_next_slot + 2) = d;
    *results_word(g_next_slot + 3) = cycles;
    g_next_slot += 4;
}

// Numerator generator for the static-divisor sweep.
inline uint32_t numerator(uint32_t i) { return 0xDEADBEEFu + i * 0x9E3779B9u; }

}  // namespace

// --- Static-divisor sweep ------------------------------------------------
template <uint32_t D>
__attribute__((noinline)) static uint32_t measure_static_div(uint32_t num_iters, volatile uint32_t* anchor) {
    uint32_t sink = 0;
    uint32_t t0 = rdcycle();
    for (uint32_t i = 0; i < num_iters; i++) {
        sink ^= numerator(i) / D;
    }
    uint32_t t1 = rdcycle();
    *anchor = sink;
    return (t1 - t0) / num_iters;
}

template <uint32_t D>
__attribute__((noinline)) static uint32_t measure_static_mod(uint32_t num_iters, volatile uint32_t* anchor) {
    uint32_t sink = 0;
    uint32_t t0 = rdcycle();
    for (uint32_t i = 0; i < num_iters; i++) {
        sink ^= numerator(i) % D;
    }
    uint32_t t1 = rdcycle();
    *anchor = sink;
    return (t1 - t0) / num_iters;
}

// --- Runtime (n, d) matrix -----------------------------------------------
__attribute__((noinline)) static uint32_t measure_runtime_div(
    uint32_t n, uint32_t d, uint32_t num_iters, volatile uint32_t* anchor) {
    volatile uint32_t vn = n;
    volatile uint32_t vd = d;
    uint32_t rn = vn;
    uint32_t rd = vd;
    uint32_t sink = 0;
    uint32_t t0 = rdcycle();
    for (uint32_t i = 0; i < num_iters; i++) {
        sink ^= (rn ^ i) / rd;
    }
    uint32_t t1 = rdcycle();
    *anchor = sink;
    return (t1 - t0) / num_iters;
}

__attribute__((noinline)) static uint32_t measure_runtime_mod(
    uint32_t n, uint32_t d, uint32_t num_iters, volatile uint32_t* anchor) {
    volatile uint32_t vn = n;
    volatile uint32_t vd = d;
    uint32_t rn = vn;
    uint32_t rd = vd;
    uint32_t sink = 0;
    uint32_t t0 = rdcycle();
    for (uint32_t i = 0; i < num_iters; i++) {
        sink ^= (rn ^ i) % rd;
    }
    uint32_t t1 = rdcycle();
    *anchor = sink;
    return (t1 - t0) / num_iters;
}

#define EMIT_STATIC(D, ANCHOR, ITERS)                                                         \
    do {                                                                                      \
        const uint32_t _cd = measure_static_div<(D)>((ITERS), (ANCHOR));                      \
        const uint32_t _cm = measure_static_mod<(D)>((ITERS), (ANCHOR));                      \
        append_result(kKindStaticDiv, 0, (D), _cd);                                           \
        append_result(kKindStaticMod, 0, (D), _cm);                                           \
        DPRINT << "STATIC d=" << (uint32_t)(D) << " div=" << _cd << " mod=" << _cm << ENDL(); \
    } while (0)

void kernel_main() {
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    volatile uint32_t* anchor = sink_ptr();

    // Initialize record count to 0; we'll set the real count at the end.
    *results_word(0) = 0;
    g_next_slot = 1;

    // --- Baseline ---------------------------------------------------------
    {
        uint32_t sink = 0;
        uint32_t t0 = rdcycle();
        for (uint32_t i = 0; i < num_iters; i++) {
            sink ^= numerator(i);
        }
        uint32_t t1 = rdcycle();
        *anchor = sink;
        const uint32_t baseline = (t1 - t0) / num_iters;
        append_result(kKindBaseline, 0, 0, baseline);
        DPRINT << "DIVMOD_PERF iters=" << num_iters << " baseline_per_iter=" << baseline << ENDL();
    }

    // --- Static-divisor sweep (one record per (divisor, op)) -------------
    // Original 10 divisors plus a wider sweep to pin down whether the `/7`
    // magic-multiply miss is a one-off, whether the compiler ever stops
    // emitting magic for larger constants, and whether `2^N - 1` constants
    // (15, 31, 63, 127) are handled differently.
    EMIT_STATIC(2, anchor, num_iters);
    EMIT_STATIC(3, anchor, num_iters);
    EMIT_STATIC(4, anchor, num_iters);
    EMIT_STATIC(5, anchor, num_iters);
    EMIT_STATIC(6, anchor, num_iters);
    EMIT_STATIC(7, anchor, num_iters);
    EMIT_STATIC(8, anchor, num_iters);
    EMIT_STATIC(9, anchor, num_iters);
    EMIT_STATIC(10, anchor, num_iters);
    EMIT_STATIC(11, anchor, num_iters);
    EMIT_STATIC(12, anchor, num_iters);
    EMIT_STATIC(13, anchor, num_iters);
    EMIT_STATIC(15, anchor, num_iters);
    EMIT_STATIC(16, anchor, num_iters);
    EMIT_STATIC(24, anchor, num_iters);
    EMIT_STATIC(31, anchor, num_iters);
    EMIT_STATIC(32, anchor, num_iters);
    EMIT_STATIC(63, anchor, num_iters);
    EMIT_STATIC(100, anchor, num_iters);
    EMIT_STATIC(127, anchor, num_iters);
    EMIT_STATIC(1000, anchor, num_iters);
    EMIT_STATIC(1024, anchor, num_iters);

    // --- Runtime (n, d) matrix -------------------------------------------
    // Numerator list - dense around small magnitudes and the n=d transition
    // (the QUAS-4272 anomaly), sparser at the high end for trend.
    constexpr uint32_t kNumerators[] = {0u,     1u,     2u,     4u,       8u,       16u,      32u,        64u,   128u,
                                        256u,   512u,   1023u,  1024u,    1025u,    1500u,    2048u,      4096u, 8192u,
                                        16384u, 32768u, 65536u, 1u << 20, 1u << 24, 1u << 28, 0xFFFFFFFFu};
    constexpr uint32_t kNumNs = sizeof(kNumerators) / sizeof(kNumerators[0]);

    constexpr uint32_t kDivisors[] = {2, 3, 4, 5, 6, 7, 8, 16, 32, 1024};
    constexpr uint32_t kNumDs = sizeof(kDivisors) / sizeof(kDivisors[0]);

    for (uint32_t ni = 0; ni < kNumNs; ni++) {
        const uint32_t n = kNumerators[ni];
        for (uint32_t di = 0; di < kNumDs; di++) {
            const uint32_t d = kDivisors[di];
            const uint32_t cd = measure_runtime_div(n, d, num_iters, anchor);
            const uint32_t cm = measure_runtime_mod(n, d, num_iters, anchor);
            append_result(kKindRuntimeDiv, n, d, cd);
            append_result(kKindRuntimeMod, n, d, cm);
        }
        // One short status print per numerator so progress is visible.
        DPRINT << "HW n=0x" << HEX() << n << DEC() << " done (" << kNumDs << " divisors)" << ENDL();
    }

    // Final record count.
    *results_word(0) = (g_next_slot - 1) / 4;
    DPRINT << "DIVMOD_PERF records=" << *results_word(0) << " at L1 0x" << HEX() << kResultsL1Offset << DEC() << ENDL();
}
