// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Timer check, on one Keraunos CCE hart (IDLE_ETH). Establishes what rdcycle can be trusted for before any benchmark
// relies on it:
//   - monotonic: back-to-back fenced reads never go backwards (aux kMonotonic*);
//   - bracket cost: the smallest and largest interval of an empty fenced bracket (cycles kEmptyMin/kEmptyMax);
//   - it counts executed cycles: a chain of A dependent adds (cycles kAddLoop) should take about A cycles;
//   - it keeps counting through stalls: a chain of L dependent uncached loads from own SRAM (kLocalLoadChase), and
//     optionally from another tile's SRAM through its global address (kRemoteLoadChase). If the counter stopped
//     during a stall, these would read about as cheap as the add chain.
//
// Runtime args: 0 base, 1 hart, 2 adds, 3 loads, 4-5 remote chase word (global address, 0 = skip).
// The chase words hold their own address, so every load depends on the one before it.

#include <cstdint>

#include "tests/tt_metal/tt_metal/test_kernels/misc/keraunos_ubench/keraunos_ubench.h"

using namespace keraunos_ubench;
namespace tc = keraunos_ubench::timer_check;

namespace {

constexpr uint32_t kEmptyBrackets = 64;
constexpr uint32_t kMonotonicReads = 1024;

uint64_t chase(volatile uint64_t* start, uint32_t loads) {
    uintptr_t p = reinterpret_cast<uintptr_t>(start);
    const uint64_t t0 = now();
    for (uint32_t i = 0; i < loads; i++) {
        p = static_cast<uintptr_t>(*reinterpret_cast<volatile uint64_t*>(p));
    }
    const uint64_t t1 = now();
    // Keep the chain live.
    asm volatile("" : : "r"(p));
    return t1 - t0;
}

}  // namespace

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0);
    const uint32_t hart = get_arg_val<uint32_t>(1);
    const uint32_t adds = get_arg_val<uint32_t>(2);
    const uint32_t loads = get_arg_val<uint32_t>(3);
    const uint64_t remote = arg_u64(4);

    volatile Result* r = result_block(base, hart);
    r->kernel = static_cast<uint32_t>(KernelId::TimerCheck);
    r->hart = hart;
    r->status = kStatusRunning;
    if (hart >= kMaxHarts) {
        r->detail = 1;
        finish(r, kStatusBadArgs);
        return;
    }

    // Monotonic.
    uint64_t backwards = 0, zero = 0;
    uint64_t prev = now();
    for (uint32_t i = 0; i < kMonotonicReads; i++) {
        const uint64_t c = now();
        backwards += c < prev;
        zero += c == prev;
        prev = c;
    }
    r->aux[tc::kMonotonicReads] = kMonotonicReads;
    r->aux[tc::kMonotonicBackwards] = backwards;
    r->aux[tc::kMonotonicZeroDelta] = zero;

    // Empty bracket.
    uint64_t lo = ~0ull, hi = 0;
    for (uint32_t i = 0; i < kEmptyBrackets; i++) {
        const uint64_t t0 = now();
        const uint64_t t1 = now();
        const uint64_t d = t1 - t0;
        lo = d < lo ? d : lo;
        hi = d > hi ? d : hi;
    }
    r->cycles[tc::kEmptyMin] = lo;
    r->cycles[tc::kEmptyMax] = hi;

    // Dependent adds: each add needs the previous result.
    {
        uint64_t x = hart;
        const uint64_t t0 = now();
        for (uint32_t i = 0; i < adds; i++) {
            asm volatile("addi %0, %0, 1" : "+r"(x));
        }
        const uint64_t t1 = now();
        asm volatile("" : : "r"(x));
        r->cycles[tc::kAddLoop] = t1 - t0;
        r->aux[0] = adds;
    }

    // Dependent uncached loads from own SRAM: the staging slot holds its own uncached address.
    {
        volatile uint64_t* word = uncached<uint64_t>(base + staging_offset(hart));
        *word = reinterpret_cast<uintptr_t>(word);
        asm volatile("fence iorw, iorw" ::: "memory");
        r->cycles[tc::kLocalLoadChase] = chase(word, loads);
        r->aux[1] = loads;
    }

    // Dependent loads from another tile's SRAM through its global address.
    if (remote != 0) {
        volatile uint64_t* word = reinterpret_cast<volatile uint64_t*>(static_cast<uintptr_t>(remote));
        *word = remote;
        asm volatile("fence iorw, iorw" ::: "memory");
        r->cycles[tc::kRemoteLoadChase] = chase(word, loads);
        r->aux[2] = loads;
    }

    r->iterations_done = 1;
    finish(r, kStatusDone);
}
