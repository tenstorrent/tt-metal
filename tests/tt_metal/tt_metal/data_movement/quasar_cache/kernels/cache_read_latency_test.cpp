// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Measures DM core cost of reading num_entries words spaced stride bytes apart from each level of the
// memory hierarchy, and of the ways fast dispatch clears a fetch queue entry.
//
// Each phase records six 64-bit deltas: mcycle, minstret, and four Rocket HPM counters programmed to
// D$ blocked cycles, long-latency interlock cycles, D$ misses, and load-use interlock cycles.
//
// Result layout at result_addr, as 32-bit words:
//   [0] mhpmevent3 read back after programming it, 0 if the counter is not implemented
//   [1..3] unused
//   then kNumPhases * kNumCounters 64-bit values, low word first
//
// Phases:
//   0 empty region (measurement overhead)
//   1 uncached reads, straight from TL1
//   2 cached reads after invalidating L1 D$ and L2, so every line misses to TL1
//   3 cached reads after invalidating only L1 D$, so lines should hit in L2
//   4 cached reads with the lines already in L1 D$
//   5 uncached zero stores
//   6 uncached zero stores, each followed by an L2 line invalidate with its fences
//   7 uncached zero stores, each followed by the invalidate register write alone
//   8 cached reads over sweep_lines 64B lines from base_addr, after invalidating them from L1 D$ and L2
//   9 the same sweep again. The sweep is several times the 4 KB L1 D$, so each line is evicted from it before
//     it is reused, but it fits in L2. No invalidate is involved, so these reads are real L2 hits.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

namespace {

constexpr std::uint32_t kNumPhases = 10;
constexpr std::uint32_t kNumCounters = 6;

// Rocket HPM event selectors: bits [7:0] pick the event set, higher bits select events within it.
constexpr std::uint64_t kEventDcacheBlocked = 1 | (1ull << (8 + 4));
constexpr std::uint64_t kEventLongLatency = 1 | (1ull << (8 + 1));
constexpr std::uint64_t kEventDcacheMiss = 2 | (1ull << (8 + 1));
constexpr std::uint64_t kEventLoadUse = 1 | (1ull << (8 + 0));

#define READ_CSR(csr)                                         \
    ({                                                        \
        std::uint64_t value;                                  \
        __asm__ __volatile__("csrr %0, " #csr : "=r"(value)); \
        value;                                                \
    })
#define WRITE_CSR(csr, value) __asm__ __volatile__("csrw " #csr ", %0" ::"r"(value))

struct Counters {
    std::uint64_t value[kNumCounters];
};

inline __attribute__((always_inline)) Counters read_counters() {
    Counters c;
    c.value[0] = READ_CSR(0xB00);  // mcycle
    c.value[1] = READ_CSR(0xB02);  // minstret
    c.value[2] = READ_CSR(0xB03);
    c.value[3] = READ_CSR(0xB04);
    c.value[4] = READ_CSR(0xB05);
    c.value[5] = READ_CSR(0xB06);
    return c;
}

inline volatile std::uint32_t* cached_word(std::uint32_t addr) {
    return reinterpret_cast<volatile std::uint32_t*>(static_cast<uintptr_t>(addr));
}
inline volatile std::uint32_t* uncached_word(std::uint32_t addr) {
    return reinterpret_cast<volatile std::uint32_t*>(static_cast<uintptr_t>(addr + MEM_L1_UNCACHED_BASE));
}

template <typename Body>
inline __attribute__((always_inline)) Counters measured(Body body) {
    __asm__ __volatile__("fence" ::: "memory");
    Counters start = read_counters();
    body();
    __asm__ __volatile__("fence" ::: "memory");
    Counters end = read_counters();
    Counters delta;
    for (std::uint32_t i = 0; i < kNumCounters; i++) {
        delta.value[i] = end.value[i] - start.value[i];
    }
    return delta;
}

}  // namespace

void kernel_main() {
    std::uint32_t base_addr = get_arg(args::base_addr);
    std::uint32_t num_entries = get_arg(args::num_entries);
    std::uint32_t stride = get_arg(args::stride);
    std::uint32_t result_addr = get_arg(args::result_addr);
    std::uint32_t sweep_lines = get_arg(args::sweep_lines);

    WRITE_CSR(0x323, kEventDcacheBlocked);
    WRITE_CSR(0x324, kEventLongLatency);
    WRITE_CSR(0x325, kEventDcacheMiss);
    WRITE_CSR(0x326, kEventLoadUse);
    const std::uint64_t event3_readback = READ_CSR(0x323);

    Counters phases[kNumPhases];
    std::uint32_t sink = 0;

    auto read_cached = [&]() {
        for (std::uint32_t i = 0; i < num_entries; i++) {
            sink += *cached_word(base_addr + i * stride);
        }
    };

    phases[0] = measured([]() {});

    phases[1] = measured([&]() {
        for (std::uint32_t i = 0; i < num_entries; i++) {
            sink += *uncached_word(base_addr + i * stride);
        }
    });

    for (std::uint32_t i = 0; i < num_entries; i++) {
        invalidate_l1_dcache(base_addr + i * stride);
        invalidate_l2_cache_line(base_addr + i * stride);
    }
    phases[2] = measured(read_cached);

    // The lines are clean because they were only read, so dropping them from L1 D$ loses nothing.
    for (std::uint32_t i = 0; i < num_entries; i++) {
        invalidate_l1_dcache(base_addr + i * stride);
    }
    phases[3] = measured(read_cached);

    phases[4] = measured(read_cached);

    phases[5] = measured([&]() {
        for (std::uint32_t i = 0; i < num_entries; i++) {
            *uncached_word(base_addr + i * stride) = 0;
        }
    });

    phases[6] = measured([&]() {
        for (std::uint32_t i = 0; i < num_entries; i++) {
            std::uint32_t entry = base_addr + i * stride;
            *uncached_word(entry) = 0;
            invalidate_l2_cache_line(entry);
        }
    });

    volatile std::uint64_t* invalidate_reg = reinterpret_cast<volatile std::uint64_t*>(L2_INVALIDATE_ADDR);
    phases[7] = measured([&]() {
        for (std::uint32_t i = 0; i < num_entries; i++) {
            std::uint32_t entry = base_addr + i * stride;
            *uncached_word(entry) = 0;
            *invalidate_reg = entry;
        }
    });

    auto sweep = [&]() {
        for (std::uint32_t i = 0; i < sweep_lines; i++) {
            sink += *cached_word(base_addr + i * 64);
        }
    };
    for (std::uint32_t i = 0; i < sweep_lines; i++) {
        invalidate_l1_dcache(base_addr + i * 64);
        invalidate_l2_cache_line(base_addr + i * 64);
    }
    phases[8] = measured(sweep);
    phases[9] = measured(sweep);

    *uncached_word(result_addr) = static_cast<std::uint32_t>(event3_readback);
    *uncached_word(result_addr + 4) = sink;
    std::uint32_t out = result_addr + 16;
    for (std::uint32_t p = 0; p < kNumPhases; p++) {
        for (std::uint32_t c = 0; c < kNumCounters; c++) {
            *uncached_word(out) = static_cast<std::uint32_t>(phases[p].value[c]);
            *uncached_word(out + 4) = static_cast<std::uint32_t>(phases[p].value[c] >> 32);
            out += 8;
        }
    }
}
