// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// CCE stop-gap build POC kernel: one command-buffer copy, one Ethernet packet
// and a two-thread handoff, on metal's device names only.
//
// Runtime args (flat tile-SRAM offsets): 0 source, 1 NOC destination,
// 2 Ethernet destination (in the link peer's SRAM), 3 result words.
// Common args: 0 pattern seed. Compile-time args: pattern_words.
// Thread 0 moves the pattern; thread 1 verifies both destinations and writes
// result[0] = words still wrong through the uncached view after polling, and
// result[1] = words wrong through the cached view after invalidating.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/kernel_thread_globals.h"
#include "dev_mem_map.h"
#include "internal/ethernet/tunneling.h"
#include "risc_common.h"

namespace {

constexpr uint32_t kPatternWords = get_named_ct_arg("pattern_words");
constexpr uint32_t kPatternBytes = kPatternWords * sizeof(uint32_t);
constexpr uint32_t kEthWordBytes = 16;
static_assert(
    kPatternWords > 0 && kPatternBytes % kEthWordBytes == 0, "pattern_words must be a positive multiple of 4");

constexpr uint32_t kEthTxq = 0;
constexpr uint32_t kArrivalPolls = 100000;

struct PocRun {
    uint32_t src;
    uint32_t noc_dst;
    uint32_t eth_dst;
    uint32_t results;
    uint32_t seed;
};

uint32_t pattern_word(uint32_t seed, uint32_t i) { return seed + i * 0x01000193u; }

volatile uint32_t* cached(uint32_t addr) { return reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(addr)); }

volatile uint32_t* uncached(uint32_t addr) {
    return reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(addr) + MEM_L1_UNCACHED_BASE);
}

uint32_t count_mismatches(const PocRun& run, volatile uint32_t* buf) {
    uint32_t bad = 0;
    for (uint32_t i = 0; i < kPatternWords; i++) {
        bad += buf[i] != pattern_word(run.seed, i);
    }
    return bad;
}

// Payload writes are not snooped, and one Ethernet packet can land as several
// writes in no stated order: poll the whole buffer until it matches.
uint32_t poll_until_match(const PocRun& run, uint32_t addr) {
    uint32_t bad = count_mismatches(run, uncached(addr));
    for (uint32_t i = 0; bad != 0 && i < kArrivalPolls; i++) {
        bad = count_mismatches(run, uncached(addr));
    }
    return bad;
}

void move_pattern(const PocRun& run) {
    volatile uint32_t* src = cached(run.src);
    for (uint32_t i = 0; i < kPatternWords; i++) {
        src[i] = pattern_word(run.seed, i);
    }
    // The iDMA and Ethernet engines read SRAM, not the CPU caches.
    flush_l2_cache_range(run.src, kPatternBytes);

    noc_async_write(run.src, get_noc_addr(run.noc_dst), kPatternBytes);
    noc_async_write_barrier();

    internal_::eth_send_packet(
        kEthTxq, run.src / kEthWordBytes, run.eth_dst / kEthWordBytes, kPatternBytes / kEthWordBytes);
    sync_threads();
}

void verify_pattern(const PocRun& run) {
    // Pull both destinations into the cache first, so a missing invalidate
    // leaves stale lines behind.
    uint32_t stale = count_mismatches(run, cached(run.noc_dst)) + count_mismatches(run, cached(run.eth_dst));
    (void)stale;

    sync_threads();

    uint32_t uncached_bad = poll_until_match(run, run.noc_dst) + poll_until_match(run, run.eth_dst);

    invalidate_l2_cache_range(run.noc_dst, kPatternBytes);
    invalidate_l2_cache_range(run.eth_dst, kPatternBytes);
    invalidate_l1_cache();
    uint32_t cached_bad = count_mismatches(run, cached(run.noc_dst)) + count_mismatches(run, cached(run.eth_dst));

    uncached(run.results)[0] = uncached_bad;
    uncached(run.results)[1] = cached_bad;
}

}  // namespace

void kernel_main() {
    const PocRun run = {
        .src = get_arg_val<uint32_t>(0),
        .noc_dst = get_arg_val<uint32_t>(1),
        .eth_dst = get_arg_val<uint32_t>(2),
        .results = get_arg_val<uint32_t>(3),
        .seed = get_common_arg_val<uint32_t>(0),
    };

    if (get_my_thread_id() == 0) {
        move_pattern(run);
    } else {
        verify_pattern(run);
    }
}
