// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// End to end check that a host write carrying the NOC snoop bit reaches a DM core's cached view.
//
// The kernel loads the slot through the cache, raises ARMED, then polls the slot through the cache the way the
// fast dispatch prefetcher polls its fetch queue. Polling keeps the line hot, so without snoop the cached copy
// cannot pick up the host's write until something invalidates it. Each poll also reads the slot uncached, which
// shows when the write has reached TL1.
//
// Results, written uncached at result_addr:
//   [0] cached value before ARMED, which must be the host's old value
//   [1] poll on which the uncached read first saw new_value, 0 if never
//   [2] poll on which the cached read first saw new_value, 0 if never
//   [3] last cached value read
//   [4] total polls

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

namespace {
inline volatile std::uint32_t* cached_word(std::uint32_t addr) {
    return reinterpret_cast<volatile std::uint32_t*>(static_cast<uintptr_t>(addr));
}
inline volatile std::uint32_t* uncached_word(std::uint32_t addr) {
    return reinterpret_cast<volatile std::uint32_t*>(static_cast<uintptr_t>(addr + MEM_L1_UNCACHED_BASE));
}
}  // namespace

void kernel_main() {
    std::uint32_t slot_addr = get_arg(args::slot_addr);
    std::uint32_t new_value = get_arg(args::new_value);
    std::uint32_t polls_after_tl1 = get_arg(args::polls_after_tl1);
    std::uint32_t max_polls = get_arg(args::max_polls);
    std::uint32_t flag_addr = get_arg(args::flag_addr);
    std::uint32_t result_addr = get_arg(args::result_addr);

    volatile std::uint32_t* cached = cached_word(slot_addr);
    volatile std::uint32_t* uncached = uncached_word(slot_addr);

    // Drop any copy left by an earlier run so the first load fills the line with the host's old value.
    invalidate_l1_dcache(slot_addr);
    invalidate_l2_cache_line(slot_addr);
    *uncached_word(result_addr) = *cached;
    __asm__ __volatile__("fence" ::: "memory");
    *uncached_word(flag_addr) = 1;

    std::uint32_t tl1_seen_at = 0;
    std::uint32_t cached_seen_at = 0;
    std::uint32_t last_cached = 0;
    std::uint32_t polls = 0;
    while (polls < max_polls) {
        polls++;
        last_cached = *cached;
        if (last_cached == new_value) {
            cached_seen_at = polls;
            break;
        }
        if (tl1_seen_at == 0 && *uncached == new_value) {
            tl1_seen_at = polls;
        }
        if (tl1_seen_at != 0 && polls - tl1_seen_at >= polls_after_tl1) {
            break;
        }
    }

    *uncached_word(result_addr + 4) = tl1_seen_at;
    *uncached_word(result_addr + 8) = cached_seen_at;
    *uncached_word(result_addr + 12) = last_cached;
    *uncached_word(result_addr + 16) = polls;
}
