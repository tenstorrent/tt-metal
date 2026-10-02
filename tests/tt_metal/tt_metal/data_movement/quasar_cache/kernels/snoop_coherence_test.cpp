// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Checks whether a host write that lands in TL1 while a DM core holds the line in its caches is visible
// through the cached view, which depends on the write carrying the NOC snoop bit.
//
// Handshake: the kernel caches every entry, raises ARMED, and spins on HOST_DONE. The host writes new
// entry values, then raises HOST_DONE. Each flag lives in its own 16B slot and is accessed uncached.
//
// Per entry the kernel records four words at result_addr + 16 * i:
//   [0] cached read right after the host write
//   [1] uncached read, the TL1 ground truth
//   [2] cached read after invalidating only the L2 line
//   [3] cached read after invalidating both the L1 D$ and L2 lines

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
    std::uint32_t base_addr = get_arg(args::base_addr);
    std::uint32_t num_entries = get_arg(args::num_entries);
    std::uint32_t stride = get_arg(args::stride);
    std::uint32_t flag_addr = get_arg(args::flag_addr);
    std::uint32_t result_addr = get_arg(args::result_addr);

    volatile std::uint32_t* armed = uncached_word(flag_addr);
    volatile std::uint32_t* host_done = uncached_word(flag_addr + 16);

    std::uint32_t sink = 0;
    for (std::uint32_t i = 0; i < num_entries; i++) {
        sink += *cached_word(base_addr + i * stride);
    }
    __asm__ __volatile__("fence" ::: "memory");
    *armed = 1;

    while (*host_done == 0) {
    }

    for (std::uint32_t i = 0; i < num_entries; i++) {
        std::uint32_t entry = base_addr + i * stride;
        std::uint32_t result = result_addr + i * 16;
        *uncached_word(result) = *cached_word(entry);
        *uncached_word(result + 4) = *uncached_word(entry);
    }

    for (std::uint32_t i = 0; i < num_entries; i++) {
        std::uint32_t entry = base_addr + i * stride;
        invalidate_l2_cache_line(entry);
        *uncached_word(result_addr + i * 16 + 8) = *cached_word(entry);
    }

    for (std::uint32_t i = 0; i < num_entries; i++) {
        std::uint32_t entry = base_addr + i * stride;
        invalidate_l1_dcache(entry);
        invalidate_l2_cache_line(entry);
        *uncached_word(result_addr + i * 16 + 12) = *cached_word(entry);
    }

    // Keeps the warm-up loads from being optimized away.
    *uncached_word(result_addr + num_entries * 16) = sink;
}
