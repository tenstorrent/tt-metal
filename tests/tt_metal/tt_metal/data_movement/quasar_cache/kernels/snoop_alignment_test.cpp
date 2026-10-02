// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Checks which snooped host write shapes keep a DM core's cached copy of a line correct. The host fills
// num_lines 64B lines with a known pattern, the kernel caches them, and the host then issues one snooped
// write per line with a different length and offset.
//
// Handshake is the same as snoop_coherence_test: ARMED at flag_addr, HOST_DONE at flag_addr + 16.
//
// For each line the kernel dumps its 16 words three times at result_addr + line * 192:
//   words  0..15 cached read after the host write
//   words 16..31 uncached read, the TL1 contents
//   words 32..47 cached read after invalidating the L2 line

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

namespace {
constexpr std::uint32_t kWordsPerLine = 16;

inline volatile std::uint32_t* cached_word(std::uint32_t addr) {
    return reinterpret_cast<volatile std::uint32_t*>(static_cast<uintptr_t>(addr));
}
inline volatile std::uint32_t* uncached_word(std::uint32_t addr) {
    return reinterpret_cast<volatile std::uint32_t*>(static_cast<uintptr_t>(addr + MEM_L1_UNCACHED_BASE));
}
}  // namespace

void kernel_main() {
    std::uint32_t base_addr = get_arg(args::base_addr);
    std::uint32_t num_lines = get_arg(args::num_lines);
    std::uint32_t flag_addr = get_arg(args::flag_addr);
    std::uint32_t result_addr = get_arg(args::result_addr);

    volatile std::uint32_t* armed = uncached_word(flag_addr);
    volatile std::uint32_t* host_done = uncached_word(flag_addr + 16);

    std::uint32_t sink = 0;
    for (std::uint32_t line = 0; line < num_lines; line++) {
        for (std::uint32_t w = 0; w < kWordsPerLine; w++) {
            sink += *cached_word(base_addr + line * 64 + w * 4);
        }
    }
    __asm__ __volatile__("fence" ::: "memory");
    *armed = 1;

    while (*host_done == 0) {
    }

    for (std::uint32_t line = 0; line < num_lines; line++) {
        const std::uint32_t src = base_addr + line * 64;
        const std::uint32_t dst = result_addr + line * kWordsPerLine * 3 * 4;
        for (std::uint32_t w = 0; w < kWordsPerLine; w++) {
            *uncached_word(dst + w * 4) = *cached_word(src + w * 4);
            *uncached_word(dst + (kWordsPerLine + w) * 4) = *uncached_word(src + w * 4);
        }
    }

    for (std::uint32_t line = 0; line < num_lines; line++) {
        const std::uint32_t src = base_addr + line * 64;
        const std::uint32_t dst = result_addr + line * kWordsPerLine * 3 * 4;
        invalidate_l2_cache_line(src);
        for (std::uint32_t w = 0; w < kWordsPerLine; w++) {
            *uncached_word(dst + (2 * kWordsPerLine + w) * 4) = *cached_word(src + w * 4);
        }
    }

    // Keeps the warm-up loads from being optimized away.
    *uncached_word(result_addr + num_lines * kWordsPerLine * 3 * 4) = sink;
}
