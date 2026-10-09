// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads or plants remapper ClientL config registers from a DM, through the same addresses the firmware uses.
//   mode 0 (read):  copies every pair's ClientL config to `result_addr` (REMAP_NUM_PAIRS words).
//   mode 1 (plant): writes `stale_value` to the ClientL config of `stale_pair_a` and `stale_pair_b`, which is
//                   what a program killed before its remapper teardown would leave behind.
//   mode 2 (clear): zeroes the ClientL config of those two pairs again.

#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"
#include "internal/tt-2xx/quasar/overlay/overlay_addresses.h"
#include "internal/tt-2xx/quasar/overlay/remapper_common.hpp"

void kernel_main() {
    constexpr uint32_t mode = get_arg(args::mode);
    constexpr uint32_t stale_pair_a = get_arg(args::stale_pair_a);
    constexpr uint32_t stale_pair_b = get_arg(args::stale_pair_b);
    constexpr uint32_t stale_value = get_arg(args::stale_value);
    static_assert(stale_pair_a < REMAP_NUM_PAIRS && stale_pair_b < REMAP_NUM_PAIRS);

    if constexpr (mode == 0) {
        uintptr_t result_addr = get_arg(args::result_addr);
        CoreLocalMem<uint32_t> result(result_addr);
        for (uint32_t pair = 0; pair < REMAP_NUM_PAIRS; pair++) {
            result[pair] = READ_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(pair));
        }
        flush_l2_cache_range(result_addr, REMAP_NUM_PAIRS * sizeof(uint32_t));
    } else {
        constexpr uint32_t value = mode == 1 ? stale_value : 0u;
        WRITE_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(stale_pair_a), value);
        WRITE_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(stale_pair_b), value);
        asm volatile("fence" ::: "memory");
    }
}
