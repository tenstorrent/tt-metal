// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads or plants remapper pair configs from a DM, through the same registers the firmware uses.
//   mode 0 (read):  copies every pair's ClientL config to `result_addr` (REMAP_NUM_PAIRS words).
//   mode 1 (plant): arms pairs `pair0` .. `pair2` with the given ClientR / ClientL configs, which is what a program
//                   killed before its remapper teardown would leave behind.
//   mode 2 (clear): disarms those three pairs again.

#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"
#include "internal/tt-2xx/quasar/overlay/overlay_addresses.h"
#include "internal/tt-2xx/quasar/overlay/remapper_common.hpp"

void kernel_main() {
    constexpr uint32_t mode = get_arg(args::mode);
    constexpr uint32_t pairs[] = {get_arg(args::pair0), get_arg(args::pair1), get_arg(args::pair2)};
    constexpr uint32_t client_r[] = {get_arg(args::client_r0), get_arg(args::client_r1), get_arg(args::client_r2)};
    constexpr uint32_t client_l[] = {get_arg(args::client_l0), get_arg(args::client_l1), get_arg(args::client_l2)};
    for (uint32_t pair : pairs) {
        ASSERT(pair < REMAP_NUM_PAIRS);
    }

    if constexpr (mode == 0) {
        uintptr_t result_addr = get_arg(args::result_addr);
        CoreLocalMem<uint32_t> result(result_addr);
        for (uint32_t pair = 0; pair < REMAP_NUM_PAIRS; pair++) {
            result[pair] = READ_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(pair));
        }
        flush_l2_cache_range(result_addr, REMAP_NUM_PAIRS * sizeof(uint32_t));
    } else {
        // ClientL carries the valid bits, so it is written last when arming and first when disarming.
        for (uint32_t i = 0; i < 3; i++) {
            if constexpr (mode == 1) {
                WRITE_REG32(REMAP_CLIENT_R_CONFIG_REG_ADDR32(pairs[i]), client_r[i]);
                WRITE_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(pairs[i]), client_l[i]);
            } else {
                WRITE_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(pairs[i]), 0u);
                WRITE_REG32(REMAP_CLIENT_R_CONFIG_REG_ADDR32(pairs[i]), 0u);
            }
        }
        // The writes are posted: read one back so they have landed before the kernel reports done.
        (void)READ_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(pairs[2]));
    }
}
