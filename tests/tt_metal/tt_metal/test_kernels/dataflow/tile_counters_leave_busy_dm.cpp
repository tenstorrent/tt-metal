// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Deliberately leaves DM-visible tile counters `tc0` and `tc1` of every Neo holding entries. The kernel
// binds no DFB, so the firmware's kernel-exit drain does not wait for those entries.

#include <cstdint>
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/overlay/llk_intf_api.hpp"

void kernel_main() {
    constexpr uint32_t num_neos = get_arg(args::num_neos);
    constexpr uint32_t capacity = get_arg(args::capacity);
    constexpr uint32_t num_entries = get_arg(args::num_entries);
    constexpr uint32_t tcs[] = {get_arg(args::tc0), get_arg(args::tc1)};
    static_assert(num_entries != 0 && num_entries <= capacity);

    for (uint32_t neo = 0; neo < num_neos; neo++) {
        for (uint32_t tc : tcs) {
            overlay::llk_intf_set_capacity(neo, tc, capacity);
            overlay::llk_intf_inc_posted(neo, tc, num_entries);
        }
    }
    // The writes are posted: read a counter back so they have landed before the kernel reports done.
    (void)overlay::llk_intf_get_occupancy(num_neos - 1, tcs[1]);
}
