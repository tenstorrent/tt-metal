// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

#include <cstdint>

// Echoes the runtime argument this launch's TRISC read into an L1 slot, one slot per TRISC thread, so the host
// can tell whether a TRISC saw this launch's argument or a stale one from an earlier launch.
void kernel_main() {
    const uint32_t result_l1_addr = get_arg(args::result_l1_addr);
    const uint32_t value = get_arg(args::value);

#if defined(TRISC_UNPACK)
    constexpr uint32_t result_slot = 0;
#elif defined(TRISC_MATH)
    constexpr uint32_t result_slot = 1;
#elif defined(TRISC_PACK)
    constexpr uint32_t result_slot = 2;
#endif
#if defined(TRISC_UNPACK) || defined(TRISC_MATH) || defined(TRISC_PACK)
    volatile tt_l1_ptr uint32_t* const out =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_l1_addr + MEM_L1_UNCACHED_BASE);
    out[result_slot] = value;
#endif
}
