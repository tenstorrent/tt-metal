// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/waypoint.h"
#include "risc_common.h"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#else
#include "api/dataflow/dataflow_api.h"
#endif

#if defined(ARCH_QUASAR)
#include "internal/tt-2xx/quasar/overlay/overlay_addresses.h"
#endif

void kernel_main() {
    // Post waypoint
    WAYPOINT("AAAA");

#if defined(COMPILE_FOR_ERISC) && !defined(COMPILE_FOR_IDLE_ERISC)
    // Active ERISC: timed wait to give watcher time to capture waypoint,
    // then exit to resume tunneling duties. Can't block for sync signal.
    uint32_t delay_cycles = get_common_arg_val<uint32_t>(0);
    riscv_wait(delay_cycles);
#else
    // TENSIX / idle ERISC: block until host signals
    uint32_t sync_flag_addr = get_common_arg_val<uint32_t>(0);
    volatile uint32_t* sync_flag =
        reinterpret_cast<volatile uint32_t*>(sync_flag_addr);

    while (*sync_flag != 1) {
#if defined(ARCH_QUASAR)
        // On Quasar DM cores, we must invalidate L2 to see updated host writes in L1
        // TODO: Use invalidate_l2_cache_line() once PR #38124 is merged
        __asm__ __volatile__("fence" ::: "memory");
        *reinterpret_cast<volatile uint64_t*>(L2_INVALIDATE_ADDR) = sync_flag_addr;
        __asm__ __volatile__("fence" ::: "memory");
#else
        invalidate_l1_cache();
#endif
    }
#endif
}
