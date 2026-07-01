// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/waypoint.h"
#include "risc_common.h"

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Post waypoint
    WAYPOINT("AAAA");

#if defined(COMPILE_FOR_ERISC) && !defined(COMPILE_FOR_IDLE_ERISC)
    // Active ERISC: timed wait to give watcher time to capture waypoint,
    // then exit to resume tunneling duties. Can't block for sync signal.
    uint32_t delay_cycles = get_common_arg_val<uint32_t>(0);
    riscv_wait(delay_cycles);
#else
    // idle ERISC: block until host signals
    uint32_t sync_flag_addr = get_common_arg_val<uint32_t>(0);
    volatile uint32_t* sync_flag =
        reinterpret_cast<volatile uint32_t*>(sync_flag_addr);

    while (*sync_flag != 1) {
        invalidate_l1_cache();
    }
#endif
}
