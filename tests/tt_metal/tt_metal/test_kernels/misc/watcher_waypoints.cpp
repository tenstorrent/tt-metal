// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/waypoint.h"
#include "debug/ring_buffer.h"

// Helper function to sync execution by forcing other riscs to wait for brisc, which in turn waits
// for a set number of cycles.
void hacky_sync(uint32_t sync_num, uint32_t wait_cycles, uint32_t sync_addr) {
    volatile tt_l1_ptr uint32_t* sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_addr);
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
    riscv_wait(wait_cycles);
#if defined(COMPILE_FOR_BRISC)
    // ERISC doesn't have to sync, doesn't have the L1 buffer
    *(sync_ptr) = sync_num;
#endif
#else
    while (*(sync_ptr) != sync_num) { ; }
#endif
}

/*
 * A test for the watcher waypointing feature.
*/
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
void kernel_main() {
#else
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
#endif
    uint32_t sync_wait_cycles = get_arg_val<uint32_t>(0);
    uint32_t sync_address     = get_arg_val<uint32_t>(1);
    WATCHER_RING_BUFFER_PUSH(sync_wait_cycles);

    // Post a new waypoint with a delay after (to let the watcher poll it)
    hacky_sync(1, sync_wait_cycles, sync_address);
    WAYPOINT("AAAA");
    hacky_sync(2, sync_wait_cycles, sync_address);
    //WAYPOINT("BBBB");
    hacky_sync(3, sync_wait_cycles, sync_address);
    //WAYPOINT("CCCC");
    hacky_sync(4, sync_wait_cycles, sync_address);
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
}
#else
}
}
#endif
