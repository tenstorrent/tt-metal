// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ckernel.h"
#include "debug/status.h"

// Wait cycles to create a ~1sec delay
#define WAIT_CYCLES 1200000000
// Flag address for syncing between cores (BRISC controls this)
#define SYNC_ADDR 409600

// Helper function to sync execution by forcing other riscs to wait for brisc, which in turn waits
// for a set number of cycles.
void hacky_sync(uint32_t sync_num) {
    volatile tt_l1_ptr uint32_t* sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SYNC_ADDR);
#if defined(COMPILE_FOR_BRISC)
    ckernel::wait(WAIT_CYCLES);
    *(sync_ptr) = sync_num;
#else
    while (*(sync_ptr) != sync_num) { ; }
#endif
}

/*
 * A test for the watcher waypointing feature.
*/
#if defined(COMPILE_FOR_BRISC) | defined(COMPILE_FOR_NCRISC)
void kernel_main() {
#else
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
#endif

    // Post a new waypoint with a delay after (to let the watcher poll it)
    hacky_sync(1);
    DEBUG_STATUS('A', 'A', 'A', 'A');
    hacky_sync(2);
    DEBUG_STATUS('B', 'B', 'B', 'B');
    hacky_sync(3);
    DEBUG_STATUS('C', 'C', 'C', 'C');
    hacky_sync(4);
#if defined(COMPILE_FOR_BRISC) | defined(COMPILE_FOR_NCRISC)
}
#else
}
}
#endif
