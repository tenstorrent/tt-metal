// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for testing sync events on TRISC.
// Tests CB APIs (all architectures) and ckernel::Semaphore (Quasar only).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#ifdef ARCH_QUASAR
#include "api/compute/experimental/semaphore.h"
#endif

#ifndef DELAY_CYCLES
#define DELAY_CYCLES 10000
#endif

#ifndef CB_ID
#define CB_ID 0
#endif

FORCE_INLINE void delay_cycles(uint32_t cycles) {
    for (uint32_t i = 0; i < cycles; i++) {
        asm volatile("nop");
    }
}

void kernel_main() {
    uint32_t api_id = get_arg_val<uint32_t>(0);

    switch (api_id) {
        // =====================================================================
        // CB APIs on TRISC (all architectures)
        // =====================================================================

        // TRISC CB Producer: reserve + delay + push
        case 100: {
            cb_reserve_back(CB_ID, 1);  // SYNC-CB-RESERVE (instant)
            delay_cycles(DELAY_CYCLES);
            cb_push_back(CB_ID, 1);  // SYNC-CB-PUSH
            break;
        }

        // TRISC CB Consumer: wait
        case 101: {
            cb_wait_front(CB_ID, 1);  // SYNC-CB-WAIT (stalls ~DELAY_CYCLES)
            break;
        }

        // TRISC CB Consumer: wait + pop
        case 102: {
            cb_wait_front(CB_ID, 1);  // SYNC-CB-WAIT
            cb_pop_front(CB_ID, 1);   // SYNC-CB-POP
            break;
        }

        // TRISC CB Consumer: delay then pop (for blocking reserve test)
        case 103: {
            cb_wait_front(CB_ID, 1);  // SYNC-CB-WAIT (instant, data ready)
            delay_cycles(DELAY_CYCLES);
            cb_pop_front(CB_ID, 1);  // SYNC-CB-POP (releases producer's reserve)
            break;
        }

        // TRISC CB Producer: push to fill, then reserve (blocks)
        case 104: {
            cb_reserve_back(CB_ID, 1);  // SYNC-CB-RESERVE (instant)
            cb_push_back(CB_ID, 1);     // SYNC-CB-PUSH (fills CB)
            cb_reserve_back(CB_ID, 1);  // SYNC-CB-RESERVE (stalls ~DELAY_CYCLES)
            break;
        }

#ifdef ARCH_QUASAR
        // =====================================================================
        // ckernel::Semaphore APIs (Quasar only)
        // =====================================================================
        case 13: {
            ckernel::Semaphore sem(0);
            delay_cycles(DELAY_CYCLES);
            sem.up(1);  // SYNC-SEM-SET
            break;
        }

        case 14: {
            ckernel::Semaphore sem(0);
            sem.down(1);  // SYNC-SEM-WAIT + SYNC-SEM-SET
            break;
        }

        case 15: {
            ckernel::Semaphore sem(0);
            sem.wait(1);  // SYNC-SEM-WAIT
            break;
        }

        case 16: {
            ckernel::Semaphore sem(0);
            sem.wait_min(1);  // SYNC-SEM-WAIT
            break;
        }

        case 17: {
            ckernel::Semaphore sem(0);
            delay_cycles(DELAY_CYCLES);
            sem.set(1);  // SYNC-SEM-SET
            break;
        }
#endif

        default: break;
    }
}
