// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unified sync events test kernel.
// Tests one API at a time based on API_ID runtime arg.
// Producer (BRISC) delays DELAY_CYCLES before signaling.
// Consumer (NCRISC) waits - duration should match delay.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"

#ifndef DELAY_CYCLES
#define DELAY_CYCLES 10000
#endif

#ifndef CB_ID
#define CB_ID 0
#endif

// API IDs:
// Raw CB APIs:
//   0  = cb_reserve_back + cb_push_back (BRISC producer)
//   1  = cb_wait_front (NCRISC consumer)
//   7  = cb_reserve_back blocking (BRISC producer)
//   8  = cb_wait_front + cb_pop_front (NCRISC consumer)
//
// Raw Semaphore APIs:
//   2  = noc_semaphore_set (BRISC producer)
//   3  = noc_semaphore_wait (NCRISC consumer)
//   4  = noc_semaphore_inc remote (BRISC producer)
//   5  = noc_semaphore_wait for remote (NCRISC consumer)
//   6  = noc_semaphore_wait_min (NCRISC consumer)
//   11 = noc_semaphore_inc_multicast (BRISC producer)
//   12 = noc_semaphore_wait for multicast (NCRISC consumer)
//   13 = noc_semaphore_set_multicast (BRISC producer)
//   14 = noc_semaphore_wait for set_multicast (NCRISC consumer)
//
// Semaphore class APIs:
//   20 = Semaphore::set() (BRISC producer)
//   21 = Semaphore::wait() (NCRISC consumer)
//   22 = Semaphore::up() local (BRISC producer)
//   23 = Semaphore::wait_min() (NCRISC consumer)
//   24 = Semaphore::up() remote (BRISC producer)
//   25 = Semaphore::wait() for remote up (NCRISC consumer)
//   26 = Semaphore::down() - wait+decrement (NCRISC, paired with set)
//   27 = Semaphore::set_multicast() (BRISC producer)
//   28 = Semaphore::wait() for set_multicast (NCRISC consumer)
//   29 = Semaphore::inc_multicast() (BRISC producer)
//   30 = Semaphore::wait() for inc_multicast (NCRISC consumer)

FORCE_INLINE void delay_cycles(uint32_t cycles) {
    for (uint32_t i = 0; i < cycles; i++) {
        asm volatile("nop");
    }
}

void kernel_main() {
    uint32_t api_id = get_arg_val<uint32_t>(0);
    uint32_t sem_id = get_arg_val<uint32_t>(1);
    uint32_t remote_noc_x = get_arg_val<uint32_t>(2);
    uint32_t remote_noc_y = get_arg_val<uint32_t>(3);
    uint32_t remote_sem_id = get_arg_val<uint32_t>(4);
    uint32_t use_remote_sem_id = get_arg_val<uint32_t>(5);

    // Convert semaphore IDs to addresses
    uint32_t sem_addr = get_semaphore(sem_id);
    uint32_t remote_sem_addr = use_remote_sem_id ? get_semaphore(remote_sem_id) : 0;
    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    switch (api_id) {
        // =====================================================================
        // CB Producer (BRISC): delay then push
        // =====================================================================
        case 0: {
            cb_reserve_back(CB_ID, 1);  // SYNC-CB-RESERVE (instant - CB empty)
            delay_cycles(DELAY_CYCLES);
            cb_push_back(CB_ID, 1);  // SYNC-CB-PUSH (releases consumer's wait)
            break;
        }

        // =====================================================================
        // CB Consumer (NCRISC): wait
        // =====================================================================
        case 1: {
            cb_wait_front(CB_ID, 1);  // SYNC-CB-WAIT (stalls ~DELAY_CYCLES)
            break;
        }

        // =====================================================================
        // Semaphore Local Producer (BRISC): delay then set
        // =====================================================================
        case 2: {
            delay_cycles(DELAY_CYCLES);
            noc_semaphore_set(sem_ptr, 1);  // SYNC-SEM-SET
            break;
        }

        // =====================================================================
        // Semaphore Local Consumer (NCRISC): wait
        // =====================================================================
        case 3: {
            noc_semaphore_wait(sem_ptr, 1);  // SYNC-SEM-WAIT (stalls ~DELAY_CYCLES)
            break;
        }

        // =====================================================================
        // Semaphore Remote Producer (BRISC): delay then inc
        // =====================================================================
        case 4: {
            if (use_remote_sem_id) {
                uint64_t remote_noc_addr = get_noc_addr(remote_noc_x, remote_noc_y, remote_sem_addr);
                delay_cycles(DELAY_CYCLES);
                noc_semaphore_inc(remote_noc_addr, 1);  // SYNC-SEM-SET-REMOTE
                noc_async_atomic_barrier();
            }
            break;
        }

        // =====================================================================
        // Semaphore Remote Consumer (NCRISC): wait for remote inc
        // =====================================================================
        case 5: {
            noc_semaphore_wait(sem_ptr, 1);  // SYNC-SEM-WAIT (stalls ~DELAY_CYCLES)
            break;
        }

        // =====================================================================
        // noc_semaphore_wait_min Consumer (NCRISC)
        // =====================================================================
        case 6: {
            noc_semaphore_wait_min(sem_ptr, 1);  // SYNC-SEM-WAIT
            break;
        }

        // =====================================================================
        // CB Reserve Producer (BRISC): push to fill CB, then reserve (blocks)
        // =====================================================================
        case 7: {
            // Reserve + push to fill the 1-page CB
            cb_reserve_back(CB_ID, 1);  // SYNC-CB-RESERVE (instant - CB empty)
            cb_push_back(CB_ID, 1);     // SYNC-CB-PUSH (fills CB)
            // Now reserve blocks until consumer pops
            cb_reserve_back(CB_ID, 1);  // SYNC-CB-RESERVE (stalls ~DELAY_CYCLES)
            break;
        }

        // =====================================================================
        // CB Reserve Consumer (NCRISC): delay then pop (releases producer's reserve)
        // =====================================================================
        case 8: {
            // Wait for producer to push first
            cb_wait_front(CB_ID, 1);  // SYNC-CB-WAIT (should be instant, data already pushed)
            // Delay before popping - this is the delay producer's reserve waits for
            delay_cycles(DELAY_CYCLES);
            cb_pop_front(CB_ID, 1);  // SYNC-CB-POP (releases producer's reserve)
            break;
        }

        // =====================================================================
        // Raw API: noc_semaphore_inc_multicast (BRISC producer)
        // =====================================================================
        case 11: {
            if (use_remote_sem_id) {
                uint64_t mcast_addr =
                    get_noc_multicast_addr(remote_noc_x, remote_noc_y, remote_noc_x, remote_noc_y, remote_sem_addr);
                delay_cycles(DELAY_CYCLES);
                noc_semaphore_inc_multicast(mcast_addr, 1, 1);  // SYNC-SEM-SET-REMOTE
                noc_async_atomic_barrier();
            }
            break;
        }

        // =====================================================================
        // Raw API: noc_semaphore_wait for multicast (NCRISC consumer)
        // =====================================================================
        case 12: {
            noc_semaphore_wait(sem_ptr, 1);  // SYNC-SEM-WAIT
            break;
        }

        // =====================================================================
        // Raw API: noc_semaphore_set_multicast (BRISC producer)
        // =====================================================================
        case 13: {
            if (use_remote_sem_id) {
                uint64_t mcast_addr =
                    get_noc_multicast_addr(remote_noc_x, remote_noc_y, remote_noc_x, remote_noc_y, remote_sem_addr);
                // Set local value first
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr) = 1;
                delay_cycles(DELAY_CYCLES);
                noc_semaphore_set_multicast(sem_addr, mcast_addr, 1);  // SYNC-SEM-SET-REMOTE
                noc_async_write_barrier();
            }
            break;
        }

        // =====================================================================
        // Raw API: noc_semaphore_wait for set_multicast (NCRISC consumer)
        // =====================================================================
        case 14: {
            noc_semaphore_wait(sem_ptr, 1);  // SYNC-SEM-WAIT
            break;
        }

        // =====================================================================
        // Semaphore class: set() (BRISC producer)
        // =====================================================================
        case 20: {
            Semaphore<> sem(0);
            delay_cycles(DELAY_CYCLES);
            sem.set(1);  // SYNC-SEM-SET
            break;
        }

        // =====================================================================
        // Semaphore class: wait() (NCRISC consumer)
        // =====================================================================
        case 21: {
            Semaphore<> sem(0);
            sem.wait(1);  // SYNC-SEM-WAIT
            break;
        }

        // =====================================================================
        // Semaphore class: up() local (BRISC producer)
        // =====================================================================
        case 22: {
            Semaphore<> sem(0);
            delay_cycles(DELAY_CYCLES);
            sem.up(1);  // SYNC-SEM-SET
            break;
        }

        // =====================================================================
        // Semaphore class: wait_min() (NCRISC consumer)
        // =====================================================================
        case 23: {
            Semaphore<> sem(0);
            sem.wait_min(1);  // SYNC-SEM-WAIT
            break;
        }

        // =====================================================================
        // Semaphore class: up() remote (BRISC producer)
        // =====================================================================
        case 24: {
            if (use_remote_sem_id) {
                Semaphore<> sem(0);
                Noc noc(0);
                delay_cycles(DELAY_CYCLES);
                sem.up(noc, remote_noc_x, remote_noc_y, 1);  // SYNC-SEM-SET-REMOTE (via noc_semaphore_inc)
                noc_async_atomic_barrier();
            }
            break;
        }

        // =====================================================================
        // Semaphore class: wait() for remote up (NCRISC consumer)
        // =====================================================================
        case 25: {
            Semaphore<> sem(0);
            sem.wait(1);  // SYNC-SEM-WAIT
            break;
        }

        // =====================================================================
        // Semaphore class: down() - wait then decrement (NCRISC consumer)
        // Paired with set() producer
        // =====================================================================
        case 26: {
            Semaphore<> sem(0);
            sem.down(1);  // SYNC-SEM-WAIT + SYNC-SEM-SET
            break;
        }

        // =====================================================================
        // Semaphore class: set_multicast() (BRISC producer)
        // =====================================================================
        case 27: {
            if (use_remote_sem_id) {
                Semaphore<> sem(0);
                Noc noc(0);
                sem.set(1);  // Set local value first
                delay_cycles(DELAY_CYCLES);
                sem.set_multicast(
                    noc, remote_noc_x, remote_noc_y, remote_noc_x, remote_noc_y, 1);  // SYNC-SEM-SET-REMOTE
                noc_async_write_barrier();
            }
            break;
        }

        // =====================================================================
        // Semaphore class: wait() for set_multicast (NCRISC consumer)
        // =====================================================================
        case 28: {
            Semaphore<> sem(0);
            sem.wait(1);  // SYNC-SEM-WAIT
            break;
        }

        // =====================================================================
        // Semaphore class: inc_multicast() (BRISC producer)
        // =====================================================================
        case 29: {
            if (use_remote_sem_id) {
                Semaphore<> sem(0);
                Noc noc(0);
                delay_cycles(DELAY_CYCLES);
                sem.inc_multicast(
                    noc, remote_noc_x, remote_noc_y, remote_noc_x, remote_noc_y, 1, 1);  // SYNC-SEM-SET-REMOTE
                noc_async_atomic_barrier();
            }
            break;
        }

        // =====================================================================
        // Semaphore class: wait() for inc_multicast (NCRISC consumer)
        // =====================================================================
        case 30: {
            Semaphore<> sem(0);
            sem.wait(1);  // SYNC-SEM-WAIT
            break;
        }

        default: break;
    }
}
