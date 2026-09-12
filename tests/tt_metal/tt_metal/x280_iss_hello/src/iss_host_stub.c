// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Spike has no host/NOC. Publish the same LIM config block that
// experiment_atomic_bench.py would write, so hart 0's AB_CONFIG_READY wait
// completes.

#include <stdint.h>

#define LIM_BASE 0x08000000UL
#define ATOMIC_BASE (LIM_BASE + 0x00160000UL)

#define AB_STOP_ADDR ((volatile uint32_t*)(ATOMIC_BASE + 0x040UL))
#define AB_CONFIG_READY_ADDR ((volatile uint32_t*)(ATOMIC_BASE + 0x080UL))
#define AB_ITERATIONS_ADDR ((volatile uint64_t*)(ATOMIC_BASE + 0x100UL))
#define AB_OP_MODE_ADDR ((volatile uint32_t*)(ATOMIC_BASE + 0x140UL))
#define AB_PHASE_MASK_ADDR ((volatile uint32_t*)(ATOMIC_BASE + 0x180UL))
#define AB_COUNTER_ADDR_ADDR ((volatile uint64_t*)(ATOMIC_BASE + 0x1C0UL))

#define AB_OP_AMOADD_D 1u
#define AB_CONFIG_READY_VALUE 0x000C0FFEu

#ifndef AB_ISS_ITERS
#define AB_ISS_ITERS 1000ULL
#endif

void iss_publish_config(void) {
    *AB_STOP_ADDR = 0;
    *AB_ITERATIONS_ADDR = AB_ISS_ITERS;
    *AB_OP_MODE_ADDR = AB_OP_AMOADD_D;
    *AB_PHASE_MASK_ADDR = 0x1u; /* 1-hart phase; multi-hart is a follow-up */
    *AB_COUNTER_ADDR_ADDR = ATOMIC_BASE + 0x2000UL;
    __asm__ volatile("fence ow, ow");
    *AB_CONFIG_READY_ADDR = AB_CONFIG_READY_VALUE;
    __asm__ volatile("fence ow, ow");
}
