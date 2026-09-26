// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ISS stand-in for the idle-FW ping-pong exit. There is no resident idle
// image at 0x08000000, so hart 0 HTIF-exits and helpers park.

#ifndef X280_RT_IDLE_H
#define X280_RT_IDLE_H

#include <stdint.h>

#include "htif.h"

extern volatile uint64_t tohost;

static inline __attribute__((noreturn)) void x280_helper_to_idle(void) {
    for (;;) {
        __asm__ volatile("wfi");
    }
}

static inline __attribute__((noreturn)) void x280_hart0_to_idle(void) {
    *SENTINEL_ADDR = SENTINEL_VALUE;
    __asm__ volatile("fence ow, ow");
    *(volatile uint64_t*)X280_BOOT_PHASE_ADDR = X280_BOOT_PHASE_RETURNED_TO_IDLE;
    __asm__ volatile("fence ow, ow");
    htif_puts("atomic_bench: hart0 returned to idle (ISS)\n");
    while (tohost) {
    }
    tohost = 1; /* HTIF exit 0 */
    for (;;) {
        __asm__ volatile("wfi");
    }
}

#endif
