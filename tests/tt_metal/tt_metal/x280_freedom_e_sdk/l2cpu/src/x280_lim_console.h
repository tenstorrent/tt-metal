// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// LIM console for freedom-metal on Blackhole L2CPU X280.

#ifndef X280_LIM_CONSOLE_H_
#define X280_LIM_CONSOLE_H_

#include <stdint.h>

// LIM layout from tt-llm-engine x280/include/x280.h (active FW links at 0x08001000).
#define X280_LIM_BASE 0x08000000UL
#define X280_ACTIVE_FW_LOAD_ADDR 0x08001000UL
#define X280_ACTIVE_FW_REGION_END 0x08120000UL

// Host loader polls this (LIM_BASE + 0x100000).
#define X280_SENTINEL_ADDR ((volatile uint64_t*)0x08100000UL)
#define X280_SENTINEL_VALUE 0xDEADBEEFCAFEBABEULL

// Fixed LIM address for host NOC readback (4 KiB above sentinel).
#define X280_CONSOLE_ADDR 0x08101000UL
#define X280_CONSOLE_MAGIC 0x2800C0FFEE000280ULL
#define X280_CONSOLE_CAPACITY 3072

struct x280_console {
    uint64_t magic;
    uint32_t len;
    uint32_t dropped;
    char data[X280_CONSOLE_CAPACITY];
};

#define X280_CONSOLE ((volatile struct x280_console*)X280_CONSOLE_ADDR)

void x280_console_init(void);
uint32_t x280_console_length(void);

#endif  // X280_LIM_CONSOLE_H_
