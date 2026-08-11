// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// LIM-resident console for freedom-metal on a Blackhole L2CPU X280.

#ifndef X280_LIM_CONSOLE_H_
#define X280_LIM_CONSOLE_H_

#include <stdint.h>

// --- LIM layout, from tt-llm-engine x280/include/x280.h ----------------------
// The X280 view of L3 LIM is [0x08000000, 0x081E0000), 1.875 MiB, split as:
//   [0x08000000, 0x08001000)  resident idle FW  (X280_IDLE_FW_LOAD_ADDR)
//   [0x08001000, 0x08120000)  active FW         (X280_ACTIVE_FW_LOAD_ADDR)
//   [0x08130000, 0x081E0000)  mailboxes / staging / FIFO pool
// This program is an active FW: it links at X280_ACTIVE_FW_LOAD_ADDR.
#define X280_LIM_BASE 0x08000000UL
#define X280_ACTIVE_FW_LOAD_ADDR 0x08001000UL
#define X280_ACTIVE_FW_REGION_END 0x08120000UL

// The sentinel tt-llm-engine's host loader already polls to confirm a firmware
// ran (x280/host/loader.py: "our LIM-mode sentinel lives at LIM_BASE +
// 0x100000"). Writing it means the existing tooling can observe this program
// without knowing anything about it.
#define X280_SENTINEL_ADDR ((volatile uint64_t*)0x08100000UL)
#define X280_SENTINEL_VALUE 0xDEADBEEFCAFEBABEULL

// Console block, pinned at a fixed absolute LIM address so the host can read it
// with a plain NOC read and no ELF symbol lookup -- the same idiom x280.h uses
// for every other host-visible structure. 4 KiB above the sentinel, well clear
// of both this program's image (~30 KiB from 0x08001000) and the mailbox pool.
#define X280_CONSOLE_ADDR 0x08101000UL
#define X280_CONSOLE_MAGIC 0x2800C0FFEE000280ULL
#define X280_CONSOLE_CAPACITY 3072

// Header is fixed-width and first so a host reader can pick up magic and len
// without agreeing on anything else.
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
