// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ISS subset of tt-llm-engine x280/include/x280.h — addresses the bench uses.

#ifndef X280_H
#define X280_H

#include <stdint.h>

#define LIM_BASE 0x08000000UL
#define LIM_SIZE 0x1E0000UL

#define SENTINEL_ADDR ((volatile uint64_t*)0x08100000UL)
#define SENTINEL_VALUE 0xDEADBEEFCAFEBABEULL

#define X280_IDLE_FW_LOAD_ADDR 0x08000000UL
#define X280_ACTIVE_FW_LOAD_ADDR 0x08001000UL

#define X280_BOOT_HANDSHAKE_BASE 0x08130200UL
#define X280_BOOT_PHASE_ADDR (X280_BOOT_HANDSHAKE_BASE + 0xC0UL)
#define X280_BOOT_PHASE_IDLE 0x000000001D1E0001ULL
#define X280_BOOT_PHASE_RUNNING_ACTIVE_FW 0x000000007E570001ULL
#define X280_BOOT_PHASE_RETURNED_TO_IDLE 0x000000001D1E0002ULL

#define CLINT_BASE 0x02000000UL
#define NUM_HARTS 4

extern uint64_t _probe_active;

#endif
