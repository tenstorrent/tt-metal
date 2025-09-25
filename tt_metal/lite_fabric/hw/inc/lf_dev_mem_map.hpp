// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define MEM_LOCAL_SIZE (8 * 1024) /* Local memory size -- on erisc1. not being shared with base firmware*/
#define MEM_LOCAL_BASE 0xFFB00000 /* Local memory base address */
#define MEM_LITE_FABRIC_NOC_ATOMIC_RET_VAL_ADDR 4

#define LITE_FABRIC_BARRIER 12

// NOTE: Base firmware data is starting at 0x70000.
// We need to ensure that the Lite Fabric memory does not overlap with it or Metal
#define MEM_LITE_FABRIC_MEMORY_BASE 0x6A000
#define MEM_LITE_FABRIC_MEMORY_SIZE (24 * 1024)
#define MEM_LITE_FABRIC_MEMORY_END (MEM_LITE_FABRIC_MEMORY_BASE + MEM_LITE_FABRIC_MEMORY_SIZE)

/* Lite Fabric Memory Layout */
/* Text (firmware code) section */
#define LITE_FABRIC_TEXT_START MEM_LITE_FABRIC_MEMORY_BASE
#define LITE_FABRIC_TEXT_SIZE 0x2000

/* Data section (in L1) */
#define LITE_FABRIC_DATA_START (LITE_FABRIC_TEXT_START + LITE_FABRIC_TEXT_SIZE)
#define LITE_FABRIC_DATA_SIZE 0x1000

/* Scratch space for init. Not used. Data is in L1 at this time */
#define LITE_FABRIC_INIT_SCRATCH (LITE_FABRIC_DATA_START + LITE_FABRIC_DATA_SIZE)
#define LITE_FABRIC_INIT_SCRATCH_SIZE 1024

/* Configuration area */
#define LITE_FABRIC_CONFIG_START (LITE_FABRIC_DATA_START + LITE_FABRIC_DATA_SIZE)
#define LITE_FABRIC_CONFIG_SIZE 0x2400

/* Stack configuration */
#define LITE_FABRIC_STACK_START (MEM_LOCAL_BASE)
#define LITE_FABRIC_STACK_SIZE 1024

/* Reset PC for ERISC1 (running lite fabric) */
#define LITE_FABRIC_RESET_PC (MEM_LOCAL_BASE | 0x14008)

/* Static assert in bh_hal_eth_asserts.hpp */
#define MEMORY_LAYOUT_END (LITE_FABRIC_CONFIG_START + LITE_FABRIC_CONFIG_SIZE)
