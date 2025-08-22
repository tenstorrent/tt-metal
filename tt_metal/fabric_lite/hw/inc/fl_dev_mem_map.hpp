// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define MEM_LOCAL_SIZE (8 * 1024) /* Local memory size -- on erisc1. not being shared with base firmware*/
#define MEM_LOCAL_BASE 0xFFB00000 /* Local memory base address */
#define MEM_FABRIC_LITE_NOC_ATOMIC_RET_VAL_ADDR 4

#define MEM_ERISC_RESERVED1 0         /* ERISC reserved area base */
#define MEM_ERISC_RESERVED1_SIZE 1024 /* ERISC reserved area size */

/* Same as dev_mem_map.h */
#define MEM_ERISC_FABRIC_LITE_BARRIER (MEM_ERISC_RESERVED1 + MEM_ERISC_RESERVED1_SIZE)

// NOTE: Base firmware data is starting at 0x70000.
// We need to ensure that the Lite Fabric memory does not overlap with it or Metal
#define MEM_FABRIC_LITE_MEMORY_BASE 0x6A000
#define MEM_FABRIC_LITE_MEMORY_SIZE (24 * 1024)
#define MEM_FABRIC_LITE_MEMORY_END (MEM_FABRIC_LITE_MEMORY_BASE + MEM_FABRIC_LITE_MEMORY_SIZE)

/* Lite Fabric Memory Layout */
/* Text (firmware code) section */
#define FABRIC_LITE_TEXT_START MEM_FABRIC_LITE_MEMORY_BASE
#define FABRIC_LITE_TEXT_SIZE 0x2000

/* Data section (in L1) */
#define FABRIC_LITE_DATA_START (FABRIC_LITE_TEXT_START + FABRIC_LITE_TEXT_SIZE)
#define FABRIC_LITE_DATA_SIZE 0x1000

/* Scratch space for init. No L1 to Local scratch. We place data into L1 */
#define FABRIC_LITE_INIT_BANK_TO_NOC_SCRATCH (FABRIC_LITE_DATA_START + FABRIC_LITE_DATA_SIZE)
#define FABRIC_LITE_BANK_TO_NOC_SIZE 1024

/* Configuration area */
#define FABRIC_LITE_CONFIG_START (FABRIC_LITE_DATA_START + FABRIC_LITE_DATA_SIZE)
#define FABRIC_LITE_CONFIG_SIZE 0x2400

/* Stack configuration */
#define FABRIC_LITE_STACK_START (FABRIC_LITE_CONFIG_START + FABRIC_LITE_CONFIG_SIZE)
#define FABRIC_LITE_STACK_SIZE 1024

/* Reset PC for ERISC1 (running lite fabric) */
#define FABRIC_LITE_RESET_PC (MEM_LOCAL_BASE | 0x14008)

/* Static assert in bh_hal_eth_asserts.hpp */
#define MEMORY_LAYOUT_END (FABRIC_LITE_STACK_START + FABRIC_LITE_STACK_SIZE)
