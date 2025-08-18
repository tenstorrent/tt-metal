/*
 * SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared Memory Definitions for Lite Fabric
 *
 * This file contains memory layout constants that are shared between
 * the linker script and C/C++ code. It uses preprocessor definitions
 * that work in both contexts.
 */

#pragma once

#define MEM_LOCAL_SIZE (8 * 1024) /* Local memory size -- on erisc1. not being shared with base firmware*/
#define MEM_LOCAL_BASE 0xFFB00000 /* Local memory base address */
#define MEM_LITE_FABRIC_NOC_ATOMIC_RET_VAL_ADDR 4

#define MEM_ERISC_RESERVED1 0         /* ERISC reserved area base */
#define MEM_ERISC_RESERVED1_SIZE 1024 /* ERISC reserved area size */

/* Same as dev_mem_map.h */
#define MEM_ERISC_FABRIC_LITE_BARRIER (MEM_ERISC_RESERVED1 + MEM_ERISC_RESERVED1_SIZE)

#define MEM_LITE_FABRIC_MEMORY_BASE 0x6A000
#define MEM_LITE_FABRIC_MEMORY_SIZE (24 * 1024)
#define MEM_LITE_FABRIC_MEMORY_END (MEM_LITE_FABRIC_MEMORY_BASE + MEM_LITE_FABRIC_MEMORY_SIZE)

/* Lite Fabric Memory Layout */
/* Text (firmware code) section */
#define LITE_FABRIC_TEXT_START MEM_LITE_FABRIC_MEMORY_BASE
#define LITE_FABRIC_TEXT_SIZE 0x2000

/* Scratch space for init. No L1 to Local scratch. We place data into L1 */
#define LITE_FABRIC_INIT_BANK_TO_NOC_SCRATCH (LITE_FABRIC_TEXT_START + LITE_FABRIC_TEXT_SIZE)
#define LITE_FABRIC_BANK_TO_NOC_SIZE 1024

/* Data section (in L1) */
#define LITE_FABRIC_DATA_START (LITE_FABRIC_INIT_BANK_TO_NOC_SCRATCH + LITE_FABRIC_BANK_TO_NOC_SIZE)
#define LITE_FABRIC_DATA_SIZE 0x1000

/* Configuration area */
#define LITE_FABRIC_CONFIG_START (LITE_FABRIC_DATA_START + LITE_FABRIC_DATA_SIZE)
#define LITE_FABRIC_CONFIG_SIZE 0x2400

/* Stack configuration */
#define LITE_FABRIC_STACK_MIN_SIZE 0x200

/* Reset PC for ERISC1 (running lite fabric) */
#define LITE_FABRIC_RESET_PC (MEM_LOCAL_BASE | 0x14008)

/* Static assert in bh_hal_eth_asserts.hpp */
#define MEMORY_LAYOUT_END (LITE_FABRIC_CONFIG_START + LITE_FABRIC_CONFIG_SIZE)
