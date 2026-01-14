// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <noc/noc_parameters.h>

#if defined(KERNEL_BUILD)
constexpr uint8_t noc_index = NOC_INDEX;
constexpr uint8_t noc_mode = NOC_MODE;
#else

extern uint8_t noc_index;
constexpr uint8_t noc_mode = DM_DEDICATED_NOC;
#endif
extern uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS];
extern int32_t bank_to_dram_offset[NUM_DRAM_BANKS];
extern uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS];
extern int32_t bank_to_l1_offset[NUM_L1_BANKS];

#ifdef ARCH_QUASAR
extern thread_local uint32_t tt_l1_ptr* rta_l1_base;
extern thread_local uint32_t tt_l1_ptr* crta_l1_base;
#else
extern uint32_t tt_l1_ptr* rta_l1_base;
extern uint32_t tt_l1_ptr* crta_l1_base;
#endif
extern uint32_t tt_l1_ptr* sem_l1_base[];

/** @file */

/**
 * \private
 */

// Use VC 1 for unicast writes, and VC 4 for mcast writes
#define NOC_UNICAST_WRITE_VC 1
#define NOC_MULTICAST_WRITE_VC 4
#define NOC_DISPATCH_MULTICAST_WRITE_VC 5  // Only to be used by the dispatch cores

#define EXCLUDE_ENABLED 1
#define EXCLUDE_ENABLED_OFFSET 22
#define EXCLUDE_DIRECTION_Y_OFFSET 21
#define EXCLUDE_DIRECTION_X_OFFSET 20
#define EXCLUDE_START_Y_OFFSET 14
#define EXCLUDE_START_X_OFFSET 8
#define DYNAMIC_NOC_DIRECTION(noc, direction) (noc == 1 ? 1 - direction : direction)
#define NOC_CLEAR_OUTSTANDING_REQ_MASK ((1ULL << (NOC_MAX_TRANSACTION_ID + 1)) - 1)

#if defined(ARCH_QUASAR)
static_assert(NUM_NOCS == 1);
#else
static_assert(NUM_NOCS == 2);
#endif
// "Scratch" in L1 has space allocated for 256 DRAM and L1 enteries, to store offsets and NOC XY data.
// (MEM_BANK_TO_NOC_XY_SCRATCH and MEM_BANK_OFFSET_SCRATCH)
static_assert((NUM_DRAM_BANKS + NUM_L1_BANKS) <= 256);
