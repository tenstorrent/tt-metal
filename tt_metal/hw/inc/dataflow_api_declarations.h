// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if __has_include("chlkc_unpack_data_format.h")
#include "chlkc_pack_data_format.h"
#include "chlkc_unpack_data_format.h"
#include "chlkc_unpack_tile_dims.h"
#define DATA_FORMATS_DEFINED
#endif
#include <noc/noc_parameters.h>

#include <stdint.h>

#include "core_config.h"
#include "circular_buffer.h"
#include "dataflow_cmd_bufs.h"
#include "debug/sanitize_noc.h"
#include "debug/waypoint.h"
#include "eth_l1_address_map.h"
#include "hostdevcommon/common_values.hpp"
#include "risc_attribs.h"
#include "umd/device/tt_silicon_driver_common.hpp"
#include "utils/utils.h"
#include "debug/assert.h"
#include "dev_msgs.h"

#if defined(COMPILE_FOR_BRISC)
constexpr uint8_t proc_type = static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0);
#else
constexpr uint8_t proc_type = static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM1);
#endif
#if defined(KERNEL_BUILD)
constexpr uint8_t noc_index = NOC_INDEX;
constexpr uint8_t noc_mode = NOC_MODE;
#else

extern uint8_t noc_index;
constexpr uint8_t noc_mode = DM_DEDICATED_NOC;
#endif

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

static_assert(NUM_NOCS == 2);
// "Scratch" in L1 has space allocated for 256 DRAM and L1 enteries, to store offsets and NOC XY data.
// (MEM_BANK_TO_NOC_XY_SCRATCH and MEM_BANK_OFFSET_SCRATCH)
static_assert((NUM_DRAM_BANKS + NUM_L1_BANKS) <= 256);

extern uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS];
extern int32_t bank_to_dram_offset[NUM_DRAM_BANKS];
extern uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS];
extern int32_t bank_to_l1_offset[NUM_L1_BANKS];

extern uint32_t tt_l1_ptr* rta_l1_base;
extern uint32_t tt_l1_ptr* crta_l1_base;
extern uint32_t tt_l1_ptr* sem_l1_base[];

template <typename T>
FORCE_INLINE T get_arg_val(int arg_idx);
static FORCE_INLINE uint32_t get_arg_addr(int arg_idx);
