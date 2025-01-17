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

/**
 * Returns the value at a given runtime argument index for unique (per-core) runtime arguments set via SetRuntimeArgs()
 * API.
 *
 * Return value: The value associated with the unique runtime argument index
 *
 * | Argument              | Description                                    | Type                  | Valid Range |
 * Required |
 * |-----------------------|------------------------------------------------|-----------------------|---------------------------|----------|
 * | arg_idx               | Unique Runtime argument index                  | uint32_t              | 0 to 255 | True |
 * | T (template argument) | Data type of the returned argument             | Any 4-byte sized type | N/A | True     |
 */
template <typename T>
FORCE_INLINE T get_arg_val(int arg_idx) {
    // only 4B args are supported (eg int32, uint32)
    static_assert("Error: only 4B args are supported" && sizeof(T) == 4);
    return *((tt_l1_ptr T*)(get_arg_addr(arg_idx)));
}

// TODO: write docs
// this issues only a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
FORCE_INLINE
void noc_async_read_one_packet(
    std::uint64_t src_noc_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */

    WAYPOINT("RP2W");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("RP2D");

    WAYPOINT("NAOW");
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, src_noc_addr, dst_local_l1_addr, size);

    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dst_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)src_noc_addr);
#ifdef ARCH_BLACKHOLE
    // Handles reading from PCIe
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(src_noc_addr >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(
        noc,
        read_cmd_buf,
        NOC_TARG_ADDR_COORDINATE,
        (uint32_t)(src_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, size);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc] += 1;

    WAYPOINT("NAOD");
}

/**
 * Initiates an asynchronous read from a specified source node located at NOC
 * coordinates (x,y) at a local address (encoded as a uint64_t using \a
 * get_noc_addr function). The destination is in L1 memory on the Tensix core
 * executing this function call. Also, see \a noc_async_read_barrier.
 *
 * The source node can be either a DRAM bank, a Tensix core or a PCIe controller.
 *
 * Return value: None
 *
 * | Argument          | Description                                        | Data type | Valid range | required |
 * |-------------------|----------------------------------------------------|-----------|------------------------------------------|----------|
 * | src_noc_addr      | Encoding of the source NOC location (x,y)+address  | uint64_t  | DOX-TODO(ref to explain valid
 * coords)    | Yes      | | dst_local_l1_addr | Address in local L1 memory                         | uint32_t  | 0..1MB
 * | Yes      | | size              | Size of data transfer in bytes                     | uint32_t  | 0..1MB | Yes |
 */
template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1>
inline void noc_async_read(
    std::uint64_t src_noc_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
        noc_async_read_one_packet(src_noc_addr, dst_local_l1_addr, size, noc);
    } else {
        WAYPOINT("NARW");
        DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, src_noc_addr, dst_local_l1_addr, size);
        ncrisc_noc_fast_read_any_len(noc, read_cmd_buf, src_noc_addr, dst_local_l1_addr, size);
        WAYPOINT("NARD");
    }
}

template <uint32_t tile_hw = 1024>
FORCE_INLINE constexpr static std::uint32_t MUL_WITH_TILE_SIZE(uint format, uint index) {
    constexpr uint8_t datum_shift = (tile_hw == 1024)  ? 10
                                    : (tile_hw == 512) ? 9
                                    : (tile_hw == 256) ? 8
                                    : (tile_hw == 128) ? 7
                                    : (tile_hw == 64)  ? 6
                                    : (tile_hw == 32)  ? 5
                                    : (tile_hw == 16)  ? 4
                                                       : 10;

    constexpr uint8_t exp_shift = (tile_hw == 1024)  ? 6
                                  : (tile_hw == 512) ? 5
                                  : (tile_hw == 256) ? 4
                                  : (tile_hw == 128) ? 4
                                  : (tile_hw == 64)  ? 4
                                  : (tile_hw == 32)  ? 4
                                  : (tile_hw == 16)  ? 4
                                                     : 6;
    switch (format & 0x1F) {
        case ((uint8_t)DataFormat::UInt8): return (index << datum_shift);
        case ((uint8_t)DataFormat::UInt16):
        case ((uint8_t)DataFormat::Float16):
        case ((uint8_t)DataFormat::Float16_b): return (index << (datum_shift + 1));
        case ((uint8_t)DataFormat::Int32):
        case ((uint8_t)DataFormat::UInt32):
        case ((uint8_t)DataFormat::Float32): return (index << (datum_shift + 2));
        case ((uint8_t)DataFormat::Bfp2):
        case ((uint8_t)DataFormat::Bfp2_b): return ((index << (datum_shift - 2)) + (index << (exp_shift)));
        case ((uint8_t)DataFormat::Bfp4):
        case ((uint8_t)DataFormat::Bfp4_b): return ((index << (datum_shift - 1)) + (index << (exp_shift)));
        case ((uint8_t)DataFormat::Bfp8):
        case ((uint8_t)DataFormat::Bfp8_b):
        // Keep default as Bfp8?
        default: return ((index << datum_shift) + (index << (exp_shift)));
    };
}

/*
    Need an alias to get_noc_addr so that the structs below don't confuse the above get_noc_addr with
    the struct variant
*/
FORCE_INLINE
std::uint64_t get_noc_addr_helper(std::uint32_t noc_xy, std::uint32_t addr) {
    /*
        Get an encoding which contains tensix core and address you want to
        write to via the noc multicast
    */
    return ((uint64_t)(noc_xy) << NOC_ADDR_COORD_SHIFT) | addr;
}
