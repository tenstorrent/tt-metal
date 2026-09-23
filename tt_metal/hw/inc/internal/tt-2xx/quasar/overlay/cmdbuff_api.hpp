// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
// Version: FFN1.3.0
/**
 * @file cmdbuff_api.hpp
 * @brief Command Buffer API for Overlay Data Movement Operations
 *
 * This API provides control over the overlay's two normal command buffers used for
 * complex data movement operations. Each overlay contains 2 normal command buffers
 * plus 1 simple command buffer (simple buffer API is separate).
 *
 * ## Implementation Details
 *
 * Command buffer operations use custom RISC-V ROCC instructions with unique opcodes
 * for each buffer. Since opcodes must be known at compile time, the command buffer id
 * is a template parameter:
 * - `xxx_cmdbuf<CMDBUF_0>()` - Functions for command buffer 0
 * - `xxx_cmdbuf<CMDBUF_1>()` - Functions for command buffer 1
 * `xxx_cmdbuf_0()` / `xxx_cmdbuf_1()` aliases are provided at the end of the file.
 *
 * ## Transaction ID Management
 *
 * Each command buffer maintains separate transaction ID ranges to avoid conflicts:
 * - **Buffer 0**: Static TID 1, wrapping range 2-6
 * - **Buffer 1**: Static TID 7, wrapping range 8-12
 *
 * @note Requires ROCC instruction definitions from rocc_instructions.hpp
 */
#pragma once

#include <type_traits>
#include "rocc_instructions.hpp"

#define CMDBUF_0 0
#define CMDBUF_1 1

namespace overlay {

/* Guard for the CMDBUF template parameter: the __builtin_riscv_ttrocc_cmdbuf_* builtins accept any
 * constant id and silently encode an out-of-range one as a different instruction. */
template <uint32_t CMDBUF>
using cmdbuf_id_t = std::enable_if_t<CMDBUF == CMDBUF_0 || CMDBUF == CMDBUF_1>;

/* Default transaction ID for both command buffers */
constexpr uint32_t CMDBUF_DEF_TRID = 0;
/* Static(starting) transaction ID for command buffer 0 */
constexpr uint32_t CMDBUF_0_TRID_STATIC = 1;
/* Start transaction ID for command buffer 0 when using TID wrapping */
constexpr uint32_t CMDBUF_0_TRID_START = 2;
/* End transaction ID for command buffer 0 when using TID wrapping */
constexpr uint32_t CMDBUF_0_TRID_END = 6;
/* Static(starting) transaction ID for command buffer 0 */
constexpr uint32_t CMDBUF_1_TRID_STATIC = 7;
/* Start transaction ID for command buffer 0 when using TID wrapping */
constexpr uint32_t CMDBUF_1_TRID_START = 8;
/* End transaction ID for command buffer 0 when using TID wrapping */
constexpr uint32_t CMDBUF_1_TRID_END = 12;

/* Read request virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_RD_REQ_VC = 1;
/* Read response virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_RD_RESP_VC = 12;
/* Write request virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_WR_REQ_VC = 1;
/* Write response virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_WR_RESP_VC = 13;
/* Multicast request virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_MCAST_REQ_VC = 8;
/* Multicast response virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_MCAST_RESP_VC = 14;
/* First of the 8 iDMA backend request virtual channels (CMDBUF_FIRST_IDMA_VC..+7).
 * Kept independent of CMDBUF_WR_REQ_VC so that moving the unicast write VC cannot
 * walk the iDMA VC-autoincrement range into the multicast request VCs. */
constexpr uint32_t CMDBUF_FIRST_IDMA_VC = 0;
/* Number of iDMA backend engines the cmdbuf round-robins packets across */
constexpr uint32_t CMDBUF_NUM_IDMA_VCS = 8;

/*
 * @fn reset_cmdbuf<CMDBUF>()
 *
 * @brief Defines an inline reset functions for resetting command buffers state
 * Should be called before any other command buffer setup functions.
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void reset_cmdbuf() {
    __builtin_riscv_ttrocc_cmdbuf_reset(CMDBUF);
}

/*
 * @fn setup_as_copy_cmdbuf<CMDBUF>
 *
 * @brief Configures command buffer for copy operations with customizable settings
 *
 * @param wr Indicates if the operation is a write (true) or read (false)
 * @param mcast Enables multicast if true; default is false
 * @param mcast_exclude A `TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u` structure specifying which cores to exclude
 *                      from multicast. Defaults to no exclusions
 * @param wrapping_en Enables address wrapping functionality; default is true
 * @param posted Enables posted transactions for better performance; default is true
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void setup_as_copy_cmdbuf(
    bool wr,
    bool mcast = false,
    TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u mcast_exclude = {0},
    bool wrapping_en = true,
    bool posted = true) {
    TT_ROCC_CMD_BUF_MISC_reg_u misc;
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

    misc.f.linked = mcast;
    misc.f.posted = wr && posted;
    misc.f.multicast = mcast;
    misc.f.write_trans = wr;
    misc.f.wrapping_en = wrapping_en;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc.val);

    if (mcast) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_EXCLUDE_REG_OFFSET / 8, mcast_exclude.val);
    }
}

/*
 * @fn setup_as_scatter_list_cmdbuf<CMDBUF>
 *
 * @brief Configures the specified command buffer as a scatter list with customizable settings.
 *
 * This function enables scatter list functionality for a command buffers
 *
 * @param wr Indicates if the operation is a write (true) or read (false).
 * @param apply_scatter_to_dest Indicates if scatter list should be applied to the destination address.
 * @param mcast Enables multicast if true; default is false.
 * @param linked Enables linked transaction
 * @param mcast_exclude A `TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u` structure specifying which cores to exclude
 *                      from multicast. Defaults to no exclusions.
 * @param scatter_list_contains_size Specifies if the scatter list includes size information.
 * @param scatter_list_contains_xy Specifies if the scatter list includes XY coordinates. When true,
 *                                 each element in the scatter list includes XY positioning.
 *
 * @note To be used with set_scatter_list_x(_)
 * @note Same thing can be achieved with setup_x() function
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void setup_as_scatter_list_cmdbuf(
    bool wr,
    bool apply_scatter_to_dest,
    bool mcast = false,
    bool linked = false,
    TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u mcast_exclude = {0},
    bool scatter_list_contains_size = false,
    bool scatter_list_contains_xy = false,
    bool wrapping_en = true,
    bool posted = true) {
    TT_ROCC_CMD_BUF_MISC_reg_u misc;
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

    misc.f.linked = linked;
    misc.f.posted = wr && posted;
    misc.f.multicast = mcast;
    misc.f.scatter_list_en = true;
    misc.f.scatter_list_to_dest_addr = apply_scatter_to_dest;
    misc.f.write_trans = wr;
    misc.f.scatter_list_has_size = scatter_list_contains_size;
    misc.f.scatter_list_has_xy = scatter_list_contains_xy;
    misc.f.wrapping_en = wrapping_en;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc.val);

    if (mcast) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_EXCLUDE_REG_OFFSET / 8, mcast_exclude.val);
    }
}

/*
 * @fn setup_as_atomic_cmdbuf<CMDBUF>
 *
 * @brief Function for configuring command buffer for atomic transactions
 *
 * @param wr Indicates if the operation is a write (true) or read (false).
 *
 * @note Overwrites existing command buffer settings
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void setup_as_atomic_cmdbuf(bool wr) {
    TT_ROCC_CMD_BUF_MISC_reg_u misc;
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

    misc.f.posted = 1;
    misc.f.write_trans = 0;
    misc.f.atomic_trans = 1;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc.val);
}

/*
 * @fn idma_setup_as_copy_cmdbuf<CMDBUF>
 *
 * @brief Function for configuring command buffer for iDMA copy operations
 *
 * @param wrapping_en Enables address wrapping; default is true
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void idma_setup_as_copy_cmdbuf(bool wrapping_en = true) {
    TT_ROCC_CMD_BUF_MISC_reg_u misc;
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

    misc.f.write_trans = 1;
    misc.f.idma_en = 1;
    misc.f.wrapping_en = wrapping_en;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc.val);
}

/*
 * @fn idma_setup_as_scatter_list_cmdbuf<CMDBUF>
 *
 * @brief Function for configuring command buffer for iDMA scatter list operations
 *
 * @param apply_scatter_to_dest Indicates if scatter list should be applied to the destination address
 * @param scatter_list_contains_size Specifies if the scatter list includes size information
 * @param scatter_list_contains_xy Specifies if the scatter list includes XY coordinates
 * @param wrapping_en Enables address wrapping; default is true
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void idma_setup_as_scatter_list_cmdbuf(
    bool apply_scatter_to_dest,
    bool scatter_list_contains_size = false,
    bool scatter_list_contains_xy = false,
    bool wrapping_en = true) {
    TT_ROCC_CMD_BUF_MISC_reg_u misc;
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

    misc.f.scatter_list_en = true;
    misc.f.scatter_list_to_dest_addr = apply_scatter_to_dest;
    misc.f.write_trans = 1;
    misc.f.scatter_list_has_size = scatter_list_contains_size;
    misc.f.scatter_list_has_xy = scatter_list_contains_xy;
    misc.f.idma_en = 1;
    misc.f.wrapping_en = wrapping_en;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc.val);
}

/*
 * @fn set_axi_opt_1_cmdbuf<CMDBUF>
 *
 * @brief Programs the AXI_OPT_1 cmdbuf register. All fields not exposed as parameters
 *        are written at their TT_ROCC_CMD_BUF_AXI_OPT_1_REG_DEFAULT values.
 *
 * @param src_protocol AXI source protocol selector
 * @param decouple_aw  Decouple AXI AW from W channel
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void set_axi_opt_1_cmdbuf(uint8_t src_protocol, uint8_t decouple_aw) {
    TT_ROCC_CMD_BUF_AXI_OPT_1_reg_u axi_opt_1;
    axi_opt_1.val = TT_ROCC_CMD_BUF_AXI_OPT_1_REG_DEFAULT;
    axi_opt_1.f.src_protocol = src_protocol;
    axi_opt_1.f.decouple_aw = decouple_aw;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_AXI_OPT_1_REG_OFFSET / 8, axi_opt_1.val);
}

/*
 * @fn setup_ongoing_cmdbuf<CMDBUF>
 *
 * @brief Function for configuring incrementing logic for command buffer
 * Addresses, vcs and transaction ids can be configure to be self incrementing after each transactions
 *
 * @param src_addr_inc_en If enabled source address will increment after issue
 * @param dest_addr_inc_en If enabled destination address will increment after issue
 * @param trid_inc_en If enabled transaction ID will increment after issue
 * @param req_vc_inc_en If enabled request VC will increment after issue
 * @param resp_vc_inc_en If enabled response VC will increment after issue
 * @param req_vc_inc_on_entire_trans Request VC increment on entire transaction
 * @param resp_vc_inc_on_entire_trans Response VC increment on entire transaction
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void setup_ongoing_cmdbuf(
    bool src_addr_inc_en,
    bool dest_addr_inc_en,
    bool trid_inc_en,
    bool req_vc_inc_en,
    bool resp_vc_inc_en,
    bool req_vc_inc_on_entire_trans = false,
    bool resp_vc_inc_on_entire_trans = false) {
    TT_ROCC_CMD_BUF_AUTOINC_reg_u ongoing;

    ongoing.f.src_addr_inc_en = src_addr_inc_en;
    ongoing.f.dest_addr_inc_en = dest_addr_inc_en;
    ongoing.f.trid_inc_en = trid_inc_en;
    ongoing.f.req_vc_inc_en = req_vc_inc_en;
    ongoing.f.resp_vc_inc_en = resp_vc_inc_en;
    ongoing.f.req_vc_inc_on_entire_trans = req_vc_inc_on_entire_trans;
    ongoing.f.resp_vc_inc_on_entire_trans = resp_vc_inc_on_entire_trans;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_AUTOINC_REG_OFFSET / 8, ongoing.val);
}

/*
 * @fn setup_vcs_cmdbuf<CMDBUF>
 *
 * @brief Function for configuring virtual channels based on global values defined in this file
 *
 * @param wr If enabled, will use write VCs
 * @param mcast If enabled, will use multicast VCs
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void setup_vcs_cmdbuf(bool wr, bool mcast = false) {
    if (wr) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            CMDBUF,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8,
            mcast ? CMDBUF_MCAST_REQ_VC : CMDBUF_WR_REQ_VC);
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            CMDBUF,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8,
            mcast ? CMDBUF_MCAST_RESP_VC : CMDBUF_WR_RESP_VC);
    } else {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, CMDBUF_RD_REQ_VC);
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, CMDBUF_RD_RESP_VC);
    }
}

/*
 * @fn setup_wrapping_req_vcs_cmdbuf<CMDBUF>
 *
 * @brief Function for configuring wrapping feature of request and response virtual channels
 *
 * @param wr Write/read mode selector
 * @param req_start_vc Starting virtual channel for requests
 * @param req_end_vc End virtual channel for requests
 * @param req_vc_offset Offset while wrapping for requests
 * @param resp_start_vc Starting virtual channel for responses
 * @param resp_end_vc End virtual channel for responses
 * @param resp_vc_offset Offset while wrapping for responses
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void setup_wrapping_vcs_cmdbuf(
    bool wr,
    uint32_t req_start_vc,
    uint32_t req_end_vc,
    uint32_t req_vc_offset = 0,
    uint32_t resp_start_vc = 0,
    uint32_t resp_end_vc = 0,
    uint32_t resp_vc_offset = 0) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, req_start_vc + req_vc_offset);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_BASE_REG_OFFSET / 8, req_start_vc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_SIZE_REG_OFFSET / 8, req_end_vc - req_start_vc + 1);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, resp_start_vc + resp_vc_offset);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_BASE_REG_OFFSET / 8, resp_start_vc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_SIZE_REG_OFFSET / 8, resp_end_vc - resp_start_vc + 1);
}

/*
 * @fn setup_trids_static_cmdbuf<CMDBUF>
 *
 * @brief Function for configuring transaction ID based on macros defined in this file
 * If wrapping feature for transaction ID is enabled, specified ID is used as offset
 *
 * @param trid_offset Transaction ID, if wrapping is enabled this serves as transaction ID offset
 * @param wrapping Enables wrapping feature for transaction IDs
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void setup_trids_cmdbuf(
    uint32_t trid_offset = CMDBUF_DEF_TRID, bool wrapping = false) {
    if constexpr (CMDBUF == CMDBUF_0) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            CMDBUF,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_WR_SENT_TR_ID_REG_OFFSET / 8,
            wrapping ? CMDBUF_0_TRID_START + trid_offset : trid_offset);
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            CMDBUF,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ACK_TR_ID_REG_OFFSET / 8,
            wrapping ? CMDBUF_0_TRID_START + trid_offset : trid_offset);
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            CMDBUF,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8,
            wrapping ? CMDBUF_0_TRID_START + trid_offset : trid_offset);
        if (wrapping) {
            __builtin_riscv_ttrocc_cmdbuf_wr_reg(
                CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_BASE_REG_OFFSET / 8, CMDBUF_0_TRID_START);
            __builtin_riscv_ttrocc_cmdbuf_wr_reg(
                CMDBUF,
                TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_SIZE_REG_OFFSET / 8,
                CMDBUF_0_TRID_END - CMDBUF_0_TRID_START + 1);
        }
    } else {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            CMDBUF,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_WR_SENT_TR_ID_REG_OFFSET / 8,
            wrapping ? CMDBUF_1_TRID_START + trid_offset : trid_offset);
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            CMDBUF,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ACK_TR_ID_REG_OFFSET / 8,
            wrapping ? CMDBUF_1_TRID_START + trid_offset : trid_offset);
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            CMDBUF,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8,
            wrapping ? CMDBUF_1_TRID_START + trid_offset : trid_offset);
        if (wrapping) {
            __builtin_riscv_ttrocc_cmdbuf_wr_reg(
                CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_BASE_REG_OFFSET / 8, CMDBUF_1_TRID_START);
            __builtin_riscv_ttrocc_cmdbuf_wr_reg(
                CMDBUF,
                TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_SIZE_REG_OFFSET / 8,
                CMDBUF_1_TRID_END - CMDBUF_1_TRID_START + 1);
        }
    }
}

/*
 * @fn swap_trid_cmdbuf<CMDBUF>
 *
 * @brief Swaps current transaction ID with new one and returns previous value
 *
 * @param new_trid New transaction ID to set
 * @return Previous transaction ID value
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint32_t swap_trid_cmdbuf(uint32_t new_trid) {
    uint32_t prev_trid =
        __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8, new_trid);
    return prev_trid;
}

/*
 * @fn setup_max_bytes_in_packet_cmdbuf<CMDBUF>
 *
 * @brief Sets maximum bytes per packet for command buffer
 *
 * @param max_bytes_in_packet Maximum bytes allowed in single packet
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void setup_max_bytes_in_packet_cmdbuf(uint64_t max_bytes_in_packet) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MAX_BYTES_IN_PACKET_REG_OFFSET / 8, max_bytes_in_packet);
}

/*
 * @fn setup_packet_tags_cmdbuf<CMDBUF>
 *
 * @brief Function for configuring snoop and flush bit of transaction
 *
 * @param snoop_bit Enables destination NIU for cache snoop mechanisms
 * @param flush_bit Enables destination NIU to commit all parts of the flit before committing the next packet
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void setup_packet_tags_cmdbuf(bool snoop_bit, bool flush_bit) {
    TT_ROCC_CMD_BUF_PACKET_TAGS_reg_u misc;

    misc.f.snoop_bit = snoop_bit;
    misc.f.flush_bit = flush_bit;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PACKET_TAGS_REG_OFFSET / 8, misc.val);
}

/*
 * @fn get_src_cmdbuf<CMDBUF>
 *
 * @brief Returns source address for transactions
 *
 * @return Source address with noc coordinates embedded
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t get_src_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8);
}

/*
 * @fn set_src_cmdbuf<CMDBUF>
 *
 * @brief Sets up source configuration
 *
 * @param addr Source address without coordinates (0-4mbs)
 * @param coordinate Coordinate generated using NOC_XY_COORD macro
 * @param base Base address, if wrapping is enabled
 * @param size Size of transfer in bytes
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void set_src_cmdbuf(
    uint64_t addr, uint64_t coordinate, uint64_t base, uint64_t size) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_BASE_REG_OFFSET / 8, base);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_SIZE_REG_OFFSET / 8, size);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET / 8, coordinate);
}
/*
 * @fn set_src_cmdbuf<CMDBUF> (3-parameter overload)
 *
 * @brief Sets up source configuration with base address
 *
 * @param addr Source address without coordinates
 * @param coordinate Coordinate generated using NOC_XY_COORD macro
 * @param base Base address for wrapping
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void set_src_cmdbuf(uint64_t addr, uint64_t coordinate, uint64_t base) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_BASE_REG_OFFSET / 8, base);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET / 8, coordinate);
}
/*
 * @fn set_src_cmdbuf<CMDBUF> (2-parameter overload)
 *
 * @brief Sets up source configuration with address and coordinate
 *
 * @param addr Source address without coordinates
 * @param coordinate Coordinate generated using NOC_XY_COORD macro
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void set_src_cmdbuf(uint64_t addr, uint64_t coordinate) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET / 8, coordinate);
}
/*
 * @fn set_src_cmdbuf<CMDBUF> (1-parameter overload)
 *
 * @brief Sets up source address only
 *
 * @param addr Source address
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void set_src_cmdbuf(uint64_t addr) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, addr);
}

/*
 * @fn get_dest_cmdbuf<CMDBUF>
 *
 * @brief Returns destination address for transactions
 *
 * @return Destination address with noc coordinates embedded
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t get_dest_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8);
}

/*
 * @fn set_dest_cmdbuf<CMDBUF>
 *
 * @brief Sets up destination configuration
 *
 * @param addr Destination address without coordinates (0-4mbs)
 * @param coordinates Coordinate generated using NOC_XY_COORD macro
 * @param base Base address, if wrapping is enabled
 * @param size Size of transfer in bytes
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void set_dest_cmdbuf(
    uint64_t addr, uint64_t coordinates, uint64_t base, uint64_t size) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_BASE_REG_OFFSET / 8, base);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_SIZE_REG_OFFSET / 8, size);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, coordinates);
}
/*
 * @fn set_dest_cmdbuf<CMDBUF> (3-parameter overload)
 *
 * @brief Sets up destination configuration with base address
 *
 * @param addr Destination address without coordinates
 * @param coordinates Coordinate generated using NOC_XY_COORD macro
 * @param base Base address for wrapping
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void set_dest_cmdbuf(uint64_t addr, uint64_t coordinates, uint64_t base) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_BASE_REG_OFFSET / 8, base);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, coordinates);
}
/*
 * @fn set_dest_cmdbuf<CMDBUF> (2-parameter overload)
 *
 * @brief Sets up destination configuration with address and coordinate
 *
 * @param addr Destination address without coordinates
 * @param coordinates Coordinate generated using NOC_XY_COORD macro
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void set_dest_cmdbuf(uint64_t addr, uint64_t coordinates) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, coordinates);
}
/*
 * @fn set_dest_cmdbuf<CMDBUF> (1-parameter overload)
 *
 * @brief Sets up destination address only
 *
 * @param addr Destination address
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void set_dest_cmdbuf(uint64_t addr) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, addr);
}

/*
 * @fn set_scatter_list_cmdbuf<CMDBUF>
 *
 * @brief Configure scatter list parameters
 *
 * @param addr Scatter list address
 * @param base Base address
 * @param index Index value
 * @param times Number of times
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void set_scatter_list_cmdbuf(
    uint64_t addr, uint64_t base, uint64_t index, uint64_t times) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_LIST_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_BASE_ADDR_REG_OFFSET / 8, base);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_INDEX_REG_OFFSET / 8, index);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_TIMES_REG_OFFSET / 8, times);
}
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void set_scatter_list_base_cmdbuf(uint64_t base) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_BASE_ADDR_REG_OFFSET / 8, base);
}
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void set_scatter_list_index_cmdbuf(uint64_t index) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_INDEX_REG_OFFSET / 8, index);
}
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void set_scatter_list_times_cmdbuf(uint64_t times) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_TIMES_REG_OFFSET / 8, times);
}
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t get_scatter_list_addr_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_LIST_ADDR_REG_OFFSET / 8);
}
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t get_scatter_list_base_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_BASE_ADDR_REG_OFFSET / 8);
}
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t get_scatter_list_index_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_INDEX_REG_OFFSET / 8);
}
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t get_scatter_list_times_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_TIMES_REG_OFFSET / 8);
}

/*
 * @fn set_len_cmdbuf<CMDBUF>
 *
 * @brief Configures size of transfer in bytes
 *
 * @param size_bytes Size in bytes
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void set_len_cmdbuf(uint64_t size_bytes) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, size_bytes);
}

/*
 * @fn issue_transaction_cmdbuf<CMDBUF>
 *
 * @brief Kicks off noc transaction with previously configured command buff
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void issue_cmdbuf() {
    __builtin_riscv_ttrocc_cmdbuf_issue_trans(CMDBUF);
}

/*
 * @fn issue_read_cmdbuf<CMDBUF>
 *
 * @brief Issues read transaction
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void issue_read_cmdbuf() {
    issue_cmdbuf<CMDBUF>();
}

/*
 * @fn issue_write_cmdbuf<CMDBUF>
 *
 * @brief Issues write transaction
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void issue_write_cmdbuf() {
    issue_cmdbuf<CMDBUF>();
}

/*
 * @fn issue_write_inline_cmdbuf<CMDBUF>
 *
 * @brief Kicks off inline noc transaction with underling custom ASM instruction
 *
 * @param data Inline data to be written
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void issue_write_inline_cmdbuf(uint64_t data) {
    __builtin_riscv_ttrocc_cmdbuf_issue_inline_trans(CMDBUF, data);
}

/*
 * @fn issue_write_inline_len_cmdbuf<CMDBUF>
 *
 * @brief Kicks off inline noc transaction with underling custom ASM instruction
 *
 * @param data Inline data to be written
 * @param dest_addr Destination address (with or without xy coordinates embedded)
 * @param size_bytes Size of data in bytes
 * @param has_xy Flag for specifying if address contains noc coordinates using NOC_XY_COORD
 * @param posted Flag if transfer should be posted or not
 * @param snoop Flag for enabling snoop bit
 * @param flush Flag for enabling flush bit
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void issue_write_inline_len_cmdbuf(
    uint64_t data,
    uint64_t dest_addr,
    uint64_t size_bytes,
    uint64_t has_xy = 1,
    uint64_t posted = 0,
    uint64_t snoop = 0,
    uint64_t flush = 0) {
    uint64_t rs2 =
        dest_addr | ((size_bytes - 1) << 57) | (has_xy << 60) | (posted << 61) | (snoop << 62) | (flush << 63);
    __builtin_riscv_ttrocc_cmdbuf_issue_inline_addr_trans(CMDBUF, data, rs2);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void interrupt_enable_cmdbuf(int id) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    val.val |= (1 << id);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, val.val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void interrupt_disable_cmdbuf(int id) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    val.val &= ~(1 << id);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, val.val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t interrupts_pending_cmdbuf() {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8);
    return val.val;
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void interrupt_clear_cmdbuf(int id) {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = ~(1 << id);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8, val.val);
}
/* Per-TRID Count Zero Interrupts (IE_0[31:0], IP_0[31:0]) */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_enable_cmdbuf(int trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8);
    val |= (1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8, val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_disable_cmdbuf(int trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8);
    val &= ~(1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8, val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint32_t per_trid_count_zero_get_interrupt_enable_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL; /* Lower 32 bits */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_count_zero_set_interrupt_enable_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8, reg_val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t per_trid_count_zero_interrupts_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL; /* Lower 32 bits */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint32_t per_trid_count_zero_get_interrupt_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL; /* Lower 32 bits */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_count_zero_set_interrupt_pending_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8, reg_val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_clear_cmdbuf(int trid) {
    uint64_t val;
    val = ~(1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8, val);
}

/* Per-TRID Write Count Zero Interrupts (IE_0[63:32], IP_0[63:32]) */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_enable_cmdbuf(int trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8);
    val |= (1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8, val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_disable_cmdbuf(int trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8);
    val &= ~(1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8, val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint32_t per_trid_wr_count_zero_get_interrupt_enable_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_wr_count_zero_set_interrupt_enable_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8);
    reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8, reg_val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t per_trid_wr_count_zero_interrupts_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint32_t per_trid_wr_count_zero_get_interrupt_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_wr_count_zero_set_interrupt_pending_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8);
    reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8, reg_val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_clear_cmdbuf(int trid) {
    uint64_t val;
    val = ~(1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8, val);
}

/* Per-TRID iDMA Count Zero Interrupts (IE_1[31:0], IP_1[31:0]) */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_enable_cmdbuf(int trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    val |= (1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8, val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_disable_cmdbuf(int trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    val &= ~(1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8, val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint32_t per_trid_idma_count_zero_get_interrupt_enable_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL; /* Lower 32 bits */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_idma_count_zero_set_interrupt_enable_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8, reg_val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t per_trid_idma_count_zero_interrupts_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL;
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint32_t per_trid_idma_count_zero_get_interrupt_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL; /* Lower 32 bits */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_idma_count_zero_set_interrupt_pending_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8, reg_val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_clear_cmdbuf(int trid) {
    uint64_t val;
    val = ~(1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8, val);
}

/* Per-TRID Tiles-to-Process TR_ACK Threshold Interrupts (IE_1[63:32], IP_1[63:32]) */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_enable_cmdbuf(int trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    val |= (1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8, val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_disable_cmdbuf(int trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    val &= ~(1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8, val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint32_t per_trid_tiles_to_process_get_interrupt_enable_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_tiles_to_process_set_interrupt_enable_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8, reg_val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t per_trid_tiles_to_process_interrupts_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL;
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint32_t per_trid_tiles_to_process_get_interrupt_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_tiles_to_process_set_interrupt_pending_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8);
    reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8, reg_val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_clear_cmdbuf(int trid) {
    uint64_t val;
    val = ~(1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8, val);
}

/* Per-TRID Write Tiles-to-Process WR_SENT Threshold Interrupts (IE_2[31:0], IP_2[31:0]) */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_enable_cmdbuf(int trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    val |= (1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8, val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_disable_cmdbuf(int trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    val &= ~(1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8, val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint32_t per_trid_wr_tiles_to_process_get_interrupt_enable_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL; /* Lower 32 bits */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_set_interrupt_enable_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8, reg_val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t per_trid_wr_tiles_to_process_interrupts_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL;
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint32_t per_trid_wr_tiles_to_process_get_interrupt_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL; /* Lower 32 bits */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_set_interrupt_pending_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8, reg_val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_clear_cmdbuf(int trid) {
    uint64_t val;
    val = ~(1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8, val);
}

/* Per-TRID iDMA Tiles-to-Process IDMA_TR_ACK Threshold Interrupts (IE_2[63:32], IP_2[63:32]) */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_enable_cmdbuf(int trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    val |= (1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8, val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_disable_cmdbuf(int trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    val &= ~(1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8, val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint32_t per_trid_idma_tiles_to_process_get_interrupt_enable_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_set_interrupt_enable_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8, reg_val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t per_trid_idma_tiles_to_process_interrupts_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL;
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint32_t per_trid_idma_tiles_to_process_get_interrupt_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_set_interrupt_pending_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8);
    reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8, reg_val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_clear_cmdbuf(int trid) {
    uint64_t val;
    val = ~(1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8, val);
}

/* Per-VC Has Space Interrupts (IE[47:32], IP[47:32]) */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_enable_cmdbuf(int vc) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    val.val |= (1ULL << (vc + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, val.val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_disable_cmdbuf(int vc) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    val.val &= ~(1ULL << (vc + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, val.val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint16_t per_vc_has_space_get_interrupt_enable_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFULL; /* Bits [47:32] */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_vc_has_space_set_interrupt_enable_cmdbuf(uint16_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFF0000FFFFFFFFULL) | (((uint64_t)(val & 0xFFFFULL)) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, reg_val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t per_vc_has_space_interrupts_pending_cmdbuf() {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8);
    return (val.val >> 32) & 0xFFFFULL; /* Bits [47:32] shifted to [15:0] */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint16_t per_vc_has_space_get_interrupt_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFULL; /* Bits [47:32] */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_vc_has_space_set_interrupt_pending_cmdbuf(uint16_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFF0000FFFFFFFFULL) | (((uint64_t)(val & 0xFFFFULL)) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8, reg_val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_clear_cmdbuf(int vc) {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = ~(1ULL << (vc + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8, val.val);
}

/* Per-iDMA-VC Has Space Interrupts (IE[63:48], IP[63:48]) */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_enable_cmdbuf(int vc) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    val.val |= (1ULL << (vc + 48));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, val.val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_disable_cmdbuf(int vc) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    val.val &= ~(1ULL << (vc + 48));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, val.val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint16_t per_idma_vc_has_space_get_interrupt_enable_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    return (val >> 48) & 0xFFFFULL; /* Bits [63:48] */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_idma_vc_has_space_set_interrupt_enable_cmdbuf(uint16_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    reg_val = (reg_val & 0x0000FFFFFFFFFFFFULL) | (((uint64_t)(val & 0xFFFFULL)) << 48);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, reg_val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t per_idma_vc_has_space_interrupts_pending_cmdbuf() {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8);
    return (val.val >> 48) & 0xFFFFULL; /* Bits [63:48] shifted to [15:0] */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint16_t per_idma_vc_has_space_get_interrupt_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8);
    return (val >> 48) & 0xFFFFULL; /* Bits [63:48] */
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_idma_vc_has_space_set_interrupt_pending_cmdbuf(uint16_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8);
    reg_val = (reg_val & 0x0000FFFFFFFFFFFFULL) | (((uint64_t)(val & 0xFFFFULL)) << 48);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8, reg_val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_clear_cmdbuf(int vc) {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = ~(1ULL << (vc + 48));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8, val.val);
}

/*
 * @fn noc_fast_read_cmdbuf<CMDBUF>
 *
 * @brief Standalone noc read function which uses custom ASM instruction, bypassing all configurations
 * Does not need any other configuring
 *
 * @param src_addr Remote source address
 * @param dest_addr Local L1 destination address
 * @param len_bytes Size of data in bytes
 * @param has_xy Flag for specifying if address contains noc coordinates using NOC_XY_COORD
 * @param posted Flag if transfer should be posted or not
 * @param snoop Flag for enabling snoop bit
 * @param flush Flag for enabling flush bit
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void noc_fast_read_cmdbuf(
    uint64_t src_addr,
    uint32_t dest_addr,
    uint64_t len_bytes,
    uint64_t has_xy = 1,
    uint64_t posted = 0,
    uint64_t snoop = 0,
    uint64_t flush = 0) {
    uint64_t rs1 = (len_bytes << 32) | dest_addr;
    uint64_t rs2 = src_addr | (has_xy << 60) | (posted << 61) | (snoop << 62) | (flush << 63);
    __builtin_riscv_ttrocc_cmdbuf_issue_read2_trans(CMDBUF, rs1, rs2);
}

/*
 * @fn noc_read_cmdbuf<CMDBUF>
 *
 * @brief Noc read function which uses other function from this API to configure all needed
 *
 * @param src_coordinate Coordinate of source core packed with NOC_XY_COORD macro
 * @param src_addr Remote source address
 * @param dest_coordinate Coordinate of destination core packed with NOC_XY_COORD macro
 * @param dest_addr Local L1 destination address
 * @param len_bytes Size of data in bytes
 * @param transaction_id Transaction ID for this operation
 * @param snoop_bit Flag for enabling snoop bit
 * @param flush_bit Flag for enabling flush bit
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void noc_read_cmdbuf(
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    bool snoop_bit = false,
    bool flush_bit = false) {
    reset_cmdbuf<CMDBUF>();
    setup_as_copy_cmdbuf<CMDBUF>(false, false, {0}, false);
    setup_ongoing_cmdbuf<CMDBUF>(false, false, false, false, false);
    setup_vcs_cmdbuf<CMDBUF>(false);
    setup_trids_cmdbuf<CMDBUF>(transaction_id);
    if (snoop_bit || flush_bit) {
        setup_packet_tags_cmdbuf<CMDBUF>(snoop_bit, flush_bit);
    }
    set_src_cmdbuf<CMDBUF>(src_addr, src_coordinate);
    set_dest_cmdbuf<CMDBUF>(dest_addr, dest_coordinate);
    set_len_cmdbuf<CMDBUF>(len_bytes);
    issue_read_cmdbuf<CMDBUF>();
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void noc_write_prep_cmdbuf(
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    bool mcast = false,
    bool snoop_bit = false,
    bool flush_bit = false,
    bool posted = true) {
    reset_cmdbuf<CMDBUF>();
    setup_as_copy_cmdbuf<CMDBUF>(true, mcast, {0}, false, posted);
    setup_ongoing_cmdbuf<CMDBUF>(false, false, false, false, false);
    setup_vcs_cmdbuf<CMDBUF>(true, mcast);
    setup_trids_cmdbuf<CMDBUF>(transaction_id);
    if (snoop_bit || flush_bit) {
        setup_packet_tags_cmdbuf<CMDBUF>(snoop_bit, flush_bit);
    }
    set_src_cmdbuf<CMDBUF>(src_addr, src_coordinate);
    set_dest_cmdbuf<CMDBUF>(dest_addr, dest_coordinate);
    set_len_cmdbuf<CMDBUF>(len_bytes);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void noc_read_prep_cmdbuf(
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    bool snoop_bit = false,
    bool flush_bit = false) {
    reset_cmdbuf<CMDBUF>();
    setup_as_copy_cmdbuf<CMDBUF>(false, false, {0}, false);
    setup_ongoing_cmdbuf<CMDBUF>(false, false, false, false, false);
    setup_vcs_cmdbuf<CMDBUF>(false);
    setup_trids_cmdbuf<CMDBUF>(transaction_id);
    if (snoop_bit || flush_bit) {
        setup_packet_tags_cmdbuf<CMDBUF>(snoop_bit, flush_bit);
    }
    set_src_cmdbuf<CMDBUF>(src_addr, src_coordinate);
    set_dest_cmdbuf<CMDBUF>(dest_addr, dest_coordinate);
    set_len_cmdbuf<CMDBUF>(len_bytes);
}

/*
 * @fn noc_fast_write_cmdbuf<CMDBUF>
 *
 * @brief Standalone noc write function which uses custom ASM instruction, bypassing all configurations
 * Does not need any other configuring (reset cmd buffer before use)
 *
 * @param src_addr Local L1 source address
 * @param dest_addr Remote destination address
 * @param len_bytes Size of data in bytes
 * @param has_xy Flag for specifying if address contains noc coordinates using NOC_XY_COORD
 * @param posted Flag if transfer should be posted or not
 * @param snoop Flag for enabling snoop bit
 * @param flush Flag for enabling flush bit
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void noc_fast_write_cmdbuf(
    uint32_t src_addr,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint64_t has_xy = 1,
    uint64_t posted = 0,
    uint64_t snoop = 0,
    uint64_t flush = 0) {
    uint64_t rs1 = (len_bytes << 32) | src_addr;
    uint64_t rs2 = dest_addr | (has_xy << 60) | (posted << 61) | (snoop << 62) | (flush << 63);
    __builtin_riscv_ttrocc_cmdbuf_issue_write2_trans(CMDBUF, rs1, rs2);
}

/*
 * @fn noc_write_cmdbuf<CMDBUF>
 *
 * @brief Noc write function which uses other function from this API to configure all needed
 *
 * @param src_coordinate Coordinate of source core
 * @param src_addr Local L1 source address
 * @param dest_coordinate Coordinate of destination core
 * @param dest_addr Remote destination address
 * @param len_bytes Size of data in bytes
 * @param transaction_id Transaction ID for this operation
 * @param mcast Enable multicast
 * @param snoop_bit Flag for enabling snoop bit
 * @param flush_bit Flag for enabling flush bit
 * @param posted Flag if transfer should be posted or not
 * @param mcast_exclude Multicast exclusion settings
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void noc_write_cmdbuf(
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    bool mcast = false,
    bool snoop_bit = false,
    bool flush_bit = false,
    bool posted = true,
    TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u mcast_exclude = {0}) {
    reset_cmdbuf<CMDBUF>();
    setup_as_copy_cmdbuf<CMDBUF>(true, mcast, mcast_exclude, false, posted);
    setup_ongoing_cmdbuf<CMDBUF>(false, false, false, false, false);
    setup_vcs_cmdbuf<CMDBUF>(true, mcast);
    setup_trids_cmdbuf<CMDBUF>(transaction_id);
    if (snoop_bit || flush_bit) {
        setup_packet_tags_cmdbuf<CMDBUF>(snoop_bit, flush_bit);
    }
    set_src_cmdbuf<CMDBUF>(src_addr, src_coordinate);
    set_dest_cmdbuf<CMDBUF>(dest_addr, dest_coordinate);
    set_len_cmdbuf<CMDBUF>(len_bytes);
    issue_write_cmdbuf<CMDBUF>();
}

/*
 * @fn idma_copy_cmdbuf<CMDBUF>
 *
 * @brief Complete iDMA copy operation
 *
 * @param src_addr Source address
 * @param dest_addr Destination address
 * @param len_bytes Size of data in bytes
 * @param transaction_id Transaction ID for this operation
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void idma_copy_cmdbuf(
    uint64_t src_addr, uint64_t dest_addr, uint64_t len_bytes, uint32_t transaction_id = CMDBUF_DEF_TRID) {
    reset_cmdbuf<CMDBUF>();
    idma_setup_as_copy_cmdbuf<CMDBUF>(false);
    setup_ongoing_cmdbuf<CMDBUF>(false, false, false, false, false);
    setup_vcs_cmdbuf<CMDBUF>(true);
    setup_trids_cmdbuf<CMDBUF>(transaction_id);
    set_src_cmdbuf<CMDBUF>(src_addr);
    set_dest_cmdbuf<CMDBUF>(dest_addr);
    set_len_cmdbuf<CMDBUF>(len_bytes);
    issue_cmdbuf<CMDBUF>();
}

/*
 * @fn noc_atomic_increment_cmdbuf<CMDBUF>
 *
 * @brief Atomic increment function
 *
 * @param noc_coordinate NOC coordinate of target
 * @param addr Remote destination address
 * @param incr Increment value
 * @param wrap Wrap value
 * @param snoop_bit Flag for enabling snoop bit
 * @param flush_bit Flag for enabling flush bit
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void noc_atomic_increment_cmdbuf(
    uint64_t noc_coordinate,
    uint64_t addr,
    uint32_t incr = 1,
    uint32_t wrap = 31,
    bool snoop_bit = false,
    bool flush_bit = false) {
    uint64_t at_len =
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0);
    setup_as_atomic_cmdbuf<CMDBUF>(true);
    if (snoop_bit || flush_bit) {
        setup_packet_tags_cmdbuf<CMDBUF>(snoop_bit, flush_bit);
    }
    set_dest_cmdbuf<CMDBUF>(addr, noc_coordinate);
    set_len_cmdbuf<CMDBUF>(at_len);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_INLINE_DATA_REG_OFFSET / 8, (uint64_t)incr);
    issue_cmdbuf<CMDBUF>();
}

/*
 * @fn free_space_cmdbuf<CMDBUF>
 *
 * @brief Returns amount of free buffer space in virtual channel buffer
 *
 * @param vc Virtual channel ID
 * @return Amount in bytes
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t free_space_cmdbuf(uint32_t vc) {
    return __builtin_riscv_ttrocc_cmdbuf_get_vc_space_vc(CMDBUF, vc);
}
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t free_space_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_get_vc_space(CMDBUF);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t idma_free_space_cmdbuf(uint32_t vc) {
    return __builtin_riscv_ttrocc_cmdbuf_idma_get_vc_space_vc(CMDBUF, vc);
}
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) uint64_t idma_free_space_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_idma_get_vc_space(CMDBUF);
}

/*
 * @fn noc_reads_acked_cmdbuf<CMDBUF>
 *
 * @brief Checks if transaction with argument trid is completed
 *
 * @param transaction_id Transaction id to check
 * @return True if all transaction is complected
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) bool noc_reads_acked_cmdbuf(uint32_t transaction_id) {
    return __builtin_riscv_ttrocc_cmdbuf_tr_ack_trid(CMDBUF, transaction_id) == 0;
}
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) bool noc_reads_acked_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_tr_ack(CMDBUF) == 0;
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) bool all_noc_reads_acked_cmdbuf() {
    bool all = true;
    if constexpr (CMDBUF == CMDBUF_0) {
        for (uint32_t k = CMDBUF_0_TRID_STATIC; k <= CMDBUF_0_TRID_END; k++) {
            all = all && __builtin_riscv_ttrocc_cmdbuf_tr_ack_trid(CMDBUF, k) == 0;
        }
    } else {
        for (uint32_t k = CMDBUF_1_TRID_STATIC; k <= CMDBUF_1_TRID_END; k++) {
            all = all && __builtin_riscv_ttrocc_cmdbuf_tr_ack_trid(CMDBUF, k) == 0;
        }
    }
    return all;
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) bool idma_acked_cmdbuf(uint32_t transaction_id) {
    return __builtin_riscv_ttrocc_cmdbuf_idma_tr_ack_trid(CMDBUF, transaction_id) == 0;
}
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) bool idma_acked_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_idma_tr_ack(CMDBUF) == 0;
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) bool all_idma_acked_cmdbuf() {
    bool all = true;
    if constexpr (CMDBUF == CMDBUF_0) {
        for (uint32_t k = CMDBUF_0_TRID_STATIC; k <= CMDBUF_0_TRID_END; k++) {
            all = all && __builtin_riscv_ttrocc_cmdbuf_idma_tr_ack_trid(CMDBUF, k) == 0;
        }
    } else {
        for (uint32_t k = CMDBUF_1_TRID_STATIC; k <= CMDBUF_1_TRID_END; k++) {
            all = all && __builtin_riscv_ttrocc_cmdbuf_idma_tr_ack_trid(CMDBUF, k) == 0;
        }
    }
    return all;
}

/*
 * @fn noc_writes_sent_cmdbuf<CMDBUF>
 *
 * @brief Checks if write with provided transaction ID is completed
 *
 * @param transaction_id Transaction id to check
 * @return True if write is complected
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) bool noc_writes_sent_cmdbuf(uint32_t transaction_id) {
    return __builtin_riscv_ttrocc_cmdbuf_wr_sent_trid(CMDBUF, transaction_id) == 0;
}
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) bool noc_writes_sent_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_wr_sent(CMDBUF) == 0;
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) bool all_noc_writes_sent_cmdbuf() {
    bool all = true;
    if constexpr (CMDBUF == CMDBUF_0) {
        for (uint32_t k = CMDBUF_0_TRID_STATIC; k <= CMDBUF_0_TRID_END; k++) {
            all = all && __builtin_riscv_ttrocc_cmdbuf_wr_sent_trid(CMDBUF, k) == 0;
        }
    } else {
        for (uint32_t k = CMDBUF_1_TRID_STATIC; k <= CMDBUF_1_TRID_END; k++) {
            all = all && __builtin_riscv_ttrocc_cmdbuf_wr_sent_trid(CMDBUF, k) == 0;
        }
    }
    return all;
}

/*
 * @fn noc_nonposted_writes_acked_cmdbuf<CMDBUF>
 *
 * @brief Checks if nonposted write transaction is acknowledged
 *
 * @param transaction_id Transaction id to check
 * @return True if nonposted write is acknowledged
 *
 */
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_cmdbuf(uint32_t transaction_id) {
    return __builtin_riscv_ttrocc_cmdbuf_tr_ack_trid(CMDBUF, transaction_id) == 0;
}
template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_tr_ack(CMDBUF) == 0;
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) bool all_noc_nonposted_writes_acked_cmdbuf() {
    bool all = true;
    if constexpr (CMDBUF == CMDBUF_0) {
        for (uint32_t k = CMDBUF_0_TRID_STATIC; k <= CMDBUF_0_TRID_END; k++) {
            all = all && __builtin_riscv_ttrocc_cmdbuf_tr_ack_trid(CMDBUF, k) == 0;
        }
    } else {
        for (uint32_t k = CMDBUF_1_TRID_STATIC; k <= CMDBUF_1_TRID_END; k++) {
            all = all && __builtin_riscv_ttrocc_cmdbuf_tr_ack_trid(CMDBUF, k) == 0;
        }
    }
    return all;
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void l1_atomic_instr_cmdbuf(uint32_t fmt, bool no_sat, uint32_t atomic_op) {
    TT_ROCC_CMD_BUF_L1_ACCUM_CFG_reg_u l1_atomic_instr;
    l1_atomic_instr.val = TT_ROCC_CMD_BUF_L1_ACCUM_CFG_REG_DEFAULT;

    l1_atomic_instr.f.l1_atomic_fmt = fmt;
    l1_atomic_instr.f.disable_sat = no_sat;
    l1_atomic_instr.f.l1_atomic_operation = atomic_op;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_L1_ACCUM_CFG_REG_OFFSET / 8, l1_atomic_instr.val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void idma_setup_as_atomic_accum_cmdbuf(bool wrapping_en = true) {
    TT_ROCC_CMD_BUF_MISC_reg_u misc;
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

    misc.f.write_trans = 1;
    misc.f.idma_en = 1;
    misc.f.wrapping_en = wrapping_en;
    misc.f.l1_accum_en = 1;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc.val);
}

template <uint32_t CMDBUF, typename = cmdbuf_id_t<CMDBUF>>
inline __attribute__((always_inline)) void idma_l1_atomic_accum_cmdbuf(
    uint64_t src_addr,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    uint64_t fmt = 0x0,
    uint64_t op = 0x9) {
    reset_cmdbuf<CMDBUF>();
    idma_setup_as_atomic_accum_cmdbuf<CMDBUF>(false);
    l1_atomic_instr_cmdbuf<CMDBUF>(fmt, false, op);
    setup_ongoing_cmdbuf<CMDBUF>(false, false, false, false, false);
    setup_vcs_cmdbuf<CMDBUF>(true);
    setup_trids_cmdbuf<CMDBUF>(transaction_id);
    set_src_cmdbuf<CMDBUF>(src_addr);
    set_dest_cmdbuf<CMDBUF>(dest_addr);
    set_len_cmdbuf<CMDBUF>(len_bytes);
    issue_cmdbuf<CMDBUF>();
}

/* Per-cmdbuf aliases: cmdbuf_0/_1 spellings of the CMDBUF-templated functions above. */
inline __attribute__((always_inline)) void reset_cmdbuf_0() { return reset_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) void reset_cmdbuf_1() { return reset_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) void setup_as_copy_cmdbuf_0(
    bool wr,
    bool mcast = false,
    TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u mcast_exclude = {0},
    bool wrapping_en = true,
    bool posted = true) {
    return setup_as_copy_cmdbuf<CMDBUF_0>(wr, mcast, mcast_exclude, wrapping_en, posted);
}
inline __attribute__((always_inline)) void setup_as_copy_cmdbuf_1(
    bool wr,
    bool mcast = false,
    TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u mcast_exclude = {0},
    bool wrapping_en = true,
    bool posted = true) {
    return setup_as_copy_cmdbuf<CMDBUF_1>(wr, mcast, mcast_exclude, wrapping_en, posted);
}
inline __attribute__((always_inline)) void setup_as_scatter_list_cmdbuf_0(
    bool wr,
    bool apply_scatter_to_dest,
    bool mcast = false,
    bool linked = false,
    TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u mcast_exclude = {0},
    bool scatter_list_contains_size = false,
    bool scatter_list_contains_xy = false,
    bool wrapping_en = true,
    bool posted = true) {
    return setup_as_scatter_list_cmdbuf<CMDBUF_0>(
        wr,
        apply_scatter_to_dest,
        mcast,
        linked,
        mcast_exclude,
        scatter_list_contains_size,
        scatter_list_contains_xy,
        wrapping_en,
        posted);
}
inline __attribute__((always_inline)) void setup_as_scatter_list_cmdbuf_1(
    bool wr,
    bool apply_scatter_to_dest,
    bool mcast = false,
    bool linked = false,
    TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u mcast_exclude = {0},
    bool scatter_list_contains_size = false,
    bool scatter_list_contains_xy = false,
    bool wrapping_en = true,
    bool posted = true) {
    return setup_as_scatter_list_cmdbuf<CMDBUF_1>(
        wr,
        apply_scatter_to_dest,
        mcast,
        linked,
        mcast_exclude,
        scatter_list_contains_size,
        scatter_list_contains_xy,
        wrapping_en,
        posted);
}
inline __attribute__((always_inline)) void setup_as_atomic_cmdbuf_0(bool wr) {
    return setup_as_atomic_cmdbuf<CMDBUF_0>(wr);
}
inline __attribute__((always_inline)) void setup_as_atomic_cmdbuf_1(bool wr) {
    return setup_as_atomic_cmdbuf<CMDBUF_1>(wr);
}
inline __attribute__((always_inline)) void idma_setup_as_copy_cmdbuf_0(bool wrapping_en = true) {
    return idma_setup_as_copy_cmdbuf<CMDBUF_0>(wrapping_en);
}
inline __attribute__((always_inline)) void idma_setup_as_copy_cmdbuf_1(bool wrapping_en = true) {
    return idma_setup_as_copy_cmdbuf<CMDBUF_1>(wrapping_en);
}
inline __attribute__((always_inline)) void idma_setup_as_scatter_list_cmdbuf_0(
    bool apply_scatter_to_dest,
    bool scatter_list_contains_size = false,
    bool scatter_list_contains_xy = false,
    bool wrapping_en = true) {
    return idma_setup_as_scatter_list_cmdbuf<CMDBUF_0>(
        apply_scatter_to_dest, scatter_list_contains_size, scatter_list_contains_xy, wrapping_en);
}
inline __attribute__((always_inline)) void idma_setup_as_scatter_list_cmdbuf_1(
    bool apply_scatter_to_dest,
    bool scatter_list_contains_size = false,
    bool scatter_list_contains_xy = false,
    bool wrapping_en = true) {
    return idma_setup_as_scatter_list_cmdbuf<CMDBUF_1>(
        apply_scatter_to_dest, scatter_list_contains_size, scatter_list_contains_xy, wrapping_en);
}
inline __attribute__((always_inline)) void set_axi_opt_1_cmdbuf_0(uint8_t src_protocol, uint8_t decouple_aw) {
    return set_axi_opt_1_cmdbuf<CMDBUF_0>(src_protocol, decouple_aw);
}
inline __attribute__((always_inline)) void set_axi_opt_1_cmdbuf_1(uint8_t src_protocol, uint8_t decouple_aw) {
    return set_axi_opt_1_cmdbuf<CMDBUF_1>(src_protocol, decouple_aw);
}
inline __attribute__((always_inline)) void setup_ongoing_cmdbuf_0(
    bool src_addr_inc_en,
    bool dest_addr_inc_en,
    bool trid_inc_en,
    bool req_vc_inc_en,
    bool resp_vc_inc_en,
    bool req_vc_inc_on_entire_trans = false,
    bool resp_vc_inc_on_entire_trans = false) {
    return setup_ongoing_cmdbuf<CMDBUF_0>(
        src_addr_inc_en,
        dest_addr_inc_en,
        trid_inc_en,
        req_vc_inc_en,
        resp_vc_inc_en,
        req_vc_inc_on_entire_trans,
        resp_vc_inc_on_entire_trans);
}
inline __attribute__((always_inline)) void setup_ongoing_cmdbuf_1(
    bool src_addr_inc_en,
    bool dest_addr_inc_en,
    bool trid_inc_en,
    bool req_vc_inc_en,
    bool resp_vc_inc_en,
    bool req_vc_inc_on_entire_trans = false,
    bool resp_vc_inc_on_entire_trans = false) {
    return setup_ongoing_cmdbuf<CMDBUF_1>(
        src_addr_inc_en,
        dest_addr_inc_en,
        trid_inc_en,
        req_vc_inc_en,
        resp_vc_inc_en,
        req_vc_inc_on_entire_trans,
        resp_vc_inc_on_entire_trans);
}
inline __attribute__((always_inline)) void setup_vcs_cmdbuf_0(bool wr, bool mcast = false) {
    return setup_vcs_cmdbuf<CMDBUF_0>(wr, mcast);
}
inline __attribute__((always_inline)) void setup_vcs_cmdbuf_1(bool wr, bool mcast = false) {
    return setup_vcs_cmdbuf<CMDBUF_1>(wr, mcast);
}
inline __attribute__((always_inline)) void setup_wrapping_vcs_cmdbuf_0(
    bool wr,
    uint32_t req_start_vc,
    uint32_t req_end_vc,
    uint32_t req_vc_offset = 0,
    uint32_t resp_start_vc = 0,
    uint32_t resp_end_vc = 0,
    uint32_t resp_vc_offset = 0) {
    return setup_wrapping_vcs_cmdbuf<CMDBUF_0>(
        wr, req_start_vc, req_end_vc, req_vc_offset, resp_start_vc, resp_end_vc, resp_vc_offset);
}
inline __attribute__((always_inline)) void setup_wrapping_vcs_cmdbuf_1(
    bool wr,
    uint32_t req_start_vc,
    uint32_t req_end_vc,
    uint32_t req_vc_offset = 0,
    uint32_t resp_start_vc = 0,
    uint32_t resp_end_vc = 0,
    uint32_t resp_vc_offset = 0) {
    return setup_wrapping_vcs_cmdbuf<CMDBUF_1>(
        wr, req_start_vc, req_end_vc, req_vc_offset, resp_start_vc, resp_end_vc, resp_vc_offset);
}
inline __attribute__((always_inline)) void setup_trids_cmdbuf_0(
    uint32_t trid_offset = CMDBUF_DEF_TRID, bool wrapping = false) {
    return setup_trids_cmdbuf<CMDBUF_0>(trid_offset, wrapping);
}
inline __attribute__((always_inline)) void setup_trids_cmdbuf_1(
    uint32_t trid_offset = CMDBUF_DEF_TRID, bool wrapping = false) {
    return setup_trids_cmdbuf<CMDBUF_1>(trid_offset, wrapping);
}
inline __attribute__((always_inline)) uint32_t swap_trid_cmdbuf_0(uint32_t new_trid) {
    return swap_trid_cmdbuf<CMDBUF_0>(new_trid);
}
inline __attribute__((always_inline)) uint32_t swap_trid_cmdbuf_1(uint32_t new_trid) {
    return swap_trid_cmdbuf<CMDBUF_1>(new_trid);
}
inline __attribute__((always_inline)) void setup_max_bytes_in_packet_cmdbuf_0(uint64_t max_bytes_in_packet) {
    return setup_max_bytes_in_packet_cmdbuf<CMDBUF_0>(max_bytes_in_packet);
}
inline __attribute__((always_inline)) void setup_max_bytes_in_packet_cmdbuf_1(uint64_t max_bytes_in_packet) {
    return setup_max_bytes_in_packet_cmdbuf<CMDBUF_1>(max_bytes_in_packet);
}
inline __attribute__((always_inline)) void setup_packet_tags_cmdbuf_0(bool snoop_bit, bool flush_bit) {
    return setup_packet_tags_cmdbuf<CMDBUF_0>(snoop_bit, flush_bit);
}
inline __attribute__((always_inline)) void setup_packet_tags_cmdbuf_1(bool snoop_bit, bool flush_bit) {
    return setup_packet_tags_cmdbuf<CMDBUF_1>(snoop_bit, flush_bit);
}
inline __attribute__((always_inline)) uint64_t get_src_cmdbuf_0() { return get_src_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) uint64_t get_src_cmdbuf_1() { return get_src_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) void set_src_cmdbuf_0(
    uint64_t addr, uint64_t coordinate, uint64_t base, uint64_t size) {
    return set_src_cmdbuf<CMDBUF_0>(addr, coordinate, base, size);
}
inline __attribute__((always_inline)) void set_src_cmdbuf_1(
    uint64_t addr, uint64_t coordinate, uint64_t base, uint64_t size) {
    return set_src_cmdbuf<CMDBUF_1>(addr, coordinate, base, size);
}
inline __attribute__((always_inline)) void set_src_cmdbuf_0(uint64_t addr, uint64_t coordinate, uint64_t base) {
    return set_src_cmdbuf<CMDBUF_0>(addr, coordinate, base);
}
inline __attribute__((always_inline)) void set_src_cmdbuf_1(uint64_t addr, uint64_t coordinate, uint64_t base) {
    return set_src_cmdbuf<CMDBUF_1>(addr, coordinate, base);
}
inline __attribute__((always_inline)) void set_src_cmdbuf_0(uint64_t addr, uint64_t coordinate) {
    return set_src_cmdbuf<CMDBUF_0>(addr, coordinate);
}
inline __attribute__((always_inline)) void set_src_cmdbuf_1(uint64_t addr, uint64_t coordinate) {
    return set_src_cmdbuf<CMDBUF_1>(addr, coordinate);
}
inline __attribute__((always_inline)) void set_src_cmdbuf_0(uint64_t addr) { return set_src_cmdbuf<CMDBUF_0>(addr); }
inline __attribute__((always_inline)) void set_src_cmdbuf_1(uint64_t addr) { return set_src_cmdbuf<CMDBUF_1>(addr); }
inline __attribute__((always_inline)) uint64_t get_dest_cmdbuf_0() { return get_dest_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) uint64_t get_dest_cmdbuf_1() { return get_dest_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) void set_dest_cmdbuf_0(
    uint64_t addr, uint64_t coordinates, uint64_t base, uint64_t size) {
    return set_dest_cmdbuf<CMDBUF_0>(addr, coordinates, base, size);
}
inline __attribute__((always_inline)) void set_dest_cmdbuf_1(
    uint64_t addr, uint64_t coordinates, uint64_t base, uint64_t size) {
    return set_dest_cmdbuf<CMDBUF_1>(addr, coordinates, base, size);
}
inline __attribute__((always_inline)) void set_dest_cmdbuf_0(uint64_t addr, uint64_t coordinates, uint64_t base) {
    return set_dest_cmdbuf<CMDBUF_0>(addr, coordinates, base);
}
inline __attribute__((always_inline)) void set_dest_cmdbuf_1(uint64_t addr, uint64_t coordinates, uint64_t base) {
    return set_dest_cmdbuf<CMDBUF_1>(addr, coordinates, base);
}
inline __attribute__((always_inline)) void set_dest_cmdbuf_0(uint64_t addr, uint64_t coordinates) {
    return set_dest_cmdbuf<CMDBUF_0>(addr, coordinates);
}
inline __attribute__((always_inline)) void set_dest_cmdbuf_1(uint64_t addr, uint64_t coordinates) {
    return set_dest_cmdbuf<CMDBUF_1>(addr, coordinates);
}
inline __attribute__((always_inline)) void set_dest_cmdbuf_0(uint64_t addr) { return set_dest_cmdbuf<CMDBUF_0>(addr); }
inline __attribute__((always_inline)) void set_dest_cmdbuf_1(uint64_t addr) { return set_dest_cmdbuf<CMDBUF_1>(addr); }
inline __attribute__((always_inline)) void set_scatter_list_cmdbuf_0(
    uint64_t addr, uint64_t base, uint64_t index, uint64_t times) {
    return set_scatter_list_cmdbuf<CMDBUF_0>(addr, base, index, times);
}
inline __attribute__((always_inline)) void set_scatter_list_cmdbuf_1(
    uint64_t addr, uint64_t base, uint64_t index, uint64_t times) {
    return set_scatter_list_cmdbuf<CMDBUF_1>(addr, base, index, times);
}
inline __attribute__((always_inline)) void set_scatter_list_base_cmdbuf_0(uint64_t base) {
    return set_scatter_list_base_cmdbuf<CMDBUF_0>(base);
}
inline __attribute__((always_inline)) void set_scatter_list_base_cmdbuf_1(uint64_t base) {
    return set_scatter_list_base_cmdbuf<CMDBUF_1>(base);
}
inline __attribute__((always_inline)) void set_scatter_list_index_cmdbuf_0(uint64_t index) {
    return set_scatter_list_index_cmdbuf<CMDBUF_0>(index);
}
inline __attribute__((always_inline)) void set_scatter_list_index_cmdbuf_1(uint64_t index) {
    return set_scatter_list_index_cmdbuf<CMDBUF_1>(index);
}
inline __attribute__((always_inline)) void set_scatter_list_times_cmdbuf_0(uint64_t times) {
    return set_scatter_list_times_cmdbuf<CMDBUF_0>(times);
}
inline __attribute__((always_inline)) void set_scatter_list_times_cmdbuf_1(uint64_t times) {
    return set_scatter_list_times_cmdbuf<CMDBUF_1>(times);
}
inline __attribute__((always_inline)) uint64_t get_scatter_list_addr_cmdbuf_0() {
    return get_scatter_list_addr_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t get_scatter_list_addr_cmdbuf_1() {
    return get_scatter_list_addr_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint64_t get_scatter_list_base_cmdbuf_0() {
    return get_scatter_list_base_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t get_scatter_list_base_cmdbuf_1() {
    return get_scatter_list_base_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint64_t get_scatter_list_index_cmdbuf_0() {
    return get_scatter_list_index_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t get_scatter_list_index_cmdbuf_1() {
    return get_scatter_list_index_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint64_t get_scatter_list_times_cmdbuf_0() {
    return get_scatter_list_times_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t get_scatter_list_times_cmdbuf_1() {
    return get_scatter_list_times_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void set_len_cmdbuf_0(uint64_t size_bytes) {
    return set_len_cmdbuf<CMDBUF_0>(size_bytes);
}
inline __attribute__((always_inline)) void set_len_cmdbuf_1(uint64_t size_bytes) {
    return set_len_cmdbuf<CMDBUF_1>(size_bytes);
}
inline __attribute__((always_inline)) void issue_cmdbuf_0() { return issue_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) void issue_cmdbuf_1() { return issue_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) void issue_read_cmdbuf_0() { return issue_read_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) void issue_read_cmdbuf_1() { return issue_read_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) void issue_write_cmdbuf_0() { return issue_write_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) void issue_write_cmdbuf_1() { return issue_write_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) void issue_write_inline_cmdbuf_0(uint64_t data) {
    return issue_write_inline_cmdbuf<CMDBUF_0>(data);
}
inline __attribute__((always_inline)) void issue_write_inline_cmdbuf_1(uint64_t data) {
    return issue_write_inline_cmdbuf<CMDBUF_1>(data);
}
inline __attribute__((always_inline)) void issue_write_inline_len_cmdbuf_0(
    uint64_t data,
    uint64_t dest_addr,
    uint64_t size_bytes,
    uint64_t has_xy = 1,
    uint64_t posted = 0,
    uint64_t snoop = 0,
    uint64_t flush = 0) {
    return issue_write_inline_len_cmdbuf<CMDBUF_0>(data, dest_addr, size_bytes, has_xy, posted, snoop, flush);
}
inline __attribute__((always_inline)) void issue_write_inline_len_cmdbuf_1(
    uint64_t data,
    uint64_t dest_addr,
    uint64_t size_bytes,
    uint64_t has_xy = 1,
    uint64_t posted = 0,
    uint64_t snoop = 0,
    uint64_t flush = 0) {
    return issue_write_inline_len_cmdbuf<CMDBUF_1>(data, dest_addr, size_bytes, has_xy, posted, snoop, flush);
}
inline __attribute__((always_inline)) void interrupt_enable_cmdbuf_0(int id) {
    return interrupt_enable_cmdbuf<CMDBUF_0>(id);
}
inline __attribute__((always_inline)) void interrupt_enable_cmdbuf_1(int id) {
    return interrupt_enable_cmdbuf<CMDBUF_1>(id);
}
inline __attribute__((always_inline)) void interrupt_disable_cmdbuf_0(int id) {
    return interrupt_disable_cmdbuf<CMDBUF_0>(id);
}
inline __attribute__((always_inline)) void interrupt_disable_cmdbuf_1(int id) {
    return interrupt_disable_cmdbuf<CMDBUF_1>(id);
}
inline __attribute__((always_inline)) uint64_t interrupts_pending_cmdbuf_0() {
    return interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t interrupts_pending_cmdbuf_1() {
    return interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void interrupt_clear_cmdbuf_0(int id) {
    return interrupt_clear_cmdbuf<CMDBUF_0>(id);
}
inline __attribute__((always_inline)) void interrupt_clear_cmdbuf_1(int id) {
    return interrupt_clear_cmdbuf<CMDBUF_1>(id);
}
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_enable_cmdbuf_0(int trid) {
    return per_trid_count_zero_interrupt_enable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_enable_cmdbuf_1(int trid) {
    return per_trid_count_zero_interrupt_enable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_disable_cmdbuf_0(int trid) {
    return per_trid_count_zero_interrupt_disable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_disable_cmdbuf_1(int trid) {
    return per_trid_count_zero_interrupt_disable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) uint32_t per_trid_count_zero_get_interrupt_enable_cmdbuf_0() {
    return per_trid_count_zero_get_interrupt_enable_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_count_zero_get_interrupt_enable_cmdbuf_1() {
    return per_trid_count_zero_get_interrupt_enable_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_count_zero_set_interrupt_enable_cmdbuf_0(uint32_t val) {
    return per_trid_count_zero_set_interrupt_enable_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_count_zero_set_interrupt_enable_cmdbuf_1(uint32_t val) {
    return per_trid_count_zero_set_interrupt_enable_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) uint64_t per_trid_count_zero_interrupts_pending_cmdbuf_0() {
    return per_trid_count_zero_interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t per_trid_count_zero_interrupts_pending_cmdbuf_1() {
    return per_trid_count_zero_interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint32_t per_trid_count_zero_get_interrupt_pending_cmdbuf_0() {
    return per_trid_count_zero_get_interrupt_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_count_zero_get_interrupt_pending_cmdbuf_1() {
    return per_trid_count_zero_get_interrupt_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_count_zero_set_interrupt_pending_cmdbuf_0(uint32_t val) {
    return per_trid_count_zero_set_interrupt_pending_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_count_zero_set_interrupt_pending_cmdbuf_1(uint32_t val) {
    return per_trid_count_zero_set_interrupt_pending_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_clear_cmdbuf_0(int trid) {
    return per_trid_count_zero_interrupt_clear_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_clear_cmdbuf_1(int trid) {
    return per_trid_count_zero_interrupt_clear_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_enable_cmdbuf_0(int trid) {
    return per_trid_wr_count_zero_interrupt_enable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_enable_cmdbuf_1(int trid) {
    return per_trid_wr_count_zero_interrupt_enable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_disable_cmdbuf_0(int trid) {
    return per_trid_wr_count_zero_interrupt_disable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_disable_cmdbuf_1(int trid) {
    return per_trid_wr_count_zero_interrupt_disable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) uint32_t per_trid_wr_count_zero_get_interrupt_enable_cmdbuf_0() {
    return per_trid_wr_count_zero_get_interrupt_enable_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_wr_count_zero_get_interrupt_enable_cmdbuf_1() {
    return per_trid_wr_count_zero_get_interrupt_enable_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_set_interrupt_enable_cmdbuf_0(uint32_t val) {
    return per_trid_wr_count_zero_set_interrupt_enable_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_set_interrupt_enable_cmdbuf_1(uint32_t val) {
    return per_trid_wr_count_zero_set_interrupt_enable_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) uint64_t per_trid_wr_count_zero_interrupts_pending_cmdbuf_0() {
    return per_trid_wr_count_zero_interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t per_trid_wr_count_zero_interrupts_pending_cmdbuf_1() {
    return per_trid_wr_count_zero_interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint32_t per_trid_wr_count_zero_get_interrupt_pending_cmdbuf_0() {
    return per_trid_wr_count_zero_get_interrupt_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_wr_count_zero_get_interrupt_pending_cmdbuf_1() {
    return per_trid_wr_count_zero_get_interrupt_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_set_interrupt_pending_cmdbuf_0(uint32_t val) {
    return per_trid_wr_count_zero_set_interrupt_pending_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_set_interrupt_pending_cmdbuf_1(uint32_t val) {
    return per_trid_wr_count_zero_set_interrupt_pending_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_clear_cmdbuf_0(int trid) {
    return per_trid_wr_count_zero_interrupt_clear_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_clear_cmdbuf_1(int trid) {
    return per_trid_wr_count_zero_interrupt_clear_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_enable_cmdbuf_0(int trid) {
    return per_trid_idma_count_zero_interrupt_enable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_enable_cmdbuf_1(int trid) {
    return per_trid_idma_count_zero_interrupt_enable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_disable_cmdbuf_0(int trid) {
    return per_trid_idma_count_zero_interrupt_disable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_disable_cmdbuf_1(int trid) {
    return per_trid_idma_count_zero_interrupt_disable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) uint32_t per_trid_idma_count_zero_get_interrupt_enable_cmdbuf_0() {
    return per_trid_idma_count_zero_get_interrupt_enable_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_idma_count_zero_get_interrupt_enable_cmdbuf_1() {
    return per_trid_idma_count_zero_get_interrupt_enable_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_set_interrupt_enable_cmdbuf_0(uint32_t val) {
    return per_trid_idma_count_zero_set_interrupt_enable_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_set_interrupt_enable_cmdbuf_1(uint32_t val) {
    return per_trid_idma_count_zero_set_interrupt_enable_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) uint64_t per_trid_idma_count_zero_interrupts_pending_cmdbuf_0() {
    return per_trid_idma_count_zero_interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t per_trid_idma_count_zero_interrupts_pending_cmdbuf_1() {
    return per_trid_idma_count_zero_interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint32_t per_trid_idma_count_zero_get_interrupt_pending_cmdbuf_0() {
    return per_trid_idma_count_zero_get_interrupt_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_idma_count_zero_get_interrupt_pending_cmdbuf_1() {
    return per_trid_idma_count_zero_get_interrupt_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_set_interrupt_pending_cmdbuf_0(uint32_t val) {
    return per_trid_idma_count_zero_set_interrupt_pending_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_set_interrupt_pending_cmdbuf_1(uint32_t val) {
    return per_trid_idma_count_zero_set_interrupt_pending_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_clear_cmdbuf_0(int trid) {
    return per_trid_idma_count_zero_interrupt_clear_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_clear_cmdbuf_1(int trid) {
    return per_trid_idma_count_zero_interrupt_clear_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_enable_cmdbuf_0(int trid) {
    return per_trid_tiles_to_process_interrupt_enable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_enable_cmdbuf_1(int trid) {
    return per_trid_tiles_to_process_interrupt_enable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_disable_cmdbuf_0(int trid) {
    return per_trid_tiles_to_process_interrupt_disable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_disable_cmdbuf_1(int trid) {
    return per_trid_tiles_to_process_interrupt_disable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) uint32_t per_trid_tiles_to_process_get_interrupt_enable_cmdbuf_0() {
    return per_trid_tiles_to_process_get_interrupt_enable_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_tiles_to_process_get_interrupt_enable_cmdbuf_1() {
    return per_trid_tiles_to_process_get_interrupt_enable_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_set_interrupt_enable_cmdbuf_0(uint32_t val) {
    return per_trid_tiles_to_process_set_interrupt_enable_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_set_interrupt_enable_cmdbuf_1(uint32_t val) {
    return per_trid_tiles_to_process_set_interrupt_enable_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) uint64_t per_trid_tiles_to_process_interrupts_pending_cmdbuf_0() {
    return per_trid_tiles_to_process_interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t per_trid_tiles_to_process_interrupts_pending_cmdbuf_1() {
    return per_trid_tiles_to_process_interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint32_t per_trid_tiles_to_process_get_interrupt_pending_cmdbuf_0() {
    return per_trid_tiles_to_process_get_interrupt_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_tiles_to_process_get_interrupt_pending_cmdbuf_1() {
    return per_trid_tiles_to_process_get_interrupt_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_set_interrupt_pending_cmdbuf_0(uint32_t val) {
    return per_trid_tiles_to_process_set_interrupt_pending_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_set_interrupt_pending_cmdbuf_1(uint32_t val) {
    return per_trid_tiles_to_process_set_interrupt_pending_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_clear_cmdbuf_0(int trid) {
    return per_trid_tiles_to_process_interrupt_clear_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_clear_cmdbuf_1(int trid) {
    return per_trid_tiles_to_process_interrupt_clear_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_enable_cmdbuf_0(int trid) {
    return per_trid_wr_tiles_to_process_interrupt_enable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_enable_cmdbuf_1(int trid) {
    return per_trid_wr_tiles_to_process_interrupt_enable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_disable_cmdbuf_0(int trid) {
    return per_trid_wr_tiles_to_process_interrupt_disable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_disable_cmdbuf_1(int trid) {
    return per_trid_wr_tiles_to_process_interrupt_disable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) uint32_t per_trid_wr_tiles_to_process_get_interrupt_enable_cmdbuf_0() {
    return per_trid_wr_tiles_to_process_get_interrupt_enable_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_wr_tiles_to_process_get_interrupt_enable_cmdbuf_1() {
    return per_trid_wr_tiles_to_process_get_interrupt_enable_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_set_interrupt_enable_cmdbuf_0(uint32_t val) {
    return per_trid_wr_tiles_to_process_set_interrupt_enable_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_set_interrupt_enable_cmdbuf_1(uint32_t val) {
    return per_trid_wr_tiles_to_process_set_interrupt_enable_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) uint64_t per_trid_wr_tiles_to_process_interrupts_pending_cmdbuf_0() {
    return per_trid_wr_tiles_to_process_interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t per_trid_wr_tiles_to_process_interrupts_pending_cmdbuf_1() {
    return per_trid_wr_tiles_to_process_interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint32_t per_trid_wr_tiles_to_process_get_interrupt_pending_cmdbuf_0() {
    return per_trid_wr_tiles_to_process_get_interrupt_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_wr_tiles_to_process_get_interrupt_pending_cmdbuf_1() {
    return per_trid_wr_tiles_to_process_get_interrupt_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_set_interrupt_pending_cmdbuf_0(uint32_t val) {
    return per_trid_wr_tiles_to_process_set_interrupt_pending_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_set_interrupt_pending_cmdbuf_1(uint32_t val) {
    return per_trid_wr_tiles_to_process_set_interrupt_pending_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_clear_cmdbuf_0(int trid) {
    return per_trid_wr_tiles_to_process_interrupt_clear_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_clear_cmdbuf_1(int trid) {
    return per_trid_wr_tiles_to_process_interrupt_clear_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_enable_cmdbuf_0(int trid) {
    return per_trid_idma_tiles_to_process_interrupt_enable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_enable_cmdbuf_1(int trid) {
    return per_trid_idma_tiles_to_process_interrupt_enable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_disable_cmdbuf_0(int trid) {
    return per_trid_idma_tiles_to_process_interrupt_disable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_disable_cmdbuf_1(int trid) {
    return per_trid_idma_tiles_to_process_interrupt_disable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) uint32_t per_trid_idma_tiles_to_process_get_interrupt_enable_cmdbuf_0() {
    return per_trid_idma_tiles_to_process_get_interrupt_enable_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_idma_tiles_to_process_get_interrupt_enable_cmdbuf_1() {
    return per_trid_idma_tiles_to_process_get_interrupt_enable_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_set_interrupt_enable_cmdbuf_0(uint32_t val) {
    return per_trid_idma_tiles_to_process_set_interrupt_enable_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_set_interrupt_enable_cmdbuf_1(uint32_t val) {
    return per_trid_idma_tiles_to_process_set_interrupt_enable_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) uint64_t per_trid_idma_tiles_to_process_interrupts_pending_cmdbuf_0() {
    return per_trid_idma_tiles_to_process_interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t per_trid_idma_tiles_to_process_interrupts_pending_cmdbuf_1() {
    return per_trid_idma_tiles_to_process_interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint32_t per_trid_idma_tiles_to_process_get_interrupt_pending_cmdbuf_0() {
    return per_trid_idma_tiles_to_process_get_interrupt_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_idma_tiles_to_process_get_interrupt_pending_cmdbuf_1() {
    return per_trid_idma_tiles_to_process_get_interrupt_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_set_interrupt_pending_cmdbuf_0(uint32_t val) {
    return per_trid_idma_tiles_to_process_set_interrupt_pending_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_set_interrupt_pending_cmdbuf_1(uint32_t val) {
    return per_trid_idma_tiles_to_process_set_interrupt_pending_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_clear_cmdbuf_0(int trid) {
    return per_trid_idma_tiles_to_process_interrupt_clear_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_clear_cmdbuf_1(int trid) {
    return per_trid_idma_tiles_to_process_interrupt_clear_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_enable_cmdbuf_0(int vc) {
    return per_vc_has_space_interrupt_enable_cmdbuf<CMDBUF_0>(vc);
}
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_enable_cmdbuf_1(int vc) {
    return per_vc_has_space_interrupt_enable_cmdbuf<CMDBUF_1>(vc);
}
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_disable_cmdbuf_0(int vc) {
    return per_vc_has_space_interrupt_disable_cmdbuf<CMDBUF_0>(vc);
}
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_disable_cmdbuf_1(int vc) {
    return per_vc_has_space_interrupt_disable_cmdbuf<CMDBUF_1>(vc);
}
inline __attribute__((always_inline)) uint16_t per_vc_has_space_get_interrupt_enable_cmdbuf_0() {
    return per_vc_has_space_get_interrupt_enable_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint16_t per_vc_has_space_get_interrupt_enable_cmdbuf_1() {
    return per_vc_has_space_get_interrupt_enable_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_vc_has_space_set_interrupt_enable_cmdbuf_0(uint16_t val) {
    return per_vc_has_space_set_interrupt_enable_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_vc_has_space_set_interrupt_enable_cmdbuf_1(uint16_t val) {
    return per_vc_has_space_set_interrupt_enable_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) uint64_t per_vc_has_space_interrupts_pending_cmdbuf_0() {
    return per_vc_has_space_interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t per_vc_has_space_interrupts_pending_cmdbuf_1() {
    return per_vc_has_space_interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint16_t per_vc_has_space_get_interrupt_pending_cmdbuf_0() {
    return per_vc_has_space_get_interrupt_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint16_t per_vc_has_space_get_interrupt_pending_cmdbuf_1() {
    return per_vc_has_space_get_interrupt_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_vc_has_space_set_interrupt_pending_cmdbuf_0(uint16_t val) {
    return per_vc_has_space_set_interrupt_pending_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_vc_has_space_set_interrupt_pending_cmdbuf_1(uint16_t val) {
    return per_vc_has_space_set_interrupt_pending_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_clear_cmdbuf_0(int vc) {
    return per_vc_has_space_interrupt_clear_cmdbuf<CMDBUF_0>(vc);
}
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_clear_cmdbuf_1(int vc) {
    return per_vc_has_space_interrupt_clear_cmdbuf<CMDBUF_1>(vc);
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_enable_cmdbuf_0(int vc) {
    return per_idma_vc_has_space_interrupt_enable_cmdbuf<CMDBUF_0>(vc);
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_enable_cmdbuf_1(int vc) {
    return per_idma_vc_has_space_interrupt_enable_cmdbuf<CMDBUF_1>(vc);
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_disable_cmdbuf_0(int vc) {
    return per_idma_vc_has_space_interrupt_disable_cmdbuf<CMDBUF_0>(vc);
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_disable_cmdbuf_1(int vc) {
    return per_idma_vc_has_space_interrupt_disable_cmdbuf<CMDBUF_1>(vc);
}
inline __attribute__((always_inline)) uint16_t per_idma_vc_has_space_get_interrupt_enable_cmdbuf_0() {
    return per_idma_vc_has_space_get_interrupt_enable_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint16_t per_idma_vc_has_space_get_interrupt_enable_cmdbuf_1() {
    return per_idma_vc_has_space_get_interrupt_enable_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_set_interrupt_enable_cmdbuf_0(uint16_t val) {
    return per_idma_vc_has_space_set_interrupt_enable_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_set_interrupt_enable_cmdbuf_1(uint16_t val) {
    return per_idma_vc_has_space_set_interrupt_enable_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) uint64_t per_idma_vc_has_space_interrupts_pending_cmdbuf_0() {
    return per_idma_vc_has_space_interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t per_idma_vc_has_space_interrupts_pending_cmdbuf_1() {
    return per_idma_vc_has_space_interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint16_t per_idma_vc_has_space_get_interrupt_pending_cmdbuf_0() {
    return per_idma_vc_has_space_get_interrupt_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint16_t per_idma_vc_has_space_get_interrupt_pending_cmdbuf_1() {
    return per_idma_vc_has_space_get_interrupt_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_set_interrupt_pending_cmdbuf_0(uint16_t val) {
    return per_idma_vc_has_space_set_interrupt_pending_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_set_interrupt_pending_cmdbuf_1(uint16_t val) {
    return per_idma_vc_has_space_set_interrupt_pending_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_clear_cmdbuf_0(int vc) {
    return per_idma_vc_has_space_interrupt_clear_cmdbuf<CMDBUF_0>(vc);
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_clear_cmdbuf_1(int vc) {
    return per_idma_vc_has_space_interrupt_clear_cmdbuf<CMDBUF_1>(vc);
}
inline __attribute__((always_inline)) void noc_fast_read_cmdbuf_0(
    uint64_t src_addr,
    uint32_t dest_addr,
    uint64_t len_bytes,
    uint64_t has_xy = 1,
    uint64_t posted = 0,
    uint64_t snoop = 0,
    uint64_t flush = 0) {
    return noc_fast_read_cmdbuf<CMDBUF_0>(src_addr, dest_addr, len_bytes, has_xy, posted, snoop, flush);
}
inline __attribute__((always_inline)) void noc_fast_read_cmdbuf_1(
    uint64_t src_addr,
    uint32_t dest_addr,
    uint64_t len_bytes,
    uint64_t has_xy = 1,
    uint64_t posted = 0,
    uint64_t snoop = 0,
    uint64_t flush = 0) {
    return noc_fast_read_cmdbuf<CMDBUF_1>(src_addr, dest_addr, len_bytes, has_xy, posted, snoop, flush);
}
inline __attribute__((always_inline)) void noc_read_cmdbuf_0(
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    bool snoop_bit = false,
    bool flush_bit = false) {
    return noc_read_cmdbuf<CMDBUF_0>(
        src_coordinate, src_addr, dest_coordinate, dest_addr, len_bytes, transaction_id, snoop_bit, flush_bit);
}
inline __attribute__((always_inline)) void noc_read_cmdbuf_1(
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    bool snoop_bit = false,
    bool flush_bit = false) {
    return noc_read_cmdbuf<CMDBUF_1>(
        src_coordinate, src_addr, dest_coordinate, dest_addr, len_bytes, transaction_id, snoop_bit, flush_bit);
}
inline __attribute__((always_inline)) void noc_write_prep_cmdbuf_0(
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    bool mcast = false,
    bool snoop_bit = false,
    bool flush_bit = false,
    bool posted = true) {
    return noc_write_prep_cmdbuf<CMDBUF_0>(
        src_coordinate,
        src_addr,
        dest_coordinate,
        dest_addr,
        len_bytes,
        transaction_id,
        mcast,
        snoop_bit,
        flush_bit,
        posted);
}
inline __attribute__((always_inline)) void noc_write_prep_cmdbuf_1(
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    bool mcast = false,
    bool snoop_bit = false,
    bool flush_bit = false,
    bool posted = true) {
    return noc_write_prep_cmdbuf<CMDBUF_1>(
        src_coordinate,
        src_addr,
        dest_coordinate,
        dest_addr,
        len_bytes,
        transaction_id,
        mcast,
        snoop_bit,
        flush_bit,
        posted);
}
inline __attribute__((always_inline)) void noc_read_prep_cmdbuf_0(
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    bool snoop_bit = false,
    bool flush_bit = false) {
    return noc_read_prep_cmdbuf<CMDBUF_0>(
        src_coordinate, src_addr, dest_coordinate, dest_addr, len_bytes, transaction_id, snoop_bit, flush_bit);
}
inline __attribute__((always_inline)) void noc_read_prep_cmdbuf_1(
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    bool snoop_bit = false,
    bool flush_bit = false) {
    return noc_read_prep_cmdbuf<CMDBUF_1>(
        src_coordinate, src_addr, dest_coordinate, dest_addr, len_bytes, transaction_id, snoop_bit, flush_bit);
}
inline __attribute__((always_inline)) void noc_fast_write_cmdbuf_0(
    uint32_t src_addr,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint64_t has_xy = 1,
    uint64_t posted = 0,
    uint64_t snoop = 0,
    uint64_t flush = 0) {
    return noc_fast_write_cmdbuf<CMDBUF_0>(src_addr, dest_addr, len_bytes, has_xy, posted, snoop, flush);
}
inline __attribute__((always_inline)) void noc_fast_write_cmdbuf_1(
    uint32_t src_addr,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint64_t has_xy = 1,
    uint64_t posted = 0,
    uint64_t snoop = 0,
    uint64_t flush = 0) {
    return noc_fast_write_cmdbuf<CMDBUF_1>(src_addr, dest_addr, len_bytes, has_xy, posted, snoop, flush);
}
inline __attribute__((always_inline)) void noc_write_cmdbuf_0(
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    bool mcast = false,
    bool snoop_bit = false,
    bool flush_bit = false,
    bool posted = true,
    TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u mcast_exclude = {0}) {
    return noc_write_cmdbuf<CMDBUF_0>(
        src_coordinate,
        src_addr,
        dest_coordinate,
        dest_addr,
        len_bytes,
        transaction_id,
        mcast,
        snoop_bit,
        flush_bit,
        posted,
        mcast_exclude);
}
inline __attribute__((always_inline)) void noc_write_cmdbuf_1(
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    bool mcast = false,
    bool snoop_bit = false,
    bool flush_bit = false,
    bool posted = true,
    TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u mcast_exclude = {0}) {
    return noc_write_cmdbuf<CMDBUF_1>(
        src_coordinate,
        src_addr,
        dest_coordinate,
        dest_addr,
        len_bytes,
        transaction_id,
        mcast,
        snoop_bit,
        flush_bit,
        posted,
        mcast_exclude);
}
inline __attribute__((always_inline)) void idma_copy_cmdbuf_0(
    uint64_t src_addr, uint64_t dest_addr, uint64_t len_bytes, uint32_t transaction_id = CMDBUF_DEF_TRID) {
    return idma_copy_cmdbuf<CMDBUF_0>(src_addr, dest_addr, len_bytes, transaction_id);
}
inline __attribute__((always_inline)) void idma_copy_cmdbuf_1(
    uint64_t src_addr, uint64_t dest_addr, uint64_t len_bytes, uint32_t transaction_id = CMDBUF_DEF_TRID) {
    return idma_copy_cmdbuf<CMDBUF_1>(src_addr, dest_addr, len_bytes, transaction_id);
}
inline __attribute__((always_inline)) void noc_atomic_increment_cmdbuf_0(
    uint64_t noc_coordinate,
    uint64_t addr,
    uint32_t incr = 1,
    uint32_t wrap = 31,
    bool snoop_bit = false,
    bool flush_bit = false) {
    return noc_atomic_increment_cmdbuf<CMDBUF_0>(noc_coordinate, addr, incr, wrap, snoop_bit, flush_bit);
}
inline __attribute__((always_inline)) void noc_atomic_increment_cmdbuf_1(
    uint64_t noc_coordinate,
    uint64_t addr,
    uint32_t incr = 1,
    uint32_t wrap = 31,
    bool snoop_bit = false,
    bool flush_bit = false) {
    return noc_atomic_increment_cmdbuf<CMDBUF_1>(noc_coordinate, addr, incr, wrap, snoop_bit, flush_bit);
}
inline __attribute__((always_inline)) uint64_t free_space_cmdbuf_0(uint32_t vc) {
    return free_space_cmdbuf<CMDBUF_0>(vc);
}
inline __attribute__((always_inline)) uint64_t free_space_cmdbuf_1(uint32_t vc) {
    return free_space_cmdbuf<CMDBUF_1>(vc);
}
inline __attribute__((always_inline)) uint64_t free_space_cmdbuf_0() { return free_space_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) uint64_t free_space_cmdbuf_1() { return free_space_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) uint64_t idma_free_space_cmdbuf_0(uint32_t vc) {
    return idma_free_space_cmdbuf<CMDBUF_0>(vc);
}
inline __attribute__((always_inline)) uint64_t idma_free_space_cmdbuf_1(uint32_t vc) {
    return idma_free_space_cmdbuf<CMDBUF_1>(vc);
}
inline __attribute__((always_inline)) uint64_t idma_free_space_cmdbuf_0() { return idma_free_space_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) uint64_t idma_free_space_cmdbuf_1() { return idma_free_space_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) bool noc_reads_acked_cmdbuf_0(uint32_t transaction_id) {
    return noc_reads_acked_cmdbuf<CMDBUF_0>(transaction_id);
}
inline __attribute__((always_inline)) bool noc_reads_acked_cmdbuf_1(uint32_t transaction_id) {
    return noc_reads_acked_cmdbuf<CMDBUF_1>(transaction_id);
}
inline __attribute__((always_inline)) bool noc_reads_acked_cmdbuf_0() { return noc_reads_acked_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) bool noc_reads_acked_cmdbuf_1() { return noc_reads_acked_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) bool all_noc_reads_acked_cmdbuf_0() {
    return all_noc_reads_acked_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) bool all_noc_reads_acked_cmdbuf_1() {
    return all_noc_reads_acked_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) bool idma_acked_cmdbuf_0(uint32_t transaction_id) {
    return idma_acked_cmdbuf<CMDBUF_0>(transaction_id);
}
inline __attribute__((always_inline)) bool idma_acked_cmdbuf_1(uint32_t transaction_id) {
    return idma_acked_cmdbuf<CMDBUF_1>(transaction_id);
}
inline __attribute__((always_inline)) bool idma_acked_cmdbuf_0() { return idma_acked_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) bool idma_acked_cmdbuf_1() { return idma_acked_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) bool all_idma_acked_cmdbuf_0() { return all_idma_acked_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) bool all_idma_acked_cmdbuf_1() { return all_idma_acked_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) bool noc_writes_sent_cmdbuf_0(uint32_t transaction_id) {
    return noc_writes_sent_cmdbuf<CMDBUF_0>(transaction_id);
}
inline __attribute__((always_inline)) bool noc_writes_sent_cmdbuf_1(uint32_t transaction_id) {
    return noc_writes_sent_cmdbuf<CMDBUF_1>(transaction_id);
}
inline __attribute__((always_inline)) bool noc_writes_sent_cmdbuf_0() { return noc_writes_sent_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) bool noc_writes_sent_cmdbuf_1() { return noc_writes_sent_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) bool all_noc_writes_sent_cmdbuf_0() {
    return all_noc_writes_sent_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) bool all_noc_writes_sent_cmdbuf_1() {
    return all_noc_writes_sent_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_cmdbuf_0(uint32_t transaction_id) {
    return noc_nonposted_writes_acked_cmdbuf<CMDBUF_0>(transaction_id);
}
inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_cmdbuf_1(uint32_t transaction_id) {
    return noc_nonposted_writes_acked_cmdbuf<CMDBUF_1>(transaction_id);
}
inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_cmdbuf_0() {
    return noc_nonposted_writes_acked_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_cmdbuf_1() {
    return noc_nonposted_writes_acked_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) bool all_noc_nonposted_writes_acked_cmdbuf_0() {
    return all_noc_nonposted_writes_acked_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) bool all_noc_nonposted_writes_acked_cmdbuf_1() {
    return all_noc_nonposted_writes_acked_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void l1_atomic_instr_cmdbuf_0(uint32_t fmt, bool no_sat, uint32_t atomic_op) {
    return l1_atomic_instr_cmdbuf<CMDBUF_0>(fmt, no_sat, atomic_op);
}
inline __attribute__((always_inline)) void l1_atomic_instr_cmdbuf_1(uint32_t fmt, bool no_sat, uint32_t atomic_op) {
    return l1_atomic_instr_cmdbuf<CMDBUF_1>(fmt, no_sat, atomic_op);
}
inline __attribute__((always_inline)) void idma_setup_as_atomic_accum_cmdbuf_0(bool wrapping_en = true) {
    return idma_setup_as_atomic_accum_cmdbuf<CMDBUF_0>(wrapping_en);
}
inline __attribute__((always_inline)) void idma_setup_as_atomic_accum_cmdbuf_1(bool wrapping_en = true) {
    return idma_setup_as_atomic_accum_cmdbuf<CMDBUF_1>(wrapping_en);
}
inline __attribute__((always_inline)) void idma_l1_atomic_accum_cmdbuf_0(
    uint64_t src_addr,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    uint64_t fmt = 0x0,
    uint64_t op = 0x9) {
    return idma_l1_atomic_accum_cmdbuf<CMDBUF_0>(src_addr, dest_addr, len_bytes, transaction_id, fmt, op);
}
inline __attribute__((always_inline)) void idma_l1_atomic_accum_cmdbuf_1(
    uint64_t src_addr,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    uint64_t fmt = 0x0,
    uint64_t op = 0x9) {
    return idma_l1_atomic_accum_cmdbuf<CMDBUF_1>(src_addr, dest_addr, len_bytes, transaction_id, fmt, op);
}

//////////////////////
/// Simple CMD Buf ///
// Below are all the functions for simple command buffer. This third command buffer,
//
//////////////////////

inline __attribute__((always_inline)) void reset_reg_cmdbuf() { __builtin_riscv_ttrocc_scmdbuf_reset(); }

inline __attribute__((always_inline)) void setup_as_copy_reg_cmdbuf(
    bool wr, bool mcast = false, TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u mcast_exclude = {0}, bool posted = true) {
    TT_ROCC_CMD_BUF_MISC_reg_u misc;
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

    misc.f.linked = mcast;
    misc.f.posted = wr && posted;
    misc.f.multicast = mcast;
    misc.f.write_trans = wr;

    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc.val);

    if (mcast) {
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_EXCLUDE_REG_OFFSET / 8, mcast_exclude.val);
    }
}

inline __attribute__((always_inline)) void setup_as_atomic_reg_cmdbuf(bool wr) {
    TT_ROCC_CMD_BUF_MISC_reg_u misc;
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

    misc.f.posted = 1;
    misc.f.write_trans = 0;
    misc.f.atomic_trans = 1;

    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc.val);
}

inline __attribute__((always_inline)) void setup_vcs_reg_cmdbuf(bool wr, bool mcast = false) {
    if (wr) {
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, mcast ? CMDBUF_MCAST_REQ_VC : CMDBUF_WR_REQ_VC);
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8,
            mcast ? CMDBUF_MCAST_RESP_VC : CMDBUF_WR_RESP_VC);
    } else {
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, CMDBUF_RD_REQ_VC);
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, CMDBUF_RD_RESP_VC);
    }
}

inline __attribute__((always_inline)) void setup_trids_reg_cmdbuf(uint32_t trid_offset = CMDBUF_DEF_TRID) {
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8, trid_offset);
}

inline __attribute__((always_inline)) uint32_t swap_trid_reg_cmdbuf(uint32_t new_trid) {
    uint32_t prev_trid =
        __builtin_riscv_ttrocc_scmdbuf_rd_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8, new_trid);
    return prev_trid;
}

inline __attribute__((always_inline)) void setup_packet_tags_reg_cmdbuf(bool snoop_bit, bool flush_bit) {
    TT_ROCC_CMD_BUF_PACKET_TAGS_reg_u misc;

    misc.f.snoop_bit = snoop_bit;
    misc.f.flush_bit = flush_bit;

    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PACKET_TAGS_REG_OFFSET / 8, misc.val);
}

inline __attribute__((always_inline)) uint64_t get_src_reg_cmdbuf() {
    return __builtin_riscv_ttrocc_scmdbuf_rd_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8);
}

inline __attribute__((always_inline)) void set_src_reg_cmdbuf(uint64_t addr, uint64_t coordinate) {
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET / 8, coordinate);
}
inline __attribute__((always_inline)) void set_src_reg_cmdbuf(uint64_t addr) {
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, addr);
}

inline __attribute__((always_inline)) uint64_t get_dest_reg_cmdbuf() {
    return __builtin_riscv_ttrocc_scmdbuf_rd_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8);
}

inline __attribute__((always_inline)) void set_dest_reg_cmdbuf(uint64_t addr, uint64_t coordinates) {
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, coordinates);
}
inline __attribute__((always_inline)) void set_dest_reg_cmdbuf(uint64_t addr) {
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, addr);
}

inline __attribute__((always_inline)) void set_len_reg_cmdbuf(uint64_t size_bytes) {
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, size_bytes);
}

inline __attribute__((always_inline)) void issue_reg_cmdbuf() { __builtin_riscv_ttrocc_scmdbuf_issue_trans(); }

inline __attribute__((always_inline)) void issue_read_reg_cmdbuf() { issue_reg_cmdbuf(); }

inline __attribute__((always_inline)) void issue_write_reg_cmdbuf() { issue_reg_cmdbuf(); }

inline __attribute__((always_inline)) void issue_write_inline_reg_cmdbuf(uint64_t data) {
    __builtin_riscv_ttrocc_scmdbuf_issue_inline_trans(data);
}

inline __attribute__((always_inline)) void issue_write_inline_len_reg_cmdbuf(
    uint64_t data,
    uint64_t dest_addr,
    uint64_t size_bytes,
    uint64_t has_xy = 1,
    uint64_t posted = 0,
    uint64_t snoop = 0,
    uint64_t flush = 0) {
    uint64_t rs2 =
        dest_addr | ((size_bytes - 1) << 57) | (has_xy << 60) | (posted << 61) | (snoop << 62) | (flush << 63);
    __builtin_riscv_ttrocc_scmdbuf_issue_inline_addr_trans(data, rs2);
}

inline __attribute__((always_inline)) void interrupt_enable_reg_cmdbuf(int id) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = __builtin_riscv_ttrocc_scmdbuf_rd_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    val.val |= (1 << id);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, val.val);
}

inline __attribute__((always_inline)) void interrupt_disable_reg_cmdbuf(int id) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = __builtin_riscv_ttrocc_scmdbuf_rd_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    val.val &= ~(1 << id);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, val.val);
}

inline __attribute__((always_inline)) uint64_t interrupts_pending_reg_cmdbuf() {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = __builtin_riscv_ttrocc_scmdbuf_rd_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8);
    return val.val;
}

inline __attribute__((always_inline)) void interrupt_clear_reg_cmdbuf(int id) {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = ~(1 << id);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8, val.val);
}

inline __attribute__((always_inline)) void noc_fast_read_reg_cmdbuf(
    uint64_t src_addr,
    uint32_t dest_addr,
    uint64_t len_bytes,
    uint64_t has_xy = 1,
    uint64_t posted = 0,
    uint64_t snoop = 0,
    uint64_t flush = 0) {
    uint64_t rs1 = (len_bytes << 32) | dest_addr;
    uint64_t rs2 = src_addr | (has_xy << 60) | (posted << 61) | (snoop << 62) | (flush << 63);
    __builtin_riscv_ttrocc_scmdbuf_issue_read2_trans(rs1, rs2);
}

inline __attribute__((always_inline)) void noc_read_reg_cmdbuf(
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    bool snoop_bit = false,
    bool flush_bit = false) {
    reset_reg_cmdbuf();
    setup_as_copy_reg_cmdbuf(false, false, {0});
    setup_vcs_reg_cmdbuf(false);
    setup_trids_reg_cmdbuf(transaction_id);
    if (snoop_bit || flush_bit) {
        setup_packet_tags_reg_cmdbuf(snoop_bit, flush_bit);
    }
    set_src_reg_cmdbuf(src_addr, src_coordinate);
    set_dest_reg_cmdbuf(dest_addr, dest_coordinate);
    set_len_reg_cmdbuf(len_bytes);
    issue_read_reg_cmdbuf();
}

inline __attribute__((always_inline)) void noc_fast_write_reg_cmdbuf(
    uint32_t src_addr,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint64_t has_xy = 1,
    uint64_t posted = 0,
    uint64_t snoop = 0,
    uint64_t flush = 0) {
    uint64_t rs1 = (len_bytes << 32) | src_addr;
    uint64_t rs2 = dest_addr | (has_xy << 60) | (posted << 61) | (snoop << 62) | (flush << 63);
    __builtin_riscv_ttrocc_scmdbuf_issue_write2_trans(rs1, rs2);
}

inline __attribute__((always_inline)) void noc_write_reg_cmdbuf(
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id = CMDBUF_DEF_TRID,
    bool mcast = false,
    bool snoop_bit = false,
    bool flush_bit = false,
    bool posted = true) {
    reset_reg_cmdbuf();
    setup_as_copy_reg_cmdbuf(true, mcast, {0}, posted);
    setup_vcs_reg_cmdbuf(true, mcast);
    setup_trids_reg_cmdbuf(transaction_id);
    if (snoop_bit || flush_bit) {
        setup_packet_tags_reg_cmdbuf(snoop_bit, flush_bit);
    }
    set_src_reg_cmdbuf(src_addr, src_coordinate);
    set_dest_reg_cmdbuf(dest_addr, dest_coordinate);
    set_len_reg_cmdbuf(len_bytes);
    issue_write_reg_cmdbuf();
}

inline __attribute__((always_inline)) void noc_atomic_increment_reg_cmdbuf(
    uint64_t noc_coordinate,
    uint64_t addr,
    uint32_t incr = 1,
    uint32_t wrap = 31,
    bool snoop_bit = false,
    bool flush_bit = false) {
    uint64_t at_len =
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0);
    setup_as_atomic_reg_cmdbuf(true);
    if (snoop_bit || flush_bit) {
        setup_packet_tags_reg_cmdbuf(snoop_bit, flush_bit);
    }
    set_dest_reg_cmdbuf(addr, noc_coordinate);
    set_len_reg_cmdbuf(at_len);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_INLINE_DATA_REG_OFFSET / 8, (uint64_t)incr);
    issue_reg_cmdbuf();
}

inline __attribute__((always_inline)) uint64_t free_space_reg_cmdbuf(uint32_t vc) {
    return __builtin_riscv_ttrocc_scmdbuf_get_vc_space_vc(vc);
}
inline __attribute__((always_inline)) uint64_t free_space_reg_cmdbuf() {
    return __builtin_riscv_ttrocc_scmdbuf_get_vc_space();
}

inline __attribute__((always_inline)) bool noc_reads_acked_reg_cmdbuf(uint32_t transaction_id) {
    return __builtin_riscv_ttrocc_scmdbuf_tr_ack_trid(transaction_id) == 0;
}
inline __attribute__((always_inline)) bool noc_reads_acked_reg_cmdbuf() {
    return __builtin_riscv_ttrocc_scmdbuf_tr_ack() == 0;
}

inline __attribute__((always_inline)) bool noc_writes_sent_reg_cmdbuf(uint32_t transaction_id) {
    return __builtin_riscv_ttrocc_scmdbuf_wr_sent_trid(transaction_id) == 0;
}
inline __attribute__((always_inline)) bool noc_writes_sent_reg_cmdbuf() {
    return __builtin_riscv_ttrocc_scmdbuf_wr_sent() == 0;
}

inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_reg_cmdbuf(uint32_t transaction_id) {
    return __builtin_riscv_ttrocc_scmdbuf_tr_ack_trid(transaction_id) == 0;
}
inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_reg_cmdbuf() {
    return __builtin_riscv_ttrocc_scmdbuf_tr_ack() == 0;
}

}  // namespace overlay
