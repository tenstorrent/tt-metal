// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
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
 * for each buffer. Since opcodes must be known at compile time, this API uses C++
 * macros to generate buffer-specific function variants:
 * - `xxx_cmdbuf_0()` - Functions for command buffer 0
 * - `xxx_cmdbuf_1()` - Functions for command buffer 1
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

#include "rocc_instructions.hpp"

#define CMDBUF_0 0
#define CMDBUF_1 1

/* Default transaction ID for both command buffers */
constexpr uint32_t CMDBUF_DEF_TRID = 0;
/* Static(starting) transcation ID for command buffer 0 */
constexpr uint32_t CMDBUF_0_TRID_STATIC = 1;
/* Start transcation ID for command buffer 0 when using TID wrapping */
constexpr uint32_t CMDBUF_0_TRID_START = 2;
/* End transcation ID for command buffer 0 when using TID wrapping */
constexpr uint32_t CMDBUF_0_TRID_END = 6;
/* Static(starting) transcation ID for command buffer 0 */
constexpr uint32_t CMDBUF_1_TRID_STATIC = 7;
/* Start transcation ID for command buffer 0 when using TID wrapping */
constexpr uint32_t CMDBUF_1_TRID_START = 8;
/* End transcation ID for command buffer 0 when using TID wrapping */
constexpr uint32_t CMDBUF_1_TRID_END = 12;

/* Read request virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_RD_REQ_VC = 1;
/* Read response virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_RD_RESP_VC = 12;
/* Write request virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_WR_REQ_VC = 2;
/* Write response virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_WR_RESP_VC = 13;
/* Multicast request virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_MCAST_REQ_VC = 8;
/* Multicast response virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_MCAST_RESP_VC = 14;

#define DEFINE_CMD_BUFS(buf_name, cmdbuf)                                                                              \
                                                                                                                       \
    /*                                                                                                                 \
     * @def reset_cmdbuf_0()                                                                                           \
     * @def reset_cmdbuf_1()                                                                                           \
     *                                                                                                                 \
     * @brief Defines an inline reset functions for reseting command buffers state                                     \
     * Should be called before any other command buffer setup functions.                                               \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void reset_##buf_name() { CMDBUF_RESET(cmdbuf); }                            \
                                                                                                                       \
    /*                                                                                                                 \
     * @def setup_as_copy_cmdbuf_0                                                                                     \
     * @def setup_as_copy_cmdbuf_1                                                                                     \
     *                                                                                                                 \
     * @brief Configures command buffer for copy operations with customizable settings                                 \
     *                                                                                                                 \
     * @param wr Indicates if the operation is a write (true) or read (false)                                          \
     * @param mcast Enables multicast if true; default is false                                                        \
     * @param mcast_exclude A `TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u` structure specifying which cores to exclude        \
     *                      from multicast. Defaults to no exclusions                                                  \
     * @param wrapping_en Enables address wrapping functionality; default is true                                      \
     * @param posted Enables posted transactions for better performance; default is true                               \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void setup_as_copy_##buf_name(                                               \
        bool wr,                                                                                                       \
        bool mcast = false,                                                                                            \
        TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u mcast_exclude = {0},                                                       \
        bool wrapping_en = true,                                                                                       \
        bool posted = true) {                                                                                          \
        TT_ROCC_CMD_BUF_MISC_reg_u misc;                                                                               \
        misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;                                                                   \
                                                                                                                       \
        misc.f.linked = mcast;                                                                                         \
        misc.f.posted = wr && posted;                                                                                  \
        misc.f.multicast = mcast;                                                                                      \
        misc.f.write_trans = wr;                                                                                       \
        misc.f.wrapping_en = wrapping_en;                                                                              \
                                                                                                                       \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET, misc.val);                         \
                                                                                                                       \
        if (mcast)                                                                                                     \
            CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_EXCLUDE_REG_OFFSET, mcast_exclude.val);   \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def setup_as_scatter_list_cmdbuf_0                                                                             \
     * @def setup_as_scatter_list_cmdbuf_1                                                                             \
     *                                                                                                                 \
     * @brief Configures the specified command buffer as a scatter list with customizable settings.                    \
     *                                                                                                                 \
     * This function enables scatter list functionality for a command buffers                                          \
     *                                                                                                                 \
     * @param wr Indicates if the operation is a write (true) or read (false).                                         \
     * @param apply_scatter_to_dest Indicates if scatter list should be applied to the destination address.            \
     * @param mcast Enables multicast if true; default is false.                                                       \
     * @param linked Enables linked transcation                                                                        \
     * @param mcast_exclude A `TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u` structure specifying which cores to exclude        \
     *                      from multicast. Defaults to no exclusions.                                                 \
     * @param scatter_list_contains_size Specifies if the scatter list includes size information.                      \
     * @param scatter_list_contains_xy Specifies if the scatter list includes XY coordinates. When true,               \
     *                                 each element in the scatter list includes XY positioning.                       \
     *                                                                                                                 \
     * @note To be used with set_scatter_list_x(_)                                                                     \
     * @note Same thing can be achieved with setup_x() function                                                        \
     *                                                                                                                 \
     */                                                                                                                \
    inline __attribute__((always_inline)) void setup_as_scatter_list_##buf_name(                                       \
        bool wr,                                                                                                       \
        bool apply_scatter_to_dest,                                                                                    \
        bool mcast = false,                                                                                            \
        bool linked = false,                                                                                           \
        TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u mcast_exclude = {0},                                                       \
        bool scatter_list_contains_size = false,                                                                       \
        bool scatter_list_contains_xy = false,                                                                         \
        bool wrapping_en = true,                                                                                       \
        bool posted = true) {                                                                                          \
        TT_ROCC_CMD_BUF_MISC_reg_u misc;                                                                               \
        misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;                                                                   \
                                                                                                                       \
        misc.f.linked = linked;                                                                                        \
        misc.f.posted = wr && posted;                                                                                  \
        misc.f.multicast = mcast;                                                                                      \
        misc.f.scatter_list_en = true;                                                                                 \
        misc.f.scatter_list_to_dest_addr = apply_scatter_to_dest;                                                      \
        misc.f.write_trans = wr;                                                                                       \
        misc.f.scatter_list_has_size = scatter_list_contains_size;                                                     \
        misc.f.scatter_list_has_xy = scatter_list_contains_xy;                                                         \
        misc.f.wrapping_en = wrapping_en;                                                                              \
                                                                                                                       \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET, misc.val);                         \
                                                                                                                       \
        if (mcast)                                                                                                     \
            CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_EXCLUDE_REG_OFFSET, mcast_exclude.val);   \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def setup_as_atomic_cmdbuf_0                                                                                   \
     * @def setup_as_atomic_cmdbuf_1                                                                                   \
     *                                                                                                                 \
     * @brief Function for configuring command buffer for atomic transactions                                          \
     *                                                                                                                 \
     * @param wr Indicates if the operation is a write (true) or read (false).                                         \
     *                                                                                                                 \
     * @note Overwrites existing command buffer settings                                                               \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void setup_as_atomic_##buf_name(bool wr) {                                   \
        TT_ROCC_CMD_BUF_MISC_reg_u misc;                                                                               \
        misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;                                                                   \
                                                                                                                       \
        misc.f.posted = 1;                                                                                             \
        misc.f.write_trans = 0;                                                                                        \
        misc.f.atomic_trans = 1;                                                                                       \
                                                                                                                       \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET, misc.val);                         \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def idma_setup_as_copy_cmdbuf_0                                                                                \
     * @def idma_setup_as_copy_cmdbuf_1                                                                                \
     *                                                                                                                 \
     * @brief Function for configuring command buffer for iDMA copy operations                                         \
     *                                                                                                                 \
     * @param wrapping_en Enables address wrapping; default is true                                                    \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void idma_setup_as_copy_##buf_name(bool wrapping_en = true) {                \
        TT_ROCC_CMD_BUF_MISC_reg_u misc;                                                                               \
        misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;                                                                   \
                                                                                                                       \
        misc.f.write_trans = 1;                                                                                        \
        misc.f.idma_en = 1;                                                                                            \
        misc.f.wrapping_en = wrapping_en;                                                                              \
                                                                                                                       \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET, misc.val);                         \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def idma_setup_as_scatter_list_cmdbuf_0                                                                        \
     * @def idma_setup_as_scatter_list_cmdbuf_1                                                                        \
     *                                                                                                                 \
     * @brief Function for configuring command buffer for iDMA scatter list operations                                 \
     *                                                                                                                 \
     * @param apply_scatter_to_dest Indicates if scatter list should be applied to the destination address             \
     * @param scatter_list_contains_size Specifies if the scatter list includes size information                       \
     * @param scatter_list_contains_xy Specifies if the scatter list includes XY coordinates                           \
     * @param wrapping_en Enables address wrapping; default is true                                                    \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void idma_setup_as_scatter_list_##buf_name(                                  \
        bool apply_scatter_to_dest,                                                                                    \
        bool scatter_list_contains_size = false,                                                                       \
        bool scatter_list_contains_xy = false,                                                                         \
        bool wrapping_en = true) {                                                                                     \
        TT_ROCC_CMD_BUF_MISC_reg_u misc;                                                                               \
        misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;                                                                   \
                                                                                                                       \
        misc.f.scatter_list_en = true;                                                                                 \
        misc.f.scatter_list_to_dest_addr = apply_scatter_to_dest;                                                      \
        misc.f.write_trans = 1;                                                                                        \
        misc.f.scatter_list_has_size = scatter_list_contains_size;                                                     \
        misc.f.scatter_list_has_xy = scatter_list_contains_xy;                                                         \
        misc.f.idma_en = 1;                                                                                            \
        misc.f.wrapping_en = wrapping_en;                                                                              \
                                                                                                                       \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET, misc.val);                         \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def setup_ongoing_cmdbuf_0                                                                                     \
     * @def setup_ongoing_cmdbuf_1                                                                                     \
     *                                                                                                                 \
     * @brief Function for configuring incrementing logic for command buffer                                           \
     * Addresses, vcs and transcation ids can be configure to be self incrementing after each transactions             \
     *                                                                                                                 \
     * @param src_addr_inc_en If enabled source address will increment after issue                                     \
     * @param dest_addr_inc_en If enabled destination address will increment after issue                               \
     * @param trid_inc_en If enabled transcation ID will increment after issue                                         \
     * @param req_vc_inc_en If enabled request VC will increment after issue                                           \
     * @param resp_vc_inc_en If enabled response VC will increment after issue                                         \
     * @param req_vc_inc_on_entire_trans Request VC increment on entire transaction                                    \
     * @param resp_vc_inc_on_entire_trans Response VC increment on entire transaction                                  \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void setup_ongoing_##buf_name(                                               \
        bool src_addr_inc_en,                                                                                          \
        bool dest_addr_inc_en,                                                                                         \
        bool trid_inc_en,                                                                                              \
        bool req_vc_inc_en,                                                                                            \
        bool resp_vc_inc_en,                                                                                           \
        bool req_vc_inc_on_entire_trans = false,                                                                       \
        bool resp_vc_inc_on_entire_trans = false) {                                                                    \
        TT_ROCC_CMD_BUF_AUTOINC_reg_u ongoing;                                                                         \
                                                                                                                       \
        ongoing.f.src_addr_inc_en = src_addr_inc_en;                                                                   \
        ongoing.f.dest_addr_inc_en = dest_addr_inc_en;                                                                 \
        ongoing.f.trid_inc_en = trid_inc_en;                                                                           \
        ongoing.f.req_vc_inc_en = req_vc_inc_en;                                                                       \
        ongoing.f.resp_vc_inc_en = resp_vc_inc_en;                                                                     \
        ongoing.f.req_vc_inc_on_entire_trans = req_vc_inc_on_entire_trans;                                             \
        ongoing.f.resp_vc_inc_on_entire_trans = resp_vc_inc_on_entire_trans;                                           \
                                                                                                                       \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_AUTOINC_REG_OFFSET, ongoing.val);                   \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def setup_vcs_cmdbuf_0                                                                                         \
     * @def setup_vcs_cmdbuf_1                                                                                         \
     *                                                                                                                 \
     * @brief Function for configuring virtual channels based on global values defined in this file                    \
     *                                                                                                                 \
     * @param wr If enabled, will use write VCs                                                                        \
     * @param mcast If enabled, will use multicast VCs                                                                 \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void setup_vcs_##buf_name(bool wr, bool mcast = false) {                     \
        if (wr) {                                                                                                      \
            CMDBUF_WR_REG(                                                                                             \
                cmdbuf,                                                                                                \
                TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET,                                                \
                mcast ? CMDBUF_MCAST_REQ_VC : CMDBUF_WR_REQ_VC);                                                       \
            CMDBUF_WR_REG(                                                                                             \
                cmdbuf,                                                                                                \
                TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET,                                               \
                mcast ? CMDBUF_MCAST_RESP_VC : CMDBUF_WR_RESP_VC);                                                     \
        } else {                                                                                                       \
            CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET, CMDBUF_RD_REQ_VC);           \
            CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET, CMDBUF_RD_RESP_VC);         \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def setup_wrapping_req_vcs_cmdbuf_0                                                                            \
     * @def setup_wrapping_req_vcs_cmdbuf_1                                                                            \
     *                                                                                                                 \
     * @brief Function for configuring wrapping feature of request and response virtual channels                       \
     *                                                                                                                 \
     * @param wr Write/read mode selector                                                                              \
     * @param req_start_vc Starting virtual channel for requests                                                       \
     * @param req_end_vc End virtual channel for requests                                                              \
     * @param req_vc_offset Offset while wrapping for requests                                                         \
     * @param resp_start_vc Starting virtual channel for responses                                                     \
     * @param resp_end_vc End virtual channel for responses                                                            \
     * @param resp_vc_offset Offset while wrapping for responses                                                       \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void setup_wrapping_vcs_##buf_name(                                          \
        bool wr,                                                                                                       \
        uint32_t req_start_vc,                                                                                         \
        uint32_t req_end_vc,                                                                                           \
        uint32_t req_vc_offset = 0,                                                                                    \
        uint32_t resp_start_vc = 0,                                                                                    \
        uint32_t resp_end_vc = 0,                                                                                      \
        uint32_t resp_vc_offset = 0) {                                                                                 \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET, req_start_vc + req_vc_offset);   \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_BASE_REG_OFFSET, req_start_vc);              \
        CMDBUF_WR_REG(                                                                                                 \
            cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_SIZE_REG_OFFSET, req_end_vc - req_start_vc + 1);       \
        CMDBUF_WR_REG(                                                                                                 \
            cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET, resp_start_vc + resp_vc_offset);          \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_BASE_REG_OFFSET, resp_start_vc);            \
        CMDBUF_WR_REG(                                                                                                 \
            cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_SIZE_REG_OFFSET, resp_end_vc - resp_start_vc + 1);    \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def setup_trids_static_cmdbuf_0                                                                                \
     * @def setup_trids_static_cmdbuf_1                                                                                \
     *                                                                                                                 \
     * @brief Function for configuring transcation ID based on macros defined in this file                             \
     * If wrapping feature for transaction ID is enabled, specifed ID is used as offset                                \
     *                                                                                                                 \
     * @param trid_offset Transaction ID, if wrapping is enabled this serves as transcation ID offset                  \
     * @param wrapping Enables wrapping feature for transcation IDs                                                    \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void setup_trids_##buf_name(                                                 \
        uint32_t trid_offset = CMDBUF_DEF_TRID, bool wrapping = false) {                                               \
        if (cmdbuf == CMDBUF_0) {                                                                                      \
            CMDBUF_WR_REG(                                                                                             \
                cmdbuf,                                                                                                \
                TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_WR_SENT_TR_ID_REG_OFFSET,                                         \
                wrapping ? CMDBUF_0_TRID_START + trid_offset : trid_offset);                                           \
            CMDBUF_WR_REG(                                                                                             \
                cmdbuf,                                                                                                \
                TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ACK_TR_ID_REG_OFFSET,                                          \
                wrapping ? CMDBUF_0_TRID_START + trid_offset : trid_offset);                                           \
            CMDBUF_WR_REG(                                                                                             \
                cmdbuf,                                                                                                \
                TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET,                                                 \
                wrapping ? CMDBUF_0_TRID_START + trid_offset : trid_offset);                                           \
            if (wrapping) {                                                                                            \
                CMDBUF_WR_REG(                                                                                         \
                    cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_BASE_REG_OFFSET, CMDBUF_0_TRID_START);          \
                CMDBUF_WR_REG(                                                                                         \
                    cmdbuf,                                                                                            \
                    TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_SIZE_REG_OFFSET,                                        \
                    CMDBUF_0_TRID_END - CMDBUF_0_TRID_START + 1);                                                      \
            }                                                                                                          \
        } else {                                                                                                       \
            CMDBUF_WR_REG(                                                                                             \
                cmdbuf,                                                                                                \
                TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_WR_SENT_TR_ID_REG_OFFSET,                                         \
                wrapping ? CMDBUF_1_TRID_START + trid_offset : trid_offset);                                           \
            CMDBUF_WR_REG(                                                                                             \
                cmdbuf,                                                                                                \
                TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ACK_TR_ID_REG_OFFSET,                                          \
                wrapping ? CMDBUF_1_TRID_START + trid_offset : trid_offset);                                           \
            CMDBUF_WR_REG(                                                                                             \
                cmdbuf,                                                                                                \
                TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET,                                                 \
                wrapping ? CMDBUF_1_TRID_START + trid_offset : trid_offset);                                           \
            if (wrapping) {                                                                                            \
                CMDBUF_WR_REG(                                                                                         \
                    cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_BASE_REG_OFFSET, CMDBUF_1_TRID_START);          \
                CMDBUF_WR_REG(                                                                                         \
                    cmdbuf,                                                                                            \
                    TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_SIZE_REG_OFFSET,                                        \
                    CMDBUF_1_TRID_END - CMDBUF_1_TRID_START + 1);                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def swap_trid_cmdbuf_0                                                                                         \
     * @def swap_trid_cmdbuf_1                                                                                         \
     *                                                                                                                 \
     * @brief Swaps current transaction ID with new one and returns previous value                                     \
     *                                                                                                                 \
     * @param new_trid New transaction ID to set                                                                       \
     * @return Previous transaction ID value                                                                           \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) uint32_t swap_trid_##buf_name(uint32_t new_trid) {                           \
        uint32_t prev_trid = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET);             \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET, new_trid);                        \
        return prev_trid;                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def setup_max_bytes_in_packet_cmdbuf_0                                                                         \
     * @def setup_max_bytes_in_packet_cmdbuf_1                                                                         \
     *                                                                                                                 \
     * @brief Sets maximum bytes per packet for command buffer                                                         \
     *                                                                                                                 \
     * @param max_bytes_in_packet Maximum bytes allowed in single packet                                               \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void setup_max_bytes_in_packet_##buf_name(uint64_t max_bytes_in_packet) {    \
        CMDBUF_WR_REG(                                                                                                 \
            cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MAX_BYTES_IN_PACKET_REG_OFFSET, max_bytes_in_packet);         \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def setup_packet_tags_cmdbuf_0                                                                                 \
     * @def setup_packet_tags_cmdbuf_1                                                                                 \
     *                                                                                                                 \
     * @brief Function for configuring snoop and flush bit of transaction                                              \
     *                                                                                                                 \
     * @param snoop_bit Enables destination NIU for cache snoop mechanims                                              \
     * @param flush_bit Enables destination NIU to commit all parts of the flit before committing the next packet      \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void setup_packet_tags_##buf_name(bool snoop_bit, bool flush_bit) {          \
        TT_ROCC_CMD_BUF_PACKET_TAGS_reg_u misc;                                                                        \
                                                                                                                       \
        misc.f.snoop_bit = snoop_bit;                                                                                  \
        misc.f.flush_bit = flush_bit;                                                                                  \
                                                                                                                       \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PACKET_TAGS_REG_OFFSET, misc.val);                  \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def get_src_cmdbuf_0                                                                                           \
     * @def get_src_cmdbuf_1                                                                                           \
     *                                                                                                                 \
     * @brief Returns source address for transactions                                                                  \
     *                                                                                                                 \
     * @return Source address with noc coordinates embedded                                                            \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) uint64_t get_src_##buf_name() {                                              \
        return CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET);                        \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def set_src_cmdbuf_0                                                                                           \
     * @def set_src_cmdbuf_1                                                                                           \
     *                                                                                                                 \
     * @brief Sets up source configuration                                                                             \
     *                                                                                                                 \
     * @param addr Source address without coordinates (0-4mbs)                                                         \
     * @param coordinate Coordinate generated using NOC_XY_COORD macro                                                 \
     * @param base Base address, if wrapping is enabled                                                                \
     * @param size Size of transfer in bytes                                                                           \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void set_src_##buf_name(                                                     \
        uint64_t addr, uint64_t coordinate, uint64_t base, uint64_t size) {                                            \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET, addr);                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_BASE_REG_OFFSET, base);                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_SIZE_REG_OFFSET, size);                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET, coordinate);                  \
    }                                                                                                                  \
    /*                                                                                                                 \
     * @def set_src_cmdbuf_0 (3-parameter overload)                                                                    \
     * @def set_src_cmdbuf_1 (3-parameter overload)                                                                    \
     *                                                                                                                 \
     * @brief Sets up source configuration with base address                                                           \
     *                                                                                                                 \
     * @param addr Source address without coordinates                                                                  \
     * @param coordinate Coordinate generated using NOC_XY_COORD macro                                                 \
     * @param base Base address for wrapping                                                                           \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void set_src_##buf_name(uint64_t addr, uint64_t coordinate, uint64_t base) { \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET, addr);                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_BASE_REG_OFFSET, base);                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET, coordinate);                  \
    }                                                                                                                  \
    /*                                                                                                                 \
     * @def set_src_cmdbuf_0 (2-parameter overload)                                                                    \
     * @def set_src_cmdbuf_1 (2-parameter overload)                                                                    \
     *                                                                                                                 \
     * @brief Sets up source configuration with address and coordinate                                                 \
     *                                                                                                                 \
     * @param addr Source address without coordinates                                                                  \
     * @param coordinate Coordinate generated using NOC_XY_COORD macro                                                 \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void set_src_##buf_name(uint64_t addr, uint64_t coordinate) {                \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET, addr);                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET, coordinate);                  \
    }                                                                                                                  \
    /*                                                                                                                 \
     * @def set_src_cmdbuf_0 (1-parameter overload)                                                                    \
     * @def set_src_cmdbuf_1 (1-parameter overload)                                                                    \
     *                                                                                                                 \
     * @brief Sets up source address only                                                                              \
     *                                                                                                                 \
     * @param addr Source address                                                                                      \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void set_src_##buf_name(uint64_t addr) {                                     \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET, addr);                         \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def get_dest_cmdbuf_0                                                                                          \
     * @def get_dest_cmdbuf_1                                                                                          \
     *                                                                                                                 \
     * @brief Returns destination address for transactions                                                             \
     *                                                                                                                 \
     * @return Destination address with noc coordinates embedded                                                       \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) uint64_t get_dest_##buf_name() {                                             \
        return CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET);                       \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def set_dest_cmdbuf_0                                                                                          \
     * @def set_dest_cmdbuf_1                                                                                          \
     *                                                                                                                 \
     * @brief Sets up destination configuration                                                                        \
     *                                                                                                                 \
     * @param addr Destination address without coordinates (0-4mbs)                                                    \
     * @param coordinates Coordinate generated using NOC_XY_COORD macro                                                \
     * @param base Base address, if wrapping is enabled                                                                \
     * @param size Size of transfer in bytes                                                                           \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void set_dest_##buf_name(                                                    \
        uint64_t addr, uint64_t coordinates, uint64_t base, uint64_t size) {                                           \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET, addr);                        \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_BASE_REG_OFFSET, base);                        \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_SIZE_REG_OFFSET, size);                        \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET, coordinates);                \
    }                                                                                                                  \
    /*                                                                                                                 \
     * @def set_dest_cmdbuf_0 (3-parameter overload)                                                                   \
     * @def set_dest_cmdbuf_1 (3-parameter overload)                                                                   \
     *                                                                                                                 \
     * @brief Sets up destination configuration with base address                                                      \
     *                                                                                                                 \
     * @param addr Destination address without coordinates                                                             \
     * @param coordinates Coordinate generated using NOC_XY_COORD macro                                                \
     * @param base Base address for wrapping                                                                           \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline                                                                                                             \
        __attribute__((always_inline)) void set_dest_##buf_name(uint64_t addr, uint64_t coordinates, uint64_t base) {  \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET, addr);                        \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_BASE_REG_OFFSET, base);                        \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET, coordinates);                \
    }                                                                                                                  \
    /*                                                                                                                 \
     * @def set_dest_cmdbuf_0 (2-parameter overload)                                                                   \
     * @def set_dest_cmdbuf_1 (2-parameter overload)                                                                   \
     *                                                                                                                 \
     * @brief Sets up destination configuration with address and coordinate                                            \
     *                                                                                                                 \
     * @param addr Destination address without coordinates                                                             \
     * @param coordinates Coordinate generated using NOC_XY_COORD macro                                                \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void set_dest_##buf_name(uint64_t addr, uint64_t coordinates) {              \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET, addr);                        \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET, coordinates);                \
    }                                                                                                                  \
    /*                                                                                                                 \
     * @def set_dest_cmdbuf_0 (1-parameter overload)                                                                   \
     * @def set_dest_cmdbuf_1 (1-parameter overload)                                                                   \
     *                                                                                                                 \
     * @brief Sets up destination address only                                                                         \
     *                                                                                                                 \
     * @param addr Destination address                                                                                 \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void set_dest_##buf_name(uint64_t addr) {                                    \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET, addr);                        \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def set_scatter_list_cmdbuf_0                                                                                  \
     * @def set_scatter_list_cmdbuf_1                                                                                  \
     *                                                                                                                 \
     * @brief Configure scatter list parameters                                                                        \
     *                                                                                                                 \
     * @param addr Scatter list address                                                                                \
     * @param base Base address                                                                                        \
     * @param index Index value                                                                                        \
     * @param times Number of times                                                                                    \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void set_scatter_list_##buf_name(                                            \
        uint64_t addr, uint64_t base, uint64_t index, uint64_t times) {                                                \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_LIST_ADDR_REG_OFFSET, addr);                \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_BASE_ADDR_REG_OFFSET, base);                \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_INDEX_REG_OFFSET, index);                   \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_TIMES_REG_OFFSET, times);                   \
    }                                                                                                                  \
    inline __attribute__((always_inline)) void set_scatter_list_base_##buf_name(uint64_t base) {                       \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_BASE_ADDR_REG_OFFSET, base);                \
    }                                                                                                                  \
    inline __attribute__((always_inline)) void set_scatter_list_index_##buf_name(uint64_t index) {                     \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_INDEX_REG_OFFSET, index);                   \
    }                                                                                                                  \
    inline __attribute__((always_inline)) void set_scatter_list_times_##buf_name(uint64_t times) {                     \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_TIMES_REG_OFFSET, times);                   \
    }                                                                                                                  \
    inline __attribute__((always_inline)) uint64_t get_scatter_list_addr_##buf_name() {                                \
        return CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_LIST_ADDR_REG_OFFSET);               \
    }                                                                                                                  \
    inline __attribute__((always_inline)) uint64_t get_scatter_list_base_##buf_name() {                                \
        return CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_BASE_ADDR_REG_OFFSET);               \
    }                                                                                                                  \
    inline __attribute__((always_inline)) uint64_t get_scatter_list_index_##buf_name() {                               \
        return CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_INDEX_REG_OFFSET);                   \
    }                                                                                                                  \
    inline __attribute__((always_inline)) uint64_t get_scatter_list_times_##buf_name() {                               \
        return CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_TIMES_REG_OFFSET);                   \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def set_len_cmdbuf_0                                                                                           \
     * @def set_len_cmdbuf_1                                                                                           \
     *                                                                                                                 \
     * @brief Configures size of transfer in bytes                                                                     \
     *                                                                                                                 \
     * @param size_bytes Size in bytes                                                                                 \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void set_len_##buf_name(uint64_t size_bytes) {                               \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET, size_bytes);                  \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def issue_transaction_cmdbuf_0                                                                                 \
     * @def issue_transaction_cmdbuf_1                                                                                 \
     *                                                                                                                 \
     * @brief Kicks off noc transaction with previously configured command buff                                        \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void issue_##buf_name() { CMDBUF_ISSUE_TRANS(cmdbuf); }                      \
                                                                                                                       \
    /*                                                                                                                 \
     * @def issue_read_cmdbuf_0                                                                                        \
     * @def issue_read_cmdbuf_1                                                                                        \
     *                                                                                                                 \
     * @brief Issues read transaction                                                                                  \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void issue_read_##buf_name() { issue_##buf_name(); }                         \
                                                                                                                       \
    /*                                                                                                                 \
     * @def issue_write_cmdbuf_0                                                                                       \
     * @def issue_write_cmdbuf_1                                                                                       \
     *                                                                                                                 \
     * @brief Issues write transaction                                                                                 \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void issue_write_##buf_name() { issue_##buf_name(); }                        \
                                                                                                                       \
    /*                                                                                                                 \
     * @def issue_write_inline_cmdbuf_0                                                                                \
     * @def issue_write_inline_cmdbuf_1                                                                                \
     *                                                                                                                 \
     * @brief Kicks off inline noc transaction with underling custom ASM instruction                                   \
     *                                                                                                                 \
     * @param data Inline data to be written                                                                           \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void issue_write_inline_##buf_name(uint64_t data) {                          \
        CMDBUF_ISSUE_INLINE_TRANS(cmdbuf, data);                                                                       \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def issue_write_inline_len_cmdbuf_0                                                                            \
     * @def issue_write_inline_len_cmdbuf_1                                                                            \
     *                                                                                                                 \
     * @brief Kicks off inline noc transaction with underling custom ASM instruction                                   \
     *                                                                                                                 \
     * @param data Inline data to be written                                                                           \
     * @param dest_addr Destination address (with or without xy coordinates embedded)                                  \
     * @param size_bytes Size of data in bytes                                                                         \
     * @param has_xy Flag for specifying if address contains noc coordinates using NOC_XY_COORD                        \
     * @param posted Flag if transfer should be posted or not                                                          \
     * @param snoop Flag for enabling snoop bit                                                                        \
     * @param flush Flag for enabling flush bit                                                                        \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void issue_write_inline_len_##buf_name(                                      \
        uint64_t data,                                                                                                 \
        uint64_t dest_addr,                                                                                            \
        uint64_t size_bytes,                                                                                           \
        uint64_t has_xy = 1,                                                                                           \
        uint64_t posted = 0,                                                                                           \
        uint64_t snoop = 0,                                                                                            \
        uint64_t flush = 0) {                                                                                          \
        uint64_t rs2 =                                                                                                 \
            dest_addr | ((size_bytes - 1) << 57) | (has_xy << 60) | (posted << 61) | (snoop << 62) | (flush << 63);    \
        CMDBUF_ISSUE_INLINE_ADDR_TRANS(cmdbuf, data, rs2);                                                             \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void interrupt_enable_##buf_name(int id) {                                   \
        TT_ROCC_CMD_BUF_IE_reg_u val;                                                                                  \
        val.val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET);                           \
        val.val |= (1 << id);                                                                                          \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET, val.val);                            \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void interrupt_disable_##buf_name(int id) {                                  \
        TT_ROCC_CMD_BUF_IE_reg_u val;                                                                                  \
        val.val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET);                           \
        val.val &= ~(1 << id);                                                                                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET, val.val);                            \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint64_t interrupts_pending_##buf_name() {                                   \
        TT_ROCC_CMD_BUF_IP_reg_u val;                                                                                  \
        val.val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET);                           \
        return val.val;                                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void interrupt_clear_##buf_name(int id) {                                    \
        TT_ROCC_CMD_BUF_IP_reg_u val;                                                                                  \
        val.val = ~(1 << id);                                                                                          \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET, val.val);                            \
    }                                                                                                                  \
    /* Per-TRID Count Zero Interrupts (IE_0[31:0], IP_0[31:0]) */                                                      \
    inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_enable_##buf_name(int trid) {             \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET);                   \
        val |= (1ULL << trid);                                                                                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_disable_##buf_name(int trid) {            \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET);                   \
        val &= ~(1ULL << trid);                                                                                        \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint32_t per_trid_count_zero_get_interrupt_enable_##buf_name() {             \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET);                   \
        return val & 0xFFFFFFFFULL; /* Lower 32 bits */                                                                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_count_zero_set_interrupt_enable_##buf_name(uint32_t val) {     \
        uint64_t reg_val;                                                                                              \
        reg_val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET);               \
        reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);                                           \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET, reg_val);                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint64_t per_trid_count_zero_interrupts_pending_##buf_name() {               \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET);                   \
        return val & 0xFFFFFFFFULL; /* Lower 32 bits */                                                                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint32_t per_trid_count_zero_get_interrupt_pending_##buf_name() {            \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET);                   \
        return val & 0xFFFFFFFFULL; /* Lower 32 bits */                                                                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_count_zero_set_interrupt_pending_##buf_name(uint32_t val) {    \
        uint64_t reg_val;                                                                                              \
        reg_val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET);               \
        reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);                                           \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET, reg_val);                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_clear_##buf_name(int trid) {              \
        uint64_t val;                                                                                                  \
        val = ~(1ULL << trid);                                                                                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    /* Per-TRID Write Count Zero Interrupts (IE_0[63:32], IP_0[63:32]) */                                              \
    inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_enable_##buf_name(int trid) {          \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET);                   \
        val |= (1ULL << (trid + 32));                                                                                  \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_disable_##buf_name(int trid) {         \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET);                   \
        val &= ~(1ULL << (trid + 32));                                                                                 \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint32_t per_trid_wr_count_zero_get_interrupt_enable_##buf_name() {          \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET);                   \
        return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */                                                        \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_wr_count_zero_set_interrupt_enable_##buf_name(uint32_t val) {  \
        uint64_t reg_val;                                                                                              \
        reg_val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET);               \
        reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET, reg_val);                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint64_t per_trid_wr_count_zero_interrupts_pending_##buf_name() {            \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET);                   \
        return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */                                                        \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint32_t per_trid_wr_count_zero_get_interrupt_pending_##buf_name() {         \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET);                   \
        return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */                                                        \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_wr_count_zero_set_interrupt_pending_##buf_name(uint32_t val) { \
        uint64_t reg_val;                                                                                              \
        reg_val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET);               \
        reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET, reg_val);                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_clear_##buf_name(int trid) {           \
        uint64_t val;                                                                                                  \
        val = ~(1ULL << (trid + 32));                                                                                  \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    /* Per-TRID iDMA Count Zero Interrupts (IE_1[31:0], IP_1[31:0]) */                                                 \
    inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_enable_##buf_name(int trid) {        \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET);                   \
        val |= (1ULL << trid);                                                                                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_disable_##buf_name(int trid) {       \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET);                   \
        val &= ~(1ULL << trid);                                                                                        \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint32_t per_trid_idma_count_zero_get_interrupt_enable_##buf_name() {        \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET);                   \
        return val & 0xFFFFFFFFULL; /* Lower 32 bits */                                                                \
    }                                                                                                                  \
                                                                                                                       \
    inline                                                                                                             \
        __attribute__((always_inline)) void per_trid_idma_count_zero_set_interrupt_enable_##buf_name(uint32_t val) {   \
        uint64_t reg_val;                                                                                              \
        reg_val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET);               \
        reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);                                           \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET, reg_val);                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint64_t per_trid_idma_count_zero_interrupts_pending_##buf_name() {          \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET);                   \
        return val & 0xFFFFFFFFULL;                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint32_t per_trid_idma_count_zero_get_interrupt_pending_##buf_name() {       \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET);                   \
        return val & 0xFFFFFFFFULL; /* Lower 32 bits */                                                                \
    }                                                                                                                  \
                                                                                                                       \
    inline                                                                                                             \
        __attribute__((always_inline)) void per_trid_idma_count_zero_set_interrupt_pending_##buf_name(uint32_t val) {  \
        uint64_t reg_val;                                                                                              \
        reg_val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET);               \
        reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);                                           \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET, reg_val);                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_clear_##buf_name(int trid) {         \
        uint64_t val;                                                                                                  \
        val = ~(1ULL << trid);                                                                                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    /* Per-TRID Tiles-to-Process TR_ACK Threshold Interrupts (IE_1[63:32], IP_1[63:32]) */                             \
    inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_enable_##buf_name(int trid) {       \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET);                   \
        val |= (1ULL << (trid + 32));                                                                                  \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_disable_##buf_name(int trid) {      \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET);                   \
        val &= ~(1ULL << (trid + 32));                                                                                 \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint32_t per_trid_tiles_to_process_get_interrupt_enable_##buf_name() {       \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET);                   \
        return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */                                                        \
    }                                                                                                                  \
                                                                                                                       \
    inline                                                                                                             \
        __attribute__((always_inline)) void per_trid_tiles_to_process_set_interrupt_enable_##buf_name(uint32_t val) {  \
        uint64_t reg_val;                                                                                              \
        reg_val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET);               \
        reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET, reg_val);                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint64_t per_trid_tiles_to_process_interrupts_pending_##buf_name() {         \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET);                   \
        return (val >> 32) & 0xFFFFFFFFULL;                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint32_t per_trid_tiles_to_process_get_interrupt_pending_##buf_name() {      \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET);                   \
        return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */                                                        \
    }                                                                                                                  \
                                                                                                                       \
    inline                                                                                                             \
        __attribute__((always_inline)) void per_trid_tiles_to_process_set_interrupt_pending_##buf_name(uint32_t val) { \
        uint64_t reg_val;                                                                                              \
        reg_val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET);               \
        reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET, reg_val);                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_clear_##buf_name(int trid) {        \
        uint64_t val;                                                                                                  \
        val = ~(1ULL << (trid + 32));                                                                                  \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    /* Per-TRID Write Tiles-to-Process WR_SENT Threshold Interrupts (IE_2[31:0], IP_2[31:0]) */                        \
    inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_enable_##buf_name(int trid) {    \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET);                   \
        val |= (1ULL << trid);                                                                                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_disable_##buf_name(int trid) {   \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET);                   \
        val &= ~(1ULL << trid);                                                                                        \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint32_t per_trid_wr_tiles_to_process_get_interrupt_enable_##buf_name() {    \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET);                   \
        return val & 0xFFFFFFFFULL; /* Lower 32 bits */                                                                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_set_interrupt_enable_##buf_name(           \
        uint32_t val) {                                                                                                \
        uint64_t reg_val;                                                                                              \
        reg_val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET);               \
        reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);                                           \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET, reg_val);                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint64_t per_trid_wr_tiles_to_process_interrupts_pending_##buf_name() {      \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET);                   \
        return val & 0xFFFFFFFFULL;                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint32_t per_trid_wr_tiles_to_process_get_interrupt_pending_##buf_name() {   \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET);                   \
        return val & 0xFFFFFFFFULL; /* Lower 32 bits */                                                                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_set_interrupt_pending_##buf_name(          \
        uint32_t val) {                                                                                                \
        uint64_t reg_val;                                                                                              \
        reg_val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET);               \
        reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);                                           \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET, reg_val);                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_clear_##buf_name(int trid) {     \
        uint64_t val;                                                                                                  \
        val = ~(1ULL << trid);                                                                                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    /* Per-TRID iDMA Tiles-to-Process IDMA_TR_ACK Threshold Interrupts (IE_2[63:32], IP_2[63:32]) */                   \
    inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_enable_##buf_name(int trid) {  \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET);                   \
        val |= (1ULL << (trid + 32));                                                                                  \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_disable_##buf_name(int trid) { \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET);                   \
        val &= ~(1ULL << (trid + 32));                                                                                 \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint32_t per_trid_idma_tiles_to_process_get_interrupt_enable_##buf_name() {  \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET);                   \
        return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */                                                        \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_set_interrupt_enable_##buf_name(         \
        uint32_t val) {                                                                                                \
        uint64_t reg_val;                                                                                              \
        reg_val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET);               \
        reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET, reg_val);                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint64_t per_trid_idma_tiles_to_process_interrupts_pending_##buf_name() {    \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET);                   \
        return (val >> 32) & 0xFFFFFFFFULL;                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint32_t per_trid_idma_tiles_to_process_get_interrupt_pending_##buf_name() { \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET);                   \
        return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */                                                        \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_set_interrupt_pending_##buf_name(        \
        uint32_t val) {                                                                                                \
        uint64_t reg_val;                                                                                              \
        reg_val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET);               \
        reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);                         \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET, reg_val);                \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_clear_##buf_name(int trid) {   \
        uint64_t val;                                                                                                  \
        val = ~(1ULL << (trid + 32));                                                                                  \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET, val);                    \
    }                                                                                                                  \
                                                                                                                       \
    /* Per-VC Has Space Interrupts (IE[47:32], IP[47:32]) */                                                           \
    inline __attribute__((always_inline)) void per_vc_has_space_interrupt_enable_##buf_name(int vc) {                  \
        TT_ROCC_CMD_BUF_IE_reg_u val;                                                                                  \
        val.val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET);                           \
        val.val |= (1ULL << (vc + 32));                                                                                \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET, val.val);                            \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_vc_has_space_interrupt_disable_##buf_name(int vc) {                 \
        TT_ROCC_CMD_BUF_IE_reg_u val;                                                                                  \
        val.val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET);                           \
        val.val &= ~(1ULL << (vc + 32));                                                                               \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET, val.val);                            \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint16_t per_vc_has_space_get_interrupt_enable_##buf_name() {                \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET);                               \
        return (val >> 32) & 0xFFFFULL; /* Bits [47:32] */                                                             \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_vc_has_space_set_interrupt_enable_##buf_name(uint16_t val) {        \
        uint64_t reg_val;                                                                                              \
        reg_val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET);                           \
        reg_val = (reg_val & 0xFFFF0000FFFFFFFFULL) | (((uint64_t)(val & 0xFFFFULL)) << 32);                           \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET, reg_val);                            \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint64_t per_vc_has_space_interrupts_pending_##buf_name() {                  \
        TT_ROCC_CMD_BUF_IP_reg_u val;                                                                                  \
        val.val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET);                           \
        return (val.val >> 32) & 0xFFFFULL; /* Bits [47:32] shifted to [15:0] */                                       \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint16_t per_vc_has_space_get_interrupt_pending_##buf_name() {               \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET);                               \
        return (val >> 32) & 0xFFFFULL; /* Bits [47:32] */                                                             \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_vc_has_space_set_interrupt_pending_##buf_name(uint16_t val) {       \
        uint64_t reg_val;                                                                                              \
        reg_val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET);                           \
        reg_val = (reg_val & 0xFFFF0000FFFFFFFFULL) | (((uint64_t)(val & 0xFFFFULL)) << 32);                           \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET, reg_val);                            \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_vc_has_space_interrupt_clear_##buf_name(int vc) {                   \
        TT_ROCC_CMD_BUF_IP_reg_u val;                                                                                  \
        val.val = ~(1ULL << (vc + 32));                                                                                \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET, val.val);                            \
    }                                                                                                                  \
                                                                                                                       \
    /* Per-iDMA-VC Has Space Interrupts (IE[63:48], IP[63:48]) */                                                      \
    inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_enable_##buf_name(int vc) {             \
        TT_ROCC_CMD_BUF_IE_reg_u val;                                                                                  \
        val.val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET);                           \
        val.val |= (1ULL << (vc + 48));                                                                                \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET, val.val);                            \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_disable_##buf_name(int vc) {            \
        TT_ROCC_CMD_BUF_IE_reg_u val;                                                                                  \
        val.val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET);                           \
        val.val &= ~(1ULL << (vc + 48));                                                                               \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET, val.val);                            \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint16_t per_idma_vc_has_space_get_interrupt_enable_##buf_name() {           \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET);                               \
        return (val >> 48) & 0xFFFFULL; /* Bits [63:48] */                                                             \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_idma_vc_has_space_set_interrupt_enable_##buf_name(uint16_t val) {   \
        uint64_t reg_val;                                                                                              \
        reg_val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET);                           \
        reg_val = (reg_val & 0x0000FFFFFFFFFFFFULL) | (((uint64_t)(val & 0xFFFFULL)) << 48);                           \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET, reg_val);                            \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint64_t per_idma_vc_has_space_interrupts_pending_##buf_name() {             \
        TT_ROCC_CMD_BUF_IP_reg_u val;                                                                                  \
        val.val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET);                           \
        return (val.val >> 48) & 0xFFFFULL; /* Bits [63:48] shifted to [15:0] */                                       \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) uint16_t per_idma_vc_has_space_get_interrupt_pending_##buf_name() {          \
        uint64_t val;                                                                                                  \
        val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET);                               \
        return (val >> 48) & 0xFFFFULL; /* Bits [63:48] */                                                             \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_idma_vc_has_space_set_interrupt_pending_##buf_name(uint16_t val) {  \
        uint64_t reg_val;                                                                                              \
        reg_val = CMDBUF_RD_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET);                           \
        reg_val = (reg_val & 0x0000FFFFFFFFFFFFULL) | (((uint64_t)(val & 0xFFFFULL)) << 48);                           \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET, reg_val);                            \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_clear_##buf_name(int vc) {              \
        TT_ROCC_CMD_BUF_IP_reg_u val;                                                                                  \
        val.val = ~(1ULL << (vc + 48));                                                                                \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET, val.val);                            \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def noc_fast_read_cmdbuf_0                                                                                     \
     * @def noc_fast_read_cmdbuf_1                                                                                     \
     *                                                                                                                 \
     * @brief Standalone noc read function which uses custom ASM instruction, bypassing all configurations             \
     * Does not need any other configuring                                                                             \
     *                                                                                                                 \
     * @param src_addr Remote source address                                                                           \
     * @param dest_addr Local L1 destination address                                                                   \
     * @param len_bytes Size of data in bytes                                                                          \
     * @param has_xy Flag for specifying if address contains noc coordinates using NOC_XY_COORD                        \
     * @param posted Flag if transfer should be posted or not                                                          \
     * @param snoop Flag for enabling snoop bit                                                                        \
     * @param flush Flag for enabling flush bit                                                                        \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void noc_fast_read_##buf_name(                                               \
        uint64_t src_addr,                                                                                             \
        uint32_t dest_addr,                                                                                            \
        uint64_t len_bytes,                                                                                            \
        uint64_t has_xy = 1,                                                                                           \
        uint64_t posted = 0,                                                                                           \
        uint64_t snoop = 0,                                                                                            \
        uint64_t flush = 0) {                                                                                          \
        uint64_t rs1 = (len_bytes << 32) | dest_addr;                                                                  \
        uint64_t rs2 = src_addr | (has_xy << 60) | (posted << 61) | (snoop << 62) | (flush << 63);                     \
        CMDBUF_ISSUE_READ2_TRANS(cmdbuf, rs1, rs2);                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def noc_read_cmdbuf_0                                                                                          \
     * @def noc_read_cmdbuf_1                                                                                          \
     *                                                                                                                 \
     * @brief Noc read function which uses other function from this API to configure all needed                        \
     *                                                                                                                 \
     * @param src_coordinate Coordinate of source core packed with NOC_XY_COORD macro                                  \
     * @param src_addr Remote source address                                                                           \
     * @param dest_coordinate Coordinate of destination core packed with NOC_XY_COORD macro                            \
     * @param dest_addr Local L1 destination address                                                                   \
     * @param len_bytes Size of data in bytes                                                                          \
     * @param transaction_id Transaction ID for this operation                                                         \
     * @param snoop_bit Flag for enabling snoop bit                                                                    \
     * @param flush_bit Flag for enabling flush bit                                                                    \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void noc_read_##buf_name(                                                    \
        uint64_t src_coordinate,                                                                                       \
        uint64_t src_addr,                                                                                             \
        uint64_t dest_coordinate,                                                                                      \
        uint64_t dest_addr,                                                                                            \
        uint64_t len_bytes,                                                                                            \
        uint32_t transaction_id = CMDBUF_DEF_TRID,                                                                     \
        bool snoop_bit = false,                                                                                        \
        bool flush_bit = false) {                                                                                      \
        reset_##buf_name();                                                                                            \
        setup_as_copy_##buf_name(false, false, {0}, false);                                                            \
        setup_ongoing_##buf_name(false, false, false, false, false);                                                   \
        setup_vcs_##buf_name(false);                                                                                   \
        setup_trids_##buf_name(transaction_id);                                                                        \
        if (snoop_bit || flush_bit)                                                                                    \
            setup_packet_tags_##buf_name(snoop_bit, flush_bit);                                                        \
        set_src_##buf_name(src_addr, src_coordinate);                                                                  \
        set_dest_##buf_name(dest_addr, dest_coordinate);                                                               \
        set_len_##buf_name(len_bytes);                                                                                 \
        issue_read_##buf_name();                                                                                       \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void noc_write_prep_##buf_name(                                              \
        uint64_t src_coordinate,                                                                                       \
        uint64_t src_addr,                                                                                             \
        uint64_t dest_coordinate,                                                                                      \
        uint64_t dest_addr,                                                                                            \
        uint64_t len_bytes,                                                                                            \
        uint32_t transaction_id = CMDBUF_DEF_TRID,                                                                     \
        bool mcast = false,                                                                                            \
        bool snoop_bit = false,                                                                                        \
        bool flush_bit = false,                                                                                        \
        bool posted = true) {                                                                                          \
        reset_##buf_name();                                                                                            \
        setup_as_copy_##buf_name(true, mcast, {0}, false, posted);                                                     \
        setup_ongoing_##buf_name(false, false, false, false, false);                                                   \
        setup_vcs_##buf_name(true, mcast);                                                                             \
        setup_trids_##buf_name(transaction_id);                                                                        \
        if (snoop_bit || flush_bit)                                                                                    \
            setup_packet_tags_##buf_name(snoop_bit, flush_bit);                                                        \
        set_src_##buf_name(src_addr, src_coordinate);                                                                  \
        set_dest_##buf_name(dest_addr, dest_coordinate);                                                               \
        set_len_##buf_name(len_bytes);                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void noc_read_prep_##buf_name(                                               \
        uint64_t src_coordinate,                                                                                       \
        uint64_t src_addr,                                                                                             \
        uint64_t dest_coordinate,                                                                                      \
        uint64_t dest_addr,                                                                                            \
        uint64_t len_bytes,                                                                                            \
        uint32_t transaction_id = CMDBUF_DEF_TRID,                                                                     \
        bool snoop_bit = false,                                                                                        \
        bool flush_bit = false) {                                                                                      \
        reset_##buf_name();                                                                                            \
        setup_as_copy_##buf_name(false, false, {0}, false);                                                            \
        setup_ongoing_##buf_name(false, false, false, false, false);                                                   \
        setup_vcs_##buf_name(false);                                                                                   \
        setup_trids_##buf_name(transaction_id);                                                                        \
        if (snoop_bit || flush_bit)                                                                                    \
            setup_packet_tags_##buf_name(snoop_bit, flush_bit);                                                        \
        set_src_##buf_name(src_addr, src_coordinate);                                                                  \
        set_dest_##buf_name(dest_addr, dest_coordinate);                                                               \
        set_len_##buf_name(len_bytes);                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def noc_fast_write_cmdbuf_0                                                                                    \
     * @def noc_fast_write_cmdbuf_1                                                                                    \
     *                                                                                                                 \
     * @brief Standalone noc write function which uses custom ASM instruction, bypassing all configurations            \
     * Does not need any other configuring (reset cmd buffer before use)                                               \
     *                                                                                                                 \
     * @param src_addr Local L1 source address                                                                         \
     * @param dest_addr Remote destination address                                                                     \
     * @param len_bytes Size of data in bytes                                                                          \
     * @param has_xy Flag for specifying if address contains noc coordinates using NOC_XY_COORD                        \
     * @param posted Flag if transfer should be posted or not                                                          \
     * @param snoop Flag for enabling snoop bit                                                                        \
     * @param flush Flag for enabling flush bit                                                                        \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void noc_fast_write_##buf_name(                                              \
        uint32_t src_addr,                                                                                             \
        uint64_t dest_addr,                                                                                            \
        uint64_t len_bytes,                                                                                            \
        uint64_t has_xy = 1,                                                                                           \
        uint64_t posted = 0,                                                                                           \
        uint64_t snoop = 0,                                                                                            \
        uint64_t flush = 0) {                                                                                          \
        uint64_t rs1 = (len_bytes << 32) | src_addr;                                                                   \
        uint64_t rs2 = dest_addr | (has_xy << 60) | (posted << 61) | (snoop << 62) | (flush << 63);                    \
        CMDBUF_ISSUE_WRITE2_TRANS(cmdbuf, rs1, rs2);                                                                   \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def noc_write_cmdbuf_0                                                                                         \
     * @def noc_write_cmdbuf_1                                                                                         \
     *                                                                                                                 \
     * @brief Noc write function which uses other function from this API to configure all needed                       \
     *                                                                                                                 \
     * @param src_coordinate Coordinate of source core                                                                 \
     * @param src_addr Local L1 source address                                                                         \
     * @param dest_coordinate Coordinate of destination core                                                           \
     * @param dest_addr Remote destination address                                                                     \
     * @param len_bytes Size of data in bytes                                                                          \
     * @param transaction_id Transaction ID for this operation                                                         \
     * @param mcast Enable multicast                                                                                   \
     * @param snoop_bit Flag for enabling snoop bit                                                                    \
     * @param flush_bit Flag for enabling flush bit                                                                    \
     * @param posted Flag if transfer should be posted or not                                                          \
     * @param mcast_exclude Multicast exclusion settings                                                               \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void noc_write_##buf_name(                                                   \
        uint64_t src_coordinate,                                                                                       \
        uint64_t src_addr,                                                                                             \
        uint64_t dest_coordinate,                                                                                      \
        uint64_t dest_addr,                                                                                            \
        uint64_t len_bytes,                                                                                            \
        uint32_t transaction_id = CMDBUF_DEF_TRID,                                                                     \
        bool mcast = false,                                                                                            \
        bool snoop_bit = false,                                                                                        \
        bool flush_bit = false,                                                                                        \
        bool posted = true,                                                                                            \
        TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u mcast_exclude = {0}) {                                                     \
        reset_##buf_name();                                                                                            \
        setup_as_copy_##buf_name(true, mcast, mcast_exclude, false, posted);                                           \
        setup_ongoing_##buf_name(false, false, false, false, false);                                                   \
        setup_vcs_##buf_name(true, mcast);                                                                             \
        setup_trids_##buf_name(transaction_id);                                                                        \
        if (snoop_bit || flush_bit)                                                                                    \
            setup_packet_tags_##buf_name(snoop_bit, flush_bit);                                                        \
        set_src_##buf_name(src_addr, src_coordinate);                                                                  \
        set_dest_##buf_name(dest_addr, dest_coordinate);                                                               \
        set_len_##buf_name(len_bytes);                                                                                 \
        issue_write_##buf_name();                                                                                      \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def idma_copy_cmdbuf_0                                                                                         \
     * @def idma_copy_cmdbuf_1                                                                                         \
     *                                                                                                                 \
     * @brief Complete iDMA copy operation                                                                             \
     *                                                                                                                 \
     * @param src_addr Source address                                                                                  \
     * @param dest_addr Destination address                                                                            \
     * @param len_bytes Size of data in bytes                                                                          \
     * @param transaction_id Transaction ID for this operation                                                         \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void idma_copy_##buf_name(                                                   \
        uint64_t src_addr, uint64_t dest_addr, uint64_t len_bytes, uint32_t transaction_id = CMDBUF_DEF_TRID) {        \
        reset_##buf_name();                                                                                            \
        idma_setup_as_copy_##buf_name(false);                                                                          \
        setup_ongoing_##buf_name(false, false, false, false, false);                                                   \
        setup_vcs_##buf_name(true);                                                                                    \
        setup_trids_##buf_name(transaction_id);                                                                        \
        set_src_##buf_name(src_addr);                                                                                  \
        set_dest_##buf_name(dest_addr);                                                                                \
        set_len_##buf_name(len_bytes);                                                                                 \
        issue_##buf_name();                                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def noc_atomic_increment_cmdbuf_0                                                                              \
     * @def noc_atomic_increment_cmdbuf_1                                                                              \
     *                                                                                                                 \
     * @brief Atomic increment function                                                                                \
     *                                                                                                                 \
     * @param noc_coordinate NOC coordinate of target                                                                  \
     * @param addr Remote destination address                                                                          \
     * @param incr Increment value                                                                                     \
     * @param wrap Wrap value                                                                                          \
     * @param snoop_bit Flag for enabling snoop bit                                                                    \
     * @param flush_bit Flag for enabling flush bit                                                                    \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) void noc_atomic_increment_##buf_name(                                        \
        uint64_t noc_coordinate,                                                                                       \
        uint64_t addr,                                                                                                 \
        uint32_t incr = 1,                                                                                             \
        uint32_t wrap = 31,                                                                                            \
        bool snoop_bit = false,                                                                                        \
        bool flush_bit = false) {                                                                                      \
        uint64_t at_len = NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) |     \
                          NOC_AT_IND_32_SRC(0);                                                                        \
        setup_as_atomic_##buf_name(true);                                                                              \
        if (snoop_bit || flush_bit)                                                                                    \
            setup_packet_tags_##buf_name(snoop_bit, flush_bit);                                                        \
        set_dest_##buf_name(addr, noc_coordinate);                                                                     \
        set_len_##buf_name(at_len);                                                                                    \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_INLINE_DATA_REG_OFFSET, (uint64_t)incr);            \
        issue_##buf_name();                                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def free_space_cmdbuf_0                                                                                        \
     * @def free_space_cmdbuf_1                                                                                        \
     *                                                                                                                 \
     * @brief Returns amount of free buffer space in virtual channel buffer                                            \
     *                                                                                                                 \
     * @param vc Virtual channel ID                                                                                    \
     * @return Amount in bytes                                                                                         \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) uint64_t free_space_##buf_name(uint32_t vc) {                                \
        return CMDBUF_GET_VC_SPACE_VC(cmdbuf, vc);                                                                     \
    }                                                                                                                  \
    inline __attribute__((always_inline)) uint64_t free_space_##buf_name() { return CMDBUF_GET_VC_SPACE(cmdbuf); }     \
                                                                                                                       \
    inline __attribute__((always_inline)) uint64_t idma_free_space_##buf_name(uint32_t vc) {                           \
        return CMDBUF_IDMA_GET_VC_SPACE_VC(cmdbuf, vc);                                                                \
    }                                                                                                                  \
    inline __attribute__((always_inline)) uint64_t idma_free_space_##buf_name() {                                      \
        return CMDBUF_IDMA_GET_VC_SPACE(cmdbuf);                                                                       \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def noc_reads_acked_cmdbuf_0                                                                                   \
     * @def noc_reads_acked_cmdbuf_1                                                                                   \
     *                                                                                                                 \
     * @brief Checks if transaction with argument trid is completed                                                    \
     *                                                                                                                 \
     * @param transaction_id Transaction id to check                                                                   \
     * @return True if all transcation is complected                                                                   \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) bool noc_reads_acked_##buf_name(uint32_t transaction_id) {                   \
        return CMDBUF_TR_ACK_TRID(cmdbuf, transaction_id) == 0;                                                        \
    }                                                                                                                  \
    inline __attribute__((always_inline)) bool noc_reads_acked_##buf_name() { return CMDBUF_TR_ACK(cmdbuf) == 0; }     \
                                                                                                                       \
    inline __attribute__((always_inline)) bool all_noc_reads_acked_##buf_name() {                                      \
        bool all = true;                                                                                               \
        if (cmdbuf == CMDBUF_0) {                                                                                      \
            for (uint32_t k = CMDBUF_0_TRID_STATIC; k <= CMDBUF_0_TRID_END; k++) {                                     \
                all = all && CMDBUF_TR_ACK_TRID(cmdbuf, k) == 0;                                                       \
            }                                                                                                          \
        } else {                                                                                                       \
            for (uint32_t k = CMDBUF_1_TRID_STATIC; k <= CMDBUF_1_TRID_END; k++) {                                     \
                all = all && CMDBUF_TR_ACK_TRID(cmdbuf, k) == 0;                                                       \
            }                                                                                                          \
        }                                                                                                              \
        return all;                                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) bool idma_acked_##buf_name(uint32_t transaction_id) {                        \
        return CMDBUF_IDMA_TR_ACK_TRID(cmdbuf, transaction_id) == 0;                                                   \
    }                                                                                                                  \
    inline __attribute__((always_inline)) bool idma_acked_##buf_name() { return CMDBUF_IDMA_TR_ACK(cmdbuf) == 0; }     \
                                                                                                                       \
    inline __attribute__((always_inline)) bool all_idma_acked_##buf_name() {                                           \
        bool all = true;                                                                                               \
        if (cmdbuf == CMDBUF_0) {                                                                                      \
            for (uint32_t k = CMDBUF_0_TRID_STATIC; k <= CMDBUF_0_TRID_END; k++) {                                     \
                all = all && CMDBUF_IDMA_TR_ACK_TRID(cmdbuf, k) == 0;                                                  \
            }                                                                                                          \
        } else {                                                                                                       \
            for (uint32_t k = CMDBUF_1_TRID_STATIC; k <= CMDBUF_1_TRID_END; k++) {                                     \
                all = all && CMDBUF_IDMA_TR_ACK_TRID(cmdbuf, k) == 0;                                                  \
            }                                                                                                          \
        }                                                                                                              \
        return all;                                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def noc_writes_sent_cmdbuf_0                                                                                   \
     * @def noc_writes_sent_cmdbuf_1                                                                                   \
     *                                                                                                                 \
     * @brief Checks if write with provided transcation ID is completed                                                \
     *                                                                                                                 \
     * @param transaction_id Transaction id to check                                                                   \
     * @return True if write is complected                                                                             \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) bool noc_writes_sent_##buf_name(uint32_t transaction_id) {                   \
        return CMDBUF_WR_SENT_TRID(cmdbuf, transaction_id) == 0;                                                       \
    }                                                                                                                  \
    inline __attribute__((always_inline)) bool noc_writes_sent_##buf_name() { return CMDBUF_WR_SENT(cmdbuf) == 0; }    \
                                                                                                                       \
    inline __attribute__((always_inline)) bool all_noc_writes_sent_##buf_name() {                                      \
        bool all = true;                                                                                               \
        if (cmdbuf == CMDBUF_0) {                                                                                      \
            for (uint32_t k = CMDBUF_0_TRID_STATIC; k <= CMDBUF_0_TRID_END; k++) {                                     \
                all = all && CMDBUF_WR_SENT_TRID(cmdbuf, k) == 0;                                                      \
            }                                                                                                          \
        } else {                                                                                                       \
            for (uint32_t k = CMDBUF_1_TRID_STATIC; k <= CMDBUF_1_TRID_END; k++) {                                     \
                all = all && CMDBUF_WR_SENT_TRID(cmdbuf, k) == 0;                                                      \
            }                                                                                                          \
        }                                                                                                              \
        return all;                                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * @def noc_nonposted_writes_acked_cmdbuf_0                                                                        \
     * @def noc_nonposted_writes_acked_cmdbuf_1                                                                        \
     *                                                                                                                 \
     * @brief Checks if nonposted write transaction is acknowledged                                                    \
     *                                                                                                                 \
     * @param transaction_id Transaction id to check                                                                   \
     * @return True if nonposted write is acknowledged                                                                 \
     *                                                                                                                 \
     * @note This macro creates 2 inline functions, 1 per each cmd buffer                                              \
     */                                                                                                                \
    inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_##buf_name(uint32_t transaction_id) {        \
        return CMDBUF_TR_ACK_TRID(cmdbuf, transaction_id) == 0;                                                        \
    }                                                                                                                  \
    inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_##buf_name() {                               \
        return CMDBUF_TR_ACK(cmdbuf) == 0;                                                                             \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) bool all_noc_nonposted_writes_acked_##buf_name() {                           \
        bool all = true;                                                                                               \
        if (cmdbuf == CMDBUF_0) {                                                                                      \
            for (uint32_t k = CMDBUF_0_TRID_STATIC; k <= CMDBUF_0_TRID_END; k++) {                                     \
                all = all && CMDBUF_TR_ACK_TRID(cmdbuf, k) == 0;                                                       \
            }                                                                                                          \
        } else {                                                                                                       \
            for (uint32_t k = CMDBUF_1_TRID_STATIC; k <= CMDBUF_1_TRID_END; k++) {                                     \
                all = all && CMDBUF_TR_ACK_TRID(cmdbuf, k) == 0;                                                       \
            }                                                                                                          \
        }                                                                                                              \
        return all;                                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void l1_atomic_instr_##buf_name(                                             \
        uint32_t fmt, bool no_sat, uint32_t atomic_op) {                                                               \
        TT_ROCC_CMD_BUF_L1_ACCUM_CFG_reg_u l1_atomic_instr;                                                            \
        l1_atomic_instr.val = TT_ROCC_CMD_BUF_L1_ACCUM_CFG_REG_DEFAULT;                                                \
                                                                                                                       \
        l1_atomic_instr.f.l1_atomic_fmt = fmt;                                                                         \
        l1_atomic_instr.f.disable_sat = no_sat;                                                                        \
        l1_atomic_instr.f.l1_atomic_operation = atomic_op;                                                             \
                                                                                                                       \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_L1_ACCUM_CFG_REG_ADDR, l1_atomic_instr.val);        \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void idma_setup_as_atomic_accum_##buf_name(bool wrapping_en = true) {        \
        TT_ROCC_CMD_BUF_MISC_reg_u misc;                                                                               \
        misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;                                                                   \
                                                                                                                       \
        misc.f.write_trans = 1;                                                                                        \
        misc.f.idma_en = 1;                                                                                            \
        misc.f.wrapping_en = wrapping_en;                                                                              \
        misc.f.l1_accum_en = 1;                                                                                        \
                                                                                                                       \
        CMDBUF_WR_REG(cmdbuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET, misc.val);                         \
    }                                                                                                                  \
                                                                                                                       \
    inline __attribute__((always_inline)) void idma_l1_atomic_accum_##buf_name(                                        \
        uint64_t src_addr,                                                                                             \
        uint64_t dest_addr,                                                                                            \
        uint64_t len_bytes,                                                                                            \
        uint32_t transaction_id = CMDBUF_DEF_TRID,                                                                     \
        uint64_t fmt = 0x0,                                                                                            \
        uint64_t op = 0x9) {                                                                                           \
        reset_##buf_name();                                                                                            \
        idma_setup_as_atomic_accum_##buf_name(false);                                                                  \
        l1_atomic_instr_##buf_name(fmt, false, op);                                                                    \
        setup_ongoing_##buf_name(false, false, false, false, false);                                                   \
        setup_vcs_##buf_name(true);                                                                                    \
        setup_trids_##buf_name(transaction_id);                                                                        \
        set_src_##buf_name(src_addr);                                                                                  \
        set_dest_##buf_name(dest_addr);                                                                                \
        set_len_##buf_name(len_bytes);                                                                                 \
        issue_##buf_name();                                                                                            \
    }

DEFINE_CMD_BUFS(cmdbuf_0, CMDBUF_0)
DEFINE_CMD_BUFS(cmdbuf_1, CMDBUF_1)

#undef DEFINE_CMD_BUFS

//////////////////////
/// Simple CMD Buf ///
// Bellow are all the functions for simple command buffer. This third command buffer,
//
//////////////////////

inline __attribute__((always_inline)) void reset_reg_cmdbuf() { SCMDBUF_RESET(); }

inline __attribute__((always_inline)) void setup_as_copy_reg_cmdbuf(
    bool wr, bool mcast = false, TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u mcast_exclude = {0}, bool posted = true) {
    TT_ROCC_CMD_BUF_MISC_reg_u misc;
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

    misc.f.linked = mcast;
    misc.f.posted = wr && posted;
    misc.f.multicast = mcast;
    misc.f.write_trans = wr;

    SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET, misc.val);

    if (mcast) {
        SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_EXCLUDE_REG_OFFSET, mcast_exclude.val);
    }
}

inline __attribute__((always_inline)) void setup_as_atomic_reg_cmdbuf(bool wr) {
    TT_ROCC_CMD_BUF_MISC_reg_u misc;
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

    misc.f.posted = 1;
    misc.f.write_trans = 0;
    misc.f.atomic_trans = 1;

    SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET, misc.val);
}

inline __attribute__((always_inline)) void setup_vcs_reg_cmdbuf(bool wr, bool mcast = false) {
    if (wr) {
        SCMDBUF_WR_REG(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET, mcast ? CMDBUF_MCAST_REQ_VC : CMDBUF_WR_REQ_VC);
        SCMDBUF_WR_REG(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET, mcast ? CMDBUF_MCAST_RESP_VC : CMDBUF_WR_RESP_VC);
    } else {
        SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET, CMDBUF_RD_REQ_VC);
        SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET, CMDBUF_RD_RESP_VC);
    }
}

inline __attribute__((always_inline)) void setup_trids_reg_cmdbuf(uint32_t trid_offset = CMDBUF_DEF_TRID) {
    SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET, trid_offset);
}

inline __attribute__((always_inline)) uint32_t swap_trid_reg_cmdbuf(uint32_t new_trid) {
    uint32_t prev_trid = SCMDBUF_RD_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET);
    SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET, new_trid);
    return prev_trid;
}

inline __attribute__((always_inline)) void setup_packet_tags_reg_cmdbuf(bool snoop_bit, bool flush_bit) {
    TT_ROCC_CMD_BUF_PACKET_TAGS_reg_u misc;

    misc.f.snoop_bit = snoop_bit;
    misc.f.flush_bit = flush_bit;

    SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PACKET_TAGS_REG_OFFSET, misc.val);
}

inline __attribute__((always_inline)) uint64_t get_src_reg_cmdbuf() {
    return SCMDBUF_RD_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET);
}

inline __attribute__((always_inline)) void set_src_reg_cmdbuf(uint64_t addr, uint64_t coordinate) {
    SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET, addr);
    SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET, coordinate);
}
inline __attribute__((always_inline)) void set_src_reg_cmdbuf(uint64_t addr) {
    SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET, addr);
}

inline __attribute__((always_inline)) uint64_t get_dest_reg_cmdbuf() {
    return SCMDBUF_RD_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET);
}

inline __attribute__((always_inline)) void set_dest_reg_cmdbuf(uint64_t addr, uint64_t coordinates) {
    SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET, addr);
    SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET, coordinates);
}
inline __attribute__((always_inline)) void set_dest_reg_cmdbuf(uint64_t addr) {
    SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET, addr);
}

inline __attribute__((always_inline)) void set_len_reg_cmdbuf(uint64_t size_bytes) {
    SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET, size_bytes);
}

inline __attribute__((always_inline)) void issue_reg_cmdbuf() { SCMDBUF_ISSUE_TRANS(); }

inline __attribute__((always_inline)) void issue_read_reg_cmdbuf() { issue_reg_cmdbuf(); }

inline __attribute__((always_inline)) void issue_write_reg_cmdbuf() { issue_reg_cmdbuf(); }

inline __attribute__((always_inline)) void issue_write_inline_reg_cmdbuf(uint64_t data) {
    SCMDBUF_ISSUE_INLINE_TRANS(data);
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
    SCMDBUF_ISSUE_INLINE_ADDR_TRANS(data, rs2);
}

inline __attribute__((always_inline)) void interrupt_enable_reg_cmdbuf(int id) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = SCMDBUF_RD_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET);
    val.val |= (1 << id);
    SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET, val.val);
}

inline __attribute__((always_inline)) void interrupt_disable_reg_cmdbuf(int id) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = SCMDBUF_RD_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET);
    val.val &= ~(1 << id);
    SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET, val.val);
}

inline __attribute__((always_inline)) uint64_t interrupts_pending_reg_cmdbuf() {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = SCMDBUF_RD_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET);
    return val.val;
}

inline __attribute__((always_inline)) void interrupt_clear_reg_cmdbuf(int id) {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = ~(1 << id);
    SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET, val.val);
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
    SCMDBUF_ISSUE_READ2_TRANS(rs1, rs2);
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
    SCMDBUF_ISSUE_WRITE2_TRANS(rs1, rs2);
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
    SCMDBUF_WR_REG(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_INLINE_DATA_REG_OFFSET, (uint64_t)incr);
    issue_reg_cmdbuf();
}

inline __attribute__((always_inline)) uint64_t free_space_reg_cmdbuf(uint32_t vc) {
    return SCMDBUF_GET_VC_SPACE_VC(vc);
}
inline __attribute__((always_inline)) uint64_t free_space_reg_cmdbuf() { return SCMDBUF_GET_VC_SPACE(); }

inline __attribute__((always_inline)) bool noc_reads_acked_reg_cmdbuf(uint32_t transaction_id) {
    return SCMDBUF_TR_ACK_TRID(transaction_id) == 0;
}
inline __attribute__((always_inline)) bool noc_reads_acked_reg_cmdbuf() { return SCMDBUF_TR_ACK() == 0; }

inline __attribute__((always_inline)) bool noc_writes_sent_reg_cmdbuf(uint32_t transaction_id) {
    return SCMDBUF_WR_SENT_TRID(transaction_id) == 0;
}
inline __attribute__((always_inline)) bool noc_writes_sent_reg_cmdbuf() { return SCMDBUF_WR_SENT() == 0; }

inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_reg_cmdbuf(uint32_t transaction_id) {
    return SCMDBUF_TR_ACK_TRID(transaction_id) == 0;
}
inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_reg_cmdbuf() { return SCMDBUF_TR_ACK() == 0; }
