// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#ifndef __DM__REMAPPER_COMMON_HPP__
#define __DM__REMAPPER_COMMON_HPP__

// Base addresses for 64 pairs of ClientL/ClientR config registers
#define REMAP_CLIENT_R_CONFIG_REG_BASE_ADDR32 0x01842000
#define REMAP_CLIENT_L_CONFIG_REG_BASE_ADDR32 0x01842004
#define REMAP_CLIENT_R_STATUS_REG_BASE_ADDR32 0x01842008
#define REMAP_CLIENT_L_STATUS_REG_BASE_ADDR32 0x0184200C
#define REMAP_GLOBAL_CONTROL_REG_ADDR32 0x01842200

// Register stride for accessing different pairs (8 bytes per pair)
#define REMAP_REG_PAIR_STRIDE 0x8

// Number of ClientL/ClientR pairs
#define REMAP_NUM_PAIRS 64

// Helper macros to calculate register addresses for each pair
#define REMAP_CLIENT_R_CONFIG_REG_ADDR32(pair_idx) \
    (REMAP_CLIENT_R_CONFIG_REG_BASE_ADDR32 + ((pair_idx) * REMAP_REG_PAIR_STRIDE))
#define REMAP_CLIENT_L_CONFIG_REG_ADDR32(pair_idx) \
    (REMAP_CLIENT_L_CONFIG_REG_BASE_ADDR32 + ((pair_idx) * REMAP_REG_PAIR_STRIDE))
#define REMAP_CLIENT_R_STATUS_REG_ADDR32(pair_idx) \
    (REMAP_CLIENT_R_STATUS_REG_BASE_ADDR32 + ((pair_idx) * REMAP_REG_PAIR_STRIDE))
#define REMAP_CLIENT_L_STATUS_REG_ADDR32(pair_idx) \
    (REMAP_CLIENT_L_STATUS_REG_BASE_ADDR32 + ((pair_idx) * REMAP_REG_PAIR_STRIDE))

typedef enum clientTypes { DM_0, DM_1, DM_2, DM_3, NEO_0, NEO_1, NEO_2, NEO_3 } tClientTypes;

typedef struct {
    uint32_t id_0 : 3;
    uint32_t cnt_sel_0 : 5;
    uint32_t id_1 : 3;
    uint32_t cnt_sel_1 : 5;
    uint32_t id_2 : 3;
    uint32_t cnt_sel_2 : 5;
    uint32_t id_3 : 3;
    uint32_t cnt_sel_3 : 5;
} tCounter_remap_clientR_config_reg_out;

typedef union {
    uint32_t val;
    tCounter_remap_clientR_config_reg_out f;
} tClientR_Config_Reg_u;

// Removed: Single register pointers replaced with pair-indexed access

// ----------------------------------------------------------------------------------------

/*
typedef struct {
    uint32_t id_L : 3;
    uint32_t cnt_sel_L : 5;
    uint32_t valid :4;
    uint32_t clientl_is_producer : 1;
    uint32_t clientr_group : 1;
    uint32_t distribute : 1;
    uint32_t cfg_entry_sel : 6;
    uint32_t cfg_update_en : 1;
    uint32_t cfg_read_en : 1;
    uint32_t remap_en : 1;
} tCounter_remap_clientL_config_reg_out;
*/

typedef struct {
    uint32_t id_L : 3;
    uint32_t cnt_sel_L : 5;
    uint32_t valid : 4;
    uint32_t clientl_is_producer : 1;
    uint32_t clientr_group : 1;
    uint32_t distribute : 1;
} tCounter_remap_clientL_config_reg_out;

typedef union {
    uint32_t val;
    tCounter_remap_clientL_config_reg_out f;
} tClientL_Config_Reg_u;

// Removed: Single register pointers replaced with pair-indexed access

// ----------------------------------------------------------------------------------------

typedef struct {
    uint32_t id_0 : 3;
    uint32_t cnt_sel_0 : 5;
    uint32_t id_1 : 3;
    uint32_t cnt_sel_1 : 5;
    uint32_t id_2 : 3;
    uint32_t cnt_sel_2 : 5;
    uint32_t id_3 : 3;
    uint32_t cnt_sel_3 : 5;
} tCounter_remap_clientR_status_reg_in;

typedef union {
    uint32_t val;
    tCounter_remap_clientR_status_reg_in f;
} tClientR_status_Reg_u;

// Removed: Single register pointers replaced with pair-indexed access

// ----------------------------------------------------------------------------------------

typedef struct {
    uint32_t id_L : 3;
    uint32_t cnt_sel_L : 5;
    uint32_t valid : 4;
    uint32_t clientl_is_producer : 1;
    uint32_t clientr_group : 1;
    uint32_t distribute : 1;
    uint32_t err_inv_client_no : 1;
    uint32_t err_inv_client_id : 1;
    uint32_t err_inv_access : 1;
} tCounter_remap_clientL_status_reg_in;

typedef union {
    uint32_t val;
    tCounter_remap_clientL_status_reg_in f;
} tClientL_status_Reg_u;

// Removed: Single register pointers replaced with pair-indexed access

#endif  // __DM__REMAPPER_COMMON_HPP__
