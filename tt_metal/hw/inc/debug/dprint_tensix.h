// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "dprint.h"
#include "compute_kernel_api.h"
#include "ckernel_debug.h"
#include "tensix_types.h"

// Given a Tensix configuration register field name, print the contents of the register.
// Uses tt_metal/hw/inc/<family>/cfg_defines.h:
//   For config section "Registers for THREAD", use banks THREAD_0_CFG, THREAD_1_CFG, THREAD_2_CFG
//   For other config sections (ALU,PACK0), use banks HW_CFG_0, HW_CFG_1
#define READ_CFG_REG_FIELD(bank,reg_field_name)\
    (dbg_read_cfgreg(bank, reg_field_name##_ADDR32) & reg_field_name##_MASK) >> reg_field_name##_SHAMT

// Helper macros
#define READ_HW_CFG_0_REG_FIELD(reg_field_name)     READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::HW_CFG_0, reg_field_name)
#define READ_HW_CFG_1_REG_FIELD(reg_field_name)     READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::HW_CFG_1, reg_field_name)
#define READ_THREAD_0_CFG_REG_FIELD(reg_field_name) READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::THREAD_0_CFG, reg_field_name)
#define READ_THREAD_1_CFG_REG_FIELD(reg_field_name) READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::THREAD_1_CFG, reg_field_name)
#define READ_THREAD_2_CFG_REG_FIELD(reg_field_name) READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::THREAD_2_CFG, reg_field_name)

// Print the contents of tile with index tile_id within the destination register
template<bool print_by_face=false>
void dprint_tensix_dest_reg(int tile_id = 0) {
    dbg_halt();
    MATH({
        // Determine the format of the data in the destination register
        uint32_t data_format_reg_field_value = READ_HW_CFG_0_REG_FIELD(ALU_FORMAT_SPEC_REG2_Dstacc);

        #ifndef ARCH_GRAYSKULL
        //ALU_ACC_CTRL_Fp32 does not exist for GS
        if (READ_HW_CFG_0_REG_FIELD(ALU_ACC_CTRL_Fp32_enabled)) {
            data_format_reg_field_value = 0; // Override the data format to tt::DataFormat::Float32
        }
        #endif

        // Print the contents
        DPRINT << FIXED() << SETPRECISION(2);
        uint32_t rd_data[8+1]; // data + array_type
        DPRINT << "Tile ID = " << tile_id << ENDL();

        // print faces 0 & 1
        int face_r_dim = 16;
        for (int row = 0; row < face_r_dim; row++) {
            // face 0
            dbg_read_dest_acc_row(row + 64 * tile_id, rd_data);
            DPRINT << SETW(6) << " " << TYPED_U32_ARRAY(TypedU32_ARRAY_Format_Tensix_Config_Register_Data_Format_Type, data_format_reg_field_value, rd_data, 8);
        }
        if constexpr (print_by_face) {
            DPRINT << ENDL();
        }

        for (int row = 0; row < face_r_dim; row++) {
            // face 1
            dbg_read_dest_acc_row(row + face_r_dim + 64 * tile_id, rd_data);
            DPRINT << SETW(6) << " " << TYPED_U32_ARRAY(TypedU32_ARRAY_Format_Tensix_Config_Register_Data_Format_Type, data_format_reg_field_value, rd_data, 8) << ENDL();
        }

        for (int row = 0; row < face_r_dim; row++) {
            // face 2
            dbg_read_dest_acc_row(row + 2*face_r_dim + 64 * tile_id, rd_data);
            DPRINT << SETW(6) << " " << TYPED_U32_ARRAY(TypedU32_ARRAY_Format_Tensix_Config_Register_Data_Format_Type, data_format_reg_field_value, rd_data, 8);
        }

        if constexpr (print_by_face) {
            DPRINT << ENDL();
        }

        for (int row = 0; row < face_r_dim; row++) {
            // face 3
            dbg_read_dest_acc_row(row + 3*face_r_dim + 64 * tile_id, rd_data);
            DPRINT << SETW(6) << " " << TYPED_U32_ARRAY(TypedU32_ARRAY_Format_Tensix_Config_Register_Data_Format_Type, data_format_reg_field_value, rd_data, 8) << ENDL();
        }
    })
    dbg_unhalt();
}

// Print the contents of the specified configuration register field.
// Example:
//   dprint_cfg_reg_field(HW_CFG_0,ALU_FORMAT_SPEC_REG2_Dstacc);
#define dprint_cfg_reg_field(bank,reg_field_name) {\
    uint32_t field_val = READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::bank, reg_field_name);\
    DPRINT << #reg_field_name << " = " << field_val << ENDL();\
}

// Print the contents of the whole configuration register. The register is specified by
// the name of any field within it.
// Example:
//    dprint_cfg_reg(HW_CFG_0,ALU_FORMAT_SPEC_REG2_Dstacc);
#define dprint_cfg_reg(bank,reg_field_name) {\
    uint32_t reg_val = dbg_read_cfgreg(ckernel::dbg_cfgreg::bank, reg_field_name##_ADDR32);\
    DPRINT << #reg_field_name << " = " << HEX() << reg_val << ENDL();\
}
