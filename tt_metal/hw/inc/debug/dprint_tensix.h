// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_debug.h"
#include "compute_kernel_api.h"
#include "dprint.h"
#include "tensix_types.h"

// Given a Tensix configuration register field name, print the contents of the register.
// Uses tt_metal/hw/inc/<family>/cfg_defines.h:
//   For config section "Registers for THREAD", use banks THREAD_0_CFG, THREAD_1_CFG, THREAD_2_CFG
//   For other config sections (ALU,PACK0), use banks HW_CFG_0, HW_CFG_1
#define READ_CFG_REG_FIELD(bank, reg_field_name) \
    (dbg_read_cfgreg(bank, reg_field_name##_ADDR32) & reg_field_name##_MASK) >> reg_field_name##_SHAMT

// Helper macros
#define READ_HW_CFG_0_REG_FIELD(reg_field_name) READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::HW_CFG_0, reg_field_name)
#define READ_HW_CFG_1_REG_FIELD(reg_field_name) READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::HW_CFG_1, reg_field_name)
#define READ_THREAD_0_CFG_REG_FIELD(reg_field_name) \
    READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::THREAD_0_CFG, reg_field_name)
#define READ_THREAD_1_CFG_REG_FIELD(reg_field_name) \
    READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::THREAD_1_CFG, reg_field_name)
#define READ_THREAD_2_CFG_REG_FIELD(reg_field_name) \
    READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::THREAD_2_CFG, reg_field_name)

// Helper function to print array
template<int WIDTH>
void dprint_array_with_data_type(uint32_t data_format, uint32_t* data, uint32_t count) {
    DPRINT << SETW(WIDTH) << " "
           << TYPED_U32_ARRAY(
                  TypedU32_ARRAY_Format_Tensix_Config_Register_Data_Format_Type, data_format, data, count)
           << ENDL();
}

inline uint16_t lo_word(uint32_t dword) { return dword & 0xFFFF;}
inline uint16_t hi_word(uint32_t dword) { return lo_word(dword >> 16);}

// Float16 = [1-bit sign, 7-bit mantissa, 8-bit exponent]
// Mantissa16 = [16-bit mantissa]
// Float32 = [1-bit sign, 8-bit exponent, 23-bit mantissa(7-bit + 16-bit)]
inline uint32_t reconstruct_float32(uint32_t float16, uint32_t mantissa16) {

    uint32_t sign = (float16 & 0x00008000) << 16;
    uint32_t exponent = (float16 & 0x000000FF) << 23;
    uint32_t mantissa = ((float16 & 0x00007F00) << 8) + mantissa16;

    return sign + exponent + mantissa;
}

// Helper function that prints one row from dest when dest is configured for storing float32 values.
// This function should be used only from dprint_tensix_dest_reg.
// Float32 in dest = [Float16, Mantissa16]
// dest_row -> [[Float16_1,Float16_0],...[Float16_15, Float16_14]]
// dest_row + 8 -> [[Mantissa16_1,Mantissa16_0],...[Mantissa16_15, Mantissa16_14]]
template<int WIDTH>
void dprint_tensix_dest_reg_row_float32(int tile_id, int face_id, int row_id) {
    constexpr int ARRAY_LEN = 16;
    uint32_t rd_data_temp[ARRAY_LEN];
    uint32_t rd_data[ARRAY_LEN + 1]; // data + array type

    int logical_row_index = tile_id * 64 + face_id * 16 + row_id;
    int physical_row_index = ((logical_row_index >> 3) << 4) + (logical_row_index & 0x7);

    // read dest row that has first half of 16 floats
    dbg_read_dest_acc_row(physical_row_index, rd_data_temp);
    // read dest row that has second half of 16 floats
    dbg_read_dest_acc_row(physical_row_index + 8, rd_data_temp + 8);

    for (int i = 0; i < 8; ++i) {
        rd_data[2 * i] = reconstruct_float32(lo_word(rd_data_temp[i]), lo_word(rd_data_temp[i + 8]));
        rd_data[2 * i + 1] = reconstruct_float32(hi_word(rd_data_temp[i]), hi_word(rd_data_temp[i + 8]));
    }

    dprint_array_with_data_type<WIDTH>((uint32_t)DataFormat::Float32, rd_data, ARRAY_LEN);
}

// Helper function that prints one row from dest when dest is configured for storing float16 values.
// This function should be used only from dprint_tensix_dest_reg.
template<int WIDTH>
void dprint_tensix_dest_reg_row_float16(uint32_t data_format, int tile_id, int face_id, int row_id) {
    int row = tile_id * 64 + face_id * 16 + row_id;
    constexpr int ARRAY_LEN = 8;
    uint32_t rd_data[ARRAY_LEN + 1]; // data + array type
    dbg_read_dest_acc_row(row, rd_data);
    dprint_array_with_data_type<WIDTH>(data_format, rd_data, 8);
}

// Print the contents of tile with index tile_id within the destination register
template <bool print_by_face = false, int PRECISION = 4, int WIDTH = 8>
void dprint_tensix_dest_reg(int tile_id = 0) {
    dbg_halt();
    MATH({
        // Determine the format of the data in the destination register
        uint32_t data_format_reg_field_value = READ_HW_CFG_0_REG_FIELD(ALU_FORMAT_SPEC_REG2_Dstacc);

#ifndef ARCH_GRAYSKULL
        // ALU_ACC_CTRL_Fp32 does not exist for GS
        if (READ_HW_CFG_0_REG_FIELD(ALU_ACC_CTRL_Fp32_enabled)) {
            data_format_reg_field_value =
                (uint32_t)DataFormat::Float32;  // Override the data format to tt::DataFormat::Float32
        }
#endif

        // Print the contents
        DPRINT << FIXED() << SETPRECISION(PRECISION);
        DPRINT << "Tile ID = " << tile_id << ENDL();

        // print faces 0
        for (int face_id = 0; face_id < 4; ++face_id) {
            for (int row_id = 0; row_id < 16; ++row_id) {
                if (data_format_reg_field_value == (uint32_t)DataFormat::Float32) {
                    dprint_tensix_dest_reg_row_float32<WIDTH>(tile_id, face_id, row_id);
                } else {
                    dprint_tensix_dest_reg_row_float16<WIDTH>(data_format_reg_field_value, tile_id, face_id, row_id);
                }
            }
            if constexpr (print_by_face) {
                DPRINT << ENDL();
            }
        }
    })
    dbg_unhalt();
}

// Print the contents of the specified configuration register field.
// Example:
//   dprint_cfg_reg_field(HW_CFG_0,ALU_FORMAT_SPEC_REG2_Dstacc);
#define dprint_cfg_reg_field(bank, reg_field_name)                                          \
    {                                                                                       \
        uint32_t field_val = READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::bank, reg_field_name); \
        DPRINT << #reg_field_name << " = " << field_val << ENDL();                          \
    }

// Print the contents of the whole configuration register. The register is specified by
// the name of any field within it.
// Example:
//    dprint_cfg_reg(HW_CFG_0,ALU_FORMAT_SPEC_REG2_Dstacc);
#define dprint_cfg_reg(bank, reg_field_name)                                                    \
    {                                                                                           \
        uint32_t reg_val = dbg_read_cfgreg(ckernel::dbg_cfgreg::bank, reg_field_name##_ADDR32); \
        DPRINT << #reg_field_name << " = " << HEX() << reg_val << ENDL();                       \
    }
