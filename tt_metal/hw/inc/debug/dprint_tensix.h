// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
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

constexpr int PRECISION = 4;
constexpr int WIDTH = 8;

constexpr uint16_t NUM_FACES_PER_TILE = 4;
constexpr uint16_t NUM_ROWS_PER_FACE = 16;
constexpr uint16_t NUM_ROWS_PER_TILE = NUM_FACES_PER_TILE * NUM_ROWS_PER_FACE;

// Helper function to print array
inline void dprint_array_with_data_type(uint32_t data_format, uint32_t* data, uint32_t count) {
    DPRINT << TYPED_U32_ARRAY(TypedU32_ARRAY_Format_Tensix_Config_Register_Data_Format_Type, data_format, data, count)
           << ENDL();
}

// Dprints data format as string given an uint
inline void dprint_data_format(uint8_t data_format) {
    switch (data_format) {
        case (uint8_t)DataFormat::Float32: DPRINT << "Float32"; break;
        case (uint8_t)DataFormat::Float16: DPRINT << "Float16"; break;
        case (uint8_t)DataFormat::Bfp8: DPRINT << "Bfp8"; break;
        case (uint8_t)DataFormat::Bfp4: DPRINT << "Bfp4"; break;
        case (uint8_t)DataFormat::Bfp2: DPRINT << "Bfp2"; break;
        case (uint8_t)DataFormat::Float16_b: DPRINT << "Float16_b"; break;
        case (uint8_t)DataFormat::Bfp8_b: DPRINT << "Bfp8_b"; break;
        case (uint8_t)DataFormat::Bfp4_b: DPRINT << "Bfp4_b"; break;
        case (uint8_t)DataFormat::Bfp2_b: DPRINT << "Bfp2_b"; break;
        case (uint8_t)DataFormat::Lf8: DPRINT << "Lf8"; break;
        case (uint8_t)DataFormat::Int8: DPRINT << "Int8"; break;
        case (uint8_t)DataFormat::UInt8: DPRINT << "UInt8"; break;
        case (uint8_t)DataFormat::UInt16: DPRINT << "UInt16"; break;
        case (uint8_t)DataFormat::Int32: DPRINT << "Int32"; break;
        case (uint8_t)DataFormat::UInt32: DPRINT << "UInt32"; break;
        case (uint8_t)DataFormat::Tf32: DPRINT << "Tf32"; break;
        default: DPRINT << "INVALID DATA FORMAT"; break;
    }
}

// if flag DEST_ACCESS_CFG_remap_addrs is enabled
// destination register row identifiers are remmaped
// bits 5:3 are rotated 543 -> 354
inline uint16_t get_remapped_row_id(uint16_t row_id) {
    // bits 5:3 are rotating -> 543 -> 354
    return (row_id & 0xFFC7) |         // clear bits [5:3]
           ((row_id & 0x0008) << 2) |  // shifting bit 3 to position 5
           ((row_id & 0x0030) >> 1);   // shifting bits 5:4 to position 4:3
}

// if flag DEST_ACCESS_CFG_swizzle_32b is enabled dest address is has bits [3:2] shuffled
inline uint16_t get_swizzled_row_id(uint16_t row_id) {
    if (row_id & 0x10) {
        switch ((row_id & 0xC) >> 2) {
            case 0: return (row_id & 0xFFF3) | 0x8;
            case 1: return (row_id & 0xFFF3);
            case 2: return (row_id & 0xFFF3) | 0xC;
            case 3:
            default: return (row_id & 0xFFF3) | 0x4;
        }
    } else {
        return (row_id & 0xFFF3) | ((row_id & 0x4) << 1) | ((row_id & 0x8) >> 1);
    }
}

inline uint16_t get_logical_row_id(uint16_t tile_id, uint16_t face_id, uint16_t row_id) {
    return NUM_ROWS_PER_TILE * tile_id + NUM_ROWS_PER_FACE * face_id + row_id;
}

// Calculates dest row address based on logical row identifiers (tile_id, face_id, row_id)
// and dest configuration.
inline uint16_t get_dest_row_id(uint16_t logical_row_id, bool is_float32) {
    uint16_t row = logical_row_id;

#ifdef ARCH_BLACKHOLE
    if (READ_HW_CFG_0_REG_FIELD(DEST_ACCESS_CFG_remap_addrs) == 1) {
        row = get_remapped_row_id(row);
    }
#endif

    if (is_float32) {
#ifdef ARCH_BLACKHOLE
        if (READ_HW_CFG_0_REG_FIELD(DEST_ACCESS_CFG_swizzle_32b) == 1) {
            row = get_swizzled_row_id(row);
        }
#endif
        // 0-7  dest rows for Float16
        // 8-15 dest rows for Mantissa
        // need to shift row index starting from bit 3
        row = ((row & 0xFFF8) << 1) | (row & 0x7);
    }

    return row;
}

inline uint16_t lo_word(uint32_t dword) { return dword & 0xFFFF; }
inline uint16_t hi_word(uint32_t dword) { return lo_word(dword >> 16); }

// Float16 = [1-bit sign, 7-bit mantissa, 8-bit exponent]
// Mantissa16 = [16-bit mantissa]
// Float32 = [1-bit sign, 8-bit exponent, 23-bit mantissa(7-bit + 16-bit)]
inline uint32_t reconstruct_float32(uint32_t float16, uint32_t mantissa16) {
    uint32_t sign = (float16 & 0x00008000) << 16;
    uint32_t exponent = (float16 & 0x000000FF) << 23;
    uint32_t mantissa = ((float16 & 0x00007F00) << 8) | mantissa16;

    return sign | exponent | mantissa;
}

// Helper function that prints one row from dest when dest is configured for storing float32 values.
// This function should be used only from dprint_tensix_dest_reg.
// Float32 in dest = [Float16, Mantissa16]
// dest_row -> [[Float16_1,Float16_0],...[Float16_15, Float16_14]]
// dest_row + 8 -> [[Mantissa16_1,Mantissa16_0],...[Mantissa16_15, Mantissa16_14]]
inline void dprint_tensix_dest_reg_row_float32(uint16_t row) {
    constexpr int ARRAY_LEN = 16;
    uint32_t rd_data[ARRAY_LEN + 1];  // data + array type

#ifdef ARCH_BLACKHOLE
    // On Blackhole, use direct dest access - Float32 values are already in correct format
    const uint32_t* addr = reinterpret_cast<const uint32_t*>(0xFFBD8000);
    for (int i = 0; i < ARRAY_LEN; ++i) {
        rd_data[i] = addr[i + (row << 4)];
    }
#else
    // On other architectures, need to reconstruct Float32 from Float16 and Mantissa16
    row = get_dest_row_id(row, true);
    uint32_t rd_data_temp[ARRAY_LEN];
    dbg_read_dest_acc_row(row, rd_data_temp);
    dbg_read_dest_acc_row(row + 8, rd_data_temp + 8);

    for (int i = 0; i < 8; ++i) {
        rd_data[2 * i] = reconstruct_float32(lo_word(rd_data_temp[i]), lo_word(rd_data_temp[i + 8]));
        rd_data[2 * i + 1] = reconstruct_float32(hi_word(rd_data_temp[i]), hi_word(rd_data_temp[i + 8]));
    }
#endif

    dprint_array_with_data_type((uint32_t)DataFormat::Float32, rd_data, ARRAY_LEN);
}

// Helper function that prints one row from dest when dest is configured for storing float16 values.
// This function should be used only from dprint_tensix_dest_reg.
inline void dprint_tensix_dest_reg_row_float16(uint32_t data_format, uint16_t row) {
    constexpr int ARRAY_LEN = 8;
    uint32_t rd_data[ARRAY_LEN + 1];  // data + array type
    row = get_dest_row_id(row, false);
    dbg_read_dest_acc_row(row, rd_data);
    dprint_array_with_data_type(data_format, rd_data, 8);
}

inline void dprint_tensix_dest_reg_row_int32(uint16_t row) {
#ifdef ARCH_BLACKHOLE
    constexpr int ARRAY_LEN = 16;
    uint32_t rd_data[ARRAY_LEN + 1];  // data + array type
    const uint32_t* addr = reinterpret_cast<const uint32_t*>(0xFFBD8000);
    for (int i = 0; i < ARRAY_LEN; ++i) {
        rd_data[i] = addr[i + (row << 4)];
    }
    dprint_array_with_data_type((uint32_t)DataFormat::Int32, rd_data, ARRAY_LEN);
#else
    DPRINT << "Int32 format not supported on this architecture" << ENDL();
#endif
}

// Print the contents of tile with index tile_id within the destination register
template <bool print_by_face = false>
void dprint_tensix_dest_reg(int tile_id = 0) {
    dbg_halt();
    MATH({
        // Determine the format of the data in the destination register
        uint32_t data_format_reg_field_value = READ_HW_CFG_0_REG_FIELD(ALU_FORMAT_SPEC_REG2_Dstacc);

        if (READ_HW_CFG_0_REG_FIELD(ALU_ACC_CTRL_Fp32_enabled)) {
            data_format_reg_field_value = (uint32_t)DataFormat::Float32;
        }

        // Print the contents
        DPRINT << FIXED() << SETW(WIDTH) << SETPRECISION(PRECISION);
        DPRINT << "Tile ID = " << tile_id << ENDL();

        uint32_t row = tile_id * NUM_ROWS_PER_TILE;
        for (int face_id = 0; face_id < NUM_FACES_PER_TILE; ++face_id) {
            for (int row_id = 0; row_id < NUM_ROWS_PER_FACE; ++row_id) {
                switch (data_format_reg_field_value) {
                    case (uint32_t)DataFormat::Float32:
                        dprint_tensix_dest_reg_row_float32(row);
                        break;
                    case (uint32_t)DataFormat::Int32:
                        dprint_tensix_dest_reg_row_int32(row);
                        break;
                    case (uint32_t)DataFormat::Float16_b:
                        dprint_tensix_dest_reg_row_float16(data_format_reg_field_value, row);
                        break;
                    default:
                        DPRINT << "Unsupported data format: " << data_format_reg_field_value << ENDL();
                        break;
                }
                row++;
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

// Print the content of the register field given the value in the register.
#define DPRINT_TENSIX_CONFIG_FIELD(reg_val, reg_field_name, name, printDec)                 \
    {                                                                                       \
        uint32_t field_value = (reg_val & reg_field_name##_MASK) >> reg_field_name##_SHAMT; \
        DPRINT << name << " = ";                                                            \
        if (printDec) {                                                                     \
            DPRINT << DEC();                                                                \
        } else {                                                                            \
            DPRINT << "0x" << HEX();                                                        \
        }                                                                                   \
        DPRINT << field_value << "; ";                                                      \
    }

inline void dprint_tensix_struct_field(
    uint32_t word, uint32_t mask, uint8_t shamt, const char* name, bool printDec = false) {
    DPRINT << name << ": ";
    if (printDec) {
        DPRINT << DEC();
    } else {
        DPRINT << "0x" << HEX();
    }
    DPRINT << ((word & mask) >> shamt) << ENDL();
}
