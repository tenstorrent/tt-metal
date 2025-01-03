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

// Calculates dest row address based on logical row identifiers (tile_id, face_id, row_id)
// and dest configuration.
inline uint16_t get_dest_row_id(
    uint16_t tile_id, uint16_t face_id, uint16_t row_id, bool is_float32, bool is_remap, bool is_swizzle) {
    uint16_t row = NUM_ROWS_PER_TILE * tile_id + NUM_ROWS_PER_FACE * face_id + row_id;

    if (is_remap) {
        row = get_remapped_row_id(row);
    }

    if (is_float32) {
        if (is_swizzle) {
            row = get_swizzled_row_id(row);
        }
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
    uint32_t rd_data_temp[ARRAY_LEN];
    uint32_t rd_data[ARRAY_LEN + 1];  // data + array type

    // read two rows [[Float16], [Mantissa]]
    dbg_read_dest_acc_row(row, rd_data_temp);
    dbg_read_dest_acc_row(row + 8, rd_data_temp + 8);

    for (int i = 0; i < 8; ++i) {
        rd_data[2 * i] = reconstruct_float32(lo_word(rd_data_temp[i]), lo_word(rd_data_temp[i + 8]));
        rd_data[2 * i + 1] = reconstruct_float32(hi_word(rd_data_temp[i]), hi_word(rd_data_temp[i + 8]));
    }

    dprint_array_with_data_type((uint32_t)DataFormat::Float32, rd_data, ARRAY_LEN);
}

// Helper function that prints one row from dest when dest is configured for storing float16 values.
// This function should be used only from dprint_tensix_dest_reg.
inline void dprint_tensix_dest_reg_row_float16(uint32_t data_format, uint16_t row) {
    constexpr int ARRAY_LEN = 8;
    uint32_t rd_data[ARRAY_LEN + 1];  // data + array type
    dbg_read_dest_acc_row(row, rd_data);
    dprint_array_with_data_type(data_format, rd_data, 8);
}

// Print the contents of tile with index tile_id within the destination register
template <bool print_by_face = false>
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

        bool is_float32 = data_format_reg_field_value == (uint32_t)DataFormat::Float32;
        bool is_swizzled = false;
        bool is_remapped = false;

#ifdef ARCH_BLACKHOLE
        is_remapped = READ_HW_CFG_0_REG_FIELD(DEST_ACCESS_CFG_remap_addrs) == 1;
        is_swizzled = READ_HW_CFG_0_REG_FIELD(DEST_ACCESS_CFG_swizzle_32b) == 1;
#endif
        // Print the contents
        DPRINT << FIXED() << SETW(WIDTH) << SETPRECISION(PRECISION);
        DPRINT << "Tile ID = " << tile_id << ENDL();

        for (int face_id = 0; face_id < NUM_FACES_PER_TILE; ++face_id) {
            for (int row_id = 0; row_id < NUM_ROWS_PER_FACE; ++row_id) {
                uint16_t row = get_dest_row_id(tile_id, face_id, row_id, is_float32, is_remapped, is_swizzled);
                if (is_float32) {
                    dprint_tensix_dest_reg_row_float32(row);
                } else {
                    dprint_tensix_dest_reg_row_float16(data_format_reg_field_value, row);
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

// Print the content of the register field given the value in the register.
#define DPRINT_TENSIX_CONFIG_FIELD(reg_val, reg_field_name, name, printDec)                     \
    {                                                                                           \
        uint32_t field_value = (reg_val & reg_field_name##_MASK) >> reg_field_name##_SHAMT;     \
        DPRINT << name << " = ";                                                                \
        if (printDec) DPRINT << DEC();                                                          \
        else DPRINT << HEX();                                                                   \
        DPRINT << field_value << "; ";                                                          \
    }

inline void dprint_tensix_struct_field(uint32_t word, uint32_t mask, uint8_t shamt, const char* name, bool printDec = false)
{
    DPRINT << name << ": ";
    if (printDec) DPRINT << DEC();
    else DPRINT << HEX();
    DPRINT << ((word & mask) >> shamt) << "; ";
}

// HARDWARE SPECIFIC FUNCTIONS

// UNPACKER CONFIG REGISTERS

// GRAYSKULL
inline void dprint_tensix_unpack_tile_descriptor_grayskull() {
    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

    //word 0
    uint32_t word = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32];
    dprint_tensix_struct_field(word, 0xf, 0, "in_data_format");
    dprint_tensix_struct_field(word, 0x10, 4, "uncompressed");
    dprint_tensix_struct_field(word, 0xe0, 5, "reserved_0");
    dprint_tensix_struct_field(word, 0xf00, 8, "blobs_per_xy_plane", true);
    dprint_tensix_struct_field(word, 0xf000, 12, "reserved_1");
    dprint_tensix_struct_field(word, 0xffff0000, 16, "x_dim", true);

    //word 1
    word = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1];
    dprint_tensix_struct_field(word, 0xffff, 0, "y_dim", true);
    dprint_tensix_struct_field(word, 0xffff0000, 16, "z_dim", true);

    //word 2
    word = cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32];
    dprint_tensix_struct_field(word, 0xffff, 0, "w_dim", true);

    // blobs_y_start is in 2 words (word2 and word3)
    uint32_t tmp_word = word;

    // word3
    word = cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + 1];
    DPRINT << "blobs_y_start: " << DEC() << (((word & 0xffff) << 16) | ((tmp_word & 0xffff0000) >> 16)) << "; "; //blobs_y_start

    dprint_tensix_struct_field(word, 0xff0000, 16, "digest_type");
    dprint_tensix_struct_field(word, 0xff000000, 24, "digest_size", true);

    DPRINT << ENDL();
}

inline void dprint_tensix_unpack_config_grayskull() {
    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

    //word 0
    uint32_t word = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32];
    dprint_tensix_struct_field(word, 0xf, 0, "out_format");
    dprint_tensix_struct_field(word, 0x30, 4, "throttle_mode");
    dprint_tensix_struct_field(word, 0xc0, 6, "cntx_cnt");
    dprint_tensix_struct_field(word, 0x100, 8, "halo_mode");
    dprint_tensix_struct_field(word, 0x200, 9, "tile_mode");
    dprint_tensix_struct_field(word, 0x400, 10, "force_shrd_exp");
    dprint_tensix_struct_field(word, 0x800, 11, "res_0");
    dprint_tensix_struct_field(word, 0x7000, 12, "upsmpl_rate", true);
    dprint_tensix_struct_field(word, 0x8000, 15, "upsmpl_and_intrlv");
    dprint_tensix_struct_field(word, 0xffff0000, 16, "shamt", true);

    //word 2
    word = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + 1];
    dprint_tensix_struct_field(word, 0xf, 0, "uncmpr_cntx0_3");
    dprint_tensix_struct_field(word, 0xfff0, 4, "res_1");
    dprint_tensix_struct_field(word, 0xf0000, 16, "uncmpr_cntx4_7");
    dprint_tensix_struct_field(word, 0xfff00000, 20, "res_2");

    //word 2
    word = cfg[THCON_SEC1_REG2_Out_data_format_ADDR32];
    dprint_tensix_struct_field(word, 0xffff, 0, "limit_addr");
    dprint_tensix_struct_field(word, 0xffff0000, 16, "fifo_sz", true);

    DPRINT << ENDL();
}

// WORMHOLE/BLACKHOLE
inline void dprint_tensix_unpack_tile_descriptor_wormhole_or_blackhole() {
    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

    //word 0
    uint32_t word = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32];
    dprint_tensix_struct_field(word, 0xf, 0, "in_data_format");
    dprint_tensix_struct_field(word, 0x10, 4, "uncompressed");
    dprint_tensix_struct_field(word, 0xe0, 5, "reserved_0");
    dprint_tensix_struct_field(word, 0xf00, 8, "blobs_per_xy_plane", true);
    dprint_tensix_struct_field(word, 0xf000, 12, "reserved_1");
    dprint_tensix_struct_field(word, 0xffff0000, 16, "x_dim", true);

    //word 1
    word = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1];
    dprint_tensix_struct_field(word, 0xffff, 0, "y_dim", true);
    dprint_tensix_struct_field(word, 0xffff0000, 16, "z_dim", true);

    //word 2
    word = cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32];
    dprint_tensix_struct_field(word, 0xffff, 0, "w_dim", true);

    uint32_t prev_word = word;

    // word3
    word = cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + 1];
    DPRINT << "blobs_y_start: " << HEX() << (((word & 0xffff) << 16) | ((prev_word & 0xffff0000) >> 16)) << "; ";
    dprint_tensix_struct_field(word, 0xff0000, 16, "digest_type");
    dprint_tensix_struct_field(word, 0xff000000, 24, "digest_size", true);

    DPRINT << ENDL();
}

inline void dprint_tensix_unpack_config_wormhole_or_blackhole() {
    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

    //word 0
    uint32_t word = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32];
    dprint_tensix_struct_field(word, 0xf, 0, "out_frmt");
    dprint_tensix_struct_field(word, 0x30, 4, "throt_md");
    dprint_tensix_struct_field(word, 0xc0, 6, "cx_cnt");
    dprint_tensix_struct_field(word, 0x100, 8, "halo_md");
    dprint_tensix_struct_field(word, 0x200, 9, "tile_md");
    dprint_tensix_struct_field(word, 0x400, 10, "unp_srreg_stupd");
    dprint_tensix_struct_field(word, 0x800, 11, "unpis");
    dprint_tensix_struct_field(word, 0x3000, 12, "ups_rate");
    dprint_tensix_struct_field(word, 0x4000, 14, "r1");
    dprint_tensix_struct_field(word, 0x8000, 15, "ups_and_int");
    dprint_tensix_struct_field(word, 0xffff0000, 16, "shamt", true);

    //word 2
    word = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + 1];
    dprint_tensix_struct_field(word, 0xf, 0, "uncmpr_cx0_3");
    dprint_tensix_struct_field(word, 0xf0, 4, "unpis_cx0_3");
    dprint_tensix_struct_field(word, 0x100, 8, "force_shrd_exp");
    dprint_tensix_struct_field(word, 0xfe00, 9, "r2");
    dprint_tensix_struct_field(word, 0xf0000, 16, "uncmpr_cx4_7");
    dprint_tensix_struct_field(word, 0xf00000, 20, "unpis_cx4_7");
    dprint_tensix_struct_field(word, 0xff000000, 24, "r3");

    //word 2
    word = cfg[THCON_SEC1_REG2_Out_data_format_ADDR32];
    dprint_tensix_struct_field(word, 0x1ffff, 0, "lmt_addr");
    dprint_tensix_struct_field(word, 0xfffe0000, 17, "r4");

    //word 3
    word = cfg[THCON_SEC1_REG2_Out_data_format_ADDR32];
    dprint_tensix_struct_field(word, 0x1ffff, 0, "fifo_sz", true);
    dprint_tensix_struct_field(word, 0xfffe0000, 17, "r5");

    DPRINT << ENDL();
}

// PACKER CONFIG REGISTERS

inline void dprint_tensix_pack_config_grayskull(uint32_t reg_addr, const volatile uint tt_reg_ptr* cfg ) {
    // word 0
    uint32_t word = cfg[reg_addr];
    dprint_tensix_struct_field(word, 0xffff, 0, "row_ptr_sec_sz", true);
    dprint_tensix_struct_field(word, 0xffff0000, 16, "exp_sec_sz", true);

    // word 1
    word = cfg[reg_addr + 1];
    dprint_tensix_struct_field(word, 0xffffffff, 0, "l1_dst_adr");

    // word 2
    word = cfg[reg_addr + 2];
    dprint_tensix_struct_field(word, 0x1, 0, "uncmpr");
    dprint_tensix_struct_field(word, 0x2, 1, "add_l1_dst_adr_ofs");
    dprint_tensix_struct_field(word, 0xc, 2, "r0");
    dprint_tensix_struct_field(word, 0xf0, 4, "out_frmt");
    dprint_tensix_struct_field(word, 0xf00, 8, "in_frmt");
    dprint_tensix_struct_field(word, 0xf000, 12, "r1");
    dprint_tensix_struct_field(word, 0x10000, 16, "src_if_sel");
    dprint_tensix_struct_field(word, 0xfe0000, 17, "pck_per_xy_pl");
    dprint_tensix_struct_field(word, 0xff000000, 24, "l1_src_adr");

    // word 3
    word = cfg[reg_addr + 3];
    dprint_tensix_struct_field(word, 0xffff, 0, "dsmpl_mask");
    dprint_tensix_struct_field(word, 0x70000, 16, "dsmpl_shcnt");
    dprint_tensix_struct_field(word, 0x80000, 19, "r_md");
    dprint_tensix_struct_field(word, 0x100000, 20, "exp_th_en");
    dprint_tensix_struct_field(word, 0xe00000, 21, "r2");
    dprint_tensix_struct_field(word, 0xff000000, 24, "exp_th", true);

    DPRINT << ENDL();
}

inline void dprint_tensix_pack_config_wormhole(uint32_t reg_addr, const volatile uint tt_reg_ptr* cfg ){
    // word 0
    uint32_t word = cfg[reg_addr];
    dprint_tensix_struct_field(word, 0xffff, 0, "r_p_s_sz", true);
    dprint_tensix_struct_field(word, 0xffff0000, 16, "e_s_sz", true);

    // word 1
    word = cfg[reg_addr + 1];
    dprint_tensix_struct_field(word, 0xffffffff, 0, "l1_dst_adr");

    // word 2
    word = cfg[reg_addr + 2];
    dprint_tensix_struct_field(word, 0x1, 0, "uncmpr");
    dprint_tensix_struct_field(word, 0x2, 1, "a_l1_d_a_o");
    dprint_tensix_struct_field(word, 0xc, 2, "r0");
    dprint_tensix_struct_field(word, 0xf0, 4, "out_frmt");
    dprint_tensix_struct_field(word, 0xf00, 8, "in_frmt");
    dprint_tensix_struct_field(word, 0xf000, 12, "r1");
    dprint_tensix_struct_field(word, 0x10000, 16, "src_if_sel");
    dprint_tensix_struct_field(word, 0xfe0000, 17, "pck_per_xy_pl", true);
    dprint_tensix_struct_field(word, 0xff000000, 24, "l1_src_adr");

    // word 3
    word = cfg[reg_addr + 3];
    dprint_tensix_struct_field(word, 0xffff, 0, "dsmpl_mask");
    dprint_tensix_struct_field(word, 0x70000, 16, "dsmpl_shcnt");
    dprint_tensix_struct_field(word, 0x80000, 19, "r_md");
    dprint_tensix_struct_field(word, 0x100000, 20, "exp_th_en");
    dprint_tensix_struct_field(word, 0x600000, 21, "pck_l1_ac_dis_pck_0_flg");
    dprint_tensix_struct_field(word, 0x800000, 23, "r2");
    dprint_tensix_struct_field(word, 0xff000000, 24, "exp_th", true);

    DPRINT << ENDL();
}

inline void dprint_tensix_pack_config_blackhole(uint32_t reg_addr, const volatile uint tt_reg_ptr* cfg) {
    // word 0
    uint32_t word = cfg[reg_addr];
    dprint_tensix_struct_field(word, 0xffff, 0, "row_ptr_sec_sz", true);
    dprint_tensix_struct_field(word, 0xffff0000, 16, "exp_sec_sz", true);

    // word 1
    word = cfg[reg_addr + 1];
    dprint_tensix_struct_field(word, 0xffffffff, 0, "l1_dst_adr");

    // word2
    word = cfg[reg_addr + 2];
    dprint_tensix_struct_field(word, 0x1, 0, "uncmps");
    dprint_tensix_struct_field(word, 0x2, 1, "aldao");
    dprint_tensix_struct_field(word, 0x4, 2, "dis_pck_0_fl");
    dprint_tensix_struct_field(word, 0x8, 3, "r0");
    dprint_tensix_struct_field(word, 0xf0, 4, "out_frmt");
    dprint_tensix_struct_field(word, 0xf00, 8, "in_frmt");
    dprint_tensix_struct_field(word, 0x1000, 12, "dsea");
    dprint_tensix_struct_field(word, 0x2000, 13, "alpis");
    dprint_tensix_struct_field(word, 0x4000, 14, "en_out_fifo");
    dprint_tensix_struct_field(word, 0x8000, 15, "sub_l1_til_h_sz", true);
    dprint_tensix_struct_field(word, 0x10000, 16, "src_if_sel");
    dprint_tensix_struct_field(word, 0x1e0000, 17, "pck_st_intf_pos");
    dprint_tensix_struct_field(word, 0x200000, 21, "apd0co");
    dprint_tensix_struct_field(word, 0x400000, 22, "add_til_h_sz", true);
    dprint_tensix_struct_field(word, 0x800000, 23, "pdypso");
    dprint_tensix_struct_field(word, 0xff000000, 24, "l1_src_adr");

    DPRINT << ENDL();
}

// COMBINED FUNCTIONS

// Print content of the register field by field. Issue: No ENDL.
inline void dprint_tensix_alu_config() {
// Only wormhole and blackhole have this register
#ifdef ARCH_GRAYSKULL
    DPRINT << "GRAYSKULL HAS NO ALU CONFIG REGISTERS" << ENDL();
    return;
#endif

    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr * cfg = get_cfg_pointer();
    uint32_t reg_val = cfg[ALU_ROUNDING_MODE_Fpu_srnd_en_ADDR32];

    DPRINT << "RND_MODE: ";
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, ALU_ROUNDING_MODE_Fpu_srnd_en, "Fpu_srnd_en", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, ALU_ROUNDING_MODE_Gasket_srnd_en, "Gasket_srnd_en", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, ALU_ROUNDING_MODE_Packer_srnd_en, "Packer_srnd_en", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, ALU_ROUNDING_MODE_Padding, "Padding", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, ALU_ROUNDING_MODE_GS_LF, "GS_LF", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, ALU_ROUNDING_MODE_Bfp8_HF, "Bfp8_HF", false);
    DPRINT << "FORMAT: ";
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, ALU_FORMAT_SPEC_REG0_SrcAUnsigned, "SrcAUnsigned", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, ALU_FORMAT_SPEC_REG0_SrcBUnsigned, "SrcBUnsigned", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, ALU_FORMAT_SPEC_REG0_SrcA, "SrcA", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, ALU_FORMAT_SPEC_REG1_SrcB, "SrcB", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, ALU_FORMAT_SPEC_REG2_Dstacc, "Dstacc", false);
    DPRINT << "ACC_CTRL: ";
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, ALU_ACC_CTRL_Fp32_enabled, "Fp32_enabled", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, ALU_ACC_CTRL_SFPU_Fp32_enabled, "SFPU_Fp32_enabled", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, ALU_ACC_CTRL_INT8_math_enabled, "INT8_math_enabled", false);
    DPRINT << ENDL();
}

inline void dprint_tensix_unpack_tile_descriptor(){
#ifdef ARCH_GRAYSKULL
    dprint_tensix_unpack_tile_descriptor_grayskull();
#else
    dprint_tensix_unpack_tile_descriptor_wormhole_or_blackhole();
#endif
}

inline void dprint_tensix_unpack_config(){
#ifdef ARCH_GRAYSKULL
    dprint_tensix_unpack_config_grayskull();
#else
    dprint_tensix_unpack_config_wormhole_or_blackhole();
#endif
}

// Choose what register you want printed with reg_id (1-4), ALL not implemented due to dprint buffer size restriction
inline void dprint_tensix_pack_config(uint reg_id = 1){
    
    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    if (reg_id) {
        uint32_t reg_addr = 0;
        switch (reg_id) {
            case 1:
                reg_addr = THCON_SEC0_REG1_Row_start_section_size_ADDR32;
                break;
            case 2:
                reg_addr = THCON_SEC0_REG8_Row_start_section_size_ADDR32;
                break;
            case 3:
                reg_addr = THCON_SEC1_REG1_Row_start_section_size_ADDR32;
                break;
            case 4:
                reg_addr = THCON_SEC1_REG8_Row_start_section_size_ADDR32;
                break;
            default:
                DPRINT << "Aborting! Invalid register id (valid ids are between 1 and 4)" << ENDL();
                return;
        }

    DPRINT << "REG_ID: " << reg_id << " ";

    #ifdef ARCH_GRAYSKULL
        dprint_tensix_pack_config_grayskull(reg_addr, cfg);
    #elif  ARCH_WORMHOLE
        dprint_tensix_pack_config_wormhole(reg_addr, cfg);
    #else
        dprint_tensix_pack_config_blackhole(reg_addr, cfg);
    #endif
    }
}

inline void dprint_tensix_pack_relu_config() {
// Only wormhole and blackhole have this register
#ifdef ARCH_GRAYSKULL
    DPRINT << "GRAYSKULL HAS NO RELU CONFIG REGISTERS" << ENDL();
    return;
#endif

    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();
    uint32_t reg_val = cfg[ALU_ACC_CTRL_Zero_Flag_disabled_src_ADDR32];

    DPRINT << "ALU_ACC_CTRL: ";
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, ALU_ACC_CTRL_Zero_Flag_disabled_src, "zero_flag_disabled_src", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, ALU_ACC_CTRL_Zero_Flag_disabled_dst, "zero_flag_disabled_dst", false);
    DPRINT << "STACC_RELU: ";
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, STACC_RELU_ApplyRelu, "apply_relu", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, STACC_RELU_ReluThreshold, "relu_threshold", true);
    DPRINT << "DISABLE_RISC_BP_Disable: ";
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, DISABLE_RISC_BP_Disable_main, "main", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, DISABLE_RISC_BP_Disable_trisc, "trisc", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, DISABLE_RISC_BP_Disable_ncrisc, "ncrisc", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, DISABLE_RISC_BP_Disable_bmp_clear_main, "bmp_clear_main", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, DISABLE_RISC_BP_Disable_bmp_clear_trisc, "bmp_clear_trisc", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, DISABLE_RISC_BP_Disable_bmp_clear_ncrisc, "bmp_clear_ncrisc", false);
    DPRINT << ENDL();
}

// Printing dest control bits
inline void dprint_tensix_dest_rd_ctrl() {
#ifdef ARCH_GRAYSKULL
    DPRINT << "GRAYSKULL HAS NO DEST RD CTRL REGISTERS" << ENDL();
    return;
#endif

    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();
    uint32_t reg_val = cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32];

    DPRINT << "PCK_DEST_RD_CTRL: ";
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, PCK_DEST_RD_CTRL_Read_32b_data, "read_32b_data", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, PCK_DEST_RD_CTRL_Read_unsigned, "read_unsigned", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, PCK_DEST_RD_CTRL_Read_int8, "read_int8", false);
    DPRINT_TENSIX_CONFIG_FIELD(reg_val, PCK_DEST_RD_CTRL_Round_10b_mant, "round_10b_mant", false);

    // Can't write to reserved? -> always prints 0
    // reg_val gets only last 4 bits
    //dprint_tensix_struct_field(reg_val, 0xfffffff0, 4, "reserved");

    DPRINT << ENDL();
}

// Printing packer edge offset
inline void dprint_tensix_pck_edge_offset_helper(uint reg_id, const volatile uint tt_reg_ptr* cfg) {

    uint32_t reg_addr = 0;
    switch (reg_id) {
        case 1:
            reg_addr = PCK_EDGE_OFFSET_SEC0_mask_ADDR32;
            break;
        case 2:
            reg_addr = PCK_EDGE_OFFSET_SEC1_mask_ADDR32;
            break;
        case 3:
            reg_addr = PCK_EDGE_OFFSET_SEC2_mask_ADDR32;
            break;
        case 4:
            reg_addr = PCK_EDGE_OFFSET_SEC3_mask_ADDR32;
            break;
        default:
            DPRINT << "Aborting! Invalid register id (valid ids are between 1 and 4)" << ENDL();
            return;
    }
    // Get pointer to registers for current state ID
    uint32_t reg_val = cfg[reg_addr];

    dprint_tensix_struct_field(reg_val, 0xffff, 0, "mask");
    dprint_tensix_struct_field(reg_val, 0x10000, 16, "mode");
    dprint_tensix_struct_field(reg_val, 0x60000, 17, "tile_row_set_select_pack0");
    dprint_tensix_struct_field(reg_val, 0x180000, 19, "tile_row_set_select_pack1");
    dprint_tensix_struct_field(reg_val, 0x600000, 21, "tile_row_set_select_pack2");
    dprint_tensix_struct_field(reg_val, 0x1800000, 23, "tile_row_set_select_pack3");
}

// Choose what register you want printed with reg_id (1-4), 0 for all
inline void dprint_tensix_pck_edge_offset(uint reg_id = 0) {
#ifdef ARCH_GRAYSKULL
    DPRINT << "GRAYSKULL HAS NO PACK EDGE OFFSET REGISTERS" << ENDL();
    return;
#endif

    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    if (reg_id) {
        DPRINT << "REG_ID: " << reg_id << " ";
        dprint_tensix_pck_edge_offset_helper(reg_id, cfg);
    }
    // Print all registers
    else {
        for (uint i = 1; i < 5; i++) {
            DPRINT << "REG_ID: " << i << " ";
            dprint_tensix_pck_edge_offset_helper(i, cfg);
        }
    }

    DPRINT << ENDL();

}

// Printing packer counters
inline void dprint_tensix_pack_counters_helper(uint reg_id, const volatile uint tt_reg_ptr* cfg) {

    uint32_t reg_addr = 0;
    switch (reg_id) {
        case 1:
            reg_addr = PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32;
            break;
        case 2:
            reg_addr = PACK_COUNTERS_SEC1_pack_per_xy_plane_ADDR32;
            break;
        case 3:
            reg_addr = PACK_COUNTERS_SEC2_pack_per_xy_plane_ADDR32;
            break;
        case 4:
            reg_addr = PACK_COUNTERS_SEC3_pack_per_xy_plane_ADDR32;
            break;
        default:
            DPRINT << "Aborting! Invalid register id (valid ids are between 1 and 4)" << ENDL();
            return;
    }
    // Get pointer to registers for current state ID
    uint32_t reg_val = cfg[reg_addr];

    dprint_tensix_struct_field(reg_val, 0xff, 0, "pack_per_xy_plane", true);
    dprint_tensix_struct_field(reg_val, 0xff00, 8, "pack_reads_per_xy_plane", true);
    dprint_tensix_struct_field(reg_val, 0x7f0000, 16, "pack_xys_per_til", true);
    dprint_tensix_struct_field(reg_val, 0x800000, 23, "pack_yz_transposed");
    dprint_tensix_struct_field(reg_val, 0xff000000, 24, "pack_per_xy_plane_offset", true);
}

// Choose what register you want printed with reg_id (1-4), 0 for all
inline void dprint_tensix_pack_counters(uint reg_id = 0) {
#ifdef ARCH_GRAYSKULL
    DPRINT << "GRAYSKULL HAS NO PACK COUNTERS REGISTERS" << ENDL();
    return;
#endif
    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    if (reg_id) {
        DPRINT << "REG_ID: " << reg_id << " ";
        dprint_tensix_pack_counters_helper(reg_id, cfg);
    }
    // Print all registers
    else {
        for (uint i = 1; i < 5; i++) {
            DPRINT << "REG_ID: " << i << " ";
            dprint_tensix_pack_counters_helper(i, cfg);
        }
    }

    DPRINT << ENDL();
}

// Printing packer strides
inline void dprint_tensix_pack_strides_helper(uint reg_id, const volatile uint tt_reg_ptr* cfg) {

    uint32_t reg_addr = 0;
    switch (reg_id) {
        case 1:
            reg_addr = PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32;
            break;
        case 2:
            reg_addr = PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32;
            break;
        default:
            DPRINT << "Aborting! Invalid register id (valid ids are between 1 and 2)" << ENDL();
            break;
    }

    // word 0 xy_stride
    uint32_t word = cfg[reg_addr];
    dprint_tensix_struct_field(word, 0xfff, 0, "x_stride", true);
    dprint_tensix_struct_field(word, 0xfff000, 12, "y_stride", true);
    
    // word 1 zw_stride
    word = cfg[reg_addr + 1];
    dprint_tensix_struct_field(word, 0xfff, 0, "z_stride", true);
    dprint_tensix_struct_field(word, 0xffff000, 12, "w_stride", true);
}

// Choose what register you want printed (1-2). 0 for all.
inline void dprint_tensix_pack_strides(uint reg_id = 0) {

    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    if (reg_id) {
        DPRINT << "REG_ID: " << reg_id << " ";
        dprint_tensix_pack_strides_helper(reg_id, cfg);
    }
    // Print all registers
    else {
        for (uint i = 1; i < 3; i++) {
            DPRINT << "REG_ID: " << i << " ";
            dprint_tensix_pack_strides_helper(i, cfg);
        }
    }

    DPRINT << ENDL();
}


