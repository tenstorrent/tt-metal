// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_debug.h"
#include "compute_kernel_api.h"
#include "dprint.h"
#include "tensix_types.h"

#include "cpack_common.h"
#include "cunpack_common.h"

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

// Make it that it returns string not dprints it
inline void dprint_data_format(uint8_t data_format) {
    switch (data_format) {
        case (uint8_t) DataFormat::Float32:
            DPRINT << "Float32";
            break;
        case (uint8_t) DataFormat::Float16:
            DPRINT << "Float16";
            break;
        case (uint8_t) DataFormat::Bfp8:
            DPRINT << "Bfp8";
            break;
        case (uint8_t) DataFormat::Bfp4:
            DPRINT << "Bfp4";
            break;
        case (uint8_t) DataFormat::Bfp2:
            DPRINT << "Bfp2";
            break;
        case (uint8_t) DataFormat::Float16_b:
            DPRINT << "Float16_b";
            break;
        case (uint8_t) DataFormat::Bfp8_b:
            DPRINT << "Bfp8_b";
            break;
        case (uint8_t) DataFormat::Bfp4_b:
            DPRINT << "Bfp4_b";
            break;
        case (uint8_t) DataFormat::Bfp2_b:
            DPRINT << "Bfp2_b";
            break;
        case (uint8_t) DataFormat::Lf8:
            DPRINT << "Lf8";
            break;
        case (uint8_t) DataFormat::Int8:
            DPRINT << "Int8";
            break;
        case (uint8_t) DataFormat::UInt8:
            DPRINT << "UInt8";
            break;
        case (uint8_t) DataFormat::UInt16:
            DPRINT << "UInt16";
            break;
        case (uint8_t) DataFormat::Int32:
            DPRINT << "Int32";
            break;
        case (uint8_t) DataFormat::UInt32:
            DPRINT << "UInt32";
            break;
        case (uint8_t) DataFormat::Tf32:
            DPRINT << "Tf32";
            break;
        default:
            DPRINT << "INVALID DATA FORMAT";
            break;
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
    DPRINT << ((word & mask) >> shamt) << ENDL();
}

// NOTE: FUNCTIONS WITHOUT ARCH NAME (GRAYSKULL, WORMHOLE, BLACKHOLE) AND WITHOUT HELPER SUFIX ARE INTENDED TO BE USE

// HARDWARE SPECIFIC FUNCTIONS

#ifdef ARCH_GRAYSKULL
inline void dprint_tensix_unpack_tile_descriptor_grayskull() {
    ckernel::unpacker::unpack_tile_descriptor_t tile_descriptor = ckernel::unpacker::read_unpack_tile_descriptor();

    DPRINT << "in_data_format: " << HEX() << tile_descriptor.in_data_format << ENDL();
    DPRINT << "uncompressed: " << HEX() << tile_descriptor.uncompressed << ENDL();
    DPRINT << "reserved_0: " << HEX() << tile_descriptor.reserved_0 << ENDL();
    DPRINT << "blobs_per_xy_plane: " << tile_descriptor.blobs_per_xy_plane << ENDL();
    DPRINT << "reserved_1: " << HEX() << tile_descriptor.reserved_1 << ENDL();
    DPRINT << "x_dim: " << tile_descriptor.x_dim << ENDL();
    DPRINT << "y_dim: " << tile_descriptor.y_dim << ENDL();
    DPRINT << "z_dim: " << tile_descriptor.z_dim << ENDL();
    DPRINT << "w_dim: " << tile_descriptor.w_dim << ENDL();
    DPRINT << "blobs_y_start: " << tile_descriptor.blobs_y_start << ENDL();
    DPRINT << "digest_type: " << HEX() << tile_descriptor.digest_type << ENDL();
    DPRINT << "digest_size: " << tile_descriptor.digest_size << ENDL();
}

inline void dprint_tensix_unpack_config_grayskull() {
    ckernel::unpacker::unpack_config_t config = ckernel::unpacker::read_unpack_config();

    DPRINT << "out_format: " << HEX() << config.out_data_format << ENDL();
    DPRINT << "throttle_mode: " << HEX() << config.throttle_mode << ENDL();
    DPRINT << "cntx_cnt: " << HEX() << config.context_count << ENDL();
    DPRINT << "halo_mode: " << HEX() << config.haloize_mode << ENDL();
    DPRINT << "tile_mode: " << HEX() << config.tileize_mode << ENDL();
    DPRINT << "force_shrd_exp: " << HEX() << config.force_shared_exp << ENDL();
    DPRINT << "res_0: " << HEX() << config.reserved_0 << ENDL();
    DPRINT << "upsmpl_rate: " << config.upsample_rate << ENDL();
    DPRINT << "upsmpl_and_intrlv: " << HEX() << config.upsamle_and_interlave << ENDL();
    DPRINT << "shamt: " << config.shift_amount << ENDL();
    DPRINT << "uncmpr_cntx0_3: " << HEX() << config.uncompress_cntx0_3 << ENDL();
    DPRINT << "res_1: " << HEX() << config.reserved_1 << ENDL();
    DPRINT << "uncmpr_cntx4_7: " << HEX() << config.uncompress_cntx4_7 << ENDL();
    DPRINT << "res_2: " << HEX() << config.reserved_2 << ENDL();
    DPRINT << "limit_addr: " << HEX() << config.limit_addr << ENDL();
    DPRINT << "fifo_sz: " << config.fifo_size << ENDL();
}

inline void dprint_tensix_pack_config_grayskull(uint32_t reg_addr, const volatile uint tt_reg_ptr* cfg) {
    ckernel::packer::pack_config_t config = ckernel::packer::read_pack_config(reg_addr, cfg);

    DPRINT << "row_ptr_sec_sz: " << config.row_ptr_section_size << ENDL();
    DPRINT << "exp_sec_sz: " << config.exp_section_size << ENDL();
    DPRINT << "l1_dst_adr: " << HEX() << config.l1_dest_addr << ENDL();
    DPRINT << "uncmpr: " << HEX() << config.uncompress << ENDL();
    DPRINT << "add_l1_dst_adr_ofs: " << HEX() << config.add_l1_dest_addr_offset << ENDL();
    DPRINT << "r0: " << HEX() << config.reserved_0 << ENDL();
    DPRINT << "out_frmt: " << HEX() << config.out_data_format << ENDL();
    DPRINT << "in_frmt: " << HEX() << config.in_data_format << ENDL();
    DPRINT << "r1: " << HEX() << config.reserved_1 << ENDL();
    DPRINT << "src_if_sel: " << HEX() << config.src_if_sel << ENDL();
    DPRINT << "pck_per_xy_pl: " << config.pack_per_xy_plane << ENDL();
    DPRINT << "l1_src_adr: " << HEX() << config.l1_src_addr << ENDL();
    DPRINT << "dsmpl_mask: " << HEX() << config.downsample_mask << ENDL();
    DPRINT << "dsmpl_shcnt: " << config.downsample_shift_count << ENDL();
    DPRINT << "r_md: " << HEX() << config.read_mode << ENDL();
    DPRINT << "exp_th_en: " << HEX() << config.exp_threshold_en << ENDL();
    DPRINT << "r2: " << HEX() << config.reserved_2 << ENDL();
    DPRINT << "exp_th: " << config.exp_threshold << ENDL();
}
#else  // ARCH_WORMHOLE or ARCH_BLACKHOLE
inline void dprint_tensix_unpack_tile_descriptor_wormhole_or_blackhole() {
    ckernel::unpacker::unpack_tile_descriptor_t tile_descriptor = ckernel::unpacker::read_unpack_tile_descriptor();

    DPRINT << "in_data_format: " << HEX() << tile_descriptor.in_data_format << ENDL();
    DPRINT << "uncompressed: " << HEX() << tile_descriptor.uncompressed << ENDL();
    DPRINT << "reserved_0: " << HEX() << tile_descriptor.reserved_0 << ENDL();
    DPRINT << "blobs_per_xy_plane: " << tile_descriptor.blobs_per_xy_plane << ENDL();
    DPRINT << "reserved_1: " << HEX() << tile_descriptor.reserved_1 << ENDL();
    DPRINT << "x_dim: " << tile_descriptor.x_dim << ENDL();
    DPRINT << "y_dim: " << tile_descriptor.y_dim << ENDL();
    DPRINT << "z_dim: " << tile_descriptor.z_dim << ENDL();
    DPRINT << "w_dim: " << tile_descriptor.w_dim << ENDL();
    DPRINT << "blobs_y_start: " << ((tile_descriptor.blobs_y_start_hi << 16) | tile_descriptor.blobs_y_start_lo)
           << ENDL();
    DPRINT << "digest_type: " << HEX() << tile_descriptor.digest_type << ENDL();
    DPRINT << "digest_size: " << tile_descriptor.digest_size << ENDL();
}

inline void dprint_tensix_unpack_config_wormhole_or_blackhole() {
    ckernel::unpacker::unpack_config_t config = ckernel::unpacker::read_unpack_config();

    DPRINT << "out_frmt: " << HEX() << config.out_data_format << ENDL();
    DPRINT << "throt_md: " << HEX() << config.throttle_mode << ENDL();
    DPRINT << "cx_cnt: " << HEX() << config.context_count << ENDL();
    DPRINT << "halo_md: " << HEX() << config.haloize_mode << ENDL();
    DPRINT << "tile_md: " << HEX() << config.tileize_mode << ENDL();
    DPRINT << "u_s_s: " << HEX() << config.unpack_src_reg_set_update << ENDL();
    DPRINT << "unpis: " << HEX() << config.unpack_if_sel << ENDL();
    DPRINT << "ups_rate: " << HEX() << config.upsample_rate << ENDL();
    DPRINT << "r1: " << HEX() << config.reserved_1 << ENDL();
    DPRINT << "ups_and_int: " << config.upsamle_and_interlave << ENDL();
    DPRINT << "shamt: " << config.shift_amount << ENDL();
    DPRINT << "u_cx0_3: " << HEX() << config.uncompress_cntx0_3 << ENDL();
    DPRINT << "u_cx4_7: " << HEX() << config.unpack_if_sel_cntx0_3 << ENDL();
    DPRINT << "frc_shrd_exp: " << HEX() << config.force_shared_exp << ENDL();
    DPRINT << "r2: " << HEX() << config.reserved_2 << ENDL();
    DPRINT << "u_cx4_7: " << HEX() << config.uncompress_cntx4_7 << ENDL();
    DPRINT << "u_cx4_7: " << HEX() << config.unpack_if_sel_cntx4_7 << ENDL();
    DPRINT << "r3: " << HEX() << config.reserved_3 << ENDL();
    DPRINT << "l_ad: " << HEX() << config.limit_addr << ENDL();
    DPRINT << "r4: " << HEX() << config.reserved_4 << ENDL();
    DPRINT << "f_sz: " << config.fifo_size << ENDL();
    DPRINT << "r5: " << HEX() << config.reserved_5 << ENDL();
}

#ifdef ARCH_WORMHOLE
inline void dprint_tensix_pack_config_wormhole(uint32_t reg_addr, const volatile uint tt_reg_ptr* cfg) {
    ckernel::packer::pack_config_t config = ckernel::packer::read_pack_config(reg_addr, cfg);

    DPRINT << "r_p_s_sz: " << config.row_ptr_section_size << ENDL();
    DPRINT << "e_s_sz: " << config.exp_section_size << ENDL();
    DPRINT << "l1_d_a: " << HEX() << config.l1_dest_addr << ENDL();
    DPRINT << "ucp: " << HEX() << config.uncompress << ENDL();
    DPRINT << "a_l1_d_a_o: " << HEX() << config.add_l1_dest_addr_offset << ENDL();
    DPRINT << "r0: " << HEX() << config.reserved_0 << ENDL();
    DPRINT << "o_fmt: " << HEX() << config.out_data_format << ENDL();
    DPRINT << "i_fmt: " << HEX() << config.in_data_format << ENDL();
    DPRINT << "r1: " << HEX() << config.reserved_1 << ENDL();
    DPRINT << "sr_if_sl: " << HEX() << config.src_if_sel << ENDL();
    DPRINT << "p_xy_pl: " << config.pack_per_xy_plane << ENDL();
    DPRINT << "l1_sr_ad: " << HEX() << config.l1_src_addr << ENDL();
    DPRINT << "d_msk: " << HEX() << config.downsample_mask << ENDL();
    DPRINT << "d_shcnt: " << config.downsample_shift_count << ENDL();
    DPRINT << "r_md: " << HEX() << config.read_mode << ENDL();
    DPRINT << "e_t_e: " << HEX() << config.exp_threshold_en << ENDL();
    DPRINT << "p_l1_ac_d_p_0_f: " << HEX() << config.pack_l1_acc_disable_pack_zero_flag << ENDL();
    DPRINT << "r2: " << HEX() << config.reserved_2 << ENDL();
    DPRINT << "exp_th: " << config.exp_threshold << ENDL();
}
#endif  // ARCH_WORMHOLE

#ifdef ARCH_BLACKHOLE
inline void dprint_tensix_pack_config_blackhole(uint32_t reg_addr, const volatile uint tt_reg_ptr* cfg) {
    ckernel::packer::pack_config_t config = ckernel::packer::read_pack_config(reg_addr, cfg);

    DPRINT << "row_ptr_sec_sz: " << HEX() << config.row_ptr_section_size << ENDL();
    DPRINT << "exp_sec_sz: " << HEX() << config.exp_section_size << ENDL();
    DPRINT << "l1_dst_adr: " << HEX() << config.l1_dest_addr << ENDL();
    DPRINT << "uncmps: " << HEX() << config.uncompress << ENDL();
    DPRINT << "aldao: " << HEX() << config.add_l1_dest_addr_offset << ENDL();
    DPRINT << "dis_pck_0_fl: " << HEX() << config.disable_pack_zero_flag << ENDL();
    DPRINT << "r0: " << HEX() << config.reserved_0 << ENDL();
    DPRINT << "out_frmt: " << HEX() << config.out_data_format << ENDL();
    DPRINT << "in_frmt: " << HEX() << config.in_data_format << ENDL();
    DPRINT << "dsea: " << HEX() << config.dis_shared_exp_assembler << ENDL();
    DPRINT << "alpis: " << HEX() << config.auto_set_last_pacr_intf_sel << ENDL();
    DPRINT << "en_out_fifo: " << HEX() << config.enable_out_fifo << ENDL();
    DPRINT << "sub_l1_til_h_sz: " << config.sub_l1_tile_header_size << ENDL();
    DPRINT << "src_if_sel: " << HEX() << config.src_if_sel << ENDL();
    DPRINT << "pck_st_intf_pos: " << HEX() << config.pack_start_intf_pos << ENDL();
    DPRINT << "apd0co: " << HEX() << config.all_pack_disable_zero_compress_ovrd << ENDL();
    DPRINT << "add_til_h_sz: " << config.add_tile_header_size << ENDL();
    DPRINT << "pdypso: " << HEX() << config.pack_dis_y_pos_start_offset << ENDL();
    DPRINT << "l1_src_adr: " << HEX() << config.l1_src_addr << ENDL();
}
#endif  // ARCH_BLACKHOLE

// Choose what register you want printed with reg_id (1-4)
inline void dprint_tensix_pack_config_helper(uint reg_id) {
    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    uint32_t reg_addr = 0;
    switch (reg_id) {
        case 1: reg_addr = THCON_SEC0_REG1_Row_start_section_size_ADDR32; break;
        case 2: reg_addr = THCON_SEC0_REG8_Row_start_section_size_ADDR32; break;
        case 3: reg_addr = THCON_SEC1_REG1_Row_start_section_size_ADDR32; break;
        case 4: reg_addr = THCON_SEC1_REG8_Row_start_section_size_ADDR32; break;
        default: DPRINT << "Aborting! Invalid register id (valid ids are between 1 and 4)" << ENDL(); return;
    }

    DPRINT << "REG_ID: " << reg_id << ENDL();

#ifdef ARCH_GRAYSKULL
    dprint_tensix_pack_config_grayskull(reg_addr, cfg);
#elif ARCH_WORMHOLE
    dprint_tensix_pack_config_wormhole(reg_addr, cfg);
#else
    dprint_tensix_pack_config_blackhole(reg_addr, cfg);
#endif
}

// Printing packer edge offset
inline void dprint_tensix_pck_edge_offset_helper(uint reg_id, const volatile uint tt_reg_ptr* cfg) {
    uint32_t reg_addr = 0;
    switch (reg_id) {
        case 1: reg_addr = PCK_EDGE_OFFSET_SEC0_mask_ADDR32; break;
        case 2: reg_addr = PCK_EDGE_OFFSET_SEC1_mask_ADDR32; break;
        case 3: reg_addr = PCK_EDGE_OFFSET_SEC2_mask_ADDR32; break;
        case 4: reg_addr = PCK_EDGE_OFFSET_SEC3_mask_ADDR32; break;
        default: DPRINT << "Aborting! Invalid register id (valid ids are between 1 and 4)" << ENDL(); return;
    }

    ckernel::packer::pck_edge_offset_t edge = ckernel::packer::read_pck_edge_offset(reg_addr, cfg);

    DPRINT << "mask: " << HEX() << edge.mask << ENDL();
    DPRINT << "mode: " << HEX() << edge.mode << ENDL();
    DPRINT << "tile_row_set_select_pack0: " << HEX() << edge.tile_row_set_select_pack0 << ENDL();
    DPRINT << "tile_row_set_select_pack1: " << HEX() << edge.tile_row_set_select_pack1 << ENDL();
    DPRINT << "tile_row_set_select_pack2: " << HEX() << edge.tile_row_set_select_pack2 << ENDL();
    DPRINT << "tile_row_set_select_pack3: " << HEX() << edge.tile_row_set_select_pack3 << ENDL();
    DPRINT << "reserved: " << HEX() << edge.reserved << ENDL();
}

// HELPER FUNCTIONS

// Printing packer counters
inline void dprint_tensix_pack_counters_helper(uint reg_id, const volatile uint tt_reg_ptr* cfg) {
    uint32_t reg_addr = 0;
    switch (reg_id) {
        case 1: reg_addr = PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32; break;
        case 2: reg_addr = PACK_COUNTERS_SEC1_pack_per_xy_plane_ADDR32; break;
        case 3: reg_addr = PACK_COUNTERS_SEC2_pack_per_xy_plane_ADDR32; break;
        case 4: reg_addr = PACK_COUNTERS_SEC3_pack_per_xy_plane_ADDR32; break;
        default: DPRINT << "Aborting! Invalid register id (valid ids are between 1 and 4)" << ENDL(); return;
    }

    ckernel::packer::pack_counters_t counters = ckernel::packer::read_pack_counters(reg_addr, cfg);

    DPRINT << "pack_per_xy_plane: " << counters.pack_per_xy_plane << ENDL();
    DPRINT << "pack_reads_per_xy_plane: " << counters.pack_reads_per_xy_plane << ENDL();
    DPRINT << "pack_xys_per_til: " << counters.pack_xys_per_til << ENDL();
    DPRINT << "pack_yz_transposed: " << HEX() << counters.pack_yz_transposed << ENDL();
    DPRINT << "pack_per_xy_plane_offset: " << counters.pack_per_xy_plane_offset << ENDL();
}

// Printing packer strides
inline void dprint_tensix_pack_strides_helper(uint reg_id, const volatile uint tt_reg_ptr* cfg) {
    uint32_t reg_addr = 0;
    switch (reg_id) {
        case 1: reg_addr = PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32; break;
        case 2: reg_addr = PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32; break;
        default: DPRINT << "Aborting! Invalid register id (valid ids are between 1 and 2)" << ENDL(); break;
    }

    // word 0 xy_stride
    uint32_t word = cfg[reg_addr];
    dprint_tensix_struct_field(word, 0xfff, 0, "x_stride", true);      // decimal
    dprint_tensix_struct_field(word, 0xfff000, 12, "y_stride", true);  // decimal

    // word 1 zw_stride
    word = cfg[reg_addr + 1];
    dprint_tensix_struct_field(word, 0xfff, 0, "z_stride", true);       // decimal
    dprint_tensix_struct_field(word, 0xffff000, 12, "w_stride", true);  // decimal
}

// FUNCTIONS TO USE

// Print content of the register field by field. Issue: No ENDL.
inline void dprint_tensix_alu_config() {
    ckernel::unpacker::alu_config_t config = ckernel::unpacker::read_alu_config();

    DPRINT << "Fpu_srnd_en: " << HEX() << config.ALU_ROUNDING_MODE_Fpu_srnd_en << ENDL();
    DPRINT << "Gasket_srnd_en: " << HEX() << config.ALU_ROUNDING_MODE_Gasket_srnd_en << ENDL();
    DPRINT << "Packer_srnd_en: " << HEX() << config.ALU_ROUNDING_MODE_Packer_srnd_en << ENDL();
    DPRINT << "Padding: " << HEX() << config.ALU_ROUNDING_MODE_Padding << ENDL();
    DPRINT << "GS_LF: " << HEX() << config.ALU_ROUNDING_MODE_GS_LF << ENDL();
    DPRINT << "Bfp8_HF: " << HEX() << config.ALU_ROUNDING_MODE_Bfp8_HF << ENDL();
    DPRINT << "SrcAUnsigned: " << HEX() << config.ALU_FORMAT_SPEC_REG0_SrcAUnsigned << ENDL();
    DPRINT << "SrcBUnsigned: " << HEX() << config.ALU_FORMAT_SPEC_REG0_SrcBUnsigned << ENDL();
    DPRINT << "SrcA: " << HEX() << config.ALU_FORMAT_SPEC_REG0_SrcA << ENDL();
    DPRINT << "SrcB: " << HEX() << config.ALU_FORMAT_SPEC_REG1_SrcB << ENDL();
    DPRINT << "Dstacc: " << HEX() << config.ALU_FORMAT_SPEC_REG2_Dstacc << ENDL();
    DPRINT << "Fp32_enabled: " << HEX() << config.ALU_ACC_CTRL_Fp32_enabled << ENDL();
    DPRINT << "SFPU_Fp32_enabled: " << HEX() << config.ALU_ACC_CTRL_SFPU_Fp32_enabled << ENDL();
    DPRINT << "INT8_math_enabled: " << HEX() << config.ALU_ACC_CTRL_INT8_math_enabled << ENDL();
}

inline void dprint_tensix_pack_relu_config() {
    ckernel::packer::relu_config_t config = ckernel::packer::read_relu_config();

    DPRINT << "zero_flag_disabled_src: " << HEX() << config.ALU_ACC_CTRL_Zero_Flag_disabled_src << ENDL();
    DPRINT << "zero_flag_disabled_dst: " << HEX() << config.ALU_ACC_CTRL_Zero_Flag_disabled_dst << ENDL();
    DPRINT << "apply_relu: " << HEX() << config.STACC_RELU_ApplyRelu << ENDL();
    DPRINT << "relu_threshold: " << config.STACC_RELU_ReluThreshold << ENDL();
    DPRINT << "main: " << HEX() << config.DISABLE_RISC_BP_Disable_main << ENDL();
    DPRINT << "trisc: " << HEX() << config.DISABLE_RISC_BP_Disable_trisc << ENDL();
    DPRINT << "ncrisc: " << HEX() << config.DISABLE_RISC_BP_Disable_ncrisc << ENDL();
    DPRINT << "bmp_clear_main: " << HEX() << config.DISABLE_RISC_BP_Disable_bmp_clear_main << ENDL();
    DPRINT << "bmp_clear_trisc: " << HEX() << config.DISABLE_RISC_BP_Disable_bmp_clear_trisc << ENDL();
    DPRINT << "bmp_clear_ncrisc: " << HEX() << config.DISABLE_RISC_BP_Disable_bmp_clear_ncrisc << ENDL();
}

// Printing dest control bits
inline void dprint_tensix_dest_rd_ctrl() {
    ckernel::packer::dest_rd_ctrl_t dest = ckernel::packer::read_dest_rd_ctrl();

    DPRINT << "read_32b_data: " << HEX() << dest.PCK_DEST_RD_CTRL_Read_32b_data << "; " << ENDL();
    DPRINT << "read_unsigned: " << HEX() << dest.PCK_DEST_RD_CTRL_Read_unsigned << "; " << ENDL();
    DPRINT << "read_int8: " << HEX() << dest.PCK_DEST_RD_CTRL_Read_int8 << "; " << ENDL();
    DPRINT << "round_10b_mant: " << HEX() << dest.PCK_DEST_RD_CTRL_Round_10b_mant << "; " << ENDL();
    DPRINT << "reserved: " << HEX() << dest.PCK_DEST_RD_CTRL_Reserved << "; " << ENDL();
}

// Choose what register you want printed with reg_id (1-4), 0 for all
inline void dprint_tensix_pck_edge_offset(uint reg_id = 0) {
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    if (reg_id) {
        DPRINT << "REG_ID: " << reg_id << ENDL();
        dprint_tensix_pck_edge_offset_helper(reg_id, cfg);
    }
    // Print all registers
    else {
        for (uint i = 1; i < 5; i++) {
            DPRINT << "REG_ID: " << i << ENDL();
            dprint_tensix_pck_edge_offset_helper(i, cfg);
        }
    }
}

// Choose what register you want printed with reg_id (1-4), 0 for all
inline void dprint_tensix_pack_counters(uint reg_id = 0) {
    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    if (reg_id) {
        DPRINT << "REG_ID: " << reg_id << ENDL();
        dprint_tensix_pack_counters_helper(reg_id, cfg);
    }
    // Print all registers
    else {
        for (uint i = 1; i < 5; i++) {
            DPRINT << "REG_ID: " << i << ENDL();
            dprint_tensix_pack_counters_helper(i, cfg);
        }
    }
}

#endif  // END OF ELSE

inline void dprint_tensix_unpack_tile_descriptor() {
#ifdef ARCH_GRAYSKULL
    dprint_tensix_unpack_tile_descriptor_grayskull();
#else
    dprint_tensix_unpack_tile_descriptor_wormhole_or_blackhole();
#endif
}

inline void dprint_tensix_unpack_config() {
#ifdef ARCH_GRAYSKULL
    dprint_tensix_unpack_config_grayskull();
#else
    dprint_tensix_unpack_config_wormhole_or_blackhole();
#endif
}

inline void dprint_tensix_pack_config(uint reg_id = 0) {
    if (reg_id) {
        dprint_tensix_pack_config_helper(reg_id);
    } else {
        dprint_tensix_pack_config_helper(1);
        dprint_tensix_pack_config_helper(2);
        dprint_tensix_pack_config_helper(3);
        dprint_tensix_pack_config_helper(4);
    }
}

// Choose what register you want printed (1-2). 0 for all.
inline void dprint_tensix_pack_strides(uint reg_id = 0) {

    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    if (reg_id) {
        DPRINT << "REG_ID: " << reg_id << ENDL();
        dprint_tensix_pack_strides_helper(reg_id, cfg);
    }
    // Print all registers
    else {
        for (uint i = 1; i < 3; i++) {
            DPRINT << "REG_ID: " << i << ENDL();
            dprint_tensix_pack_strides_helper(i, cfg);
        }
    }
}
