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

#ifdef ARCH_GRAYSKULL
inline void dprint_tensix_unpack_tile_descriptor_grayskull() {
    ckernel::unpacker::unpack_tile_descriptor_t tile_descriptor = ckernel::unpacker::read_unpack_tile_descriptor();

    DPRINT << "in_data_format: " << HEX() << tile_descriptor.in_data_format << "; ";
    DPRINT << "uncompressed: " << HEX() << tile_descriptor.uncompressed << "; ";
    DPRINT << "reserved_0: " << HEX() << tile_descriptor.reserved_0 << "; ";
    DPRINT << "blobs_per_xy_plane: " << tile_descriptor.blobs_per_xy_plane << "; ";
    DPRINT << "reserved_1: " << HEX() << tile_descriptor.reserved_1 << "; ";
    DPRINT << "x_dim: " << tile_descriptor.x_dim << "; ";
    DPRINT << "y_dim: " << tile_descriptor.y_dim << "; ";
    DPRINT << "z_dim: " << tile_descriptor.z_dim << "; ";
    DPRINT << "w_dim: " << tile_descriptor.w_dim << "; ";
    DPRINT << "blobs_y_start: " << tile_descriptor.blobs_y_start << "; ";
    DPRINT << "digest_type: " << HEX() << tile_descriptor.digest_type << "; ";
    DPRINT << "digest_size: " << tile_descriptor.digest_size << "; ";
    DPRINT << ENDL();
}

inline void dprint_tensix_unpack_config_grayskull() {
    ckernel::unpacker::unpack_config_t config = ckernel::unpacker::read_unpack_config();

    DPRINT << "out_format: " << HEX() << config.out_data_format << "; ";
    DPRINT << "throttle_mode: " << HEX() << config.throttle_mode << "; ";
    DPRINT << "cntx_cnt: " << HEX() << config.context_count << "; ";
    DPRINT << "halo_mode: " << HEX() << config.haloize_mode << "; ";
    DPRINT << "tile_mode: " << HEX() << config.tileize_mode << "; ";
    DPRINT << "force_shrd_exp: " << HEX() << config.force_shared_exp << "; ";
    DPRINT << "res_0: " << HEX() << config.reserved_0 << "; ";
    DPRINT << "upsmpl_rate: " << config.upsample_rate << "; ";
    DPRINT << "upsmpl_and_intrlv: " << HEX() << config.upsamle_and_interlave << "; ";
    DPRINT << "shamt: " << config.shift_amount << "; ";
    DPRINT << "uncmpr_cntx0_3: " << HEX() << config.uncompress_cntx0_3 << "; ";
    DPRINT << "res_1: " << HEX() << config.reserved_1 << "; ";
    DPRINT << "uncmpr_cntx4_7: " << HEX() << config.uncompress_cntx4_7 << "; ";
    DPRINT << "res_2: " << HEX() << config.reserved_2 << "; ";
    DPRINT << "limit_addr: " << HEX() << config.limit_addr << "; ";
    DPRINT << "fifo_sz: " << config.fifo_size << "; ";
    DPRINT << ENDL();
}

inline void dprint_tensix_pack_config_grayskull(uint32_t reg_addr, const volatile uint tt_reg_ptr* cfg) {
    ckernel::packer::pack_config_t config = ckernel::packer::read_pack_config(reg_addr, cfg);

    DPRINT << "row_ptr_sec_sz: " << config.row_ptr_section_size << "; ";
    DPRINT << "exp_sec_sz: " << config.exp_section_size << "; ";
    DPRINT << "l1_dst_adr: " << HEX() << config.l1_dest_addr << "; ";
    DPRINT << "uncmpr: " << HEX() << config.uncompress << "; ";
    DPRINT << "add_l1_dst_adr_ofs: " << HEX() << config.add_l1_dest_addr_offset << "; ";
    DPRINT << "r0: " << HEX() << config.reserved_0 << "; ";
    DPRINT << "out_frmt: " << HEX() << config.out_data_format << "; ";
    DPRINT << "in_frmt: " << HEX() << config.in_data_format << "; ";
    DPRINT << "r1: " << HEX() << config.reserved_1 << "; ";
    DPRINT << "src_if_sel: " << HEX() << config.src_if_sel << "; ";
    DPRINT << "pck_per_xy_pl: " << config.pack_per_xy_plane << "; ";
    DPRINT << "l1_src_adr: " << HEX() << config.l1_src_addr << "; ";
    DPRINT << "dsmpl_mask: " << HEX() << config.downsample_mask << "; ";
    DPRINT << "dsmpl_shcnt: " << config.downsample_shift_count << "; ";
    DPRINT << "r_md: " << HEX() << config.read_mode << "; ";
    DPRINT << "exp_th_en: " << HEX() << config.exp_threshold_en << "; ";
    DPRINT << "r2: " << HEX() << config.reserved_2 << "; ";
    DPRINT << "exp_th: " << config.exp_threshold << "; ";
    DPRINT << ENDL();
}
#else  // ARCH_WORMHOLE or ARCH_BLACKHOLE
inline void dprint_tensix_unpack_tile_descriptor_wormhole_or_blackhole() {
    ckernel::unpacker::unpack_tile_descriptor_t tile_descriptor = ckernel::unpacker::read_unpack_tile_descriptor();

    DPRINT << "in_data_format: " << HEX() << tile_descriptor.in_data_format << "; ";
    DPRINT << "uncompressed: " << HEX() << tile_descriptor.uncompressed << "; ";
    DPRINT << "reserved_0: " << HEX() << tile_descriptor.reserved_0 << "; ";
    DPRINT << "blobs_per_xy_plane: " << tile_descriptor.blobs_per_xy_plane << "; ";
    DPRINT << "reserved_1: " << HEX() << tile_descriptor.reserved_1 << "; ";
    DPRINT << "x_dim: " << tile_descriptor.x_dim << "; ";
    DPRINT << "y_dim: " << tile_descriptor.y_dim << "; ";
    DPRINT << "z_dim: " << tile_descriptor.z_dim << "; ";
    DPRINT << "w_dim: " << tile_descriptor.w_dim << "; ";
    DPRINT << "blobs_y_start: " << ((tile_descriptor.blobs_y_start_hi << 16) | tile_descriptor.blobs_y_start_lo) << "; ";
    DPRINT << "digest_type: " << HEX() << tile_descriptor.digest_type << "; ";
    DPRINT << "digest_size: " << tile_descriptor.digest_size << "; ";
    DPRINT << ENDL();
}

inline void dprint_tensix_unpack_config_wormhole_or_blackhole() {
    ckernel::unpacker::unpack_config_t config = ckernel::unpacker::read_unpack_config();

    DPRINT << "out_frmt: " << HEX() << config.out_data_format << "; ";
    DPRINT << "throt_md: " << HEX() << config.throttle_mode << "; ";
    DPRINT << "cx_cnt: " << HEX() << config.context_count << "; ";
    DPRINT << "halo_md: " << HEX() << config.haloize_mode << "; ";
    DPRINT << "tile_md: " << HEX() << config.tileize_mode << "; ";
    DPRINT << "u_s_s: " << HEX() << config.unpack_src_reg_set_update << "; ";
    DPRINT << "unpis: " << HEX() << config.unpack_if_sel << "; ";
    DPRINT << "ups_rate: " << HEX() << config.upsample_rate << "; ";
    DPRINT << "r1: " << HEX() << config.reserved_1 << "; ";
    DPRINT << "ups_and_int: " << config.upsamle_and_interlave << "; ";
    DPRINT << "shamt: " << config.shift_amount << " ;";
    DPRINT << "u_cx0_3: " << HEX() << config.uncompress_cntx0_3 << "; ";
    DPRINT << "u_cx4_7: " << HEX() << config.unpack_if_sel_cntx0_3 << "; ";
    DPRINT << "frc_shrd_exp: " << HEX() << config.force_shared_exp << "; ";
    DPRINT << "r2: " << HEX() << config.reserved_2 << "; ";
    DPRINT << "u_cx4_7: " << HEX() << config.uncompress_cntx4_7 << "; ";
    DPRINT << "u_cx4_7: " << HEX() << config.unpack_if_sel_cntx4_7 << "; ";
    DPRINT << "r3: " << HEX() << config.reserved_3 << "; ";
    DPRINT << "l_ad: " << HEX() << config.limit_addr << "; ";
    DPRINT << "r4: " << HEX() << config.reserved_4 << "; ";
    DPRINT << "f_sz: " << config.fifo_size << "; ";
    DPRINT << "r5: " << HEX() << config.reserved_5 << "; ";
    DPRINT << ENDL();
}

#ifdef ARCH_WORMHOLE
inline void dprint_tensix_pack_config_wormhole(uint32_t reg_addr, const volatile uint tt_reg_ptr* cfg) {
    ckernel::packer::pack_config_t config = ckernel::packer::read_pack_config(reg_addr, cfg);

    DPRINT << "r_p_s_sz: " << config.row_ptr_section_size << "; ";
    DPRINT << "e_s_sz: " << config.exp_section_size << "; ";
    DPRINT << "l1_d_a: " << HEX() << config.l1_dest_addr << "; ";
    DPRINT << "ucp: " << HEX() << config.uncompress << "; ";
    DPRINT << "a_l1_d_a_o: " << HEX() << config.add_l1_dest_addr_offset << "; ";
    DPRINT << "r0: " << HEX() << config.reserved_0 << "; ";
    DPRINT << "o_fmt: " << HEX() << config.out_data_format << "; ";
    DPRINT << "i_fmt: " << HEX() << config.in_data_format << "; ";
    DPRINT << "r1: " << HEX() << config.reserved_1 << "; ";
    DPRINT << "sr_if_sl: " << HEX() << config.src_if_sel << "; ";
    DPRINT << "p_xy_pl: " << config.pack_per_xy_plane << "; ";
    DPRINT << "l1_sr_ad: " << HEX() << config.l1_src_addr << "; ";
    DPRINT << "d_msk: " << HEX() << config.downsample_mask << "; ";
    DPRINT << "d_shcnt: " << config.downsample_shift_count << "; ";
    DPRINT << "r_md: " << HEX() << config.read_mode << "; ";
    DPRINT << "e_t_e: " << HEX() << config.exp_threshold_en << "; ";
    DPRINT << "p_l1_ac_d_p_0_f: " << HEX() << config.pack_l1_acc_disable_pack_zero_flag << "; ";
    DPRINT << "r2: " << HEX() << config.reserved_2 << "; ";
    DPRINT << "exp_th: " << config.exp_threshold << "; ";
    DPRINT << ENDL();
}
#endif  // ARCH_WORMHOLE

#ifdef ARCH_BLACKHOLE
inline void dprint_tensix_pack_config_blackhole(uint32_t reg_addr, const volatile uint tt_reg_ptr* cfg) {
    ckernel::packer::pack_config_t config = ckernel::packer::read_pack_config(reg_addr, cfg);

    DPRINT << "row_ptr_sec_sz: " << HEX() << config.row_ptr_section_size << "; ";
    DPRINT << "exp_sec_sz: " << HEX() << config.exp_section_size << "; ";
    DPRINT << "l1_dst_adr: " << HEX() << config.l1_dest_addr << "; ";
    DPRINT << "uncmps: " << HEX() << config.uncompress << "; ";
    DPRINT << "aldao: " << HEX() << config.add_l1_dest_addr_offset << "; ";
    DPRINT << "dis_pck_0_fl: " << HEX() << config.disable_pack_zero_flag << "; ";
    DPRINT << "r0: " << HEX() << config.reserved_0 << "; ";
    DPRINT << "out_frmt: " << HEX() << config.out_data_format << "; ";
    DPRINT << "in_frmt: " << HEX() << config.in_data_format << "; ";
    DPRINT << "dsea: " << HEX() << config.dis_shared_exp_assembler << "; ";
    DPRINT << "alpis: " << HEX() << config.auto_set_last_pacr_intf_sel << "; ";
    DPRINT << "en_out_fifo: " << HEX() << config.enable_out_fifo << "; ";
    DPRINT << "sub_l1_til_h_sz: " << config.sub_l1_tile_header_size << "; ";
    DPRINT << "src_if_sel: " << HEX() << config.src_if_sel << "; ";
    DPRINT << "pck_st_intf_pos: " << HEX() << config.pack_start_intf_pos << "; ";
    DPRINT << "apd0co: " << HEX() << config.all_pack_disable_zero_compress_ovrd << "; ";
    DPRINT << "add_til_h_sz: " << config.add_tile_header_size << "; ";
    DPRINT << "pdypso: " << HEX() << config.pack_dis_y_pos_start_offset << "; ";
    DPRINT << "l1_src_adr: " << HEX() << config.l1_src_addr << "; ";
    DPRINT << ENDL();
}
#endif  // ARCH_BLACKHOLE

// Print content of the register field by field. Issue: No ENDL.
inline void dprint_tensix_alu_config() {
    // Only wormhole and blackhole have this register

    ckernel::unpacker::alu_config_t config = ckernel::unpacker::read_alu_config();

    DPRINT << "Fpu_srnd_en: " << HEX() << config.ALU_ROUNDING_MODE_Fpu_srnd_en << "; ";
    DPRINT << "Gasket_srnd_en: " << HEX() << config.ALU_ROUNDING_MODE_Gasket_srnd_en << "; ";
    DPRINT << "Packer_srnd_en: " << HEX() << config.ALU_ROUNDING_MODE_Packer_srnd_en << "; ";
    DPRINT << "Padding: " << HEX() << config.ALU_ROUNDING_MODE_Padding << "; ";
    DPRINT << "GS_LF: " << HEX() << config.ALU_ROUNDING_MODE_GS_LF << "; ";
    DPRINT << "Bfp8_HF: " << HEX() << config.ALU_ROUNDING_MODE_Bfp8_HF << "; ";
    DPRINT << "SrcAUnsigned: " << HEX() << config.ALU_FORMAT_SPEC_REG0_SrcAUnsigned << "; ";
    DPRINT << "SrcBUnsigned: " << HEX() << config.ALU_FORMAT_SPEC_REG0_SrcBUnsigned << "; ";
    DPRINT << "SrcA: " << HEX() << config.ALU_FORMAT_SPEC_REG0_SrcA << "; ";
    DPRINT << "SrcB: " << HEX() << config.ALU_FORMAT_SPEC_REG1_SrcB << "; ";
    DPRINT << "Dstacc: " << HEX() << config.ALU_FORMAT_SPEC_REG2_Dstacc << "; ";
    DPRINT << "Fp32_enabled: " << HEX() << config.ALU_ACC_CTRL_Fp32_enabled << "; ";
    DPRINT << "SFPU_Fp32_enabled: " << HEX() << config.ALU_ACC_CTRL_SFPU_Fp32_enabled << "; ";
    DPRINT << "INT8_math_enabled: " << HEX() << config.ALU_ACC_CTRL_INT8_math_enabled << "; ";
    DPRINT << ENDL();
}

inline void dprint_tensix_pack_relu_config() {
    // Only wormhole and blackhole have this register

    ckernel::packer::relu_config_t config = ckernel::packer::read_relu_config();

    DPRINT << "zero_flag_disabled_src: " << HEX() << config.ALU_ACC_CTRL_Zero_Flag_disabled_src << "; ";
    DPRINT << "zero_flag_disabled_dst: " << HEX() << config.ALU_ACC_CTRL_Zero_Flag_disabled_dst << "; ";
    DPRINT << "apply_relu: " << HEX() << config.STACC_RELU_ApplyRelu << "; ";
    DPRINT << "relu_threshold: " << config.STACC_RELU_ReluThreshold << "; ";
    DPRINT << "main: " << HEX() << config.DISABLE_RISC_BP_Disable_main << "; ";
    DPRINT << "trisc: " << HEX() << config.DISABLE_RISC_BP_Disable_trisc << "; ";
    DPRINT << "ncrisc: " << HEX() << config.DISABLE_RISC_BP_Disable_ncrisc << "; ";
    DPRINT << "bmp_clear_main: " << HEX() << config.DISABLE_RISC_BP_Disable_bmp_clear_main << "; ";
    DPRINT << "bmp_clear_trisc: " << HEX() << config.DISABLE_RISC_BP_Disable_bmp_clear_trisc << "; ";
    DPRINT << "bmp_clear_ncrisc: " << HEX() << config.DISABLE_RISC_BP_Disable_bmp_clear_ncrisc << "; ";
    DPRINT << ENDL();
}

// Printing dest control bits
inline void dprint_tensix_dest_rd_ctrl() {
    ckernel::packer::dest_rd_ctrl_t dest = ckernel::packer::read_dest_rd_ctrl();

    DPRINT << "read_32b_data: " << HEX() << dest.PCK_DEST_RD_CTRL_Read_32b_data << "; ";
    DPRINT << "read_unsigned: " << HEX() << dest.PCK_DEST_RD_CTRL_Read_unsigned << "; ";
    DPRINT << "read_int8: " << HEX() << dest.PCK_DEST_RD_CTRL_Read_int8 << "; ";
    DPRINT << "round_10b_mant: " << HEX() << dest.PCK_DEST_RD_CTRL_Round_10b_mant << "; ";
    DPRINT << "reserved: " << HEX() << dest.PCK_DEST_RD_CTRL_Reserved << "; ";
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

    ckernel::packer::pck_edge_offset_t edge = ckernel::packer::read_pck_edge_offset(reg_addr, cfg);

    DPRINT << "mask: " << HEX() << edge.mask << "; ";
    DPRINT << "mode: " << HEX() << edge.mode << "; ";
    DPRINT << "tile_row_set_select_pack0: " << HEX() << edge.tile_row_set_select_pack0 << "; ";
    DPRINT << "tile_row_set_select_pack1: " << HEX() << edge.tile_row_set_select_pack1 << "; ";
    DPRINT << "tile_row_set_select_pack2: " << HEX() << edge.tile_row_set_select_pack2 << "; ";
    DPRINT << "tile_row_set_select_pack3: " << HEX() << edge.tile_row_set_select_pack3 << "; ";
    DPRINT << "reserved: " << HEX() << edge.reserved << "; ";
}

// Choose what register you want printed with reg_id (1-4), 0 for all
inline void dprint_tensix_pck_edge_offset(uint reg_id = 0) {
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

    ckernel::packer::pack_counters_t counters = ckernel::packer::read_pack_counters(reg_addr, cfg);

    DPRINT << "pack_per_xy_plane: " << counters.pack_per_xy_plane << "; ";
    DPRINT << "pack_reads_per_xy_plane: " << counters.pack_reads_per_xy_plane << "; ";
    DPRINT << "pack_xys_per_til: " << counters.pack_xys_per_til << "; ";
    DPRINT << "pack_yz_transposed: " << HEX() << counters.pack_yz_transposed << "; ";
    DPRINT << "pack_per_xy_plane_offset: " << counters.pack_per_xy_plane_offset << "; ";
}

// Choose what register you want printed with reg_id (1-4), 0 for all
inline void dprint_tensix_pack_counters(uint reg_id = 0) {
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

// Choose what register you want printed with reg_id (1-4), ALL not implemented due to dprint buffer size restriction
inline void dprint_tensix_pack_config(uint reg_id = 1) {
    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    if (reg_id) {
        uint32_t reg_addr = 0;
        switch (reg_id) {
            case 1: reg_addr = THCON_SEC0_REG1_Row_start_section_size_ADDR32; break;
            case 2: reg_addr = THCON_SEC0_REG8_Row_start_section_size_ADDR32; break;
            case 3: reg_addr = THCON_SEC1_REG1_Row_start_section_size_ADDR32; break;
            case 4: reg_addr = THCON_SEC1_REG8_Row_start_section_size_ADDR32; break;
            default: DPRINT << "Aborting! Invalid register id (valid ids are between 1 and 4)" << ENDL(); return;
        }

        DPRINT << "REG_ID: " << reg_id << " ";

#ifdef ARCH_GRAYSKULL
        dprint_tensix_pack_config_grayskull(reg_addr, cfg);
#elif ARCH_WORMHOLE
        dprint_tensix_pack_config_wormhole(reg_addr, cfg);
#else
        dprint_tensix_pack_config_blackhole(reg_addr, cfg);
#endif
    }
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
    dprint_tensix_struct_field(word, 0xfff, 0, "x_stride", true); // decimal
    dprint_tensix_struct_field(word, 0xfff000, 12, "y_stride", true); // decimal

    // word 1 zw_stride
    word = cfg[reg_addr + 1];
    dprint_tensix_struct_field(word, 0xfff, 0, "z_stride", true); // decimal
    dprint_tensix_struct_field(word, 0xffff000, 12, "w_stride", true); // decimal
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
