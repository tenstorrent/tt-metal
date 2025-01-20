// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <array>

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

// Dprints data format as string given an uint
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
    else {
        DPRINT << "0x" << HEX();
    }
    DPRINT << ((word & mask) >> shamt) << ENDL();
}

// NOTE: FUNCTIONS WITHOUT ARCH NAME (GRAYSKULL, WORMHOLE, BLACKHOLE) AND WITHOUT HELPER SUFIX ARE INTENDED TO BE USE

// UNPACK TILE DESCRIPTOR

// These function's argument should be return value of read_unpack_tile_descriptor()

inline void dprint_tensix_unpack_tile_descriptor_in_data_format(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    dprint_data_format(tile_descriptor.in_data_format);
    DPRINT << ENDL();
}

inline void dprint_tensix_unpack_tile_descriptor_uncompressed(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << "0x" << HEX() << tile_descriptor.uncompressed << ENDL();
}

inline void dprint_tensix_unpack_tile_descriptor_reserved_0(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << "0x" << HEX() << tile_descriptor.reserved_0 << ENDL();
}

inline void dprint_tensix_unpack_tile_descriptor_blobs_per_xy_plane(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << DEC() << tile_descriptor.blobs_per_xy_plane << ENDL();
}

inline void dprint_tensix_unpack_tile_descriptor_reserved_1(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << "0x" << HEX() << tile_descriptor.reserved_1 << ENDL();
}

inline void dprint_tensix_unpack_tile_descriptor_x_dim(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << DEC() << tile_descriptor.x_dim << ENDL();
}

inline void dprint_tensix_unpack_tile_descriptor_y_dim(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << DEC() << tile_descriptor.y_dim << ENDL();
}

inline void dprint_tensix_unpack_tile_descriptor_z_dim(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << DEC() << tile_descriptor.z_dim << ENDL();
}

inline void dprint_tensix_unpack_tile_descriptor_w_dim(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << DEC() << tile_descriptor.w_dim << ENDL();
}

inline void dprint_tensix_unpack_tile_descriptor_blobs_y_start(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
#ifdef ARCH_GRAYSKULL
    DPRINT << DEC() << tile_descriptor.blobs_y_start << ENDL();
#else
    DPRINT << DEC() << ((tile_descriptor.blobs_y_start_hi << 16) | tile_descriptor.blobs_y_start_lo) << ENDL();
#endif
}

inline void dprint_tensix_unpack_tile_descriptor_digest_type(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << "0x" << HEX() << tile_descriptor.digest_type << ENDL();
}

inline void dprint_tensix_unpack_tile_descriptor_digest_size(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << DEC() << tile_descriptor.digest_size << ENDL();
}

// UNPACK CONFIG

// These function's argument should be return value of read_unpack_config()

inline void dprint_tensix_unpack_config_out_data_format(const ckernel::unpacker::unpack_config_t& config) {
    dprint_data_format(config.out_data_format);
    DPRINT << ENDL();
}

inline void dprint_tensix_unpack_config_throttle_mode(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.throttle_mode << ENDL();
}

inline void dprint_tensix_unpack_config_context_count(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.context_count << ENDL();
}

inline void dprint_tensix_unpack_config_haloize_mode(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.haloize_mode << ENDL();
}

inline void dprint_tensix_unpack_config_tileize_mode(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.tileize_mode << ENDL();
}

inline void dprint_tensix_unpack_config_force_shared_exp(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.force_shared_exp << ENDL();
}

#ifdef ARCH_GRAYSKULL
inline void dprint_tensix_unpack_config_reserved_0(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_0 << ENDL();
}
#endif

inline void dprint_tensix_unpack_config_upsample_rate(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << DEC() << config.upsample_rate << ENDL();
}

inline void dprint_tensix_unpack_config_upsample_and_interlave(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.upsamle_and_interlave << ENDL();
}

inline void dprint_tensix_unpack_config_shift_amount(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << DEC() << config.shift_amount << ENDL();
}

inline void dprint_tensix_unpack_config_uncompress_cntx0_3(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.uncompress_cntx0_3 << ENDL();
}

inline void dprint_tensix_unpack_config_reserved_1(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_1 << ENDL();
}

inline void dprint_tensix_unpack_config_uncompress_cntx4_7(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.uncompress_cntx4_7 << ENDL();
}

inline void dprint_tensix_unpack_config_reserved_2(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_2 << ENDL();
}

inline void dprint_tensix_unpack_config_limit_addr(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.limit_addr << ENDL();
}

inline void dprint_tensix_unpack_config_fifo_size(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << DEC() << config.fifo_size << ENDL();
}

#ifndef ARCH_GRAYSKULL  // ARCH_WORMHOLE OR ARCH_BLACKHOLE
inline void dprint_tensix_unpack_config_unpack_src_reg_set_update(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.unpack_src_reg_set_update << ENDL();
}

inline void dprint_tensix_unpack_config_unpack_if_sel(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.unpack_if_sel << ENDL();
}

inline void dprint_tensix_unpack_config_unpack_if_sel_cntx0_3(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.unpack_if_sel_cntx0_3 << ENDL();
}

inline void dprint_tensix_unpack_config_unpack_if_sel_cntx4_7(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.unpack_if_sel_cntx4_7 << ENDL();
}

inline void dprint_tensix_unpack_config_reserved_3(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_3 << ENDL();
}

inline void dprint_tensix_unpack_config_reserved_4(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_4 << ENDL();
}

inline void dprint_tensix_unpack_config_reserved_5(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_5 << ENDL();
}
#endif

// PACK CONFIG

// // These function's argument should be return value of read_pack_config()

inline void dprint_tensix_pack_config_row_ptr_section_size(const ckernel::packer::pack_config_t& config) {
    DPRINT << DEC() << config.row_ptr_section_size << ENDL();
}

inline void dprint_tensix_pack_config_exp_section_size(const ckernel::packer::pack_config_t& config) {
    DPRINT << DEC() << config.exp_section_size << ENDL();
}

inline void dprint_tensix_pack_config_l1_dest_addr(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.l1_dest_addr << ENDL();
}

inline void dprint_tensix_pack_config_uncompressed(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.uncompress << ENDL();
}

inline void dprint_tensix_pack_config_add_l1_dest_addr_offset(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.add_l1_dest_addr_offset << ENDL();
}

inline void dprint_tensix_pack_config_reserved_0(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_0 << ENDL();
}

inline void dprint_tensix_pack_config_out_data_format(const ckernel::packer::pack_config_t& config) {
    dprint_data_format(config.out_data_format);
    DPRINT << ENDL();
}

inline void dprint_tensix_pack_config_in_data_format(const ckernel::packer::pack_config_t& config) {
    dprint_data_format(config.in_data_format);
    DPRINT << ENDL();
}

#ifndef ARCH_BLACKHOLE  // ARCH_GRAYSKULL OR ARCH_WORMHOLE
inline void dprint_tensix_pack_config_reserved_1(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_1 << ENDL();
}
#endif

inline void dprint_tensix_pack_config_src_if_sel(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.src_if_sel << ENDL();
}

#ifndef ARCH_BLACKHOLE  // ARCH_GRAYSKULL OR ARCH_WORMHOLE
inline void dprint_tensix_pack_config_pack_per_xy_plane(const ckernel::packer::pack_config_t& config) {
    DPRINT << DEC() << config.pack_per_xy_plane << ENDL();
}
#endif

inline void dprint_tensix_pack_config_l1_src_addr(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.l1_src_addr << ENDL();
}

#ifndef ARCH_BLACKHOLE  // ARCH_GRAYSKULL OR ARCH_WORMHOLE
inline void dprint_tensix_pack_config_downsample_mask(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.downsample_mask << ENDL();
}

inline void dprint_tensix_pack_config_downsample_shift_count(const ckernel::packer::pack_config_t& config) {
    DPRINT << DEC() << config.downsample_shift_count << ENDL();
}

inline void dprint_tensix_pack_config_read_mode(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.read_mode << ENDL();
}

inline void dprint_tensix_pack_config_exp_threshold_en(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.exp_threshold_en << ENDL();
}

inline void dprint_tensix_pack_config_reserved_2(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_2 << ENDL();
}

inline void dprint_tensix_pack_config_exp_threshold(const ckernel::packer::pack_config_t& config) {
    DPRINT << DEC() << config.exp_threshold << ENDL();
}
#endif

#ifdef ARCH_WORMHOLE
inline void dprint_tensix_pack_config_l1_acc_disable_pack_zero_flag(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.pack_l1_acc_disable_pack_zero_flag << ENDL();
}
#endif

#ifdef ARCH_BLACKHOLE
inline void dprint_tensix_pack_config_disable_pack_zero_flag(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.disable_pack_zero_flag << ENDL();
}

inline void dprint_tensix_pack_config_dis_shared_exp_assembler(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.dis_shared_exp_assembler << ENDL();
}

inline void dprint_tensix_pack_config_auto_set_last_pacr_intf_sel(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.auto_set_last_pacr_intf_sel << ENDL();
}

inline void dprint_tensix_pack_config_enable_out_fifo(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.enable_out_fifo << ENDL();
}

inline void dprint_tensix_pack_config_sub_l1_tile_header_size(const ckernel::packer::pack_config_t& config) {
    DPRINT << DEC() << config.sub_l1_tile_header_size << ENDL();
}

inline void dprint tensix_pack_config_pack_start_intf_pos(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.pack_start_intf_pos << ENDL();
}

inline void dprint_tensix_pack_config_all_pack_disable_zero_compress_ovrd(
    const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.all_pack_disable_zero_compress_ovrd << ENDL();
}

inline void dprint_tensix_pack_config_add_tile_header_size(const ckernel::packer::pack_config_t& config) {
    DPRINT << DEC() << config.add_tile_header_size << ENDL();
}

inline void dprint_tensix_pack_config_pack_dis_y_pos_start_offset(const ckernel::packer::pack_config_t& config) {
    DPRINT < "0x" << HEX() << config.pack_dis_y_pos_start_offset << ENDL();
}
#endif

// HARDWARE SPECIFIC FUNCTIONS

#ifdef ARCH_GRAYSKULL
inline void dprint_tensix_unpack_tile_descriptor_grayskull_helper(
    const ckernel::unpacker::tile_descriptor_t& tile_descriptor) {
    DPRINT << "in_data_format: ";
    dprint_tensix_unpack_tile_descriptor_in_data_format(tile_descriptor);
    DPRINT << "uncompressed: ";
    dprint_tensix_unpack_tile_descriptor_uncompressed(tile_descriptor);
    DPRINT << "reserved_0: ";
    dprint_tensix_unpack_tile_descriptor_reserved_0(tile_descriptor);
    DPRINT << "blobs_per_xy_plane: " dprint_tensix_unpack_tile_descriptor_blobs_per_xy_plane(tile_descriptor);
    DPRINT << "reserved_1: ";
    dprint_tensix_unpack_tile_descriptor_reserved_1(tile_descriptor);
    DPRINT << "x_dim: ";
    dprint_tensix_unpack_tile_descriptor_x_dim(tile_descriptor);
    DPRINT << "y_dim: ";
    dprint_tensix_unpacK_tile_descriptor_y_dim(tile_descriptor);
    DPRINT << "z_dim: ";
    dprint_tensix_unpack_tile_descriptor_z_dim(tile_descriptor);
    DPRINT << "w_dim: ";
    dprint_tensix_unpack_tile_descriptor_w_dim(tile_descriptor);
    DPRINT << "blobs_y_start: ";
    dprint_tensix_unpack_tile_descriptor_blobs_y_start(tile_descriptor);
    DPRINT << "digest_type: ";
    dprint_tensix_unpack_tile_descriptor_digest_type(tile_descriptor);
    DPRINT << "digest_size: ";
    dprint_tensix_unpack_tile_descriptor_digest_type(tile_descriptor);
}

inline void dprint_tensix_unpack_tile_descriptor_grayskull(uint reg_id) {
    std::array<ckernel::unpacker::unpack_tile_descriptor_t, 2> tile_descriptor_vec;
    tile_descriptor_vec = ckernel::unpacker::read_unpack_tile_descriptor();
    if (reg_id >= 1 && reg_id <= 2) {
        DPRINT << "REG_ID: " << reg_id << ENDL();
        dprint_tensix_unpack_tile_descriptor_grayskull_helper(tile_descriptor_vec[reg_id - 1]);
    } else if (reg_id == 0) {
        for (uint i = 1; i <= 2; i++) {
            DPRINT << "REG_ID: " << i << ENDL();
            dprint_tensix_unpack_tile_descriptor_grayskull_helper(tile_descriptor_vec[i - 1]);
            if (i != 2) {
                DPRINT << ENDL();
            }
        }
    } else {
        DPRINT << "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND 2." << ENDL();
    }
}

inline void dprint_tensix_unpack_config_grayskull_helper(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "out_data_format: ";
    dprint_tensix_unpack_config_out_data_format(config);
    DPRINT << "throttle_mode: ";
    dprint_tensix_unpack_config_throttle_mode(config);
    DPRINT << "context_count: ";
    dprint_tensix_unpack_config_context_count(config);
    DPRINT << "haloize_mode: ";
    dprint_tensix_unpack_config_haloize_mode(config);
    DPRINT << "tileize_mode: ";
    dprint_tensix_unpack_config_tileize_mode(config);
    DPRINT << "force_shared_exp: ";
    dprint_tensix_unpack_config_force_shared_exp(config) DPRINT << "reserved_0: ";
    dprint_tensix_unpack_config_reserved_0(config);
    DPRINT << "upsample_rate: ";
    dprint_tensix_unpack_config_upsample_rate(config);
    DPRINT << "upsamle_and_interlave: ";
    dprint_tensix_unpack_config_upsample_and_interlave(config);
    DPRINT << "shift_amount: ";
    dprint_tensix_unpack_config_shift_amount(config);
    DPRINT << "uncompress_cntx0_3: ";
    dprint_tensix_unpack_config_uncompress_cntx0_3(config);
    DPRINT << "reserved_1: ";
    dprint_tensix_unpack_config_reserved_1(config);
    DPRINT << "uncompress_cntx4_7: ";
    dprint_tensix_unpack_config_uncompress_cntx4_7(config);
    DPRINT << "reserved_2: ";
    dprint_tensix_unpack_config_reserved_2(config);
    DPRINT << "limit_addr: ";
    dprint_tensix_unpack_config_limit_addr(config);
    DPRINT << "fifo_size: ";
    dprint_tensix_unpack_config_fifo_size(config);
}

inline void dprint_tensix_unpack_config_grayskull() {
    std::array<ckernel::unpacker::unpack_config_t, 2> config_vec;
    config_vec = ckernel::unpacker::read_unpack_config();
    if (reg_id >= 1 && reg_id <= 2) {
        DPRINT << "REG_ID: " << reg_id << ENDL();
        dprint_tensix_unpack_config_grayskull_helper(config_vec[reg_id - 1]);
    } else if (reg_id == 0) {
        for (uint i = 1; i <= 2; i++) {
            DPRINT << "REG_ID: " << i << ENDL();
            dprint_tensix_unpack_config_grayskull_helper(config_vec[i - 1]);
            if (i != 2) {
                DPRINT << ENDL();
            }
        }
    } else {
        DPRINT << "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND 2." << ENDL();
    }
}

inline void dprint_tensix_pack_config_grayskull(const ckernel::packer::pack_config_t& config) {
    DPRINT << "row_ptr_section_size: ";
    dprint_tensix_pack_config_row_ptr_section_size(config);
    DPRINT << "exp_section_size: ";
    dprint_tensix_pack_config_exp_section_size(config);
    DPRINT << "l1_dest_addr: ";
    dprint_tensix_pack_config_l1_dest_addr(config);
    DPRINT << "uncompress: ";
    dprint_tensix_pack_config_uncompress(config);
    DPRINT << "add_l1_dest_addr_offset: ";
    dprint_tensix_pack_config_add_l1_dest_addr_offset(config);
    DPRINT << "reserved_0: ";
    dprint_tensix_pack_config_reserved_0(config);
    DPRINT << "out_data_format: ";
    dprint_tensix_pack_config_out_data_format(config);
    DPRINT << "in_data_format: ";
    dprint_tensix_pack_config_in_data_format(config);
    DPRINT << "reserved_1: ";
    dprint_tensix_pack_config_reserved_1(config);
    DPRINT << "src_if_sel: ";
    dprint_tensix_pack_config_src_if_sel(config);
    DPRINT << "pack_per_xy_plane: ";
    dprint_tensix_pack_config_pack_per_xy_plane(config);
    DPRINT << "l1_src_addr: ";
    dprint_tensix_pack_conifg_l1_src_addr(config);
    DPRINT << "downsample_mask: ";
    dprint_tensix_pack_config_downsample_mask(config);
    DPRINT << "downsample_shift_count: ";
    dprint_tensix_pack_config_downsample_shift_count(config);
    DPRINT << "read_mode: ";
    dprint_tensix_pack_config_read_mode(config);
    DPRINT << "exp_threshold_en: ";
    dprint_tensix_pack_config_exp_threshold_en(config);
    DPRINT << "reserved_2: ";
    dprint_tensix_pack_config_reserved_2(config);
    DPRINT << "exp_threshold: ";
    dprint_tensix_pack_config_exp_threshold(config);
}
#else  // ARCH_WORMHOLE or ARCH_BLACKHOLE
inline void dprint_tensix_unpack_tile_descriptor_wormhole_or_blackhole_helper(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << "in_data_format: ";
    dprint_tensix_unpack_tile_descriptor_in_data_format(tile_descriptor);
    DPRINT << "uncompressed: ";
    dprint_tensix_unpack_tile_descriptor_uncompressed(tile_descriptor);
    DPRINT << "reserved_0: ";
    dprint_tensix_unpack_tile_descriptor_reserved_0(tile_descriptor);
    DPRINT << "blobs_per_xy_plane: ";
    dprint_tensix_unpack_tile_descriptor_blobs_per_xy_plane(tile_descriptor);
    DPRINT << "reserved_1: ";
    dprint_tensix_unpack_tile_descriptor_reserved_1(tile_descriptor);
    DPRINT << "x_dim: ";
    dprint_tensix_unpack_tile_descriptor_x_dim(tile_descriptor);
    DPRINT << "y_dim: ";
    dprint_tensix_unpack_tile_descriptor_y_dim(tile_descriptor);
    DPRINT << "z_dim: ";
    dprint_tensix_unpack_tile_descriptor_z_dim(tile_descriptor);
    DPRINT << "w_dim: ";
    dprint_tensix_unpack_tile_descriptor_w_dim(tile_descriptor);
    DPRINT << "blobs_y_start: ";
    dprint_tensix_unpack_tile_descriptor_blobs_y_start(tile_descriptor);
    DPRINT << "digest_type: ";
    dprint_tensix_unpack_tile_descriptor_digest_type(tile_descriptor);
    DPRINT << "digest_size: ";
    dprint_tensix_unpack_tile_descriptor_digest_size(tile_descriptor);
}

// Choose which register you want (1-2). 0 for both.
inline void dprint_tensix_unpack_tile_descriptor_wormhole_or_blackhole(uint reg_id) {
    std::array<ckernel::unpacker::unpack_tile_descriptor_t, 2> tile_descriptor_vec;
    tile_descriptor_vec = ckernel::unpacker::read_unpack_tile_descriptor();
    if (reg_id >= 1 && reg_id <= 2) {
        DPRINT << "REG_ID: " << reg_id << ENDL();
        dprint_tensix_unpack_tile_descriptor_wormhole_or_blackhole_helper(tile_descriptor_vec[reg_id - 1]);
    } else if (reg_id == 0) {
        for (uint i = 1; i <= 2; i++) {
            DPRINT << "REG_ID: " << i << ENDL();
            dprint_tensix_unpack_tile_descriptor_wormhole_or_blackhole_helper(tile_descriptor_vec[i - 1]);
            if (i != 2) {
                DPRINT << ENDL();
            }
        }
    } else {
        DPRINT << "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND 2." << ENDL();
    }
}

inline void dprint_tensix_unpack_config_wormhole_or_blackhole_helper(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "out_data_format: ";
    dprint_tensix_unpack_config_out_data_format(config);
    DPRINT << "throttle_mode: ";
    dprint_tensix_unpack_config_throttle_mode(config);
    DPRINT << "context_count: ";
    dprint_tensix_unpack_config_context_count(config);
    DPRINT << "haloize_mode: ";
    dprint_tensix_unpack_config_haloize_mode(config);
    DPRINT << "tileize_mode: ";
    dprint_tensix_unpack_config_tileize_mode(config);
    DPRINT << "unpack_src_reg_set_update: ";
    dprint_tensix_unpack_config_unpack_src_reg_set_update(config);
    DPRINT << "unpack_if_sel: ";
    dprint_tensix_unpack_config_unpack_if_sel(config);
    DPRINT << "upsample_rate: ";
    dprint_tensix_unpack_config_upsample_rate(config);
    DPRINT << "reserved_1: ";
    dprint_tensix_unpack_config_reserved_1(config);
    DPRINT << "upsample_and_interlave: ";
    dprint_tensix_unpack_config_upsample_and_interlave(config);
    DPRINT << "shift_amount: ";
    dprint_tensix_unpack_config_shift_amount(config);
    DPRINT << "uncompress_cntx0_3: ";
    dprint_tensix_unpack_config_uncompress_cntx0_3(config);
    DPRINT << "unpack_if_sel_cntx0_3: ";
    dprint_tensix_unpack_config_unpack_if_sel_cntx0_3(config);
    DPRINT << "force_shared_exp: ";
    dprint_tensix_unpack_config_force_shared_exp(config);
    DPRINT << "reserved_2: ";
    dprint_tensix_unpack_config_reserved_2(config);
    DPRINT << "uncompress_cntx4_7: ";
    dprint_tensix_unpack_config_uncompress_cntx4_7(config);
    DPRINT << "unpack_if_sel_cntx4_7: ";
    dprint_tensix_unpack_config_unpack_if_sel_cntx4_7(config);
    DPRINT << "reserved_3: ";
    dprint_tensix_unpack_config_reserved_3(config);
    DPRINT << "limit_addr: ";
    dprint_tensix_unpack_config_limit_addr(config);
    DPRINT << "reserved_4: ";
    dprint_tensix_unpack_config_reserved_4(config);
    DPRINT << "fifo_size: ";
    dprint_tensix_unpack_config_fifo_size(config);
    DPRINT << "reserved_5: ";
    dprint_tensix_unpack_config_reserved_5(config);
}

// Choose which register you want (1-2). 0 for both.
inline void dprint_tensix_unpack_config_wormhole_or_blackhole(uint reg_id) {
    std::array<ckernel::unpacker::unpack_config_t, 2> config_vec;
    config_vec = ckernel::unpacker::read_unpack_config();
    if (reg_id >= 1 && reg_id <= 2) {
        DPRINT << "REG_ID: " << reg_id << ENDL();
        dprint_tensix_unpack_config_wormhole_or_blackhole_helper(config_vec[reg_id - 1]);
    } else if (reg_id == 0) {
        for (uint i = 1; i <= 2; i++) {
            DPRINT << "REG_ID: " << i << ENDL();
            dprint_tensix_unpack_config_wormhole_or_blackhole_helper(config_vec[i - 1]);
            if (i != 2) {
                DPRINT << ENDL();
            }
        }
    } else {
        DPRINT << "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND 2." << ENDL();
    }
}

#ifdef ARCH_WORMHOLE
inline void dprint_tensix_pack_config_wormhole(const ckernel::packer::pack_config_t& config) {
    DPRINT << "row_ptr_section_size: ";
    dprint_tensix_pack_config_row_ptr_section_size(config);
    DPRINT << "exp_section_size: ";
    dprint_tensix_pack_config_exp_section_size(config);
    DPRINT << "l1_dest_addr: ";
    dprint_tensix_pack_config_l1_dest_addr(config);
    DPRINT << "uncompress: ";
    dprint_tensix_pack_config_uncompressed(config);
    DPRINT << "add_l1_dest_addr_offset: ";
    dprint_tensix_pack_config_add_l1_dest_addr_offset(config);
    DPRINT << "reserved_0: ";
    dprint_tensix_pack_config_reserved_0(config);
    DPRINT << "out_data_format: ";
    dprint_tensix_pack_config_out_data_format(config);
    DPRINT << "in_data_format: ";
    dprint_tensix_pack_config_in_data_format(config);
    DPRINT << "reserved_1: ";
    dprint_tensix_pack_config_reserved_1(config);
    DPRINT << "src_if_sel: ";
    dprint_tensix_pack_config_src_if_sel(config);
    DPRINT << "pack_per_xy_plane: ";
    dprint_tensix_pack_config_pack_per_xy_plane(config);
    DPRINT << "l1_src_addr: ";
    dprint_tensix_pack_config_l1_src_addr(config);
    DPRINT << "downsample_mask: ";
    dprint_tensix_pack_config_downsample_mask(config);
    DPRINT << "downsample_shift_count: ";
    dprint_tensix_pack_config_downsample_shift_count(config);
    DPRINT << "read_mode: ";
    dprint_tensix_pack_config_read_mode(config);
    DPRINT << "exp_threshold_en: ";
    dprint_tensix_pack_config_exp_threshold_en(config);
    DPRINT << "pack_l1_acc_disable_pack_zero_flag: ";
    dprint_tensix_pack_config_l1_acc_disable_pack_zero_flag(config);
    DPRINT << "reserved_2: ";
    dprint_tensix_pack_config_reserved_2(config);
    DPRINT << "exp_threshold: ";
    dprint_tensix_pack_config_exp_threshold(config);
}
#endif  // ARCH_WORMHOLE

#ifdef ARCH_BLACKHOLE
inline void dprint_tensix_pack_config_blackhole(const ckernel::packer::pack_config_t& config) {
    DPRINT << "row_ptr_section_size: ";
    dprint_tensix_pack_config_row_ptr_section_size(config);
    DPRINT << "exp_section_size: ";
    dprint_tensix_pack_config_exp_section_size(config);
    DPRINT << "l1_dest_addr: ";
    dprint_tensix_pack_config_l1_dest_addr(config);
    DPRINT << "uncompress: ";
    dprint_tensix_pack_config_uncompress(config);
    DPRINT << "add_l1_dest_addr_offset: ";
    dprint_tensix_pack_config_add_l1_dest_addr_offset(config);
    DPRINT << "disable_pack_zero_flag: ";
    dprint_tensix_pack_config_disable_pack_zero_flag(config);
    DPRINT << "reserved_0: ";
    dprint_tensix_pack_config_reserved_0(config);
    DPRINT << "out_data_format: ";
    dprint_tensix_pack_config_out_data_format(config);
    DPRINT << "in_data_format: ";
    dprint_tensix_pack_config_in_data_format(config);
    DPRINT << "dis_shared_exp_assembler: ";
    dprint_tensix_pack_config_dis_shared_exp_assembler(config);
    DPRINT << "auto_set_last_pacr_intf_sel: ";
    dprint_tensix_pack_config_auto_set_last_pacr_intf_sel(config);
    DPRINT << "enable_out_fifo: ";
    dprint_tensix_pack_config_enable_out_fifo(config);
    DPRINT << "sub_l1_tile_header_size: ";
    dprint_tensix_pack_config_sub_l1_tile_header_size(config);
    DPRINT << "src_if_sel: ";
    dprint_tensix_pack_config_src_if_sel(config);
    DPRINT << "pack_start_intf_pos: ";
    dprint_tensix_pack_config_pack_start_intf_pos(config);
    DPRINT << "all_pack_disable_zero_compress_ovrd: ";
    dprint_tensix_pack_config_all_pack_disable_zero_compress_ovrd(config);
    DPRINT << "add_tile_header_size: ";
    dprint_tensix_pack_config_add_tile_header_size(config);
    DPRINT << "pack_dis_y_pos_start_offset: ";
    dprint_tensix_pack_config_pack_dis_y_pos_start_offset(config);
    DPRINT << "l1_src_addr: ";
    dprint_tensix_pack_config_l1_src_addr(config);
}
#endif  // ARCH_BLACKHOLE

// PCK_EDGE_OFFSET

// These function's argument should be return value of read_pack_edge_offset()

inline void dprint_tensix_pack_edge_offset_mask(const ckernel::packer::pck_edge_offset_t& edge) {
    DPRINT << "0x" << HEX() << edge.mask << ENDL();
}

inline void dprint_tensix_pack_edge_offset_mode(const ckernel::packer::pck_edge_offset_t& edge) {
    DPRINT << "0x" << HEX() << edge.mode << ENDL();
}

inline void dprint_tensix_pack_edge_offset_tile_row_set_select_pack0(const ckernel::packer::pck_edge_offset_t& edge) {
    DPRINT << "0x" << HEX() << edge.tile_row_set_select_pack0 << ENDL();
}

inline void dprint_tensix_pack_edge_offset_tile_row_set_select_pack1(const ckernel::packer::pck_edge_offset_t& edge) {
    DPRINT << "0x" << HEX() << edge.tile_row_set_select_pack1 << ENDL();
}

inline void dprint_tensix_pack_edge_offset_tile_row_set_select_pack2(const ckernel::packer::pck_edge_offset_t& edge) {
    DPRINT << "0x" << HEX() << edge.tile_row_set_select_pack2 << ENDL();
}

inline void dprint_tensix_pack_edge_offset_tile_row_set_select_pack3(const ckernel::packer::pck_edge_offset_t& edge) {
    DPRINT << "0x" << HEX() << edge.tile_row_set_select_pack3 << ENDL();
}

inline void dprint_tensix_pack_edge_offset_reserved(const ckernel::packer::pck_edge_offset_t& edge) {
    DPRINT << "0x" << HEX() << edge.reserved << ENDL();
}

// Printing packer edge offset
inline void dprint_tensix_pack_edge_offset_helper(const ckernel::packer::pck_edge_offset_t& edge) {
    DPRINT << "mask: ";
    dprint_tensix_pack_edge_offset_mask(edge);
    DPRINT << "mode: ";
    dprint_tensix_pack_edge_offset_mode(edge);
    DPRINT << "tile_row_set_select_pack0: ";
    dprint_tensix_pack_edge_offset_tile_row_set_select_pack0(edge);
    DPRINT << "tile_row_set_select_pack1: ";
    dprint_tensix_pack_edge_offset_tile_row_set_select_pack1(edge);
    DPRINT << "tile_row_set_select_pack2: ";
    dprint_tensix_pack_edge_offset_tile_row_set_select_pack2(edge);
    DPRINT << "tile_row_set_select_pack3: ";
    dprint_tensix_pack_edge_offset_tile_row_set_select_pack3(edge);
    DPRINT << "reserved: ";
    dprint_tensix_pack_edge_offset_reserved(edge);
}

// PACK COUNTERS

// These functions' argument should be return value of read_pack_counters()

inline void dprint_tensix_pack_counters_pack_per_xy_plane(const ckernel::packer::pack_counters_t& counters) {
    DPRINT << DEC() << counters.pack_per_xy_plane << ENDL();
}

inline void dprint_tensix_pack_counters_pack_reads_per_xy_plane(const ckernel::packer::pack_counters_t& counters) {
    DPRINT << DEC() << counters.pack_reads_per_xy_plane << ENDL();
}

inline void dprint_tensix_pack_counters_pack_xys_per_til(const ckernel::packer::pack_counters_t& counters) {
    DPRINT << DEC() << counters.pack_xys_per_til << ENDL();
}

inline void dprint_tensix_pack_counters_pack_yz_transposed(const ckernel::packer::pack_counters_t& counters) {
    DPRINT << "0x" << HEX() << counters.pack_yz_transposed << ENDL();
}

inline void dprint_tensix_pack_counters_pack_per_xy_plane_offset(const ckernel::packer::pack_counters_t& counters) {
    DPRINT << DEC() << counters.pack_per_xy_plane_offset << ENDL();
}

// Printing packer counters
inline void dprint_tensix_pack_counters_helper(const ckernel::packer::pack_counters_t& counters) {
    DPRINT << "pack_per_xy_plane: ";
    dprint_tensix_pack_counters_pack_per_xy_plane(counters);
    DPRINT << "pack_reads_per_xy_plane: ";
    dprint_tensix_pack_counters_pack_reads_per_xy_plane(counters);
    DPRINT << "pack_xys_per_til: ";
    dprint_tensix_pack_counters_pack_xys_per_til(counters);
    DPRINT << "pack_yz_transposed: ";
    dprint_tensix_pack_counters_pack_yz_transposed(counters);
    DPRINT << "pack_per_xy_plane_offset: ";
    dprint_tensix_pack_counters_pack_per_xy_plane_offset(counters);
}

// PACK STRIDES

inline void dprint_tensix_pack_strides_x_stride(const uint32_t& word) {
    dprint_tensix_struct_field(word, 0xfff, 0, "x_stride", true);  // decimal
}

inline void dprint_tensix_pack_strides_y_stride(const uint32_t& word) {
    dprint_tensix_struct_field(word, 0xfff000, 12, "y_stride", true);  // decimal
}

inline void dprint_tensix_pack_strides_z_stride(const uint32_t& word) {
    dprint_tensix_struct_field(word, 0xfff, 0, "z_stride", true);  // decimal
}

inline void dprint_tensix_pack_strides_w_stride(const uint32_t& word) {
    dprint_tensix_struct_field(word, 0xffff000, 12, "w_stride", true);  // decimal
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
    dprint_tensix_pack_strides_x_stride(word);
    dprint_tensix_pack_strides_y_stride(word);

    // word 1 zw_stride
    word = cfg[reg_addr + 1];
    dprint_tensix_pack_strides_z_stride(word);
    dprint_tensix_pack_strides_w_stride(word);
}

// ALU CONFIG

// These functions' argument should be return value of read_alu_config()

inline void dprint_tensix_alu_config_alu_rounding_mode_fpu_srnd_en(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ROUNDING_MODE_Fpu_srnd_en << ENDL();
}

inline void dprint_tensix_alu_config_alu_rounding_mode_gasket_srnd_en(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ROUNDING_MODE_Gasket_srnd_en << ENDL();
}

inline void dprint_tensix_alu_config_alu_rounding_mode_packer_srnd_en(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ROUNDING_MODE_Packer_srnd_en << ENDL();
}

inline void dprint_tensix_alu_config_alu_rounding_mode_padding(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ROUNDING_MODE_Padding << ENDL();
}

inline void dprint_tensix_alu_config_alu_rounding_mode_gs_lf(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ROUNDING_MODE_GS_LF << ENDL();
}

inline void dprint_tensix_alu_config_alu_rounding_mode_bfp8_hf(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ROUNDING_MODE_Bfp8_HF << ENDL();
}

inline void dprint_tensix_alu_config_alu_format_spec_reg0_srcaunsigned(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_FORMAT_SPEC_REG0_SrcAUnsigned << ENDL();
}

inline void dprint_tensix_alu_config_alu_format_spec_reg0_srcbunsigned(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_FORMAT_SPEC_REG0_SrcBUnsigned << ENDL();
}

inline void dprint_tensix_alu_config_alu_format_spec_reg0_srca(const ckernel::unpacker::alu_config_t& config) {
    dprint_data_format(config.ALU_FORMAT_SPEC_REG0_SrcA);
    DPRINT << ENDL();
}

inline void dprint_tensix_alu_config_alu_format_spec_reg1_srcb(const ckernel::unpacker::alu_config_t& config) {
    dprint_data_format(config.ALU_FORMAT_SPEC_REG1_SrcB);
    DPRINT << ENDL();
}

inline void dprint_tensix_alu_config_alu_format_spec_reg2_dstacc(const ckernel::unpacker::alu_config_t& config) {
    dprint_data_format(config.ALU_FORMAT_SPEC_REG2_Dstacc);
    DPRINT << ENDL();
}

inline void dprint_tensix_alu_config_alu_acc_ctrl_fp32_enabled(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ACC_CTRL_Fp32_enabled << ENDL();
}

inline void dprint_tensix_alu_config_alu_acc_ctrl_sfpu_fp32_enabled(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ACC_CTRL_SFPU_Fp32_enabled << ENDL();
}

inline void dprint_tensix_alu_config_alu_acc_ctrl_int8_math_enabled(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ACC_CTRL_INT8_math_enabled << ENDL();
}

// Print content of the register field by field.
inline void dprint_tensix_alu_config() {
    MATH(ckernel::unpacker::alu_config_t config = ckernel::unpacker::read_alu_config();

         DPRINT << "ALU_ROUNDING_MODE_Fpu_srnd_en: ";
         dprint_tensix_alu_config_alu_rounding_mode_fpu_srnd_en(config);
         DPRINT << "ALU_ROUNDING_MODE_Gasket_srnd_en: ";
         dprint_tensix_alu_config_alu_rounding_mode_gasket_srnd_en(config);
         DPRINT << "ALU_ROUNDING_MODE_Packer_srnd_en: ";
         dprint_tensix_alu_config_alu_rounding_mode_packer_srnd_en(config);
         DPRINT << "ALU_ROUNDING_MODE_Padding: ";
         dprint_tensix_alu_config_alu_rounding_mode_padding(config);
         DPRINT << "ALU_ROUNDING_MODE_GS_LF: ";
         dprint_tensix_alu_config_alu_rounding_mode_gs_lf(config);
         DPRINT << "ALU_ROUNDING_MODE_Bfp8_HF: ";
         dprint_tensix_alu_config_alu_rounding_mode_bfp8_hf(config);
         DPRINT << "ALU_FORMAT_SPEC_REG0_SrcAUnsigned: ";
         dprint_tensix_alu_config_alu_format_spec_reg0_srcaunsigned(config);
         DPRINT << "ALU_FORMAT_SPEC_REG0_SrcBUnsigned: ";
         dprint_tensix_alu_config_alu_format_spec_reg0_srcbunsigned(config);
         DPRINT << "ALU_FORMAT_SPEC_REG0_SrcA: ";
         dprint_tensix_alu_config_alu_format_spec_reg0_srca(config);
         DPRINT << "ALU_FORMAT_SPEC_REG1_SrcB: ";
         dprint_tensix_alu_config_alu_format_spec_reg1_srcb(config);
         DPRINT << "ALU_FORMAT_SPEC_REG2_Dstacc: ";
         dprint_tensix_alu_config_alu_format_spec_reg2_dstacc(config);
         DPRINT << "ALU_ACC_CTRL_Fp32_enabled: ";
         dprint_tensix_alu_config_alu_acc_ctrl_fp32_enabled(config);
         DPRINT << "ALU_ACC_CTRL_SFPU_Fp32_enabled: ";
         dprint_tensix_alu_config_alu_acc_ctrl_sfpu_fp32_enabled(config);
         DPRINT << "ALU_ACC_CTRL_INT8_math_enabled: ";
         dprint_tensix_alu_config_alu_acc_ctrl_int8_math_enabled(config);)
}

// PACK RELU CONFIG

// These functions' argument should be return value of read_relu_config()

inline void dprint_tensix_pack_relu_config_alu_acc_ctrl_zero_flag_disabled_src(
    const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ACC_CTRL_Zero_Flag_disabled_src << ENDL();
}

inline void dprint_tensix_pack_relu_config_alu_acc_ctrl_zero_flag_disabled_dst(
    const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ACC_CTRL_Zero_Flag_disabled_dst << ENDL();
}

inline void dprint_tensix_pack_relu_config_stacc_relu_apply_relu(const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.STACC_RELU_ApplyRelu << ENDL();
}

inline void dprint_tensix_pack_relu_config_stacc_relu_relu_threshold(const ckernel::packer::relu_config_t& config) {
    DPRINT << DEC() << config.STACC_RELU_ReluThreshold << ENDL();
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_main(const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.DISABLE_RISC_BP_Disable_main << ENDL();
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_trisc(const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.DISABLE_RISC_BP_Disable_trisc << ENDL();
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_ncrisc(
    const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.DISABLE_RISC_BP_Disable_ncrisc << ENDL();
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_main(
    const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.DISABLE_RISC_BP_Disable_bmp_clear_main << ENDL();
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_trisc(
    const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.DISABLE_RISC_BP_Disable_bmp_clear_trisc << ENDL();
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_ncrisc(
    const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.DISABLE_RISC_BP_Disable_bmp_clear_ncrisc << ENDL();
}

inline void dprint_tensix_pack_relu_config() {
    MATH(ckernel::packer::relu_config_t config = ckernel::packer::read_relu_config();

         DPRINT << "ALU_ACC_CTRL_Zero_Flag_disabled_src: ";
         dprint_tensix_pack_relu_config_alu_acc_ctrl_zero_flag_disabled_src(config);
         DPRINT << "ALU_ACC_CTRL_Zero_Flag_disabled_dst: ";
         dprint_tensix_pack_relu_config_alu_acc_ctrl_zero_flag_disabled_dst(config);
         DPRINT << "STACC_RELU_ApplyRelu: ";
         dprint_tensix_pack_relu_config_stacc_relu_apply_relu(config);
         DPRINT << "STACC_RELU_ReluThreshold: ";
         dprint_tensix_pack_relu_config_stacc_relu_relu_threshold(config);
         DPRINT << "DISABLE_RISC_BP_Disable_main: ";
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_main(config);
         DPRINT << "DISABLE_RISC_BP_Disable_trisc: ";
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_trisc(config);
         DPRINT << "DISABLE_RISC_BP_Disable_ncrisc: ";
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_ncrisc(config);
         DPRINT << "DISABLE_RISC_BP_Disable_bmp_clear_main: ";
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_main(config);
         DPRINT << "DISABLE_RISC_BP_Disable_bmp_clear_trisc: ";
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_trisc(config);
         DPRINT << "DISABLE_RISC_BP_Disable_bmp_clear_ncrisc: ";
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_ncrisc(config);)
}

// PACK DEST RD CTRL

// These functions' argument should be return value of read_dest_rd_ctrl()

inline void dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_32b_data(
    const ckernel::packer::dest_rd_ctrl_t& dest) {
    DPRINT << "0x" << HEX() << dest.PCK_DEST_RD_CTRL_Read_32b_data << "; " << ENDL();
}

inline void dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_unsigned(
    const ckernel::packer::dest_rd_ctrl_t& dest) {
    DPRINT << "0x" << HEX() << dest.PCK_DEST_RD_CTRL_Read_unsigned << "; " << ENDL();
}

inline void dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_int8(const ckernel::packer::dest_rd_ctrl_t& dest) {
    DPRINT << "0x" << HEX() << dest.PCK_DEST_RD_CTRL_Read_int8 << "; " << ENDL();
}

inline void dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_round_10b_mant(
    const ckernel::packer::dest_rd_ctrl_t& dest) {
    DPRINT << "0x" << HEX() << dest.PCK_DEST_RD_CTRL_Round_10b_mant << "; " << ENDL();
}

inline void dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_reserved(const ckernel::packer::dest_rd_ctrl_t& dest) {
    DPRINT << "0x" << HEX() << dest.PCK_DEST_RD_CTRL_Reserved << "; " << ENDL();
}

// Printing dest control bits
inline void dprint_tensix_dest_rd_ctrl() {
    PACK(ckernel::packer::dest_rd_ctrl_t dest = ckernel::packer::read_dest_rd_ctrl();

         DPRINT << "PCK_DEST_RD_CTRL_Read_32b_data: ";
         dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_32b_data(dest);
         DPRINT << "PCK_DEST_RD_CTRL_Read_unsigned: ";
         dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_unsigned(dest);
         DPRINT << "PCK_DEST_RD_CTRL_Read_int8: ";
         dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_int8(dest);
         DPRINT << "PCK_DEST_RD_CTRL_Round_10b_mant: ";
         dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_round_10b_mant(dest);
         DPRINT << "PCK_DEST_RD_CTRL_Reserved: ";
         dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_reserved(dest);)
}

// Choose what register you want printed with reg_id (1-4), 0 for all
inline void dprint_tensix_pack_edge_offset(uint reg_id = 0) {
    std::array<ckernel::packer::pck_edge_offset_t, 4> edge_vec;
    PACK(
        edge_vec = ckernel::packer::read_pack_edge_offset();
        if (reg_id >= 1 && reg_id <= 4) {
            DPRINT << "REG_ID: " << reg_id << ENDL();
            dprint_tensix_pack_edge_offset_helper(edge_vec[reg_id - 1]);
        }
        // Print all registers
        else if (reg_id == 0) {
            for (uint i = 1; i <= 4; i++) {
                DPRINT << "REG_ID: " << i << ENDL();
                dprint_tensix_pack_edge_offset_helper(edge_vec[i - 1]);
                if (i != 4) {
                    DPRINT << ENDL();
                }
            }
        } else DPRINT
        << "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND 4." << ENDL();)
}

// Choose what register you want printed with reg_id (1-4), 0 for all
inline void dprint_tensix_pack_counters(uint reg_id = 0) {
    std::array<ckernel::packer::pack_counters_t, 4> counters_vec;
    PACK(
        counters_vec = ckernel::packer::read_pack_counters();
        if (reg_id >= 1 && reg_id <= 4) {
            DPRINT << "REG_ID: " << reg_id << ENDL();
            dprint_tensix_pack_counters_helper(counters_vec[reg_id - 1]);
        }
        // Print all registers
        else if (reg_id == 0) {
            for (uint i = 1; i <= 4; i++) {
                DPRINT << "REG_ID: " << i << ENDL();
                dprint_tensix_pack_counters_helper(counters_vec[i - 1]);
                if (i != 4) {
                    DPRINT << ENDL();
                }
            }
        } else DPRINT
        << "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND 4." << ENDL();)
}

#endif  // END OF ELSE

// Choose what register you want printed with reg_id (1-4)
inline void dprint_tensix_pack_config_helper(const ckernel::packer::pack_config_t& config) {
#ifdef ARCH_GRAYSKULL
    dprint_tensix_pack_config_grayskull(config);
#elif ARCH_WORMHOLE
    dprint_tensix_pack_config_wormhole(config);
#else
    dprint_tensix_pack_config_blackhole(config);
#endif
}

// Choose register you want (1-2). 0 for both.
inline void dprint_tensix_unpack_tile_descriptor(uint reg_id = 0) {
    UNPACK(
#ifdef ARCH_GRAYSKULL
        dprint_tensix_unpack_tile_descriptor_grayskull(reg_id);
#else
        dprint_tensix_unpack_tile_descriptor_wormhole_or_blackhole(reg_id);
#endif
    )
}

// Choose register you want (1-2). 0 for both.
inline void dprint_tensix_unpack_config(uint reg_id = 0) {
    UNPACK(
#ifdef ARCH_GRAYSKULL
        dprint_tensix_unpack_config_grayskull(reg_id);
#else
        dprint_tensix_unpack_config_wormhole_or_blackhole(reg_id);
#endif
    )
}

// Choose what register you want by id (1-4). 0 for all.
inline void dprint_tensix_pack_config(uint reg_id = 0) {
#ifdef ARCH_BLACKHOLE
    constexpr uint num_of_instances = 1;
#else
    constexpr uint num_of_instances = 4;
#endif
    std::array<ckernel::packer::pack_config_t, num_of_instances> config_vec;
    MATH(
        config_vec = ckernel::packer::read_pack_config(); if (reg_id >= 1 && reg_id <= num_of_instances) {
            DPRINT << "REG_ID: " << reg_id << ENDL();
            dprint_tensix_pack_config_helper(config_vec[reg_id - 1]);
        } else if (reg_id == 0) for (uint i = 1; i <= num_of_instances; i++) {
            DPRINT << "REG_ID: " << i << ENDL();
            dprint_tensix_pack_config_helper(config_vec[i - 1]);
            if (i != num_of_instances) {
                DPRINT << ENDL();
            }
        } else DPRINT << "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND 4."
                      << ENDL();)
}

// Choose what register you want printed (1-2). 0 for all.
inline void dprint_tensix_pack_strides(uint reg_id = 0) {
    PACK(
        // Get pointer to registers for current state ID
        volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

        if (reg_id >= 1 && reg_id <= 2) {
            DPRINT << "REG_ID: " << reg_id << ENDL();
            dprint_tensix_pack_strides_helper(reg_id, cfg);
        }
        // Print all registers
        else if (reg_id == 0) {
            for (uint i = 1; i <= 2; i++) {
                DPRINT << "REG_ID: " << i << ENDL();
                dprint_tensix_pack_strides_helper(i, cfg);
                if (i != 2) {
                    DPRINT << ENDL();
                }
            }
        } else DPRINT
        << "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND 4." << ENDL();)
}
