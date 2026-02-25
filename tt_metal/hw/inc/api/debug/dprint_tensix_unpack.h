// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "dprint.h"
#include "dprint_tensix.h"
#include "cunpack_common.h"

// NOTE: FUNCTIONS WITHOUT HELPER SUFIX ARE INTENDED TO BE USED

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
    DPRINT << DEC() << ((tile_descriptor.blobs_y_start_hi << 16) | tile_descriptor.blobs_y_start_lo) << ENDL();
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

#if defined(ARCH_WORMHOLE) || defined(ARCH_BLACKHOLE)
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

// HARDWARE SPECIFIC FUNCTIONS

inline void dprint_tensix_unpack_tile_descriptor_helper(
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
inline void dprint_tensix_unpack_tile_descriptor(uint reg_id = 0) {
    std::array<ckernel::unpacker::unpack_tile_descriptor_t, ckernel::unpacker::NUM_UNPACKERS> tile_descriptor_vec;
    UNPACK(
        tile_descriptor_vec = ckernel::unpacker::read_unpack_tile_descriptor();
        if (reg_id >= 1 && reg_id <= ckernel::unpacker::NUM_UNPACKERS) {
            DPRINT << "REG_ID: " << reg_id << ENDL();
            dprint_tensix_unpack_tile_descriptor_helper(tile_descriptor_vec[reg_id - 1]);
        } else if (reg_id == 0) {
            for (uint i = 1; i <= ckernel::unpacker::NUM_UNPACKERS; i++) {
                DPRINT << "REG_ID: " << i << ENDL();
                dprint_tensix_unpack_tile_descriptor_helper(tile_descriptor_vec[i - 1]);
                if (i != ckernel::unpacker::NUM_UNPACKERS) {
                    DPRINT << ENDL();
                }
            }
        } else {
            DPRINT << "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND " << ckernel::unpacker::NUM_UNPACKERS
                   << "." << ENDL();
        })
}

inline void dprint_tensix_unpack_config_helper(const ckernel::unpacker::unpack_config_t& config) {
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
inline void dprint_tensix_unpack_config(uint reg_id = 0) {
    std::array<ckernel::unpacker::unpack_config_t, ckernel::unpacker::NUM_UNPACKERS> config_vec;
    UNPACK(
        config_vec = ckernel::unpacker::read_unpack_config();
        if (reg_id >= 1 && reg_id <= ckernel::unpacker::NUM_UNPACKERS) {
            DPRINT << "REG_ID: " << reg_id << ENDL();
            dprint_tensix_unpack_config_helper(config_vec[reg_id - 1]);
        } else if (reg_id == 0) {
            for (uint i = 1; i <= ckernel::unpacker::NUM_UNPACKERS; i++) {
                DPRINT << "REG_ID: " << i << ENDL();
                dprint_tensix_unpack_config_helper(config_vec[i - 1]);
                if (i != ckernel::unpacker::NUM_UNPACKERS) {
                    DPRINT << ENDL();
                }
            }
        } else {
            DPRINT << "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND " << ckernel::unpacker::NUM_UNPACKERS
                   << "." << ENDL();
        })
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
