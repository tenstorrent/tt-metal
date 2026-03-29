// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "dprint.h"
#include "dprint_tensix.h"
#include "cunpack_common.h"

// NOTE: FUNCTIONS WITHOUT HELPER SUFFIX ARE INTENDED TO BE USED

// UNPACK TILE DESCRIPTOR

// These function's argument should be return value of read_unpack_tile_descriptor()

inline void dprint_tensix_unpack_tile_descriptor_in_data_format(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    dprint_data_format(tile_descriptor.in_data_format);
    DPRINT << ENDL();
    DEVICE_PRINT("\n");
}

inline void dprint_tensix_unpack_tile_descriptor_uncompressed(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << "0x" << HEX() << tile_descriptor.uncompressed << ENDL();
    DEVICE_PRINT("0x{:08X}\n", tile_descriptor.uncompressed);
}

inline void dprint_tensix_unpack_tile_descriptor_reserved_0(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << "0x" << HEX() << tile_descriptor.reserved_0 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", tile_descriptor.reserved_0);
}

inline void dprint_tensix_unpack_tile_descriptor_blobs_per_xy_plane(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << DEC() << tile_descriptor.blobs_per_xy_plane << ENDL();
    DEVICE_PRINT("{}\n", tile_descriptor.blobs_per_xy_plane);
}

inline void dprint_tensix_unpack_tile_descriptor_reserved_1(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << "0x" << HEX() << tile_descriptor.reserved_1 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", tile_descriptor.reserved_1);
}

inline void dprint_tensix_unpack_tile_descriptor_x_dim(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << DEC() << tile_descriptor.x_dim << ENDL();
    DEVICE_PRINT("{}\n", tile_descriptor.x_dim);
}

inline void dprint_tensix_unpack_tile_descriptor_y_dim(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << DEC() << tile_descriptor.y_dim << ENDL();
    DEVICE_PRINT("{}\n", tile_descriptor.y_dim);
}

inline void dprint_tensix_unpack_tile_descriptor_z_dim(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << DEC() << tile_descriptor.z_dim << ENDL();
    DEVICE_PRINT("{}\n", tile_descriptor.z_dim);
}

inline void dprint_tensix_unpack_tile_descriptor_w_dim(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << DEC() << tile_descriptor.w_dim << ENDL();
    DEVICE_PRINT("{}\n", tile_descriptor.w_dim);
}

inline void dprint_tensix_unpack_tile_descriptor_blobs_y_start(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << DEC() << ((tile_descriptor.blobs_y_start_hi << 16) | tile_descriptor.blobs_y_start_lo) << ENDL();
    DEVICE_PRINT("{}\n", ((tile_descriptor.blobs_y_start_hi << 16) | tile_descriptor.blobs_y_start_lo));
}

inline void dprint_tensix_unpack_tile_descriptor_digest_type(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << "0x" << HEX() << tile_descriptor.digest_type << ENDL();
    DEVICE_PRINT("0x{:08X}\n", tile_descriptor.digest_type);
}

inline void dprint_tensix_unpack_tile_descriptor_digest_size(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << DEC() << tile_descriptor.digest_size << ENDL();
    DEVICE_PRINT("{}\n", tile_descriptor.digest_size);
}

// UNPACK CONFIG

// These function's argument should be return value of read_unpack_config()

inline void dprint_tensix_unpack_config_out_data_format(const ckernel::unpacker::unpack_config_t& config) {
    dprint_data_format(config.out_data_format);
    DPRINT << ENDL();
    DEVICE_PRINT("\n");
}

inline void dprint_tensix_unpack_config_throttle_mode(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.throttle_mode << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.throttle_mode);
}

inline void dprint_tensix_unpack_config_context_count(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.context_count << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.context_count);
}

inline void dprint_tensix_unpack_config_haloize_mode(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.haloize_mode << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.haloize_mode);
}

inline void dprint_tensix_unpack_config_tileize_mode(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.tileize_mode << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.tileize_mode);
}

inline void dprint_tensix_unpack_config_force_shared_exp(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.force_shared_exp << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.force_shared_exp);
}

inline void dprint_tensix_unpack_config_upsample_rate(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << DEC() << config.upsample_rate << ENDL();
    DEVICE_PRINT("{}\n", config.upsample_rate);
}

inline void dprint_tensix_unpack_config_upsample_and_interlave(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.upsamle_and_interlave << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.upsamle_and_interlave);
}

inline void dprint_tensix_unpack_config_shift_amount(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << DEC() << config.shift_amount << ENDL();
    DEVICE_PRINT("{}\n", config.shift_amount);
}

inline void dprint_tensix_unpack_config_uncompress_cntx0_3(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.uncompress_cntx0_3 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.uncompress_cntx0_3);
}

inline void dprint_tensix_unpack_config_reserved_1(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_1 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.reserved_1);
}

inline void dprint_tensix_unpack_config_uncompress_cntx4_7(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.uncompress_cntx4_7 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.uncompress_cntx4_7);
}

inline void dprint_tensix_unpack_config_reserved_2(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_2 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.reserved_2);
}

inline void dprint_tensix_unpack_config_limit_addr(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.limit_addr << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.limit_addr);
}

inline void dprint_tensix_unpack_config_fifo_size(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << DEC() << config.fifo_size << ENDL();
    DEVICE_PRINT("{}\n", config.fifo_size);
}

#if defined(ARCH_WORMHOLE) || defined(ARCH_BLACKHOLE)
inline void dprint_tensix_unpack_config_unpack_src_reg_set_update(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.unpack_src_reg_set_update << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.unpack_src_reg_set_update);
}

inline void dprint_tensix_unpack_config_unpack_if_sel(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.unpack_if_sel << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.unpack_if_sel);
}

inline void dprint_tensix_unpack_config_unpack_if_sel_cntx0_3(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.unpack_if_sel_cntx0_3 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.unpack_if_sel_cntx0_3);
}

inline void dprint_tensix_unpack_config_unpack_if_sel_cntx4_7(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.unpack_if_sel_cntx4_7 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.unpack_if_sel_cntx4_7);
}

inline void dprint_tensix_unpack_config_reserved_3(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_3 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.reserved_3);
}

inline void dprint_tensix_unpack_config_reserved_4(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_4 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.reserved_4);
}

inline void dprint_tensix_unpack_config_reserved_5(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_5 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.reserved_5);
}
#endif

// HARDWARE SPECIFIC FUNCTIONS

inline void dprint_tensix_unpack_tile_descriptor_helper(
    const ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    DPRINT << "in_data_format: ";
    DEVICE_PRINT("in_data_format: ");
    dprint_tensix_unpack_tile_descriptor_in_data_format(tile_descriptor);
    DPRINT << "uncompressed: ";
    DEVICE_PRINT("uncompressed: ");
    dprint_tensix_unpack_tile_descriptor_uncompressed(tile_descriptor);
    DPRINT << "reserved_0: ";
    DEVICE_PRINT("reserved_0: ");
    dprint_tensix_unpack_tile_descriptor_reserved_0(tile_descriptor);
    DPRINT << "blobs_per_xy_plane: ";
    DEVICE_PRINT("blobs_per_xy_plane: ");
    dprint_tensix_unpack_tile_descriptor_blobs_per_xy_plane(tile_descriptor);
    DPRINT << "reserved_1: ";
    DEVICE_PRINT("reserved_1: ");
    dprint_tensix_unpack_tile_descriptor_reserved_1(tile_descriptor);
    DPRINT << "x_dim: ";
    DEVICE_PRINT("x_dim: ");
    dprint_tensix_unpack_tile_descriptor_x_dim(tile_descriptor);
    DPRINT << "y_dim: ";
    DEVICE_PRINT("y_dim: ");
    dprint_tensix_unpack_tile_descriptor_y_dim(tile_descriptor);
    DPRINT << "z_dim: ";
    DEVICE_PRINT("z_dim: ");
    dprint_tensix_unpack_tile_descriptor_z_dim(tile_descriptor);
    DPRINT << "w_dim: ";
    DEVICE_PRINT("w_dim: ");
    dprint_tensix_unpack_tile_descriptor_w_dim(tile_descriptor);
    DPRINT << "blobs_y_start: ";
    DEVICE_PRINT("blobs_y_start: ");
    dprint_tensix_unpack_tile_descriptor_blobs_y_start(tile_descriptor);
    DPRINT << "digest_type: ";
    DEVICE_PRINT("digest_type: ");
    dprint_tensix_unpack_tile_descriptor_digest_type(tile_descriptor);
    DPRINT << "digest_size: ";
    DEVICE_PRINT("digest_size: ");
    dprint_tensix_unpack_tile_descriptor_digest_size(tile_descriptor);
}

// Choose which register you want (1-2). 0 for both.
inline void dprint_tensix_unpack_tile_descriptor(uint reg_id = 0) {
    std::array<ckernel::unpacker::unpack_tile_descriptor_t, ckernel::unpacker::NUM_UNPACKERS> tile_descriptor_vec;
    UNPACK(
        tile_descriptor_vec = ckernel::unpacker::read_unpack_tile_descriptor();
        if (reg_id >= 1 && reg_id <= ckernel::unpacker::NUM_UNPACKERS) {
            DPRINT << "REG_ID: " << reg_id << ENDL();
            DEVICE_PRINT("REG_ID: {}\n", reg_id);
            dprint_tensix_unpack_tile_descriptor_helper(tile_descriptor_vec[reg_id - 1]);
        } else if (reg_id == 0) {
            for (uint i = 1; i <= ckernel::unpacker::NUM_UNPACKERS; i++) {
                DPRINT << "REG_ID: " << i << ENDL();
                DEVICE_PRINT("REG_ID: {}\n", i);
                dprint_tensix_unpack_tile_descriptor_helper(tile_descriptor_vec[i - 1]);
                if (i != ckernel::unpacker::NUM_UNPACKERS) {
                    DPRINT << ENDL();
                    DEVICE_PRINT("\n");
                }
            }
        } else {
            DPRINT << "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND " << ckernel::unpacker::NUM_UNPACKERS
                   << "." << ENDL();
            DEVICE_PRINT(
                "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND {}.\n", ckernel::unpacker::NUM_UNPACKERS);
        })
}

inline void dprint_tensix_unpack_config_helper(const ckernel::unpacker::unpack_config_t& config) {
    DPRINT << "out_data_format: ";
    DEVICE_PRINT("out_data_format: ");
    dprint_tensix_unpack_config_out_data_format(config);
    DPRINT << "throttle_mode: ";
    DEVICE_PRINT("throttle_mode: ");
    dprint_tensix_unpack_config_throttle_mode(config);
    DPRINT << "context_count: ";
    DEVICE_PRINT("context_count: ");
    dprint_tensix_unpack_config_context_count(config);
    DPRINT << "haloize_mode: ";
    DEVICE_PRINT("haloize_mode: ");
    dprint_tensix_unpack_config_haloize_mode(config);
    DPRINT << "tileize_mode: ";
    DEVICE_PRINT("tileize_mode: ");
    dprint_tensix_unpack_config_tileize_mode(config);
    DPRINT << "unpack_src_reg_set_update: ";
    DEVICE_PRINT("unpack_src_reg_set_update: ");
    dprint_tensix_unpack_config_unpack_src_reg_set_update(config);
    DPRINT << "unpack_if_sel: ";
    DEVICE_PRINT("unpack_if_sel: ");
    dprint_tensix_unpack_config_unpack_if_sel(config);
    DPRINT << "upsample_rate: ";
    DEVICE_PRINT("upsample_rate: ");
    dprint_tensix_unpack_config_upsample_rate(config);
    DPRINT << "reserved_1: ";
    DEVICE_PRINT("reserved_1: ");
    dprint_tensix_unpack_config_reserved_1(config);
    DPRINT << "upsample_and_interlave: ";
    DEVICE_PRINT("upsample_and_interlave: ");
    dprint_tensix_unpack_config_upsample_and_interlave(config);
    DPRINT << "shift_amount: ";
    DEVICE_PRINT("shift_amount: ");
    dprint_tensix_unpack_config_shift_amount(config);
    DPRINT << "uncompress_cntx0_3: ";
    DEVICE_PRINT("uncompress_cntx0_3: ");
    dprint_tensix_unpack_config_uncompress_cntx0_3(config);
    DPRINT << "unpack_if_sel_cntx0_3: ";
    DEVICE_PRINT("unpack_if_sel_cntx0_3: ");
    dprint_tensix_unpack_config_unpack_if_sel_cntx0_3(config);
    DPRINT << "force_shared_exp: ";
    DEVICE_PRINT("force_shared_exp: ");
    dprint_tensix_unpack_config_force_shared_exp(config);
    DPRINT << "reserved_2: ";
    DEVICE_PRINT("reserved_2: ");
    dprint_tensix_unpack_config_reserved_2(config);
    DPRINT << "uncompress_cntx4_7: ";
    DEVICE_PRINT("uncompress_cntx4_7: ");
    dprint_tensix_unpack_config_uncompress_cntx4_7(config);
    DPRINT << "unpack_if_sel_cntx4_7: ";
    DEVICE_PRINT("unpack_if_sel_cntx4_7: ");
    dprint_tensix_unpack_config_unpack_if_sel_cntx4_7(config);
    DPRINT << "reserved_3: ";
    DEVICE_PRINT("reserved_3: ");
    dprint_tensix_unpack_config_reserved_3(config);
    DPRINT << "limit_addr: ";
    DEVICE_PRINT("limit_addr: ");
    dprint_tensix_unpack_config_limit_addr(config);
    DPRINT << "reserved_4: ";
    DEVICE_PRINT("reserved_4: ");
    dprint_tensix_unpack_config_reserved_4(config);
    DPRINT << "fifo_size: ";
    DEVICE_PRINT("fifo_size: ");
    dprint_tensix_unpack_config_fifo_size(config);
    DPRINT << "reserved_5: ";
    DEVICE_PRINT("reserved_5: ");
    dprint_tensix_unpack_config_reserved_5(config);
}

// Choose which register you want (1-2). 0 for both.
inline void dprint_tensix_unpack_config(uint reg_id = 0) {
    std::array<ckernel::unpacker::unpack_config_t, ckernel::unpacker::NUM_UNPACKERS> config_vec;
    UNPACK(
        config_vec = ckernel::unpacker::read_unpack_config();
        if (reg_id >= 1 && reg_id <= ckernel::unpacker::NUM_UNPACKERS) {
            DPRINT << "REG_ID: " << reg_id << ENDL();
            DEVICE_PRINT("REG_ID: {}\n", reg_id);
            dprint_tensix_unpack_config_helper(config_vec[reg_id - 1]);
        } else if (reg_id == 0) {
            for (uint i = 1; i <= ckernel::unpacker::NUM_UNPACKERS; i++) {
                DPRINT << "REG_ID: " << i << ENDL();
                DEVICE_PRINT("REG_ID: {}\n", i);
                dprint_tensix_unpack_config_helper(config_vec[i - 1]);
                if (i != ckernel::unpacker::NUM_UNPACKERS) {
                    DPRINT << ENDL();
                    DEVICE_PRINT("\n");
                }
            }
        } else {
            DPRINT << "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND " << ckernel::unpacker::NUM_UNPACKERS
                   << "." << ENDL();
            DEVICE_PRINT(
                "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND {}.\n", ckernel::unpacker::NUM_UNPACKERS);
        })
}

// ALU CONFIG

// These functions' argument should be return value of read_alu_config()

inline void dprint_tensix_alu_config_alu_rounding_mode_fpu_srnd_en(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ROUNDING_MODE_Fpu_srnd_en << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.ALU_ROUNDING_MODE_Fpu_srnd_en);
}

inline void dprint_tensix_alu_config_alu_rounding_mode_gasket_srnd_en(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ROUNDING_MODE_Gasket_srnd_en << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.ALU_ROUNDING_MODE_Gasket_srnd_en);
}

inline void dprint_tensix_alu_config_alu_rounding_mode_packer_srnd_en(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ROUNDING_MODE_Packer_srnd_en << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.ALU_ROUNDING_MODE_Packer_srnd_en);
}

inline void dprint_tensix_alu_config_alu_rounding_mode_padding(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ROUNDING_MODE_Padding << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.ALU_ROUNDING_MODE_Padding);
}

inline void dprint_tensix_alu_config_alu_rounding_mode_gs_lf(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ROUNDING_MODE_GS_LF << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.ALU_ROUNDING_MODE_GS_LF);
}

inline void dprint_tensix_alu_config_alu_rounding_mode_bfp8_hf(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ROUNDING_MODE_Bfp8_HF << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.ALU_ROUNDING_MODE_Bfp8_HF);
}

inline void dprint_tensix_alu_config_alu_format_spec_reg0_srcaunsigned(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_FORMAT_SPEC_REG0_SrcAUnsigned << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.ALU_FORMAT_SPEC_REG0_SrcAUnsigned);
}

inline void dprint_tensix_alu_config_alu_format_spec_reg0_srcbunsigned(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_FORMAT_SPEC_REG0_SrcBUnsigned << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.ALU_FORMAT_SPEC_REG0_SrcBUnsigned);
}

inline void dprint_tensix_alu_config_alu_format_spec_reg0_srca(const ckernel::unpacker::alu_config_t& config) {
    dprint_data_format(config.ALU_FORMAT_SPEC_REG0_SrcA);
    DPRINT << ENDL();
    DEVICE_PRINT("\n");
}

inline void dprint_tensix_alu_config_alu_format_spec_reg1_srcb(const ckernel::unpacker::alu_config_t& config) {
    dprint_data_format(config.ALU_FORMAT_SPEC_REG1_SrcB);
    DPRINT << ENDL();
    DEVICE_PRINT("\n");
}

inline void dprint_tensix_alu_config_alu_format_spec_reg2_dstacc(const ckernel::unpacker::alu_config_t& config) {
    dprint_data_format(config.ALU_FORMAT_SPEC_REG2_Dstacc);
    DPRINT << ENDL();
    DEVICE_PRINT("\n");
}

inline void dprint_tensix_alu_config_alu_acc_ctrl_fp32_enabled(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ACC_CTRL_Fp32_enabled << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.ALU_ACC_CTRL_Fp32_enabled);
}

inline void dprint_tensix_alu_config_alu_acc_ctrl_sfpu_fp32_enabled(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ACC_CTRL_SFPU_Fp32_enabled << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.ALU_ACC_CTRL_SFPU_Fp32_enabled);
}

inline void dprint_tensix_alu_config_alu_acc_ctrl_int8_math_enabled(const ckernel::unpacker::alu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ACC_CTRL_INT8_math_enabled << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.ALU_ACC_CTRL_INT8_math_enabled);
}

// Print content of the register field by field.
inline void dprint_tensix_alu_config() {
    MATH(ckernel::unpacker::alu_config_t config = ckernel::unpacker::read_alu_config();

         DPRINT << "ALU_ROUNDING_MODE_Fpu_srnd_en: ";
         DEVICE_PRINT("ALU_ROUNDING_MODE_Fpu_srnd_en: ");
         dprint_tensix_alu_config_alu_rounding_mode_fpu_srnd_en(config);
         DPRINT << "ALU_ROUNDING_MODE_Gasket_srnd_en: ";
         DEVICE_PRINT("ALU_ROUNDING_MODE_Gasket_srnd_en: ");
         dprint_tensix_alu_config_alu_rounding_mode_gasket_srnd_en(config);
         DPRINT << "ALU_ROUNDING_MODE_Packer_srnd_en: ";
         DEVICE_PRINT("ALU_ROUNDING_MODE_Packer_srnd_en: ");
         dprint_tensix_alu_config_alu_rounding_mode_packer_srnd_en(config);
         DPRINT << "ALU_ROUNDING_MODE_Padding: ";
         DEVICE_PRINT("ALU_ROUNDING_MODE_Padding: ");
         dprint_tensix_alu_config_alu_rounding_mode_padding(config);
         DPRINT << "ALU_ROUNDING_MODE_GS_LF: ";
         DEVICE_PRINT("ALU_ROUNDING_MODE_GS_LF: ");
         dprint_tensix_alu_config_alu_rounding_mode_gs_lf(config);
         DPRINT << "ALU_ROUNDING_MODE_Bfp8_HF: ";
         DEVICE_PRINT("ALU_ROUNDING_MODE_Bfp8_HF: ");
         dprint_tensix_alu_config_alu_rounding_mode_bfp8_hf(config);
         DPRINT << "ALU_FORMAT_SPEC_REG0_SrcAUnsigned: ";
         DEVICE_PRINT("ALU_FORMAT_SPEC_REG0_SrcAUnsigned: ");
         dprint_tensix_alu_config_alu_format_spec_reg0_srcaunsigned(config);
         DPRINT << "ALU_FORMAT_SPEC_REG0_SrcBUnsigned: ";
         DEVICE_PRINT("ALU_FORMAT_SPEC_REG0_SrcBUnsigned: ");
         dprint_tensix_alu_config_alu_format_spec_reg0_srcbunsigned(config);
         DPRINT << "ALU_FORMAT_SPEC_REG0_SrcA: ";
         DEVICE_PRINT("ALU_FORMAT_SPEC_REG0_SrcA: ");
         dprint_tensix_alu_config_alu_format_spec_reg0_srca(config);
         DPRINT << "ALU_FORMAT_SPEC_REG1_SrcB: ";
         DEVICE_PRINT("ALU_FORMAT_SPEC_REG1_SrcB: ");
         dprint_tensix_alu_config_alu_format_spec_reg1_srcb(config);
         DPRINT << "ALU_FORMAT_SPEC_REG2_Dstacc: ";
         DEVICE_PRINT("ALU_FORMAT_SPEC_REG2_Dstacc: ");
         dprint_tensix_alu_config_alu_format_spec_reg2_dstacc(config);
         DPRINT << "ALU_ACC_CTRL_Fp32_enabled: ";
         DEVICE_PRINT("ALU_ACC_CTRL_Fp32_enabled: ");
         dprint_tensix_alu_config_alu_acc_ctrl_fp32_enabled(config);
         DPRINT << "ALU_ACC_CTRL_SFPU_Fp32_enabled: ";
         DEVICE_PRINT("ALU_ACC_CTRL_SFPU_Fp32_enabled: ");
         dprint_tensix_alu_config_alu_acc_ctrl_sfpu_fp32_enabled(config);
         DPRINT << "ALU_ACC_CTRL_INT8_math_enabled: ";
         DEVICE_PRINT("ALU_ACC_CTRL_INT8_math_enabled: ");
         dprint_tensix_alu_config_alu_acc_ctrl_int8_math_enabled(config);)
}
