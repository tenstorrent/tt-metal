// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "dprint.h"
#include "dprint_tensix.h"
#include "cpack_common.h"

// NOTE: FUNCTIONS WITHOUT HELPER SUFFIX ARE INTENDED TO BE USED

// PACK CONFIG

// These function's argument should be return value of read_pack_config()

inline void dprint_tensix_pack_config_row_ptr_section_size(const ckernel::packer::pack_config_t& config) {
    DPRINT << DEC() << config.row_ptr_section_size << ENDL();
    DEVICE_PRINT("{}\n", config.row_ptr_section_size);
}

inline void dprint_tensix_pack_config_exp_section_size(const ckernel::packer::pack_config_t& config) {
    DPRINT << DEC() << config.exp_section_size << ENDL();
    DEVICE_PRINT("{}\n", config.exp_section_size);
}

inline void dprint_tensix_pack_config_l1_dest_addr(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.l1_dest_addr << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.l1_dest_addr);
}

inline void dprint_tensix_pack_config_uncompressed(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.uncompress << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.uncompress);
}

inline void dprint_tensix_pack_config_add_l1_dest_addr_offset(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.add_l1_dest_addr_offset << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.add_l1_dest_addr_offset);
}

inline void dprint_tensix_pack_config_reserved_0(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_0 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.reserved_0);
}

inline void dprint_tensix_pack_config_out_data_format(const ckernel::packer::pack_config_t& config) {
    dprint_data_format(config.out_data_format);
    DPRINT << ENDL();
    DEVICE_PRINT("\n");
}

inline void dprint_tensix_pack_config_in_data_format(const ckernel::packer::pack_config_t& config) {
    dprint_data_format(config.in_data_format);
    DPRINT << ENDL();
    DEVICE_PRINT("\n");
}

#if defined(ARCH_WORMHOLE)
inline void dprint_tensix_pack_config_reserved_1(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_1 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.reserved_1);
}
#endif

inline void dprint_tensix_pack_config_src_if_sel(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.src_if_sel << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.src_if_sel);
}

#if defined(ARCH_WORMHOLE)
inline void dprint_tensix_pack_config_pack_per_xy_plane(const ckernel::packer::pack_config_t& config) {
    DPRINT << DEC() << config.pack_per_xy_plane << ENDL();
    DEVICE_PRINT("{}\n", config.pack_per_xy_plane);
}
#endif

inline void dprint_tensix_pack_config_l1_src_addr(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.l1_src_addr << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.l1_src_addr);
}

#if defined(ARCH_WORMHOLE)
inline void dprint_tensix_pack_config_downsample_mask(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.downsample_mask << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.downsample_mask);
}

inline void dprint_tensix_pack_config_downsample_shift_count(const ckernel::packer::pack_config_t& config) {
    DPRINT << DEC() << config.downsample_shift_count << ENDL();
    DEVICE_PRINT("{}\n", config.downsample_shift_count);
}

inline void dprint_tensix_pack_config_read_mode(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.read_mode << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.read_mode);
}

inline void dprint_tensix_pack_config_exp_threshold_en(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.exp_threshold_en << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.exp_threshold_en);
}

inline void dprint_tensix_pack_config_reserved_2(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.reserved_2 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.reserved_2);
}

inline void dprint_tensix_pack_config_exp_threshold(const ckernel::packer::pack_config_t& config) {
    DPRINT << DEC() << config.exp_threshold << ENDL();
    DEVICE_PRINT("{}\n", config.exp_threshold);
}
#endif

#ifdef ARCH_WORMHOLE
inline void dprint_tensix_pack_config_l1_acc_disable_pack_zero_flag(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.pack_l1_acc_disable_pack_zero_flag << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.pack_l1_acc_disable_pack_zero_flag);
}
#endif

#ifdef ARCH_BLACKHOLE
inline void dprint_tensix_pack_config_disable_pack_zero_flag(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.disable_pack_zero_flag << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.disable_pack_zero_flag);
}

inline void dprint_tensix_pack_config_dis_shared_exp_assembler(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.dis_shared_exp_assembler << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.dis_shared_exp_assembler);
}

inline void dprint_tensix_pack_config_auto_set_last_pacr_intf_sel(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.auto_set_last_pacr_intf_sel << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.auto_set_last_pacr_intf_sel);
}

inline void dprint_tensix_pack_config_enable_out_fifo(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.enable_out_fifo << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.enable_out_fifo);
}

inline void dprint_tensix_pack_config_sub_l1_tile_header_size(const ckernel::packer::pack_config_t& config) {
    DPRINT << DEC() << config.sub_l1_tile_header_size << ENDL();
    DEVICE_PRINT("{}\n", config.sub_l1_tile_header_size);
}

inline void dprint_tensix_pack_config_pack_start_intf_pos(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.pack_start_intf_pos << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.pack_start_intf_pos);
}

inline void dprint_tensix_pack_config_all_pack_disable_zero_compress_ovrd(
    const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.all_pack_disable_zero_compress_ovrd << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.all_pack_disable_zero_compress_ovrd);
}

inline void dprint_tensix_pack_config_add_tile_header_size(const ckernel::packer::pack_config_t& config) {
    DPRINT << DEC() << config.add_tile_header_size << ENDL();
    DEVICE_PRINT("{}\n", config.add_tile_header_size);
}

inline void dprint_tensix_pack_config_pack_dis_y_pos_start_offset(const ckernel::packer::pack_config_t& config) {
    DPRINT << "0x" << HEX() << config.pack_dis_y_pos_start_offset << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.pack_dis_y_pos_start_offset);
}
#endif

#ifdef ARCH_WORMHOLE
inline void dprint_tensix_pack_config_helper(const ckernel::packer::pack_config_t& config) {
    DPRINT << "row_ptr_section_size: ";
    DEVICE_PRINT("row_ptr_section_size: ");
    dprint_tensix_pack_config_row_ptr_section_size(config);
    DPRINT << "exp_section_size: ";
    DEVICE_PRINT("exp_section_size: ");
    dprint_tensix_pack_config_exp_section_size(config);
    DPRINT << "l1_dest_addr: ";
    DEVICE_PRINT("l1_dest_addr: ");
    dprint_tensix_pack_config_l1_dest_addr(config);
    DPRINT << "uncompress: ";
    DEVICE_PRINT("uncompress: ");
    dprint_tensix_pack_config_uncompressed(config);
    DPRINT << "add_l1_dest_addr_offset: ";
    DEVICE_PRINT("add_l1_dest_addr_offset: ");
    dprint_tensix_pack_config_add_l1_dest_addr_offset(config);
    DPRINT << "reserved_0: ";
    DEVICE_PRINT("reserved_0: ");
    dprint_tensix_pack_config_reserved_0(config);
    DPRINT << "out_data_format: ";
    DEVICE_PRINT("out_data_format: ");
    dprint_tensix_pack_config_out_data_format(config);
    DPRINT << "in_data_format: ";
    DEVICE_PRINT("in_data_format: ");
    dprint_tensix_pack_config_in_data_format(config);
    DPRINT << "reserved_1: ";
    DEVICE_PRINT("reserved_1: ");
    dprint_tensix_pack_config_reserved_1(config);
    DPRINT << "src_if_sel: ";
    DEVICE_PRINT("src_if_sel: ");
    dprint_tensix_pack_config_src_if_sel(config);
    DPRINT << "pack_per_xy_plane: ";
    DEVICE_PRINT("pack_per_xy_plane: ");
    dprint_tensix_pack_config_pack_per_xy_plane(config);
    DPRINT << "l1_src_addr: ";
    DEVICE_PRINT("l1_src_addr: ");
    dprint_tensix_pack_config_l1_src_addr(config);
    DPRINT << "downsample_mask: ";
    DEVICE_PRINT("downsample_mask: ");
    dprint_tensix_pack_config_downsample_mask(config);
    DPRINT << "downsample_shift_count: ";
    DEVICE_PRINT("downsample_shift_count: ");
    dprint_tensix_pack_config_downsample_shift_count(config);
    DPRINT << "read_mode: ";
    DEVICE_PRINT("read_mode: ");
    dprint_tensix_pack_config_read_mode(config);
    DPRINT << "exp_threshold_en: ";
    DEVICE_PRINT("exp_threshold_en: ");
    dprint_tensix_pack_config_exp_threshold_en(config);
    DPRINT << "pack_l1_acc_disable_pack_zero_flag: ";
    DEVICE_PRINT("pack_l1_acc_disable_pack_zero_flag: ");
    dprint_tensix_pack_config_l1_acc_disable_pack_zero_flag(config);
    DPRINT << "reserved_2: ";
    DEVICE_PRINT("reserved_2: ");
    dprint_tensix_pack_config_reserved_2(config);
    DPRINT << "exp_threshold: ";
    DEVICE_PRINT("exp_threshold: ");
    dprint_tensix_pack_config_exp_threshold(config);
}
#endif  // ARCH_WORMHOLE

#ifdef ARCH_BLACKHOLE
inline void dprint_tensix_pack_config_helper(const ckernel::packer::pack_config_t& config) {
    DPRINT << "row_ptr_section_size: ";
    DEVICE_PRINT("row_ptr_section_size: ");
    dprint_tensix_pack_config_row_ptr_section_size(config);
    DPRINT << "exp_section_size: ";
    DEVICE_PRINT("exp_section_size: ");
    dprint_tensix_pack_config_exp_section_size(config);
    DPRINT << "l1_dest_addr: ";
    DEVICE_PRINT("l1_dest_addr: ");
    dprint_tensix_pack_config_l1_dest_addr(config);
    DPRINT << "uncompress: ";
    DEVICE_PRINT("uncompress: ");
    dprint_tensix_pack_config_uncompressed(config);
    DPRINT << "add_l1_dest_addr_offset: ";
    DEVICE_PRINT("add_l1_dest_addr_offset: ");
    dprint_tensix_pack_config_add_l1_dest_addr_offset(config);
    DPRINT << "disable_pack_zero_flag: ";
    DEVICE_PRINT("disable_pack_zero_flag: ");
    dprint_tensix_pack_config_disable_pack_zero_flag(config);
    DPRINT << "reserved_0: ";
    DEVICE_PRINT("reserved_0: ");
    dprint_tensix_pack_config_reserved_0(config);
    DPRINT << "out_data_format: ";
    DEVICE_PRINT("out_data_format: ");
    dprint_tensix_pack_config_out_data_format(config);
    DPRINT << "in_data_format: ";
    DEVICE_PRINT("in_data_format: ");
    dprint_tensix_pack_config_in_data_format(config);
    DPRINT << "dis_shared_exp_assembler: ";
    DEVICE_PRINT("dis_shared_exp_assembler: ");
    dprint_tensix_pack_config_dis_shared_exp_assembler(config);
    DPRINT << "auto_set_last_pacr_intf_sel: ";
    DEVICE_PRINT("auto_set_last_pacr_intf_sel: ");
    dprint_tensix_pack_config_auto_set_last_pacr_intf_sel(config);
    DPRINT << "enable_out_fifo: ";
    DEVICE_PRINT("enable_out_fifo: ");
    dprint_tensix_pack_config_enable_out_fifo(config);
    DPRINT << "sub_l1_tile_header_size: ";
    DEVICE_PRINT("sub_l1_tile_header_size: ");
    dprint_tensix_pack_config_sub_l1_tile_header_size(config);
    DPRINT << "src_if_sel: ";
    DEVICE_PRINT("src_if_sel: ");
    dprint_tensix_pack_config_src_if_sel(config);
    DPRINT << "pack_start_intf_pos: ";
    DEVICE_PRINT("pack_start_intf_pos: ");
    dprint_tensix_pack_config_pack_start_intf_pos(config);
    DPRINT << "all_pack_disable_zero_compress_ovrd: ";
    DEVICE_PRINT("all_pack_disable_zero_compress_ovrd: ");
    dprint_tensix_pack_config_all_pack_disable_zero_compress_ovrd(config);
    DPRINT << "add_tile_header_size: ";
    DEVICE_PRINT("add_tile_header_size: ");
    dprint_tensix_pack_config_add_tile_header_size(config);
    DPRINT << "pack_dis_y_pos_start_offset: ";
    DEVICE_PRINT("pack_dis_y_pos_start_offset: ");
    dprint_tensix_pack_config_pack_dis_y_pos_start_offset(config);
    DPRINT << "l1_src_addr: ";
    DEVICE_PRINT("l1_src_addr: ");
    dprint_tensix_pack_config_l1_src_addr(config);
}
#endif  // ARCH_BLACKHOLE

// PACK RELU CONFIG

// These functions' argument should be return value of read_relu_config()

inline void dprint_tensix_pack_relu_config_alu_acc_ctrl_zero_flag_disabled_src(
    const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ACC_CTRL_Zero_Flag_disabled_src << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.ALU_ACC_CTRL_Zero_Flag_disabled_src);
}

inline void dprint_tensix_pack_relu_config_alu_acc_ctrl_zero_flag_disabled_dst(
    const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.ALU_ACC_CTRL_Zero_Flag_disabled_dst << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.ALU_ACC_CTRL_Zero_Flag_disabled_dst);
}

inline void dprint_tensix_pack_relu_config_stacc_relu_apply_relu(const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.STACC_RELU_ApplyRelu << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.STACC_RELU_ApplyRelu);
}

inline void dprint_tensix_pack_relu_config_stacc_relu_relu_threshold(const ckernel::packer::relu_config_t& config) {
    DPRINT << DEC() << config.STACC_RELU_ReluThreshold << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.STACC_RELU_ReluThreshold);
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_main(const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.DISABLE_RISC_BP_Disable_main << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.DISABLE_RISC_BP_Disable_main);
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_trisc(const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.DISABLE_RISC_BP_Disable_trisc << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.DISABLE_RISC_BP_Disable_trisc);
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_ncrisc(
    const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.DISABLE_RISC_BP_Disable_ncrisc << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.DISABLE_RISC_BP_Disable_ncrisc);
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_main(
    const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.DISABLE_RISC_BP_Disable_bmp_clear_main << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.DISABLE_RISC_BP_Disable_bmp_clear_main);
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_trisc(
    const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.DISABLE_RISC_BP_Disable_bmp_clear_trisc << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.DISABLE_RISC_BP_Disable_bmp_clear_trisc);
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_ncrisc(
    const ckernel::packer::relu_config_t& config) {
    DPRINT << "0x" << HEX() << config.DISABLE_RISC_BP_Disable_bmp_clear_ncrisc << ENDL();
    DEVICE_PRINT("0x{:08X}\n", config.DISABLE_RISC_BP_Disable_bmp_clear_ncrisc);
}

inline void dprint_tensix_pack_relu_config() {
    MATH(ckernel::packer::relu_config_t config = ckernel::packer::read_relu_config();

         DPRINT << "ALU_ACC_CTRL_Zero_Flag_disabled_src: ";
         DEVICE_PRINT("ALU_ACC_CTRL_Zero_Flag_disabled_src: ");
         dprint_tensix_pack_relu_config_alu_acc_ctrl_zero_flag_disabled_src(config);
         DPRINT << "ALU_ACC_CTRL_Zero_Flag_disabled_dst: ";
         DEVICE_PRINT("ALU_ACC_CTRL_Zero_Flag_disabled_dst: ");
         dprint_tensix_pack_relu_config_alu_acc_ctrl_zero_flag_disabled_dst(config);
         DPRINT << "STACC_RELU_ApplyRelu: ";
         DEVICE_PRINT("STACC_RELU_ApplyRelu: ");
         dprint_tensix_pack_relu_config_stacc_relu_apply_relu(config);
         DPRINT << "STACC_RELU_ReluThreshold: ";
         DEVICE_PRINT("STACC_RELU_ReluThreshold: ");
         dprint_tensix_pack_relu_config_stacc_relu_relu_threshold(config);
         DPRINT << "DISABLE_RISC_BP_Disable_main: ";
         DEVICE_PRINT("DISABLE_RISC_BP_Disable_main: ");
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_main(config);
         DPRINT << "DISABLE_RISC_BP_Disable_trisc: ";
         DEVICE_PRINT("DISABLE_RISC_BP_Disable_trisc: ");
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_trisc(config);
         DPRINT << "DISABLE_RISC_BP_Disable_ncrisc: ";
         DEVICE_PRINT("DISABLE_RISC_BP_Disable_ncrisc: ");
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_ncrisc(config);
         DPRINT << "DISABLE_RISC_BP_Disable_bmp_clear_main: ";
         DEVICE_PRINT("DISABLE_RISC_BP_Disable_bmp_clear_main: ");
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_main(config);
         DPRINT << "DISABLE_RISC_BP_Disable_bmp_clear_trisc: ";
         DEVICE_PRINT("DISABLE_RISC_BP_Disable_bmp_clear_trisc: ");
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_trisc(config);
         DPRINT << "DISABLE_RISC_BP_Disable_bmp_clear_ncrisc: ";
         DEVICE_PRINT("DISABLE_RISC_BP_Disable_bmp_clear_ncrisc: ");
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_ncrisc(config);)
}

// PACK DEST RD CTRL

// These functions' argument should be return value of read_dest_rd_ctrl()

inline void dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_32b_data(
    const ckernel::packer::dest_rd_ctrl_t& dest) {
    DPRINT << "0x" << HEX() << dest.PCK_DEST_RD_CTRL_Read_32b_data << ENDL();
    DEVICE_PRINT("0x{:08X}\n", dest.PCK_DEST_RD_CTRL_Read_32b_data);
}

inline void dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_unsigned(
    const ckernel::packer::dest_rd_ctrl_t& dest) {
    DPRINT << "0x" << HEX() << dest.PCK_DEST_RD_CTRL_Read_unsigned << ENDL();
    DEVICE_PRINT("0x{:08X}\n", dest.PCK_DEST_RD_CTRL_Read_unsigned);
}

inline void dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_int8(const ckernel::packer::dest_rd_ctrl_t& dest) {
    DPRINT << "0x" << HEX() << dest.PCK_DEST_RD_CTRL_Read_int8 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", dest.PCK_DEST_RD_CTRL_Read_int8);
}

inline void dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_round_10b_mant(
    const ckernel::packer::dest_rd_ctrl_t& dest) {
    DPRINT << "0x" << HEX() << dest.PCK_DEST_RD_CTRL_Round_10b_mant << ENDL();
    DEVICE_PRINT("0x{:08X}\n", dest.PCK_DEST_RD_CTRL_Round_10b_mant);
}

inline void dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_reserved(const ckernel::packer::dest_rd_ctrl_t& dest) {
    DPRINT << "0x" << HEX() << dest.PCK_DEST_RD_CTRL_Reserved << ENDL();
    DEVICE_PRINT("0x{:08X}\n", dest.PCK_DEST_RD_CTRL_Reserved);
}

// Printing dest control bits
inline void dprint_tensix_dest_rd_ctrl() {
    PACK(ckernel::packer::dest_rd_ctrl_t dest = ckernel::packer::read_dest_rd_ctrl();

         DPRINT << "PCK_DEST_RD_CTRL_Read_32b_data: ";
         DEVICE_PRINT("PCK_DEST_RD_CTRL_Read_32b_data: ");
         dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_32b_data(dest);
         DPRINT << "PCK_DEST_RD_CTRL_Read_unsigned: ";
         DEVICE_PRINT("PCK_DEST_RD_CTRL_Read_unsigned: ");
         dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_unsigned(dest);
         DPRINT << "PCK_DEST_RD_CTRL_Read_int8: ";
         DEVICE_PRINT("PCK_DEST_RD_CTRL_Read_int8: ");
         dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_int8(dest);
         DPRINT << "PCK_DEST_RD_CTRL_Round_10b_mant: ";
         DEVICE_PRINT("PCK_DEST_RD_CTRL_Round_10b_mant: ");
         dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_round_10b_mant(dest);
         DPRINT << "PCK_DEST_RD_CTRL_Reserved: ";
         DEVICE_PRINT("PCK_DEST_RD_CTRL_Reserved: ");
         dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_reserved(dest);)
}

// PACK STRIDES
#ifdef ARCH_BLACKHOLE
inline void dprint_tensix_pack_strides_x_stride(const uint32_t& word) {
    dprint_tensix_struct_field(word, 0xffff, 0, "x_stride", true);  // decimal
}

inline void dprint_tensix_pack_strides_y_stride(const uint32_t& word) {
    dprint_tensix_struct_field(word, 0xffff0000, 16, "y_stride", true);  // decimal
}

inline void dprint_tensix_pack_strides_z_stride(const uint32_t& word) {
    dprint_tensix_struct_field(word, 0xffff, 0, "z_stride", true);  // decimal
}

inline void dprint_tensix_pack_strides_w_stride(const uint32_t& word) {
    dprint_tensix_struct_field(word, 0xffff0000, 16, "w_stride", true);  // decimal
}
#else
inline void dprint_tensix_pack_strides_x_stride(const uint32_t& word) {
    dprint_tensix_struct_field(word, 0xffff, 0, "x_stride", true);  // decimal
}

inline void dprint_tensix_pack_strides_y_stride(const uint32_t& word) {
    dprint_tensix_struct_field(word, 0xffff0000, 16, "y_stride", true);  // decimal
}

inline void dprint_tensix_pack_strides_z_stride(const uint32_t& word) {
    dprint_tensix_struct_field(word, 0xffff, 0, "z_stride", true);  // decimal
}

inline void dprint_tensix_pack_strides_w_stride(const uint32_t& word) {
    dprint_tensix_struct_field(word, 0xffff0000, 16, "w_stride", true);  // decimal
}
#endif

// Printing packer strides
inline void dprint_tensix_pack_strides_helper(uint reg_id, const volatile uint tt_reg_ptr* cfg) {
    uint32_t reg_addr = 0;
    switch (reg_id) {
        case 1: reg_addr = PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32; break;
        case 2: reg_addr = PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32; break;
        default:
            DPRINT << "Aborting! Invalid register id (valid ids are between 1 and 2)" << ENDL();
            DEVICE_PRINT("Aborting! Invalid register id (valid ids are between 1 and 2)\n");
            break;
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

// PCK_EDGE_OFFSET

// These function's argument should be return value of read_pack_edge_offset()

inline void dprint_tensix_pack_edge_offset_mask(const ckernel::packer::pck_edge_offset_t& edge) {
    DPRINT << "0x" << HEX() << edge.mask << ENDL();
    DEVICE_PRINT("0x{:08X}\n", edge.mask);
}

inline void dprint_tensix_pack_edge_offset_mode(const ckernel::packer::pck_edge_offset_t& edge) {
    DPRINT << "0x" << HEX() << edge.mode << ENDL();
    DEVICE_PRINT("0x{:08X}\n", edge.mode);
}

inline void dprint_tensix_pack_edge_offset_tile_row_set_select_pack0(const ckernel::packer::pck_edge_offset_t& edge) {
    DPRINT << "0x" << HEX() << edge.tile_row_set_select_pack0 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", edge.tile_row_set_select_pack0);
}

inline void dprint_tensix_pack_edge_offset_tile_row_set_select_pack1(const ckernel::packer::pck_edge_offset_t& edge) {
    DPRINT << "0x" << HEX() << edge.tile_row_set_select_pack1 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", edge.tile_row_set_select_pack1);
}

inline void dprint_tensix_pack_edge_offset_tile_row_set_select_pack2(const ckernel::packer::pck_edge_offset_t& edge) {
    DPRINT << "0x" << HEX() << edge.tile_row_set_select_pack2 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", edge.tile_row_set_select_pack2);
}

inline void dprint_tensix_pack_edge_offset_tile_row_set_select_pack3(const ckernel::packer::pck_edge_offset_t& edge) {
    DPRINT << "0x" << HEX() << edge.tile_row_set_select_pack3 << ENDL();
    DEVICE_PRINT("0x{:08X}\n", edge.tile_row_set_select_pack3);
}

inline void dprint_tensix_pack_edge_offset_reserved(const ckernel::packer::pck_edge_offset_t& edge) {
    DPRINT << "0x" << HEX() << edge.reserved << ENDL();
    DEVICE_PRINT("0x{:08X}\n", edge.reserved);
}

// Printing packer edge offset
inline void dprint_tensix_pack_edge_offset_helper(const ckernel::packer::pck_edge_offset_t& edge, uint reg_id) {
    DPRINT << "mask: ";
    DEVICE_PRINT("mask: ");
    dprint_tensix_pack_edge_offset_mask(edge);
    if (reg_id == 1) {
        DPRINT << "mode: ";
        DEVICE_PRINT("mode: ");
        dprint_tensix_pack_edge_offset_mode(edge);
        DPRINT << "tile_row_set_select_pack0: ";
        DEVICE_PRINT("tile_row_set_select_pack0: ");
        dprint_tensix_pack_edge_offset_tile_row_set_select_pack0(edge);
        DPRINT << "tile_row_set_select_pack1: ";
        DEVICE_PRINT("tile_row_set_select_pack1: ");
        dprint_tensix_pack_edge_offset_tile_row_set_select_pack1(edge);
        DPRINT << "tile_row_set_select_pack2: ";
        DEVICE_PRINT("tile_row_set_select_pack2: ");
        dprint_tensix_pack_edge_offset_tile_row_set_select_pack2(edge);
        DPRINT << "tile_row_set_select_pack3: ";
        DEVICE_PRINT("tile_row_set_select_pack3: ");
        dprint_tensix_pack_edge_offset_tile_row_set_select_pack3(edge);
        DPRINT << "reserved: ";
        DEVICE_PRINT("reserved: ");
        dprint_tensix_pack_edge_offset_reserved(edge);
    }
}

// Choose what register you want printed with reg_id (1-4), 0 for all
inline void dprint_tensix_pack_edge_offset(uint reg_id = 0) {
    std::array<ckernel::packer::pck_edge_offset_t, ckernel::packer::NUM_PACKERS> edge_vec;
    PACK(
        edge_vec = ckernel::packer::read_pack_edge_offset();
        if (reg_id >= 1 && reg_id <= ckernel::packer::NUM_PACKERS) {
            if (ckernel::packer::NUM_PACKERS > 1) {
                DPRINT << "REG_ID: " << reg_id << ENDL();
                DEVICE_PRINT("REG_ID: {}\n", reg_id);
            }
            dprint_tensix_pack_edge_offset_helper(edge_vec[reg_id - 1], reg_id);
        }
        // Print all registers
        else if (reg_id == 0) {
            for (uint i = 1; i <= ckernel::packer::NUM_PACKERS; i++) {
                if (ckernel::packer::NUM_PACKERS > 1) {
                    DPRINT << "REG_ID: " << i << ENDL();
                    DEVICE_PRINT("REG_ID: {}\n", i);
                }
                dprint_tensix_pack_edge_offset_helper(edge_vec[i - 1], i);
                if (i != ckernel::packer::NUM_PACKERS) {
                    DPRINT << ENDL();
                    DEVICE_PRINT("\n");
                }
            }
        } else {
            DPRINT << "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND " << ckernel::packer::NUM_PACKERS
                   << "." << ENDL();
            DEVICE_PRINT(
                "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND {}.\n", ckernel::packer::NUM_PACKERS);
        })
}

// PACK COUNTERS

// These functions' argument should be return value of read_pack_counters()

inline void dprint_tensix_pack_counters_pack_per_xy_plane(const ckernel::packer::pack_counters_t& counters) {
    DPRINT << DEC() << counters.pack_per_xy_plane << ENDL();
    DEVICE_PRINT("{}\n", counters.pack_per_xy_plane);
}

inline void dprint_tensix_pack_counters_pack_reads_per_xy_plane(const ckernel::packer::pack_counters_t& counters) {
    DPRINT << DEC() << counters.pack_reads_per_xy_plane << ENDL();
    DEVICE_PRINT("{}\n", counters.pack_reads_per_xy_plane);
}

inline void dprint_tensix_pack_counters_pack_xys_per_til(const ckernel::packer::pack_counters_t& counters) {
    DPRINT << DEC() << counters.pack_xys_per_til << ENDL();
    DEVICE_PRINT("{}\n", counters.pack_xys_per_til);
}

inline void dprint_tensix_pack_counters_pack_yz_transposed(const ckernel::packer::pack_counters_t& counters) {
    DPRINT << "0x" << HEX() << counters.pack_yz_transposed << ENDL();
    DEVICE_PRINT("0x{:X}\n", counters.pack_yz_transposed);
}

inline void dprint_tensix_pack_counters_pack_per_xy_plane_offset(const ckernel::packer::pack_counters_t& counters) {
    DPRINT << DEC() << counters.pack_per_xy_plane_offset << ENDL();
    DEVICE_PRINT("{}\n", counters.pack_per_xy_plane_offset);
}

// Printing packer counters
inline void dprint_tensix_pack_counters_helper(const ckernel::packer::pack_counters_t& counters) {
    DPRINT << "pack_per_xy_plane: ";
    DEVICE_PRINT("pack_per_xy_plane: ");
    dprint_tensix_pack_counters_pack_per_xy_plane(counters);
    DPRINT << "pack_reads_per_xy_plane: ";
    DEVICE_PRINT("pack_reads_per_xy_plane: ");
    dprint_tensix_pack_counters_pack_reads_per_xy_plane(counters);
    DPRINT << "pack_xys_per_til: ";
    DEVICE_PRINT("pack_xys_per_til: ");
    dprint_tensix_pack_counters_pack_xys_per_til(counters);
    DPRINT << "pack_yz_transposed: ";
    DEVICE_PRINT("pack_yz_transposed: ");
    dprint_tensix_pack_counters_pack_yz_transposed(counters);
    DPRINT << "pack_per_xy_plane_offset: ";
    DEVICE_PRINT("pack_per_xy_plane_offset: ");
    dprint_tensix_pack_counters_pack_per_xy_plane_offset(counters);
}

// Choose what register you want printed with reg_id (1-4), 0 for all
inline void dprint_tensix_pack_counters(uint reg_id = 0) {
    std::array<ckernel::packer::pack_counters_t, ckernel::packer::NUM_PACKERS> counters_vec;
    PACK(
        counters_vec = ckernel::packer::read_pack_counters();
        if (reg_id >= 1 && reg_id <= ckernel::packer::NUM_PACKERS) {
            if (ckernel::packer::NUM_PACKERS > 1) {
                DPRINT << "REG_ID: " << reg_id << ENDL();
                DEVICE_PRINT("REG_ID: {}\n", reg_id);
            }
            dprint_tensix_pack_counters_helper(counters_vec[reg_id - 1]);
        }
        // Print all registers
        else if (reg_id == 0) {
            for (uint i = 1; i <= ckernel::packer::NUM_PACKERS; i++) {
                if (ckernel::packer::NUM_PACKERS > 1) {
                    DPRINT << "REG_ID: " << i << ENDL();
                    DEVICE_PRINT("REG_ID: {}\n", i);
                }
                dprint_tensix_pack_counters_helper(counters_vec[i - 1]);
                if (i != ckernel::packer::NUM_PACKERS) {
                    DPRINT << ENDL();
                    DEVICE_PRINT("\n");
                }
            }
        } else {
            DPRINT << "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND " << ckernel::packer::NUM_PACKERS
                   << "." << ENDL();
            DEVICE_PRINT(
                "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND {}.\n", ckernel::packer::NUM_PACKERS);
        })
}

// Choose what register you want by id (1-4). 0 for all.
inline void dprint_tensix_pack_config(uint reg_id = 0) {
    std::array<ckernel::packer::pack_config_t, ckernel::packer::NUM_PACKERS> config_vec;
    MATH(
        config_vec = ckernel::packer::read_pack_config(); if (reg_id >= 1 && reg_id <= ckernel::packer::NUM_PACKERS) {
            if (ckernel::packer::NUM_PACKERS > 1) {
                DPRINT << "REG_ID: " << reg_id << ENDL();
                DEVICE_PRINT("REG_ID: {}\n", reg_id);
            }
            dprint_tensix_pack_config_helper(config_vec[reg_id - 1]);
        } else if (reg_id == 0) for (uint i = 1; i <= ckernel::packer::NUM_PACKERS; i++) {
            if (ckernel::packer::NUM_PACKERS > 1) {
                DPRINT << "REG_ID: " << i << ENDL();
                DEVICE_PRINT("REG_ID: {}\n", i);
            }
            dprint_tensix_pack_config_helper(config_vec[i - 1]);
            if (i != ckernel::packer::NUM_PACKERS) {
                DPRINT << ENDL();
                DEVICE_PRINT("\n");
            }
        } else {
            DPRINT << "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND " << ckernel::packer::NUM_PACKERS
                   << "." << ENDL();
            DEVICE_PRINT(
                "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND {}.\n", ckernel::packer::NUM_PACKERS);
        })
}

// Choose what register you want printed (1-2). 0 for all.
inline void dprint_tensix_pack_strides(uint reg_id = 0) {
    PACK(
        // Get pointer to registers for current state ID
        volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

        if (reg_id >= 1 && reg_id <= 2) {
            DPRINT << "REG_ID: " << reg_id << ENDL();
            DEVICE_PRINT("REG_ID: {}\n", reg_id);
            dprint_tensix_pack_strides_helper(reg_id, cfg);
        }
        // Print all registers
        else if (reg_id == 0) {
            for (uint i = 1; i <= 2; i++) {
                DPRINT << "REG_ID: " << i << ENDL();
                DEVICE_PRINT("REG_ID: {}\n", i);
                dprint_tensix_pack_strides_helper(i, cfg);
                if (i != 2) {
                    DPRINT << ENDL();
                    DEVICE_PRINT("\n");
                }
            }
        } else {
            DPRINT << "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND 2." << ENDL();
            DEVICE_PRINT("INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND 2.\n");
        })
}
