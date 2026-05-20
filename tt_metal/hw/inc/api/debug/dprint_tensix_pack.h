// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
    DEVICE_PRINT("{}\n", config.row_ptr_section_size);
}

inline void dprint_tensix_pack_config_exp_section_size(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("{}\n", config.exp_section_size);
}

inline void dprint_tensix_pack_config_l1_dest_addr(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.l1_dest_addr);
}

inline void dprint_tensix_pack_config_uncompressed(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.uncompress);
}

inline void dprint_tensix_pack_config_add_l1_dest_addr_offset(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.add_l1_dest_addr_offset);
}

inline void dprint_tensix_pack_config_reserved_0(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.reserved_0);
}

inline void dprint_tensix_pack_config_out_data_format(const ckernel::packer::pack_config_t& config) {
    dprint_data_format(config.out_data_format);
    DEVICE_PRINT("\n");
}

inline void dprint_tensix_pack_config_in_data_format(const ckernel::packer::pack_config_t& config) {
    dprint_data_format(config.in_data_format);
    DEVICE_PRINT("\n");
}

#if defined(ARCH_WORMHOLE)
inline void dprint_tensix_pack_config_reserved_1(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.reserved_1);
}
#endif

inline void dprint_tensix_pack_config_src_if_sel(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.src_if_sel);
}

#if defined(ARCH_WORMHOLE)
inline void dprint_tensix_pack_config_pack_per_xy_plane(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("{}\n", config.pack_per_xy_plane);
}
#endif

inline void dprint_tensix_pack_config_l1_src_addr(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.l1_src_addr);
}

#if defined(ARCH_WORMHOLE)
inline void dprint_tensix_pack_config_downsample_mask(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.downsample_mask);
}

inline void dprint_tensix_pack_config_downsample_shift_count(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("{}\n", config.downsample_shift_count);
}

inline void dprint_tensix_pack_config_read_mode(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.read_mode);
}

inline void dprint_tensix_pack_config_exp_threshold_en(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.exp_threshold_en);
}

inline void dprint_tensix_pack_config_reserved_2(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.reserved_2);
}

inline void dprint_tensix_pack_config_exp_threshold(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("{}\n", config.exp_threshold);
}
#endif

#ifdef ARCH_WORMHOLE
inline void dprint_tensix_pack_config_l1_acc_disable_pack_zero_flag(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.pack_l1_acc_disable_pack_zero_flag);
}
#endif

#ifdef ARCH_BLACKHOLE
inline void dprint_tensix_pack_config_disable_pack_zero_flag(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.disable_pack_zero_flag);
}

inline void dprint_tensix_pack_config_dis_shared_exp_assembler(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.dis_shared_exp_assembler);
}

inline void dprint_tensix_pack_config_auto_set_last_pacr_intf_sel(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.auto_set_last_pacr_intf_sel);
}

inline void dprint_tensix_pack_config_enable_out_fifo(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.enable_out_fifo);
}

inline void dprint_tensix_pack_config_sub_l1_tile_header_size(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("{}\n", config.sub_l1_tile_header_size);
}

inline void dprint_tensix_pack_config_pack_start_intf_pos(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.pack_start_intf_pos);
}

inline void dprint_tensix_pack_config_all_pack_disable_zero_compress_ovrd(
    const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.all_pack_disable_zero_compress_ovrd);
}

inline void dprint_tensix_pack_config_add_tile_header_size(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("{}\n", config.add_tile_header_size);
}

inline void dprint_tensix_pack_config_pack_dis_y_pos_start_offset(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.pack_dis_y_pos_start_offset);
}
#endif

#ifdef ARCH_WORMHOLE
inline void dprint_tensix_pack_config_helper(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("row_ptr_section_size: ");
    dprint_tensix_pack_config_row_ptr_section_size(config);
    DEVICE_PRINT("exp_section_size: ");
    dprint_tensix_pack_config_exp_section_size(config);
    DEVICE_PRINT("l1_dest_addr: ");
    dprint_tensix_pack_config_l1_dest_addr(config);
    DEVICE_PRINT("uncompress: ");
    dprint_tensix_pack_config_uncompressed(config);
    DEVICE_PRINT("add_l1_dest_addr_offset: ");
    dprint_tensix_pack_config_add_l1_dest_addr_offset(config);
    DEVICE_PRINT("reserved_0: ");
    dprint_tensix_pack_config_reserved_0(config);
    DEVICE_PRINT("out_data_format: ");
    dprint_tensix_pack_config_out_data_format(config);
    DEVICE_PRINT("in_data_format: ");
    dprint_tensix_pack_config_in_data_format(config);
    DEVICE_PRINT("reserved_1: ");
    dprint_tensix_pack_config_reserved_1(config);
    DEVICE_PRINT("src_if_sel: ");
    dprint_tensix_pack_config_src_if_sel(config);
    DEVICE_PRINT("pack_per_xy_plane: ");
    dprint_tensix_pack_config_pack_per_xy_plane(config);
    DEVICE_PRINT("l1_src_addr: ");
    dprint_tensix_pack_config_l1_src_addr(config);
    DEVICE_PRINT("downsample_mask: ");
    dprint_tensix_pack_config_downsample_mask(config);
    DEVICE_PRINT("downsample_shift_count: ");
    dprint_tensix_pack_config_downsample_shift_count(config);
    DEVICE_PRINT("read_mode: ");
    dprint_tensix_pack_config_read_mode(config);
    DEVICE_PRINT("exp_threshold_en: ");
    dprint_tensix_pack_config_exp_threshold_en(config);
    DEVICE_PRINT("pack_l1_acc_disable_pack_zero_flag: ");
    dprint_tensix_pack_config_l1_acc_disable_pack_zero_flag(config);
    DEVICE_PRINT("reserved_2: ");
    dprint_tensix_pack_config_reserved_2(config);
    DEVICE_PRINT("exp_threshold: ");
    dprint_tensix_pack_config_exp_threshold(config);
}
#endif  // ARCH_WORMHOLE

#ifdef ARCH_BLACKHOLE
inline void dprint_tensix_pack_config_helper(const ckernel::packer::pack_config_t& config) {
    DEVICE_PRINT("row_ptr_section_size: ");
    dprint_tensix_pack_config_row_ptr_section_size(config);
    DEVICE_PRINT("exp_section_size: ");
    dprint_tensix_pack_config_exp_section_size(config);
    DEVICE_PRINT("l1_dest_addr: ");
    dprint_tensix_pack_config_l1_dest_addr(config);
    DEVICE_PRINT("uncompress: ");
    dprint_tensix_pack_config_uncompressed(config);
    DEVICE_PRINT("add_l1_dest_addr_offset: ");
    dprint_tensix_pack_config_add_l1_dest_addr_offset(config);
    DEVICE_PRINT("disable_pack_zero_flag: ");
    dprint_tensix_pack_config_disable_pack_zero_flag(config);
    DEVICE_PRINT("reserved_0: ");
    dprint_tensix_pack_config_reserved_0(config);
    DEVICE_PRINT("out_data_format: ");
    dprint_tensix_pack_config_out_data_format(config);
    DEVICE_PRINT("in_data_format: ");
    dprint_tensix_pack_config_in_data_format(config);
    DEVICE_PRINT("dis_shared_exp_assembler: ");
    dprint_tensix_pack_config_dis_shared_exp_assembler(config);
    DEVICE_PRINT("auto_set_last_pacr_intf_sel: ");
    dprint_tensix_pack_config_auto_set_last_pacr_intf_sel(config);
    DEVICE_PRINT("enable_out_fifo: ");
    dprint_tensix_pack_config_enable_out_fifo(config);
    DEVICE_PRINT("sub_l1_tile_header_size: ");
    dprint_tensix_pack_config_sub_l1_tile_header_size(config);
    DEVICE_PRINT("src_if_sel: ");
    dprint_tensix_pack_config_src_if_sel(config);
    DEVICE_PRINT("pack_start_intf_pos: ");
    dprint_tensix_pack_config_pack_start_intf_pos(config);
    DEVICE_PRINT("all_pack_disable_zero_compress_ovrd: ");
    dprint_tensix_pack_config_all_pack_disable_zero_compress_ovrd(config);
    DEVICE_PRINT("add_tile_header_size: ");
    dprint_tensix_pack_config_add_tile_header_size(config);
    DEVICE_PRINT("pack_dis_y_pos_start_offset: ");
    dprint_tensix_pack_config_pack_dis_y_pos_start_offset(config);
    DEVICE_PRINT("l1_src_addr: ");
    dprint_tensix_pack_config_l1_src_addr(config);
}
#endif  // ARCH_BLACKHOLE

// PACK RELU CONFIG

// These functions' argument should be return value of read_relu_config()

inline void dprint_tensix_pack_relu_config_alu_acc_ctrl_zero_flag_disabled_src(
    const ckernel::packer::relu_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.ALU_ACC_CTRL_Zero_Flag_disabled_src);
}

inline void dprint_tensix_pack_relu_config_alu_acc_ctrl_zero_flag_disabled_dst(
    const ckernel::packer::relu_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.ALU_ACC_CTRL_Zero_Flag_disabled_dst);
}

inline void dprint_tensix_pack_relu_config_stacc_relu_apply_relu(const ckernel::packer::relu_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.STACC_RELU_ApplyRelu);
}

inline void dprint_tensix_pack_relu_config_stacc_relu_relu_threshold(const ckernel::packer::relu_config_t& config) {
    DEVICE_PRINT("{}\n", config.STACC_RELU_ReluThreshold);
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_main(const ckernel::packer::relu_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.DISABLE_RISC_BP_Disable_main);
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_trisc(const ckernel::packer::relu_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.DISABLE_RISC_BP_Disable_trisc);
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_ncrisc(
    const ckernel::packer::relu_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.DISABLE_RISC_BP_Disable_ncrisc);
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_main(
    const ckernel::packer::relu_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.DISABLE_RISC_BP_Disable_bmp_clear_main);
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_trisc(
    const ckernel::packer::relu_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.DISABLE_RISC_BP_Disable_bmp_clear_trisc);
}

inline void dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_ncrisc(
    const ckernel::packer::relu_config_t& config) {
    DEVICE_PRINT("0x{:x}\n", config.DISABLE_RISC_BP_Disable_bmp_clear_ncrisc);
}

inline void dprint_tensix_pack_relu_config() {
    MATH(ckernel::packer::relu_config_t config = ckernel::packer::read_relu_config();

         DEVICE_PRINT("ALU_ACC_CTRL_Zero_Flag_disabled_src: ");
         dprint_tensix_pack_relu_config_alu_acc_ctrl_zero_flag_disabled_src(config);
         DEVICE_PRINT("ALU_ACC_CTRL_Zero_Flag_disabled_dst: ");
         dprint_tensix_pack_relu_config_alu_acc_ctrl_zero_flag_disabled_dst(config);
         DEVICE_PRINT("STACC_RELU_ApplyRelu: ");
         dprint_tensix_pack_relu_config_stacc_relu_apply_relu(config);
         DEVICE_PRINT("STACC_RELU_ReluThreshold: ");
         dprint_tensix_pack_relu_config_stacc_relu_relu_threshold(config);
         DEVICE_PRINT("DISABLE_RISC_BP_Disable_main: ");
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_main(config);
         DEVICE_PRINT("DISABLE_RISC_BP_Disable_trisc: ");
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_trisc(config);
         DEVICE_PRINT("DISABLE_RISC_BP_Disable_ncrisc: ");
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_ncrisc(config);
         DEVICE_PRINT("DISABLE_RISC_BP_Disable_bmp_clear_main: ");
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_main(config);
         DEVICE_PRINT("DISABLE_RISC_BP_Disable_bmp_clear_trisc: ");
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_trisc(config);
         DEVICE_PRINT("DISABLE_RISC_BP_Disable_bmp_clear_ncrisc: ");
         dprint_tensix_pack_relu_config_disable_risc_bp_disable_bmp_clear_ncrisc(config);)
}

// PACK DEST RD CTRL

// These functions' argument should be return value of read_dest_rd_ctrl()

inline void dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_32b_data(
    const ckernel::packer::dest_rd_ctrl_t& dest) {
    DEVICE_PRINT("0x{:x}\n", dest.PCK_DEST_RD_CTRL_Read_32b_data);
}

inline void dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_unsigned(
    const ckernel::packer::dest_rd_ctrl_t& dest) {
    DEVICE_PRINT("0x{:x}\n", dest.PCK_DEST_RD_CTRL_Read_unsigned);
}

inline void dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_int8(const ckernel::packer::dest_rd_ctrl_t& dest) {
    DEVICE_PRINT("0x{:x}\n", dest.PCK_DEST_RD_CTRL_Read_int8);
}

inline void dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_round_10b_mant(
    const ckernel::packer::dest_rd_ctrl_t& dest) {
    DEVICE_PRINT("0x{:x}\n", dest.PCK_DEST_RD_CTRL_Round_10b_mant);
}

inline void dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_reserved(const ckernel::packer::dest_rd_ctrl_t& dest) {
    DEVICE_PRINT("0x{:x}\n", dest.PCK_DEST_RD_CTRL_Reserved);
}

// Printing dest control bits
inline void dprint_tensix_dest_rd_ctrl() {
    PACK(ckernel::packer::dest_rd_ctrl_t dest = ckernel::packer::read_dest_rd_ctrl();

         DEVICE_PRINT("PCK_DEST_RD_CTRL_Read_32b_data: ");
         dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_32b_data(dest);
         DEVICE_PRINT("PCK_DEST_RD_CTRL_Read_unsigned: ");
         dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_unsigned(dest);
         DEVICE_PRINT("PCK_DEST_RD_CTRL_Read_int8: ");
         dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_read_int8(dest);
         DEVICE_PRINT("PCK_DEST_RD_CTRL_Round_10b_mant: ");
         dprint_tensix_pack_dest_rd_ctrl_pck_dest_rd_ctrl_round_10b_mant(dest);
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
        default: DEVICE_PRINT("Aborting! Invalid register id (valid ids are between 1 and 2)\n"); break;
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
    DEVICE_PRINT("0x{:x}\n", edge.mask);
}

inline void dprint_tensix_pack_edge_offset_mode(const ckernel::packer::pck_edge_offset_t& edge) {
    DEVICE_PRINT("0x{:x}\n", edge.mode);
}

inline void dprint_tensix_pack_edge_offset_tile_row_set_select_pack0(const ckernel::packer::pck_edge_offset_t& edge) {
    DEVICE_PRINT("0x{:x}\n", edge.tile_row_set_select_pack0);
}

inline void dprint_tensix_pack_edge_offset_tile_row_set_select_pack1(const ckernel::packer::pck_edge_offset_t& edge) {
    DEVICE_PRINT("0x{:x}\n", edge.tile_row_set_select_pack1);
}

inline void dprint_tensix_pack_edge_offset_tile_row_set_select_pack2(const ckernel::packer::pck_edge_offset_t& edge) {
    DEVICE_PRINT("0x{:x}\n", edge.tile_row_set_select_pack2);
}

inline void dprint_tensix_pack_edge_offset_tile_row_set_select_pack3(const ckernel::packer::pck_edge_offset_t& edge) {
    DEVICE_PRINT("0x{:x}\n", edge.tile_row_set_select_pack3);
}

inline void dprint_tensix_pack_edge_offset_reserved(const ckernel::packer::pck_edge_offset_t& edge) {
    DEVICE_PRINT("0x{:x}\n", edge.reserved);
}

// Printing packer edge offset
inline void dprint_tensix_pack_edge_offset_helper(const ckernel::packer::pck_edge_offset_t& edge, uint reg_id) {
    DEVICE_PRINT("mask: ");
    dprint_tensix_pack_edge_offset_mask(edge);
    if (reg_id == 1) {
        DEVICE_PRINT("mode: ");
        dprint_tensix_pack_edge_offset_mode(edge);
        DEVICE_PRINT("tile_row_set_select_pack0: ");
        dprint_tensix_pack_edge_offset_tile_row_set_select_pack0(edge);
        DEVICE_PRINT("tile_row_set_select_pack1: ");
        dprint_tensix_pack_edge_offset_tile_row_set_select_pack1(edge);
        DEVICE_PRINT("tile_row_set_select_pack2: ");
        dprint_tensix_pack_edge_offset_tile_row_set_select_pack2(edge);
        DEVICE_PRINT("tile_row_set_select_pack3: ");
        dprint_tensix_pack_edge_offset_tile_row_set_select_pack3(edge);
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
                DEVICE_PRINT("REG_ID: {}\n", reg_id);
            }
            dprint_tensix_pack_edge_offset_helper(edge_vec[reg_id - 1], reg_id);
        }
        // Print all registers
        else if (reg_id == 0) {
            for (uint i = 1; i <= ckernel::packer::NUM_PACKERS; i++) {
                if (ckernel::packer::NUM_PACKERS > 1) {
                    DEVICE_PRINT("REG_ID: {}\n", i);
                }
                dprint_tensix_pack_edge_offset_helper(edge_vec[i - 1], i);
                if (i != ckernel::packer::NUM_PACKERS) {
                    DEVICE_PRINT("\n");
                }
            }
        } else {
            DEVICE_PRINT(
                "INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND {}.\n", ckernel::packer::NUM_PACKERS);
        })
}

// PACK COUNTERS

// These functions' argument should be return value of read_pack_counters()

inline void dprint_tensix_pack_counters_pack_per_xy_plane(const ckernel::packer::pack_counters_t& counters) {
    DEVICE_PRINT("{}\n", counters.pack_per_xy_plane);
}

inline void dprint_tensix_pack_counters_pack_reads_per_xy_plane(const ckernel::packer::pack_counters_t& counters) {
    DEVICE_PRINT("{}\n", counters.pack_reads_per_xy_plane);
}

inline void dprint_tensix_pack_counters_pack_xys_per_til(const ckernel::packer::pack_counters_t& counters) {
    DEVICE_PRINT("{}\n", counters.pack_xys_per_til);
}

inline void dprint_tensix_pack_counters_pack_yz_transposed(const ckernel::packer::pack_counters_t& counters) {
    DEVICE_PRINT("0x{:x}\n", counters.pack_yz_transposed);
}

inline void dprint_tensix_pack_counters_pack_per_xy_plane_offset(const ckernel::packer::pack_counters_t& counters) {
    DEVICE_PRINT("{}\n", counters.pack_per_xy_plane_offset);
}

// Printing packer counters
inline void dprint_tensix_pack_counters_helper(const ckernel::packer::pack_counters_t& counters) {
    DEVICE_PRINT("pack_per_xy_plane: ");
    dprint_tensix_pack_counters_pack_per_xy_plane(counters);
    DEVICE_PRINT("pack_reads_per_xy_plane: ");
    dprint_tensix_pack_counters_pack_reads_per_xy_plane(counters);
    DEVICE_PRINT("pack_xys_per_til: ");
    dprint_tensix_pack_counters_pack_xys_per_til(counters);
    DEVICE_PRINT("pack_yz_transposed: ");
    dprint_tensix_pack_counters_pack_yz_transposed(counters);
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
                DEVICE_PRINT("REG_ID: {}\n", reg_id);
            }
            dprint_tensix_pack_counters_helper(counters_vec[reg_id - 1]);
        }
        // Print all registers
        else if (reg_id == 0) {
            for (uint i = 1; i <= ckernel::packer::NUM_PACKERS; i++) {
                if (ckernel::packer::NUM_PACKERS > 1) {
                    DEVICE_PRINT("REG_ID: {}\n", i);
                }
                dprint_tensix_pack_counters_helper(counters_vec[i - 1]);
                if (i != ckernel::packer::NUM_PACKERS) {
                    DEVICE_PRINT("\n");
                }
            }
        } else {
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
                DEVICE_PRINT("REG_ID: {}\n", reg_id);
            }
            dprint_tensix_pack_config_helper(config_vec[reg_id - 1]);
        } else if (reg_id == 0) for (uint i = 1; i <= ckernel::packer::NUM_PACKERS; i++) {
            if (ckernel::packer::NUM_PACKERS > 1) {
                DEVICE_PRINT("REG_ID: {}\n", i);
            }
            dprint_tensix_pack_config_helper(config_vec[i - 1]);
            if (i != ckernel::packer::NUM_PACKERS) {
                DEVICE_PRINT("\n");
            }
        } else {
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
            DEVICE_PRINT("REG_ID: {}\n", reg_id);
            dprint_tensix_pack_strides_helper(reg_id, cfg);
        }
        // Print all registers
        else if (reg_id == 0) {
            for (uint i = 1; i <= 2; i++) {
                DEVICE_PRINT("REG_ID: {}\n", i);
                dprint_tensix_pack_strides_helper(i, cfg);
                if (i != 2) {
                    DEVICE_PRINT("\n");
                }
            }
        } else { DEVICE_PRINT("INVALID REGISTER ID! PLEASE CHOOSE A NUMBER BETWEEN 0 AND 2.\n"); })
}
