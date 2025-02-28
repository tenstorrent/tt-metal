// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint_tensix_pack.h"
#include "debug/dprint_tensix_unpack.h"
#include "ckernel.h"

#include <array>

// Register names
#define ALU_CONFIG 0
#define UNPACK_TILE_DESCRIPTOR 1
#define UNPACK_CONFIG 2
#define PACK_CONFIG 3
#define RELU_CONFIG 4
#define DEST_RD_CTRL 5
#define PACK_EDGE_OFFSET 6
#define PACK_COUNTERS 7
#define PACK_STRIDES 8

namespace NAMESPACE {
#if defined(ARCH_WORMHOLE) or defined(ARCH_BLACKHOLE)
void generate_alu_config(ckernel::unpacker::alu_config_t& config) {
    config.ALU_ROUNDING_MODE_Fpu_srnd_en = 1;
    config.ALU_ROUNDING_MODE_Gasket_srnd_en = 0;
    config.ALU_ROUNDING_MODE_Packer_srnd_en = 1;
    config.ALU_ROUNDING_MODE_Padding = 15;
    config.ALU_ROUNDING_MODE_GS_LF = 0;
    config.ALU_ROUNDING_MODE_Bfp8_HF = 1;
    config.ALU_FORMAT_SPEC_REG0_SrcAUnsigned = 1;
    config.ALU_FORMAT_SPEC_REG0_SrcBUnsigned = 0;
    config.ALU_FORMAT_SPEC_REG0_SrcA = 0;
    config.ALU_FORMAT_SPEC_REG1_SrcB = 1;
    config.ALU_FORMAT_SPEC_REG2_Dstacc = 0;
    config.ALU_ACC_CTRL_Fp32_enabled = 0;
    config.ALU_ACC_CTRL_SFPU_Fp32_enabled = 0;
    config.ALU_ACC_CTRL_INT8_math_enabled = 1;
}
#endif

void generate_unpack_tile_descriptor(ckernel::unpacker::unpack_tile_descriptor_t& tile_descriptor) {
    tile_descriptor.in_data_format = 5;
    tile_descriptor.uncompressed = 1;
    tile_descriptor.reserved_0 = 0;
    tile_descriptor.blobs_per_xy_plane = 10;
    tile_descriptor.reserved_1 = 7;
    tile_descriptor.x_dim = 2;
    tile_descriptor.y_dim = 4;
    tile_descriptor.z_dim = 8;
    tile_descriptor.w_dim = 16;
#ifdef ARCH_GRAYSKULL
    tile_descriptor.blobs_y_start = 32;
#else  // ARCH_WORMHOLE or ARCH_BLACKHOLE
    tile_descriptor.blobs_y_start_lo = 32;
    tile_descriptor.blobs_y_start_hi = 0;
#endif
    tile_descriptor.digest_type = 0;
    tile_descriptor.digest_size = 0;
}

void generate_unpack_config(ckernel::unpacker::unpack_config_t& config) {
    config.out_data_format = 0;
    config.throttle_mode = 1;
    config.context_count = 2;
    config.haloize_mode = 0;
    config.tileize_mode = 1;
    config.upsample_rate = 3;
    config.reserved_1 = 0;
    config.upsamle_and_interlave = 0;
    config.shift_amount = 16;
    config.uncompress_cntx0_3 = 5;
    config.force_shared_exp = 0;
    config.reserved_2 = 0;
    config.uncompress_cntx4_7 = 2;
    config.limit_addr = 28;
    config.fifo_size = 29;

#ifdef ARCH_GRAYSKULL
    config.reserved_0 = 0;
#else  // ARCH_WORMHOLE or ARCH_BLACKHOLE
    config.reserved_3 = 0;
    config.reserved_4 = 0;
    config.reserved_5 = 0;
    config.unpack_if_sel_cntx0_3 = 6;
    config.unpack_if_sel_cntx4_7 = 3;
    config.unpack_src_reg_set_update = 1;
    config.unpack_if_sel = 0;
#endif
}

void generate_pack_config(ckernel::packer::pack_config_t& config) {
    config.row_ptr_section_size = 12;
    config.exp_section_size = 24;
    config.l1_dest_addr = 16;
    config.uncompress = 0;
    config.add_l1_dest_addr_offset = 1;
    config.reserved_0 = 0;
    config.out_data_format = 5;
    config.in_data_format = 5;
    config.src_if_sel = 1;
    config.l1_src_addr = 8;
#if defined(ARCH_WORMHOLE) or defined(ARCH_GRAYSKULL)
    config.reserved_1 = 0;
    config.pack_per_xy_plane = 0;
    config.downsample_mask = 12;
    config.downsample_shift_count = 4;
    config.read_mode = 0;
    config.exp_threshold_en = 1;
#ifdef ARCH_WORMHOLE
    config.pack_l1_acc_disable_pack_zero_flag = 2;
#endif
    config.reserved_2 = 0;
    config.exp_threshold = 12;
#endif
#ifdef ARCH_BLACKHOLE
    config.disable_pack_zero_flag = 1;
    config.dis_shared_exp_assembler = 0;
    config.auto_set_last_pacr_intf_sel = 0;
    config.enable_out_fifo = 1;
    config.sub_l1_tile_header_size = 0;
    config.pack_start_intf_pos = 2;
    config.all_pack_disable_zero_compress_ovrd = 0;
    config.add_tile_header_size = 1;
    config.pack_dis_y_pos_start_offset = 0;
#endif
}

#if defined(ARCH_WORMHOLE) or defined(ARCH_BLACKHOLE)
void generate_relu_config(ckernel::packer::relu_config_t& config) {
    config.ALU_ACC_CTRL_Zero_Flag_disabled_src = 0;
    config.ALU_ACC_CTRL_Zero_Flag_disabled_dst = 0;
    config.STACC_RELU_ApplyRelu = 1;
    config.STACC_RELU_ReluThreshold = 8;
    config.DISABLE_RISC_BP_Disable_main = 0;
    config.DISABLE_RISC_BP_Disable_trisc = 0;
    config.DISABLE_RISC_BP_Disable_ncrisc = 0;
    config.DISABLE_RISC_BP_Disable_bmp_clear_main = 0;
    config.DISABLE_RISC_BP_Disable_bmp_clear_trisc = 0;
    config.DISABLE_RISC_BP_Disable_bmp_clear_ncrisc = 0;
}
#endif

#if defined(ARCH_WORMHOLE) or defined(ARCH_BLACKHOLE)
void generate_dest_rd_ctrl(ckernel::packer::dest_rd_ctrl_t& dest) {
    dest.PCK_DEST_RD_CTRL_Read_32b_data = 1;
    dest.PCK_DEST_RD_CTRL_Read_unsigned = 0;
    dest.PCK_DEST_RD_CTRL_Read_int8 = 1;
    dest.PCK_DEST_RD_CTRL_Round_10b_mant = 1;
    dest.PCK_DEST_RD_CTRL_Reserved = 0;
}
#endif

void generate_pack_edge_offset(ckernel::packer::pck_edge_offset_t& edge) {
    edge.mask = 16;
    edge.mode = 1;
    edge.tile_row_set_select_pack0 = 0;
    edge.tile_row_set_select_pack1 = 1;
    edge.tile_row_set_select_pack2 = 2;
    edge.tile_row_set_select_pack3 = 3;
    edge.reserved = 0;
}

void generate_pack_counters(ckernel::packer::pack_counters_t& counter) {
    counter.pack_per_xy_plane = 4;
    counter.pack_reads_per_xy_plane = 8;
    counter.pack_xys_per_til = 2;
    counter.pack_yz_transposed = 0;
    counter.pack_per_xy_plane_offset = 6;
}

#if defined(ARCH_WORMHOLE) or defined(ARCH_BLACKHOLE)
void write_alu_config(volatile uint tt_reg_ptr* cfg, uint32_t address, const ckernel::unpacker::alu_config_u& config) {
    MATH(cfg[address] = config.val;)
}
#endif

void write_unpack_tile_descriptor(
    volatile uint tt_reg_ptr* cfg,
    uint32_t address,
    uint num_of_words,
    const ckernel::unpacker::unpack_tile_descriptor_u& tile_descriptor) {
    UNPACK(for (uint i = 0; i < num_of_words; i++) { cfg[address + i] = tile_descriptor.val[i]; })
}

void write_unpack_config(
    volatile uint tt_reg_ptr* cfg,
    uint32_t address,
    uint num_of_words,
    const ckernel::unpacker::unpack_config_u& config) {
    UNPACK(for (uint i = 0; i < num_of_words; i++) { cfg[address + i] = config.val[i]; })
}

void write_pack_config(
    volatile uint tt_reg_ptr* cfg, uint32_t address, uint num_of_words, const ckernel::packer::pack_config_u& config) {
    MATH(for (uint i = 0; i < num_of_words; i++) { cfg[address + i] = config.val[i]; })
}

#if defined(ARCH_WORMHOLE) or defined(ARCH_BLACKHOLE)
void write_relu_config(
    volatile uint tt_reg_ptr* cfg, uint32_t address, uint num_of_words, const ckernel::packer::relu_config_u& config) {
    MATH(for (uint i = 0; i < num_of_words; i++) { cfg[address + i] = config.val[i]; })
}
#endif

#if defined(ARCH_WORMHOLE) or defined(ARCH_BLACKHOLE)
void write_dest_rd_ctrl(volatile uint tt_reg_ptr* cfg, uint32_t address, const ckernel::packer::dest_rd_ctrl_u& dest) {
    PACK(cfg[address] = dest.val;)
}
#endif

void write_pack_edge_offset(
    volatile uint tt_reg_ptr* cfg, uint32_t address, const ckernel::packer::pck_edge_offset_u& edge) {
    PACK(cfg[address] = edge.val;)
}

void write_pack_counters(
    volatile uint tt_reg_ptr* cfg, uint32_t address, const ckernel::packer::pack_counters_u& counter) {
    PACK(cfg[address] = counter.val;)
}

void MAIN {
    uint32_t register_name = get_compile_time_arg_val(0);

    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    switch (register_name) {
#if defined(ARCH_WORMHOLE) or defined(ARCH_BLACKHOLE)
        case ALU_CONFIG:
            ckernel::unpacker::alu_config_u alu_config;
            generate_alu_config(alu_config.f);
            ckernel::unpacker::alu_config_u alu_config_original;
            MATH(alu_config_original.f = ckernel::unpacker::read_alu_config();)
            write_alu_config(cfg, ALU_ROUNDING_MODE_Fpu_srnd_en_ADDR32, alu_config);

            dprint_tensix_alu_config();

            write_alu_config(cfg, ALU_ROUNDING_MODE_Fpu_srnd_en_ADDR32, alu_config_original);
            break;
#endif
        case UNPACK_TILE_DESCRIPTOR:
            ckernel::unpacker::unpack_tile_descriptor_u tile_descriptor;
            generate_unpack_tile_descriptor(tile_descriptor.f);
            std::array<ckernel::unpacker::unpack_tile_descriptor_t, ckernel::unpacker::NUM_UNPACKERS>
                tile_descriptor_vec;
            UNPACK(tile_descriptor_vec = ckernel::unpacker::read_unpack_tile_descriptor();)
            write_unpack_tile_descriptor(cfg, THCON_SEC0_REG0_TileDescriptor_ADDR32, 4, tile_descriptor);
            write_unpack_tile_descriptor(cfg, THCON_SEC1_REG0_TileDescriptor_ADDR32, 4, tile_descriptor);

            dprint_tensix_unpack_tile_descriptor();

            tile_descriptor.f = tile_descriptor_vec[0];
            write_unpack_tile_descriptor(cfg, THCON_SEC0_REG0_TileDescriptor_ADDR32, 4, tile_descriptor);
            tile_descriptor.f = tile_descriptor_vec[1];
            write_unpack_tile_descriptor(cfg, THCON_SEC1_REG0_TileDescriptor_ADDR32, 4, tile_descriptor);
            break;
        case UNPACK_CONFIG: uint num_of_words_unpack_config;
#ifdef ARCH_GRAYSKULL
            num_of_words_unpack_config = 3;
#else
            num_of_words_unpack_config = 4;
#endif
            ckernel::unpacker::unpack_config_u unpack_config;
            generate_unpack_config(unpack_config.f);
            std::array<ckernel::unpacker::unpack_config_t, ckernel::unpacker::NUM_UNPACKERS> unpack_config_vec;
            UNPACK(unpack_config_vec = ckernel::unpacker::read_unpack_config();)
            write_unpack_config(cfg, THCON_SEC0_REG2_Out_data_format_ADDR32, num_of_words_unpack_config, unpack_config);
            write_unpack_config(cfg, THCON_SEC1_REG2_Out_data_format_ADDR32, num_of_words_unpack_config, unpack_config);

            dprint_tensix_unpack_config();

            unpack_config.f = unpack_config_vec[0];
            write_unpack_config(cfg, THCON_SEC0_REG2_Out_data_format_ADDR32, num_of_words_unpack_config, unpack_config);
            unpack_config.f = unpack_config_vec[1];
            write_unpack_config(cfg, THCON_SEC1_REG2_Out_data_format_ADDR32, num_of_words_unpack_config, unpack_config);
            break;
        case PACK_CONFIG: uint num_of_words_pack_config;
#ifdef ARCH_BLACKHOLE
            num_of_words_pack_config = 3;
#else
            num_of_words_pack_config = 4;
#endif
            ckernel::packer::pack_config_u pack_config;
            generate_pack_config(pack_config.f);
            std::array<ckernel::packer::pack_config_t, ckernel::packer::NUM_PACKERS> pack_config_vec;
            MATH(pack_config_vec = ckernel::packer::read_pack_config();)
            write_pack_config(
                cfg, THCON_SEC0_REG1_Row_start_section_size_ADDR32, num_of_words_pack_config, pack_config);
#if defined(ARCH_GRAYSKULL) or defined(ARCH_WORMHOLE)
            write_pack_config(
                cfg, THCON_SEC0_REG8_Row_start_section_size_ADDR32, num_of_words_pack_config, pack_config);
            write_pack_config(
                cfg, THCON_SEC1_REG1_Row_start_section_size_ADDR32, num_of_words_pack_config, pack_config);
            write_pack_config(
                cfg, THCON_SEC1_REG8_Row_start_section_size_ADDR32, num_of_words_pack_config, pack_config);
#endif

            dprint_tensix_pack_config();

            pack_config.f = pack_config_vec[0];
            write_pack_config(
                cfg, THCON_SEC0_REG1_Row_start_section_size_ADDR32, num_of_words_pack_config, pack_config);
#if defined(ARCH_GRAYSKULL) or defined(ARCH_WORMHOLE)
            pack_config.f = pack_config_vec[1];
            write_pack_config(
                cfg, THCON_SEC0_REG8_Row_start_section_size_ADDR32, num_of_words_pack_config, pack_config);
            pack_config.f = pack_config_vec[2];
            write_pack_config(
                cfg, THCON_SEC1_REG1_Row_start_section_size_ADDR32, num_of_words_pack_config, pack_config);
            pack_config.f = pack_config_vec[3];
            write_pack_config(
                cfg, THCON_SEC1_REG8_Row_start_section_size_ADDR32, num_of_words_pack_config, pack_config);
#endif
            break;
#if defined(ARCH_WORMHOLE) or defined(ARCH_BLACKHOLE)
        case RELU_CONFIG:
            ckernel::packer::relu_config_u relu_config;
            generate_relu_config(relu_config.r);
            ckernel::packer::relu_config_u relu_config_original;
            MATH(relu_config_original.r = ckernel::packer::read_relu_config();)
            write_relu_config(cfg, ALU_ACC_CTRL_Zero_Flag_disabled_src_ADDR32, 1, relu_config);
            dprint_tensix_pack_relu_config();

            write_relu_config(cfg, ALU_ACC_CTRL_Zero_Flag_disabled_src_ADDR32, 1, relu_config_original);
            break;
#endif
#if defined(ARCH_WORMHOLE) or defined(ARCH_BLACKHOLE)
        case DEST_RD_CTRL:
            ckernel::packer::dest_rd_ctrl_u dest;
            generate_dest_rd_ctrl(dest.f);
            ckernel::packer::dest_rd_ctrl_u dest_original;
            PACK(dest_original.f = ckernel::packer::read_dest_rd_ctrl();)
            write_dest_rd_ctrl(cfg, PCK_DEST_RD_CTRL_Read_32b_data_ADDR32, dest);

            dprint_tensix_dest_rd_ctrl();

            write_dest_rd_ctrl(cfg, PCK_DEST_RD_CTRL_Read_32b_data_ADDR32, dest_original);
            break;
#endif
        case PACK_EDGE_OFFSET:
            ckernel::packer::pck_edge_offset_u edge;
            generate_pack_edge_offset(edge.f);
            std::array<ckernel::packer::pck_edge_offset_t, ckernel::packer::NUM_PACKERS> edge_vec;
            PACK(edge_vec = ckernel::packer::read_pack_edge_offset();)
            write_pack_edge_offset(cfg, PCK_EDGE_OFFSET_SEC0_mask_ADDR32, edge);
#if defined(ARCH_GRAYSKULL) or defined(ARCH_WORMHOLE)
            write_pack_edge_offset(cfg, PCK_EDGE_OFFSET_SEC1_mask_ADDR32, edge);
            write_pack_edge_offset(cfg, PCK_EDGE_OFFSET_SEC2_mask_ADDR32, edge);
            write_pack_edge_offset(cfg, PCK_EDGE_OFFSET_SEC3_mask_ADDR32, edge);
#endif

            dprint_tensix_pack_edge_offset();

            edge.f = edge_vec[0];
            write_pack_edge_offset(cfg, PCK_EDGE_OFFSET_SEC0_mask_ADDR32, edge);
#if defined(ARCH_GRAYSKULL) or defined(ARCH_WORMHOLE)
            edge.f = edge_vec[1];
            write_pack_edge_offset(cfg, PCK_EDGE_OFFSET_SEC1_mask_ADDR32, edge);
            edge.f = edge_vec[2];
            write_pack_edge_offset(cfg, PCK_EDGE_OFFSET_SEC2_mask_ADDR32, edge);
            edge.f = edge_vec[3];
            write_pack_edge_offset(cfg, PCK_EDGE_OFFSET_SEC3_mask_ADDR32, edge);
#endif
            break;
        case PACK_COUNTERS:
            ckernel::packer::pack_counters_u counter;
            generate_pack_counters(counter.f);
            std::array<ckernel::packer::pack_counters_t, ckernel::packer::NUM_PACKERS> counter_vec;
            PACK(counter_vec = ckernel::packer::read_pack_counters();)
            write_pack_counters(cfg, PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32, counter);
#if defined(ARCH_GRAYSKULL) or defined(ARCH_WORMHOLE)
            write_pack_counters(cfg, PACK_COUNTERS_SEC1_pack_per_xy_plane_ADDR32, counter);
            write_pack_counters(cfg, PACK_COUNTERS_SEC2_pack_per_xy_plane_ADDR32, counter);
            write_pack_counters(cfg, PACK_COUNTERS_SEC3_pack_per_xy_plane_ADDR32, counter);
#endif

            dprint_tensix_pack_counters();

            counter.f = counter_vec[0];
            write_pack_counters(cfg, PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32, counter);
#if defined(ARCH_GRAYSKULL) or defined(ARCH_WORMHOLE)
            counter.f = counter_vec[1];
            write_pack_counters(cfg, PACK_COUNTERS_SEC1_pack_per_xy_plane_ADDR32, counter);
            counter.f = counter_vec[2];
            write_pack_counters(cfg, PACK_COUNTERS_SEC2_pack_per_xy_plane_ADDR32, counter);
            counter.f = counter_vec[3];
            write_pack_counters(cfg, PACK_COUNTERS_SEC3_pack_per_xy_plane_ADDR32, counter);
#endif
            break;
    }
}
}  // namespace NAMESPACE
