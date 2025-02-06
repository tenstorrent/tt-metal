// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint_tensix_pack.h"
#include "debug/dprint_tensix_unpack.h"

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
#ifndef ARCH_GRAYSKULL
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
#else
   tile_descriptor.blobs_y_start_lo = 32;
   tile_descriptor.blobs_y_start_hi = 0;
#endif
   tile_descriptor.digest_type = 0;
   tile_descriptor.digest_size = 0;
}

//0,1,2,0,1,1,0,3,1,0,16,5,6,0,4,2,3,15,28,7,29,8
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
#else
   config.reserved_3 = 0;
   config.reserved_4 = 0;
   config.reserved_5 = 0;
   config.unpack_if_sel_cntx0_3 = 6;
   config.unpack_if_sel_cntx4_7 = 3;
   config.unpack_src_reg_set_update = 1;
   config.unpack_if_sel = 0;
#endif
}

void write_alu_config(volatile uint tt_reg_ptr* cfg, uint32_t address, const ckernel::unpacker::alu_config_u &config) {
   cfg[address] = config.val;
}

void write_unpack_tile_descriptor(volatile uint tt_reg_ptr* cfg, uint32_t address, uint num_of_words, const ckernel::unpacker::unpack_tile_descriptor_u &tile_descriptor) {
   for (uint32_t i = 0; i < num_of_words; i++)
      cfg[address + i] = tile_descriptor.val[i];
}

void write_unpack_config(volatile uint tt_reg_ptr* cfg, uint32_t address, uint num_of_words, const ckernel::unpacker::unpack_config_u &config) {
   for (uint32_t i = 0; i < num_of_words; i++)
      cfg[address + i] = config.val[i];
}

void MAIN {
   uint32_t register_name = get_compile_time_arg_val(0);

   // Get pointer to registers for current state ID 
   volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

   switch (register_name) {
      #ifndef ARCH_GRAYSKULL
      case ALU_CONFIG:
         ckernel::unpacker::alu_config_u alu_config;
         generate_alu_config(alu_config.f);
         write_alu_config(cfg, ALU_ROUNDING_MODE_Fpu_srnd_en_ADDR32, alu_config);
         dprint_tensix_alu_config();
         break;
      #endif
      case UNPACK_TILE_DESCRIPTOR:
         ckernel::unpacker::unpack_tile_descriptor_u tile_descriptor;
         generate_unpack_tile_descriptor(tile_descriptor.f);
         write_unpack_tile_descriptor(cfg, THCON_SEC0_REG0_TileDescriptor_ADDR32, 4, tile_descriptor);
         write_unpack_tile_descriptor(cfg, THCON_SEC1_REG0_TileDescriptor_ADDR32, 4, tile_descriptor);
         dprint_tensix_unpack_tile_descriptor();
         break;
      case UNPACK_CONFIG:
         ckernel::unpacker::unpack_config_u config;
         generate_unpack_config(config.f);
         write_unpack_config(cfg, THCON_SEC0_REG2_Out_data_format_ADDR32, 4, config);
         write_unpack_config(cfg, THCON_SEC1_REG2_Out_data_format_ADDR32, 4, config);
         dprint_tensix_unpack_config();
         break;
   }
}
}  // namespace NAMESPACE
