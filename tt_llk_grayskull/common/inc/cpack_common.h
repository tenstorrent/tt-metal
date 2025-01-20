// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "fw_debug.h"
#include "llk_defs.h"

#define TT_OP_SETDMAREG_SHFT(Payload_SigSelSize, Payload_SigSelShft, SetSignalsMode, RegIndex16b) \
  TT_OP(0x45, (((Payload_SigSelSize) << 22) + ((Payload_SigSelShft)) + ((SetSignalsMode) << 7) + ((RegIndex16b) << 0)))
#define TT_SETDMAREG_SHFT(Payload_SigSelSize, Payload_SigSelShft, SetSignalsMode, RegIndex16b) \
  ckernel::instrn_buffer[0] = TT_OP_SETDMAREG_SHFT(Payload_SigSelSize, Payload_SigSelShft, SetSignalsMode, RegIndex16b)

namespace ckernel::packer
{
   constexpr uint32_t PACK_CNT = 4;

   //Pack src format, save src format to make reconfig writes only
   uint32_t tile_desc_pack_src_format;

   constexpr uint PACK_SEL(const uint pack_count)
   {
     return (pack_count == 1) ? 0x1 :
            (pack_count == 2) ? 0x3 :
            (pack_count == 4) ? 0xF : 0x0;
   }

   // Pack config
   typedef struct {
     uint32_t row_ptr_section_size : 16;
     uint32_t exp_section_size : 16;
     uint32_t l1_dest_addr: 32;
     uint32_t uncompress  : 1;
     uint32_t add_l1_dest_addr_offset  : 1;
     uint32_t reserved_0  : 2;
     uint32_t out_data_format  : 4;
     uint32_t in_data_format  : 4;
     uint32_t reserved_1  : 4;
     uint32_t src_if_sel  : 1;
     uint32_t pack_per_xy_plane  : 7;
     uint32_t l1_src_addr : 8;
     uint32_t downsample_mask : 16;
     uint32_t downsample_shift_count  : 3;
     uint32_t read_mode : 1;
     uint32_t exp_threshold_en  : 1;
     uint32_t reserved_2 : 3;
     uint32_t exp_threshold : 8;
   } pack_config_t;

   typedef union {
     uint32_t val[4];
     pack_config_t f;
   } pack_config_u;

   // Set unpacker offsets to 0, except for unpacker 0, channel 1, X, which is the tile X dimension
   inline void packer_addr_counter_init()
   {
       TTI_SETADCXY(0b100, 0, 0, 0, 0, 0b1011);
       TTI_SETADCZW(0b100, 0, 0, 0, 0, 0b1111);
   }

   inline void set_packer_config(const uint pack_dst_format){
      // Get pointer to registers for current state ID
      volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

      // Set packer config
      pack_config_u config;
      for (uint i=0; i<4; i++) {
         config.val[i] = 0;
      }
      config.f.exp_section_size = ((uint)(pack_dst_format&0x2) == 0) ? 0 : 4 ;
      config.f.uncompress   = 1;
      config.f.out_data_format   = (uint)pack_dst_format;
      config.f.in_data_format    = (uint)tile_desc_pack_src_format;
      config.f.pack_per_xy_plane = 1;

      TT_SETDMAREG(0, (config.val[2] & 0xffff), 0, LO_16(p_gpr_pack::TMP0));
      TT_SETDMAREG(0, ((config.val[2]>>16) & 0xffff), 0, HI_16(p_gpr_pack::TMP0));
      TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG1_Row_start_section_size_ADDR32+2);
      TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG8_Row_start_section_size_ADDR32+2);
      TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, THCON_SEC1_REG1_Row_start_section_size_ADDR32+2);
      TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, THCON_SEC1_REG8_Row_start_section_size_ADDR32+2);

      // Fix for #1392
      // Break dependancy between data format update and exception section size update below by inserting nop cycles
      TTI_DMANOP; TTI_DMANOP;

      if ((uint)(pack_dst_format&0x2) != 0) {
         // Override exp section size for packers 1,2,3
         // Tile header + exp size + datum size
         if ((uint)(pack_dst_format&0x1F) == (uint)DataFormat::Bfp8 || (uint)(pack_dst_format&0x1F) == (uint)DataFormat::Bfp8_b) {
            TTI_WRCFG(p_gpr_pack::BFP8_EXP_SEC_SIZE+0, p_cfg::WRCFG_32b, THCON_SEC0_REG1_Row_start_section_size_ADDR32);
            TTI_WRCFG(p_gpr_pack::BFP8_EXP_SEC_SIZE+1, p_cfg::WRCFG_32b, THCON_SEC0_REG8_Row_start_section_size_ADDR32);
            TTI_WRCFG(p_gpr_pack::BFP8_EXP_SEC_SIZE+2, p_cfg::WRCFG_32b, THCON_SEC1_REG1_Row_start_section_size_ADDR32);
            TTI_WRCFG(p_gpr_pack::BFP8_EXP_SEC_SIZE+3, p_cfg::WRCFG_32b, THCON_SEC1_REG8_Row_start_section_size_ADDR32);
         } else if ((uint)(pack_dst_format&0x1F) == (uint)DataFormat::Bfp4 || (uint)(pack_dst_format&0x1F) == (uint)DataFormat::Bfp4_b) {
            TTI_WRCFG(p_gpr_pack::BFP4_EXP_SEC_SIZE+0, p_cfg::WRCFG_32b, THCON_SEC0_REG1_Row_start_section_size_ADDR32);
            TTI_WRCFG(p_gpr_pack::BFP4_EXP_SEC_SIZE+1, p_cfg::WRCFG_32b, THCON_SEC0_REG8_Row_start_section_size_ADDR32);
            TTI_WRCFG(p_gpr_pack::BFP4_EXP_SEC_SIZE+2, p_cfg::WRCFG_32b, THCON_SEC1_REG1_Row_start_section_size_ADDR32);
            TTI_WRCFG(p_gpr_pack::BFP4_EXP_SEC_SIZE+3, p_cfg::WRCFG_32b, THCON_SEC1_REG8_Row_start_section_size_ADDR32);
         } else if ((uint)(pack_dst_format&0x1F) == (uint)DataFormat::Bfp2 || (uint)(pack_dst_format&0x1F) == (uint)DataFormat::Bfp2_b) {
            TTI_WRCFG(p_gpr_pack::BFP2_EXP_SEC_SIZE+0, p_cfg::WRCFG_32b, THCON_SEC0_REG1_Row_start_section_size_ADDR32);
            TTI_WRCFG(p_gpr_pack::BFP2_EXP_SEC_SIZE+1, p_cfg::WRCFG_32b, THCON_SEC0_REG8_Row_start_section_size_ADDR32);
            TTI_WRCFG(p_gpr_pack::BFP2_EXP_SEC_SIZE+2, p_cfg::WRCFG_32b, THCON_SEC1_REG1_Row_start_section_size_ADDR32);
            TTI_WRCFG(p_gpr_pack::BFP2_EXP_SEC_SIZE+3, p_cfg::WRCFG_32b, THCON_SEC1_REG8_Row_start_section_size_ADDR32);
         } else {
            FWASSERT("Other data formats not supported", false);
         }
      }
      TTI_DMANOP; TTI_DMANOP;
   }

   //reconfig the packer dst format
   inline void reconfig_packer_data_format(const uint pack_dst_format, const uint tile_size)
   {
      set_packer_config(pack_dst_format);

      uint tile_header_0 = tile_size;
      TT_SETDMAREG(0, (tile_header_0 & 0xffff), 0, LO_16(p_gpr_pack::TILE_HEADER));
      TT_SETDMAREG(0, ((tile_header_0 >> 16) & 0xffff), 0, HI_16(p_gpr_pack::TILE_HEADER));

   }

   inline void wait_for_unpack_config_done()
   {
       while (semaphore_read(semaphore::UNPACK_PACK_CONFIG_SYNC) == 0) {}
   }

   template <bool untilize = false>
   inline void configure_pack(const uint pack_src_format, const uint pack_dst_format, const uint tile_size, uint relu_config = 0, bool skip_alu_format_set=false)
   {
      // Get pointer to registers for current state ID
      volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

      if (pack_src_format != pack_dst_format) {
         TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK);
         tensix_sync();
      }

      const uint pack_per_xy_plane = 16;

      uint x_stride = (uint)(pack_src_format&0x3) == (uint)DataFormat::Float32 ? 4 :
                      (uint)(pack_src_format&0x3) == (uint)DataFormat::Float16 ? 2 : 1;
      uint y_stride = 16*x_stride;
      uint z_stride = PACK_CNT*16*y_stride;

      // Strides (not needed)
      cfg[PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32] = (y_stride<<PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT) |
                                                    (x_stride<<PCK0_ADDR_CTRL_XY_REG_0_Xstride_SHAMT);  // X and Y stride for src address (ch0)
      cfg[PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32] = (z_stride<<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT);  // Z stride for src address (ch0)

      tile_desc_pack_src_format = pack_src_format;

      // Set packer config
      pack_config_u config;
      for (uint i=0; i<4; i++) {
         config.val[i] = 0;
      }
      config.f.exp_section_size = ((uint)(pack_dst_format&0x2) == 0) ? 0 : 4 ;
      config.f.uncompress   = 1;
      config.f.out_data_format   = (uint)pack_dst_format;
      config.f.in_data_format    = (uint)tile_desc_pack_src_format;
      config.f.pack_per_xy_plane = 1;

      cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32+0]=config.val[0];
      cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32+0]=config.val[0];
      cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32+0]=config.val[0];
      cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32+0]=config.val[0];
      cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32+2]=config.val[2];
      cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32+2]=config.val[2];
      cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32+2]=config.val[2];
      cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32+2]=config.val[2];

      if ((uint)(pack_dst_format&0x2) != 0) {
         // Override exp section size for packers 1,2,3
         // Tile header + exp size + datum size
         if ((uint)(pack_dst_format&0x1F) == (uint)DataFormat::Bfp8 || (uint)(pack_dst_format&0x1F) == (uint)DataFormat::Bfp8_b) {
            config.f.exp_section_size = 1 + 2 + 16;
            cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32+0]=config.val[0];
            config.f.exp_section_size = 1 + 1 + 32;
            cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32+0]=config.val[0];
            config.f.exp_section_size = 1 + 0 + 48;
            cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32+0]=config.val[0];
         } else if ((uint)(pack_dst_format&0x1F) == (uint)DataFormat::Bfp4 || (uint)(pack_dst_format&0x1F) == (uint)DataFormat::Bfp4_b) {
            config.f.exp_section_size = 1 + 2 + 8;
            cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32+0]=config.val[0];
            config.f.exp_section_size = 1 + 1 + 16;
            cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32+0]=config.val[0];
            config.f.exp_section_size = 1 + 0 + 24;
            cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32+0]=config.val[0];
         } else if ((uint)(pack_dst_format&0x1F) == (uint)DataFormat::Bfp2 || (uint)(pack_dst_format&0x1F) == (uint)DataFormat::Bfp2_b) {
            config.f.exp_section_size = 1 + 2 + 4;
            cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32+0]=config.val[0];
            config.f.exp_section_size = 1 + 1 + 8;
            cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32+0]=config.val[0];
            config.f.exp_section_size = 1 + 0 + 12;
            cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32+0]=config.val[0];
         } else {
            FWASSERT("Other data formats not supported", false);
         }
      }

      // Preload exp section size for fast format reconfig
      regfile[p_gpr_pack::BFP8_EXP_SEC_SIZE]   = 4 << THCON_SEC0_REG1_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::BFP8_EXP_SEC_SIZE+1] = 19 << THCON_SEC1_REG1_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::BFP8_EXP_SEC_SIZE+2] = 34 << THCON_SEC0_REG8_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::BFP8_EXP_SEC_SIZE+3] = 49 << THCON_SEC1_REG8_Exp_section_size_SHAMT;

      regfile[p_gpr_pack::BFP4_EXP_SEC_SIZE]   = 4 << THCON_SEC0_REG1_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::BFP4_EXP_SEC_SIZE+1] = 11 << THCON_SEC1_REG1_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::BFP4_EXP_SEC_SIZE+2] = 18 << THCON_SEC0_REG8_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::BFP4_EXP_SEC_SIZE+3] = 25 << THCON_SEC1_REG8_Exp_section_size_SHAMT;

      regfile[p_gpr_pack::BFP2_EXP_SEC_SIZE]   = 4 << THCON_SEC0_REG1_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::BFP2_EXP_SEC_SIZE+1] = 7 << THCON_SEC1_REG1_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::BFP2_EXP_SEC_SIZE+2] = 10 << THCON_SEC0_REG8_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::BFP2_EXP_SEC_SIZE+3] = 13 << THCON_SEC1_REG8_Exp_section_size_SHAMT;

      // PACK_COUNTERS_SEC0_pack_per_xy_plane = cfg_reg_array[3][0 +: 8];
      // PACK_COUNTERS_SEC0_pack_reads_per_xy_plane = cfg_reg_array[3][8 +: 8];
      // PACK_COUNTERS_SEC0_pack_xys_per_tile = cfg_reg_array[3][16 +: 7];
      // PACK_COUNTERS_SEC0_pack_yz_transposed = cfg_reg_array[3][23 +: 1];
      for (uint i=0; i<4; i++) cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32+i]=pack_per_xy_plane | (pack_per_xy_plane<<8) | ((untilize?1:0)<<16); // Auto last generation is disabled

      cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32]=0xffff;
      for (uint i=0; i<4; i++) {
         cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32+i] = 0x0;
      }

      regfile[p_gpr_pack::TILE_HEADER]   = tile_size;
      regfile[p_gpr_pack::TILE_HEADER+1] = 0;
      regfile[p_gpr_pack::TILE_HEADER+2] = 0;
      regfile[p_gpr_pack::TILE_HEADER+3] = 0;
      sync_regfile_write(p_gpr_pack::TILE_HEADER+3);

      // Config RELU
      uint apply_relu = reg_read((uint)&cfg[STACC_RELU_ApplyRelu_ADDR32]);
      uint relu_threshold  = reg_read((uint)&cfg[STACC_RELU_ReluThreshold_ADDR32]);
      apply_relu      &= (~STACC_RELU_ApplyRelu_MASK);
      relu_threshold  &= (~STACC_RELU_ReluThreshold_MASK);
      struct {
         uint apply:16;
         uint threshold:16;
      } tmp_relu_cfg = {.apply = relu_config&0xf, .threshold = (relu_config>>16)&0xffff};
      apply_relu |= tmp_relu_cfg.apply<<STACC_RELU_ApplyRelu_SHAMT;
      relu_threshold |= tmp_relu_cfg.threshold<<STACC_RELU_ReluThreshold_SHAMT;
      cfg[STACC_RELU_ApplyRelu_ADDR32] = apply_relu;
      cfg[STACC_RELU_ReluThreshold_ADDR32] = relu_threshold;

      // Assume face height 16
      TTI_SETADCXX(p_setadc::PAC, (256/pack_per_xy_plane)-1, 0x0);

      // Store value for num_msg_received register when tile count is 1
      //regfile[p_gpr_pack::PACK_STREAM_SYNC] = 0;

      // MT: Ensure thread safety between unpacker and packer threads by using semaphore
      if (!skip_alu_format_set) {
         wait_for_unpack_config_done();
         uint alu_dst_format = pack_src_format;
         cfg_rmw(ALU_FORMAT_SPEC_REG2_Dstacc_RMW, alu_dst_format);
         semaphore_get(semaphore::UNPACK_PACK_CONFIG_SYNC);
      }
   }

   inline uint8_t get_packer_dest_offset_index()
   {
      return (dest_offset_id ? p_gpr_pack::DEST_OFFSET_HI : p_gpr_pack::DEST_OFFSET_LO);
   }

   inline uint32_t get_packer_dest_offset()
   {
      return (dest_offset_id ? 0x200 : 0x0);
   }

   inline void flip_packer_dest_offset_id()
   {
      dest_offset_id = 1 - dest_offset_id;
   }

   // Flip packer dest register offset to 0 or 0x200
   // flip-flopping between two halfs
   template <DstSync Dst>
   inline void select_packer_dest_registers()
   {
      if constexpr (Dst == DstSync::SyncFull)
      {
         TTI_WRCFG(p_gpr_pack::DEST_OFFSET_LO,     p_cfg::WRCFG_128b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
      } else {
         TT_WRCFG(get_packer_dest_offset_index(),     p_cfg::WRCFG_128b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
      }
   }

   // Program packer destination addresses from GPRs
   // Note: DMAREG_SHFT takes a pre-shifted SigSel to generate cleaner asm
   template <PackSelMask PackSel=PACK_ALL>
   inline void program_packer_destination(uint32_t addr, uint8_t pack_dst_format)
   {
      const uint8_t fmt = (uint8_t)(pack_dst_format & 0x3);
      if constexpr (PackSel == PACK_ALL) {
         const uint32_t offset1 = fmt == (uint8_t)DataFormat::Float32 ? (0x40 << 8) : fmt == (uint8_t)DataFormat::Float16 ? (0x20 << 8) : (0x1 << 8);
         const uint32_t offset2 = fmt == (uint8_t)DataFormat::Float32 ? (0x80 << 8) : fmt == (uint8_t)DataFormat::Float16 ? (0x40 << 8) : (0x2 << 8);
         const uint32_t offset3 = fmt == (uint8_t)DataFormat::Float32 ? (0xC0 << 8) : fmt == (uint8_t)DataFormat::Float16 ? (0x60 << 8) : (0x3 << 8);
         addr <<= 8;

         TT_SETDMAREG_SHFT(0, addr, 0, LO_16(p_gpr_pack::OUTPUT_ADDR+0));
         TT_SETDMAREG_SHFT(0, addr+offset1, 0, LO_16(p_gpr_pack::OUTPUT_ADDR+1));
         TT_SETDMAREG_SHFT(0, addr+offset2, 0, LO_16(p_gpr_pack::OUTPUT_ADDR+2));
         TT_SETDMAREG_SHFT(0, addr+offset3, 0, LO_16(p_gpr_pack::OUTPUT_ADDR+3));
	      TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK0 | p_stall::PACK1);
         TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR,     p_cfg::WRCFG_32b, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
         TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+1,   p_cfg::WRCFG_32b, THCON_SEC0_REG8_L1_Dest_addr_ADDR32);
         TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK2 | p_stall::PACK3);
         TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+2,   p_cfg::WRCFG_32b, THCON_SEC1_REG1_L1_Dest_addr_ADDR32);
         TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+3,   p_cfg::WRCFG_32b, THCON_SEC1_REG8_L1_Dest_addr_ADDR32);
     } else if constexpr (PackSel == PACK_01) {
         const uint32_t offset1 = fmt == (uint8_t)DataFormat::Float32 ? (0x40 << 8) : fmt == (uint8_t)DataFormat::Float16 ? (0x20 << 8) : (0x1 << 8);
         addr <<= 8;

         TT_SETDMAREG_SHFT(0, addr, 0, LO_16(p_gpr_pack::OUTPUT_ADDR+0));
         TT_SETDMAREG_SHFT(0, addr+offset1, 0, LO_16(p_gpr_pack::OUTPUT_ADDR+1));
         TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK0|p_stall::PACK1);
         TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG1_L1_Dest_addr_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR);
         TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG8_L1_Dest_addr_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR+1);
     } else if constexpr (PackSel == PACK_0) {
         TT_SETDMAREG(0, addr, 0, LO_16(p_gpr_pack::OUTPUT_ADDR+0));
         TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK0);
         TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG1_L1_Dest_addr_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR);
     } else {
        FWASSERT("Unsupported pack select mask!", false);
     }
    }

   template <uint32_t block_ct_dim, uint32_t full_ct_dim, bool diagonal = false, uint32_t row_num_datums = TILE_C_DIM>
   inline void program_packer_untilized_destination(const uint32_t addr, const uint32_t pack_dst_format)
   {
      if constexpr (diagonal) {
         const uint32_t block_size = SCALE_DATUM_SIZE(pack_dst_format, FACE_C_DIM);
         constexpr uint32_t offset0 = 0;
         const uint32_t offset1 = (1*block_size)/16;
         // const uint32_t offset2 = (2*block_size)/16;
         // const uint32_t offset3 = (3*block_size)/16;

         TT_SETDMAREG(0, addr+offset0, 0, LO_16(p_gpr_pack::OUTPUT_ADDR+0));
         TT_SETDMAREG(0, addr+offset1, 0, LO_16(p_gpr_pack::OUTPUT_ADDR+1));
         // TT_SETDMAREG(0, addr+offset2, 0, LO_16(p_gpr_pack::OUTPUT_ADDR+2));
         // TT_SETDMAREG(0, addr+offset3, 0, LO_16(p_gpr_pack::OUTPUT_ADDR+3));
         TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK0 | p_stall::PACK1);
         TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+0,   p_cfg::WRCFG_32b, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
         TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+1,   p_cfg::WRCFG_32b, THCON_SEC0_REG8_L1_Dest_addr_ADDR32);
         // TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK2 | p_stall::PACK3);
         // TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+2,   p_cfg::WRCFG_32b, THCON_SEC1_REG1_L1_Dest_addr_ADDR32);
         // TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+3,   p_cfg::WRCFG_32b, THCON_SEC1_REG8_L1_Dest_addr_ADDR32);
      } else {
         const uint32_t block_size = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * TILE_C_DIM * (TILE_R_DIM/4));
         constexpr uint32_t offset0 = 0;
         const uint32_t offset1 = (1*row_num_datums*block_size) / 16 / TILE_C_DIM;
         const uint32_t offset2 = (2*row_num_datums*block_size) / 16 / TILE_C_DIM;
         const uint32_t offset3 = (3*row_num_datums*block_size) / 16 / TILE_C_DIM;

         TT_SETDMAREG(0, addr+offset0, 0, LO_16(p_gpr_pack::OUTPUT_ADDR+0));
         TT_SETDMAREG(0, addr+offset1, 0, LO_16(p_gpr_pack::OUTPUT_ADDR+1));
         TT_SETDMAREG(0, addr+offset2, 0, LO_16(p_gpr_pack::OUTPUT_ADDR+2));
         TT_SETDMAREG(0, addr+offset3, 0, LO_16(p_gpr_pack::OUTPUT_ADDR+3));
         TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK0 | p_stall::PACK1);
         TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+0,   p_cfg::WRCFG_32b, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
         TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+1,   p_cfg::WRCFG_32b, THCON_SEC0_REG8_L1_Dest_addr_ADDR32);
         TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK2 | p_stall::PACK3);
         TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+2,   p_cfg::WRCFG_32b, THCON_SEC1_REG1_L1_Dest_addr_ADDR32);
         TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+3,   p_cfg::WRCFG_32b, THCON_SEC1_REG8_L1_Dest_addr_ADDR32);
      }
   }

   // Write tile header to l1
   inline void write_tile_header()
   {
      TTI_STOREIND (1, 0, p_ind::LD_16B, LO_16(0), p_ind::INC_NONE, p_gpr_pack::TILE_HEADER, p_gpr_pack::OUTPUT_ADDR);
   }
}
