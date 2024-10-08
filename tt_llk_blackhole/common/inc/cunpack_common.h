// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "ckernel.h"
#include "ckernel_globals.h"

#ifdef PERF_DUMP
#include "perf_res_decouple.h"
#endif

namespace ckernel::unpacker
{
   constexpr uint32_t TILE_DESC_SIZE = 2; //Unpacker descriptor size in dwords
   constexpr uint32_t CONFIG_SIZE = 2; //Unpacker configuration size in dwords

   // Unpack tile descriptor
   typedef struct {
     // word 0
     uint32_t in_data_format : 4;
     uint32_t uncompressed: 1;
     uint32_t reserved_0  : 3;
     uint32_t blobs_per_xy_plane  : 4;
     uint32_t reserved_1  : 4;
     uint32_t x_dim       : 16;
     // word 1
     uint32_t y_dim       : 16;
     uint32_t z_dim       : 16;
     // word 2
     uint32_t w_dim       : 16;
     uint32_t blobs_y_start_lo : 16;
     // word 3
     uint32_t blobs_y_start_hi : 16;
     uint32_t digest_type : 8;  // Not used
     uint32_t digest_size : 8;  // Not used
   } unpack_tile_descriptor_t; // Unpack configuration

   static_assert(sizeof(unpack_tile_descriptor_t) == (sizeof(uint32_t) * 4));

   typedef union {
     uint32_t val[4];
     unpack_tile_descriptor_t f;
   } unpack_tile_descriptor_u;

   // Unpack config
   typedef struct {
     // word 0
     uint32_t out_data_format         : 4;
     uint32_t throttle_mode           : 2;
     uint32_t context_count           : 2;
     uint32_t haloize_mode            : 1; // this controls xy transpose on unpacker
     uint32_t tileize_mode            : 1;
     uint32_t unpack_src_reg_set_update : 1;
     uint32_t unpack_if_sel           : 1;
     uint32_t upsample_rate           : 2;
     uint32_t reserved_1              : 1;
     uint32_t upsamle_and_interlave   : 1;
     uint32_t shift_amount            : 16;
     // word 1
     uint32_t uncompress_cntx0_3      : 4;
     uint32_t unpack_if_sel_cntx0_3   : 4;
     uint32_t force_shared_exp        : 1;
     uint32_t reserved_2              : 7;
     uint32_t uncompress_cntx4_7      : 4;
     uint32_t unpack_if_sel_cntx4_7   : 4;
     uint32_t reserved_3              : 8;
     // word 2
     uint32_t limit_addr              : 17;
     uint32_t reserved_4              : 15;
     // word 3
     uint32_t fifo_size               : 17;
     uint32_t reserved_5              : 15;
   } unpack_config_t;

   static_assert(sizeof(unpack_config_t) == (sizeof(uint32_t) * 4));

   typedef union {
     uint32_t val[4];
     unpack_config_t f;
   } unpack_config_u;

   // ALU config
   typedef struct{
     uint32_t ALU_ROUNDING_MODE_Fpu_srnd_en      : 1;
     uint32_t ALU_ROUNDING_MODE_Gasket_srnd_en   : 1;
     uint32_t ALU_ROUNDING_MODE_Packer_srnd_en   : 1;
     uint32_t ALU_ROUNDING_MODE_Padding          :10;
     uint32_t ALU_ROUNDING_MODE_GS_LF            : 1;
     uint32_t ALU_ROUNDING_MODE_Bfp8_HF          : 1;
     uint32_t ALU_FORMAT_SPEC_REG0_SrcAUnsigned  : 1;
     uint32_t ALU_FORMAT_SPEC_REG0_SrcBUnsigned  : 1;
     uint32_t ALU_FORMAT_SPEC_REG0_SrcA          : 4;
     uint32_t ALU_FORMAT_SPEC_REG1_SrcB          : 4;
     uint32_t ALU_FORMAT_SPEC_REG2_Dstacc        : 4;
     uint32_t ALU_ACC_CTRL_Fp32_enabled          : 1;
     uint32_t ALU_ACC_CTRL_SFPU_Fp32_enabled     : 1;
     uint32_t ALU_ACC_CTRL_INT8_math_enabled     : 1;
   } alu_config_t;

   static_assert(sizeof(alu_config_t) == sizeof(uint32_t));

   typedef union {
     uint32_t val;
     alu_config_t f;
   } alu_config_u;

   // Set unpacker offsets to 0, except for unpacker 0, channel 1, X, which is the tile X dimension
   inline void unpacker_addr_counter_init()
   {
       TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1011);
       TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
   }

   inline void unpacker_iteration_cleanup(uint &context)
   {
       // Indicate that unpacker is done, and we can program the next one
       t6_semaphore_get(semaphore::UNPACK_SYNC);
       context = 1 - context;
       if (context == 1) {
           TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0104);
       } else {
           TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
       }
   }

   inline void unpacker_wrapup()
   {
       // Clear unpacker0 tile offset address
       TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, THCON_SEC0_REG7_Offset_address_ADDR32);
       TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, THCON_SEC0_REG7_Offset_cntx1_address_ADDR32);

       // Clear unpacker1 tile offset address
       TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, THCON_SEC1_REG7_Offset_address_ADDR32);
       TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, THCON_SEC1_REG7_Offset_cntx1_address_ADDR32);

       // Clear context offset and counter
       TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x1010);
   }

   inline uint unpack_16B_address(const uint addr)
   {
       return (addr << FIFO_BASE_ADDRESS_ALIGN_BITS) >> 4;
   }

   inline void flush_xsearch_cache(const uint unpacker)
   {
       TTI_UNPACR(unpacker, 0, 0, 0, 0, 0, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 1, 0);
   }

   // Wait for threshold of busy contexts to fall below total available contexts
   inline void wait_for_next_context(const uint num_contexts)
   {
       while (semaphore_read(semaphore::UNPACK_SYNC) >= num_contexts) {}
   }

   inline void switch_config_context(uint &unp_cfg_context)
   {
      // Switch config context
      unp_cfg_context = 1 - unp_cfg_context;
      if (unp_cfg_context == 0) {
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
      } else {
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0101);
      }

   }

   inline void reset_config_context()
   {
      // Switch config context
      unp_cfg_context = 0;
      TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
   }

   // Sync on unpacker idle via waiting busy contexts counter 0
   inline void wait_for_idle()
   {
       while (semaphore_read(semaphore::UNPACK_SYNC) > 0) {}
   }

   inline void enable_int8_fpu_math() {
      alu_config_u alu_payload = {.val = 0};
      alu_payload.f.ALU_ACC_CTRL_INT8_math_enabled = 1;
      cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, ALU_ACC_CTRL_INT8_math_enabled_MASK>(alu_payload.val);
   }

   template<bool row_pool=false, bool is_fp32_dest_acc_en = false, bool fpu_srnd_en = false, bool pack_srnd_en = false>
   inline void configure_unpack_AB(
     const uint unpA_src_format,
     const uint unpB_src_format,
     const uint unpA_dst_format,
     const uint unpB_dst_format,
     const uint unpA_face_r_dim=FACE_R_DIM,
     const uint unpB_face_r_dim=FACE_R_DIM,
     const bool transpose_xy_srca_en=false,
     const uint unpA_num_faces = 4,
     const uint unpB_num_faces = 4)
   {
      // Check that unpacker is done (all contexts freed up) before starting hw configuration
      wait_for_idle();

      // Reset address counters
      unpacker_addr_counter_init();

      const uint unpA_src_format_masked = (uint)unpA_src_format & 0x0F;
      const uint unpB_src_format_masked = (uint)unpB_src_format & 0x0F;
      const uint unpA_dst_format_masked = (uint)unpA_dst_format & 0x0F;
      const uint unpB_dst_format_masked = (uint)unpB_dst_format & 0x0F;

      // Get pointer to registers for current state ID
      volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

      uint unpA_ch1_x_stride = (uint) (unpA_dst_format_masked&0x3) == (uint) DataFormat::Float32 ? 4 : (uint) (unpA_dst_format_masked&0x3) == (uint) DataFormat::Float16 ? 2 : 1;
      uint unpB_ch1_x_stride = (uint) (unpB_dst_format_masked&0x3) == (uint) DataFormat::Float32 ? 4 : (uint) (unpB_dst_format_masked&0x3) == (uint) DataFormat::Float16 ? 2 : 1;
      uint unpA_ch1_z_stride = FACE_C_DIM*FACE_R_DIM*unpA_ch1_x_stride;
      uint unpB_ch1_z_stride = FACE_C_DIM*FACE_R_DIM*unpB_ch1_x_stride;
      uint exp_width = ((uint)unpA_dst_format_masked>>2)&0x1; //0=5-bit, 1=8-bit

      // Strides for incrementing ch1 address to srcA and srcB
      cfg[UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] = (0                 << UNP0_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT) |
                                                    (unpA_ch1_z_stride << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT);  // Z and W(not used) stride for dest address (ch1)

      cfg[UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] = (0                 << UNP1_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT) |
                                                    (unpB_ch1_z_stride << UNP1_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT);  // Z and W(not used) stride for dest address (ch1)

      // Math ALU_FORMAT_REG
      t6_mutex_acquire(mutex::REG_RMW);
      uint alu_src_format = (0x0 << ALU_FORMAT_SPEC_REG_SrcA_val_SHAMT);

      constexpr uint mask0 = (1 << (ALU_FORMAT_SPEC_REG_Dstacc_override_SHAMT + 1)) - 1;
      cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, ALU_FORMAT_SPEC_REG_SrcA_val_SHAMT, mask0>(alu_src_format);

      alu_config_u alu_payload = {.val = 0};

      uint32_t fp32_dest_acc_en = (is_fp32_dest_acc_en) ? (1) : (0);
      uint32_t int8_math_enabled = ((uint)unpA_dst_format_masked == (uint)DataFormat::Int8) ||
                                   ((uint)unpB_dst_format_masked == (uint)DataFormat::Int8) ||
                                   ((uint)unpA_dst_format_masked == (uint)DataFormat::Int32) ||
                                   ((uint)unpB_dst_format_masked == (uint)DataFormat::Int32);

      constexpr uint alu_format_mask = ALU_FORMAT_SPEC_REG0_SrcAUnsigned_MASK | ALU_FORMAT_SPEC_REG0_SrcBUnsigned_MASK;

      if ((uint)unpA_src_format == (uint)DataFormat::UInt8) {
         alu_payload.f.ALU_FORMAT_SPEC_REG0_SrcAUnsigned = 1;
      }
      if ((uint)unpB_src_format == (uint)DataFormat::UInt8) {
         alu_payload.f.ALU_FORMAT_SPEC_REG0_SrcBUnsigned = 1;
      }

      // FP32 accumulation and SFPU to read dest as FP32
      // NOTE: This assumes these config fields are adjacent and in same register!!
      static_assert(ALU_ACC_CTRL_Fp32_enabled_ADDR32 == ALU_FORMAT_SPEC_REG0_SrcA_ADDR32);
      static_assert(ALU_ACC_CTRL_Fp32_enabled_ADDR32 == ALU_ACC_CTRL_SFPU_Fp32_enabled_ADDR32);
      constexpr uint alu_dest_format_mask = ALU_ACC_CTRL_SFPU_Fp32_enabled_MASK | ALU_ACC_CTRL_Fp32_enabled_MASK;
      alu_payload.f.ALU_ACC_CTRL_Fp32_enabled = fp32_dest_acc_en;
      alu_payload.f.ALU_ACC_CTRL_SFPU_Fp32_enabled = fp32_dest_acc_en;
      constexpr uint alu_stoch_rnd_mask = ALU_ROUNDING_MODE_Fpu_srnd_en_MASK | ALU_ROUNDING_MODE_Gasket_srnd_en_MASK | ALU_ROUNDING_MODE_Packer_srnd_en_MASK;
      alu_payload.f.ALU_ROUNDING_MODE_Fpu_srnd_en = fpu_srnd_en;
      alu_payload.f.ALU_ROUNDING_MODE_Gasket_srnd_en = pack_srnd_en;
      alu_payload.f.ALU_ROUNDING_MODE_Packer_srnd_en = pack_srnd_en;

      constexpr uint alu_mask = alu_format_mask | alu_dest_format_mask | alu_stoch_rnd_mask;

      cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, alu_mask>(alu_payload.val);

      uint32_t src_zeroflags_disable = ((uint)unpA_dst_format == (uint)DataFormat::UInt16) || ((uint)unpB_dst_format == (uint)DataFormat::UInt16);
      cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(src_zeroflags_disable);

      //Set FP8 E4M3 mode, bit is accessible by unpacker/packer
      if((unpA_src_format&0x1F) == (uint)DataFormat::Fp8_e4m3) {
        cfg_reg_rmw_tensix<THCON_SEC0_REG1_Unp_LF8_4b_exp_RMW>(1);
      }

      if((unpB_src_format&0x1F) == (uint)DataFormat::Fp8_e4m3) {
        cfg_reg_rmw_tensix<THCON_SEC1_REG1_Unp_LF8_4b_exp_RMW>(1);
      }

      t6_mutex_release(mutex::REG_RMW);

      // Set tile descriptor
      unpack_tile_descriptor_u tile_descriptor;
      for (uint i=0; i<TILE_DESC_SIZE; i++) {
         tile_descriptor.val[i] = 0;
      }
      tile_descriptor.f.in_data_format  = (uint) unpA_src_format_masked;
      tile_descriptor.f.uncompressed = 1; // Input tile is uncompressed
      tile_descriptor.f.x_dim        = 0; // Not used for unpA as value is overriden by per context x_dim set below. Used for unpB
      tile_descriptor.f.y_dim        = 1;
      tile_descriptor.f.z_dim        = unpA_num_faces;
      //tile_descriptor.f.blobs_per_xy_plane = 0;
      //tile_descriptor.f.blobs_y_start = 0;
      for (uint i=0; i<TILE_DESC_SIZE; i++) cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32+i]=tile_descriptor.val[i];
      tile_descriptor.f.in_data_format  = row_pool ? (uint) DataFormat::Float32 : unpB_src_format_masked;
      tile_descriptor.f.x_dim        = unpB_face_r_dim*FACE_C_DIM;
      tile_descriptor.f.z_dim        = unpB_num_faces;
      for (uint i=0; i<TILE_DESC_SIZE; i++) cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32+i]=tile_descriptor.val[i];

      // Set unpacker config
      unpack_config_u config;
      for (uint i=0; i<CONFIG_SIZE; i++) {
         config.val[i] = 0;
      }
      config.f.out_data_format = unpA_dst_format_masked;
      config.f.throttle_mode   = 2;
      config.f.context_count   = 0;
      config.f.haloize_mode    = transpose_xy_srca_en ? 1 : 0;
      //config.f.upsample_rate   = 0;
      //config.f.upsamle_and_interlave  = 0;
      //config.f.shift_amount = 0;
      config.f.uncompress_cntx0_3 = 0xf;
      config.f.uncompress_cntx4_7 = 0xf;
      //config.f.limit_addr = 0; // Set dynamically
      //config.f.fifo_size = 0; // Set dynamically
      for (uint i=0; i<CONFIG_SIZE; i++) cfg[THCON_SEC0_REG2_Out_data_format_ADDR32+i]=config.val[i];

      config.f.out_data_format = row_pool ? ((uint) DataFormat::Float16 | (exp_width<<2)) : unpB_dst_format_masked;
      config.f.haloize_mode    = 0;

      for (uint i=0; i<CONFIG_SIZE; i++) cfg[THCON_SEC1_REG2_Out_data_format_ADDR32+i]=config.val[i];

      uint unpA_x_end = (unpA_face_r_dim == 0) ? 1 : (unpA_face_r_dim << 4) - 1;
      TTI_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
      TTI_SETADCXX(p_setadc::UNP_B, (unpB_face_r_dim << 4)-1, 0x0);

      // Program base address for all 2 sections (each section address is loaded to corresponding context)
      // Load dummy data to unused location if face height is 0
      const uint Dest_cntx0_address = unpA_face_r_dim == 0 ? 22*16 : 4 * 16;
      const uint Dest_cntx1_address = unpA_face_r_dim == 0 ? 22*16 : 4 * 16;
      cfg[THCON_SEC0_REG5_Dest_cntx0_address_ADDR32] = Dest_cntx0_address | (Dest_cntx1_address << 16);

      // Program unpacker0 per context x_dim (face size in l1)
      // Overrides value set by tile descriptor when thread override bit is set in unpack instruction
      const uint face_dim = unpA_face_r_dim*FACE_C_DIM;
      cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = face_dim | (face_dim << 16);

      constexpr uint face_dim_16x16 = FACE_R_DIM*FACE_C_DIM;
      regfile[p_gpr_unpack::FACE_DIM_16x16] = (face_dim_16x16/1 ) | ((face_dim_16x16/1 )<<16);
      regfile[p_gpr_unpack::FACE_DIM_8x16]  = (face_dim_16x16/2 ) | ((face_dim_16x16/2 )<<16);
      regfile[p_gpr_unpack::FACE_DIM_4x16]  = (face_dim_16x16/4 ) | ((face_dim_16x16/4 )<<16);
      regfile[p_gpr_unpack::FACE_DIM_2x16]  = (face_dim_16x16/8 ) | ((face_dim_16x16/8 )<<16);
      regfile[p_gpr_unpack::FACE_DIM_1x16]  = (face_dim_16x16/16) | ((face_dim_16x16/16)<<16);
      sync_regfile_write(p_gpr_unpack::FACE_DIM_1x16);

      TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4);

      // Enable address counter for unpacker ch1/dst address
      // final address is calculated as: Dest_cntx0/1_address + address_counter_ch1
      // used for face by face unpacking of entire tile into srcA
      cfg[UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_ADDR32] = 0x1<<UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_SHAMT;

      /*
      // Workaround for HW bug (fp32 dest and movd2a/b is used with srcA/B configured with 5-bit exponent)
      if (is_fp32_dest_acc_en && (exp_width == 0)) {
          reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 1<<11); // Set debug feature disable bit 11
                                                                 // workaround for bug tenstorrent/budabackend#1372
      }
      */
      // Workaround for HW bug (int32 dest and movd2a/b is used with srcA/B configured as int8)
      if (int8_math_enabled || (fp32_dest_acc_en && ((uint)unpA_dst_format == (uint)DataFormat::UInt16))) {
          reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 1<<11); // Set debug feature disable bit 11
                                                                 // workaround for bug tenstorrent/budabackend#1948
      }

      // Clear context ID
      reset_config_context();
   }

   template <std::uint32_t UNP_SEL = p_setadc::UNP_AB>
   inline void config_unpacker_x_end(const uint32_t face_r_dim)
   {
      switch (face_r_dim) {
         case 1:
            TTI_SETADCXX(UNP_SEL, 1*FACE_C_DIM-1, 0x0);
            break;
         case 2:
            TTI_SETADCXX(UNP_SEL, 2*FACE_C_DIM-1, 0x0);
            break;
         case 4:
            TTI_SETADCXX(UNP_SEL, 4*FACE_C_DIM-1, 0x0);
            break;
         case 8:
            TTI_SETADCXX(UNP_SEL, 8*FACE_C_DIM-1, 0x0);
            break;
         default:
            TTI_SETADCXX(UNP_SEL, FACE_R_DIM*FACE_C_DIM-1, 0x0);
            break;
      }
   }

   template <bool INSERT_FENCE=false, std::uint32_t UNP_SEL = p_setadc::UNP_AB>
   inline void config_unpacker_0_face_dim(const uint32_t face_r_dim)
   {
      //tile x dim registers are only for unpacker 0
      static_assert(UNP_SEL != p_setadc::UNP_B);
      TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK0);
      switch (face_r_dim) {
         case 1:
            // TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_1x16);
            TTI_WRCFG(p_gpr_unpack::FACE_DIM_1x16, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
            break;
         case 2:
            // TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_2x16);
            TTI_WRCFG(p_gpr_unpack::FACE_DIM_2x16, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
            break;
         case 4:
            // TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_4x16);
            TTI_WRCFG(p_gpr_unpack::FACE_DIM_4x16, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
            break;
         case 8:
            // TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_8x16);
            TTI_WRCFG(p_gpr_unpack::FACE_DIM_8x16, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
            break;
         default:
            // TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_16x16);
            TTI_WRCFG(p_gpr_unpack::FACE_DIM_16x16, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
            break;
      }

      if constexpr (INSERT_FENCE) {
         TTI_DMANOP; // Insert fence if reg2flop is followed by an unpack
      }
   }

   inline constexpr bool is_32bit_input(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format) {
       const uint input_df = unpack_src_format & 0xF;
       const uint output_df = unpack_dst_format & 0xF;
       return ((input_df == (uint)DataFormat::Int32)  || (input_df == (uint)DataFormat::Float32)) &&
              ((output_df == (uint)DataFormat::Int32) || (output_df == (uint)DataFormat::Float32));
   }

   inline void wait_for_dest_available() {
      t6_semaphore_wait_on_max<p_stall::UNPACK>(semaphore::UNPACK_TO_DEST);
   }

   inline void unpack_to_dest_tile_done(uint &context_id) {
      t6_semaphore_post<p_stall::UNPACK0>(semaphore::UNPACK_TO_DEST);
      TTI_WRCFG(p_gpr_unpack::UNPACK_STRIDE, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32); // Restore unpack stride
      // Restore config context
      if (context_id == 0) {
         cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(0);
         cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(4*16);
      } else {
         cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(0);
         cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(4*16);
      }
      TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4); // re-enable address bit swizzle
   }


   inline void set_dst_write_addr(const uint32_t &context_id, const uint32_t &unpack_dst_format)
   {
      uint32_t dst_byte_addr = 16*(4 + mailbox_read(ThreadId::MathThreadId)); // Apply fixed offset of 4*16 to dest address
      TTI_SETC16(SRCA_SET_Base_ADDR32, 0x0); // Disable address bit swizzle
      TTI_RDCFG(p_gpr_unpack::UNPACK_STRIDE, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32); // Save current stride
      uint unpA_ch1_x_stride = (uint) (unpack_dst_format&0x3) == (uint) DataFormat::Float32 ? 4 : (uint) (unpack_dst_format&0x3) == (uint) DataFormat::Float16 ? 2 : 1;
      uint unpA_ch1_z_stride = FACE_C_DIM*FACE_R_DIM*unpA_ch1_x_stride;
      TT_SETDMAREG(0, LOWER_HALFWORD(unpA_ch1_z_stride << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT), 0, LO_16(p_gpr_unpack::TMP_LO));
      TTI_WRCFG(p_gpr_unpack::TMP_LO, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32); // Set unpack stride
      if (context_id == 0) {
         cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(1);
         cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(dst_byte_addr);
      } else {
         cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(1);
         cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(dst_byte_addr);
      }

   }

}
