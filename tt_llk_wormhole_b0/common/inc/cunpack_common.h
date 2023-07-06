#pragma once

#include "ckernel.h"
#include "ckernel_globals.h"
#include "fw_debug.h"

#ifdef PERF_DUMP
#include "perf_res_decouple.h"
#endif

namespace ckernel::unpacker
{
   constexpr uint32_t OPERAND_BASE_ID = 0; 
   constexpr uint32_t INTERMEDIATE_BASE_ID = 24; 
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

   inline void configure_unpack_AB(
     uint unpA_operand, 
     uint unpB_operand, 
     uint srca_face_height=16,
     uint srcb_face_height=16,
     bool row_pool=false,
     bool transpose_xy_srca_en=false,
     bool is_fp32_dest_acc_en=false,
     bool srnd_fpu_en = false)
   {
      // Check that unpacker is done (all contexts freed up) before starting hw configuration
      wait_for_idle();	     

      // Reset address counters	      
      unpacker_addr_counter_init();

      // Get pointer to registers for current state ID
      volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

      uint unpA_ch1_x_stride = (uint) (unpack_dst_format[unpA_operand]&0x3) == (uint) DataFormat::Float32 ? 4 : (uint) (unpack_dst_format[unpA_operand]&0x3) == (uint) DataFormat::Float16 ? 2 : 1;
      uint unpB_ch1_x_stride = (uint) (unpack_dst_format[unpB_operand]&0x3) == (uint) DataFormat::Float32 ? 4 : (uint) (unpack_dst_format[unpB_operand]&0x3) == (uint) DataFormat::Float16 ? 2 : 1;
      uint unpA_ch1_y_stride = 16*srca_face_height*unpA_ch1_x_stride;
      uint unpB_ch1_y_stride = 16*srcb_face_height*unpB_ch1_x_stride;
      uint unpB_ch1_z_stride = 16*srcb_face_height*unpB_ch1_x_stride;
      uint exp_width = ((uint)unpack_dst_format[unpA_operand]>>2)&0x1; //0=5-bit, 1=8-bit
   
      // Strides
      cfg[UNP0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32] = (unpA_ch1_y_stride << UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT) |
                                                    (            0 << UNP0_ADDR_CTRL_XY_REG_0_Xstride_SHAMT);  // X and Y stride for dest address (ch1)
      //cfg[UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] =  // Z and W stride for dest address (ch1)
      cfg[UNP1_ADDR_CTRL_XY_REG_1_Xstride_ADDR32] = (unpB_ch1_y_stride << UNP1_ADDR_CTRL_XY_REG_0_Ystride_SHAMT) |
                                                    (            0 << UNP1_ADDR_CTRL_XY_REG_0_Xstride_SHAMT);  // X and Y stride for dest address (ch1)
      cfg[UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] = (0                 << UNP1_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT) | 
                                                    (unpB_ch1_z_stride << UNP1_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT);  // Z and W stride for dest address (ch1)

      // Math ALU_FORMAT_REG
      t6_mutex_acquire(mutex::REG_RMW);
      uint alu_src_format = (0x0 << ALU_FORMAT_SPEC_REG_SrcA_val_SHAMT);

      constexpr uint mask0 = (1 << (ALU_FORMAT_SPEC_REG_Dstacc_override_SHAMT + 1)) - 1;
      cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, ALU_FORMAT_SPEC_REG_SrcA_val_SHAMT, mask0>(alu_src_format);

      alu_config_u alu_payload = {.val = 0};

      uint32_t fp32_dest_acc_en = (is_fp32_dest_acc_en) ? (1) : (0);

      alu_payload.f.ALU_FORMAT_SPEC_REG0_SrcA = unpack_dst_format[unpA_operand];
      alu_payload.f.ALU_FORMAT_SPEC_REG1_SrcB = row_pool ? ((uint) DataFormat::Float16 | (exp_width<<2)) : unpack_dst_format[unpB_operand];
      // FP32 accumulation and SFPU to read dest as FP32
      // NOTE: This assumes these config fields are adjacent and in same register!!
      static_assert(ALU_ACC_CTRL_Fp32_enabled_ADDR32 == ALU_FORMAT_SPEC_REG0_SrcA_ADDR32);
      static_assert(ALU_ACC_CTRL_Fp32_enabled_ADDR32 == ALU_ACC_CTRL_SFPU_Fp32_enabled_ADDR32);
      alu_payload.f.ALU_ACC_CTRL_Fp32_enabled = fp32_dest_acc_en;
      alu_payload.f.ALU_ACC_CTRL_SFPU_Fp32_enabled = fp32_dest_acc_en;
      alu_payload.f.ALU_ACC_CTRL_INT8_math_enabled = ((uint)unpack_dst_format[unpA_operand] == (uint)DataFormat::Int8) || 
                                                     ((uint)unpack_dst_format[unpB_operand] == (uint)DataFormat::Int8);

      constexpr uint mask1 = ALU_ACC_CTRL_INT8_math_enabled_MASK | ALU_ACC_CTRL_SFPU_Fp32_enabled_MASK | ALU_ACC_CTRL_Fp32_enabled_MASK | ALU_FORMAT_SPEC_REG1_SrcB_MASK | ALU_FORMAT_SPEC_REG0_SrcA_MASK;

      cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, mask1>(alu_payload.val);

      cfg_reg_rmw_tensix<ALU_ROUNDING_MODE_Fpu_srnd_en_RMW>(srnd_fpu_en);

      t6_mutex_release(mutex::REG_RMW);

   
      // Set tile descriptor
      unpack_tile_descriptor_u tile_descriptor;
      for (uint i=0; i<TILE_DESC_SIZE; i++) {
         tile_descriptor.val[i] = 0;
      }
      tile_descriptor.f.in_data_format  = (uint) unpack_src_format[unpA_operand];
      tile_descriptor.f.uncompressed = 1; // Input tile is uncompressed
      tile_descriptor.f.x_dim        = 256; 
      tile_descriptor.f.y_dim        = 1; 
      tile_descriptor.f.z_dim        = 4; 
      //tile_descriptor.f.blobs_per_xy_plane = 0;
      //tile_descriptor.f.blobs_y_start = 0;
      for (uint i=0; i<TILE_DESC_SIZE; i++) cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32+i]=tile_descriptor.val[i];
      tile_descriptor.f.in_data_format  = row_pool ? (uint) DataFormat::Float32 : unpack_src_format[unpB_operand];
      for (uint i=0; i<TILE_DESC_SIZE; i++) cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32+i]=tile_descriptor.val[i];
   
      // Set unpacker config
      unpack_config_u config;
      for (uint i=0; i<CONFIG_SIZE; i++) {
         config.val[i] = 0;
      }
      config.f.out_data_format = unpack_dst_format[unpA_operand];
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

      config.f.out_data_format = row_pool ? ((uint) DataFormat::Float16 | (exp_width<<2)) : unpack_dst_format[unpB_operand];
      config.f.haloize_mode    = 0;

      for (uint i=0; i<CONFIG_SIZE; i++) cfg[THCON_SEC1_REG2_Out_data_format_ADDR32+i]=config.val[i];
      
      uint unp0_x_end = (srca_face_height == 0) ? 1 : (srca_face_height << 4) - 1;
      TTI_SETADCXX(p_setadc::UNP0, unp0_x_end, 0x0);
      TTI_SETADCXX(p_setadc::UNP1, (srcb_face_height << 4)-1, 0x0);
   
      // Program base address for all 2 sections (each section address is loaded to corresponding context)
      // Load dummy data to unused location if face height is 0
      const uint Dest_cntx0_address = srca_face_height == 0 ? 22*16 : 4 * 16;
      const uint Dest_cntx1_address = srca_face_height == 0 ? 22*16 : 4 * 16; 
      cfg[THCON_SEC0_REG5_Dest_cntx0_address_ADDR32] = Dest_cntx0_address | (Dest_cntx1_address << 16);
   
      // Program unpacker0 per context x_dim
      const uint Tile_x_dim = 256;
      cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = Tile_x_dim | (Tile_x_dim << 16);

      regfile[p_gpr_unpack::TILE_SIZE_A]   = GET_L1_TILE_SIZE((uint)unpack_src_format[unpA_operand]);
      regfile[p_gpr_unpack::TILE_SIZE_B]   = GET_L1_TILE_SIZE((uint)unpack_src_format[unpB_operand]);
      sync_regfile_write(p_gpr_unpack::TILE_SIZE_B);

      TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4);

      /*
      // Workaround for HW bug (fp32 dest and movd2a/b is used with srcA/B configured with 5-bit exponent)
      if (is_fp32_dest_acc_en && (exp_width == 0)) {
          reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 1<<11); // Set debug feature disable bit 11 
                                                                 // workaround for bug tenstorrent/budabackend#1372
      }
      */

      // Clear context ID
      reset_config_context();
   }


   inline void configure_unpack_AB_tile_size(const std::uint32_t in0_tile_dims[2] = default_tile_dims, const std::uint32_t in1_tile_dims[2] = default_tile_dims) {

     TTI_SETADCXX(0b11, TILE_WIDTH*TILE_HEIGHT-1, 0x0);

     /*
     if ((in0_tile_dims[TileDim::R_IDX] <= FACE_HEIGHT) && (in0_tile_dims[TileDim::C_IDX] <= FACE_WIDTH)) {
       TTI_SETADCXX(0b10, FACE_HEIGHT*FACE_WIDTH-1, 0x0); //16x16
     } else if (in0_tile_dims[TileDim::R_IDX] <= FACE_HEIGHT) { 
       TTI_SETADCXX(0b10, FACE_HEIGHT*TILE_WIDTH-1, 0x0); //16x32
     } else if (in0_tile_dims[TileDim::C_IDX] <= FACE_WIDTH) { 
       TTI_SETADCXX(0b10, FACE_HEIGHT*FACE_WIDTH-1, 0x0); //16x16+16x16
     }

     if ((in1_tile_dims[TileDim::R_IDX] <= FACE_HEIGHT) && (in1_tile_dims[TileDim::C_IDX] <= FACE_WIDTH)) {
       TTI_SETADCXX(0b01, FACE_HEIGHT*FACE_WIDTH-1, 0x0); //16x16
     } else if (in1_tile_dims[TileDim::R_IDX] <= FACE_HEIGHT) { 
       TTI_SETADCXX(0b01, FACE_HEIGHT*TILE_WIDTH-1, 0x0); //16x32
     } else if (in1_tile_dims[TileDim::C_IDX] <= FACE_WIDTH) { 
       TTI_SETADCXX(0b01, FACE_HEIGHT*FACE_WIDTH-1, 0x0); //16x16+16x16
     }
     */

   }

   inline uint32_t get_operand_id(uint32_t operand) 
   {
      return (operand>=INTERMEDIATE_BASE_ID) ? operand - 8 : operand - OPERAND_BASE_ID;
   }
}

