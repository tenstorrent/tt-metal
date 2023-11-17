#pragma once

#include "ckernel.h"
#include "ckernel_globals.h"
#include "fw_debug.h"

#ifdef PERF_DUMP
#include "perf_res_decouple.h"
#endif

namespace ckernel::unpacker
{
   constexpr uint32_t TILE_DESC_SIZE = 2; //Unpacker descriptor size in dwords
   constexpr uint32_t CONFIG_SIZE = 2; //Unpacker configuration size in dwords

   // Unpack tile descriptor
   typedef struct {
     uint32_t in_data_format : 4;
     uint32_t uncompressed: 1;
     uint32_t reserved_0  : 3;
     uint32_t blobs_per_xy_plane  : 4;
     uint32_t reserved_1  : 4;
     uint32_t x_dim       : 16;
     uint32_t y_dim       : 16;
     uint32_t z_dim       : 16;
     uint32_t w_dim       : 16;
     uint32_t blobs_y_start : 32;
     uint32_t digest_type : 8;  // Not used
     uint32_t digest_size : 8;  // Not used
   } unpack_tile_descriptor_t; // Unpack configuration
   
   typedef union {
     uint32_t val[4];
     unpack_tile_descriptor_t f;
   } unpack_tile_descriptor_u;
   
   // Unpack config
   typedef struct {
      //word 0
     uint32_t out_data_format : 4;
     uint32_t throttle_mode: 2;
     uint32_t context_count  : 2;
     uint32_t haloize_mode  : 1;
     uint32_t tileize_mode  : 1;
     uint32_t force_shared_exp  : 1;
     uint32_t reserved_0  : 1;
     uint32_t upsample_rate  : 3;
     uint32_t upsamle_and_interlave  : 1;
     uint32_t shift_amount : 16;
     //word 1
     uint32_t uncompress_cntx0_3 : 4;
     uint32_t reserved_1  : 12;
     uint32_t uncompress_cntx4_7 : 4;
     uint32_t reserved_2  : 12;
     //word 2
     uint32_t limit_addr : 16;
     uint32_t fifo_size : 16;
   } unpack_config_t; 
   
   typedef union {
     uint32_t val[4];
     unpack_config_t f;
   } unpack_config_u;

   typedef struct {
     uint32_t z: 12;
     uint32_t w: 12;
     uint32_t reserved : 8;
   } unpack_zw_stride_t; 

   typedef union {
     uint32_t val;
     unpack_zw_stride_t f;
   } unpack_zw_stride_u;

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

   inline void wait_for_pack_config_done()
   {
       while (semaphore_read(semaphore::UNPACK_PACK_CONFIG_SYNC) > 0) {}
   }

   inline void configure_unpack_AB(
     const uint unpA_src_format, 
     const uint unpB_src_format, 
     const uint unpA_dst_format, 
     const uint unpB_dst_format, 
     const uint srca_face_height=16,
     const uint srcb_face_height=16,
     const bool row_pool=false,
     const bool skip_alu_format_set=false)
   {
      // Check that unpacker is done (all contexts freed up) before starting hw configuration
      wait_for_idle();	     

      // Reset address counters	      
      unpacker_addr_counter_init();

      // Get pointer to registers for current state ID
      volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

      uint unpA_ch1_x_stride = (uint) (unpA_dst_format&0x3) == (uint) DataFormat::Float32 ? 4 : (uint) (unpA_dst_format&0x3) == (uint) DataFormat::Float16 ? 2 : 1;
      uint unpB_ch1_x_stride = (uint) (unpB_dst_format&0x3) == (uint) DataFormat::Float32 ? 4 : (uint) (unpB_dst_format&0x3) == (uint) DataFormat::Float16 ? 2 : 1;
      uint unpA_ch1_z_stride = 16*srca_face_height*unpA_ch1_x_stride;
      uint unpB_ch1_z_stride = 16*srcb_face_height*unpB_ch1_x_stride;
      uint exp_width = ((uint)unpA_dst_format>>2)&0x1; //0=5-bit, 1=8-bit
   
      // Math ALU_FORMAT_REG
      // MT: Ensure thread safety between unpacker and math threads by using semaphore
      if (!skip_alu_format_set) {
         uint alu_src_format = 
            ((row_pool ? ((uint) DataFormat::Float16 | (exp_width<<2)) : unpB_dst_format) << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT) // Row polling dest format is always 16-bit float
         | (unpA_dst_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT)
         | (0x0 << ALU_FORMAT_SPEC_REG_SrcA_val_SHAMT);
         cfg[ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32] = alu_src_format;
         semaphore_post(semaphore::UNPACK_PACK_CONFIG_SYNC);
      }

      // Strides
      cfg[UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] = (unpA_ch1_z_stride << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT) |
                                                    (                0 << UNP0_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT);  // Z and W stride for dest address (ch1)
      cfg[UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] = (unpB_ch1_z_stride << UNP1_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT) |
                                                    (                0 << UNP1_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT);  // Z and W stride for dest address (ch1)

      // Set tile descriptor
      unpack_tile_descriptor_u tile_descriptor;
      for (uint i=0; i<TILE_DESC_SIZE; i++) {
         tile_descriptor.val[i] = 0;
      }
      tile_descriptor.f.in_data_format  = unpA_src_format;
      tile_descriptor.f.uncompressed = 1; // Input tile is uncompressed
      tile_descriptor.f.x_dim        = 256; 
      tile_descriptor.f.y_dim        = 1; 
      tile_descriptor.f.z_dim        = 4; 
      //tile_descriptor.f.blobs_per_xy_plane = 0;
      //tile_descriptor.f.blobs_y_start = 0;
      for (uint i=0; i<TILE_DESC_SIZE; i++) cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32+i]=tile_descriptor.val[i];
      tile_descriptor.f.in_data_format  = row_pool ? (uint) DataFormat::Float32 : unpB_src_format;
      for (uint i=0; i<TILE_DESC_SIZE; i++) cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32+i]=tile_descriptor.val[i];
   
      // Set unpacker config
      unpack_config_u config;
      for (uint i=0; i<CONFIG_SIZE; i++) {
         config.val[i] = 0;
      }
      config.f.out_data_format = unpA_dst_format;
      config.f.throttle_mode   = 2;
      //config.f.upsample_rate   = 0;
      //config.f.upsamle_and_interlave  = 0;
      //config.f.shift_amount = 0;
      config.f.uncompress_cntx0_3 = 0xf;
      config.f.uncompress_cntx4_7 = 0xf;
      //config.f.limit_addr = 0; // Set dynamically
      //config.f.fifo_size = 0; // Set dynamically
      for (uint i=0; i<CONFIG_SIZE; i++) cfg[THCON_SEC0_REG2_Out_data_format_ADDR32+i]=config.val[i];

      config.f.out_data_format = row_pool ? ((uint) DataFormat::Float16 | (exp_width<<2)) : unpB_dst_format;
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

      // Store config used by tilizer
      regfile[p_gpr_unpack::FACE_DIM_1x16] = (Tile_x_dim/16) | ((Tile_x_dim/16)<<16);
      sync_regfile_write(p_gpr_unpack::FACE_DIM_1x16);

      if (!skip_alu_format_set) {
         wait_for_pack_config_done();
         gl_alu_format_spec_reg = cfg[ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32];
      }

      // Clear context ID
      reset_config_context();
   }
   
   inline uint32_t cfg_rmw_mmio_rd_tensix_wr(uint addr, uint shamt,  uint mask, uint new_val, uint rmw_val) {
      // Write only to the needed data bits
      new_val <<= shamt;
      new_val &= mask;
      rmw_val &= ~mask;

      // Or new data bits
      rmw_val |= new_val;

      TT_SETDMAREG(0, (rmw_val & 0xffff), 0, LO_16(p_gpr_unpack::TMP0));
      TT_SETDMAREG(0, ((rmw_val >> 16) & 0xffff), 0, HI_16(p_gpr_unpack::TMP0));
      
      TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, addr);
      TTI_NOP;TTI_NOP;

      return rmw_val;
   }

   inline void reconfig_unpacker_data_format(const uint src_format, const uint dst_format, const uint32_t tile_addr, const uint32_t out_df_addr, const uint32_t out_df_stride) {

      //volatile uint *cfg = get_cfg_pointer();
      // Set first 32 bites of tile descriptor, only need data format change
      unpack_tile_descriptor_u tile_descriptor = {0};

      tile_descriptor.f.in_data_format  = src_format;
      tile_descriptor.f.uncompressed = 1; // Input tile is uncompressed
      tile_descriptor.f.x_dim        = 256; 

      //cfg[tile_addr]=tile_descriptor.val[0];
      TT_SETDMAREG(0, LOWER_HALFWORD(tile_descriptor.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
      TT_SETDMAREG(0, UPPER_HALFWORD(tile_descriptor.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
      TT_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, tile_addr);
      TTI_NOP;TTI_NOP;
   
      // Set first 32 bites of tile unpacker config, only need data format change
      unpack_config_u config = {0};

      config.f.out_data_format = dst_format;
      config.f.throttle_mode = 2;

      //cfg[out_df_addr]=config.val[0];
      TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
      TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
      TT_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, out_df_addr);
      TTI_NOP;TTI_NOP;

      // Set ch1/dst address stride (needed for matmul)
      uint x_stride = (uint) (dst_format&0x3) == (uint) DataFormat::Float32 ? 4 : (uint) (dst_format&0x3) == (uint)DataFormat::Float16 ? 2 : 1;
      uint z_stride = 16*16*x_stride;
      unpack_zw_stride_u zw_stride = {0};
      zw_stride.f.z = z_stride;
      TT_SETDMAREG(0, LOWER_HALFWORD(zw_stride.val), 0, LO_16(p_gpr_unpack::TMP1));
      TT_SETDMAREG(0, UPPER_HALFWORD(zw_stride.val), 0, HI_16(p_gpr_unpack::TMP1));

      TT_WRCFG(p_gpr_unpack::TMP1, p_cfg::WRCFG_32b, out_df_stride);
      TTI_NOP;TTI_NOP;

   
      // Clear context ID
      //reset_config_context();
    }
}

