#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "fw_debug.h"


namespace ckernel::packer
{
   constexpr uint32_t OUTPUT_BASE    = 16; 
   constexpr uint32_t OUTPUT_BASE_ID = 0; 
   constexpr uint32_t PACK_CNT       = 4; 
   
   
   constexpr uint PACK_SEL(const uint pack_count)
   {
     return (pack_count == 1) ? 0x1 :
            (pack_count == 2) ? 0x3 :
            (pack_count == 4) ? 0xF : 0x0;
   }

   // Pack config
   typedef struct {
      //word 0
     uint32_t row_ptr_section_size : 16;
     uint32_t exp_section_size : 16;
      //word 1
     uint32_t l1_dest_addr: 32;
      //word 2
     uint32_t uncompress  : 1;
     uint32_t add_l1_dest_addr_offset  : 1;
     uint32_t reserved_0  : 2;
     uint32_t out_data_format  : 4;
     uint32_t in_data_format  : 4;
     uint32_t reserved_1  : 4;
     uint32_t src_if_sel  : 1;
     uint32_t pack_per_xy_plane  : 7;
     uint32_t l1_src_addr : 8;
     //word 3
     uint32_t downsample_mask : 16;
     uint32_t downsample_shift_count  : 3;
     uint32_t read_mode : 1;
     uint32_t exp_threshold_en  : 1;
     uint32_t pack_l1_acc_disable_pack_zero_flag : 2;
     uint32_t reserved_2 : 1;
     uint32_t exp_threshold : 8;
   } pack_config_t;

   static_assert(sizeof(pack_config_t) == (sizeof(uint32_t)*4));
   
   typedef union {
     uint32_t val[4];
     pack_config_t f;
   } pack_config_u;

   // Relu Config
   typedef struct {
      uint32_t ALU_ACC_CTRL_Zero_Flag_disabled_src      :1;
      uint32_t ALU_ACC_CTRL_Zero_Flag_disabled_dst      :1;
      uint32_t STACC_RELU_ApplyRelu                     :4;
      uint32_t STACC_RELU_ReluThreshold                 :16;
      uint32_t DISABLE_RISC_BP_Disable_main             :1;
      uint32_t DISABLE_RISC_BP_Disable_trisc            :3;
      uint32_t DISABLE_RISC_BP_Disable_ncrisc           :1;
      uint32_t DISABLE_RISC_BP_Disable_bmp_clear_main   :1;
      uint32_t DISABLE_RISC_BP_Disable_bmp_clear_trisc  :3;
      uint32_t DISABLE_RISC_BP_Disable_bmp_clear_ncrisc :1;
   }relu_config_t;

   static_assert(sizeof(relu_config_t) == (sizeof(uint32_t)));

   typedef union {
      uint32_t val[1];
      relu_config_t r;
   }relu_config_u;

   // Dest rd control
   typedef struct {
      uint32_t PCK_DEST_RD_CTRL_Read_32b_data : 1;
      uint32_t PCK_DEST_RD_CTRL_Read_unsigned : 1;
      uint32_t PCK_DEST_RD_CTRL_Read_int8 : 1;
      uint32_t PCK_DEST_RD_CTRL_Round_10b_mant : 1;
      uint32_t PCK_DEST_RD_CTRL_Reserved : 28;
   }dest_rd_ctrl_t;

   static_assert(sizeof(dest_rd_ctrl_t) == (sizeof(uint32_t)));

   typedef union {
      uint32_t val;
      dest_rd_ctrl_t f;
   } dest_rd_ctrl_u;

   // Set unpacker offsets to 0, except for unpacker 0, channel 1, X, which is the tile X dimension
   inline void packer_addr_counter_init()
   {
       TTI_SETADCXY(0b100, 0, 0, 0, 0, 0b1011);
       TTI_SETADCZW(0b100, 0, 0, 0, 0, 0b1111);
   }

   inline void set_packer_strides(const uint operand_id){

      // Get pointer to registers for current state ID
      volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

      uint x_stride = (uint)(pack_src_format[operand_id]&0x3) == (uint)DataFormat::Float32 ? 4 : 
                      (uint)(pack_src_format[operand_id]&0x3) == (uint)DataFormat::Float16 ? 2 : 1;
      uint y_stride = 16*x_stride;
      uint z_stride = PACK_CNT*16*y_stride;
      uint w_stride = z_stride;
   
      TT_SETDMAREG(0, LOWER_HALFWORD((y_stride<<PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, LO_16(p_gpr_pack::TMP0)); //x-stride not used!
      TT_SETDMAREG(0, UPPER_HALFWORD((y_stride<<PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, HI_16(p_gpr_pack::TMP0));
      TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
      TT_SETDMAREG(0, LOWER_HALFWORD((w_stride<<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT)), 0, LO_16(p_gpr_pack::TMP0)); //z-stride not used!
      TT_SETDMAREG(0, UPPER_HALFWORD((w_stride<<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT)), 0, HI_16(p_gpr_pack::TMP0));
      TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
      TTI_NOP; TTI_NOP;
   }

   template <bool is_fp32_dest_acc_en>
   inline void set_packer_config(const uint operand_id){

      // Get pointer to registers for current state ID
      volatile uint tt_reg_ptr *cfg = get_cfg_pointer();
      
      // Set packer config
      pack_config_u config;
      for (uint i=0; i<4; i++) {
         config.val[i] = 0;
      }

      config.f.exp_section_size = (((uint)pack_dst_format[operand_id] == (uint)DataFormat::Lf8) || 
                                   ((uint)pack_dst_format[operand_id] == (uint)DataFormat::Int8)) ? 0 : 4; // set to 4 as exp section size is not used for non-bfp formats except for lf8/int8

      config.f.uncompress   = 1;
      config.f.out_data_format   = (uint)pack_dst_format[operand_id];
      config.f.in_data_format    = (uint)pack_src_format[operand_id];
      config.f.pack_per_xy_plane = 1;

      // Program:
      // THCON_SEC0_REG1_Row_start_section_size = cfg_reg_array[1][0 +: 16];
      // THCON_SEC0_REG1_Exp_section_size = cfg_reg_array[1][16 +: 16];
      // This is filled with garbage, and will be set up on every pack: 
      //           THCON_SEC0_REG1_L1_Dest_addr = cfg_reg_array[1][32 +: 32];    
      // THCON_SEC0_REG1_Disable_zero_compress = cfg_reg_array[1][64 +: 1];
      // THCON_SEC0_REG1_Add_l1_dest_addr_offset = cfg_reg_array[1][65 +: 1];
      // THCON_SEC0_REG1_Unused0 = cfg_reg_array[1][66 +: 2];
      // THCON_SEC0_REG1_Out_data_format = cfg_reg_array[1][68 +: 4];
      // THCON_SEC0_REG1_In_data_format = cfg_reg_array[1][72 +: 4];
      // THCON_SEC0_REG1_Unused00 = cfg_reg_array[1][76 +: 4];
      // THCON_SEC0_REG1_Source_interface_selection = cfg_reg_array[1][80 +: 1];
      // THCON_SEC0_REG1_Packs_per_xy_plane = cfg_reg_array[1][81 +: 7];
      // THCON_SEC0_REG1_L1_source_addr = cfg_reg_array[1][88 +: 8];
      // THCON_SEC0_REG1_Downsample_mask = cfg_reg_array[1][96 +: 16];
      // THCON_SEC0_REG1_Downsample_shift_count = cfg_reg_array[1][112 +: 3];
      // THCON_SEC0_REG1_Read_mode = cfg_reg_array[1][115 +: 1];
      // THCON_SEC0_REG1_Exp_threshold_en = cfg_reg_array[1][116 +: 1];
      // THCON_SEC0_REG1_Unused1 = cfg_reg_array[1][117 +: 3];
      // THCON_SEC0_REG1_Exp_threshold = cfg_reg_array[1][120 +: 8];
      // for (uint i=0; i<4; i++) cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32+i]=config.val[i];
      // for (uint i=0; i<4; i++) cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32+i]=config.val[i];
      // for (uint i=0; i<4; i++) cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32+i]=config.val[i];
      // for (uint i=0; i<4; i++) cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32+i]=config.val[i];
      cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32+0]=config.val[0];
      cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32+0]=config.val[0];
      cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32+0]=config.val[0];
      cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32+0]=config.val[0];
      cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32+2]=config.val[2];
      cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32+2]=config.val[2];
      cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32+2]=config.val[2];
      cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32+2]=config.val[2];
      cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32+3]=config.val[3];
      cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32+3]=config.val[3];
      cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32+3]=config.val[3];
      cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32+3]=config.val[3];

      dest_rd_ctrl_u dest_rd_ctrl;
      dest_rd_ctrl.val = 0;
      dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_32b_data = ((uint)pack_src_format[operand_id] == (uint)DataFormat::Int8) | (is_fp32_dest_acc_en ? 1 : 0);
      cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32] = dest_rd_ctrl.val;

      if (IS_BFP_FORMAT(pack_dst_format[operand_id])) {
         // Override exp section size for packers 1,2,3
         // Tile header + exp size + datum size 
         if ((uint)(pack_dst_format[operand_id]&0x1F) == (uint)DataFormat::Bfp8 || (uint)(pack_dst_format[operand_id]&0x1F) == (uint)DataFormat::Bfp8_b) {
            config.f.exp_section_size = 1 + 2 + 16;
            cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32+0]=config.val[0];
            config.f.exp_section_size = 1 + 1 + 32;
            cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32+0]=config.val[0];
            config.f.exp_section_size = 1 + 0 + 48;
            cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32+0]=config.val[0];
         } else if ((uint)(pack_dst_format[operand_id]&0x1F) == (uint)DataFormat::Bfp4 || (uint)(pack_dst_format[operand_id]&0x1F) == (uint)DataFormat::Bfp4_b) {
            config.f.exp_section_size = 1 + 2 + 8;
            cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32+0]=config.val[0];
            config.f.exp_section_size = 1 + 1 + 16;
            cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32+0]=config.val[0];
            config.f.exp_section_size = 1 + 0 + 24;
            cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32+0]=config.val[0];
         } else if ((uint)(pack_dst_format[operand_id]&0x1F) == (uint)DataFormat::Bfp2 || (uint)(pack_dst_format[operand_id]&0x1F) == (uint)DataFormat::Bfp2_b) {
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

      // Save to GPR for quick data format reconfig 
      regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP]  = (         4) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::EXP1_SEC_SIZE_BFP8] = (1 + 2 + 16) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::EXP2_SEC_SIZE_BFP8] = (1 + 1 + 32) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::EXP3_SEC_SIZE_BFP8] = (1 + 0 + 48) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::EXP1_SEC_SIZE_BFP4] = (1 + 2 + 8 ) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::EXP2_SEC_SIZE_BFP4] = (1 + 1 + 16) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::EXP3_SEC_SIZE_BFP4] = (1 + 0 + 24) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::EXP1_SEC_SIZE_BFP2] = (1 + 2 + 4 ) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::EXP2_SEC_SIZE_BFP2] = (1 + 1 + 8 ) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
      regfile[p_gpr_pack::EXP3_SEC_SIZE_BFP2] = (1 + 0 + 12) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
      sync_regfile_write(p_gpr_pack::EXP3_SEC_SIZE_BFP2);
   }

   inline void set_packer_l1_offset(const uint operand_id){

      uint32_t l1_offset_1 = IS_BFP_FORMAT(pack_dst_format[operand_id]) ? 1 : (((uint8_t)(pack_dst_format[operand_id]&0x3) == (uint8_t)DataFormat::Float32)  ? 0x40 : 
                                                                               ((uint8_t)(pack_dst_format[operand_id]&0x3) == (uint8_t)DataFormat::Float16)  ? 0x20 : 0x10);
      uint32_t l1_offset_2 = 2 * l1_offset_1;
      uint32_t l1_offset_3 = 3 * l1_offset_1;

      //HW automatically offsets packers base address by tile header size
      //with new L1 addressing mode, the effective address for pack1/2/3
      //will be pack[i] += pack[0], which leads to double counting of tile header
      //subtract by this amount when programming the offset
      constexpr uint32_t PACK_TILE_HEADER_OFFSET = 1; //in 16B
      l1_offset_1 -= PACK_TILE_HEADER_OFFSET;
      l1_offset_2 -= PACK_TILE_HEADER_OFFSET;
      l1_offset_3 -= PACK_TILE_HEADER_OFFSET;
      TT_SETDMAREG(0, LOWER_HALFWORD(l1_offset_1), 0, LO_16(p_gpr_pack::TMP_LO));
      TTI_REG2FLOP(2,0,0,0,THCON_SEC0_REG8_L1_Dest_addr_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);
      TT_SETDMAREG(0, LOWER_HALFWORD(l1_offset_2), 0, LO_16(p_gpr_pack::TMP_LO));
      TTI_REG2FLOP(2,0,0,0,THCON_SEC1_REG1_L1_Dest_addr_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);
      TT_SETDMAREG(0, LOWER_HALFWORD(l1_offset_3), 0, LO_16(p_gpr_pack::TMP_LO));
      TTI_REG2FLOP(2,0,0,0,THCON_SEC1_REG8_L1_Dest_addr_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);
   }   


   inline void reconfig_packer_data_format(const uint operand_id)
   {
      // Get pointer to registers for current state ID
      volatile uint *cfg = get_cfg_pointer();

      // Configure packers
      pack_config_u config;
      config.val[2] = 0; // Only need to modify word[2][15:0]

      config.f.uncompress   = 1;
      config.f.out_data_format   = (uint)pack_dst_format[operand_id];
      config.f.in_data_format    = (uint)pack_src_format[operand_id];
      TT_SETDMAREG(0, LOWER_HALFWORD(config.val[2]), 0, LO_16(p_gpr_pack::TMP_LO));
      TTI_REG2FLOP(2,0,0,0,THCON_SEC0_REG1_Row_start_section_size_ADDR32+2-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO); //16-bit write
      TTI_REG2FLOP(2,0,0,0,THCON_SEC0_REG8_Row_start_section_size_ADDR32+2-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);
      TTI_REG2FLOP(2,0,0,0,THCON_SEC1_REG1_Row_start_section_size_ADDR32+2-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);
      TTI_REG2FLOP(2,0,0,0,THCON_SEC1_REG8_Row_start_section_size_ADDR32+2-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);

      if (IS_BFP_FORMAT(pack_dst_format[operand_id])) {
         // Override exp section size for packers 1,2,3
         // Tile header + exp size + datum size 
         TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG1_Row_start_section_size_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP0_SEC_SIZE_BFP);
         if ((uint)(pack_dst_format[operand_id]&0x1F) == (uint)DataFormat::Bfp8 || (uint)(pack_dst_format[operand_id]&0x1F) == (uint)DataFormat::Bfp8_b) {
            TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG8_Row_start_section_size_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP1_SEC_SIZE_BFP8);
            TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG1_Row_start_section_size_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP2_SEC_SIZE_BFP8);
            TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG8_Row_start_section_size_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP3_SEC_SIZE_BFP8);
         } else if ((uint)(pack_dst_format[operand_id]&0x1F) == (uint)DataFormat::Bfp4 || (uint)(pack_dst_format[operand_id]&0x1F) == (uint)DataFormat::Bfp4_b) {
            TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG8_Row_start_section_size_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP1_SEC_SIZE_BFP4);
            TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG1_Row_start_section_size_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP2_SEC_SIZE_BFP4);
            TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG8_Row_start_section_size_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP3_SEC_SIZE_BFP4);
         } else if ((uint)(pack_dst_format[operand_id]&0x1F) == (uint)DataFormat::Bfp2 || (uint)(pack_dst_format[operand_id]&0x1F) == (uint)DataFormat::Bfp2_b) {
            TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG8_Row_start_section_size_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP1_SEC_SIZE_BFP2);
            TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG1_Row_start_section_size_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP2_SEC_SIZE_BFP2);
            TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG8_Row_start_section_size_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP3_SEC_SIZE_BFP2);
         } else {
            FWASSERT("Other data formats not supported", false);
         }
      } else if (((uint)pack_dst_format[operand_id] == (uint)DataFormat::Lf8) || 
                 ((uint)pack_dst_format[operand_id] == (uint)DataFormat::Int8)) {
         TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG1_Row_start_section_size_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
         TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG8_Row_start_section_size_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
         TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG1_Row_start_section_size_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
         TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG8_Row_start_section_size_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
      }   

      // Set l1 address offset
      set_packer_l1_offset(operand_id);

      TT_SETDMAREG(0, LOWER_HALFWORD(GET_L1_TILE_SIZE((uint)pack_dst_format[operand_id])), 0, LO_16(p_gpr_pack::TILE_HEADER));


      // Flush packer pipeline before strides gasket alu format change
      TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
      cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG2_Dstacc_RMW>(pack_src_format[operand_id]);

      tensix_sync(); //FIXME: why stallwait on cfg write doesn't work!

      // Set packer strides
      set_packer_strides(operand_id);


   }

   template <bool is_fp32_dest_acc_en>
   inline void configure_pack(uint pack_output, uint relu_config = 0)
   {
      // Get pointer to registers for current state ID
      volatile uint *cfg = get_cfg_pointer();

      if (pack_src_format[pack_output] != pack_dst_format[pack_output]) {
         TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK);
         tensix_sync();	         
      }	      

      set_packer_strides(pack_output);

      t6_mutex_acquire(mutex::REG_RMW);

      uint alu_dst_format = pack_src_format[pack_output];

      cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG2_Dstacc_RMW>(alu_dst_format);

      t6_mutex_release(mutex::REG_RMW);

      set_packer_config<is_fp32_dest_acc_en>(pack_output);

      set_packer_l1_offset(pack_output);

      // PACK_COUNTERS_SEC0_pack_per_xy_plane = cfg_reg_array[3][0 +: 8];
      // PACK_COUNTERS_SEC0_pack_reads_per_xy_plane = cfg_reg_array[3][8 +: 8];
      // PACK_COUNTERS_SEC0_pack_xys_per_tile = cfg_reg_array[3][16 +: 7];
      // PACK_COUNTERS_SEC0_pack_yz_transposed = cfg_reg_array[3][23 +: 1];
      for (uint i=0; i<4; i++) cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32+i]=0; // disable auto last generation

      for (uint i=0; i<4; i++) {
         cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+i]=0xffffffff;
         cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32+i] = 0x0;
      }

      regfile[p_gpr_pack::TILE_HEADER]   = GET_L1_TILE_SIZE((uint)pack_dst_format[pack_output]);
      regfile[p_gpr_pack::TILE_HEADER+1] = 0;
      regfile[p_gpr_pack::TILE_HEADER+2] = 0;
      regfile[p_gpr_pack::TILE_HEADER+3] = 0;
      sync_regfile_write(p_gpr_pack::TILE_HEADER+3);

      relu_config_u hw_relu_config;
      // Config RELU
      uint32_t current_relu_val = reg_read((uint)&cfg[STACC_RELU_ApplyRelu_ADDR32]);
      hw_relu_config.val[0] = current_relu_val;

      hw_relu_config.r.STACC_RELU_ApplyRelu      = relu_config&0xffff;
      hw_relu_config.r.STACC_RELU_ReluThreshold  = (relu_config>>16)&0xffff;

      cfg[STACC_RELU_ApplyRelu_ADDR32] = hw_relu_config.val[0];

      // Assume face height 16
      const uint pack_per_xy_plane = 16;
      TTI_SETADCXX(p_setadc::PAC, (256/pack_per_xy_plane)-1, 0x0);
   }

   template <DstTileFaceLayout FaceLayout, bool untilize, bool is_fp32_dest_acc_en>
   inline void init_packer_dest_offset_registers()
   {
      //Issue #3064: to avoid sfpu and packer stalling when dest is in FP32 mode
      //             use dest offset of 0x200 instead of 0x100
      //             Wormhole a0/b0 HW translates these addreses to the correct dest bank,
      //             however dest capacity is unchanged (e.g 0x100 to 0x1FF should be unused now)
      constexpr uint32_t DEST_OFFSET_SHIFT = 0; //is_fp32_dest_acc_en ? (1) : (0);
      constexpr uint32_t DEST_HALF_OFFSET = DEST_REGISTER_HALF_SIZE >> DEST_OFFSET_SHIFT;
      
      if constexpr (untilize) {
         if constexpr (FaceLayout == ColMajor) {
            // Packer0 :  0,32,  1,33 ...  7, 39
	    // Packer1 :  8,40,  9,41 ... 15, 47
	    // Packer2 : 16,48, 17,49 ... 23, 55		  
	    // Packer3 : 23,56, 24,57 ... 31, 63		  
            regfile[p_gpr_pack::DEST_OFFSET_LO]   = 0x0;
            regfile[p_gpr_pack::DEST_OFFSET_LO+1] = 0x0 + 0x8;
            regfile[p_gpr_pack::DEST_OFFSET_LO+2] = 0x0 + 0x10;
            regfile[p_gpr_pack::DEST_OFFSET_LO+3] = 0x0 + 0x18;
            regfile[p_gpr_pack::DEST_OFFSET_HI]   = DEST_HALF_OFFSET;
            regfile[p_gpr_pack::DEST_OFFSET_HI+1] = DEST_HALF_OFFSET + 0x8;
            regfile[p_gpr_pack::DEST_OFFSET_HI+2] = DEST_HALF_OFFSET + 0x10;
            regfile[p_gpr_pack::DEST_OFFSET_HI+3] = DEST_HALF_OFFSET + 0x18;
         } else {		 
            // Packer0 :  0,16,  1,17 ...  7, 23
	    // Packer1 :  8,24,  9,25 ... 15, 31
	    // Packer2 : 32,48, 33,49 ... 39, 55		  
	    // Packer3 : 40,56, 41,57 ... 47, 63		  
            regfile[p_gpr_pack::DEST_OFFSET_LO]   = 0x0;
            regfile[p_gpr_pack::DEST_OFFSET_LO+1] = 0x0 + 0x8;
            regfile[p_gpr_pack::DEST_OFFSET_LO+2] = 0x0 + 0x20;
            regfile[p_gpr_pack::DEST_OFFSET_LO+3] = 0x0 + 0x28;
            regfile[p_gpr_pack::DEST_OFFSET_HI]   = DEST_HALF_OFFSET;
            regfile[p_gpr_pack::DEST_OFFSET_HI+1] = DEST_HALF_OFFSET + 0x8;
            regfile[p_gpr_pack::DEST_OFFSET_HI+2] = DEST_HALF_OFFSET + 0x20;
            regfile[p_gpr_pack::DEST_OFFSET_HI+3] = DEST_HALF_OFFSET + 0x28;
	 }    
      } else { 
         if constexpr (FaceLayout == ColMajor) {
            regfile[p_gpr_pack::DEST_OFFSET_LO]   = 0x0;
            regfile[p_gpr_pack::DEST_OFFSET_LO+1] = 0x0 + 0x20;
            regfile[p_gpr_pack::DEST_OFFSET_LO+2] = 0x0 + 0x10;
            regfile[p_gpr_pack::DEST_OFFSET_LO+3] = 0x0 + 0x30;
            regfile[p_gpr_pack::DEST_OFFSET_HI]   = DEST_HALF_OFFSET;
            regfile[p_gpr_pack::DEST_OFFSET_HI+1] = DEST_HALF_OFFSET + 0x20;
            regfile[p_gpr_pack::DEST_OFFSET_HI+2] = DEST_HALF_OFFSET + 0x10;
            regfile[p_gpr_pack::DEST_OFFSET_HI+3] = DEST_HALF_OFFSET + 0x30;
         } else { // Default to row major layout
            regfile[p_gpr_pack::DEST_OFFSET_LO]   = 0x0;
            regfile[p_gpr_pack::DEST_OFFSET_LO+1] = 0x0 + 0x10;
            regfile[p_gpr_pack::DEST_OFFSET_LO+2] = 0x0 + 0x20;
            regfile[p_gpr_pack::DEST_OFFSET_LO+3] = 0x0 + 0x30;
            regfile[p_gpr_pack::DEST_OFFSET_HI]   = DEST_HALF_OFFSET;
            regfile[p_gpr_pack::DEST_OFFSET_HI+1] = DEST_HALF_OFFSET + 0x10;
            regfile[p_gpr_pack::DEST_OFFSET_HI+2] = DEST_HALF_OFFSET + 0x20;
            regfile[p_gpr_pack::DEST_OFFSET_HI+3] = DEST_HALF_OFFSET + 0x30;
         }
      }
      sync_regfile_write(p_gpr_pack::DEST_OFFSET_HI+3);
   }

   inline uint8_t get_packer_dest_offset_index()
   {
           return (dest_offset_id ? p_gpr_pack::DEST_OFFSET_HI : p_gpr_pack::DEST_OFFSET_LO);
   }

   inline uint32_t get_packer_dest_offset()
   {
           return (dest_offset_id ? DEST_REGISTER_HALF_SIZE : 0x0);
   }

   inline void flip_packer_dest_offset_id()
   {
           dest_offset_id = 1 - dest_offset_id;
   }
   
   // Flip packer dest register offset to 0 or DEST_REGISTER_HALF_SIZE
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
           TTI_DMANOP;TTI_DMANOP;
   }

   // Program packer destination addresses from GPRs
   template <PackSelMask PackSel=PACK_ALL>
   inline void program_packer_destination(uint32_t addr, uint8_t pack_output)
   {
      uint32_t new_l1_addr = (1 << 31) | addr;
      TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
      TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));

      //TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
      TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG1_L1_Dest_addr_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR);

      TTI_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 0); // pack flush

      TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
   }

   inline void program_packer_dest_offset_registers(uint32_t dest_tile_offset)
   {
      TT_SETDMAREG(0, LOWER_HALFWORD(dest_tile_offset), 0, LO_16(p_gpr_pack::TEMP_TILE_OFFSET));
      TT_SETDMAREG(0, UPPER_HALFWORD(dest_tile_offset), 0, HI_16(p_gpr_pack::TEMP_TILE_OFFSET));
      TTI_WRCFG(p_gpr_pack::TEMP_TILE_OFFSET, p_cfg::WRCFG_32b, PCK0_ADDR_BASE_REG_0_Base_ADDR32);
      TTI_DMANOP;TTI_DMANOP;
   }

   inline void reconfigure_packer_l1_acc(const std::uint32_t pack_l1_acc){

      //assumes all configured packers have these fields as common values
      // pack_config_u pack_config;
      // pack_config.val[3] = 0;
      // pack_config.f.pack_l1_acc_disable_pack_zero_flag = pack_l1_acc ? (0b11) : (0b00);

      // TT_SETDMAREG(0, pack_config.val[3] & 0xffff, 0, LO_16(p_gpr_pack::TMP0));
      // TT_SETDMAREG(0, (pack_config.val[3] >> 16) & 0xffff, 0, HI_16(p_gpr_pack::TMP0));
      // TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG1_Pack_L1_Acc_ADDR32);
      // TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG8_Pack_L1_Acc_ADDR32);
      // TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, THCON_SEC1_REG1_Pack_L1_Acc_ADDR32);
      // TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, THCON_SEC1_REG8_Pack_L1_Acc_ADDR32);
      // TTI_DMANOP;TTI_DMANOP;

      // TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::TRISC_CFG);

      const uint32_t pack_l1_acc_disable_pack_zero_flag = pack_l1_acc ? (0b11) : (0b00);

      cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pack_L1_Acc_ADDR32, THCON_SEC0_REG1_Pack_L1_Acc_SHAMT, THCON_SEC0_REG1_Disable_pack_zero_flags_MASK | THCON_SEC0_REG1_Pack_L1_Acc_MASK>(pack_l1_acc_disable_pack_zero_flag);
      cfg_reg_rmw_tensix<THCON_SEC0_REG8_Pack_L1_Acc_ADDR32, THCON_SEC0_REG8_Pack_L1_Acc_SHAMT, THCON_SEC0_REG8_Disable_pack_zero_flags_MASK | THCON_SEC0_REG8_Pack_L1_Acc_MASK>(pack_l1_acc_disable_pack_zero_flag);
      cfg_reg_rmw_tensix<THCON_SEC1_REG1_Pack_L1_Acc_ADDR32, THCON_SEC1_REG1_Pack_L1_Acc_SHAMT, THCON_SEC1_REG1_Disable_pack_zero_flags_MASK | THCON_SEC1_REG1_Pack_L1_Acc_MASK>(pack_l1_acc_disable_pack_zero_flag);
      cfg_reg_rmw_tensix<THCON_SEC1_REG8_Pack_L1_Acc_ADDR32, THCON_SEC1_REG8_Pack_L1_Acc_SHAMT, THCON_SEC1_REG8_Disable_pack_zero_flags_MASK | THCON_SEC1_REG8_Pack_L1_Acc_MASK>(pack_l1_acc_disable_pack_zero_flag);
   }

   // Write tile header to l1
   inline void write_tile_header()
   {
      TTI_STOREIND (1, 0, p_ind::LD_16B, LO_16(0), p_ind::INC_NONE, p_gpr_pack::TILE_HEADER, p_gpr_pack::OUTPUT_ADDR);
   }

   inline uint32_t get_output_id(uint32_t output) 
   {
      return ((output) - OUTPUT_BASE);
   }

   inline constexpr uint32_t get_output_base_id() 
   {
      return (OUTPUT_BASE_ID);
   }
}
