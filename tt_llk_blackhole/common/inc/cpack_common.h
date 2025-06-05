// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "llk_defs.h"

namespace ckernel::packer
{
using DataFormatType = std::underlying_type_t<DataFormat>;

constexpr uint replay_buf_offset = 16; // split replay buffer usage between fpu/sfpu
                                       // fist 16 for sfpu, next 16 for fpu
constexpr uint32_t NUM_PACKERS = 1;    // Number of packers

// Pack config
typedef struct
{
    // word 0
    uint32_t row_ptr_section_size : 16;
    uint32_t exp_section_size     : 16;
    // word 1
    uint32_t l1_dest_addr : 32;
    // word 2
    uint32_t uncompress                          : 1;
    uint32_t add_l1_dest_addr_offset             : 1;
    uint32_t disable_pack_zero_flag              : 1;
    uint32_t reserved_0                          : 1;
    uint32_t out_data_format                     : 4;
    uint32_t in_data_format                      : 4;
    uint32_t dis_shared_exp_assembler            : 1;
    uint32_t auto_set_last_pacr_intf_sel         : 1;
    uint32_t enable_out_fifo                     : 1;
    uint32_t sub_l1_tile_header_size             : 1;
    uint32_t src_if_sel                          : 1;
    uint32_t pack_start_intf_pos                 : 4;
    uint32_t all_pack_disable_zero_compress_ovrd : 1;
    uint32_t add_tile_header_size                : 1;
    uint32_t pack_dis_y_pos_start_offset         : 1;
    uint32_t l1_src_addr                         : 8;
    //   The bit unp_lf8_4b_exp is configured in the unpack, remove word 3 to avoid potential race condition
    //   //word 3
    //   uint32_t downsample_mask : 16;
    //   uint32_t downsample_shift_count  : 3;
    //   uint32_t pack_l1_acc : 1; //Not new to BH, but moved
    //   //uint32_t read_mode : 1; //Removed in BH
    //   uint32_t exp_threshold_en  : 1;
    //   uint32_t reserved_2 : 1;
    //   uint32_t unp_lf8_4b_exp: 1;
    //   uint32_t pac_lf8_4b_exp: 1;
    //   uint32_t exp_threshold : 8;
} pack_config_t;

static_assert(sizeof(pack_config_t) == (sizeof(uint32_t) * 3));

typedef union
{
    uint32_t val[3];
    pack_config_t f;
} pack_config_u;

// Relu Config
typedef struct
{
    uint32_t ALU_ACC_CTRL_Zero_Flag_disabled_src      : 1;
    uint32_t ALU_ACC_CTRL_Zero_Flag_disabled_dst      : 1;
    uint32_t STACC_RELU_ApplyRelu                     : 4;
    uint32_t STACC_RELU_ReluThreshold                 : 16;
    uint32_t DISABLE_RISC_BP_Disable_main             : 1;
    uint32_t DISABLE_RISC_BP_Disable_trisc            : 3;
    uint32_t DISABLE_RISC_BP_Disable_ncrisc           : 1;
    uint32_t DISABLE_RISC_BP_Disable_bmp_clear_main   : 1;
    uint32_t DISABLE_RISC_BP_Disable_bmp_clear_trisc  : 3;
    uint32_t DISABLE_RISC_BP_Disable_bmp_clear_ncrisc : 1;
} relu_config_t;

static_assert(sizeof(relu_config_t) == (sizeof(uint32_t)));

typedef union
{
    uint32_t val[1];
    relu_config_t r;
} relu_config_u;

// Dest rd control
typedef struct
{
    uint32_t PCK_DEST_RD_CTRL_Read_32b_data  : 1;
    uint32_t PCK_DEST_RD_CTRL_Read_unsigned  : 1;
    uint32_t PCK_DEST_RD_CTRL_Read_int8      : 1;
    uint32_t PCK_DEST_RD_CTRL_Round_10b_mant : 1;
    uint32_t PCK_DEST_RD_CTRL_Reserved       : 28;
} dest_rd_ctrl_t;

static_assert(sizeof(dest_rd_ctrl_t) == (sizeof(uint32_t)));

typedef union
{
    uint32_t val;
    dest_rd_ctrl_t f;
} dest_rd_ctrl_u;

// PACK_EDGE_OFFSET_SEC[0:3] register sutructure
//
// Lower 16b represent a mask that is applied on a single row of one face on the packer output
// Higher 16b contain information about which TILE_ROW_SET_MAPPING register is used for each packer (only in PACK_EDGE_OFFSET_SEC0)
//
// There are 4 PACK_EDGE_OFFSET_SEC[0:3] registers and 4 TILE_ROW_SET_MAPPING[0:3] registers.
// TILE_ROW_SET_MAPPING[0:3] have 2 bits for each row inside a face that determine which PACK_EDGE_OFFSET_SEC[0:3] mask is used.
// Only PACK_EDGE_OFFSET_SEC0 register has higher 16b configured to determine TILE_ROW_SET_MAPPING[0:3] registers used for each packer.
// Other PACK_EDGE_OFFSET_SEC[1:3] registers are used only for the masks in the lower 16b.
typedef struct
{
    uint32_t mask                      : 16;
    uint32_t mode                      : 1;
    uint32_t tile_row_set_select_pack0 : 2;
    uint32_t tile_row_set_select_pack1 : 2;
    uint32_t tile_row_set_select_pack2 : 2;
    uint32_t tile_row_set_select_pack3 : 2;
    uint32_t reserved                  : 7;
} pck_edge_offset_t;

static_assert(sizeof(pck_edge_offset_t) == (sizeof(uint32_t)));

typedef union
{
    uint32_t val;
    pck_edge_offset_t f;
} pck_edge_offset_u;

// Pack counters
typedef struct
{
    uint32_t pack_per_xy_plane        : 8;
    uint32_t pack_reads_per_xy_plane  : 8;
    uint32_t pack_xys_per_til         : 7;
    uint32_t pack_yz_transposed       : 1;
    uint32_t pack_per_xy_plane_offset : 8;
} pack_counters_t;

static_assert(sizeof(pack_counters_t) == (sizeof(uint32_t)));

typedef union
{
    uint32_t val;
    pack_counters_t f;
} pack_counters_u;

// Set unpacker offsets to 0, except for unpacker 0, channel 1, X, which is the tile X dimension
inline void packer_addr_counter_init()
{
    TTI_SETADCXY(0b100, 0, 0, 0, 0, 0b1011);
    TTI_SETADCZW(0b100, 0, 0, 0, 0, 0b1111);
}

template <bool untilize = false, bool tilize = false>
inline void set_packer_strides(const uint pack_src_format, const uint pack_dst_format, const uint tile_c_dim)
{
    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    uint x_stride = (uint)(pack_src_format & 0x3) == static_cast<DataFormatType>(DataFormat::Float32)   ? 4
                    : (uint)(pack_src_format & 0x3) == static_cast<DataFormatType>(DataFormat::Float16) ? 2
                                                                                                        : 1;
    uint y_stride = FACE_C_DIM * x_stride;
    uint w_stride = TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * x_stride;

    // Untilize mode has 2 packer interfaces active, so z counter needs to jump by 2
    // faces, since z counter is only 1 bit (can't be programmed to inc by 2)
    const uint z_stride = ((untilize ^ tilize) && (tile_c_dim == TILE_C_DIM)) ? 2 * FACE_R_DIM * y_stride : FACE_R_DIM * y_stride;

    TT_SETDMAREG(0, LOWER_HALFWORD((y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, LO_16(p_gpr_pack::TMP0)); // x-stride not used!
    TT_SETDMAREG(0, UPPER_HALFWORD((y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, HI_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, LOWER_HALFWORD((z_stride << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT)), 0, LO_16(p_gpr_pack::TMP1));
    TT_SETDMAREG(0, UPPER_HALFWORD((w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT)), 0, HI_16(p_gpr_pack::TMP1));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
    TTI_NOP;
    TTI_NOP;

    if constexpr (tilize && !untilize)
    {
        const uint z_stride_ch1 = FACE_R_DIM * y_stride;
        TT_SETDMAREG(0, LOWER_HALFWORD((z_stride_ch1 << PCK0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT)), 0, LO_16(p_gpr_pack::TMP1));
        TT_SETDMAREG(0, UPPER_HALFWORD((z_stride_ch1 << PCK0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT)), 0, HI_16(p_gpr_pack::TMP1));
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
        TTI_NOP;
        TTI_NOP;
    }
}

template <bool is_fp32_dest_acc_en>
inline void set_packer_config(const uint pack_src_format, const uint pack_dst_format, const uint num_faces = 4, const bool partial_face = false)
{
    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    const uint pack_output_src_format = (uint)pack_src_format & 0xF;
    const uint pack_output_dst_format = (uint)pack_dst_format & 0xF;

    // Set packer config
    pack_config_u config;
    for (uint i = 0; i < 3; i++)
    {
        config.val[i] = 0;
    }

    config.f.exp_section_size =
        ((pack_output_dst_format == static_cast<DataFormatType>(DataFormat::Lf8)) || (pack_output_dst_format == static_cast<DataFormatType>(DataFormat::Int8)))
            ? 0
            : (partial_face ? 1 : num_faces); // set to num_faces as exp section size is not used for non-bfp formats except for lf8/int8

    config.f.uncompress      = 1;
    config.f.out_data_format = pack_output_dst_format;
    config.f.in_data_format  = pack_output_src_format;

    // Workaround for bug in HW: tenstorrent/budabackend#1394
    if constexpr (is_fp32_dest_acc_en)
    {
        uint exp_threshold_en  = 0;
        uint exp_threshold_val = 0;
        if (IS_BFP_A_FORMAT(pack_output_dst_format))
        {
            exp_threshold_en  = 1;
            exp_threshold_val = 113;
        }
        // EXP threshold is updated in the config word 3 which has a bit programmed by the unpacker as well
        constexpr uint exp_threshold_rmw_mask = THCON_SEC0_REG1_Exp_threshold_en_MASK | THCON_SEC0_REG1_Exp_threshold_MASK;
        uint exp_threshold_rmw_data = (exp_threshold_val << THCON_SEC0_REG1_Exp_threshold_SHAMT) | (exp_threshold_en << THCON_SEC0_REG1_Exp_threshold_en_SHAMT);
        cfg_reg_rmw_tensix<THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 3, 0, exp_threshold_rmw_mask>(exp_threshold_rmw_data);
    }

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
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2] = config.val[2];
    // cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32+3]=config.val[3];

    dest_rd_ctrl_u dest_rd_ctrl;
    dest_rd_ctrl.val = 0;

    bool is_32b_format = pack_src_format == static_cast<DataFormatType>(DataFormat::Int32) ||
                         pack_src_format == static_cast<DataFormatType>(DataFormat::UInt32) ||
                         pack_src_format == static_cast<DataFormatType>(DataFormat::Float32);
    bool is_int8_format = pack_src_format == static_cast<DataFormatType>(DataFormat::Int8) || pack_src_format == static_cast<DataFormatType>(DataFormat::UInt8);

    dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_32b_data = is_32b_format || is_fp32_dest_acc_en;
    dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_int8     = !(is_fp32_dest_acc_en || is_32b_format) && is_int8_format;

    if (pack_dst_format == static_cast<DataFormatType>(DataFormat::UInt8))
    {
        dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_unsigned = 1;
    }

    // Round to 10 bit mantissa from fp32 dest
    if (is_fp32_dest_acc_en && (pack_src_format == static_cast<DataFormatType>(DataFormat::Float16)))
    {
        dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Round_10b_mant = 1;
    }
    cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32] = dest_rd_ctrl.val;

    // Save to GPR for quick data format reconfig
    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP] = (partial_face ? 1 : num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    sync_regfile_write(p_gpr_pack::EXP0_SEC_SIZE_BFP);
}

template <bool is_fp32_dest_acc_en>
inline void reconfig_packer_data_format(
    const uint pack_src_format, const uint pack_dst_format, const uint tile_size, const uint face_r_dim, const uint tile_c_dim)
{
    // Get pointer to registers for current state ID
    volatile uint* cfg = get_cfg_pointer();

    const uint pack_output_src_format = (uint)pack_src_format & 0xF;
    const uint pack_output_dst_format = (uint)pack_dst_format & 0xF;

    // Configure packers
    pack_config_u config;
    config.val[2] = 0; // Only need to modify word[2][15:0]

    config.f.uncompress      = 1;
    config.f.out_data_format = pack_output_dst_format;
    config.f.in_data_format  = pack_output_src_format;
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[2]), 0, LO_16(p_gpr_pack::TMP_LO));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP_LO, p_cfg::WRCFG_32b, THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2);

    dest_rd_ctrl_u dest_rd_ctrl;
    dest_rd_ctrl.val = 0;

    bool is_32b_format = pack_src_format == static_cast<DataFormatType>(DataFormat::Int32) ||
                         pack_src_format == static_cast<DataFormatType>(DataFormat::UInt32) ||
                         pack_src_format == static_cast<DataFormatType>(DataFormat::Float32);
    bool is_int8_format = pack_src_format == static_cast<DataFormatType>(DataFormat::Int8) || pack_src_format == static_cast<DataFormatType>(DataFormat::UInt8);

    dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_32b_data = is_32b_format || is_fp32_dest_acc_en;
    dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_int8     = !(is_fp32_dest_acc_en || is_32b_format) && is_int8_format;

    if (pack_dst_format == static_cast<DataFormatType>(DataFormat::UInt8))
    {
        dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_unsigned = 1;
    }
    // Round to 10 bit mantissa from fp32 dest
    if (is_fp32_dest_acc_en && (pack_src_format == static_cast<DataFormatType>(DataFormat::Float16)))
    {
        dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Round_10b_mant = 1;
    }
    cfg_reg_rmw_tensix<
        PCK_DEST_RD_CTRL_Read_32b_data_ADDR32,
        PCK_DEST_RD_CTRL_Read_32b_data_SHAMT,
        PCK_DEST_RD_CTRL_Read_32b_data_MASK | PCK_DEST_RD_CTRL_Read_unsigned_MASK | PCK_DEST_RD_CTRL_Round_10b_mant_MASK>(dest_rd_ctrl.val);

    if (IS_BFP_FORMAT(pack_output_dst_format))
    {
        TTI_WRCFG(p_gpr_pack::EXP0_SEC_SIZE_BFP, p_cfg::WRCFG_32b, THCON_SEC0_REG1_Row_start_section_size_ADDR32);
    }
    else if (
        (pack_output_dst_format == static_cast<DataFormatType>(DataFormat::Lf8)) || (pack_output_dst_format == static_cast<DataFormatType>(DataFormat::Int8)))
    {
        TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, THCON_SEC0_REG1_Row_start_section_size_ADDR32);
    }

    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_pack::TILE_HEADER));

    // Workaround for HW bug: tenstorrent/budabackend#1394
    if constexpr (is_fp32_dest_acc_en)
    {
        uint exp_threshold_en  = 0;
        uint exp_threshold_val = 0;
        if (IS_BFP_A_FORMAT(pack_output_dst_format))
        {
            exp_threshold_en  = 1;
            exp_threshold_val = 113;
        }
        // EXP threshold is updated in the config word 3 which has a bit programmed by the unpacker as well
        constexpr uint exp_threshold_rmw_mask = THCON_SEC0_REG1_Exp_threshold_en_MASK | THCON_SEC0_REG1_Exp_threshold_MASK;
        uint exp_threshold_rmw_data = (exp_threshold_val << THCON_SEC0_REG1_Exp_threshold_SHAMT) | (exp_threshold_en << THCON_SEC0_REG1_Exp_threshold_en_SHAMT);
        cfg_reg_rmw_tensix<THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 3, 0, exp_threshold_rmw_mask>(exp_threshold_rmw_data);
    }

    // Flush packer pipeline before strides gasket alu format change
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG2_Dstacc_RMW>(pack_output_src_format);

    // Set packer strides
    set_packer_strides(pack_output_src_format, pack_output_dst_format, tile_c_dim);
}

template <bool is_fp32_dest_acc_en, bool untilize = false, bool tilize = false>
inline void configure_pack(
    const uint pack_src_format,
    const uint pack_dst_format,
    const uint tile_size,
    const uint face_r_dim   = FACE_R_DIM,
    const uint tile_c_dim   = TILE_C_DIM,
    const uint num_faces    = 4,
    const bool partial_face = false,
    const bool narrow_tile  = false,
    const uint relu_config  = 0)
{
    // Get pointer to registers for current state ID
    volatile uint* cfg = get_cfg_pointer();

    const uint pack_output_src_format = (uint)pack_src_format & 0xF;
    const uint pack_output_dst_format = (uint)pack_dst_format & 0xF;

    set_packer_strides<untilize, tilize>(pack_src_format, pack_dst_format, tile_c_dim);

    t6_mutex_acquire(mutex::REG_RMW);

    // Set Fp8 E4M3 mode for packer
    if ((pack_dst_format & 0x1F) == static_cast<DataFormatType>(DataFormat::Fp8_e4m3))
    {
        cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pac_LF8_4b_exp_RMW>(1);
    }

    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG2_Dstacc_RMW>(pack_output_src_format);

    t6_mutex_release(mutex::REG_RMW);

    set_packer_config<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, num_faces, partial_face);

    // PACK_COUNTERS_SEC0_pack_per_xy_plane = cfg_reg_array[3][0 +: 8];
    // PACK_COUNTERS_SEC0_pack_reads_per_xy_plane = cfg_reg_array[3][8 +: 8];
    // PACK_COUNTERS_SEC0_pack_xys_per_tile = cfg_reg_array[3][16 +: 7];
    // PACK_COUNTERS_SEC0_pack_yz_transposed = cfg_reg_array[3][23 +: 1];
    pack_counters_u pack_counters;
    pack_counters.val                       = 0;
    pack_counters.f.pack_reads_per_xy_plane = face_r_dim; // Number of reads per face
                                                          // Used for resetting tile posistion generator for edge masks
    for (uint i = 0; i < 4; i++)
    {
        cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 + i] = pack_counters.val; // disable auto last generation
    }

    pck_edge_offset_u pck_edge_offset;
    pck_edge_offset.val    = 0;
    pck_edge_offset.f.mask = 0xffff;

    cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32]                = pck_edge_offset.val;
    cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = 0x0; // All packers use row set mapping 0, edge offset 0 mask

    regfile[p_gpr_pack::TILE_HEADER]     = tile_size;
    regfile[p_gpr_pack::TILE_HEADER + 1] = 0;
    regfile[p_gpr_pack::TILE_HEADER + 2] = 0;
    regfile[p_gpr_pack::TILE_HEADER + 3] = 0;
    sync_regfile_write(p_gpr_pack::TILE_HEADER + 3);

    relu_config_u hw_relu_config;
    // Config RELU
    uint32_t current_relu_val = reg_read((uint)&cfg[STACC_RELU_ApplyRelu_ADDR32]);
    hw_relu_config.val[0]     = current_relu_val;

    hw_relu_config.r.STACC_RELU_ApplyRelu     = relu_config & 0xffff;
    hw_relu_config.r.STACC_RELU_ReluThreshold = (relu_config >> 16) & 0xffff;

    cfg[STACC_RELU_ApplyRelu_ADDR32] = hw_relu_config.val[0];

    // In Blackhole, x_start/x_end must be within 1 row size (i.e. from 0 to 15)
    TT_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
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
        TTI_WRCFG(p_gpr_pack::DEST_OFFSET_LO, p_cfg::WRCFG_128b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
    }
    else
    {
        TT_WRCFG(get_packer_dest_offset_index(), p_cfg::WRCFG_128b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
    }
    TTI_DMANOP;
    TTI_DMANOP;
}

// Program packer destination addresses from GPRs
inline void program_packer_destination(uint32_t addr)
{
    /*
       The GPR OUTPUT_ADDR is only used by the packer mop when writing tile headers.
       Since we do not write tile headers in tt-metal, we do not need to wait for
       packer to finish or say put a stallwait at this point.
       We just need to make sure we wait before issuing the WRCFG.
    */
    uint32_t new_l1_addr = (1 << 31) | addr;
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON | p_stall::PACK);
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);

    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
}

// RT: If multiple contexts are used, for issue #https://github.com/tenstorrent/tt-llk-bh/issues/20
// then this function needs to be re-written
template <uint32_t block_ct_dim, uint32_t full_ct_dim, bool diagonal = false>
inline void program_packer_untilized_destination(const uint32_t addr, const uint32_t pack_dst_format)
{
    // const uint32_t block_size = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * TILE_C_DIM * (TILE_R_DIM/4));
    // constexpr uint32_t offset0 = 0;
    // const uint32_t offset1 = (1*block_size)/16;
    // const uint32_t offset2 = (2*block_size)/16;
    // const uint32_t offset3 = (3*block_size)/16;

    // TT_SETDMAREG(0, LOWER_HALFWORD(addr+offset0), 0, LO_16(p_gpr_pack::OUTPUT_ADDR+0));
    // TT_SETDMAREG(0, UPPER_HALFWORD(addr+offset0), 0, HI_16(p_gpr_pack::OUTPUT_ADDR+0));
    // TT_SETDMAREG(0, LOWER_HALFWORD(addr+offset1), 0, LO_16(p_gpr_pack::OUTPUT_ADDR+1));
    // TT_SETDMAREG(0, UPPER_HALFWORD(addr+offset1), 0, HI_16(p_gpr_pack::OUTPUT_ADDR+1));
    // TT_SETDMAREG(0, LOWER_HALFWORD(addr+offset2), 0, LO_16(p_gpr_pack::OUTPUT_ADDR+2));
    // TT_SETDMAREG(0, UPPER_HALFWORD(addr+offset2), 0, HI_16(p_gpr_pack::OUTPUT_ADDR+2));
    // TT_SETDMAREG(0, LOWER_HALFWORD(addr+offset3), 0, LO_16(p_gpr_pack::OUTPUT_ADDR+3));
    // TT_SETDMAREG(0, UPPER_HALFWORD(addr+offset3), 0, HI_16(p_gpr_pack::OUTPUT_ADDR+3));
    // TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);

    // TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
    // TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+1, 0, THCON_SEC0_REG8_L1_Dest_addr_ADDR32);
    // TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+2, 0, THCON_SEC1_REG1_L1_Dest_addr_ADDR32);
    // TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+3, 0, THCON_SEC1_REG8_L1_Dest_addr_ADDR32);
    // TTI_NOP; TTI_NOP;
}

inline void program_packer_dest_offset_registers(uint32_t dest_tile_offset)
{
    TT_SETDMAREG(0, LOWER_HALFWORD(dest_tile_offset), 0, LO_16(p_gpr_pack::TEMP_TILE_OFFSET));
    TT_SETDMAREG(0, UPPER_HALFWORD(dest_tile_offset), 0, HI_16(p_gpr_pack::TEMP_TILE_OFFSET));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON | p_stall::PACK);
    TTI_WRCFG(p_gpr_pack::TEMP_TILE_OFFSET, p_cfg::WRCFG_32b, PCK0_ADDR_BASE_REG_0_Base_ADDR32);
    TTI_DMANOP;
    TTI_DMANOP;
}

inline void reconfigure_packer_l1_acc(const std::uint32_t pack_l1_acc)
{
    // assumes all configured packers have these fields as common values
    //  pack_config_u pack_config;
    //  pack_config.val[3] = 0;
    //  pack_config.f.pack_l1_acc_disable_pack_zero_flag = pack_l1_acc ? (0b11) : (0b00);

    // TT_SETDMAREG(0, pack_config.val[3] & 0xffff, 0, LO_16(p_gpr_pack::TMP0));
    // TT_SETDMAREG(0, (pack_config.val[3] >> 16) & 0xffff, 0, HI_16(p_gpr_pack::TMP0));
    // TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG1_Pack_L1_Acc_ADDR32);
    // TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG8_Pack_L1_Acc_ADDR32);
    // TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, THCON_SEC1_REG1_Pack_L1_Acc_ADDR32);
    // TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, THCON_SEC1_REG8_Pack_L1_Acc_ADDR32);
    // TTI_DMANOP;TTI_DMANOP;

    // TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::TRISC_CFG);

    const uint32_t pack_l1_acc_disable_pack_zero_flag = pack_l1_acc ? (0b11) : (0b00);

    cfg_reg_rmw_tensix<
        THCON_SEC0_REG1_Pack_L1_Acc_ADDR32,
        THCON_SEC0_REG1_Pack_L1_Acc_SHAMT,
        THCON_SEC0_REG1_Disable_pack_zero_flags_MASK | THCON_SEC0_REG1_Pack_L1_Acc_MASK>(pack_l1_acc_disable_pack_zero_flag);
}

// Write tile header to l1
inline void write_tile_header()
{
    TTI_STOREIND(1, 0, p_ind::LD_16B, LO_16(0), p_ind::INC_NONE, p_gpr_pack::TILE_HEADER, p_gpr_pack::OUTPUT_ADDR);
}

// READERS FOR CONFIG STRUCTS

inline pack_config_t read_pack_config_helper(uint32_t reg_addr, const volatile uint tt_reg_ptr* cfg)
{
    pack_config_u config = {.val = 0};

    config.val[0] = cfg[reg_addr];
    config.val[1] = cfg[reg_addr + 1];
    config.val[2] = cfg[reg_addr + 2];

    return config.f;
}

inline std::array<pack_config_t, NUM_PACKERS> read_pack_config()
{
    std::array<pack_config_t, NUM_PACKERS> config_vec;

    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    config_vec[0] = read_pack_config_helper(THCON_SEC0_REG1_Row_start_section_size_ADDR32, cfg);

    return config_vec;
}

inline relu_config_t read_relu_config()
{
    relu_config_u config;

    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();
    config.val[0]                 = cfg[ALU_ACC_CTRL_Zero_Flag_disabled_src_ADDR32];

    return config.r;
}

inline dest_rd_ctrl_t read_dest_rd_ctrl()
{
    dest_rd_ctrl_u dest;

    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    dest.val = cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32];

    return dest.f;
}

inline pck_edge_offset_t read_pack_edge_offset_helper(uint32_t reg_addr, const volatile uint tt_reg_ptr* cfg)
{
    pck_edge_offset_u edge = {.val = 0};

    edge.val = cfg[reg_addr];

    return edge.f;
}

inline std::array<pck_edge_offset_t, NUM_PACKERS> read_pack_edge_offset()
{
    std::array<pck_edge_offset_t, NUM_PACKERS> edge_vec;

    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    edge_vec[0] = read_pack_edge_offset_helper(PCK_EDGE_OFFSET_SEC0_mask_ADDR32, cfg);

    return edge_vec;
}

inline pack_counters_t read_pack_counters_helper(uint32_t reg_addr, const volatile uint tt_reg_ptr* cfg)
{
    pack_counters_u counters = {.val = 0};
    counters.val             = cfg[reg_addr];

    return counters.f;
}

inline std::array<pack_counters_t, NUM_PACKERS> read_pack_counters()
{
    std::array<pack_counters_t, NUM_PACKERS> config_vec;

    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    config_vec[0] = read_pack_counters_helper(PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32, cfg);

    return config_vec;
}
} // namespace ckernel::packer
