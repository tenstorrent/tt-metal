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

constexpr uint32_t PACK_CNT    = 4;
constexpr uint32_t NUM_PACKERS = 4; // Number of packers

constexpr uint PACK_SEL(const uint pack_count)
{
    return (pack_count == 1) ? 0x1 : (pack_count == 2) ? 0x3 : (pack_count == 4) ? 0xF : 0x0;
}

constexpr uint replay_buf_offset = 16; // split replay buffer usage between fpu/sfpu
                                       // fist 16 for sfpu, next 16 for fpu

// Pack config
typedef struct
{
    // word 0
    uint32_t row_ptr_section_size : 16;
    uint32_t exp_section_size     : 16;
    // word 1
    uint32_t l1_dest_addr : 32;
    // word 2
    uint32_t uncompress              : 1;
    uint32_t add_l1_dest_addr_offset : 1;
    uint32_t reserved_0              : 2;
    uint32_t out_data_format         : 4;
    uint32_t in_data_format          : 4;
    uint32_t reserved_1              : 4;
    uint32_t src_if_sel              : 1;
    uint32_t pack_per_xy_plane       : 7;
    uint32_t l1_src_addr             : 8;
    // word 3
    uint32_t downsample_mask                    : 16;
    uint32_t downsample_shift_count             : 3;
    uint32_t read_mode                          : 1;
    uint32_t exp_threshold_en                   : 1;
    uint32_t pack_l1_acc_disable_pack_zero_flag : 2;
    uint32_t reserved_2                         : 1;
    uint32_t exp_threshold                      : 8;

} pack_config_t;

static_assert(sizeof(pack_config_t) == (sizeof(uint32_t) * 4));

typedef union
{
    uint32_t val[4];
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

inline void set_packer_strides(const uint pack_src_format, const uint pack_dst_format)
{
    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    uint x_stride = (uint)(pack_src_format & 0x3) == static_cast<DataFormatType>(DataFormat::Float32)   ? 4
                    : (uint)(pack_src_format & 0x3) == static_cast<DataFormatType>(DataFormat::Float16) ? 2
                                                                                                        : 1;
    uint y_stride = FACE_R_DIM * x_stride;
    uint z_stride = PACK_CNT * FACE_C_DIM * y_stride;
    uint w_stride = z_stride;

    TT_SETDMAREG(0, LOWER_HALFWORD((x_stride << PCK0_ADDR_CTRL_XY_REG_0_Xstride_SHAMT)), 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD((y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, HI_16(p_gpr_pack::TMP0));
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TT_SETDMAREG(0, LOWER_HALFWORD((w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT)), 0, LO_16(p_gpr_pack::TMP0)); // z-stride not used!
    TT_SETDMAREG(0, UPPER_HALFWORD((w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT)), 0, HI_16(p_gpr_pack::TMP0));
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
    TTI_NOP;
    TTI_NOP;
}

template <bool is_fp32_dest_acc_en>
inline void set_packer_config(const uint pack_src_format, const uint pack_dst_format, const uint num_faces = 4, const bool partial_face = false)
{
    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    // Set packer config
    pack_config_u config;
    for (uint i = 0; i < 4; i++)
    {
        config.val[i] = 0;
    }

    config.f.exp_section_size =
        ((pack_dst_format == static_cast<DataFormatType>(DataFormat::Lf8)) || ((pack_dst_format & 0xF) == static_cast<DataFormatType>(DataFormat::Int8)))
            ? 0
            : (partial_face ? 1 : num_faces); // set to num_faces as exp section size is not used for non-bfp formats except for lf8/int8

    config.f.uncompress        = 1;
    config.f.out_data_format   = pack_dst_format;
    config.f.in_data_format    = pack_src_format;
    config.f.pack_per_xy_plane = 1;

    // Workaround for bug in HW: tenstorrent/budabackend#1394
    if constexpr (is_fp32_dest_acc_en)
    {
        if (IS_A_FORMAT(pack_dst_format))
        {
            config.f.exp_threshold_en = 1;
            config.f.exp_threshold    = 113;
        }
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
    // for (uint i=0; i<4; i++) cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32+i]=config.val[i];
    // for (uint i=0; i<4; i++) cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32+i]=config.val[i];
    // for (uint i=0; i<4; i++) cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32+i]=config.val[i];
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
    cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
    cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 0] = config.val[0];
    cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 0] = config.val[0];
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2] = config.val[2];
    cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 2] = config.val[2];
    cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 2] = config.val[2];
    cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 2] = config.val[2];
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 3] = config.val[3];
    cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 3] = config.val[3];
    cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 3] = config.val[3];
    cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 3] = config.val[3];

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

    if (IS_BFP_FORMAT(pack_dst_format))
    {
        // Override exp section size for packers 1,2,3
        // Tile header + exp size + datum size
        if ((uint)(pack_dst_format & 0x1F) == static_cast<DataFormatType>(DataFormat::Bfp8) ||
            (uint)(pack_dst_format & 0x1F) == static_cast<DataFormatType>(DataFormat::Bfp8_b))
        {
            config.f.exp_section_size                              = 1 + ((num_faces > 2) ? 2 : 0) + 16;
            cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 0] = config.val[0];
            config.f.exp_section_size                              = 1 + 1 + 32;
            cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
            config.f.exp_section_size                              = 1 + 0 + 48;
            cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 0] = config.val[0];
        }
        else if (
            (uint)(pack_dst_format & 0x1F) == static_cast<DataFormatType>(DataFormat::Bfp4) ||
            (uint)(pack_dst_format & 0x1F) == static_cast<DataFormatType>(DataFormat::Bfp4_b))
        {
            config.f.exp_section_size                              = 1 + ((num_faces > 2) ? 2 : 0) + 8;
            cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 0] = config.val[0];
            config.f.exp_section_size                              = 1 + 1 + 16;
            cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
            config.f.exp_section_size                              = 1 + 0 + 24;
            cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 0] = config.val[0];
        }
        else if (
            (uint)(pack_dst_format & 0x1F) == static_cast<DataFormatType>(DataFormat::Bfp2) ||
            (uint)(pack_dst_format & 0x1F) == static_cast<DataFormatType>(DataFormat::Bfp2_b))
        {
            config.f.exp_section_size                              = 1 + ((num_faces > 2) ? 2 : 0) + 4;
            cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 0] = config.val[0];
            config.f.exp_section_size                              = 1 + 1 + 8;
            cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
            config.f.exp_section_size                              = 1 + 0 + 12;
            cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 0] = config.val[0];
        }
        else
        {
            FWASSERT("Other data formats not supported", false);
        }
    }

    // Save to GPR for quick data format reconfig
    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP]  = (partial_face ? 1 : num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    regfile[p_gpr_pack::EXP1_SEC_SIZE_BFP8] = (1 + ((num_faces > 2) ? 2 : 0) + 16) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    regfile[p_gpr_pack::EXP2_SEC_SIZE_BFP8] = (1 + 1 + 32) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    regfile[p_gpr_pack::EXP3_SEC_SIZE_BFP8] = (1 + 0 + 48) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    regfile[p_gpr_pack::EXP1_SEC_SIZE_BFP4] = (1 + ((num_faces > 2) ? 2 : 0) + 8) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    regfile[p_gpr_pack::EXP2_SEC_SIZE_BFP4] = (1 + 1 + 16) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    regfile[p_gpr_pack::EXP3_SEC_SIZE_BFP4] = (1 + 0 + 24) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    regfile[p_gpr_pack::EXP1_SEC_SIZE_BFP2] = (1 + ((num_faces > 2) ? 2 : 0) + 4) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    regfile[p_gpr_pack::EXP2_SEC_SIZE_BFP2] = (1 + 1 + 8) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    regfile[p_gpr_pack::EXP3_SEC_SIZE_BFP2] = (1 + 0 + 12) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    sync_regfile_write(p_gpr_pack::EXP3_SEC_SIZE_BFP2);
}

inline void set_packer_l1_offset(const uint pack_dst_format, const uint face_r_dim = FACE_R_DIM)
{
    const uint face_dim = face_r_dim * FACE_C_DIM;

    uint32_t l1_offset_1 = IS_BFP_FORMAT(pack_dst_format)
                               ? 1
                               : (((uint8_t)(pack_dst_format & 0x3) == static_cast<DataFormatType>(DataFormat::Float32))   ? (face_dim / 16) * 4
                                  : ((uint8_t)(pack_dst_format & 0x3) == static_cast<DataFormatType>(DataFormat::Float16)) ? (face_dim / 16) * 2
                                                                                                                           : (face_dim / 16));
    uint32_t l1_offset_2 = 2 * l1_offset_1;
    uint32_t l1_offset_3 = 3 * l1_offset_1;

    // HW automatically offsets packers base address by tile header size
    // with new L1 addressing mode, the effective address for pack1/2/3
    // will be pack[i] += pack[0], which leads to double counting of tile header
    // subtract by this amount when programming the offset
    constexpr uint32_t PACK_TILE_HEADER_OFFSET = 1; // in 16B
    l1_offset_1 -= PACK_TILE_HEADER_OFFSET;
    l1_offset_2 -= PACK_TILE_HEADER_OFFSET;
    l1_offset_3 -= PACK_TILE_HEADER_OFFSET;
    TT_SETDMAREG(0, LOWER_HALFWORD(l1_offset_1), 0, LO_16(p_gpr_pack::TMP_LO));
    TTI_REG2FLOP(2, 0, 0, 0, THCON_SEC0_REG8_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);
    TT_SETDMAREG(0, LOWER_HALFWORD(l1_offset_2), 0, LO_16(p_gpr_pack::TMP_LO));
    TTI_REG2FLOP(2, 0, 0, 0, THCON_SEC1_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);
    TT_SETDMAREG(0, LOWER_HALFWORD(l1_offset_3), 0, LO_16(p_gpr_pack::TMP_LO));
    TTI_REG2FLOP(2, 0, 0, 0, THCON_SEC1_REG8_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);
}

template <bool is_fp32_dest_acc_en>
inline void reconfig_packer_data_format(const uint pack_src_format, const uint pack_dst_format, const uint tile_size, const uint face_r_dim = FACE_R_DIM)
{
    // Get pointer to registers for current state ID
    volatile uint* cfg = get_cfg_pointer();

    // Configure packers
    pack_config_u config;
    config.val[2] = 0; // Only need to modify word[2][15:0]

    config.f.uncompress      = 1;
    config.f.out_data_format = pack_dst_format;
    config.f.in_data_format  = pack_src_format;
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[2]), 0, LO_16(p_gpr_pack::TMP_LO));
    TTI_REG2FLOP(2, 0, 0, 0, THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO); // 16-bit write
    TTI_REG2FLOP(2, 0, 0, 0, THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 2 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);
    TTI_REG2FLOP(2, 0, 0, 0, THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 2 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);
    TTI_REG2FLOP(2, 0, 0, 0, THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 2 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);

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

    if (IS_BFP_FORMAT(pack_dst_format))
    {
        // Override exp section size for packers 1,2,3
        // Tile header + exp size + datum size
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP0_SEC_SIZE_BFP);
        if ((pack_dst_format & 0x1F) == static_cast<DataFormatType>(DataFormat::Bfp8) ||
            (pack_dst_format & 0x1F) == static_cast<DataFormatType>(DataFormat::Bfp8_b))
        {
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP1_SEC_SIZE_BFP8);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP2_SEC_SIZE_BFP8);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP3_SEC_SIZE_BFP8);
        }
        else if (
            (pack_dst_format & 0x1F) == static_cast<DataFormatType>(DataFormat::Bfp4) ||
            (pack_dst_format & 0x1F) == static_cast<DataFormatType>(DataFormat::Bfp4_b))
        {
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP1_SEC_SIZE_BFP4);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP2_SEC_SIZE_BFP4);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP3_SEC_SIZE_BFP4);
        }
        else if (
            (pack_dst_format & 0x1F) == static_cast<DataFormatType>(DataFormat::Bfp2) ||
            (pack_dst_format & 0x1F) == static_cast<DataFormatType>(DataFormat::Bfp2_b))
        {
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP1_SEC_SIZE_BFP2);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP2_SEC_SIZE_BFP2);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP3_SEC_SIZE_BFP2);
        }
        else
        {
            FWASSERT("Other data formats not supported", false);
        }
    }
    else if ((pack_dst_format == static_cast<DataFormatType>(DataFormat::Lf8)) || ((pack_dst_format & 0xF) == static_cast<DataFormatType>(DataFormat::Int8)))
    {
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
    }

    // Set l1 address offset
    set_packer_l1_offset(pack_dst_format, face_r_dim);

    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_pack::TILE_HEADER));

    // Workaround for HW bug: tenstorrent/budabackend#1394
    if constexpr (is_fp32_dest_acc_en)
    {
        if (IS_BFP_A_FORMAT(pack_dst_format))
        {
            config.val[3]             = 0; // Only need to modify word[2][15:0]
            config.f.exp_threshold_en = 1;
            config.f.exp_threshold    = 113;
            TT_SETDMAREG(0, UPPER_HALFWORD(config.val[3]), 0, HI_16(p_gpr_pack::TMP_HI));
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 3 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_HI);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 3 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_HI);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 3 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_HI);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 3 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_HI);
        }
        else
        {
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 3 - THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 3 - THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 3 - THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 3 - THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
        }
    }

    // Flush packer pipeline before strides gasket alu format change
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG2_Dstacc_RMW>(pack_src_format);

    tensix_sync(); // FIXME: why stallwait on cfg write doesn't work!

    // Set packer strides
    set_packer_strides(pack_src_format, pack_dst_format);
}

template <bool is_fp32_dest_acc_en, bool untilize>
inline void configure_pack(
    const uint pack_src_format,
    const uint pack_dst_format,
    const uint tile_size,
    const uint face_r_dim   = FACE_R_DIM,
    const uint num_faces    = 4,
    const bool partial_face = false,
    const bool narrow_tile  = false,
    const uint relu_config  = 0)
{
    // Get pointer to registers for current state ID
    volatile uint* cfg = get_cfg_pointer();

    if (pack_src_format != pack_dst_format)
    {
        TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK);
        tensix_sync();
    }

    set_packer_strides(pack_src_format, pack_dst_format);

    t6_mutex_acquire(mutex::REG_RMW);

    const uint alu_dst_format = pack_src_format;

    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG2_Dstacc_RMW>(alu_dst_format);

    t6_mutex_release(mutex::REG_RMW);

    set_packer_config<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, num_faces, partial_face);

    set_packer_l1_offset(pack_dst_format, face_r_dim);

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

    const uint face_dim = face_r_dim * FACE_C_DIM;

    // To untilize narrow tile (32x16) we just pack 2 faces back to back
    // Number of datums to pack per row
    const uint pack_x_dim = (narrow_tile || !untilize) ? face_dim : FACE_R_DIM;

    TT_SETADCXX(p_setadc::PAC, pack_x_dim - 1, 0x0);
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
template <PackSelMask PackSel = PACK_ALL>
inline void program_packer_destination(uint32_t addr)
{
    uint32_t new_l1_addr = (1 << 31) | addr;
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));

    // TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR);

    TTI_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 0); // pack flush

    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
}

template <uint32_t block_ct_dim, uint32_t full_ct_dim, bool diagonal = false, uint32_t row_num_datums = TILE_C_DIM>
inline void program_packer_untilized_destination(const uint32_t addr, const uint32_t pack_dst_format)
{
    if constexpr (diagonal)
    {
        const uint32_t block_size  = SCALE_DATUM_SIZE(pack_dst_format, FACE_C_DIM);
        constexpr uint32_t offset0 = 0;
        const uint32_t offset1     = (1 * block_size) / 16;
        // const uint32_t offset2 = (2*block_size)/16;
        // const uint32_t offset3 = (3*block_size)/16;

        TT_SETDMAREG(0, LOWER_HALFWORD(addr + offset0), 0, LO_16(p_gpr_pack::OUTPUT_ADDR + 0));
        TT_SETDMAREG(0, UPPER_HALFWORD(addr + offset0), 0, HI_16(p_gpr_pack::OUTPUT_ADDR + 0));
        TT_SETDMAREG(0, LOWER_HALFWORD(addr + offset1), 0, LO_16(p_gpr_pack::OUTPUT_ADDR + 1));
        TT_SETDMAREG(0, UPPER_HALFWORD(addr + offset1), 0, HI_16(p_gpr_pack::OUTPUT_ADDR + 1));
        // TT_SETDMAREG(0, LOWER_HALFWORD(addr+offset2), 0, LO_16(p_gpr_pack::OUTPUT_ADDR+2));
        // TT_SETDMAREG(0, UPPER_HALFWORD(addr+offset2), 0, HI_16(p_gpr_pack::OUTPUT_ADDR+2));
        // TT_SETDMAREG(0, LOWER_HALFWORD(addr+offset3), 0, LO_16(p_gpr_pack::OUTPUT_ADDR+3));
        // TT_SETDMAREG(0, UPPER_HALFWORD(addr+offset3), 0, HI_16(p_gpr_pack::OUTPUT_ADDR+3));

        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR);
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR + 1);
        // TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG1_L1_Dest_addr_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR+2);
        // TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG8_L1_Dest_addr_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR+3);

        TTI_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 0); // pack flush
    }
    else
    {
        // Each packer packs 8 rows of full_ct_dim*TILE_C_DIM datums
        const uint32_t block_size  = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * TILE_C_DIM * (TILE_R_DIM / 4));
        constexpr uint32_t offset0 = 0;
        const uint32_t offset1     = (1 * row_num_datums * block_size) / 16 / TILE_C_DIM;
        const uint32_t offset2     = (2 * row_num_datums * block_size) / 16 / TILE_C_DIM;
        const uint32_t offset3     = (3 * row_num_datums * block_size) / 16 / TILE_C_DIM;

        TT_SETDMAREG(0, LOWER_HALFWORD(addr + offset0), 0, LO_16(p_gpr_pack::OUTPUT_ADDR + 0));
        TT_SETDMAREG(0, UPPER_HALFWORD(addr + offset0), 0, HI_16(p_gpr_pack::OUTPUT_ADDR + 0));
        TT_SETDMAREG(0, LOWER_HALFWORD(addr + offset1), 0, LO_16(p_gpr_pack::OUTPUT_ADDR + 1));
        TT_SETDMAREG(0, UPPER_HALFWORD(addr + offset1), 0, HI_16(p_gpr_pack::OUTPUT_ADDR + 1));
        TT_SETDMAREG(0, LOWER_HALFWORD(addr + offset2), 0, LO_16(p_gpr_pack::OUTPUT_ADDR + 2));
        TT_SETDMAREG(0, UPPER_HALFWORD(addr + offset2), 0, HI_16(p_gpr_pack::OUTPUT_ADDR + 2));
        TT_SETDMAREG(0, LOWER_HALFWORD(addr + offset3), 0, LO_16(p_gpr_pack::OUTPUT_ADDR + 3));
        TT_SETDMAREG(0, UPPER_HALFWORD(addr + offset3), 0, HI_16(p_gpr_pack::OUTPUT_ADDR + 3));

        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR);
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR + 1);
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR + 2);
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG8_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR + 3);

        TTI_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 0); // pack flush
    }
}

inline void program_packer_dest_offset_registers(uint32_t dest_tile_offset)
{
    TT_SETDMAREG(0, LOWER_HALFWORD(dest_tile_offset), 0, LO_16(p_gpr_pack::TEMP_TILE_OFFSET));
    TT_SETDMAREG(0, UPPER_HALFWORD(dest_tile_offset), 0, HI_16(p_gpr_pack::TEMP_TILE_OFFSET));
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
    cfg_reg_rmw_tensix<
        THCON_SEC0_REG8_Pack_L1_Acc_ADDR32,
        THCON_SEC0_REG8_Pack_L1_Acc_SHAMT,
        THCON_SEC0_REG8_Disable_pack_zero_flags_MASK | THCON_SEC0_REG8_Pack_L1_Acc_MASK>(pack_l1_acc_disable_pack_zero_flag);
    cfg_reg_rmw_tensix<
        THCON_SEC1_REG1_Pack_L1_Acc_ADDR32,
        THCON_SEC1_REG1_Pack_L1_Acc_SHAMT,
        THCON_SEC1_REG1_Disable_pack_zero_flags_MASK | THCON_SEC1_REG1_Pack_L1_Acc_MASK>(pack_l1_acc_disable_pack_zero_flag);
    cfg_reg_rmw_tensix<
        THCON_SEC1_REG8_Pack_L1_Acc_ADDR32,
        THCON_SEC1_REG8_Pack_L1_Acc_SHAMT,
        THCON_SEC1_REG8_Disable_pack_zero_flags_MASK | THCON_SEC1_REG8_Pack_L1_Acc_MASK>(pack_l1_acc_disable_pack_zero_flag);
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
    config.val[3] = cfg[reg_addr + 3];

    return config.f;
}

inline std::array<pack_config_t, NUM_PACKERS> read_pack_config()
{
    std::array<pack_config_t, NUM_PACKERS> config_vec;

    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    config_vec[0] = read_pack_config_helper(THCON_SEC0_REG1_Row_start_section_size_ADDR32, cfg);
    config_vec[1] = read_pack_config_helper(THCON_SEC0_REG8_Row_start_section_size_ADDR32, cfg);
    config_vec[2] = read_pack_config_helper(THCON_SEC1_REG1_Row_start_section_size_ADDR32, cfg);
    config_vec[3] = read_pack_config_helper(THCON_SEC1_REG8_Row_start_section_size_ADDR32, cfg);

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
    edge.val               = cfg[reg_addr];

    return edge.f;
}

inline std::array<pck_edge_offset_t, NUM_PACKERS> read_pack_edge_offset()
{
    std::array<pck_edge_offset_t, NUM_PACKERS> edge_vec;

    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    edge_vec[0] = read_pack_edge_offset_helper(PCK_EDGE_OFFSET_SEC0_mask_ADDR32, cfg);
    edge_vec[1] = read_pack_edge_offset_helper(PCK_EDGE_OFFSET_SEC1_mask_ADDR32, cfg);
    edge_vec[2] = read_pack_edge_offset_helper(PCK_EDGE_OFFSET_SEC2_mask_ADDR32, cfg);
    edge_vec[3] = read_pack_edge_offset_helper(PCK_EDGE_OFFSET_SEC3_mask_ADDR32, cfg);

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
    std::array<pack_counters_t, NUM_PACKERS> counters_vec;

    // Get pointer to registers for current state ID
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();

    counters_vec[0] = read_pack_counters_helper(PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32, cfg);
    counters_vec[1] = read_pack_counters_helper(PACK_COUNTERS_SEC1_pack_per_xy_plane_ADDR32, cfg);
    counters_vec[2] = read_pack_counters_helper(PACK_COUNTERS_SEC2_pack_per_xy_plane_ADDR32, cfg);
    counters_vec[3] = read_pack_counters_helper(PACK_COUNTERS_SEC3_pack_per_xy_plane_ADDR32, cfg);

    return counters_vec;
}

} // namespace ckernel::packer
