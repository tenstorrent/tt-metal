# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Blackhole unpack family (``tools.llk_state_audit.bh.families.unpack``).

The ``*`` snippet constants below are verbatim slices of the unpack scope: the folded tree
(78385cf + patches 01/02) for most files, and main (8a7beeeaede) for
``MAIN_LLK_UNPACK_A_TOPK_XL_COPY`` (its ternary ``upk0_reg`` cfg index). ``CFG_DEFINES`` and
``CKERNEL_GPR_MAP`` are the real definitions of the fields and GPRs the snippets use.
``STUBS`` are minimal stand-ins for headers the snippets include (not under test). Expected
values follow the C++ semantics of each snippet (struct bit layouts, GPR halves, branch
conditions), not a snapshot of the tool output.
"""

import tempfile
import unittest
from pathlib import Path

from tools.llk_state_audit.bh import effects_db
from tools.llk_state_audit.bh.families import unpack

LLK = "tt_metal/tt-llk/tt_llk_blackhole"
API = "tt_metal/hw/ckernels/blackhole/metal/llk_api"

CFG_DEFINES = r"""
// Registers for THREAD
#define SRCA_SET_Base_ADDR32 5
#define SRCA_SET_Base_SHAMT 0
#define SRCA_SET_Base_MASK 0x3
#define UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32 41
#define UNPACK_MISC_CFG_CfgContextOffset_0_SHAMT 0
#define UNPACK_MISC_CFG_CfgContextOffset_0_MASK 0xf
// Registers for ALU
#define ALU_ROUNDING_MODE_Fpu_srnd_en_ADDR32 1
#define ALU_ROUNDING_MODE_Fpu_srnd_en_SHAMT 0
#define ALU_ROUNDING_MODE_Fpu_srnd_en_MASK 0x1
#define ALU_ROUNDING_MODE_Gasket_srnd_en_ADDR32 1
#define ALU_ROUNDING_MODE_Gasket_srnd_en_SHAMT 1
#define ALU_ROUNDING_MODE_Gasket_srnd_en_MASK 0x2
#define ALU_ROUNDING_MODE_Packer_srnd_en_ADDR32 1
#define ALU_ROUNDING_MODE_Packer_srnd_en_SHAMT 2
#define ALU_ROUNDING_MODE_Packer_srnd_en_MASK 0x4
#define ALU_ROUNDING_MODE_Padding_ADDR32 1
#define ALU_ROUNDING_MODE_Padding_SHAMT 3
#define ALU_ROUNDING_MODE_Padding_MASK 0x1ff8
#define ALU_ROUNDING_MODE_GS_LF_ADDR32 1
#define ALU_ROUNDING_MODE_GS_LF_SHAMT 13
#define ALU_ROUNDING_MODE_GS_LF_MASK 0x2000
#define ALU_ROUNDING_MODE_Bfp8_HF_ADDR32 1
#define ALU_ROUNDING_MODE_Bfp8_HF_SHAMT 14
#define ALU_ROUNDING_MODE_Bfp8_HF_MASK 0x4000
#define ALU_FORMAT_SPEC_REG0_SrcAUnsigned_ADDR32 1
#define ALU_FORMAT_SPEC_REG0_SrcAUnsigned_SHAMT 15
#define ALU_FORMAT_SPEC_REG0_SrcAUnsigned_MASK 0x8000
#define ALU_FORMAT_SPEC_REG0_SrcBUnsigned_ADDR32 1
#define ALU_FORMAT_SPEC_REG0_SrcBUnsigned_SHAMT 16
#define ALU_FORMAT_SPEC_REG0_SrcBUnsigned_MASK 0x10000
#define ALU_FORMAT_SPEC_REG0_SrcA_ADDR32 1
#define ALU_FORMAT_SPEC_REG0_SrcA_SHAMT 17
#define ALU_FORMAT_SPEC_REG0_SrcA_MASK 0x1e0000
#define ALU_FORMAT_SPEC_REG1_SrcB_ADDR32 1
#define ALU_FORMAT_SPEC_REG1_SrcB_SHAMT 21
#define ALU_FORMAT_SPEC_REG1_SrcB_MASK 0x1e00000
#define ALU_FORMAT_SPEC_REG2_Dstacc_ADDR32 1
#define ALU_FORMAT_SPEC_REG2_Dstacc_SHAMT 25
#define ALU_FORMAT_SPEC_REG2_Dstacc_MASK 0x1e000000
#define ALU_ACC_CTRL_Fp32_enabled_ADDR32 1
#define ALU_ACC_CTRL_Fp32_enabled_SHAMT 29
#define ALU_ACC_CTRL_Fp32_enabled_MASK 0x20000000
#define ALU_ACC_CTRL_SFPU_Fp32_enabled_ADDR32 1
#define ALU_ACC_CTRL_SFPU_Fp32_enabled_SHAMT 30
#define ALU_ACC_CTRL_SFPU_Fp32_enabled_MASK 0x40000000
#define ALU_ACC_CTRL_INT8_math_enabled_ADDR32 1
#define ALU_ACC_CTRL_INT8_math_enabled_SHAMT 31
#define ALU_ACC_CTRL_INT8_math_enabled_MASK 0x80000000
#define ALU_ACC_CTRL_Zero_Flag_disabled_src_ADDR32 2
#define ALU_ACC_CTRL_Zero_Flag_disabled_src_SHAMT 0
#define ALU_ACC_CTRL_Zero_Flag_disabled_src_MASK 0x1
#define ALU_ACC_CTRL_Zero_Flag_disabled_dst_ADDR32 2
#define ALU_ACC_CTRL_Zero_Flag_disabled_dst_SHAMT 1
#define ALU_ACC_CTRL_Zero_Flag_disabled_dst_MASK 0x2
// Registers for UNPACK0
#define UNP0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32 44
#define UNP0_ADDR_CTRL_XY_REG_0_Xstride_SHAMT 0
#define UNP0_ADDR_CTRL_XY_REG_0_Xstride_MASK 0xffff
#define UNP0_ADDR_CTRL_XY_REG_0_Ystride_ADDR32 44
#define UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT 16
#define UNP0_ADDR_CTRL_XY_REG_0_Ystride_MASK 0xffff0000
#define UNP0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32 45
#define UNP0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT 0
#define UNP0_ADDR_CTRL_ZW_REG_0_Zstride_MASK 0xffff
#define UNP0_ADDR_CTRL_ZW_REG_0_Wstride_ADDR32 45
#define UNP0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT 16
#define UNP0_ADDR_CTRL_ZW_REG_0_Wstride_MASK 0xffff0000
#define UNP1_ADDR_CTRL_XY_REG_0_Xstride_ADDR32 46
#define UNP1_ADDR_CTRL_XY_REG_0_Xstride_SHAMT 0
#define UNP1_ADDR_CTRL_XY_REG_0_Xstride_MASK 0xffff
#define UNP1_ADDR_CTRL_XY_REG_0_Ystride_ADDR32 46
#define UNP1_ADDR_CTRL_XY_REG_0_Ystride_SHAMT 16
#define UNP1_ADDR_CTRL_XY_REG_0_Ystride_MASK 0xffff0000
#define UNP1_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32 47
#define UNP1_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT 0
#define UNP1_ADDR_CTRL_ZW_REG_0_Zstride_MASK 0xffff
#define UNP1_ADDR_CTRL_ZW_REG_0_Wstride_ADDR32 47
#define UNP1_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT 16
#define UNP1_ADDR_CTRL_ZW_REG_0_Wstride_MASK 0xffff0000
#define UNP0_FORCED_SHARED_EXP_shared_exp_ADDR32 50
#define UNP0_FORCED_SHARED_EXP_shared_exp_SHAMT 0
#define UNP0_FORCED_SHARED_EXP_shared_exp_MASK 0xff
#define UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_ADDR32 50
#define UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_SHAMT 8
#define UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_MASK 0x100
// Registers for UNPACK1
#define UNP0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32 56
#define UNP0_ADDR_CTRL_XY_REG_1_Xstride_SHAMT 0
#define UNP0_ADDR_CTRL_XY_REG_1_Xstride_MASK 0xffff
#define UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32 56
#define UNP0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT 16
#define UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK 0xffff0000
#define UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32 57
#define UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT 0
#define UNP0_ADDR_CTRL_ZW_REG_1_Zstride_MASK 0xffff
#define UNP0_ADDR_CTRL_ZW_REG_1_Wstride_ADDR32 57
#define UNP0_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT 16
#define UNP0_ADDR_CTRL_ZW_REG_1_Wstride_MASK 0xffff0000
#define UNP1_ADDR_CTRL_XY_REG_1_Xstride_ADDR32 58
#define UNP1_ADDR_CTRL_XY_REG_1_Xstride_SHAMT 0
#define UNP1_ADDR_CTRL_XY_REG_1_Xstride_MASK 0xffff
#define UNP1_ADDR_CTRL_XY_REG_1_Ystride_ADDR32 58
#define UNP1_ADDR_CTRL_XY_REG_1_Ystride_SHAMT 16
#define UNP1_ADDR_CTRL_XY_REG_1_Ystride_MASK 0xffff0000
#define UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32 59
#define UNP1_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT 0
#define UNP1_ADDR_CTRL_ZW_REG_1_Zstride_MASK 0xffff
#define UNP1_ADDR_CTRL_ZW_REG_1_Wstride_ADDR32 59
#define UNP1_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT 16
#define UNP1_ADDR_CTRL_ZW_REG_1_Wstride_MASK 0xffff0000
// Registers for THCON
#define THCON_SEC0_REG0_TileDescriptor_ADDR32 64
#define THCON_SEC0_REG0_TileDescriptor_SHAMT 0
#define THCON_SEC0_REG0_TileDescriptor_MASK 0xffffffffffffffffffffffffffffffff
#define THCON_SEC0_REG1_Unp_LF8_4b_exp_ADDR32 71
#define THCON_SEC0_REG1_Unp_LF8_4b_exp_SHAMT 22
#define THCON_SEC0_REG1_Unp_LF8_4b_exp_MASK 0x400000
#define THCON_SEC0_REG2_Out_data_format_ADDR32 72
#define THCON_SEC0_REG2_Out_data_format_SHAMT 0
#define THCON_SEC0_REG2_Out_data_format_MASK 0xf
#define THCON_SEC0_REG2_Throttle_mode_ADDR32 72
#define THCON_SEC0_REG2_Throttle_mode_SHAMT 4
#define THCON_SEC0_REG2_Throttle_mode_MASK 0x30
#define THCON_SEC0_REG2_Context_count_ADDR32 72
#define THCON_SEC0_REG2_Context_count_SHAMT 6
#define THCON_SEC0_REG2_Context_count_MASK 0xc0
#define THCON_SEC0_REG2_Haloize_mode_ADDR32 72
#define THCON_SEC0_REG2_Haloize_mode_SHAMT 8
#define THCON_SEC0_REG2_Haloize_mode_MASK 0x100
#define THCON_SEC0_REG2_Tileize_mode_ADDR32 72
#define THCON_SEC0_REG2_Tileize_mode_SHAMT 9
#define THCON_SEC0_REG2_Tileize_mode_MASK 0x200
#define THCON_SEC0_REG2_Unpack_Src_Reg_Set_Upd_ADDR32 72
#define THCON_SEC0_REG2_Unpack_Src_Reg_Set_Upd_SHAMT 10
#define THCON_SEC0_REG2_Unpack_Src_Reg_Set_Upd_MASK 0x400
#define THCON_SEC0_REG2_Unpack_If_Sel_ADDR32 72
#define THCON_SEC0_REG2_Unpack_If_Sel_SHAMT 11
#define THCON_SEC0_REG2_Unpack_If_Sel_MASK 0x800
#define THCON_SEC0_REG2_Upsample_rate_ADDR32 72
#define THCON_SEC0_REG2_Upsample_rate_SHAMT 12
#define THCON_SEC0_REG2_Upsample_rate_MASK 0x3000
#define THCON_SEC0_REG2_Ovrd_data_format_ADDR32 72
#define THCON_SEC0_REG2_Ovrd_data_format_SHAMT 14
#define THCON_SEC0_REG2_Ovrd_data_format_MASK 0x4000
#define THCON_SEC0_REG2_Upsample_and_interleave_ADDR32 72
#define THCON_SEC0_REG2_Upsample_and_interleave_SHAMT 15
#define THCON_SEC0_REG2_Upsample_and_interleave_MASK 0x8000
#define THCON_SEC0_REG2_Shift_amount_cntx0_ADDR32 72
#define THCON_SEC0_REG2_Shift_amount_cntx0_SHAMT 16
#define THCON_SEC0_REG2_Shift_amount_cntx0_MASK 0xf0000
#define THCON_SEC0_REG2_Shift_amount_cntx1_ADDR32 72
#define THCON_SEC0_REG2_Shift_amount_cntx1_SHAMT 20
#define THCON_SEC0_REG2_Shift_amount_cntx1_MASK 0xf00000
#define THCON_SEC0_REG2_Shift_amount_cntx2_ADDR32 72
#define THCON_SEC0_REG2_Shift_amount_cntx2_SHAMT 24
#define THCON_SEC0_REG2_Shift_amount_cntx2_MASK 0xf000000
#define THCON_SEC0_REG2_Shift_amount_cntx3_ADDR32 72
#define THCON_SEC0_REG2_Shift_amount_cntx3_SHAMT 28
#define THCON_SEC0_REG2_Shift_amount_cntx3_MASK 0xf0000000
#define THCON_SEC0_REG2_Disable_zero_compress_cntx0_ADDR32 73
#define THCON_SEC0_REG2_Disable_zero_compress_cntx0_SHAMT 0
#define THCON_SEC0_REG2_Disable_zero_compress_cntx0_MASK 0x1
#define THCON_SEC0_REG2_Disable_zero_compress_cntx1_ADDR32 73
#define THCON_SEC0_REG2_Disable_zero_compress_cntx1_SHAMT 1
#define THCON_SEC0_REG2_Disable_zero_compress_cntx1_MASK 0x2
#define THCON_SEC0_REG2_Disable_zero_compress_cntx2_ADDR32 73
#define THCON_SEC0_REG2_Disable_zero_compress_cntx2_SHAMT 2
#define THCON_SEC0_REG2_Disable_zero_compress_cntx2_MASK 0x4
#define THCON_SEC0_REG2_Disable_zero_compress_cntx3_ADDR32 73
#define THCON_SEC0_REG2_Disable_zero_compress_cntx3_SHAMT 3
#define THCON_SEC0_REG2_Disable_zero_compress_cntx3_MASK 0x8
#define THCON_SEC0_REG2_Unpack_if_sel_cntx0_ADDR32 73
#define THCON_SEC0_REG2_Unpack_if_sel_cntx0_SHAMT 4
#define THCON_SEC0_REG2_Unpack_if_sel_cntx0_MASK 0x10
#define THCON_SEC0_REG2_Unpack_if_sel_cntx1_ADDR32 73
#define THCON_SEC0_REG2_Unpack_if_sel_cntx1_SHAMT 5
#define THCON_SEC0_REG2_Unpack_if_sel_cntx1_MASK 0x20
#define THCON_SEC0_REG2_Unpack_if_sel_cntx2_ADDR32 73
#define THCON_SEC0_REG2_Unpack_if_sel_cntx2_SHAMT 6
#define THCON_SEC0_REG2_Unpack_if_sel_cntx2_MASK 0x40
#define THCON_SEC0_REG2_Unpack_if_sel_cntx3_ADDR32 73
#define THCON_SEC0_REG2_Unpack_if_sel_cntx3_SHAMT 7
#define THCON_SEC0_REG2_Unpack_if_sel_cntx3_MASK 0x80
#define THCON_SEC0_REG2_Force_shared_exp_ADDR32 73
#define THCON_SEC0_REG2_Force_shared_exp_SHAMT 8
#define THCON_SEC0_REG2_Force_shared_exp_MASK 0x100
#define THCON_SEC0_REG2_Context_count_non_log2_ADDR32 73
#define THCON_SEC0_REG2_Context_count_non_log2_SHAMT 9
#define THCON_SEC0_REG2_Context_count_non_log2_MASK 0xe00
#define THCON_SEC0_REG2_Context_count_non_log2_en_ADDR32 73
#define THCON_SEC0_REG2_Context_count_non_log2_en_SHAMT 12
#define THCON_SEC0_REG2_Context_count_non_log2_en_MASK 0x1000
#define THCON_SEC0_REG2_Disable_zero_compress_cntx4_ADDR32 73
#define THCON_SEC0_REG2_Disable_zero_compress_cntx4_SHAMT 16
#define THCON_SEC0_REG2_Disable_zero_compress_cntx4_MASK 0x10000
#define THCON_SEC0_REG2_Disable_zero_compress_cntx5_ADDR32 73
#define THCON_SEC0_REG2_Disable_zero_compress_cntx5_SHAMT 17
#define THCON_SEC0_REG2_Disable_zero_compress_cntx5_MASK 0x20000
#define THCON_SEC0_REG2_Disable_zero_compress_cntx6_ADDR32 73
#define THCON_SEC0_REG2_Disable_zero_compress_cntx6_SHAMT 18
#define THCON_SEC0_REG2_Disable_zero_compress_cntx6_MASK 0x40000
#define THCON_SEC0_REG2_Disable_zero_compress_cntx7_ADDR32 73
#define THCON_SEC0_REG2_Disable_zero_compress_cntx7_SHAMT 19
#define THCON_SEC0_REG2_Disable_zero_compress_cntx7_MASK 0x80000
#define THCON_SEC0_REG2_Unpack_if_sel_cntx4_ADDR32 73
#define THCON_SEC0_REG2_Unpack_if_sel_cntx4_SHAMT 20
#define THCON_SEC0_REG2_Unpack_if_sel_cntx4_MASK 0x100000
#define THCON_SEC0_REG2_Unpack_if_sel_cntx5_ADDR32 73
#define THCON_SEC0_REG2_Unpack_if_sel_cntx5_SHAMT 21
#define THCON_SEC0_REG2_Unpack_if_sel_cntx5_MASK 0x200000
#define THCON_SEC0_REG2_Unpack_if_sel_cntx6_ADDR32 73
#define THCON_SEC0_REG2_Unpack_if_sel_cntx6_SHAMT 22
#define THCON_SEC0_REG2_Unpack_if_sel_cntx6_MASK 0x400000
#define THCON_SEC0_REG2_Unpack_if_sel_cntx7_ADDR32 73
#define THCON_SEC0_REG2_Unpack_if_sel_cntx7_SHAMT 23
#define THCON_SEC0_REG2_Unpack_if_sel_cntx7_MASK 0x800000
#define THCON_SEC0_REG2_Metadata_x_end_ADDR32 73
#define THCON_SEC0_REG2_Metadata_x_end_SHAMT 24
#define THCON_SEC0_REG2_Metadata_x_end_MASK 0xff000000
#define THCON_SEC0_REG2_Unpack_limit_address_ADDR32 74
#define THCON_SEC0_REG2_Unpack_limit_address_SHAMT 0
#define THCON_SEC0_REG2_Unpack_limit_address_MASK 0x1ffff
#define THCON_SEC0_REG2_Unpack_fifo_size_ADDR32 75
#define THCON_SEC0_REG2_Unpack_fifo_size_SHAMT 0
#define THCON_SEC0_REG2_Unpack_fifo_size_MASK 0x1ffff
#define THCON_SEC0_REG3_Base_address_ADDR32 76
#define THCON_SEC0_REG3_Base_address_SHAMT 0
#define THCON_SEC0_REG3_Base_address_MASK 0xffffffff
#define THCON_SEC0_REG3_Base_cntx1_address_ADDR32 77
#define THCON_SEC0_REG3_Base_cntx1_address_SHAMT 0
#define THCON_SEC0_REG3_Base_cntx1_address_MASK 0xffffffff
#define THCON_SEC0_REG5_Dest_cntx0_address_ADDR32 84
#define THCON_SEC0_REG5_Dest_cntx0_address_SHAMT 0
#define THCON_SEC0_REG5_Dest_cntx0_address_MASK 0xffff
#define THCON_SEC0_REG5_Dest_cntx1_address_ADDR32 84
#define THCON_SEC0_REG5_Dest_cntx1_address_SHAMT 16
#define THCON_SEC0_REG5_Dest_cntx1_address_MASK 0xffff0000
#define THCON_SEC0_REG5_Dest_cntx2_address_ADDR32 85
#define THCON_SEC0_REG5_Dest_cntx2_address_SHAMT 0
#define THCON_SEC0_REG5_Dest_cntx2_address_MASK 0xffff
#define THCON_SEC0_REG5_Dest_cntx3_address_ADDR32 85
#define THCON_SEC0_REG5_Dest_cntx3_address_SHAMT 16
#define THCON_SEC0_REG5_Dest_cntx3_address_MASK 0xffff0000
#define THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32 86
#define THCON_SEC0_REG5_Tile_x_dim_cntx0_SHAMT 0
#define THCON_SEC0_REG5_Tile_x_dim_cntx0_MASK 0xffff
#define THCON_SEC0_REG5_Tile_x_dim_cntx1_ADDR32 86
#define THCON_SEC0_REG5_Tile_x_dim_cntx1_SHAMT 16
#define THCON_SEC0_REG5_Tile_x_dim_cntx1_MASK 0xffff0000
#define THCON_SEC0_REG5_Tile_x_dim_cntx2_ADDR32 87
#define THCON_SEC0_REG5_Tile_x_dim_cntx2_SHAMT 0
#define THCON_SEC0_REG5_Tile_x_dim_cntx2_MASK 0xffff
#define THCON_SEC0_REG5_Tile_x_dim_cntx3_ADDR32 87
#define THCON_SEC0_REG5_Tile_x_dim_cntx3_SHAMT 16
#define THCON_SEC0_REG5_Tile_x_dim_cntx3_MASK 0xffff0000
#define THCON_SEC1_REG0_TileDescriptor_ADDR32 112
#define THCON_SEC1_REG0_TileDescriptor_SHAMT 0
#define THCON_SEC1_REG0_TileDescriptor_MASK 0xffffffffffffffffffffffffffffffff
#define THCON_SEC1_REG1_Unp_LF8_4b_exp_ADDR32 119
#define THCON_SEC1_REG1_Unp_LF8_4b_exp_SHAMT 22
#define THCON_SEC1_REG1_Unp_LF8_4b_exp_MASK 0x400000
#define THCON_SEC1_REG2_Out_data_format_ADDR32 120
#define THCON_SEC1_REG2_Out_data_format_SHAMT 0
#define THCON_SEC1_REG2_Out_data_format_MASK 0xf
#define THCON_SEC1_REG2_Throttle_mode_ADDR32 120
#define THCON_SEC1_REG2_Throttle_mode_SHAMT 4
#define THCON_SEC1_REG2_Throttle_mode_MASK 0x30
#define THCON_SEC1_REG2_Context_count_ADDR32 120
#define THCON_SEC1_REG2_Context_count_SHAMT 6
#define THCON_SEC1_REG2_Context_count_MASK 0xc0
#define THCON_SEC1_REG2_Haloize_mode_ADDR32 120
#define THCON_SEC1_REG2_Haloize_mode_SHAMT 8
#define THCON_SEC1_REG2_Haloize_mode_MASK 0x100
#define THCON_SEC1_REG2_Tileize_mode_ADDR32 120
#define THCON_SEC1_REG2_Tileize_mode_SHAMT 9
#define THCON_SEC1_REG2_Tileize_mode_MASK 0x200
#define THCON_SEC1_REG2_Unpack_Src_Reg_Set_Upd_ADDR32 120
#define THCON_SEC1_REG2_Unpack_Src_Reg_Set_Upd_SHAMT 10
#define THCON_SEC1_REG2_Unpack_Src_Reg_Set_Upd_MASK 0x400
#define THCON_SEC1_REG2_Unpack_If_Sel_ADDR32 120
#define THCON_SEC1_REG2_Unpack_If_Sel_SHAMT 11
#define THCON_SEC1_REG2_Unpack_If_Sel_MASK 0x800
#define THCON_SEC1_REG2_Upsample_rate_ADDR32 120
#define THCON_SEC1_REG2_Upsample_rate_SHAMT 12
#define THCON_SEC1_REG2_Upsample_rate_MASK 0x3000
#define THCON_SEC1_REG2_Ovrd_data_format_ADDR32 120
#define THCON_SEC1_REG2_Ovrd_data_format_SHAMT 14
#define THCON_SEC1_REG2_Ovrd_data_format_MASK 0x4000
#define THCON_SEC1_REG2_Upsample_and_interleave_ADDR32 120
#define THCON_SEC1_REG2_Upsample_and_interleave_SHAMT 15
#define THCON_SEC1_REG2_Upsample_and_interleave_MASK 0x8000
#define THCON_SEC1_REG2_Shift_amount_cntx0_ADDR32 120
#define THCON_SEC1_REG2_Shift_amount_cntx0_SHAMT 16
#define THCON_SEC1_REG2_Shift_amount_cntx0_MASK 0xf0000
#define THCON_SEC1_REG2_Shift_amount_cntx1_ADDR32 120
#define THCON_SEC1_REG2_Shift_amount_cntx1_SHAMT 20
#define THCON_SEC1_REG2_Shift_amount_cntx1_MASK 0xf00000
#define THCON_SEC1_REG2_Shift_amount_cntx2_ADDR32 120
#define THCON_SEC1_REG2_Shift_amount_cntx2_SHAMT 24
#define THCON_SEC1_REG2_Shift_amount_cntx2_MASK 0xf000000
#define THCON_SEC1_REG2_Shift_amount_cntx3_ADDR32 120
#define THCON_SEC1_REG2_Shift_amount_cntx3_SHAMT 28
#define THCON_SEC1_REG2_Shift_amount_cntx3_MASK 0xf0000000
#define THCON_SEC1_REG2_Disable_zero_compress_cntx0_ADDR32 121
#define THCON_SEC1_REG2_Disable_zero_compress_cntx0_SHAMT 0
#define THCON_SEC1_REG2_Disable_zero_compress_cntx0_MASK 0x1
#define THCON_SEC1_REG2_Disable_zero_compress_cntx1_ADDR32 121
#define THCON_SEC1_REG2_Disable_zero_compress_cntx1_SHAMT 1
#define THCON_SEC1_REG2_Disable_zero_compress_cntx1_MASK 0x2
#define THCON_SEC1_REG2_Disable_zero_compress_cntx2_ADDR32 121
#define THCON_SEC1_REG2_Disable_zero_compress_cntx2_SHAMT 2
#define THCON_SEC1_REG2_Disable_zero_compress_cntx2_MASK 0x4
#define THCON_SEC1_REG2_Disable_zero_compress_cntx3_ADDR32 121
#define THCON_SEC1_REG2_Disable_zero_compress_cntx3_SHAMT 3
#define THCON_SEC1_REG2_Disable_zero_compress_cntx3_MASK 0x8
#define THCON_SEC1_REG2_Unpack_if_sel_cntx0_ADDR32 121
#define THCON_SEC1_REG2_Unpack_if_sel_cntx0_SHAMT 4
#define THCON_SEC1_REG2_Unpack_if_sel_cntx0_MASK 0x10
#define THCON_SEC1_REG2_Unpack_if_sel_cntx1_ADDR32 121
#define THCON_SEC1_REG2_Unpack_if_sel_cntx1_SHAMT 5
#define THCON_SEC1_REG2_Unpack_if_sel_cntx1_MASK 0x20
#define THCON_SEC1_REG2_Unpack_if_sel_cntx2_ADDR32 121
#define THCON_SEC1_REG2_Unpack_if_sel_cntx2_SHAMT 6
#define THCON_SEC1_REG2_Unpack_if_sel_cntx2_MASK 0x40
#define THCON_SEC1_REG2_Unpack_if_sel_cntx3_ADDR32 121
#define THCON_SEC1_REG2_Unpack_if_sel_cntx3_SHAMT 7
#define THCON_SEC1_REG2_Unpack_if_sel_cntx3_MASK 0x80
#define THCON_SEC1_REG2_Force_shared_exp_ADDR32 121
#define THCON_SEC1_REG2_Force_shared_exp_SHAMT 8
#define THCON_SEC1_REG2_Force_shared_exp_MASK 0x100
#define THCON_SEC1_REG2_Context_count_non_log2_ADDR32 121
#define THCON_SEC1_REG2_Context_count_non_log2_SHAMT 9
#define THCON_SEC1_REG2_Context_count_non_log2_MASK 0xe00
#define THCON_SEC1_REG2_Context_count_non_log2_en_ADDR32 121
#define THCON_SEC1_REG2_Context_count_non_log2_en_SHAMT 12
#define THCON_SEC1_REG2_Context_count_non_log2_en_MASK 0x1000
#define THCON_SEC1_REG2_Disable_zero_compress_cntx4_ADDR32 121
#define THCON_SEC1_REG2_Disable_zero_compress_cntx4_SHAMT 16
#define THCON_SEC1_REG2_Disable_zero_compress_cntx4_MASK 0x10000
#define THCON_SEC1_REG2_Disable_zero_compress_cntx5_ADDR32 121
#define THCON_SEC1_REG2_Disable_zero_compress_cntx5_SHAMT 17
#define THCON_SEC1_REG2_Disable_zero_compress_cntx5_MASK 0x20000
#define THCON_SEC1_REG2_Disable_zero_compress_cntx6_ADDR32 121
#define THCON_SEC1_REG2_Disable_zero_compress_cntx6_SHAMT 18
#define THCON_SEC1_REG2_Disable_zero_compress_cntx6_MASK 0x40000
#define THCON_SEC1_REG2_Disable_zero_compress_cntx7_ADDR32 121
#define THCON_SEC1_REG2_Disable_zero_compress_cntx7_SHAMT 19
#define THCON_SEC1_REG2_Disable_zero_compress_cntx7_MASK 0x80000
#define THCON_SEC1_REG2_Unpack_if_sel_cntx4_ADDR32 121
#define THCON_SEC1_REG2_Unpack_if_sel_cntx4_SHAMT 20
#define THCON_SEC1_REG2_Unpack_if_sel_cntx4_MASK 0x100000
#define THCON_SEC1_REG2_Unpack_if_sel_cntx5_ADDR32 121
#define THCON_SEC1_REG2_Unpack_if_sel_cntx5_SHAMT 21
#define THCON_SEC1_REG2_Unpack_if_sel_cntx5_MASK 0x200000
#define THCON_SEC1_REG2_Unpack_if_sel_cntx6_ADDR32 121
#define THCON_SEC1_REG2_Unpack_if_sel_cntx6_SHAMT 22
#define THCON_SEC1_REG2_Unpack_if_sel_cntx6_MASK 0x400000
#define THCON_SEC1_REG2_Unpack_if_sel_cntx7_ADDR32 121
#define THCON_SEC1_REG2_Unpack_if_sel_cntx7_SHAMT 23
#define THCON_SEC1_REG2_Unpack_if_sel_cntx7_MASK 0x800000
#define THCON_SEC1_REG2_Metadata_x_end_ADDR32 121
#define THCON_SEC1_REG2_Metadata_x_end_SHAMT 24
#define THCON_SEC1_REG2_Metadata_x_end_MASK 0xff000000
#define THCON_SEC1_REG2_Unpack_limit_address_ADDR32 122
#define THCON_SEC1_REG2_Unpack_limit_address_SHAMT 0
#define THCON_SEC1_REG2_Unpack_limit_address_MASK 0x1ffff
#define THCON_SEC1_REG2_Unpack_fifo_size_ADDR32 123
#define THCON_SEC1_REG2_Unpack_fifo_size_SHAMT 0
#define THCON_SEC1_REG2_Unpack_fifo_size_MASK 0x1ffff
#define THCON_SEC1_REG3_Base_address_ADDR32 124
#define THCON_SEC1_REG3_Base_address_SHAMT 0
#define THCON_SEC1_REG3_Base_address_MASK 0xffffffff
#define THCON_SEC1_REG3_Base_cntx1_address_ADDR32 125
#define THCON_SEC1_REG3_Base_cntx1_address_SHAMT 0
#define THCON_SEC1_REG3_Base_cntx1_address_MASK 0xffffffff
// Registers for GLOBAL
#define SCRATCH_SEC0_val_ADDR32 209
#define SCRATCH_SEC0_val_SHAMT 0
#define SCRATCH_SEC0_val_MASK 0xffffffff
#define SCRATCH_SEC1_val_ADDR32 210
#define SCRATCH_SEC1_val_SHAMT 0
#define SCRATCH_SEC1_val_MASK 0xffffffff
#define SCRATCH_SEC2_val_ADDR32 211
#define SCRATCH_SEC2_val_SHAMT 0
#define SCRATCH_SEC2_val_MASK 0xffffffff
"""

CKERNEL_GPR_MAP = r"""
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Hand-coded parameter encoding for various GPR mappings
#include <cstdint>

namespace ckernel
{

// Common GPR mapping across all threads
struct p_gpr
{
    constexpr static std::uint32_t ZERO         = 0; // Always stores 0
    constexpr static std::uint32_t DBG_RESERVED = 1; // Reserved for future use
    constexpr static std::uint32_t DBG_MSG      = 2; // Firmware debug message
    constexpr static std::uint32_t DBG_CKID     = 3; // Ckernel ID
};

// Unpack GPR thread
struct p_gpr_unpack
{
    constexpr static std::uint32_t OPERAND_BASE_ADDR       = 4;  // Operand base address used by zero buffer function
    constexpr static std::uint32_t OPERAND_OFFSET_ADDR     = 5;  // Operand offset address used by zero buffer function
    constexpr static std::uint32_t ZERO_0                  = 8;  // Zero data
    constexpr static std::uint32_t ZERO_1                  = 9;  // Zero data
    constexpr static std::uint32_t ZERO_2                  = 10; // Zero data
    constexpr static std::uint32_t ZERO_3                  = 11; // Zero data
    constexpr static std::uint32_t TMP0                    = 12; // Temp data
    constexpr static std::uint32_t TMP1                    = 13; // Temp data
    constexpr static std::uint32_t TILE_SIZE               = 14; // Tile size
    constexpr static std::uint32_t TILE_OFFSET             = 15; // Tile offset
    constexpr static std::uint32_t L1_BUFFER_ADDR          = 17; // Holds address of fixed l1 buffer used for reduce in1
    constexpr static std::uint32_t TMP_LO                  = 18; // Temp data. Upper 16-bits always 0
    constexpr static std::uint32_t TMP_HI                  = 19; // Temp data. Lower 16-bits always 0
    constexpr static std::uint32_t PERF_FIRST_UNP_LO       = 32; // timestamp for first-unpack-instruction (low 32b)
    constexpr static std::uint32_t PERF_FIRST_UNP_HI       = 33; // timestamp for first-unpack-instruction (high 32b)
    constexpr static std::uint32_t TILE_SIZE_A             = 36; // Holds tile size for unpacker 0
    constexpr static std::uint32_t TILE_SIZE_B             = 37; // Holds tile size for unpacker 1
    constexpr static std::uint32_t KT_DIM                  = 38; // Holds matmul kt_dim
    constexpr static std::uint32_t FACE_DIM_16x16          = 40; // Holds face dimension (16x16)
    constexpr static std::uint32_t FACE_DIM_8x16           = 41; // Holds face dimension (8x16)
    constexpr static std::uint32_t FACE_DIM_4x16           = 42; // Holds face dimension (4x16)
    constexpr static std::uint32_t FACE_DIM_2x16           = 43; // Holds face dimension (2x16)
    constexpr static std::uint32_t FACE_DIM_1x16           = 44; // Holds face dimension (1x16)
    constexpr static std::uint32_t PERF_UNPACK_NUM_TILES_0 = 45; // num tiles for input operands 0-1
    constexpr static std::uint32_t PERF_UNPACK_NUM_TILES_1 = 46; // num tiles for input operands 2-3
    constexpr static std::uint32_t PERF_UNPACK_NUM_TILES_2 = 47; // num tiles for input operands 4-5
    constexpr static std::uint32_t PERF_UNPACK_NUM_TILES_3 = 48; // num tiles for input operands 6-7
    // Slot 52 was the legacy UNPACK_STRIDE GPR used to snapshot/restore the channel-1 Z-stride
    // around unpack-to-dest. Removed in favor of canonical_unpA_z_stride() in the bracket pair
    // (set_dst_write_addr / unpack_to_dest_tile_done), which recomputes the baseline instead of
    // preserving caller-side state. Slot is reserved as free until the next unpacker GPR consolidation.
    constexpr static std::uint32_t SR_UNPACK_TILIZER_STATE_0   = 54; // Save unpack state before tilizer is enabled for quick restore
    constexpr static std::uint32_t SR_UNPACK_TILIZER_STATE_1   = 55;
    constexpr static std::uint32_t SR_UNPACK_UNTILIZER_STATE_0 = 56; // Save unpack state before tilizer is enabled for quick restore
    constexpr static std::uint32_t SR_UNPACK_UNTILIZER_STATE_1 = 57;
    constexpr static std::uint32_t SR_UNPACK_UNTILIZER_STATE_2 = 58;
    constexpr static std::uint32_t SR_UNPACK_UNTILIZER_STATE_3 = 59;
};

// Math GPR thread
struct p_gpr_math
{
    constexpr static std::uint32_t PERF_DBUS_CNTL           = 4;  // Control debug bus perf counter selection
    constexpr static std::uint32_t PERF_MEM_DUMP_CNTL_CLEAR = 5;  // Clear write to memory flag
    constexpr static std::uint32_t PERF_MEM_DUMP_CNTL_SET   = 6;  // Set write to memory flag
    constexpr static std::uint32_t PERF_CNT_START           = 7;  // Start perf counter
    constexpr static std::uint32_t PERF_CNT_STOP            = 8;  // Stop perf counter
    constexpr static std::uint32_t PERF_EPOCH_BASE_ADDR     = 9;  // Perf event ID
    constexpr static std::uint32_t PERF_EPOCH_OFFSET        = 10; // The offset address for epoch variables
    constexpr static std::uint32_t DEST_OP0_BASE            = 48; // dest base for sfpu op0
    constexpr static std::uint32_t DEST_OP1_BASE            = 49; // dest base for sfpu op1
    constexpr static std::uint32_t DEST_REGW_OFFSET         = 50; // dest rwc base (1st set)
    constexpr static std::uint32_t DEST_REGW_INCR           = 51; // dest rwc incr (1st set)
    constexpr static std::uint32_t DEST_REGW_OFFSET2        = 52; // dest rwc base (2nd set)
    constexpr static std::uint32_t DEST_REGW_INCR2          = 53; // dest rwc incr (2nd set)
    constexpr static std::uint32_t TMP0                     = 60;
    constexpr static std::uint32_t NUM_DRAM_REQS            = 61;
};

// Pack GPR thread
struct p_gpr_pack
{
    constexpr static std::uint32_t DEST_OFFSET_LO = 4;  // dest lower bank offsets
    constexpr static std::uint32_t DEST_OFFSET_HI = 8;  // dest upper bank offsets
    constexpr static std::uint32_t OUTPUT_ADDR    = 12; // output address that packer is writing to
    constexpr static std::uint32_t TILE_HEADER    = 16; // tile header - ID + tile size

    constexpr static std::uint32_t TEMP_TILE_OFFSET    = 20; // Temp var which holds tile offset in dest
    constexpr static std::uint32_t NUM_MSGS_RECEIVED   = 24; // holds tile count and word size
    constexpr static std::uint32_t ONE_MSG_RECEIVED    = 25; // by default holds 1 tile count and word size for streaming per tile
    constexpr static std::uint32_t HEADER_ADDR         = 26; // Holds the address of the header (used by pack shift kernel only)
    constexpr static std::uint32_t TMP0                = 28; // Temp data
    constexpr static std::uint32_t TMP1                = 29; // Temp data
    constexpr static std::uint32_t TMP_LO              = 30; // Temp data, upper 16-bit always 0
    constexpr static std::uint32_t TMP_HI              = 31; // Temp data, lower 16-bit always 0
    constexpr static std::uint32_t PACK_STREAM_SYNC    = 32; // sync between pack and output stream [32:63]
    constexpr static std::uint32_t OUTPUT_ADDR_OFFSET  = 50; // output offset address that's added to OUTPUT_ADDR
    constexpr static std::uint32_t PERF_PACK_NUM_TILES = 51; // output operand num tiles
    constexpr static std::uint32_t EXP0_SEC_SIZE_BFP   = 52; // pack0,1,2,3 exp section size for bfp8,4,2
    constexpr static std::uint32_t EXP1_SEC_SIZE_BFP8  = 53; // pack1 exp section size for bfp8
    constexpr static std::uint32_t EXP2_SEC_SIZE_BFP8  = 54; // pack2 exp section size for bfp8
    constexpr static std::uint32_t EXP3_SEC_SIZE_BFP8  = 55; // pack2 exp section size for bfp8
    constexpr static std::uint32_t EXP1_SEC_SIZE_BFP4  = 57; // pack1 exp section size for bfp4
    constexpr static std::uint32_t EXP2_SEC_SIZE_BFP4  = 58; // pack2 exp section size for bfp4
    constexpr static std::uint32_t EXP3_SEC_SIZE_BFP4  = 59; // pack3 exp section size for bfp4
    constexpr static std::uint32_t EXP1_SEC_SIZE_BFP2  = 61; // pack1 exp section size for bfp2
    constexpr static std::uint32_t EXP2_SEC_SIZE_BFP2  = 62; // pack2 exp section size for bfp2
    constexpr static std::uint32_t EXP3_SEC_SIZE_BFP2  = 63; // pack3 exp section size for bfp2
};

} // namespace ckernel

"""

CUNPACK_COMMON = r"""
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "llk_assert.h"

namespace ckernel::unpacker
{
constexpr std::uint32_t TILE_DESC_SIZE = 2; // Unpacker descriptor size in dwords
constexpr std::uint32_t CONFIG_SIZE    = 2; // Unpacker configuration size in dwords
constexpr std::uint32_t NUM_UNPACKERS  = 2; // Number of unpackers

// Unpack tile descriptor
typedef struct
{
    // word 0
    std::uint32_t in_data_format : DATA_FORMAT_BIT_COUNT;
    std::uint32_t uncompressed       : 1;
    std::uint32_t reserved_0         : 3;
    std::uint32_t blobs_per_xy_plane : 4;
    std::uint32_t reserved_1         : 4;
    std::uint32_t x_dim              : 16;
    // word 1
    std::uint32_t y_dim : 16;
    std::uint32_t z_dim : 16;
    // word 2
    std::uint32_t w_dim            : 16;
    std::uint32_t blobs_y_start_lo : 16;
    // word 3
    std::uint32_t blobs_y_start_hi : 16;
    std::uint32_t digest_type      : 8; // Not used
    std::uint32_t digest_size      : 8; // Not used
} unpack_tile_descriptor_t;             // Unpack configuration

static_assert(sizeof(unpack_tile_descriptor_t) == (sizeof(std::uint32_t) * 4));

typedef union
{
    std::uint32_t val[4];
    unpack_tile_descriptor_t f;
} unpack_tile_descriptor_u;

// Unpack config
typedef struct
{
    // word 0
    std::uint32_t out_data_format : DATA_FORMAT_BIT_COUNT;
    std::uint32_t throttle_mode             : 2;
    std::uint32_t context_count             : 2;
    std::uint32_t haloize_mode              : 1; // this controls xy transpose on unpacker
    std::uint32_t tileize_mode              : 1;
    std::uint32_t unpack_src_reg_set_update : 1;
    std::uint32_t unpack_if_sel             : 1;
    std::uint32_t upsample_rate             : 2;
    std::uint32_t reserved_1                : 1;
    std::uint32_t upsamle_and_interlave     : 1;
    std::uint32_t shift_amount              : 16;
    // word 1
    std::uint32_t uncompress_cntx0_3    : 4;
    std::uint32_t unpack_if_sel_cntx0_3 : 4;
    std::uint32_t force_shared_exp      : 1;
    std::uint32_t reserved_2            : 7;
    std::uint32_t uncompress_cntx4_7    : 4;
    std::uint32_t unpack_if_sel_cntx4_7 : 4;
    std::uint32_t reserved_3            : 8;
    // word 2
    std::uint32_t limit_addr : 17;
    std::uint32_t reserved_4 : 15;
    // word 3
    std::uint32_t fifo_size  : 17;
    std::uint32_t reserved_5 : 15;
} unpack_config_t;

static_assert(sizeof(unpack_config_t) == (sizeof(std::uint32_t) * 4));

typedef union
{
    std::uint32_t val[4];
    unpack_config_t f;
} unpack_config_u;

// ALU config
typedef struct
{
    std::uint32_t ALU_ROUNDING_MODE_Fpu_srnd_en     : 1;
    std::uint32_t ALU_ROUNDING_MODE_Gasket_srnd_en  : 1;
    std::uint32_t ALU_ROUNDING_MODE_Packer_srnd_en  : 1;
    std::uint32_t ALU_ROUNDING_MODE_Padding         : 10;
    std::uint32_t ALU_ROUNDING_MODE_GS_LF           : 1;
    std::uint32_t ALU_ROUNDING_MODE_Bfp8_HF         : 1;
    std::uint32_t ALU_FORMAT_SPEC_REG0_SrcAUnsigned : 1;
    std::uint32_t ALU_FORMAT_SPEC_REG0_SrcBUnsigned : 1;
    std::uint32_t ALU_FORMAT_SPEC_REG0_SrcA         : 4;
    std::uint32_t ALU_FORMAT_SPEC_REG1_SrcB         : 4;
    std::uint32_t ALU_FORMAT_SPEC_REG2_Dstacc       : 4;
    std::uint32_t ALU_ACC_CTRL_Fp32_enabled         : 1;
    std::uint32_t ALU_ACC_CTRL_SFPU_Fp32_enabled    : 1;
    std::uint32_t ALU_ACC_CTRL_INT8_math_enabled    : 1;
} alu_config_t;

static_assert(sizeof(alu_config_t) == sizeof(std::uint32_t));

typedef union
{
    std::uint32_t val;
    alu_config_t f;
} alu_config_u;

// Set unpacker offsets to 0, except for unpacker 0, channel 1, X, which is the tile X dimension
inline void unpacker_addr_counter_init()
{
    TTI_SETADCXY(p_setadc::UNP_A | p_setadc::UNP_B, 0, 0, 0, 0, 0b1011);
    TTI_SETADCZW(p_setadc::UNP_A | p_setadc::UNP_B, 0, 0, 0, 0, 0b1111);
}

inline void unpacker_iteration_cleanup(std::uint32_t &context)
{
    // Indicate that unpacker is done, and we can program the next one
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    context = 1 - context;
    if (context == 1)
    {
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0104);
    }
    else
    {
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

inline std::uint32_t unpack_16B_address(const std::uint32_t addr)
{
    return (addr << FIFO_BASE_ADDRESS_ALIGN_BITS) >> 4;
}

inline void flush_xsearch_cache(const std::uint32_t unpacker)
{
    TTI_UNPACR(unpacker, 0, 0, 0, 0, 0, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 1, 0);
}

// Wait for threshold of busy contexts to fall below total available contexts
inline void wait_for_next_context(const std::uint32_t num_contexts)
{
    while (semaphore_read(semaphore::UNPACK_SYNC) >= num_contexts)
    {
    }
}

inline void switch_config_context(std::uint32_t &unp_cfg_context)
{
    // Switch config context
    unp_cfg_context = 1 - unp_cfg_context;
    if (unp_cfg_context == 0)
    {
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    }
    else
    {
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
    while (semaphore_read(semaphore::UNPACK_SYNC) > 0)
    {
    }
}

/**
 * \brief Returns true if unpacker I/O uses 32-bit formats (Int32 or Float32).
 *
 * Used to determine unpack-to-dest mode and related configuration when both
 * input and output are 32-bit. Masks low nibble of format codes for comparison.
 *
 * \param unpack_src_format Unpacker input (L1) data format.
 * \param unpack_dst_format Unpacker output (register) data format.
 * \return true if both formats are Int32 or Float32; false otherwise.
 */
inline constexpr bool is_32bit_input(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format)
{
    const DataFormat input_df  = static_cast<DataFormat>(masked_data_format(unpack_src_format));
    const DataFormat output_df = static_cast<DataFormat>(masked_data_format(unpack_dst_format));
    return (input_df == DataFormat::Int32 || input_df == DataFormat::Float32) && (output_df == DataFormat::Int32 || output_df == DataFormat::Float32);
}

// Canonical srcA channel-1 X stride in bytes, derived from the unpacker dst format.
// Used to compute the canonical Y/Z strides programmed into UNP0_ADDR_CTRL_*.
inline constexpr std::uint32_t canonical_unpA_x_stride(const std::uint32_t unpack_dst_format)
{
    return (unpack_dst_format & 0x3) == to_underlying(DataFormat::Float32) ? 4 : (unpack_dst_format & 0x3) == to_underlying(DataFormat::Float16) ? 2 : 1;
}

// Canonical srcA channel-1 Y stride: face row step (one face row = FACE_R_DIM datums).
//
// This is the value programmed by configure_unpack_AB and the srca data-format reconfig,
// so it serves as the documented baseline that operations which mutate Y-stride
// (e.g. tilizeA_B) must restore on uninit. It does not depend on the operand's face_r_dim;
// FACE_R_DIM is the constant face row count (matches set_packer_strides and unpack_untilize).
inline constexpr std::uint32_t canonical_unpA_y_stride(const std::uint32_t unpack_dst_format)
{
    return FACE_R_DIM * canonical_unpA_x_stride(unpack_dst_format);
}

/*
 * Single source of truth for whether _llk_unpack_A_ takes the "unpack to dest" (SrcA -> DEST) path.
 * It is only taken for genuinely 32-bit input; otherwise the MOP falls through to the normal/broadcast
 * path. Both _llk_unpack_A_init_ (which programs the X counter) and _llk_unpack_A_mop_config_ (which
 * programs the MOP) gate on this, so they cannot diverge if the policy ever changes.
 *
 * \param unpack_to_dest    Whether the caller requested unpack-to-dest.
 * \param unpack_src_format Unpacker input (L1) data format.
 * \param unpack_dst_format Unpacker output (register) data format.
 */
inline constexpr bool should_unpack_to_dest(const bool unpack_to_dest, const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format)
{
    return unpack_to_dest && is_32bit_input(unpack_src_format, unpack_dst_format);
}

// Canonical srcA channel-1 Z stride: face-sized row step.
//
// Programmed by configure_unpack_AB and the srca data-format reconfig, and re-asserted by
// unpack-to-dest brackets. Z-stride does not depend on operand face_r_dim; FACE_R_DIM is
// the constant face row count because data is not stored densely in src/dest registers
// (matches the explicit comment in _llk_unpack_reconfig_data_format_srca_impl_).
inline constexpr std::uint32_t canonical_unpA_z_stride(const std::uint32_t unpack_dst_format)
{
    return FACE_C_DIM * FACE_R_DIM * canonical_unpA_x_stride(unpack_dst_format);
}

// Canonical srcA tile-descriptor X-dim programmed by configure_unpack_AB. 0 because
// Tile_x_dim_cntx0 overrides it for srcA.
constexpr std::uint32_t CANONICAL_UNPA_TILE_X_DIM = 0;

// Mask for the upper halfword of a TileDescriptor config word, where the X/Y/Z dim fields live.
constexpr std::uint32_t TILE_DESC_UPPER_HALFWORD_MASK = 0xffff0000;

// Canonical Tile_x_dim_cntx0 word: face_dim packed into both cntx0 (low 16) and cntx1 (high 16),
// where face_dim = face_r_dim * FACE_C_DIM. configure_unpack_AB programs this value.
inline constexpr std::uint32_t canonical_unpA_tile_x_dim_cntx(const std::uint32_t face_r_dim)
{
    const std::uint32_t face_dim = face_r_dim * FACE_C_DIM;
    return face_dim | (face_dim << 16);
}

template <bool is_fp32_dest_acc_en, bool row_pool = false, bool fpu_srnd_en = false, bool pack_srnd_en = false>
inline void configure_unpack_AB(
    const std::uint32_t unpA_src_format,
    const std::uint32_t unpB_src_format,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t unpB_dst_format,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM,
    const bool transpose_xy_srca_en     = false,
    const std::uint32_t unpA_num_faces  = 4,
    const std::uint32_t unpB_num_faces  = 4)
{
    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");

    LLK_ASSERT(
        is_unpacker_format_conversion_supported_fp32_acc(
            static_cast<DataFormat>(unpA_src_format), static_cast<DataFormat>(unpA_dst_format), is_fp32_dest_acc_en),
        "Unsupported unpacker to register conversion.");
    LLK_ASSERT(
        is_unpacker_format_conversion_supported_fp32_acc(
            static_cast<DataFormat>(unpB_src_format), static_cast<DataFormat>(unpB_dst_format), is_fp32_dest_acc_en),
        "Unsupported unpacker to register conversion.");

    // Check that unpacker is done (all contexts freed up) before starting hw configuration
    wait_for_idle();

    // Reset address counters
    unpacker_addr_counter_init();

    const std::uint32_t unpA_src_format_masked = masked_data_format(unpA_src_format);
    const std::uint32_t unpB_src_format_masked = masked_data_format(unpB_src_format);
    const std::uint32_t unpA_dst_format_masked = masked_data_format(unpA_dst_format);
    const std::uint32_t unpB_dst_format_masked = masked_data_format(unpB_dst_format);

    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();

    std::uint32_t unpA_ch1_x_stride = datum_size_in_bytes(unpA_dst_format_masked);
    std::uint32_t unpB_ch1_x_stride = datum_size_in_bytes(unpB_dst_format_masked);
    std::uint32_t unpA_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpA_ch1_x_stride;
    std::uint32_t unpB_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpB_ch1_x_stride;
    std::uint32_t exp_width         = (static_cast<std::uint32_t>(unpA_dst_format_masked) >> 2) & 0x1; // 0=5-bit, 1=8-bit

    // Strides for incrementing ch1 address to srcA and srcB
    TT_CFG_STORE(
        cfg,
        UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32,
        (0 << UNP0_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT) |
            (unpA_ch1_z_stride << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT)); // Z and W(not used) stride for dest address (ch1)

    TT_CFG_STORE(
        cfg,
        UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32,
        (0 << UNP1_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT) |
            (unpB_ch1_z_stride << UNP1_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT)); // Z and W(not used) stride for dest address (ch1)

    // Establish the canonical Y-stride baseline for srcA (ch1) here, so per-op inits that
    // mutate Y-stride (e.g. tilizeA_B) can deterministically restore this baseline on uninit
    // rather than snapshotting the previous register value into a GPR.
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32, UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT, UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(
        canonical_unpA_y_stride(unpA_dst_format_masked));

    // Math ALU_FORMAT_REG
    t6_mutex_acquire(mutex::REG_RMW);
    alu_config_u alu_payload = {.val = 0};

    constexpr std::uint32_t alu_format_mask = ALU_FORMAT_SPEC_REG0_SrcAUnsigned_MASK | ALU_FORMAT_SPEC_REG0_SrcBUnsigned_MASK;

    if (unpA_src_format == to_underlying(DataFormat::UInt8))
    {
        alu_payload.f.ALU_FORMAT_SPEC_REG0_SrcAUnsigned = 1;
    }
    if (unpB_src_format == to_underlying(DataFormat::UInt8))
    {
        alu_payload.f.ALU_FORMAT_SPEC_REG0_SrcBUnsigned = 1;
    }

    // FP32 accumulation and SFPU to read dest as FP32
    // NOTE: This assumes these config fields are adjacent and in same register!!
    static_assert(ALU_ACC_CTRL_Fp32_enabled_ADDR32 == ALU_FORMAT_SPEC_REG0_SrcA_ADDR32);
    static_assert(ALU_ACC_CTRL_Fp32_enabled_ADDR32 == ALU_ACC_CTRL_SFPU_Fp32_enabled_ADDR32);
    constexpr std::uint32_t alu_stoch_rnd_mask =
        ALU_ROUNDING_MODE_Fpu_srnd_en_MASK | ALU_ROUNDING_MODE_Gasket_srnd_en_MASK | ALU_ROUNDING_MODE_Packer_srnd_en_MASK;
    alu_payload.f.ALU_ROUNDING_MODE_Fpu_srnd_en    = fpu_srnd_en;
    alu_payload.f.ALU_ROUNDING_MODE_Gasket_srnd_en = pack_srnd_en;
    alu_payload.f.ALU_ROUNDING_MODE_Packer_srnd_en = pack_srnd_en;

    constexpr std::uint32_t alu_mask = alu_format_mask | alu_stoch_rnd_mask;

    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, alu_mask>(alu_payload.val);

    // Set FP8 E4M3 mode, bit is accessible by unpacker/packer
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Unp_LF8_4b_exp_RMW>(((unpA_src_format & 0x1F) == (std::uint32_t)DataFormat::Fp8_e4m3) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC1_REG1_Unp_LF8_4b_exp_RMW>(((unpB_src_format & 0x1F) == (std::uint32_t)DataFormat::Fp8_e4m3) ? 1 : 0);

    t6_mutex_release(mutex::REG_RMW);

    // Set tile descriptor
    unpack_tile_descriptor_u tile_descriptor;
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++)
    {
        tile_descriptor.val[i] = 0;
    }
    tile_descriptor.f.in_data_format = static_cast<std::uint32_t>(unpA_src_format_masked);
    tile_descriptor.f.uncompressed   = 1; // Input tile is uncompressed
    tile_descriptor.f.x_dim          = 0; // Not used for unpA as value is overridden by per context x_dim set below. Used for unpB
    tile_descriptor.f.y_dim          = 1;
    tile_descriptor.f.z_dim          = unpA_num_faces;
    // tile_descriptor.f.blobs_per_xy_plane = 0;
    // tile_descriptor.f.blobs_y_start = 0;
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++)
    {
        TT_CFG_STORE(cfg, THCON_SEC0_REG0_TileDescriptor_ADDR32 + i, tile_descriptor.val[i]);
    }
    tile_descriptor.f.in_data_format = row_pool ? to_underlying(DataFormat::Float32) : unpB_src_format_masked;
    tile_descriptor.f.x_dim          = unpB_face_r_dim * FACE_C_DIM;
    tile_descriptor.f.z_dim          = unpB_num_faces;
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++)
    {
        TT_CFG_STORE(cfg, THCON_SEC1_REG0_TileDescriptor_ADDR32 + i, tile_descriptor.val[i]);
    }

    // Set unpacker config
    unpack_config_u config;
    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++)
    {
        config.val[i] = 0;
    }
    config.f.out_data_format = unpA_dst_format_masked;
    config.f.throttle_mode   = 2;
    config.f.context_count   = 0;
    config.f.haloize_mode    = transpose_xy_srca_en ? 1 : 0;
    // config.f.upsample_rate   = 0;
    // config.f.upsamle_and_interlave  = 0;
    // config.f.shift_amount = 0;
    config.f.uncompress_cntx0_3 = 0xf;
    config.f.uncompress_cntx4_7 = 0xf;
    // config.f.limit_addr = 0; // Set dynamically
    // config.f.fifo_size = 0; // Set dynamically
    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++)
    {
        TT_CFG_STORE(cfg, THCON_SEC0_REG2_Out_data_format_ADDR32 + i, config.val[i]);
    }

    config.f.out_data_format = row_pool ? (to_underlying(DataFormat::Float16) | (exp_width << 2)) : unpB_dst_format_masked;
    config.f.haloize_mode    = 0;

    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++)
    {
        TT_CFG_STORE(cfg, THCON_SEC1_REG2_Out_data_format_ADDR32 + i, config.val[i]);
    }

    // Program base address for all 2 sections (each section address is loaded to corresponding context)
    // Load dummy data to unused location if face height is 0
    const std::uint32_t Dest_cntx0_address         = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    const std::uint32_t Dest_cntx1_address         = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    TT_CFG_STORE(cfg, THCON_SEC0_REG5_Dest_cntx0_address_ADDR32, Dest_cntx0_address | (Dest_cntx1_address << 16));

    // Program unpacker0 per context x_dim (face size in l1)
    // Overrides value set by tile descriptor when thread override bit is set in unpack instruction
    const std::uint32_t face_dim                 = unpA_face_r_dim * FACE_C_DIM;
    TT_CFG_STORE(cfg, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, face_dim | (face_dim << 16));

    constexpr std::uint32_t face_dim_16x16 = FACE_R_DIM * FACE_C_DIM;
    regfile[p_gpr_unpack::FACE_DIM_16x16]  = (face_dim_16x16 / 1) | ((face_dim_16x16 / 1) << 16);
    regfile[p_gpr_unpack::FACE_DIM_8x16]   = (face_dim_16x16 / 2) | ((face_dim_16x16 / 2) << 16);
    regfile[p_gpr_unpack::FACE_DIM_4x16]   = (face_dim_16x16 / 4) | ((face_dim_16x16 / 4) << 16);
    regfile[p_gpr_unpack::FACE_DIM_2x16]   = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    regfile[p_gpr_unpack::FACE_DIM_1x16]   = (face_dim_16x16 / 16) | ((face_dim_16x16 / 16) << 16);
    sync_regfile_write(p_gpr_unpack::FACE_DIM_1x16);

    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4);

    // Enable address counter for unpacker ch1/dst address
    // final address is calculated as: Dest_cntx0/1_address + address_counter_ch1
    // used for face by face unpacking of entire tile into srcA
    TT_CFG_STORE(cfg, UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_ADDR32, 0x1 << UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_SHAMT);

    // Clear context ID
    reset_config_context();
}

template <std::uint32_t UNP_SEL = p_setadc::UNP_AB>
inline void config_unpacker_x_end(const std::uint32_t face_r_dim)
{
    static_assert(UNP_SEL == p_setadc::UNP_A || UNP_SEL == p_setadc::UNP_B || UNP_SEL == p_setadc::UNP_AB, "UNP_SEL must be UNP_A, UNP_B, or UNP_AB");
    LLK_ASSERT(
        face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == FACE_R_DIM, "face_r_dim must be 1, 2, 4, 8, or FACE_R_DIM");

    switch (face_r_dim)
    {
        case 1:
            TTI_SETADCXX(UNP_SEL, 1 * FACE_C_DIM - 1, 0x0);
            break;
        case 2:
            TTI_SETADCXX(UNP_SEL, 2 * FACE_C_DIM - 1, 0x0);
            break;
        case 4:
            TTI_SETADCXX(UNP_SEL, 4 * FACE_C_DIM - 1, 0x0);
            break;
        case 8:
            TTI_SETADCXX(UNP_SEL, 8 * FACE_C_DIM - 1, 0x0);
            break;
        default:
            TTI_SETADCXX(UNP_SEL, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
            break;
    }
}

inline void wait_for_dest_available()
{
    t6_semaphore_wait_on_max<p_stall::STALL_UNPACK>(semaphore::UNPACK_TO_DEST);
}

// Restore srcA channel-1 Z-stride to the canonical baseline derived from unpack_dst_format.
// This pairs with set_dst_write_addr to bracket the unpack-to-dest section: rather than
// snapshotting the prior register value into a GPR, we recompute the canonical baseline so the
// restore is deterministic and order-independent.
inline void unpack_to_dest_tile_done(std::uint32_t &context_id, const std::uint32_t unpack_dst_format)
{
    t6_semaphore_post<p_stall::UNPACK0>(semaphore::UNPACK_TO_DEST);
    TT_SETDMAREG(0, LOWER_HALFWORD(canonical_unpA_z_stride(unpack_dst_format) << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT), 0, LO_16(p_gpr_unpack::TMP_LO));
    TTI_WRCFG(p_gpr_unpack::TMP_LO, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32); // Restore canonical Z-stride
    // Restore config context
    if (context_id == 0)
    {
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(0);
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(4 * 16);
    }
    else
    {
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(0);
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(4 * 16);
    }
    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4); // re-enable address bit swizzle
}

inline void set_dst_write_addr(const std::uint32_t &context_id, const std::uint32_t &unpack_dst_format)
{
    std::uint32_t dst_byte_addr = 16 * (4 + mailbox_read(ThreadId::MathThreadId)); // Apply fixed offset of 4*16 to dest address
    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x0);                                         // Disable address bit swizzle
    // Set unpacker Z-stride to the canonical baseline for unpack_dst_format. Paired with
    // unpack_to_dest_tile_done's canonical restore; no GPR snapshot of the prior value.
    TT_SETDMAREG(0, LOWER_HALFWORD(canonical_unpA_z_stride(unpack_dst_format) << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT), 0, LO_16(p_gpr_unpack::TMP_LO));
    TTI_WRCFG(p_gpr_unpack::TMP_LO, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    if (context_id == 0)
    {
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(1);
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(dst_byte_addr);
    }
    else
    {
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(1);
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(dst_byte_addr);
    }
}
} // namespace ckernel::unpacker
"""

LLK_UNPACK_A = r"""
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_unpack.h"

using namespace ckernel;
using namespace ckernel::unpacker;

namespace llk_unpack_a_detail
{
template <EltwiseBinaryReuseDestType binary_reuse_dest>
constexpr std::uint32_t dest_reuse_dummy_unpack()
{
    static_assert(binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA || binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB);
    constexpr std::uint32_t source = binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA ? SrcA : SrcB;
    return TT_OP_UNPACR_NOP(source, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1 /* wait like UNPACR */, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
}

} // namespace llk_unpack_a_detail

/**
 * @brief Program the unpacker MOP for a single-operand (A) unpack.
 *
 * Selects the UNPACR instruction sequence based on broadcast type, dest-reuse mode and
 * whether data is unpacked straight to the dest register, covering transpose-of-faces and
 * 32-bit-to-dest paths.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @tparam acc_to_dest: Accumulate the operand into the dest register rather than overwriting it.
 * @tparam binary_reuse_dest: Reuse dest as a source operand, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 * @param transpose_of_faces: Whether faces are reordered (transposed) during the unpack.
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim).
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 */
template <
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false>
inline void _llk_unpack_A_mop_config_(
    const bool transpose_of_faces, const ckernel::TensorShape tensor_shape, const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format = 0)
{
    static_assert(
        !((BType != BroadcastType::NONE) && acc_to_dest && (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)), "Not supported configuration!");
    static_assert(
        !(((acc_to_dest) || (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE)) && (unpack_to_dest)),
        "Not supported configuration when unpacking to dest!");
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_A_mop_config_", tensor_shape);
    const std::uint8_t num_faces = tensor_shape.total_num_faces();

    static constexpr std::uint32_t unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srca_to_dest =
        TT_OP_UNPACR(SrcA, 0b00010001 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // ch0/ch1 z_inc
    static constexpr std::uint32_t unpack_srca_to_dest_column =
        TT_OP_UNPACR(SrcA, 0b00100010 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // ch0/ch1 z_inc
    static constexpr std::uint32_t unpack_srca_to_dest_transpose_of_faces =
        TT_OP_UNPACR(SrcA, 0b00010010, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc srcA ch1_z+=1, ch0_z+=2
    static constexpr std::uint32_t unpack_srca_set_dvalid = TT_OP_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    static constexpr std::uint32_t unpack_srcb =
        TT_OP_UNPACR(SrcB, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_inc_z_0 =
        TT_OP_UNPACR(SrcB, 0b0 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_set_dvalid = TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    static constexpr std::uint32_t srca_set_z_1           = TT_OP_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 1, 0b0001); // set srcA ch0_z = 1
    static constexpr std::uint32_t srcb_set_z_2           = TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 2, 0b0001); // set srcB ch0_z = 2
    static constexpr std::uint32_t srcb_clear_z           = TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0001); // set srcB ch0_z = 0

    if (should_unpack_to_dest(unpack_to_dest, unpack_src_format, unpack_dst_format))
    {
        if (transpose_of_faces && num_faces == 4)
        {
            const std::uint32_t outerloop = 2;
            const std::uint32_t innerloop = 2;
            ckernel_template tmp(outerloop, innerloop, unpack_srca_to_dest_transpose_of_faces);
            tmp.set_end_op(TT_OP_SETADCZW(p_setadc::UNP_A, 0, 2, 0, 1, 0b0101));
            tmp.program();
        }
        else if (BType == BroadcastType::ROW || BType == BroadcastType::SCALAR)
        {
            constexpr std::uint32_t outerloop = BType == BroadcastType::ROW ? 2 : 1;
            constexpr std::uint32_t innerloop = 1;
            ckernel_template tmp(outerloop, innerloop, unpack_srca_to_dest);
            tmp.program();
        }
        else if (BType == BroadcastType::COL)
        {
            constexpr std::uint32_t outerloop = 2;
            constexpr std::uint32_t innerloop = 1;
            ckernel_template tmp(outerloop, innerloop, unpack_srca_to_dest_column);
            tmp.program();
        }
        else
        {
            const std::uint32_t outerloop     = num_faces;
            constexpr std::uint32_t innerloop = 1;
            ckernel_template tmp(outerloop, innerloop, unpack_srca_to_dest);
            tmp.program();
        }
    }
    else if constexpr (BType == BroadcastType::COL)
    {
        if constexpr (acc_to_dest)
        {
            // Use unpacker-bank readiness for the dummy SrcA publication so unpack can prepare
            // the next bank while math consumes the current bank.
            static constexpr std::uint32_t unpack_srca_reuse = (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
                                                                   ? llk_unpack_a_detail::dest_reuse_dummy_unpack<EltwiseBinaryReuseDestType::DEST_TO_SRCA>()
                                                                   : unpack_srca_set_dvalid;

            constexpr std::uint32_t innerloop = 1;
            constexpr std::uint32_t outerloop = 2; // TODO: add support for num_faces, add support for dest to srcB
            ckernel_template tmp(outerloop, innerloop, unpack_srca_reuse, unpack_srca_reuse);
            tmp.set_start_op(unpack_srcb);
            tmp.set_end_op(srcb_set_z_2);
            tmp.program();
        }
        else
        {
            constexpr std::uint32_t innerloop = 1;
            constexpr std::uint32_t outerloop = 1; // TODO: add support for num_faces
            ckernel_template tmp(outerloop, innerloop, unpack_srcb, srcb_set_z_2);
            tmp.set_start_op(unpack_srca_set_dvalid);
            tmp.set_end_op(unpack_srcb);
            tmp.program();
        }
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        const std::uint32_t outerloop = tensor_shape.num_faces_r_dim;
        const std::uint32_t innerloop = tensor_shape.num_faces_c_dim;
        if constexpr (acc_to_dest)
        {
            // Publish one dummy SrcA DVALID per tensor face, using unpacker-bank readiness so
            // unpack can prepare the next bank while math consumes the current bank.
            static constexpr std::uint32_t unpack_srca_reuse = (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
                                                                   ? llk_unpack_a_detail::dest_reuse_dummy_unpack<EltwiseBinaryReuseDestType::DEST_TO_SRCA>()
                                                                   : unpack_srca_set_dvalid;
            ckernel_template tmp(outerloop, innerloop, unpack_srcb, unpack_srca_reuse);
            tmp.set_end_op(srcb_clear_z);
            tmp.program();
        }
        else
        {
            ckernel_template tmp(outerloop, innerloop, unpack_srcb);
            tmp.set_end_op(srcb_clear_z);
            tmp.program();
        }
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        static_assert((!acc_to_dest) && "accumulate into dest with broadcast scaler is not supported!");
        constexpr std::uint32_t outerloop = 1;
        constexpr std::uint32_t innerloop = 1;
        ckernel_template tmp(outerloop, innerloop, unpack_srcb_inc_z_0);
        tmp.set_start_op(unpack_srca_set_dvalid);
        tmp.program();
    }
    else
    {
        if (transpose_of_faces)
        {
            constexpr std::uint32_t replay_buf_len = 2;
            load_replay_buf(
                0,
                replay_buf_len,
                [num_faces]
                {
                    TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
                    if (num_faces > 2)
                    {
                        TTI_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc srcA ch0_z+=2
                    }
                    else
                    {
                        TTI_UNPACR(SrcA, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc srcA ch0_z+=1
                    }
                });

            const std::uint32_t outerloop = num_faces < 4 ? 1 : 2;
            const std::uint32_t innerloop = num_faces < 2 ? 1 : 2;
            ckernel_template tmp(outerloop, innerloop, lltt::replay_insn(0, replay_buf_len)); // Unpack faces 0/2 && 1/3 to srcA
                                                                                              // or 0/1 for 2 face tile
            if (num_faces > 2)
            {
                tmp.set_end_op(srca_set_z_1);
            }
            tmp.program();
        }
        else
        {
            if constexpr (acc_to_dest)
            {
                // Use unpacker-bank readiness for destination-reuse dummy source publication so
                // unpack can prepare the next bank while math consumes the current bank.
                static constexpr std::uint32_t unpack_srca_reuse =
                    (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
                        ? llk_unpack_a_detail::dest_reuse_dummy_unpack<EltwiseBinaryReuseDestType::DEST_TO_SRCA>()
                        : unpack_srca;

                static constexpr std::uint32_t unpack_srcb_reuse =
                    (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)
                        ? llk_unpack_a_detail::dest_reuse_dummy_unpack<EltwiseBinaryReuseDestType::DEST_TO_SRCB>()
                        : unpack_srcb;

                const std::uint32_t outerloop     = num_faces;
                constexpr std::uint32_t innerloop = 1;
                ckernel_template tmp(outerloop, innerloop, unpack_srca_reuse, unpack_srcb_reuse);
                tmp.program();
            }
            else
            {
                const std::uint32_t outerloop     = num_faces;
                constexpr std::uint32_t innerloop = 1;
                ckernel_template tmp(outerloop, innerloop, unpack_srcb_set_dvalid);
                tmp.set_start_op(unpack_srca);
                tmp.program();
            }
        }
    }
}
"""

LLK_UNPACK_AB_FACE_COMPRESSED_MM = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_code_sequence.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// The unpack code sequence: one character is one instruction, recorded into the replay buffer at init (see
// The two BFP precisions need their own address counters so one instruction stream can alternate format
// face to face, and the unpacker has only one set of its own. It borrows other threads' ADC sets via the
// SETADC thread-id override: bfp2 counts in the math thread's set, bfp4 in the pack thread's. That is why
// every SETADC* site below holds mutex::THREAD2_ADC -- see ckernel_common_ops.h for the override macros.

/**
 * @brief Configure the unpack thread for a face-granular compressed matmul.
 *
 * @tparam transpose: Haloize the SrcA read, values = <true/false>
 * @param unpB_face_r_dim: Activation rows per face, 1 or 8. Sets unpacker 1's X end.
 * @note Call this before @ref _llk_unpack_AB_face_compressed_mm_, and
 *       @ref _llk_unpack_AB_face_compressed_mm_uninit_ after the last one, to put back the tile descriptor
 *       num_faces this forces to a single face.
 * @note On the math thread, pair with @ref _llk_math_face_compressed_mm_init_.
 */
template <bool transpose = false>
inline void _llk_unpack_AB_face_compressed_mm_init_(const std::uint32_t unpB_face_r_dim)
{
    LLK_ASSERT(unpB_face_r_dim == 1 || unpB_face_r_dim == 8, "face_compressed_mm (unpack): unsupported activation face_r_dim (expected 1 or 8)");

    // The config writes below need no preceding stall. The unpacker samples these fields when it issues an
    // UNPACR, not while the unpack runs, so they are free to change as soon as the previous UNPACR has been
    // issued -- which program order on this thread already guarantees.
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose ? 1 : 0);

    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Ovrd_data_format_RMW>(1); // read dataformat from per cntx registers
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_data_format_cntx0_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp2_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_out_data_format_cntx0_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp2_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_data_format_cntx1_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp4_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_out_data_format_cntx1_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp4_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_data_format_cntx2_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp2_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_out_data_format_cntx2_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp2_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_data_format_cntx3_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp4_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_out_data_format_cntx3_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp4_b));

    // Replicate onto cntx2/3 what hw_configure programs for cntx0/1, the only contexts used before. Both
    // values come from there: the x_dim through the same canonical helper, the dest address as a literal.
    // Each word packs the cntx2 value low and the cntx3 value high.
    constexpr std::uint32_t canonical_x_dim_cntx = canonical_unpA_tile_x_dim_cntx(FACE_R_DIM);
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx2_ADDR32, 0, 0xffffffff>(canonical_x_dim_cntx);
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx2_address_ADDR32, 0, 0xffffffff>(((4 * 16) << 16) | (4 * 16));

    // override z dim
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 16, 0xFF0000>(1);

    _llk_unpack_AB_face_compressed_mm_mop_config_();

    constexpr std::uint32_t unpA_x_end = FACE_R_DIM * FACE_C_DIM - 1;
    const std::uint32_t unpB_x_end     = 4 * unpB_face_r_dim * FACE_C_DIM - 1;

    t6_mutex_acquire(mutex::THREAD2_ADC);
    // the thread override covers CH0 XY and CH1 X, CH0 ZW stays shared
    // reset the borrowed counters, bfp2's in the math set and bfp4's in the pack set
    TTI_SETADCXY_THREAD_OVERRIDE(p_setadc::UNP_AB, p_setadc::THREAD_OVRD_MATH, 0, 0, 0, 0, SETADC_CH01(p_setadc::XY));
    TTI_SETADCXY_THREAD_OVERRIDE(p_setadc::UNP_AB, p_setadc::THREAD_OVRD_PACK, 0, 0, 0, 0, SETADC_CH01(p_setadc::XY));
    TTI_SETADCXY(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::XY)); // unp1's own set, unp0's rides along
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW)); // ZW has no override, both formats share it
    // set CH1 X for both unpackers; bfp4 and bfp2 sit in different thread overrides, so both get set
    TTI_SETADC_THREAD_OVERRIDE(p_setadc::UNP0, p_setadc::CH_1, p_setadc::SET_X, p_setadc::THREAD_OVRD_MATH, unpA_x_end);
    TTI_SETADC_THREAD_OVERRIDE(p_setadc::UNP0, p_setadc::CH_1, p_setadc::SET_X, p_setadc::THREAD_OVRD_PACK, unpA_x_end);
    TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);
    t6_mutex_release(mutex::THREAD2_ADC);
}
"""

LLK_UNPACK_AB_MATMUL = r"""
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * @brief Program the unpacker MOP/replay buffer for a matmul operand unpack.
 *
 * Builds a replay buffer that unpacks the streamed (non-reused) operand and advances its L1 base
 * address by one tile each step. Which operand is reused versus streamed is chosen by comparing
 * ct_dim and rt_dim (reuse_a = ct_dim >= rt_dim); operand A maps to SrcB and operand B to SrcA.
 * The address bump is suppressed under kernel broadcast.
 *
 * @tparam kernel_broadcast_a: Tile count to wrap operand A around for kernel broadcast (0 = disabled).
 * @tparam kernel_broadcast_b: Tile count to wrap operand B around for kernel broadcast (0 = disabled).
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param unpA_partial_face: Whether operand A is unpacked face-by-face (partial faces).
 * @param unpB_partial_face: Whether operand B is unpacked face-by-face (partial faces).
 */
template <std::uint32_t kernel_broadcast_a = 0, std::uint32_t kernel_broadcast_b = 0>
inline void _llk_unpack_AB_matmul_mop_config_(
    const std::uint32_t ct_dim, const std::uint32_t rt_dim, const bool unpA_partial_face, const bool unpB_partial_face)
{
    // in0/inA - loaded to SrcB
    // in1/inB - loaded to SrcA

    const bool reuse_a                      = ct_dim >= rt_dim;
    const std::uint32_t replay_buf_prog_len = (reuse_a && unpA_partial_face) ? 18 : ((!reuse_a && unpB_partial_face) ? 18 : 12);
    const std::uint32_t replay_buf_run_len  = replay_buf_prog_len / 2;

    if (reuse_a)
    {
        static_assert(kernel_broadcast_b <= 1, "kernel_broadcast>1 on matmul input 1 is not supported with reuse enabled");
        load_replay_buf(
            0,
            replay_buf_prog_len,
            // Lambda function to set up replay buffer
            [unpA_partial_face]
            {
                if (unpA_partial_face)
                {
                    TTI_UNPACR_NOP(SrcA, 0, 0, 0 /*Set Dvalid*/, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
                    TTI_UNPACR(
                        SrcA, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                    TTI_UNPACR(
                        SrcA, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                    TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
                }
                else
                {
                    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                }
                if constexpr (kernel_broadcast_b == 1)
                {
                    TTI_NOP;
                    TTI_NOP;
                    TTI_NOP;
                    TTI_NOP;
                }
                else
                {
                    TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_address_ADDR32);
                    TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
                    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
                    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
                }
                // Added to ensure WRCFG instruction has finished, since it takes 2 cycles.
                TTI_NOP;

                if (unpA_partial_face)
                {
                    TTI_UNPACR_NOP(SrcA, 0, 0, 0 /*Set Dvalid*/, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
                    TTI_UNPACR(
                        SrcA, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                    TTI_UNPACR(
                        SrcA, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                    TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
                }
                else
                {
                    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                }
                if constexpr (kernel_broadcast_b == 1)
                {
                    TTI_NOP;
                    TTI_NOP;
                    TTI_NOP;
                    TTI_NOP;
                }
                else
                {
                    TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
                    TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
                    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
                    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
                }
                // Added to ensure WRCFG instruction has finished, since it takes 2 cycles.
                TTI_NOP;
            });
    }
    else
    {
        static_assert(kernel_broadcast_a <= 1, "kernel_broadcast>1 on matmul input 0 is not supported with reuse enabled");
        load_replay_buf(
            0,
            replay_buf_prog_len,
            // Lambda function to set up replay buffer
            [unpB_partial_face]
            {
                if (unpB_partial_face)
                {
                    TTI_UNPACR_NOP(SrcB, 0, 0, 0 /*Set Dvalid*/, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
                    TTI_UNPACR(
                        SrcB, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                    TTI_UNPACR(
                        SrcB, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                    TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
                }
                else
                {
                    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                }
                if constexpr (kernel_broadcast_a == 1)
                {
                    TTI_NOP;
                    TTI_NOP;
                    TTI_NOP;
                    TTI_NOP;
                }
                else
                {
                    TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC1_REG3_Base_address_ADDR32);
                    TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP_LO);
                    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
                    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC1_REG3_Base_address_ADDR32);
                }
                // Added to ensure WRCFG instruction has finished, since it takes 2 cycles.
                TTI_NOP;

                if (unpB_partial_face)
                {
                    TTI_UNPACR_NOP(SrcB, 0, 0, 0 /*Set Dvalid*/, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
                    TTI_UNPACR(
                        SrcB, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                    TTI_UNPACR(
                        SrcB, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                    TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
                }
                else
                {
                    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                }
                if constexpr (kernel_broadcast_a == 1)
                {
                    TTI_NOP;
                    TTI_NOP;
                    TTI_NOP;
                    TTI_NOP;
                }
                else
                {
                    TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
                    TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP_LO);
                    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
                    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
                }
                // Added to ensure WRCFG instruction has finished, since it takes 2 cycles.
                TTI_NOP;
            });
    }

    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,                                    // src B
        false,                                    // halo - just used for 4 unpacks
        lltt::replay_insn(0, replay_buf_run_len), // runs when context is 0
        0,
        0,
        0,
        lltt::replay_insn(replay_buf_run_len, replay_buf_run_len), // runs when context is 1
        0,
        0);

    tmp.program();
}
"""

LLK_UNPACK_AB_REDUCE_CUSTOM = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::unpacker;
/**
 * Initializes unpacker configuration for block-based reduce_max_row operations.
 * Sets up tile dimensions and saves unpacker state that will be modified during operation.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile is num_faces faces (4 for a 32x32 tile, 2 for a 16x32 tiny tile)
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * @param tensor_shape Shape of the operand tile (4 faces for 32x32, 2 faces for a 16x32 tiny tile).
 *
 * This function should NOT be used as a substitute for native reduce unpacking LLK initialization.
 * Use the standard _llk_unpack_AB_reduce_init_ for general-purpose reduction operations.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false, bool respect_trigger = false>
inline void _llk_unpack_AB_reduce_block_max_row_init_(const ckernel::TensorShape tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    if constexpr (is_fp32_dest_acc_en)
    {
        // Set necessary config regs for MOVB2D hi16/lo16 to work
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
    }
    // REDUCE_ROW requires transpose itself; additionally, within_face_16x16_transpose flag could require transpose;
    // if we have the flag set with REDUCE_ROW, we don't need to do anything
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(1);

    TTI_SETADCXX(p_setadc::UNP_B, FACE_R_DIM * FACE_C_DIM - 1, 0x0);         // Unpack a single face of a scaler
    TT_SETADCXX(p_setadc::UNP_A, tensor_shape.total_tensor_size() - 1, 0x0); // Unpack all faces of an operand

    // save the following state that is going to be modified:
    // tile x, y, and z dims for both unpackers
    // CH1 Z stride for both unpackers
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);

    TTI_SETDMAREG(0, 1 * FACE_C_DIM * FACE_R_DIM, 0, LO_16(p_gpr_unpack::TMP0));
    TTI_SETDMAREG(0, 1 * FACE_C_DIM * FACE_R_DIM, 0, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    // y_dim = number of faces (4 for a 32x32 tile, 2 for a 16x32 tiny tile). Hardcoding 4 makes the
    // unpacker stride a full 32x32 tile between block tiles, over-reading the next tile (out-of-bounds
    // / wrong tile) for a genuine 16x32 operand.
    if (tensor_shape.num_faces_r_dim == 1)
    {
        TTI_SETDMAREG(0, 2 /* y_dim: 2 faces (16x32 tiny tile) */, 0, LO_16(p_gpr_unpack::TMP0));
    }
    else
    {
        TTI_SETDMAREG(0, 4 /* y_dim: 4 faces (32x32 tile) */, 0, LO_16(p_gpr_unpack::TMP0));
    }
    TTI_SETDMAREG(0, 1 /* z_dim */, 0, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);

    _llk_unpack_AB_reduce_block_max_row_mop_config_<block_ct_dim, respect_trigger>(); // Unpack operand and scaler
}
/**
 * Uninitializes block-based reduce_max_row unpacker operation.
 * Restores the unpacker state that was saved during initialization.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for native reduce unpacking cleanup.
 * Standard _llk_unpack_AB_reduce_init_ operations typically don't require explicit cleanup.
 */
template <bool respect_trigger = false>
inline void _llk_unpack_AB_reduce_block_max_row_uninit_()
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);

    if constexpr (respect_trigger)
    {
        t6_semaphore_get(semaphore::FPU_SFPU);
    }
}
"""

LLK_UNPACK_A_CUSTOM = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

inline void _llk_unpack_A_custom_(const std::uint32_t address)
{
    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");
    // Clear z/w start counters
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Should be propagated to the unpacker as parameter if we wish to use this LLK standalone
    // cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(1); // Disable haloize mode

    // Program srcA base address
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    if (unp_cfg_context == 0)
    {
        TT_CFG_STORE(cfg, THCON_SEC0_REG3_Base_address_ADDR32, address);
    }
    else
    {
        TT_CFG_STORE(cfg, THCON_SEC0_REG3_Base_cntx1_address_ADDR32, address);
    }

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    TTI_SETADCXX(p_setadc::UNP_A, 1023, 0x0);
    TTI_UNPACR(SrcA, 0b0 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}

"""

LLK_UNPACK_COMMON = r"""
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_memory_checks.h"

using namespace ckernel;
using namespace ckernel::unpacker;
/**
 * @brief Reprogram only the SrcA unpacker tile size and face geometry, leaving the data format untouched.
 *
 * Updates the SrcA tile-size GPR and programs the per-context X-dim (face width) and Z-dim (num_faces).
 * This is the shape half of @ref _llk_unpack_reconfig_data_format_srca_impl_, split out so a caller that
 * changes only the tile shape (not the format) can reprogram it without the THCON format writes or the
 * int8/unsigned RMW. The ch1 Z/Y strides are format-derived and are NOT touched here (they are owned by the
 * format reconfig); a shape-only change does not alter them.
 *
 * @tparam issue_stall: Issue the STALL_CFG/UNPACK0 drain before the config writes. Leave true for standalone
 *                      callers; @ref _llk_unpack_reconfig_data_format_srca_impl_ passes false because it has
 *                      already stalled for its format writes.
 * @param tile_size: New tile size (bytes) of operand A, stored to the tile-size GPR (tracks the shape).
 * @param unpack_face_r_dim: Rows per face.
 * @param unpack_num_faces: Number of faces, valid values = <1, 2, 4>.
 * @note A face_r_dim change also needs the op init re-run: the unpacker ADC X-end is programmed by the op
 *       init (config_unpacker_x_end / SETADCXX), not here. A num_faces-only change is fully handled here.
 */
template <bool issue_stall = true>
inline void _llk_unpack_reconfig_tile_shape_srca_(
    const std::uint32_t tile_size, const std::uint32_t unpack_face_r_dim = FACE_R_DIM, const std::uint32_t unpack_num_faces = 4)
{
    LLK_ASSERT(unpack_num_faces == 1 || unpack_num_faces == 2 || unpack_num_faces == 4, "unpack_num_faces must be 1, 2, or 4");

    if constexpr (issue_stall)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    }

    // Tile size (bytes) tracks the tile shape; refresh the GPR the unpack MOP uses to step the SrcA L1 base.
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));

    // Program unpacker0 per context x_dim (face size in l1)
    // Overrides value set by tile descriptor when thread override bit is set in unpack instruction
    const std::uint32_t face_dim = unpack_face_r_dim * FACE_C_DIM;
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(face_dim | (face_dim << 16));

    // Set Z-dim to number of faces
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 0, TILE_DESC_UPPER_HALFWORD_MASK>(0 | (unpack_num_faces << 16));
}
/**
 * @brief Reprogram only the SrcB unpacker tile size and face geometry, leaving the data format untouched.
 *
 * Updates the SrcB tile-size GPR and programs the tile-descriptor X-dim (face width) and Z-dim (num_faces).
 * Shape half of @ref _llk_unpack_reconfig_data_format_srcb_impl_, split out so a caller that changes only the
 * tile shape can reprogram it without the THCON format writes or the int8/unsigned RMW. The ch1 Z-stride is
 * format-derived and is NOT touched here (owned by the format reconfig).
 *
 * @tparam issue_stall: Issue the STALL_CFG/UNPACK1 drain before the config writes. Leave true for standalone
 *                      callers; @ref _llk_unpack_reconfig_data_format_srcb_impl_ passes false because it has
 *                      already stalled for its format writes.
 * @param tile_size: New tile size (bytes) of operand B, stored to the tile-size GPR (tracks the shape).
 * @param unpack_face_r_dim: Rows per face.
 * @param unpack_num_faces: Number of faces, valid values = <1, 2, 4>.
 * @note See @ref _llk_unpack_reconfig_tile_shape_srca_: a face_r_dim change also needs the op init re-run.
 */
template <bool issue_stall = true>
inline void _llk_unpack_reconfig_tile_shape_srcb_(
    const std::uint32_t tile_size, const std::uint32_t unpack_face_r_dim = FACE_R_DIM, const std::uint32_t unpack_num_faces = 4)
{
    LLK_ASSERT(unpack_num_faces == 1 || unpack_num_faces == 2 || unpack_num_faces == 4, "unpack_num_faces must be 1, 2, or 4");

    if constexpr (issue_stall)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK1);
    }

    // Tile size (bytes) tracks the tile shape; refresh the GPR the unpack MOP uses to step the SrcB L1 base.
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));

    // Set X-dim to face_r_dim * FACE_C_DIM
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, TILE_DESC_UPPER_HALFWORD_MASK>((unpack_face_r_dim * FACE_C_DIM) << 16);

    // Set Z-dim to number of faces
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32 + 1, 0, TILE_DESC_UPPER_HALFWORD_MASK>(0 | (unpack_num_faces << 16));
}
/**
 * @brief Reconfigure the operand A (SrcA) data format at runtime.
 *
 * Updates the SrcA tile-descriptor source format, FP8 exponent mode, destination format and
 * tile-size GPR; optionally reprograms dim/stride registers for face-row-major unpacking.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @tparam dim_stride_target: Whether to reprogram dim/stride, values = <IGNORE/FACE_ROW_MAJOR>
 * @tparam skip_int8: Skip re-deriving the SrcA signedness bit from the new format. Leave false (the default)
 *                    unless the caller guarantees the reconfig never crosses an Int8/UInt8/Int32 boundary and
 *                    wants to avoid the extra RMW; skipping it can leave a stale SrcAUnsigned bit (tt-metal#34499).
 * @param unpack_src_format: New source data format of operand A in L1.
 * @param unpack_dst_format: New destination data format operand A is converted to.
 * @param tile_size: New tile size of operand A stored to the tile-size GPR.
 * @param unpack_face_r_dim: Rows per face, used when reprogramming dim/stride.
 * @param unpack_num_faces: Number of faces, valid values = <1, 2, 4>.
 * @note The SrcA-unsigned ALU bit (ALU_FORMAT_SPEC_REG0_SrcAUnsigned), and the math-side INT8 math-enable in
 *       @ref _llk_math_reconfig_data_format_srca_, are re-derived from the new format by default, so a reconfig
 *       that transitions to OR from an 8-bit-integer / Int8 / Int32 format (e.g. UInt8 -> float) is handled
 *       automatically. Set skip_int8 = true only when the caller guarantees no such boundary is crossed;
 *       skipping leaves the previous unsigned/INT8 state, which the next op would then misinterpret.
 */
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool skip_int8 = false>
inline void _llk_unpack_reconfig_data_format_srca_impl_(
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t unpack_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpack_num_faces  = 4)
{
    LLK_ASSERT(unpack_num_faces == 1 || unpack_num_faces == 2 || unpack_num_faces == 4, "unpack_num_faces must be 1, 2, or 4");

    LLK_ASSERT(
        is_unpacker_format_conversion_supported_fp32_acc(
            static_cast<DataFormat>(unpack_src_format), static_cast<DataFormat>(unpack_dst_format), is_fp32_dest_acc_en),
        "Unsupported unpacker to register conversion.");

    // Drains only the config writes up to the scope end at the end of this function.
#if __has_builtin(__builtin_rvtt_stallwait_scoped)
    __instrn_buffer[0] = __builtin_rvtt_stallwait_scoped((unsigned)(p_stall::STALL_CFG), (unsigned)(p_stall::UNPACK0));
#else
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
#endif
    if constexpr (!skip_int8)
    {
        // Always re-derive SrcAUnsigned from the new format so a reconfig cannot leave a stale signedness bit
        // (tt-metal#34499). For non-int8 formats this simply writes 0, which is exactly what a reconfig away
        // from UInt8 needs. FP32 dest is required only when the new register (dst) format drives int8/int32 math,
        // so key the assertion on unpack_dst_format (matching math-side reconfig), not the L1 source: a supported
        // dequant like Int8->Float16_b needs no FP32 dest.
        LLK_ASSERT(
            is_fp32_dest_acc_en || !is_int8_or_int32_format(unpack_dst_format),
            "Reconfiguring unpack to/from Int8/UInt8/Int32 formats requires FP32 Dest mode enabled");
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcAUnsigned_RMW>((unpack_src_format == to_underlying(DataFormat::UInt8)) ? 1 : 0);
    }

    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(unpack_src_format);

    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Unp_LF8_4b_exp_RMW>(((unpack_src_format & 0x1F) == (std::uint32_t)DataFormat::Fp8_e4m3) ? 1 : 0);

    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(unpack_dst_format);
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A)); // update gpr which holds tile size A

    // The ch1 (register-side) Z/Y strides are format-derived (datum size), so re-commit them on EVERY format
    // change, independent of dim_stride_target. They drive partial-face unpack addressing (e.g. partial-face
    // matmul, which reads them without setting them), so leaving them stale across a datum-size change (e.g.
    // fp16 -> fp32) mislands faces (tt-llk#1161 follow-up). Per-op brackets (unpack-to-dest, bcastA_B) restore
    // to this same canonical baseline, so re-committing it here is what keeps those restores correct.
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(canonical_unpA_z_stride(unpack_dst_format));
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32, UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT, UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(
        canonical_unpA_y_stride(unpack_dst_format));

    if constexpr (dim_stride_target == p_dim_stride_target::FACE_ROW_MAJOR)
    {
        // Reprogram tile size + face geometry. issue_stall=false: the STALL_CFG/UNPACK0 above already drained.
        // (tile_size is re-written here; it is the same value stored just above -- a harmless duplicate.)
        _llk_unpack_reconfig_tile_shape_srca_<false /*issue_stall*/>(tile_size, unpack_face_r_dim, unpack_num_faces);
    }
#if __has_builtin(__builtin_rvtt_stallwait_scoped)
    __builtin_rvtt_stall_scope_end();
#endif
}
/**
 * @brief Validates L1 addresses and configures unpack base addresses in the configuration registers
 *
 * This helper function validates that both address_a and address_b are within the valid L1 memory region,
 * then configures the appropriate THCON base address registers based on the unpack configuration context.
 *
 * @param address_a: Address for unpacker A (THCON_SEC0).
 * @param address_b: Address for unpacker B (THCON_SEC1).
 * @param cfg: Pointer to configuration registers.
 */
inline void _llk_unpack_configure_addresses_(const std::uint32_t address_a, const std::uint32_t address_b, volatile std::uint32_t tt_reg_ptr *cfg)
{
    LLK_ASSERT(is_valid_L1_address(address_a), "L1 address_a must be in valid L1 memory region");
    LLK_ASSERT(is_valid_L1_address(address_b), "L1 address_b must be in valid L1 memory region");

    if (0 == unp_cfg_context)
    {
        TT_CFG_STORE(cfg, THCON_SEC0_REG3_Base_address_ADDR32, address_a);
        TT_CFG_STORE(cfg, THCON_SEC1_REG3_Base_address_ADDR32, address_b);
    }
    else
    {
        TT_CFG_STORE(cfg, THCON_SEC0_REG3_Base_cntx1_address_ADDR32, address_a);
        TT_CFG_STORE(cfg, THCON_SEC1_REG3_Base_cntx1_address_ADDR32, address_b);
    }
}
"""

LLK_UNPACK_COMMON_API = r"""
// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sanitizer/api.h"
#include "internal/circular_buffer_interface.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "api/debug/waypoint.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_operands.h"
#include "llk_param_structs.h"
#include "llk_assert.h"
#include "llk_fp32_dest_acc.h"
#include "llk_unpack_common.h"

/*************************************************************************
 * LLK UNPACK COMMON
 *************************************************************************/

/**
 * Configure the unpacker hardware for operands A and B.
 *
 * Face geometry (face_r_dim, num_faces) and tile size for both operands are
 * derived from the CB metadata associated with each operand id. This is the
 * primary entry point: callers no longer need to thread face geometry through
 * the API, since per-CB face geometry is recorded in the CB descriptor at
 * program creation time.
 *
 * @tparam is_fp32_dest_acc_en   Enable FP32 accumulation in the destination register.
 * @param  unpA_operand          Operand index for unpack source A (In0).
 * @param  unpB_operand          Operand index for unpack source B (In1).
 */
/**
 * Reconfigure the srcA unpacker for a new operand's data format.
 *
 * Face geometry (face_r_dim, num_faces) and tile size are derived from the new operand's CB metadata.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam dim_stride_target   Dimension/stride programming target for the unpacker.
 * @tparam skip_int8           Skip re-deriving the SrcUnsigned bit from the new format; leave false unless the
 *                             caller guarantees no Int8/UInt8/Int32 boundary is crossed (tt-metal#34499).
 * @param  srca_new_operand    New operand id to configure srcA for.
 */
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool skip_int8 = false>
inline void llk_unpack_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    const std::uint32_t srca_operand_id = get_operand_id(srca_new_operand);
    const std::uint32_t num_faces = get_operand_num_faces(srca_operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(srca_operand_id);

    // Currently, there is a constraint that tile size is equal to the fifo page size
    // TODO NC: tile size should be computed in the LLK instead, as the part of #34495
    const std::uint32_t tile_size = get_local_cb_interface(srca_operand_id).fifo_page_size;
    SAN_HOOK(reconfigure(
        StateVal<Operand<Exu::Unpack>::InputFormatA>(unpack_src_format[srca_operand_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatA>(unpack_dst_format[srca_operand_id]),
        StateDiscard<std::uint32_t>(tile_size),
        StateDiscard<std::uint32_t>(face_r_dim),
        StateDiscard<std::uint32_t>(num_faces)));
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, dim_stride_target, skip_int8>(
        unpack_src_format[srca_operand_id], unpack_dst_format[srca_operand_id], tile_size, face_r_dim, num_faces);
}
"""

LLK_UNPACK_FAST_TILIZE = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_unpack_tilize.h"

// ============================================================================
// BH Fast-Tilize Unpack
//
// Software tilization via UNPACR address modes. Processes 4 tiles at a time
// (unit_dim=4). Each UNPACR reads 128 datums (4 tile widths) into 8 SrcA rows.
// CH1_Z stride = 256 bytes (8 contiguous SrcA rows per read).
// MASK_LOOP MOP with zmask=0x80808080 fires dvalid every 8th read
// (32 reads total, 4 dvalids per unit).
// ============================================================================

// BH fast-tilize: block height is always 1 (one row of tiles per call).
// Multiple rows are handled by the caller, looping over rows and calling
// the block function once per chunk per row.
//
// init_unit_dim: unit_dim of the first chunk (= decompose_row(ct_dim)[0]).
// Initialising X to the first chunk's width avoids one reinit_xdim call per row.
// Formula: ct_dim > 5 ? 4 : ct_dim == 5 ? 2 : ct_dim.
inline void _llk_unpack_fast_tilize_init_(const std::uint32_t unpack_dst_format, const std::uint32_t ct_dim, const std::uint32_t init_unit_dim)
{
    // Context-safe writes only: Tile_x_dim (WRCFG below writes the full 32-bit word,
    // covering cntx0 low-16 and cntx1 high-16), TileDescriptor (shared across contexts),
    // Zstride (RMW on shared reg), and SETADCXX (thread-scoped counter, not per
    // cfg context - see ISA SETADCXX). Per-call context switching happens in
    // _llk_unpack_fast_tilize_block_.

    // Save state
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_2, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);
    // Save the unpacker Out_data_format word so we can restore it in uninit if the
    // fp32/tf32 -> bf16 downgrade below modifies it.
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_3, THCON_SEC0_REG2_Out_data_format_ADDR32);

    // BH fast-tilize forces a 16-bit DEST view in the math thread (MOVA2D cannot
    // safely write Dst32b in this flow - see _llk_math_fast_tilize_init_). That makes
    // TF32/Float32 SrcA incompatible: MOVA2D(TF32) + Dst16b is ISA UB, and MOVA2D(bf16)
    // + Dst32b is also UB. The only consistent combination is bf16 SrcA + Dst16b, so
    // downgrade SrcA output to Float16_b here when the caller requested fp32/tf32.
    // The unpacker performs the fp32 -> bf16 conversion on the L1 -> SrcA path. This
    // matches the precision Metal previously consumed from the fp32 fast-tilize path.
    const std::uint32_t effective_dst_format =
        (unpack_dst_format == static_cast<std::uint32_t>(DataFormat::Float32) || unpack_dst_format == static_cast<std::uint32_t>(DataFormat::Tf32))
            ? static_cast<std::uint32_t>(DataFormat::Float16_b)
            : unpack_dst_format;
    if (effective_dst_format != unpack_dst_format)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(effective_dst_format);
    }

    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    // Tile_x_dim = 32, Tile_y_dim = ct_dim, Tile_z_dim = 16
    TT_SETDMAREG(0, TILE_C_DIM, 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, TILE_C_DIM, 0, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);

    TT_SETDMAREG(0, ct_dim, 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, FACE_R_DIM, 0, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);

    // BH HW bug: TileDescriptor words 1-3 (YDim, ZDim) are not tracked as unpacker
    // resources. Explicit stall ensures the write completes.
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);

    // X counter end = first chunk's tile width. CH1_Z stride stays at 4-wide
    // (8 SrcA rows) regardless of unit_dim - natural gaps in SrcA for unit_dim < 4.
    // Matching init_unit_dim to the first chunk eliminates one reinit_xdim per row.
    TT_SETADCXX(p_setadc::UNP_A, init_unit_dim * TILE_C_DIM - 1, 0x0);

    // CH1 Z stride: controls SrcA dest address gap between reads.
    // Uses effective_dst_format because Float32/Tf32 are downgraded to bf16 above.
    const std::uint32_t ch1_x_stride =
        (effective_dst_format == static_cast<std::uint32_t>(DataFormat::Float32) || effective_dst_format == static_cast<std::uint32_t>(DataFormat::Int32) ||
         effective_dst_format == static_cast<std::uint32_t>(DataFormat::Tf32))
            ? 4
            : 2;
    // stride = 4 * 32 * 2 = 256 bytes = 8 contiguous SrcA rows per read
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(4 * TILE_C_DIM * ch1_x_stride);

    _llk_unpack_fast_tilize_mop_config_();
}
// Reconfigure X counter for a different unit_dim without full reinit.
// CH1_Z stride and MOP stay unchanged - only the read width changes.
inline void _llk_unpack_fast_tilize_reinit_xdim_(const std::uint32_t unit_dim)
{
    TT_SETADCXX(p_setadc::UNP_A, unit_dim * TILE_C_DIM - 1, 0x0);
}
"""

LLK_UNPACK_TILIZE = r"""
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "../../common/tensor_shape.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;
/**
 * @brief Program the unpacker MOP for the non-8-bit tilize path (Blackhole whole-tile workaround).
 *
 * Selects between unpacking to SrcA (with a SrcB dvalid NOP) or straight to dest. 8-bit formats
 * use the inline 2-context path in @ref _llk_unpack_tilize_ instead (no MOP).
 *
 * @param unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 */
inline void _llk_unpack_tilize_mop_config_(const bool unpack_to_dest = false)
{
    // SrcB SET_DVALID + UNP_ZEROSRC: math's ELWADD branch (FP32 dest-accumulate) reads SrcB,
    // so it must be both dvalid and zeroed. MOVA2D branch ignores it (harmless handshake).
    static constexpr std::uint32_t unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srca_to_dest =
        TT_OP_UNPACR(0, 0b00010001 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_set_dvalid = TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);

    ckernel_template tmp(1 /*outerloop*/, 1 /*innerloop*/, unpack_to_dest ? unpack_srca_to_dest : unpack_srcb_set_dvalid);

    if (!unpack_to_dest)
    {
        tmp.set_start_op(unpack_srca);
    }

    tmp.program();
}
/**
 * @brief Initialize the unpacker for a tilize operation.
 *
 * Disables face transpose, configures the unpacker into tileize mode (throttle, shift amount,
 * per-tile X/Z dims) for the given block column dimension, decides whether 32-bit datums must be
 * unpacked to dest, and:
 *   - 8-bit formats: programs the 1x16 face x_dim per context and (for num_faces==4) the
 *     SCRATCH_SEC0_val preload used by the inline CFGSHIFTMASK. No MOP is programmed —
 *     the UNPACR sequence is issued inline by @ref _llk_unpack_tilize_.
 *   - non-8-bit formats: programs the whole-tile x_dim/z_dim descriptor state and the
 *     MOP template via @ref _llk_unpack_tilize_mop_config_ (Blackhole workaround).
 *
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 * @param ct_dim: Number of column tiles in the block, used to size the column dimension.
 * @param face_r_dim: Rows per face, valid values = <2, 4, 8, 16>.
 * @param narrow_tile: Whether the tile is narrow (single column of faces). Not supported on the 8-bit path (asserted false).
 * @param num_faces: Number of faces in the tile, valid values = <2, 4>.
 * @note Call @ref _llk_unpack_tilize_uninit_ to restore the modified tile-descriptor state.
 * @ref _llk_unpack_tilize_ is the matching execute call.
 * @ref _llk_math_eltwise_unary_datacopy_init_ (A2D, PackMode::Tilize) is the matching init on the math thread.
 */
inline void _llk_unpack_tilize_init_(
    const std::uint32_t unpack_src_format = 0,
    const std::uint32_t unpack_dst_format = 0,
    const std::uint32_t ct_dim            = 0,
    const std::uint32_t face_r_dim        = FACE_R_DIM,
    const bool narrow_tile                = false,
    const std::uint32_t num_faces         = 4)
{
    LLK_ASSERT(face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16, "face_r_dim must be 1, 2, 4, 8, or 16 for tilize");
    LLK_ASSERT(num_faces == 2 || num_faces == 4, "num_faces must be 2 or 4 for tilize");
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    // Program Ch1.Z stride from unpack_dst_format's datum size, required by the
    // UNPACR AddrMode 0b00010001 used on the 8-bit tilize path.
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(canonical_unpA_z_stride(unpack_dst_format));

    // Clear Ch1.X and Ch1.Y start-counts once at init. Tilize's UNPACRs never
    // increment Ch1.X or Ch1.Y (AddrMode 0b00010001 sets only Ch0.Z++/Ch1.Z++),
    // so the counts stay at 0 through the whole tilize_block. With the counts at
    // 0, Ch1.X/Y strides multiply out of the SrcA write address (SrcA_row =
    // Ch1.Z*Zstride + Ch1.Y*Ystride + Ch1.X*Xstride + Base) and need not be
    // programmed here.
    TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1111);

    // Z_dim = num_faces, the per-operand baseline. The non-8-bit branch below overwrites it with 1,
    // and that is the write the teardown reverts.
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 16, TILE_DESC_UPPER_HALFWORD_MASK>(num_faces);

    const std::uint32_t block_c_dim = ct_dim * (narrow_tile ? FACE_C_DIM : TILE_C_DIM);

    // In case of 32-bit numbers, we have to unpack into dest register
    // For integers, always unpack to dest. For Float32, only if unpack_dst_format is Float32 (lossless tilize mode)
    auto unpack_source_format = static_cast<DataFormat>(unpack_src_format);
    auto unpack_dest_format   = static_cast<DataFormat>(unpack_dst_format);

    const bool unpack_to_dest =
        (unpack_source_format == DataFormat::UInt32) || (unpack_source_format == DataFormat::Int32) || (unpack_dest_format == DataFormat::Float32);

    LLK_ASSERT(
        is_unpacker_format_conversion_supported_dest(static_cast<DataFormat>(unpack_src_format), static_cast<DataFormat>(unpack_dst_format), unpack_to_dest),
        "Unsupported unpacker format conversion.");

    // Set face dim
    TT_SETADCXX(p_setadc::UNP_A, face_r_dim * FACE_C_DIM - 1, 0x0);

    const bool is_8bit_format        = IS_8BIT_FORMAT(unpack_src_format);
    const std::uint32_t shift_amount = (SCALE_DATUM_SIZE(unpack_src_format, block_c_dim)) >> 4;

    // Override default settings to enable tilize mode
    unpack_config_u config   = {0};
    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    config.f.tileize_mode    = 1;
    config.f.shift_amount    = shift_amount;

    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG2_Out_data_format_ADDR32);
    if (is_8bit_format)
    {
        LLK_ASSERT(!narrow_tile, "8-bit tilize narrow_tile not supported");
        LLK_ASSERT(!unpack_to_dest, "8-bit tilize unpack_to_dest not supported");

        // x_dim per context = FACE_C_DIM (1x16 face read). FACE_DIM_1x16 holds
        // FACE_C_DIM packed in both halfwords, so a 32b WRCFG programs both cntx0 and cntx1.
        TTI_WRCFG(p_gpr_unpack::FACE_DIM_1x16, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);

        // SCRATCH_SEC0_val is the increment CFGSHIFTMASK adds to Base_address between
        // top and bottom face pairs. Only used when num_faces==4 (top pair to bottom pair).
        // For num_faces==2 (top pair only) no CFGSHIFTMASK fires, so the preload is skipped.
        //   bot_face_offset = shift_amount * face_r_dim       (in 16B words, jump from top to bottom faces)
        //   -2 compensates for Z=2 L1 offset (Z * XDim * DatumSize = 2 * 16 * 1 = 32 bytes = 2 16B-words)
        if (num_faces == 4)
        {
            const std::uint32_t base_address_increment = shift_amount * face_r_dim - 2;
            TT_SETDMAREG(0, LOWER_HALFWORD(base_address_increment), 0, LO_16(p_gpr_unpack::TMP0));
            TT_SETDMAREG(0, UPPER_HALFWORD(base_address_increment), 0, HI_16(p_gpr_unpack::TMP0));
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
            TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, SCRATCH_SEC0_val_ADDR32);
        }
    }
    else
    {
        // Whole-tile unpacking (non-8-bit BH workaround): x_dim covers the entire tile, z_dim=1
        const std::uint32_t Tile_x_dim = face_r_dim * num_faces * FACE_C_DIM;
        const std::uint32_t Tile_z_dim = 1;
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(Tile_x_dim | (Tile_x_dim << 16));
        // Set x-dim to cover entire tile (face_r_dim * num_faces * FACE_C_DIM)
        cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, TILE_DESC_UPPER_HALFWORD_MASK>(0 | (Tile_x_dim << 16));
        // Set z-dim to 1 as X dim is set to cover the entire tile, so no need to iterate over faces.
        cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 0, TILE_DESC_UPPER_HALFWORD_MASK>(0 | (Tile_z_dim << 16));

        // Set x-end for Unpackers to (face_r_dim * num_faces * FACE_C_DIM - 1)
        TT_SETADCXX(p_setadc::UNP0, Tile_x_dim - 1, 0x0);

        _llk_unpack_tilize_mop_config_(unpack_to_dest);
    }
}
/**
 * @brief Restore unpacker state after a tilize operation.
 *
 * Reverts the config @ref _llk_unpack_tilize_init_ wrote: tile-descriptor Z dim (dropped to 1 by
 * the non-8-bit init path) and X dim (written by that path only, never by the 8-bit one), unpack
 * config word 0 (clearing tilize and haloize mode) and Tile_x_dim_cntx0. x-start/x-end and the MOP
 * are reprogrammed by the next operation's init, so they are not restored here.
 *
 * @param unpack_dst_format: Destination data format to restore in the unpack config.
 * @param tensor_shape: Tile geometry; total_num_faces() restores the descriptor Z dimension
 *                      (valid values = <1, 2, 4>) and face_r_dim restores the canonical Tile_x_dim.
 * @note Call @ref _llk_unpack_tilize_init_ before this function.
 */
inline void _llk_unpack_tilize_uninit_(const std::uint32_t unpack_dst_format, const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    const std::uint32_t num_faces  = tensor_shape.total_num_faces();
    const std::uint32_t face_r_dim = tensor_shape.face_r_dim;

    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    // X-dim is 0 because Tile_x_dim_cntx0 below is what the unpacker consumes for srcA.
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 16, TILE_DESC_UPPER_HALFWORD_MASK>(num_faces);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 16, TILE_DESC_UPPER_HALFWORD_MASK>(CANONICAL_UNPA_TILE_X_DIM);

    // Zeroed config clears tileize_mode and haloize_mode along with the rest of word 0.
    unpack_config_u config = {0};

    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    // Load unpack config[0]
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG2_Out_data_format_ADDR32);
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(canonical_unpA_tile_x_dim_cntx(face_r_dim));
    TTI_NOP;
}
"""

MAIN_LLK_UNPACK_A_TOPK_XL_COPY = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// TopK-XL copy: single UNPACR (unpack_srca_to_dest encoding) per tile run for FP32 and 16-bit
// unpack-to-dest — no MOP wrapper — instead of face-by-face loops from stock _llk_unpack_A_mop_config_.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
// Minimal unpack-to-dest execute path for TopK-XL copy (one TTI_UNPACR per tile).
inline void _llk_unpack_topk_xl_copy_(
    const std::uint32_t address, const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t elements_this_tile)
{
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();

    wait_for_next_context(2);

    const std::uint32_t upk0_reg = (unp_cfg_context == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;
    cfg[upk0_reg]                = address;

    semaphore_post(semaphore::UNPACK_SYNC);

    if (elements_this_tile == 0)
    {
        // Zero-element tile: clear SrcA to -inf so math datacopy writes full -inf padding.
        // This avoids SETADC underflow behavior and avoids reading source tile data.
        TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, p_unpacr_nop::CLR_SRC_NEGINF, p_unpacr_nop::CLR_SRC);
        TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);

        t6_semaphore_get(semaphore::UNPACK_SYNC);
        switch_config_context(unp_cfg_context);
        return;
    }

    if (is_32bit_input(unpack_src_format, unpack_dst_format))
    {
        if (elements_this_tile < 1024)
        {
            // clear to -inf first for padding
            TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, p_unpacr_nop::CLR_SRC_NEGINF, p_unpacr_nop::CLR_SRC);
            TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
        }
        set_dst_write_addr(unp_cfg_context, unpack_dst_format);
        wait_for_dest_available();
    }

    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Run MOP
    ckernel::ckernel_template::run();

    t6_semaphore_get(semaphore::UNPACK_SYNC);

    if (is_32bit_input(unpack_src_format, unpack_dst_format))
    {
        unpack_to_dest_tile_done(unp_cfg_context, unpack_dst_format);
    }

    switch_config_context(unp_cfg_context);
}
"""

SDPA_WEIGHTED_REDUCE = r"""
#ifdef TRISC_UNPACK
// Raw (no-MOP) replacement for llk_unpack_AB(qk, weights, 0, 0): load tile 0 of
// qk -> SrcA (both faces) and tile 0 of weights -> SrcB. One SrcB unpack is enough
// since the MVMULs only read SrcB cols 0..7 (weights face0). Three UNPACR replace
// the standard AB template MOP, with SetDatValid deferred to the last unpack of
// each source so both srcs go valid only once fully loaded.
inline void weighted_reduce_unpack_impl(
    const std::uint32_t weights_cb, const std::uint32_t qk_cb, const std::uint32_t qk_tile_index = 0) {
    const std::uint32_t qk_id = get_operand_id(qk_cb);            // operandA -> SrcA
    const std::uint32_t weights_id = get_operand_id(weights_cb);  // operandB -> SrcB
    const std::uint32_t address_a =
        get_local_cb_interface(qk_id).fifo_rd_ptr - 1 + qk_tile_index * get_local_cb_interface(qk_id).fifo_page_size;
    const std::uint32_t address_b = get_local_cb_interface(weights_id).fifo_rd_ptr - 1;

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);  // reset both unpackers' counters
    // Wait for a free context, program SrcA/SrcB base addresses, release to unpacker.
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // qk face0 -> SrcA rows 0..15; AddrMode advances the L1 read + SrcA write to
    // face1, no dvalid yet.
    TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    // weights -> SrcB row 0; sets SrcB dvalid.
    TTI_UNPACR(SrcB, 0b00000000, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    // qk face1 -> SrcA rows 16..31; sets SrcA dvalid.
    TTI_UNPACR(SrcA, 0b00000000, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);  // reset both unpackers' counters
}
#endif
"""

#: Minimal stand-ins for headers the snippets include (not under test themselves).
STUBS = {
    f"{LLK}/common/inc/ckernel_defs.h": """
namespace ckernel
{
constexpr std::uint32_t FACE_R_DIM            = 16;
constexpr std::uint32_t FACE_C_DIM            = 16;
constexpr std::uint32_t TILE_C_DIM            = 32;
constexpr std::uint32_t TILE_NUM_FACES        = 4;
constexpr std::uint32_t DATA_FORMAT_BIT_COUNT = 4;
#define LO_16(REG) (2 * (REG))
#define HI_16(REG) (2 * (REG) + 1)
#define LOWER_HALFWORD(x) ((x) & 0xFFFF)
#define UPPER_HALFWORD(x) ((x) >> 16)
inline constexpr static std::uint32_t masked_data_format(std::uint32_t data_format) { return data_format & 0xF; }
}
""",
    f"{LLK}/common/inc/ckernel_globals.h": """
extern std::uint32_t unp_cfg_context;
""",
    f"{LLK}/common/inc/ckernel.h": """
namespace ckernel
{
template <std::uint32_t CfgAddr32, std::uint32_t Shamt, std::uint32_t Mask>
inline void cfg_reg_rmw_tensix(std::uint32_t val)
{
    TT_RMWCIB0(Mask & 0xff, val << Shamt, CfgAddr32);
}
inline std::uint32_t semaphore_read(const std::uint32_t index) { return pc_buf_base[index]; }
inline void semaphore_post(const std::uint32_t index) { pc_buf_base[index] = 0; }
template <typename F>
inline void load_replay_buf(std::uint32_t start, std::uint32_t len, F fn)
{
    TTI_REPLAY(start, len, 0, 1);
    fn();
}
}
""",
    f"{LLK}/common/inc/ckernel_template.h": """
namespace ckernel
{
class ckernel_template
{
public:
    void set_end_op(std::uint32_t end_op0) { m_end_op0 = end_op0; }
    void set_start_op(std::uint32_t start_op0) { m_start_op0 = start_op0; }
    void program() { mop_cfg[0] = m_outer_loop_len; }
    static void run() { TTI_MOP(1, 0, 0); }
};
class ckernel_unpack_template
{
public:
    void program() const { mop_cfg[0] = m_unpackB; }
    static void run(const std::uint8_t count, const std::uint32_t zmask) { TT_MOP(0, count - 1, zmask); }
};
}
""",
}


def build_files() -> dict[str, str]:
    return {
        **STUBS,
        "tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h": CFG_DEFINES,
        f"{LLK}/common/inc/ckernel_gpr_map.h": CKERNEL_GPR_MAP,
        f"{LLK}/common/inc/cunpack_common.h": CUNPACK_COMMON,
        f"{LLK}/llk_lib/llk_unpack_common.h": LLK_UNPACK_COMMON,
        f"{LLK}/llk_lib/llk_unpack_A.h": LLK_UNPACK_A,
        f"{LLK}/llk_lib/llk_unpack_tilize.h": LLK_UNPACK_TILIZE,
        f"{LLK}/llk_lib/llk_unpack_AB_matmul.h": LLK_UNPACK_AB_MATMUL,
        f"{LLK}/llk_lib/experimental/llk_unpack_A_custom.h": LLK_UNPACK_A_CUSTOM,
        f"{LLK}/llk_lib/experimental/llk_unpack_A_topk_xl_copy.h": MAIN_LLK_UNPACK_A_TOPK_XL_COPY,
        f"{LLK}/llk_lib/experimental/llk_unpack_AB_reduce_custom.h": LLK_UNPACK_AB_REDUCE_CUSTOM,
        f"{LLK}/llk_lib/experimental/llk_unpack_AB_face_compressed_mm.h": LLK_UNPACK_AB_FACE_COMPRESSED_MM,
        f"{LLK}/llk_lib/experimental/llk_unpack_fast_tilize.h": LLK_UNPACK_FAST_TILIZE,
        f"{API}/llk_unpack_common_api.h": LLK_UNPACK_COMMON_API,
        "tt_metal/hw/inc/api/compute/experimental/sdpa_weighted_reduce.h": SDPA_WEIGHTED_REDUCE,
    }


def write_tree(root: Path) -> Path:
    for relative, text in build_files().items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def line_of(relative: str, needle: str, nth: int = 0) -> int:
    """1-based line of the ``nth`` occurrence of ``needle`` in the file as written."""
    lines = build_files()[relative].split("\n")
    hits = [number for number, line in enumerate(lines, 1) if needle in line]
    return hits[nth]


SEC0_READS = [name for name, _ in unpack.unpacr_read_names(0)]
SEC1_READS = [name for name, _ in unpack.unpacr_read_names(1)]


class UnpackFamilyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = write_tree(Path(cls._tmp.name))
        cls.db = effects_db.build(cls.root, families=["unpack"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # ---- helpers -------------------------------------------------------- #
    def fn(self, name: str):
        (model,) = self.db.by_name(name)
        return model

    def mine(self, name: str, pattern: str | None = None):
        """unpack-family direct effects as (pattern, resource, access, value, condition, line, detail, thread)."""
        rows = []
        for e in self.fn(name).effects:
            if e.family != "unpack" or (pattern is not None and e.pattern != pattern):
                continue
            rows.append(
                (
                    e.pattern,
                    e.resource,
                    e.access,
                    e.value,
                    e.condition,
                    int(e.evidence.rsplit(":", 1)[1]),
                    e.detail,
                    e.thread,
                )
            )
        return rows

    def fields_at(
        self,
        name: str,
        line: int,
        prefix: str,
        pattern: str = "unpack.cfg_struct_field",
    ):
        """resource -> value of the field writes of ``pattern`` at ``line`` whose resource starts with ``prefix``."""
        return {
            row[1]: row[3]
            for row in self.mine(name, pattern)
            if row[5] == line and row[1].startswith(prefix) and row[2] == "write"
        }

    def reads_at(self, name: str, line: int):
        return sorted(
            row[1] for row in self.mine(name, "unpack.unpacr.read") if row[5] == line
        )

    # ---- read set ------------------------------------------------------- #
    def test_unpacr_read_set_names_real_cfg_words(self) -> None:
        self.assertEqual(len(SEC0_READS), 23)
        self.assertEqual(len(SEC1_READS), 19)
        cfg = self.db.tables.cfg
        for name, granularity in unpack.unpacr_read_names(0) + unpack.unpacr_read_names(
            1
        ):
            if not name.startswith("CFG.") or name.endswith("+1"):
                continue
            field = cfg.fields[name[4:]]  # every CFG name is a real cfg_defines field
            if (
                granularity == "word"
            ):  # word reads are named after the word's first field
                self.assertEqual(
                    cfg.field_for_word(field.space, field.addr32), name[4:]
                )
        self.assertEqual(
            unpack._norm_unpacr_reads("unpacker")[0].name, "UNPACK.?unpacker"
        )
        self.assertFalse(unpack._norm_unpacr_reads("unpacker")[0].resolved)

    # ---- struct staging: configure_unpack_AB (cunpack_common.h) ---------- #
    def test_configure_unpack_AB_tile_descriptors_from_struct(self) -> None:
        path = f"{LLK}/common/inc/cunpack_common.h"
        sec0 = line_of(
            path, "TT_CFG_STORE(cfg, THCON_SEC0_REG0_TileDescriptor_ADDR32 + i"
        )
        sec1 = line_of(
            path, "TT_CFG_STORE(cfg, THCON_SEC1_REG0_TileDescriptor_ADDR32 + i"
        )
        self.assertEqual(
            self.fields_at("configure_unpack_AB", sec0, "UNPACK."),
            {
                "UNPACK.THCON_SEC0_TILE_DESC.in_data_format": "static_cast<std::uint32_t>(unpA_src_format_masked)",
                "UNPACK.THCON_SEC0_TILE_DESC.uncompressed": "1",
                "UNPACK.THCON_SEC0_TILE_DESC.blobs_per_xy_plane": "0",
                "UNPACK.THCON_SEC0_TILE_DESC.x_dim": "0",
                "UNPACK.THCON_SEC0_TILE_DESC.y_dim": "1",
                "UNPACK.THCON_SEC0_TILE_DESC.z_dim": "unpA_num_faces",
            },
        )
        # the SEC1 store sees the fields re-assigned between the two stores
        self.assertEqual(
            self.fields_at("configure_unpack_AB", sec1, "UNPACK."),
            {
                "UNPACK.THCON_SEC1_TILE_DESC.in_data_format": "row_pool ? to_underlying(DataFormat::Float32) : unpB_src_format_masked",
                "UNPACK.THCON_SEC1_TILE_DESC.uncompressed": "1",
                "UNPACK.THCON_SEC1_TILE_DESC.blobs_per_xy_plane": "0",
                "UNPACK.THCON_SEC1_TILE_DESC.x_dim": "unpB_face_r_dim * FACE_C_DIM",
                "UNPACK.THCON_SEC1_TILE_DESC.y_dim": "1",
                "UNPACK.THCON_SEC1_TILE_DESC.z_dim": "unpB_num_faces",
            },
        )

    def test_configure_unpack_AB_config_words_follow_param(self) -> None:
        path = f"{LLK}/common/inc/cunpack_common.h"
        sec0 = line_of(
            path, "TT_CFG_STORE(cfg, THCON_SEC0_REG2_Out_data_format_ADDR32 + i"
        )
        sec1 = line_of(
            path, "TT_CFG_STORE(cfg, THCON_SEC1_REG2_Out_data_format_ADDR32 + i"
        )
        word0 = self.fields_at("configure_unpack_AB", sec0, "CFG.THCON_SEC0_REG2_")
        self.assertEqual(
            word0["CFG.THCON_SEC0_REG2_Haloize_mode"], "transpose_xy_srca_en ? 1 : 0"
        )
        self.assertEqual(
            word0["CFG.THCON_SEC0_REG2_Out_data_format"], "unpA_dst_format_masked"
        )
        self.assertEqual(word0["CFG.THCON_SEC0_REG2_Throttle_mode"], "2")
        self.assertEqual(word0["CFG.THCON_SEC0_REG2_Tileize_mode"], "0")
        self.assertEqual(
            word0["CFG.THCON_SEC0_REG2_Ovrd_data_format"], "0"
        )  # struct reserved_1 bit
        # word 1: uncompress_cntx0_3 = 0xf covers Disable_zero_compress_cntx0..3 (bits 0-3)
        for cntx in range(8):
            self.assertEqual(
                word0[f"CFG.THCON_SEC0_REG2_Disable_zero_compress_cntx{cntx}"], "1"
            )
        self.assertEqual(word0["CFG.THCON_SEC0_REG2_Unpack_if_sel_cntx0"], "0")
        self.assertEqual(len(word0), 34)  # every field of both config words
        word1 = self.fields_at("configure_unpack_AB", sec1, "CFG.THCON_SEC1_REG2_")
        self.assertEqual(word1["CFG.THCON_SEC1_REG2_Haloize_mode"], "0")
        self.assertEqual(
            word1["CFG.THCON_SEC1_REG2_Out_data_format"],
            "row_pool ? (to_underlying(DataFormat::Float16) | (exp_width << 2)) : unpB_dst_format_masked",
        )
        # transpose flows from the caller: bound to true the field is "true ? 1 : 0"
        effects = self.db.call_effects(
            "configure_unpack_AB",
            template_args=("false",),
            args=("a", "b", "c", "d", "16", "16", "true"),
            evidence="kernel.cpp:1",
        )
        halo = {
            e.value
            for e in effects
            if e.resource == "CFG.THCON_SEC0_REG2_Haloize_mode"
            and e.pattern == "unpack.cfg_struct_field"
        }
        self.assertEqual(halo, {"true ? 1 : 0"})

    def test_configure_unpack_AB_alu_payload_and_field_companions(self) -> None:
        path = f"{LLK}/common/inc/cunpack_common.h"
        alu = line_of(
            path, "cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, alu_mask>"
        )
        self.assertEqual(
            self.fields_at("configure_unpack_AB", alu, "CFG.ALU_"),
            {  # conditional struct assignments become (cond) ? v : prev; the mask limits the fields
                "CFG.ALU_FORMAT_SPEC_REG0_SrcAUnsigned": "(unpA_src_format == to_underlying(DataFormat::UInt8)) ? (1) : (0)",
                "CFG.ALU_FORMAT_SPEC_REG0_SrcBUnsigned": "(unpB_src_format == to_underlying(DataFormat::UInt8)) ? (1) : (0)",
                "CFG.ALU_ROUNDING_MODE_Fpu_srnd_en": "fpu_srnd_en",
                "CFG.ALU_ROUNDING_MODE_Gasket_srnd_en": "pack_srnd_en",
                "CFG.ALU_ROUNDING_MODE_Packer_srnd_en": "pack_srnd_en",
            },
        )
        # <XY_REG_1_Ystride_ADDR32, XY_REG_0_Ystride_SHAMT, XY_REG_1_Ystride_MASK>: the word is XY_REG_1
        # (the core now names the field itself, so the identical unpack companion is dropped)
        ystride = line_of(
            path, "cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32"
        )
        self.assertEqual(
            {
                e.resource: e.value
                for e in self.fn("configure_unpack_AB").effects
                if e.evidence.endswith(f":{ystride}")
                and e.resource.startswith("CFG.")
                and e.access == "write"
            },
            {
                "CFG.UNP0_ADDR_CTRL_XY_REG_1_Ystride": "canonical_unpA_y_stride(unpA_dst_format_masked)"
            },
        )
        # a whole-word store of a multi-field word: both 16-bit halves
        dest = line_of(
            path, "TT_CFG_STORE(cfg, THCON_SEC0_REG5_Dest_cntx0_address_ADDR32"
        )
        self.assertEqual(
            self.fields_at("configure_unpack_AB", dest, "CFG.", "unpack.cfg_field"),
            {
                "CFG.THCON_SEC0_REG5_Dest_cntx0_address": "Dest_cntx0_address",
                "CFG.THCON_SEC0_REG5_Dest_cntx1_address": "Dest_cntx1_address",
            },
        )
        adder = line_of(
            path, "TT_CFG_STORE(cfg, UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_ADDR32"
        )
        self.assertEqual(  # the shared-exp field of the same word is cleared by the store
            self.fields_at("configure_unpack_AB", adder, "CFG.", "unpack.cfg_field"),
            {
                "CFG.UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr": "1",
                "CFG.UNP0_FORCED_SHARED_EXP_shared_exp": "0",
            },
        )

    # ---- GPR flow: tilize init / uninit (llk_unpack_tilize.h) ------------ #
    def test_tilize_init_config_word_through_gpr_halves(self) -> None:
        path = f"{LLK}/llk_lib/llk_unpack_tilize.h"
        wrcfg = line_of(
            path,
            "TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG2_Out_data_format_ADDR32)",
        )
        rows = [r for r in self.mine("_llk_unpack_tilize_init_") if r[5] == wrcfg]
        self.assertIn(
            (
                "unpack.wrcfg_value",
                "CFG.THCON_SEC0_REG2_Out_data_format",
                "write",
                "config.val[0]",
                "",
                wrcfg,
                "",
                "unpack",
            ),
            rows,
        )
        word = self.fields_at("_llk_unpack_tilize_init_", wrcfg, "CFG.THCON_SEC0_REG2_")
        self.assertEqual(
            word,
            {
                "CFG.THCON_SEC0_REG2_Out_data_format": "unpack_dst_format",
                "CFG.THCON_SEC0_REG2_Throttle_mode": "2",
                "CFG.THCON_SEC0_REG2_Context_count": "0",
                "CFG.THCON_SEC0_REG2_Haloize_mode": "0",
                "CFG.THCON_SEC0_REG2_Tileize_mode": "1",
                "CFG.THCON_SEC0_REG2_Unpack_Src_Reg_Set_Upd": "0",
                "CFG.THCON_SEC0_REG2_Unpack_If_Sel": "0",
                "CFG.THCON_SEC0_REG2_Upsample_rate": "0",
                "CFG.THCON_SEC0_REG2_Ovrd_data_format": "0",
                "CFG.THCON_SEC0_REG2_Upsample_and_interleave": "0",
                # the 16-bit struct shift_amount spans the four 4-bit per-context cfg fields
                "CFG.THCON_SEC0_REG2_Shift_amount_cntx0": "shift_amount",
                "CFG.THCON_SEC0_REG2_Shift_amount_cntx1": "((shift_amount) >> 4) & 0xf",
                "CFG.THCON_SEC0_REG2_Shift_amount_cntx2": "((shift_amount) >> 8) & 0xf",
                "CFG.THCON_SEC0_REG2_Shift_amount_cntx3": "((shift_amount) >> 12) & 0xf",
            },
        )
        # branch-dependent writes keep their if/else condition
        x_dim = line_of(
            path,
            "cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(Tile_x_dim",
        )
        self.assertEqual(
            sorted(
                r
                for r in self.mine("_llk_unpack_tilize_init_", "unpack.cfg_field")
                if r[5] == x_dim
            ),
            [
                (
                    "unpack.cfg_field",
                    "CFG.THCON_SEC0_REG5_Tile_x_dim_cntx0",
                    "write",
                    "Tile_x_dim",
                    "!(is_8bit_format)",
                    x_dim,
                    "",
                    "unpack",
                ),
                (
                    "unpack.cfg_field",
                    "CFG.THCON_SEC0_REG5_Tile_x_dim_cntx1",
                    "write",
                    "Tile_x_dim",
                    "!(is_8bit_format)",
                    x_dim,
                    "",
                    "unpack",
                ),
            ],
        )
        scratch = line_of(
            path,
            "TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, SCRATCH_SEC0_val_ADDR32)",
        )
        self.assertIn(
            (
                "unpack.wrcfg_value",
                "CFG.SCRATCH_SEC0_val",
                "write",
                "base_address_increment",
                "is_8bit_format && num_faces == 4",
                scratch,
                "",
                "unpack",
            ),
            self.mine("_llk_unpack_tilize_init_"),
        )

    def test_tilize_uninit_exact_effects(self) -> None:
        path = f"{LLK}/llk_lib/llk_unpack_tilize.h"
        z_dim = line_of(
            path,
            "cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 16, TILE_DESC_UPPER_HALFWORD_MASK>(num_faces)",
            1,
        )
        x_dim = line_of(
            path,
            "cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 16, TILE_DESC_UPPER_HALFWORD_MASK>(CANONICAL_UNPA_TILE_X_DIM)",
        )
        wrcfg = line_of(
            path,
            "TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG2_Out_data_format_ADDR32)",
        )
        x_cntx = line_of(path, "canonical_unpA_tile_x_dim_cntx(face_r_dim)")
        zeros = [
            "Context_count",
            "Haloize_mode",
            "Ovrd_data_format",
            "Shift_amount_cntx0",
            "Shift_amount_cntx1",
            "Shift_amount_cntx2",
            "Shift_amount_cntx3",
            "Tileize_mode",
            "Unpack_If_Sel",
            "Unpack_Src_Reg_Set_Upd",
            "Upsample_and_interleave",
            "Upsample_rate",
        ]
        expected = [
            (
                "unpack.cfg_field",
                "UNPACK.THCON_SEC0_TILE_DESC.z_dim",
                "write",
                "num_faces",
                "",
                z_dim,
                "",
                "unpack",
            ),
            (
                "unpack.cfg_field",
                "UNPACK.THCON_SEC0_TILE_DESC.x_dim",
                "write",
                "0",
                "",
                x_dim,
                "",
                "unpack",
            ),
            (
                "unpack.wrcfg_value",
                "CFG.THCON_SEC0_REG2_Out_data_format",
                "write",
                "config.val[0]",
                "",
                wrcfg,
                "",
                "unpack",
            ),
            (
                "unpack.cfg_struct_field",
                "CFG.THCON_SEC0_REG2_Out_data_format",
                "write",
                "unpack_dst_format",
                "",
                wrcfg,
                "",
                "unpack",
            ),
            (
                "unpack.cfg_struct_field",
                "CFG.THCON_SEC0_REG2_Throttle_mode",
                "write",
                "2",
                "",
                wrcfg,
                "",
                "unpack",
            ),
        ]
        expected += [
            (
                "unpack.cfg_struct_field",
                f"CFG.THCON_SEC0_REG2_{n}",
                "write",
                "0",
                "",
                wrcfg,
                "",
                "unpack",
            )
            for n in zeros
        ]
        expected += [
            (
                "unpack.cfg_field",
                "CFG.THCON_SEC0_REG5_Tile_x_dim_cntx1",
                "write",
                "((canonical_unpA_tile_x_dim_cntx(face_r_dim)) >> 16) & 0xffff",
                "",
                x_cntx,
                "",
                "unpack",
            ),
        ]
        self.assertEqual(
            sorted(self.mine("_llk_unpack_tilize_uninit_")), sorted(expected)
        )
        # the <X_ADDR32, 0, 0xffffffff> mask covers both halves: the core names cntx0 (same value as
        # the unpack companion, which is therefore dropped) and cntx1 (raw word value)
        core = sorted(
            (e.resource, e.value)
            for e in self.fn("_llk_unpack_tilize_uninit_").effects
            if e.family == "core"
            and e.evidence.endswith(f":{x_cntx}")
            and e.resource.startswith("CFG.")
        )
        self.assertEqual(
            core,
            [
                (
                    "CFG.THCON_SEC0_REG5_Tile_x_dim_cntx0",
                    "canonical_unpA_tile_x_dim_cntx(face_r_dim)",
                ),
                (
                    "CFG.THCON_SEC0_REG5_Tile_x_dim_cntx1",
                    "canonical_unpA_tile_x_dim_cntx(face_r_dim)",
                ),
            ],
        )

    # ---- MOP programs: deferred UNPACR reads at the program() site -------- #
    def test_tilize_mop_reads_follow_operand_ternary(self) -> None:
        path = f"{LLK}/llk_lib/llk_unpack_tilize.h"
        program = line_of(path, "tmp.program();")
        rows = self.mine("_llk_unpack_tilize_mop_config_")
        self.assertEqual(
            {r[5] for r in rows}, {program}
        )  # nothing at the static constexpr definitions
        self.assertTrue(all(r[6] == "deferred" for r in rows))
        reads = {(r[1], r[4]) for r in rows if r[0] == "unpack.unpacr.read"}
        # loop op: unpack_to_dest ? unpack_srca_to_dest (unpacker 0) : unpack_srcb_set_dvalid (UNPACR_NOP)
        # start op: unpack_srca, set only when !unpack_to_dest
        self.assertEqual(
            reads,
            {(n, "unpack_to_dest") for n in SEC0_READS}
            | {(n, "!unpack_to_dest") for n in SEC0_READS},
        )
        self.assertEqual(
            sorted((r[1], r[3], r[4]) for r in rows if r[0] == "unpack.unpacr.adc_inc"),
            [
                ("ADC.UNP_A.ZW", "inc:0b00010001", "unpack_to_dest"),
                ("ADC.UNP_A.ZW", "inc:0b1", "!unpack_to_dest"),
            ],
        )

    def test_unpack_A_mop_branches_select_the_unpacker(self) -> None:
        path = f"{LLK}/llk_lib/llk_unpack_A.h"
        scalar = line_of(path, "tmp.program();", 8)  # BroadcastType::SCALAR branch
        none = line_of(
            path, "tmp.program();", 11
        )  # BroadcastType::NONE, no transpose, no acc_to_dest
        self.assertEqual(
            self.reads_at("_llk_unpack_A_mop_config_", scalar), sorted(SEC1_READS)
        )
        self.assertEqual(
            self.reads_at("_llk_unpack_A_mop_config_", none), sorted(SEC0_READS)
        )
        conditions = {
            r[4]
            for r in self.mine("_llk_unpack_A_mop_config_", "unpack.unpacr.read")
            if r[5] == scalar
        }
        self.assertEqual(len(conditions), 1)
        self.assertTrue(conditions.pop().endswith("&& BType == BroadcastType::SCALAR"))
        # else if (ROW || SCALAR): the || part keeps its parentheses inside the && chain
        row_or_scalar = line_of(path, "tmp.program();", 1)
        self.assertEqual(
            {
                r[4]
                for r in self.mine("_llk_unpack_A_mop_config_", "unpack.unpacr.read")
                if r[5] == row_or_scalar
            },
            {
                "should_unpack_to_dest(unpack_to_dest, unpack_src_format, unpack_dst_format) && "
                "!(transpose_of_faces && num_faces == 4) && (BType == BroadcastType::ROW || BType == BroadcastType::SCALAR)"
            },
        )
        definitions = line_of(
            path, "static constexpr std::uint32_t unpack_srcb_inc_z_0"
        )
        self.assertFalse(
            [
                r
                for r in self.mine("_llk_unpack_A_mop_config_")
                if r[5] in (definitions, definitions + 1)
            ]
        )
        # bound to SCALAR from a caller, the other broadcast branches fold away
        effects = self.db.call_effects(
            "_llk_unpack_A_mop_config_",
            template_args=("BroadcastType::SCALAR",),
            args=("false", "ckernel::DEFAULT_TENSOR_SHAPE", "0", "0"),
            evidence="kernel.cpp:7",
        )
        td = {
            e.evidence
            for e in effects
            if e.resource == "UNPACK.THCON_SEC1_TILE_DESC.in_data_format"
        }
        self.assertEqual(td, {f"{path}:{scalar}"})

    # ---- direct UNPACR, ternary cfg index (main), SW reads --------------- #
    def test_unpack_A_custom_direct_unpacr_and_context_reads(self) -> None:
        path = f"{LLK}/llk_lib/experimental/llk_unpack_A_custom.h"
        unpacr = line_of(path, "TTI_UNPACR(SrcA")
        rows = self.mine("_llk_unpack_A_custom_")
        reads = sorted(r[1] for r in rows if r[0] == "unpack.unpacr.read")
        self.assertEqual(reads, sorted(SEC0_READS))
        self.assertTrue(
            all(
                r[5] == unpacr and r[6] == "" and r[7] == "unpack"
                for r in rows
                if r[0] == "unpack.unpacr.read"
            )
        )
        self.assertFalse(
            [r for r in rows if r[0] == "unpack.unpacr.adc_inc"]
        )  # AddrMode 0b0
        self.assertEqual(
            sorted((r[1], r[4], r[5]) for r in rows if r[0] == "unpack.sw_read"),
            [
                ("SW.unp_cfg_context", "", line_of(path, "if (unp_cfg_context == 0)")),
                (
                    "SW.unp_cfg_context",
                    "",
                    line_of(path, "switch_config_context(unp_cfg_context)"),
                ),
            ],
        )

    def test_main_ternary_cfg_index_becomes_two_writes(self) -> None:
        path = f"{LLK}/llk_lib/experimental/llk_unpack_A_topk_xl_copy.h"
        line = line_of(path, "cfg[upk0_reg]")
        self.assertEqual(
            sorted(self.mine("_llk_unpack_topk_xl_copy_", "unpack.cfg_ternary")),
            [
                (
                    "unpack.cfg_ternary",
                    "CFG.THCON_SEC0_REG3_Base_address",
                    "write",
                    "address",
                    "unp_cfg_context == 0",
                    line,
                    "",
                    "unpack",
                ),
                (
                    "unpack.cfg_ternary",
                    "CFG.THCON_SEC0_REG3_Base_cntx1_address",
                    "write",
                    "address",
                    "!(unp_cfg_context == 0)",
                    line,
                    "",
                    "unpack",
                ),
            ],
        )
        # the core alone can only say CFG.?upk0_reg; SUPERSEDES drops it in favour of the two writes
        core = [
            e.resource
            for e in self.fn("_llk_unpack_topk_xl_copy_").effects
            if e.pattern == "core.cfg_assign"
        ]
        self.assertEqual(core, [])

    # ---- conditional GPR halves, opaque restores ------------------------ #
    def test_reduce_custom_init_gpr_halves_fold_if_else(self) -> None:
        path = f"{LLK}/llk_lib/experimental/llk_unpack_AB_reduce_custom.h"
        wrcfg = line_of(
            path,
            "TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1)",
        )
        x_dim = line_of(
            path,
            "TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32)",
        )
        rows = sorted(self.mine("_llk_unpack_AB_reduce_block_max_row_init_"))
        self.assertEqual(
            rows,
            sorted(
                [
                    (
                        "unpack.wrcfg_value",
                        "CFG.THCON_SEC0_REG5_Tile_x_dim_cntx0",
                        "write",
                        "0x1000100",
                        "",
                        x_dim,
                        "",
                        "unpack",
                    ),
                    (
                        "unpack.cfg_field",
                        "CFG.THCON_SEC0_REG5_Tile_x_dim_cntx0",
                        "write",
                        "0x100",
                        "",
                        x_dim,
                        "",
                        "unpack",
                    ),
                    (
                        "unpack.cfg_field",
                        "CFG.THCON_SEC0_REG5_Tile_x_dim_cntx1",
                        "write",
                        "0x100",
                        "",
                        x_dim,
                        "",
                        "unpack",
                    ),
                    (
                        "unpack.wrcfg_value",
                        "CFG.THCON_SEC0_REG0_TileDescriptor+1",
                        "write",
                        "((tensor_shape.num_faces_r_dim == 1) ? (2) : (4)) | ((1) << 16)",
                        "",
                        wrcfg,
                        "",
                        "unpack",
                    ),
                    (
                        "unpack.cfg_field",
                        "UNPACK.THCON_SEC0_TILE_DESC.y_dim",
                        "write",
                        "(tensor_shape.num_faces_r_dim == 1) ? (2) : (4)",
                        "",
                        wrcfg,
                        "",
                        "unpack",
                    ),
                    (
                        "unpack.cfg_field",
                        "UNPACK.THCON_SEC0_TILE_DESC.z_dim",
                        "write",
                        "1",
                        "",
                        wrcfg,
                        "",
                        "unpack",
                    ),
                ]
            ),
        )

    def test_reduce_custom_uninit_restores_saved_gprs(self) -> None:
        path = f"{LLK}/llk_lib/experimental/llk_unpack_AB_reduce_custom.h"
        td = line_of(path, "TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1")
        self.assertEqual(
            self.fields_at(
                "_llk_unpack_AB_reduce_block_max_row_uninit_",
                td,
                "UNPACK.",
                "unpack.cfg_field",
            ),
            {
                "UNPACK.THCON_SEC0_TILE_DESC.y_dim": "GPR(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1)",
                "UNPACK.THCON_SEC0_TILE_DESC.z_dim": "((GPR(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1)) >> 16) & 0xffff",
            },
        )

    def test_fast_tilize_init_constant_and_param_halves(self) -> None:
        path = f"{LLK}/llk_lib/experimental/llk_unpack_fast_tilize.h"
        x_dim = line_of(
            path,
            "TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32)",
        )
        td = line_of(
            path,
            "TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1)",
        )
        rows = {
            (r[0], r[1]): r[3]
            for r in self.mine("_llk_unpack_fast_tilize_init_")
            if r[5] in (x_dim, td)
        }
        self.assertEqual(
            rows,
            {
                (
                    "unpack.wrcfg_value",
                    "CFG.THCON_SEC0_REG5_Tile_x_dim_cntx0",
                ): "0x200020",  # TILE_C_DIM in both halves
                ("unpack.cfg_field", "CFG.THCON_SEC0_REG5_Tile_x_dim_cntx0"): "0x20",
                ("unpack.cfg_field", "CFG.THCON_SEC0_REG5_Tile_x_dim_cntx1"): "0x20",
                (
                    "unpack.wrcfg_value",
                    "CFG.THCON_SEC0_REG0_TileDescriptor+1",
                ): "(ct_dim) | ((FACE_R_DIM) << 16)",
                ("unpack.cfg_field", "UNPACK.THCON_SEC0_TILE_DESC.y_dim"): "ct_dim",
                ("unpack.cfg_field", "UNPACK.THCON_SEC0_TILE_DESC.z_dim"): "0x10",
            },
        )
        self.assertEqual(self.fn("_llk_unpack_fast_tilize_reinit_xdim_").role, "init")

    # ---- replay recording, DMA-register ALU, cross-thread ADC ----------- #
    def test_matmul_replay_program_is_deferred(self) -> None:
        path = f"{LLK}/llk_lib/llk_unpack_AB_matmul.h"
        rows = self.mine("_llk_unpack_AB_matmul_mop_config_")
        replay = sorted(
            (r[3], r[4], r[6]) for r in rows if r[0] == "unpack.replay_program"
        )
        self.assertEqual(
            [(v.split("(")[1], c, d) for v, c, d in replay],
            [
                (
                    "kernel_broadcast_a,unpB_partial_face)",
                    "!(reuse_a)",
                    "start=0;len=replay_buf_prog_len",
                ),
                (
                    "kernel_broadcast_b,unpA_partial_face)",
                    "reuse_a",
                    "start=0;len=replay_buf_prog_len",
                ),
            ],
        )
        self.assertNotEqual(
            replay[0][0].split("(")[0], replay[1][0].split("(")[0]
        )  # different programs differ
        recorded = [
            r
            for r in rows
            if r[0]
            in ("unpack.unpacr.read", "unpack.dmareg.opb_read", "unpack.wrcfg_value")
        ]
        self.assertTrue(recorded and all(r[6] == "deferred" for r in recorded))
        first = line_of(
            path,
            "TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/",
        )
        self.assertEqual(
            {r[4] for r in rows if r[0] == "unpack.unpacr.read" and r[5] == first},
            {"reuse_a && !(unpA_partial_face)"},
        )
        wrcfg = line_of(
            path,
            "TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG3_Base_address_ADDR32)",
        )
        self.assertIn(
            (
                "unpack.wrcfg_value",
                "CFG.THCON_SEC0_REG3_Base_address",
                "write",
                "ADDDMAREG(p_gpr_unpack::TILE_SIZE_A, p_gpr_unpack::TMP0)",
                "reuse_a && !(kernel_broadcast_b == 1)",
                wrcfg,
                "deferred",
                "unpack",
            ),
            rows,
        )
        add = line_of(
            path,
            "TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A)",
        )
        # the core now reads OpB and marks replay-recorded instructions deferred itself, so the
        # identical unpack companion is dropped in favour of the core effect
        core = [
            (e.pattern, e.resource, e.access, e.condition, e.detail)
            for e in self.fn("_llk_unpack_AB_matmul_mop_config_").effects
            if e.evidence.endswith(f":{add}") and e.access == "read"
        ]
        self.assertIn(
            (
                "core.dmareg_alu.read_b",
                "GPR.unpack.TMP0",
                "read",
                "reuse_a && !(kernel_broadcast_b == 1)",
                "deferred",
            ),
            core,
        )

    def test_face_compressed_init_writes_other_threads_adc(self) -> None:
        path = f"{LLK}/llk_lib/experimental/llk_unpack_AB_face_compressed_mm.h"
        rows = sorted(
            (r[1], r[3], r[5], r[7])
            for r in self.mine(
                "_llk_unpack_AB_face_compressed_mm_init_", "unpack.setadc_override"
            )
        )
        xy_math = line_of(
            path,
            "TTI_SETADCXY_THREAD_OVERRIDE(p_setadc::UNP_AB, p_setadc::THREAD_OVRD_MATH",
        )
        xy_pack = line_of(
            path,
            "TTI_SETADCXY_THREAD_OVERRIDE(p_setadc::UNP_AB, p_setadc::THREAD_OVRD_PACK",
        )
        x_math = line_of(
            path,
            "TTI_SETADC_THREAD_OVERRIDE(p_setadc::UNP0, p_setadc::CH_1, p_setadc::SET_X, p_setadc::THREAD_OVRD_MATH",
        )
        x_pack = line_of(
            path,
            "TTI_SETADC_THREAD_OVERRIDE(p_setadc::UNP0, p_setadc::CH_1, p_setadc::SET_X, p_setadc::THREAD_OVRD_PACK",
        )
        xy = "0, 0, 0, 0, SETADC_CH01(p_setadc::XY)"
        self.assertEqual(
            rows,
            sorted(
                [
                    ("ADC.UNP_A.XY", xy, xy_math, "math"),
                    ("ADC.UNP_B.XY", xy, xy_math, "math"),
                    ("ADC.UNP_A.XY", xy, xy_pack, "pack"),
                    ("ADC.UNP_B.XY", xy, xy_pack, "pack"),
                    ("ADC.UNP_A.X", "unpA_x_end", x_math, "math"),
                    ("ADC.UNP_A.X", "unpA_x_end", x_pack, "pack"),
                ]
            ),
        )

    # ---- llk_api parameter flow, compute-API raw UNPACR, helpers --------- #
    def test_llk_api_reconfig_srca_param_flow(self) -> None:
        (model,) = [
            f
            for f in self.db.by_name("llk_unpack_reconfig_data_format_srca")
            if len(f.params) == 1
        ]
        effects = self.db.effects(model.key)
        values = {
            e.resource: (e.value, e.condition)
            for e in effects
            if e.access == "write" and e.family == "unpack"
        }
        self.assertEqual(
            values["UNPACK.THCON_SEC0_TILE_DESC.in_data_format"],
            ("unpack_src_format[(get_operand_id(srca_new_operand))]", ""),
        )
        self.assertEqual(
            values["UNPACK.THCON_SEC0_TILE_DESC.z_dim"],
            (
                "get_operand_num_faces((get_operand_id(srca_new_operand)))",
                "dim_stride_target == p_dim_stride_target::FACE_ROW_MAJOR",
            ),
        )
        self.assertEqual(
            values["CFG.THCON_SEC0_REG5_Tile_x_dim_cntx1"][1],
            "dim_stride_target == p_dim_stride_target::FACE_ROW_MAJOR",
        )
        self.assertEqual(model.role, "reconfig")
        self.assertEqual(model.thread, "unpack")

    def test_compute_api_raw_unpacr_outside_scope(self) -> None:
        rows = self.mine("weighted_reduce_unpack_impl")
        self.assertEqual(
            {r[0] for r in rows}, {"unpack.unpacr.read", "unpack.unpacr.adc_inc"}
        )  # no scoped refinements
        path = "tt_metal/hw/inc/api/compute/experimental/sdpa_weighted_reduce.h"
        first = line_of(path, "TTI_UNPACR(SrcA, 0b00010001")
        second = line_of(path, "TTI_UNPACR(SrcB, 0b00000000")
        self.assertEqual(
            self.reads_at("weighted_reduce_unpack_impl", first), sorted(SEC0_READS)
        )
        self.assertEqual(
            self.reads_at("weighted_reduce_unpack_impl", second), sorted(SEC1_READS)
        )
        self.assertEqual(  # 0b00010001: ch0_z and ch1_z increments only
            [(r[1], r[3], r[5], r[7]) for r in rows if r[0] == "unpack.unpacr.adc_inc"],
            [("ADC.UNP_A.ZW", "inc:0b00010001", first, "unpack")],
        )

    def test_unpacker_select_parameter_resolves_in_caller(self) -> None:
        (direct,) = {r[1] for r in self.mine("flush_xsearch_cache")}
        self.assertEqual(direct, "UNPACK.?unpacker")
        effects = self.db.call_effects(
            "flush_xsearch_cache", args=("SrcB",), evidence="kernel.cpp:3"
        )
        self.assertEqual(
            sorted(e.resource for e in effects if e.pattern == "unpack.unpacr.read"),
            sorted(SEC1_READS),
        )
        self.assertTrue(all(e.resolved for e in effects))

    def test_semaphore_cfgshiftmask_and_roles(self) -> None:
        self.assertEqual(  # semaphore_read is a core pattern now (unpack's own one disables itself)
            [
                (e.resource, e.access)
                for e in self.fn("wait_for_next_context").effects
                if e.resource.startswith("SEM.")
            ],
            [("SEM.UNPACK_SYNC", "read")],
        )
        self.assertEqual(self.fn("_llk_unpack_configure_addresses_").role, "helper")
        self.assertEqual(self.fn("switch_config_context").role, "helper")
        self.assertEqual(self.fn("_llk_unpack_tilize_init_").role, "init")

    def test_helpers(self) -> None:
        self.assertEqual(
            unpack.split_ternary("(x == y::Z) ? ns::f<T::A>() : b"),
            ("x == y::Z", "ns::f<T::A>()", "b"),
        )
        self.assertEqual(
            unpack.split_ternary("a ? b ? c : d : e"), ("a", "b ? c : d", "e")
        )
        self.assertIsNone(unpack.split_ternary("p_setadc::UNP_A"))
        tables = self.db.tables
        self.assertEqual(
            unpack.extract_bits(tables, "face_dim | (face_dim << 16)", 0, 16, 16),
            "face_dim",
        )
        self.assertEqual(
            unpack.extract_bits(
                tables,
                "(0 << UNP0_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT) | (z << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT)",
                0,
                16,
                16,
            ),
            "0",
        )
        self.assertEqual(
            unpack.extract_bits(tables, "v", 0, 16, 16), "((v) >> 16) & 0xffff"
        )
        self.assertEqual(
            unpack.extract_bits(tables, "num_faces", 16, 16, 16), "num_faces"
        )
        self.assertEqual(unpack.cfg_int(tables, "0b00'01'00'01"), 0b00010001)
        self.assertEqual(unpack.unpacker_index("Srcs::SrcB"), 1)
        self.assertIsNone(unpack.unpacker_index("unpacker"))
        structs, unions = unpack.struct_layouts(
            tables
        )  # parsed from the tree's cunpack_common.h
        config = structs["unpack_config_t"]
        self.assertEqual(unions["unpack_config_u"], "unpack_config_t")
        shift = next(f for f in config.fields if f.name == "shift_amount")
        self.assertEqual((shift.word, shift.bit, shift.width), (0, 16, 16))
        uncompress = next(f for f in config.fields if f.name == "uncompress_cntx4_7")
        self.assertEqual((uncompress.word, uncompress.bit), (1, 16))
        self.assertEqual(
            config,
            unpack.layout_from_widths(
                "unpack_config_t", unpack._FALLBACK_STRUCTS["unpack_config_t"]
            ),
        )

    def test_deterministic(self) -> None:
        first = effects_db.dumps_effects(self.db.to_json(transitive=False))
        second = effects_db.dumps_effects(
            effects_db.build(self.root, families=["unpack"]).to_json(transitive=False)
        )
        self.assertEqual(first, second)
        self.assertNotIn(str(self.root), first)


if __name__ == "__main__":
    unittest.main()
