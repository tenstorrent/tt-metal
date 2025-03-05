// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef PERF_DUMP
#include "perf_res_decouple.h"
#endif

// MT: This should be dissolved and moved to the appropriate place
#include "tensix.h"

// Hand-coded parameter encoding for various common instructions
namespace ckernel
{

struct p_setrwc
{
#ifdef PERF_DUMP

#if SKIP_UNP == 1
    constexpr static uint CLR_A  = 0x0;
    constexpr static uint CLR_B  = 0x0;
    constexpr static uint CLR_AB = 0x0;
#else
    constexpr static uint CLR_A  = 0x1;
    constexpr static uint CLR_B  = 0x2;
    constexpr static uint CLR_AB = 0x3;
#endif

#else
    constexpr static uint CLR_A  = 0x1;
    constexpr static uint CLR_B  = 0x2;
    constexpr static uint CLR_AB = 0x3;
#endif
    constexpr static uint CLR_NONE = 0x0;

    constexpr static uint SET_A     = 0x1;
    constexpr static uint SET_B     = 0x2;
    constexpr static uint SET_AB    = 0x3;
    constexpr static uint SET_D     = 0x4;
    constexpr static uint SET_AD    = 0x5;
    constexpr static uint SET_BD    = 0x6;
    constexpr static uint SET_ABD   = 0x7;
    constexpr static uint SET_F     = 0x8;
    constexpr static uint SET_A_F   = 0x9;
    constexpr static uint SET_B_F   = 0xa;
    constexpr static uint SET_AB_F  = 0xb;
    constexpr static uint SET_D_F   = 0xc;
    constexpr static uint SET_AD_F  = 0xd;
    constexpr static uint SET_BD_F  = 0xe;
    constexpr static uint SET_ABD_F = 0xf;

    constexpr static uint CR_A         = 0x1;
    constexpr static uint CR_B         = 0x2;
    constexpr static uint CR_AB        = 0x3;
    constexpr static uint CR_D         = 0x4;
    constexpr static uint CR_AD        = 0x5;
    constexpr static uint CR_BD        = 0x6;
    constexpr static uint CR_ABD       = 0x7;
    constexpr static uint C_TO_CR_MODE = 0x8;
};

struct p_setibrwc
{
    constexpr static uint SET_BIAS = 0x0;
    constexpr static uint INC_BIAS = 0x1;
    constexpr static uint CR_NONE  = 0x0;
    constexpr static uint CR_BIAS  = 0x1;
};

struct p_unpacr
{
    constexpr static uint RAREFYB_DISABLE = 0x0;
    constexpr static uint RAREFYB_ENABLE  = 0x1;

    constexpr static uint TILE0_ADDRCNT_CONTEXT = (0); // Address counter context for tile 0
    constexpr static uint TILE1_ADDRCNT_CONTEXT = (0); // Address counter context for tile 1
    constexpr static uint TILE2_ADDRCNT_CONTEXT = (1); // Address counter context for tile 2
    constexpr static uint TILE3_ADDRCNT_CONTEXT = (1); // Address counter context for tile 3
    constexpr static uint TILE0_CFG_CONTEXT     = (0); // Config context for tile 0
    constexpr static uint TILE1_CFG_CONTEXT     = (0); // Config context for tile 1
    constexpr static uint TILE2_CFG_CONTEXT     = (0); // Config context for tile 2
    constexpr static uint TILE3_CFG_CONTEXT     = (0); // Config context for tile 3
    constexpr static uint AUTO_INC_CONTEXT      = (1); // Auto increment config context (max value set through unpacker config command)

    constexpr static uint UNP_POP           = 0x0;
    constexpr static uint UNP_CLRSRC        = 0x1;
    constexpr static uint UNP_NOP           = 0x2;
    constexpr static uint UNP_POP_STREAM    = 0x3;
    constexpr static uint UNP_CLRSRC_ZERO   = 0x0;
    constexpr static uint UNP_CLRSRC_NEGINF = 0x1;
    constexpr static uint UNP_CLRSRC_ONE    = 0x2;
    constexpr static uint UNP_CLRSRC_IMM    = 0x3;

    constexpr static uint UNP_CLRSRC_RESET_ALL_BANKS = 0x1;
    constexpr static uint UNP_CLRSRC_ONE_FP16A       = 0x0;
    constexpr static uint UNP_CLRSRC_ONE_FP16B       = 0x1;
    constexpr static uint UNP_CLRSRC_ONE_TF32        = 0x1;
    constexpr static uint UNP_CLRSRC_ONE_INT8        = 0x2;
};

// TODO: RT Review this struct, bits do not match for UNPACR_NOP
struct p_unpacr_nop
{
    constexpr static uint UNP_POP = 0b000;
    constexpr static uint CLR_SRC = 0b01;
    constexpr static uint UNP_NOP = 0b010;

    constexpr static uint UNP_ZEROSRC   = 0b001;
    constexpr static uint UNP_NEGINFSRC = 0b101;

    constexpr static uint SET_DVALID = 0x1;

    constexpr static uint UNP_ZEROSRC_RESET_ALL_BANKS    = 0b1001; // default is clear current bank
    constexpr static uint UNP_ZEROSRC_STALL_RESET_WR_RDY = 0b10001;
    constexpr static uint UNP_ZEROSRC_SET_DVALID         = 0b1000001;

    constexpr static uint UNP0 = 0x0;
    constexpr static uint UNP1 = 0x1;

    constexpr static uint CLR_SRC_0      = 0b00;
    constexpr static uint CLR_SRC_NEGINF = 0b01;
    constexpr static uint CLR_SRC_1      = 0b10;
    constexpr static uint CLR_SRC_IMM    = 0b11;
};

struct p_srcb
{
    constexpr static uint FORWARD_PASS  = 0x0;
    constexpr static uint BACKWARD_PASS = 0x1;
};

struct p_setadc
{
    constexpr static uint UNP0   = 0b001;
    constexpr static uint UNP1   = 0b010;
    constexpr static uint UNP_A  = 0b001;
    constexpr static uint UNP_B  = 0b010;
    constexpr static uint UNP_AB = 0b011;
    constexpr static uint PAC    = 0b100;

    constexpr static uint SET_X = 0;
    constexpr static uint SET_Y = 1;
    constexpr static uint SET_Z = 2;
    constexpr static uint SET_W = 3;

    constexpr static uint CH_0 = 0;
    constexpr static uint CH_1 = 1;
};

struct p_pacr
{
    constexpr static uint P_ZERO_OUTPUT_DISABLED = 0x0;
    constexpr static uint P_ZERO_OUTPUT_ENABLED  = 0x1;

    constexpr static uint DST_ACCESS_NORMAL_MODE  = 0b0;
    constexpr static uint DST_ACCESS_STRIDED_MODE = 0b1;

    constexpr static uint NO_ROW_PAD_ZERO                          = 0b000;
    constexpr static uint ROW_PAD_ZERO_ALL_PACR                    = 0b001;
    constexpr static uint ROW_PAD_ZERO_ALL_PACR_16DATUM_ALGN       = 0b101;
    constexpr static uint ROW_PAD_ZERO_NO_CONCAT_PACR              = 0b010;
    constexpr static uint ROW_PAD_ZERO_NO_CONCAT_PACR_16DATUM_ALGN = 0b110;
    constexpr static uint ROW_PAD_ZERO_LAST_PACR                   = 0b011;
    constexpr static uint ROW_PAD_ZERO_LAST_PACR_16DATUM_ALGN      = 0b111;

    constexpr static uint CFG_CTXT_0 = 0b00;
    constexpr static uint CFG_CTXT_1 = 0b01;
    constexpr static uint CFG_CTXT_2 = 0b10;
    constexpr static uint CFG_CTXT_3 = 0b11;

    constexpr static uint ADDR_CNT_CTXT_0 = 0b00;
    constexpr static uint ADDR_CNT_CTXT_1 = 0b01;
    constexpr static uint ADDR_CNT_CTXT_2 = 0b10;

    constexpr static uint ALL_INTF_ACTIVE          = 0b0000;
    constexpr static uint ALL_INTF_ACTIVE_ONES     = 0b1111;
    constexpr static uint SINGLE_INTF_ACTIVE       = 0b0001;
    constexpr static uint TWO_INTFS_ACTIVE         = 0b0011;
    constexpr static uint THREE_INTFS_ACTIVE       = 0b0111;
    constexpr static uint _0th_AND_2nd_INTF_ACTIVE = 0b0101;
    constexpr static uint _1st_AND_3rd_INTF_ACTIVE = 0b1010;

    constexpr static uint ZERO_WRITE    = 0b1;
    constexpr static uint NO_ZERO_WRITE = 0b0;

    constexpr static uint NO_CTXT_CTRL               = 0b00;
    constexpr static uint RTL_FLOPS_CTXT_SEL         = 0b01;
    constexpr static uint RTL_FLOPS_CTXT_RST_AND_NOP = 0b10;
    constexpr static uint RTL_FLOPS_CTXT_SEL_NO_RST  = 0b11;
};

struct p_ind
{
    constexpr static uint HIER_REGFILE = 0x0;
    constexpr static uint HIER_L1      = 0x1;

    constexpr static uint INC_NONE = 0x0;
    constexpr static uint INC_2B   = 0x1;
    constexpr static uint INC_4B   = 0x2;
    constexpr static uint INC_16B  = 0x3;

    constexpr static uint LD_16B   = 0;
    constexpr static uint LD_32bit = 1;
    constexpr static uint LD_16bit = 2;
    constexpr static uint LD_8bit  = 3;
};

struct p_mova2d
{
    constexpr static uint MATH_HALO_ROWS = 0x0;
    constexpr static uint MOV_1_ROW      = 0x0;
    constexpr static uint MOV_8_ROWS     = 0x2;
};

struct p_movd2a
{
    constexpr static uint MOV_1_ROW  = 0x0;
    constexpr static uint MOV_4_ROWS = 0x2;
};

struct p_movb2d
{
    constexpr static uint SRC_ZERO_OFFSET          = 0x0;
    constexpr static uint SRC_ROW16_OFFSET         = 0x10;
    constexpr static uint MOV_1_ROW                = 0x0;
    constexpr static uint MOV_1_ROW_D0_BRCST       = 0x1;
    constexpr static uint MOV_8_ROW_BRCST          = 0x2;
    constexpr static uint MOV_8_ROW_BRCST_D0_BRCST = 0x3;
    constexpr static uint MOV_4_ROWS               = 0x4;
    constexpr static uint MOV_4_ROWS_D0_BRCST      = 0x5;
};

struct p_movd2b
{
    constexpr static uint SRC_ZERO_OFFSET  = 0x0;
    constexpr static uint SRC_ROW16_OFFSET = 0x10;
    constexpr static uint MOV_1_ROW        = 0x0;
    constexpr static uint MOV_4_ROWS       = 0x2;
};

struct p_movb2a
{
    constexpr static uint SRCA_ZERO_OFFSET  = 0x0;
    constexpr static uint SRCB_ZERO_OFFSET  = 0x0;
    constexpr static uint SRCB_ROW16_OFFSET = 0x10;
    constexpr static uint MOV_1_ROW         = 0x0;
    constexpr static uint MOV_4_ROWS        = 0x2;
};

struct p_stall
{
    // What to stall on
    constexpr static uint NONE    = 0x0;
    constexpr static uint THCON   = 0x1;
    constexpr static uint UNPACK0 = 0x2;
    constexpr static uint UNPACK1 = 0x4;
    constexpr static uint UNPACK  = UNPACK0 | UNPACK1; // Added to satisfy the LLK code
    constexpr static uint PACK0   = 0x8;
    constexpr static uint PACK    = PACK0;
    constexpr static uint MATH    = 0x10;
    // constexpr static uint SEM_ZERO   = 0x20;
    // constexpr static uint SEM_MAX    = 0x40;
    constexpr static uint SRCA_CLR  = 0x20;
    constexpr static uint SRCB_CLR  = 0x40;
    constexpr static uint SRCA_VLD  = 0x80;
    constexpr static uint SRCB_VLD  = 0x100;
    constexpr static uint XMOV      = 0x200;
    constexpr static uint TRISC_CFG = 0x400;
    constexpr static uint SFPU1     = 0x800;
    constexpr static uint WAIT_SFPU = 0x800;
    constexpr static uint CFGEXU    = 0x1000;

    constexpr static uint ALL_THREAD_RES = THCON | UNPACK0 | UNPACK1 | PACK0 | PACK | MATH | XMOV;

    // What to stall
    constexpr static uint STALL_TDMA   = 0x1;
    constexpr static uint STALL_SYNC   = 0x2;
    constexpr static uint STALL_PACK   = 0x4;
    constexpr static uint STALL_UNPACK = 0x8;
    //    constexpr static uint STALL_XSEARCH = 0x10;
    constexpr static uint STALL_XMOV   = 0x10;
    constexpr static uint STALL_THCON  = 0x20;
    constexpr static uint STALL_MATH   = 0x40;
    constexpr static uint STALL_CFG    = 0x80;
    constexpr static uint STALL_SFPU   = 0x100;
    constexpr static uint STALL_THREAD = 0x1ff;

    constexpr static uint STALL_ON_ZERO = 0x1;
    constexpr static uint STALL_ON_MAX  = 0x2;

    constexpr static uint SEMAPHORE_0    = 0x1;
    constexpr static uint SEMAPHORE_1    = 0x2;
    constexpr static uint SEMAPHORE_2    = 0x4;
    constexpr static uint SEMAPHORE_3    = 0x8;
    constexpr static uint SEMAPHORE_4    = 0x10;
    constexpr static uint SEMAPHORE_5    = 0x20;
    constexpr static uint SEMAPHORE_6    = 0x40;
    constexpr static uint SEMAPHORE_7    = 0x80;
    constexpr static uint SEMAPHORE_BIAS = SEMAPHORE_4;
};

struct p_zeroacc
{
    constexpr static uint CLR_SPECIFIC = 0b000;
    constexpr static uint CLR_16       = 0b001;
    constexpr static uint CLR_HALF     = 0b010;
    constexpr static uint CLR_ALL      = 0b011;
    constexpr static uint CLR_HALF_32B = 0b110;
    constexpr static uint CLR_ALL_32B  = 0b111;
};

struct p_zerosrc
{
    constexpr static uint CLR_A  = 0x1;
    constexpr static uint CLR_B  = 0x2;
    constexpr static uint CLR_AB = 0x3;
};

struct p_shiftx
{
    constexpr static uint SHIFT_1 = 0x0;
    constexpr static uint SHIFT_2 = 0x1;
    constexpr static uint SHIFT_4 = 0x2;
    constexpr static uint SHIFT_8 = 0x3;

    constexpr static uint RESERVED0    = 0x0;
    constexpr static uint RESERVED1    = 0x1;
    constexpr static uint RIGHT_AWAY0  = 0x2;
    constexpr static uint LEFT_TOWARD0 = 0x3;
};

struct p_cfg
{
    constexpr static uint WRCFG_128b = 0x1;
    constexpr static uint WRCFG_32b  = 0x0;
};

struct p_alu
{
    constexpr static uint AND = 0x0;
    constexpr static uint OR  = 0x1;
    constexpr static uint XOR = 0x2;
};

struct p_gpool
{
    constexpr static uint DIM_1X16  = 0x0;
    constexpr static uint DIM_16X16 = 0x1;
    constexpr static uint INDEX_DIS = 0x0;
    constexpr static uint INDEX_EN  = 0x1;
};

struct p_elwise
{
    constexpr static uint SRCB_NO_BCAST  = 0x0;
    constexpr static uint DEST_ACCUM_EN  = 0x1;
    constexpr static uint DEST_ACCUM_DIS = 0x0;
    constexpr static uint SRCB_BCAST_COL = 0x1;
    constexpr static uint SRCB_BCAST_ROW = 0x2;
    constexpr static uint SRCB_BCAST_ALL = 0x3;

    constexpr static uint CLR_A  = 0x1;
    constexpr static uint CLR_B  = 0x2;
    constexpr static uint CLR_AB = 0x3;
};

struct p_sfpu
{
    // SFPU registers
    constexpr static uint LREG0 = 0;
    constexpr static uint LREG1 = 1;
    constexpr static uint LREG2 = 2;
    constexpr static uint LREG3 = 3;
    constexpr static uint LREG4 = 4;
    constexpr static uint LREG5 = 5;
    constexpr static uint LREG6 = 6;
    constexpr static uint LREG7 = 7;

    // HW provided constants
    constexpr static uint LCONST_0_8373 = 8;
    constexpr static uint LCONST_0      = 9;
    constexpr static uint LCONST_1      = 10;

    // Programmable constants
    constexpr static uint LREG11      = 11;
    constexpr static uint LREG12      = 12;
    constexpr static uint LREG13      = 13;
    constexpr static uint LREG14      = 14;
    constexpr static uint LCONST_neg1 = 11;

    constexpr static uint LTILEID = 15;

    constexpr static uint kCONST_1_FP16B  = 0x3F80;
    constexpr static uint kCONST_1_FP16A  = 0x3C00;
    constexpr static uint kCONST_0        = 0x0000;
    constexpr static uint kCONST_Exp_8Bit = 0;
    constexpr static uint kCONST_Exp_5Bit = 1;
};

struct p_sfpswap
{
    // SFPSWAP instruction modes
    constexpr static uint UNCONDITIONALLY = 0;
    constexpr static uint ALL_ROWS_MAX    = 1;
    constexpr static uint ROWS_01_MAX     = 2;
    constexpr static uint ROWS_02_MAX     = 3;
    constexpr static uint ROWS_03_MAX     = 4;
    constexpr static uint ROW_0_MAX       = 5;
    constexpr static uint ROW_1_MAX       = 6;
    constexpr static uint ROW_2_MAX       = 5;
    constexpr static uint ROW_3_MAX       = 6;
};

struct p_exp
{
    constexpr static uint FRAC_BITS = 3;
    constexpr static uint C23_73    = 0x4340; // Based on FRAC_BITS
    // ADJ_EXP = -0x4300 + 0x003F
    //  0x4300 : 0100 0011 0000 0000
    //  0x003F : 0000 0000 0011 1111
    // -0x4300 : 1011 1101 0000 0000
    // ADJ_EXP : 1011 1101 0011 1111 (-0x4300 + 0x003F = 0xBD3F)
    constexpr static uint ADJ_EXP = 0xBD3F;
};

} // namespace ckernel
