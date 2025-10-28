// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Hand-coded parameter encoding for various common instructions
namespace ckernel
{

struct p_setrwc
{
    constexpr static uint CLR_A    = 0x1;
    constexpr static uint CLR_B    = 0x2;
    constexpr static uint CLR_AB   = 0x3;
    constexpr static uint CLR_NONE = 0x0;

    constexpr static uint SET_A     = 0x1;
    constexpr static uint SET_B     = 0x2;
    constexpr static uint SET_AB    = 0x3;
    constexpr static uint SET_D     = 0x4;
    constexpr static uint SET_AD    = 0x5;
    constexpr static uint SET_BD    = 0x6;
    constexpr static uint SET_ABD   = 0x7;
    constexpr static uint SET_F     = 0x8;
    constexpr static uint SET_ABD_F = 0xf;
    constexpr static uint SET_AB_F  = 0xb;

    constexpr static uint CR_A         = 0x1;
    constexpr static uint CR_B         = 0x2;
    constexpr static uint CR_D         = 0x4;
    constexpr static uint C_TO_CR_MODE = 0x8;
    constexpr static uint CR_ABD       = 0x7;
};

struct p_setibrwc
{
    constexpr static uint SET_BIAS = 0x0;
    constexpr static uint INC_BIAS = 0x0;
    constexpr static uint CR_NONE  = 0x0;
    constexpr static uint CR_BIAS  = 0x1;
};

struct p_unpacr
{
    constexpr static uint UNP_A    = 0b000;
    constexpr static uint UNP_B    = 0b001;
    constexpr static uint UNP_S    = 0b010;
    constexpr static uint UNP_DEST = 0b011;

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

    constexpr static uint UNP_STALL_FPU_RD = 0x0;
    constexpr static uint UNP_STALL_UNP_WR = 0x1;

    constexpr static uint UNP_CLRSRC_RESET_ALL_BANKS = 0x1;
    constexpr static uint UNP_CLRSRC_ONE_FP16A       = 0x0;
    constexpr static uint UNP_CLRSRC_ONE_FP16B       = 0x1;
    constexpr static uint UNP_CLRSRC_ONE_TF32        = 0x1;
    constexpr static uint UNP_CLRSRC_ONE_INT8        = 0x2;
};

struct p_set_inc_sel
{
    constexpr static uint TILE_SEL = 0b000;
    constexpr static uint FACE_SEL = 0b001;
    constexpr static uint ROW_SEL  = 0b010;
};

struct p_srcb
{
    constexpr static uint FORWARD_PASS  = 0x0;
    constexpr static uint BACKWARD_PASS = 0x1;
};

struct p_setadc
{
    // RT: Is this outdated?
    constexpr static uint UNP0 = 0b001;
    constexpr static uint UNP1 = 0b010;
    constexpr static uint PAC  = 0b100;

    constexpr static uint SET_X = 0;
    constexpr static uint SET_Y = 1;
    constexpr static uint SET_Z = 2;
    constexpr static uint SET_W = 3;

    constexpr static uint CH_0 = 0;
    constexpr static uint CH_1 = 1;
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

struct p_movd2a
{
    constexpr static uint MOV_1_ROW  = 0x0;
    constexpr static uint MOV_4_ROWS = 0x1;
    constexpr static uint MOV_8_ROWS = 0x2;
};

struct p_movd2b
{
    constexpr static uint MOV_1_ROW  = 0x0;
    constexpr static uint MOV_4_ROWS = 0x1;
    constexpr static uint MOV_8_ROWS = 0x2;

    constexpr static uint SRC_ROW16_OFFSET = 0x10;
    constexpr static uint SRC_ROW32_OFFSET = 0x20;
};

struct p_movb2a
{
    constexpr static uint MOV_1_ROW   = 0x0;
    constexpr static uint MOV_4_ROWS  = 0x1;
    constexpr static uint MOV_8_ROWS  = 0x2;
    constexpr static uint MOV_16_ROWS = 0x3;

    constexpr static uint SRCA_ZERO_OFFSET  = 0x0;
    constexpr static uint SRCB_ROW16_OFFSET = 0x10;
    constexpr static uint SRCB_ROW32_OFFSET = 0x20;
};

struct p_mov_src_to_dest
{
    constexpr static uint MOV_1_ROW   = 0x0;
    constexpr static uint MOV_4_ROWS  = 0x1;
    constexpr static uint MOV_8_ROWS  = 0x2;
    constexpr static uint MOV_16_ROWS = 0x3;

    constexpr static uint SRC_ROW16_OFFSET = 0x10;
    constexpr static uint SRC_ROW32_OFFSET = 0x20;
};

struct p_stall
{
    // What to stall on
    constexpr static uint NOTHING         = 0;
    constexpr static uint THCON           = 1;
    constexpr static uint UNPACK0         = 2;
    constexpr static uint UNPACK0_DONE_RD = 3;
    constexpr static uint UNPACK1         = 4;
    constexpr static uint UNPACK1_DONE_RD = 5;
    constexpr static uint UNPACK2         = 6;
    constexpr static uint UNPACK2_DONE_RD = 7;
    constexpr static uint PACK0           = 8;
    constexpr static uint PACK0_DONE_WR   = 9;
    constexpr static uint PACK1           = 10;
    constexpr static uint PACK1_DONE_WR   = 11;
    constexpr static uint MATH            = 12;
    constexpr static uint SRCA_CLR        = 13;
    constexpr static uint SRCB_CLR        = 14;
    constexpr static uint UNPACK_SRCS_RDY = 15;
    constexpr static uint PACK_SRCS_RDY   = 16;
    constexpr static uint SRCA_VLD        = 17;
    constexpr static uint SRCB_VLD        = 18;
    constexpr static uint SFPU_SRCS_RDY   = 19;
    constexpr static uint XMOV            = 20;
    constexpr static uint TRISC_CFG       = 21;
    constexpr static uint SFPU1           = 22; // lol name collisions
    constexpr static uint CFGEXU          = 23;

    constexpr static uint WAIT_SFPU = SFPU1;
    constexpr static uint PACK      = PACK0;

    constexpr static struct
    {
        int dont_use_this;
    } ALL_THREAD_RES = {0}; // Use non-integer so that compiler throws an error if you try to use this

    // What to stall
    constexpr static uint STALL_TDMA   = 0x1;
    constexpr static uint STALL_SYNC   = 0x2;
    constexpr static uint STALL_PACK   = 0x4;
    constexpr static uint STALL_UNPACK = 0x8;
    // constexpr static uint STALL_XSEARCH     = 0x10;
    constexpr static uint STALL_XMOV   = 0x10;
    constexpr static uint STALL_THCON  = 0x20;
    constexpr static uint STALL_MATH   = 0x40;
    constexpr static uint STALL_CFG    = 0x80;
    constexpr static uint STALL_SFPU   = 0x100;
    constexpr static uint STALL_THREAD = 0x1ff;

    constexpr static uint STALL_ON_ZERO = 0x1;
    constexpr static uint STALL_ON_MAX  = 0x2;

    constexpr static uint SEMAPHORE_0 = 0x1;
    constexpr static uint SEMAPHORE_1 = 0x2;
    constexpr static uint SEMAPHORE_2 = 0x4;
    constexpr static uint SEMAPHORE_3 = 0x8;
    constexpr static uint SEMAPHORE_4 = 0x10;
    constexpr static uint SEMAPHORE_5 = 0x20;
    constexpr static uint SEMAPHORE_6 = 0x40;
    constexpr static uint SEMAPHORE_7 = 0x80;
};

struct p_zeroacc
{
    constexpr static uint CLR_SPECIFIC       = 0x0;
    constexpr static uint CLR_16             = 0x1;
    constexpr static uint CLR_HALF           = 0x2;
    constexpr static uint CLR_ALL            = 0x3;
    constexpr static uint CLR_STRIPED_Z_FACE = 0x4;
};

struct p_zerosrc
{
    constexpr static uint CLR_A  = 0x1;
    constexpr static uint CLR_B  = 0x2;
    constexpr static uint CLR_AB = 0x3;

    constexpr static uint CURR_BANK = 0x0;
    constexpr static uint ALL_BANKS = 0x1;

    constexpr static uint READ_BANK  = 0x1;
    constexpr static uint WRITE_BANK = 0x0;
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

    constexpr static uint CLR_NONE      = 0x0;
    constexpr static uint CLR_SRCA_VLD  = 0x1;
    constexpr static uint CLR_SRCB_VLD  = 0x2;
    constexpr static uint CLR_SRCAB_VLD = 0x3;
};

struct p_elwise
{
    constexpr static uint SRCB_NO_BCAST  = 0x0;
    constexpr static uint SRCB_BCAST_COL = 0x1;
    constexpr static uint SRCB_BCAST_ROW = 0x2;
    constexpr static uint SRCB_BCAST_ALL = 0x3;

    constexpr static uint DISABLE_ACCUM = 0x0;
    constexpr static uint ENABLE_ACCUM  = 0x1;

    constexpr static uint CLR_NONE      = 0x0;
    constexpr static uint CLR_SRCA_VLD  = 0x1;
    constexpr static uint CLR_SRCB_VLD  = 0x2;
    constexpr static uint CLR_SRCAB_VLD = 0x3;
};

struct p_sfpu
{
    constexpr static uint LREG0       = 0;
    constexpr static uint LREG1       = 1;
    constexpr static uint LREG2       = 2;
    constexpr static uint LREG3       = 3;
    constexpr static uint LREG4       = 4;
    constexpr static uint LREG5       = 5;
    constexpr static uint LREG6       = 6;
    constexpr static uint LREG7       = 7;
    constexpr static uint LCONST_0    = 9;
    constexpr static uint LCONST_1    = 10;
    constexpr static uint LCONST_neg1 = 11;

    struct sfpmem
    {
        constexpr static uint DEFAULT = 0b0000; // format is determined by combination of SrcB exponent width of ALU_FORMAT_SPEC_REG and also ACC_CTRL_SFPU_Fp32
        constexpr static uint FP16A   = 0b0001; // stored data will be interpreted as fp16 (fp16_a) format
        constexpr static uint FP16B   = 0b0010; // stored data will be interpreted as bfloat (fp16_b) format
        constexpr static uint FP32    = 0b0011; // stored data will be interpreted as fp32 format
        constexpr static uint INT32   = 0b0100; // stored data will be interpreted as int32 (sign + magnitude) format
        constexpr static uint UINT8   = 0b0101; // stored data will be interpreted as unsigned int8 format
        constexpr static uint UINT16  = 0b0110; // stored data will be interpreted as unsigned int16 format
                                                // TODO - Luka: add the other formats
    };

    struct mad_mode
    {
        constexpr static uint INDEX_ADDR_D  = 0x4;
        constexpr static uint INDEX_ADDR_A  = 0x8;
        constexpr static uint INDEX_ADDR_AD = 0xC;
    };

    struct col_offset
    {
        constexpr static uint EVEN_COL = 0x0;
        constexpr static uint ODD_COL  = 0x2;
    };

    struct cc
    {
        constexpr static uint SET_CC    = 0x2;
        constexpr static uint CLR_CC    = 0x1;
        constexpr static uint SET_CC_EN = 0x1;
        constexpr static uint CLR_CC_EN = 0x0;
    };

    struct sfp_stochrnd_mod
    {
        constexpr static uint FP32_TO_FP16A  = 0x0;
        constexpr static uint FP32_TO_FP16B  = 0x1;
        constexpr static uint FP32_TO_UINT8  = 0x2;
        constexpr static uint FP32_TO_INT8   = 0x3;
        constexpr static uint INT32_TO_UINT8 = 0x4;
        constexpr static uint INT32_TO_INT8  = 0x5;
        constexpr static uint FP32_TO_UINT16 = 0x6; // TODO: does Uint16 even exist?
        constexpr static uint FP32_TO_INT16  = 0x7;
    };

    struct sfp_stochrnd_rnd_mod
    {
        constexpr static uint NearEven   = 0x0;
        constexpr static uint Stochastic = 0x1;
        constexpr static uint RoundZero  = 0x2;
    };
};

struct p_cleardvalid
{
    constexpr static uint UNPACK_TO_DEST = 0b0001;
    constexpr static uint FPU            = 0b0010;
    constexpr static uint SFPU           = 0b0100;
    constexpr static uint PACK           = 0b1000;

    constexpr static uint CLR_NONE      = 0b00;
    constexpr static uint CLR_SRCA_VLD  = 0b01;
    constexpr static uint CLR_SRCB_VLD  = 0b10;
    constexpr static uint CLR_SRCAB_VLD = 0b11;
};

struct p_pacr
{
    constexpr static uint PACK0 = 0b011;
    constexpr static uint PACK1 = 0b100;

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

    constexpr static uint ZERO_WRITE    = 0b1;
    constexpr static uint NO_ZERO_WRITE = 0b0;

    constexpr static uint NO_CTXT_CTRL               = 0b00;
    constexpr static uint RTL_FLOPS_CTXT_SEL         = 0b01;
    constexpr static uint RTL_FLOPS_CTXT_RST_AND_NOP = 0b10;
    constexpr static uint RTL_FLOPS_CTXT_SEL_NO_RST  = 0b11;
};

struct p_ttsync
{
    // Resource bits.
    // These must line up with the definition of ttsync_instrn_rsrc_t in tt_tensix_pkg.sv .
    constexpr static uint CFG_RD = 1 << 0;
    constexpr static uint CFG_WR = 1 << 1;
    constexpr static uint GPR_RD = 1 << 2;
    constexpr static uint GPR_WR = 1 << 3;

    // Class numbers
    //<ttsync_class_numbers> //Please do not remove this line, it is used by gen_tt_instrn_resources_used.lua
    constexpr static uint DEFAULT_CLASS = 0; // Everything else
    constexpr static uint ATOMICS_CLASS = 1; // ATCAS, ATGETM, ATINCGET, ATINCGETPTR, ATRELM, ATSWAP
    constexpr static uint CFG_CLASS =
        2; // CFGSHIFTMASK, CFGSHIFTMASK_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE, RMWCIB0,
           // RMWCIB0_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE, RMWCIB1, RMWCIB1_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE, RMWCIB2,
           // RMWCIB2_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE, RMWCIB3_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE
    constexpr static uint GPR_CLASS     = 3;   // ADDGPR, BITWOPGPR, CMPGPR, MULGPR, SETGPR, SHIFTGPR, SUBGPR
    constexpr static uint LDSTIND_CLASS = 4;   // LOADIND, STOREIND
    constexpr static uint LOADREG_CLASS = 5;   // LOADREG
    constexpr static uint MOP_CLASS     = 6;   // MOP
    constexpr static uint PACK_CLASS    = 7;   // PACR0_FACE, PACR0_FACE_INC, PACR0_ROW, PACR0_ROW_INC, PACR0_TILE, PACR0_TILE_INC, PACR1_FACE, PACR1_FACE_INC,
                                               // PACR1_ROW, PACR1_ROW_INC, PACR1_TILE, PACR1_TILE_INC, PACR_STRIDE, PACR_UNTILIZE, RV_PACR
    constexpr static uint RDCFG_CLASS    = 8;  // RDCFG
    constexpr static uint REPLAY_CLASS   = 9;  // REPLAY
    constexpr static uint RV_WRCFG_CLASS = 10; // RV_WRCFG
    constexpr static uint SHADOW_CLASS   = 11; // COMMIT_SHADOW
    constexpr static uint STOREREG_CLASS = 12; // STOREREG
    constexpr static uint UNPACK_CLASS =
        13; // RV_UNPACR, UNPACR0_FACE, UNPACR0_FACE_INC, UNPACR0_ROW, UNPACR0_ROW_INC, UNPACR0_STRIDE, UNPACR0_TILE, UNPACR0_TILE_INC, UNPACR1_FACE,
            // UNPACR1_FACE_INC, UNPACR1_ROW, UNPACR1_ROW_INC, UNPACR1_STRIDE, UNPACR1_TILE, UNPACR1_TILE_INC, UNPACR2_FACE, UNPACR2_FACE_INC, UNPACR2_ROW,
            // UNPACR2_ROW_INC, UNPACR2_STRIDE, UNPACR2_TILE, UNPACR2_TILE_INC, UNPACR_DEST_FACE, UNPACR_DEST_FACE_INC, UNPACR_DEST_ROW, UNPACR_DEST_ROW_INC,
            // UNPACR_DEST_STRIDE, UNPACR_DEST_TILE, UNPACR_DEST_TILE_INC, UNPACR_NOP, UNPACR_TILE_MISC, UNPACR_TILIZE
    constexpr static uint WRCFG_CLASS = 14; // WRCFG
                                            //</ttsync_class_numbers> //Please do not remove this line, it is used by gen_tt_instrn_resources_used.lua
};

struct p_sfpnonlinear
{
    constexpr static uint RECIP_MODE = 0x0;
    constexpr static uint RELU_MODE  = 0x2;
    constexpr static uint SQRT_MODE  = 0x3;
    constexpr static uint EXP_MODE   = 0x4;
    constexpr static uint TANH_MODE  = 0x5;
};

} // namespace ckernel
