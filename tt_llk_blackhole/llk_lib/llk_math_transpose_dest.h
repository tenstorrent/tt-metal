// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_math_common.h"

using namespace ckernel;

// local function declarations
template <bool is_32bit>
inline void transpose_dest_configure_addrmod();
template <bool transpose_of_faces, bool is_32bit>
inline void transpose_dest_configure_mop();

// Notes on these template parameters:
// 1. <transpose_of_faces=false, is_32bit=false>: not supported.
// 2. <transpose_of_faces=false, is_32bit=true>: 4x 16x16 face transpose; can be combined with _llk_unpack_A_ with transpose_of_faces=true.
// 3. <transpose_of_faces=true, is_32bit=false>: the default case (full 32x32 tile transpose, non-32-bit).
// 4. <transpose_of_faces=true, is_32bit=true>: full 32x32 tile transpose for 32-bit.
//
// We may want to revisit these template parameters, and perhaps the
// transpose_dest API generally as it's not currently widely used:
// https://github.com/tenstorrent/tt-llk/issues/290
template <bool is_fp32_dest_acc_en, bool transpose_of_faces = true, bool is_32bit = false>
inline void _llk_math_transpose_dest_(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
    math::reset_counters(p_setrwc::SET_ABD_F);

    // Wait condition SRCA_VLD is required as MOVB2A doesn't automatically wait
    // for SrcA[MatrixUnit.SrcABank].AllowedClient == SrcClient::MatrixUnit.
    // Wait condition SRCB_VLD is required as MOVD2B doesn't automatically wait
    // for SrcB[MatrixUnit.SrcBBank].AllowedClient == SrcClient::MatrixUnit.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);

    if constexpr (is_32bit)
    {
        if constexpr (is_fp32_dest_acc_en)
        {
            // Needs to be disabled for MOVD2B/B2D on BH (Issue ##449)
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
        }
        if constexpr (transpose_of_faces)
        {
            // 4x 32b face transpositions followed by 8x middle-face row swaps.
            ckernel_unpack_template::run(instrn_buffer, 12, 0xff0);
        }
        else
        {
            // 4x 32b face transpositions.
            ckernel_unpack_template::run(instrn_buffer, 4, 0);
        }
        if constexpr (is_fp32_dest_acc_en)
        {
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
        }
    }
    else
    {
        ckernel_unpack_template::run(instrn_buffer, 2, 2);
    }

    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

template <bool is_32bit>
inline void transpose_dest_configure_addrmod()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 16},
    }
        .set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = is_32bit ? 2 : 0x3ff & -16},
    }
        .set(ADDR_MOD_2);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 32},
    }
        .set(ADDR_MOD_3);
}

template <bool transpose_of_faces, bool is_32bit>
inline void transpose_dest_configure_mop()
{
    if (is_32bit)
    {
        load_replay_buf(
            16,
            16,
            []
            {
#pragma GCC unroll 2
                for (int dest_32b_lo = 0; dest_32b_lo < 2; ++dest_32b_lo)
                {
                    TTI_MOVD2B(dest_32b_lo, 16, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
                    TTI_MOVD2B(dest_32b_lo, 20, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 4);
                    TTI_MOVD2B(dest_32b_lo, 24, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 8);
                    TTI_MOVD2B(dest_32b_lo, 28, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12);
                    TTI_MOVB2D(dest_32b_lo, 16, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0);
                    TTI_MOVB2D(dest_32b_lo, 20, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 4);
                    TTI_MOVB2D(dest_32b_lo, 24, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 8);
                    TTI_MOVB2D(dest_32b_lo, 28, dest_32b_lo == 1 ? ADDR_MOD_0 : ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 12);
                }
            });

        uint macro0 = TT_OP_SFPNOP;
        uint macro1 = TT_OP_SFPNOP;

        if (transpose_of_faces)
        {
            // Macro 0: SFPLOAD VD; SFPMOV LReg[16],VD; SFPSTORE LReg[0].
            // Intended for use with VD=LReg[1].

            // Set InstructionTemplate[0] to SFPMOV.
            TTI_SFPMOV(0, 0, 12, 0);

            // StoreSubUnit: schedule SFPSTORE with VD=0 after 1 cycle.
            TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (0x80 | (1 << 3) | 3) << 8);
            // SimpleSubUnit: schedule SFPMOV LReg[16],VD after 0 cycles.
            TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (0x40 | 4) << 0);
            // Set Sequence[0].
            TTI_SFPCONFIG(0, 4, 0);

            // Macro 1: SFPLOAD VD; delay; SFPSTORE LReg[16].
            // Intended for use with VD=LReg[0].

            // StoreSubUnit: schedule SFPSTORE with VD=LReg[16] after 1 cycle.
            TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (0x40 | (1 << 3) | 3) << 8);
            // Other sub-units: nothing scheduled.
            TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0);
            // Set Sequence[1].
            TTI_SFPCONFIG(0, 5, 0);

            // Misc: {UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1} for macros 0 and 1.
            TTI_SFPCONFIG(0x330, 8, 1);

            // Macro 0: SFPLOAD LReg[1], 16 (addr_mod_1); SFPMOV LReg[16],LReg[1]; SFPSTORE LReg[0].
            // Note: 0x3ff mask is used to ensure negative offset value is 10 bits.
            macro0 = TT_OP_SFPLOADMACRO((0 << 2) | 1, 4, ADDR_MOD_1, 0x3ff & -48);

            // Macro 1: SFPLOAD LReg[0], 32 (addr_mod_2); delay; SFPSTORE LReg[16].
            // Note: 0x3ff mask is used to ensure negative offset value is 10 bits.
            macro1 = TT_OP_SFPLOADMACRO((1 << 2) | 0, 4, ADDR_MOD_2, 0x3ff & -32);
        }

        // A 32b face transpose consists of: (movd2b_hi, transpose, movb2d_hi_d2b_lo, transpose, movb2d_lo).
        uint movd2b_hi        = lltt::replay_insn(16, 4);
        uint movb2d_hi_d2b_lo = lltt::replay_insn(20, 8);
        uint movb2d_lo        = lltt::replay_insn(28, 4);
        uint transpose        = TT_OP_TRNSPSRCB;

        // MOP config:
        // - zmask 0-bits: 32b 16x16 face transpose.
        // - zmask 1-bits: 32b 32x1 middle face row swap via SFPU.
        ckernel_unpack_template tmp(true, true, movd2b_hi, transpose, movb2d_hi_d2b_lo, transpose, /* skip A */ macro0, /* B */ movb2d_lo, /* skip B */ macro1);
        tmp.program(instrn_buffer);
    }
    else
    {
        load_replay_buf(
            16,
            15,
            []
            {
                // ABCD
                TTI_MOVB2A(0, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 16);
                TTI_MOVB2A(4, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 20);
                TTI_MOVB2A(8, ADDR_MOD_1, p_movb2a::MOV_4_ROWS, 24);
                TTI_MOVB2A(12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, 28); // dst += 16

                // EFGHIJKLM
                TTI_MOVD2B(0, 16, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
                TTI_MOVD2B(0, 20, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 4);
                TTI_MOVD2B(0, 24, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 8);
                TTI_MOVD2B(0, 28, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12);
                TTI_TRNSPSRCB;
                TTI_MOVB2D(0, 16, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0);
                TTI_MOVB2D(0, 20, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 4);
                TTI_MOVB2D(0, 24, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 8);
                TTI_MOVB2D(0, 28, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 12); // dst += 16

                // NO
                // Note: the 0x3ff mask ensures the negative offsets are 10 bits.
                TTI_MOVA2D(0, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0x3ff & (0 - 32)); // dst -= 16
                TTI_MOVA2D(0, 8, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0x3ff & (8 - 16)); // dst -= 16
            });

        // The next 7 instructions expand as follows:
        // Face 0: 4x MOVD2B, TRNSPSRCB, 4x MOVB2D, dst += 16 (EFGHIJKLM)
        // Face 1: 4x MOVD2B, TRNSPSRCB, 4x MOVB2A, dst += 16 (EFGHI, ABCD..)
        // Face 2: 4x MOVD2B (dst -= 16), TRNSPSRCB, 4x MOVB2D, dst += 32 (..EFG, P, IJKL, Q)
        // Face 3: 4x MOVD2B, TRNSPSRCB, 4x MOVB2D, dst += 16 (EFGHIJKLM..)
        // Face 1: 2x MOVA2D (2x dst -= 16) (..NO)

        uint EFGHIJKLM   = lltt::replay_insn(20, 9);
        uint EFGHI       = lltt::replay_insn(20, 5);
        uint ABCDEFG     = lltt::replay_insn(16, 7);
        uint P           = TT_OP_MOVD2B(0, 28, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, 12); // dst -= 16
        uint IJKL        = lltt::replay_insn(24, 4);
        uint Q           = TT_OP_MOVB2D(0, 28, ADDR_MOD_3, p_movb2d::MOV_4_ROWS, 12); // dst += 32
        uint EFGHIJKLMNO = lltt::replay_insn(20, 11);

        // The following MOP config simply runs the above 7 instructions in order (when executed with zmask 0b10):
        ckernel_unpack_template tmp(true, true, EFGHIJKLM, EFGHI, ABCDEFG, P, /* skip A */ Q, /* B */ IJKL, /* skip B */ EFGHIJKLMNO);
        tmp.program(instrn_buffer);
    }
}

template <bool transpose_of_faces = true, bool is_32bit = false>
inline void _llk_math_transpose_dest_init_()
{
    transpose_dest_configure_addrmod<is_32bit>();
    transpose_dest_configure_mop<transpose_of_faces, is_32bit>();

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
}
