/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include "ckernel.h"

namespace ckernel
{

class ckernel_template
{

    // Here is the basic outline of a MOP loop and the definition of the
    // variables used.
    // LOOP_OUTER: <OUTER_LOOP_COUNT>
    //   START_OP
    //   LOOP_INNER: <INNER_LOOP_COUNT>
    //     LOOP_OP0
    //     LOOP_OP1
    //   END_LOOP_INNER
    //   END_OP0
    //   END_OP1
    // END_LOOP_OUTER

    const uint m_outer_loop_len;
    const uint m_inner_loop_len;
    uint m_loop_op0;
    uint m_loop_op1;
    uint m_end_op0, m_end_op1, m_start_op0;
    uint
        m_loop0_last_instr; // In the last iteration of the outer loop, this instruction replaces the inner loop instruction, if constructed with one inner loop instruction
                            // or the second inner loop instruction, if constructed with two inner loop instructions (see below example).
    uint
        m_loop1_last_instr; // In the last iteration of the inner loop, this instruction replaces the inner loop instruction, if constructed with one inner loop instruction
                            // or the second inner loop instruction, if constructed with two inner loop instructions (see below example).

    // Note: The last iteration of inner loop will also be the last iteration of the outer loop when outer loop length = 1.
    // This means that in this case, last_inner_loop_instr will be replaced by the last_outer_loop_instr
    // NOTE:
    // This is how m_loop0_last_instr and m_loop1_last_instr are executed:
    //
    // if(last_inner_loop_iter && last_outer_loop_iter) m_loop_op1 = m_loop0_last_instr;
    // else if(last_inner_loop_iter)                    m_loop_op1 = m_loop1_last_instr;
    // else                                             m_loop_op1 = m_loop_op1;

public:
    ckernel_template(uint outer_loop_len, uint inner_loop_len, uint loop_op);
    ckernel_template(uint outer_loop_len, uint inner_loop_len, uint loop_op0, uint loop_op1);
    void set_loop_op0(uint loop_op);
    void set_loop_op1(uint loop_op);
    void set_end_ops(uint end_op0, uint end_op1);
    void set_end_op(uint end_op0);
    void set_start_op(uint start_op0);
    void set_last_inner_loop_instr(uint op);
    void set_last_outer_loop_instr(uint op);

    void program(volatile uint *instrn_buffer);         // just programs the registers
    static void run(volatile uint *instrn_buffer);      // runs - assumes that registers were already programmed
    void program_and_run(volatile uint *instrn_buffer); // calls program, then run
};

class ckernel_unpack_template
{

    // Unpack template is a single-loop template that allows for some dynamic selection of instructions based on zmask
    // The basic template looks like this:
    //
    //  LOOP:
    //    if (zmask[iteration]):
    //      UNPACR_A0
    //      UNPACR_A1
    //      UNPACR_A2
    //      UNPACR_A3
    //      UNPACR_B
    //    else
    //      SKIP_A0
    //      SKIP_A1
    //      SKIP_A2
    //      SKIP_A3
    //      SKIP_B
    //
    //  The configuration allows the following changes:
    //    - B enable - if 0, removes the UNPACR_B/SKIP_B instruction
    //    - HALO enable - if 0, removes A1/2/3 instructions
    //    - Each of the UNPACR/SKIP instructions can be anything at all, although the most common use is for UNPACR to unpack, and SKIP to be an
    //      unpack NOP that increments the context counter

    const bool m_unpackB;
    const bool m_unpack_halo;

    const uint m_A0_instr, m_A1_instr, m_A2_instr, m_A3_instr;
    const uint m_B_instr;

    const uint m_skipA_instr;
    const uint m_skipB_instr;

public:
    ckernel_unpack_template(bool unpackB, bool unpackHalo, uint A0_instr, uint A1_instr, uint A2_instr, uint A3_instr, uint skipA_instr,

        uint B_instr, uint skipB_instr)
        : m_unpackB(unpackB)
        , m_unpack_halo(unpackHalo)
        , m_A0_instr(A0_instr)
        , m_A1_instr(A1_instr)
        , m_A2_instr(A2_instr)
        , m_A3_instr(A3_instr)
        , m_B_instr(B_instr)
        , m_skipA_instr(skipA_instr)
        , m_skipB_instr(skipB_instr){};

public:
    // Default ZeroSrcA UNPACR_NOP
    static constexpr uint DEF_ZEROSRCA = TT_OP_UNPACR_NOP(p_unpacr_nop::UNP0, p_unpacr_nop::UNP_ZEROSRC);
    static constexpr uint DEF_NINFSRCA = TT_OP_UNPACR_NOP(p_unpacr_nop::UNP0, p_unpacr_nop::UNP_NEGINFSRC);
    static constexpr uint DEF_UNPACR_NOP = TT_OP_UNPACR_NOP(p_unpacr_nop::UNP0, p_unpacr_nop::UNP_NOP);

    // Default skip A/B instructions that increment Z counters by 1
    static constexpr uint DEF_SKIP_A = TT_OP_INCADCZW(0b001, 0, 0, 0, 1);
    static constexpr uint DEF_SKIP_B = TT_OP_INCADCZW(0b010, 0, 0, 0, 1);

    // Default non-halo A instruction
    static constexpr uint DEF_A_instr = TT_OP_UNPACR(0, 0b1, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_A_cntx_ovrd_instr = TT_OP_UNPACR(0, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    // Default B instruction with rarefy
    static constexpr uint DEF_B_rarefy_instr = TT_OP_UNPACR(1, 0b01, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_ENABLE, 0, 0, 0, 0, 1);

    // Default B instruction with rarefy and context override, coupled with halo on A
    static constexpr uint DEF_B_rarefy_cntx_ovrd_instr = TT_OP_UNPACR(1, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_ENABLE, 0, 0, 0, 0, 1);

    // Default B instruction without rarefy, no z increment and with context override, coupled with halo on A for factored conv
    static constexpr uint DEF_B_cntx_ovrd_no_z_inc_instr = TT_OP_UNPACR(1, 0b00, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    // Default B instruction without rarefy and context override
    static constexpr uint DEF_B_cntx_ovrd_instr = TT_OP_UNPACR(1, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    // Default B instruction without rarefy
    static constexpr uint DEF_B_instr = TT_OP_UNPACR(1, 0b01, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    // Deafult halo A instructions
    static constexpr uint DEF_A0_instr = TT_OP_UNPACR(
        0, 0b00, 0, p_unpacr::TILE0_CFG_CONTEXT, p_unpacr::TILE0_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);
    static constexpr uint DEF_A1_instr = TT_OP_UNPACR(
        0, 0b00, 0, p_unpacr::TILE1_CFG_CONTEXT, p_unpacr::TILE1_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);
    static constexpr uint DEF_A2_instr = TT_OP_UNPACR(
        0, 0b00, 0, p_unpacr::TILE2_CFG_CONTEXT, p_unpacr::TILE2_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);
    static constexpr uint DEF_A3_instr = TT_OP_UNPACR(
        0, 0b01, 0, p_unpacr::TILE3_CFG_CONTEXT, p_unpacr::TILE3_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);

    // Special case where all later strips are skipped, so this one has to set DVALID because it is last, and increment Z
    static constexpr uint DEF_A0_last_instr = TT_OP_UNPACR(
        0, 0b01, 0, p_unpacr::TILE0_CFG_CONTEXT, p_unpacr::TILE0_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);
    static constexpr uint DEF_A1_last_instr = TT_OP_UNPACR(
        0, 0b01, 0, p_unpacr::TILE1_CFG_CONTEXT, p_unpacr::TILE1_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);
    static constexpr uint DEF_A2_last_instr = TT_OP_UNPACR(
        0, 0b01, 0, p_unpacr::TILE2_CFG_CONTEXT, p_unpacr::TILE2_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);

    // Halo A instructions that skip actual unpacking, but increment context
    static constexpr uint SKIP_A0_instr = TT_OP_UNPACR(
        0, 0b00, 1, p_unpacr::TILE0_CFG_CONTEXT, p_unpacr::TILE0_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);
    static constexpr uint SKIP_A1_instr = TT_OP_UNPACR(
        0, 0b00, 1, p_unpacr::TILE1_CFG_CONTEXT, p_unpacr::TILE1_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);
    static constexpr uint SKIP_A2_instr = TT_OP_UNPACR(
        0, 0b00, 1, p_unpacr::TILE2_CFG_CONTEXT, p_unpacr::TILE2_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);
    static constexpr uint SKIP_A3_instr = TT_OP_UNPACR(
        0, 0b00, 1, p_unpacr::TILE3_CFG_CONTEXT, p_unpacr::TILE3_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);

    // Factored conv halo A instructions
    static constexpr uint DEF_A0_fconv_instr = TT_OP_UNPACR(
        0, 0b00, 0, p_unpacr::TILE0_CFG_CONTEXT, p_unpacr::TILE0_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);
    static constexpr uint DEF_A1_fconv_instr = TT_OP_UNPACR(
        0, 0b00, 0, p_unpacr::TILE1_CFG_CONTEXT, p_unpacr::TILE1_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);
    static constexpr uint DEF_A2_fconv_instr = TT_OP_UNPACR(
        0, 0b00, 0, p_unpacr::TILE2_CFG_CONTEXT, p_unpacr::TILE2_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);
    static constexpr uint DEF_A3_fconv_instr = TT_OP_UNPACR(
        0, 0b00, 0, p_unpacr::TILE3_CFG_CONTEXT, p_unpacr::TILE3_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);

    // Special case where all later strips are skipped, so this one has to set DVALID because it is last, and increment Z (factored conv)
    static constexpr uint DEF_A0_fconv_last_instr = TT_OP_UNPACR(
        0, 0b00, 0, p_unpacr::TILE0_CFG_CONTEXT, p_unpacr::TILE0_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);
    static constexpr uint DEF_A1_fconv_last_instr = TT_OP_UNPACR(
        0, 0b00, 0, p_unpacr::TILE1_CFG_CONTEXT, p_unpacr::TILE1_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);
    static constexpr uint DEF_A2_fconv_last_instr = TT_OP_UNPACR(
        0, 0b00, 0, p_unpacr::TILE2_CFG_CONTEXT, p_unpacr::TILE2_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 1, 0, 1);

    static constexpr uint DEF_Strip0_instr = TT_OP_UNPACR(
        0, 0, 0, 0, p_unpacr::TILE0_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip1_instr = TT_OP_UNPACR(
        0, 0, 0, 1, p_unpacr::TILE1_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip2_instr = TT_OP_UNPACR(
        0, 0, 0, 2, p_unpacr::TILE2_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip3_instr = TT_OP_UNPACR(
        0, 0, 0, 3, p_unpacr::TILE3_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    static constexpr uint DEF_Strip0_last_instr = TT_OP_UNPACR(
        0, 1, 0, 0, p_unpacr::TILE0_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip1_last_instr = TT_OP_UNPACR(
        0, 1, 0, 1, p_unpacr::TILE1_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip2_last_instr = TT_OP_UNPACR(
        0, 1, 0, 2, p_unpacr::TILE2_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip3_last_instr = TT_OP_UNPACR(
        0, 1, 0, 3, p_unpacr::TILE3_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    static constexpr uint DEF_Strip0_data_valid_instr = TT_OP_UNPACR(
        0, 0, 0, 0, p_unpacr::TILE0_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip1_data_valid_instr = TT_OP_UNPACR(
        0, 0, 0, 1, p_unpacr::TILE1_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip2_data_valid_instr = TT_OP_UNPACR(
        0, 0, 0, 2, p_unpacr::TILE2_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip3_data_valid_instr = TT_OP_UNPACR(
        0, 0, 0, 3, p_unpacr::TILE3_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    //
    // Convenience factory methods
    //
    static ckernel_unpack_template lzA(bool neginf, uint A_instr = DEF_A_cntx_ovrd_instr, uint skipA_instr = DEF_SKIP_A);

    static ckernel_unpack_template lA(uint A_instr = DEF_A_cntx_ovrd_instr, uint skipA_instr = DEF_SKIP_A);

    static ckernel_unpack_template lB(uint B_instr = DEF_B_cntx_ovrd_instr, uint skipB_instr = DEF_SKIP_B);

    static ckernel_unpack_template lhA(const uint32_t halo_mask);

    static ckernel_unpack_template flhA(const uint32_t halo_mask);

    static ckernel_unpack_template lBhA(const uint32_t halo_mask, const bool rarefy = true);

    static ckernel_unpack_template flBhA(const uint32_t halo_mask);

    static ckernel_unpack_template lBA(uint A_instr = DEF_A_instr, uint skipA_instr = DEF_SKIP_A,

        uint B_instr = DEF_B_instr, uint skipB_instr = DEF_SKIP_B);

    // More abstraction to re-use above templates for kernel to run loop of N instructions
    static ckernel_unpack_template loopx1instr(uint instr0, uint skip0 = TT_OP_NOP);
    static ckernel_unpack_template loopx2instr(uint instr0, uint instr1, uint skip0 = TT_OP_NOP, uint skip1 = TT_OP_NOP);

    void program(volatile uint *instrn_buffer) const;                                                  // just programs the registers
    static void run(volatile uint *instrn_buffer, const uint8_t count, const uint32_t zmask);          // runs - assumes that registers were already programmed
    static void run(volatile uint *instrn_buffer, const uint8_t count);                                // runs - assumes that registers were already programmed
    void program_and_run(volatile uint *instrn_buffer, const uint8_t count, const uint32_t zmask = 0); // calls program, then run
};

} // namespace ckernel
