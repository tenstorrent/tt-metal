// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
    uint m_loop0_last_instr; // In the last iteration of the outer loop, this instruction replaces the inner loop instruction, if constructed with one inner
                             // loop instruction or the second inner loop instruction, if constructed with two inner loop instructions (see below example).
    uint m_loop1_last_instr; // In the last iteration of the inner loop, this instruction replaces the inner loop instruction, if constructed with one inner
                             // loop instruction or the second inner loop instruction, if constructed with two inner loop instructions (see below example).

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
    ckernel_unpack_template(
        bool unpackB,
        bool unpackHalo,
        uint A0_instr,
        uint A1_instr,
        uint A2_instr,
        uint A3_instr,
        uint skipA_instr,

        uint B_instr,
        uint skipB_instr) :
        m_unpackB(unpackB),
        m_unpack_halo(unpackHalo),
        m_A0_instr(A0_instr),
        m_A1_instr(A1_instr),
        m_A2_instr(A2_instr),
        m_A3_instr(A3_instr),
        m_B_instr(B_instr),
        m_skipA_instr(skipA_instr),
        m_skipB_instr(skipB_instr)
    {
    }

public:
    // Default ZeroSrcA UNPACR_NOP
    static constexpr uint DEF_ZEROSRCA   = TT_OP_UNPACR_NOP(p_unpacr_nop::UNP0, p_unpacr_nop::UNP_ZEROSRC);
    static constexpr uint DEF_NINFSRCA   = TT_OP_UNPACR_NOP(p_unpacr_nop::UNP0, p_unpacr_nop::UNP_NEGINFSRC);
    static constexpr uint DEF_UNPACR_NOP = TT_OP_UNPACR_NOP(p_unpacr_nop::UNP0, p_unpacr_nop::UNP_NOP);

    // Default skip A/B instructions that increment Z counters by 1
    static constexpr uint DEF_SKIP_A = TT_OP_INCADCZW(0b001, 0, 0, 0, 1);
    static constexpr uint DEF_SKIP_B = TT_OP_INCADCZW(0b010, 0, 0, 0, 1);

    // Default non-halo A instruction
    static constexpr uint DEF_A_instr           = TT_OP_UNPACR(0, 0b1, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
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

    static constexpr uint DEF_Strip0_instr = TT_OP_UNPACR(0, 0, 0, 0, p_unpacr::TILE0_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip1_instr = TT_OP_UNPACR(0, 0, 0, 1, p_unpacr::TILE1_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip2_instr = TT_OP_UNPACR(0, 0, 0, 2, p_unpacr::TILE2_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip3_instr = TT_OP_UNPACR(0, 0, 0, 3, p_unpacr::TILE3_ADDRCNT_CONTEXT, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    static constexpr uint DEF_Strip0_last_instr = TT_OP_UNPACR(0, 1, 0, 0, p_unpacr::TILE0_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip1_last_instr = TT_OP_UNPACR(0, 1, 0, 1, p_unpacr::TILE1_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip2_last_instr = TT_OP_UNPACR(0, 1, 0, 2, p_unpacr::TILE2_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip3_last_instr = TT_OP_UNPACR(0, 1, 0, 3, p_unpacr::TILE3_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    static constexpr uint DEF_Strip0_data_valid_instr =
        TT_OP_UNPACR(0, 0, 0, 0, p_unpacr::TILE0_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip1_data_valid_instr =
        TT_OP_UNPACR(0, 0, 0, 1, p_unpacr::TILE1_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip2_data_valid_instr =
        TT_OP_UNPACR(0, 0, 0, 2, p_unpacr::TILE2_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_Strip3_data_valid_instr =
        TT_OP_UNPACR(0, 0, 0, 3, p_unpacr::TILE3_ADDRCNT_CONTEXT, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

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

    static ckernel_unpack_template lBA(
        uint A_instr     = DEF_A_instr,
        uint skipA_instr = DEF_SKIP_A,

        uint B_instr     = DEF_B_instr,
        uint skipB_instr = DEF_SKIP_B);

    // More abstraction to re-use above templates for kernel to run loop of N instructions
    static ckernel_unpack_template loopx1instr(uint instr0, uint skip0 = TT_OP_NOP);
    static ckernel_unpack_template loopx2instr(uint instr0, uint instr1, uint skip0 = TT_OP_NOP, uint skip1 = TT_OP_NOP);

    void program(volatile uint *instrn_buffer) const;                                                  // just programs the registers
    static void run(volatile uint *instrn_buffer, const uint8_t count, const uint32_t zmask);          // runs - assumes that registers were already programmed
    static void run(volatile uint *instrn_buffer, const uint8_t count);                                // runs - assumes that registers were already programmed
    void program_and_run(volatile uint *instrn_buffer, const uint8_t count, const uint32_t zmask = 0); // calls program, then run
};

inline ckernel_template::ckernel_template(uint outer_loop_len, uint inner_loop_len, uint loop_op) :
    m_outer_loop_len(outer_loop_len),
    m_inner_loop_len(inner_loop_len),
    m_loop_op0(loop_op),
    m_loop_op1(TT_OP_NOP),
    m_end_op0(TT_OP_NOP),
    m_end_op1(TT_OP_NOP),
    m_start_op0(TT_OP_NOP)
{
    m_loop0_last_instr = loop_op;
    m_loop1_last_instr = loop_op;
}

inline ckernel_template::ckernel_template(uint outer_loop_len, uint inner_loop_len, uint loop_op0, uint loop_op1) :
    m_outer_loop_len(outer_loop_len),
    m_inner_loop_len(inner_loop_len),
    m_loop_op0(loop_op0),
    m_loop_op1(loop_op1),
    m_end_op0(TT_OP_NOP),
    m_end_op1(TT_OP_NOP),
    m_start_op0(TT_OP_NOP)
{
    m_loop0_last_instr = loop_op1;
    m_loop1_last_instr = loop_op1;
}

inline void ckernel_template::set_loop_op0(uint loop_op)
{
    m_loop_op0 = loop_op;
}

inline void ckernel_template::set_loop_op1(uint loop_op)
{
    m_loop_op1 = loop_op;
}

inline void ckernel_template::set_end_ops(uint end_op0, uint end_op1)
{
    m_end_op0 = end_op0;
    m_end_op1 = end_op1;
}

inline void ckernel_template::set_end_op(uint end_op0)
{
    set_end_ops(end_op0, TT_OP_NOP);
}

inline void ckernel_template::set_start_op(uint start_op0)
{
    m_start_op0 = start_op0;
}

inline void ckernel_template::set_last_inner_loop_instr(uint op)
{
    m_loop1_last_instr = op;
}

inline void ckernel_template::set_last_outer_loop_instr(uint op)
{
    m_loop0_last_instr = op;
}

inline void ckernel_template::program_and_run(volatile uint *instrn_buffer)
{
    program(instrn_buffer);
    run(instrn_buffer);
}

inline void ckernel_template::run(volatile uint *instrn_buffer)
{
    TTI_MOP(1, 0, 0); // run the double-loop template
}

inline void ckernel_template::program(volatile uint *instrn_buffer)
{
    volatile uint *mop_cfg = reinterpret_cast<volatile uint *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[0] = m_outer_loop_len;
    mop_cfg[1] = m_inner_loop_len;
    mop_cfg[2] = m_start_op0;
    mop_cfg[3] = m_end_op0;
    mop_cfg[4] = m_end_op1;
    mop_cfg[5] = m_loop_op0;
    mop_cfg[6] = m_loop_op1;
    mop_cfg[7] = m_loop0_last_instr;
    mop_cfg[8] = m_loop1_last_instr;
}

inline void ckernel_unpack_template::program_and_run(volatile uint *instrn_buffer, const uint8_t count, const uint32_t zmask)
{
    program(instrn_buffer);
    run(instrn_buffer, count, zmask);
}

inline void ckernel_unpack_template::run(volatile uint *instrn_buffer, const uint8_t count, const uint32_t zmask)
{
    FWASSERT("Unpack template only supports loops up to 128", count <= 128);
    TT_MOP_CFG(zmask >> 16);              // Set the top 16 bits of zmask - we could skip this for count <= 16
    TT_MOP(0, count - 1, zmask & 0xFFFF); // Run the template
}

// Version without zmask, should be slightly faster by eliminating one instruction.
inline void ckernel_unpack_template::run(volatile uint *instrn_buffer, const uint8_t count)
{
    FWASSERT("Unpack template only supports loops up to 128", count <= 128);
    TT_MOP(0, count - 1, 0); // Run the template
}

inline void ckernel_unpack_template::program(volatile uint *instrn_buffer) const
{
    volatile uint *mop_cfg = reinterpret_cast<volatile uint *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[1] = m_unpackB | (m_unpack_halo << 1);
    mop_cfg[2] = m_B_instr;
    mop_cfg[3] = m_A0_instr;
    mop_cfg[4] = m_A1_instr;
    mop_cfg[5] = m_A2_instr;
    mop_cfg[6] = m_A3_instr;
    mop_cfg[7] = m_skipA_instr;
    mop_cfg[8] = m_skipB_instr;
}

inline ckernel_unpack_template ckernel_unpack_template::lA(uint A_instr, uint skipA_instr)
{
    return ckernel_unpack_template(
        false, // src B
        false, // halo
        A_instr,
        0,
        0,
        0,
        skipA_instr,
        0,
        0);
}

inline ckernel_unpack_template ckernel_unpack_template::lB(uint B_instr, uint skipB_instr)
{
    return ckernel_unpack_template(
        false, // src B
        false, // halo
        B_instr,
        0,
        0,
        0,
        skipB_instr,
        0,
        0);
}

inline ckernel_unpack_template ckernel_unpack_template::lzA(bool neginf, uint A_instr, uint skipA_instr)
{
    return ckernel_unpack_template(
        false, // src B
        true,  // halo
        neginf ? DEF_NINFSRCA : DEF_ZEROSRCA,
        A_instr,
        DEF_UNPACR_NOP,
        DEF_UNPACR_NOP,
        skipA_instr,
        0,
        0);
}

inline ckernel_unpack_template ckernel_unpack_template::lhA(const uint32_t halo_mask)
{
    // Figure out which unpack is last
    const uint last_mask = (halo_mask == 0x1) ? 0x1 : (halo_mask <= 0x3) ? 0x2 : (halo_mask <= 0x7) ? 0x4 : 0;

    return ckernel_unpack_template(
        false, // src B
        true,  // halo
        ((halo_mask >> 0) & 0x1) ? ((last_mask >> 0) & 0x1) ? DEF_A0_last_instr : DEF_A0_instr : SKIP_A0_instr,
        ((halo_mask >> 1) & 0x1) ? ((last_mask >> 1) & 0x1) ? DEF_A1_last_instr : DEF_A1_instr : SKIP_A1_instr,
        ((halo_mask >> 2) & 0x1) ? ((last_mask >> 2) & 0x1) ? DEF_A2_last_instr : DEF_A2_instr : SKIP_A2_instr,
        ((halo_mask >> 3) & 0x1) ? DEF_A3_instr : SKIP_A3_instr,
        DEF_SKIP_A,
        0,
        0);
}

inline ckernel_unpack_template ckernel_unpack_template::flhA(const uint32_t halo_mask)
{
    // Figure out which unpack is last
    const uint last_mask = (halo_mask == 0x1) ? 0x1 : (halo_mask <= 0x3) ? 0x2 : (halo_mask <= 0x7) ? 0x4 : 0;

    return ckernel_unpack_template(
        false, // src B
        true,  // halo
        ((halo_mask >> 0) & 0x1) ? ((last_mask >> 0) & 0x1) ? DEF_A0_fconv_last_instr : DEF_A0_fconv_instr : SKIP_A0_instr,
        ((halo_mask >> 1) & 0x1) ? ((last_mask >> 1) & 0x1) ? DEF_A1_fconv_last_instr : DEF_A1_fconv_instr : SKIP_A1_instr,
        ((halo_mask >> 2) & 0x1) ? ((last_mask >> 2) & 0x1) ? DEF_A2_fconv_last_instr : DEF_A2_fconv_instr : SKIP_A2_instr,
        ((halo_mask >> 3) & 0x1) ? DEF_A3_fconv_instr : SKIP_A3_instr,
        TT_OP_NOP,
        0,
        0);
}

inline ckernel_unpack_template ckernel_unpack_template::lBhA(const uint32_t halo_mask, const bool rarefy)
{
    // Figure out which unpack is last
    const uint last_mask = (halo_mask == 0x1) ? 0x1 : (halo_mask <= 0x3) ? 0x2 : (halo_mask <= 0x7) ? 0x4 : 0;

    return ckernel_unpack_template(
        true, // src B
        true, // halo
        ((halo_mask >> 0) & 0x1) ? ((last_mask >> 0) & 0x1) ? DEF_A0_last_instr : DEF_A0_instr : SKIP_A0_instr,
        ((halo_mask >> 1) & 0x1) ? ((last_mask >> 1) & 0x1) ? DEF_A1_last_instr : DEF_A1_instr : SKIP_A1_instr,
        ((halo_mask >> 2) & 0x1) ? ((last_mask >> 2) & 0x1) ? DEF_A2_last_instr : DEF_A2_instr : SKIP_A2_instr,
        ((halo_mask >> 3) & 0x1) ? DEF_A3_instr : SKIP_A3_instr,
        DEF_SKIP_A,
        rarefy ? DEF_B_rarefy_cntx_ovrd_instr : DEF_B_cntx_ovrd_instr,
        DEF_SKIP_B);
}

inline ckernel_unpack_template ckernel_unpack_template::flBhA(const uint32_t halo_mask)
{
    // Figure out which unpack is last
    const uint last_mask = (halo_mask == 0x1) ? 0x1 : (halo_mask <= 0x3) ? 0x2 : (halo_mask <= 0x7) ? 0x4 : 0;

    return ckernel_unpack_template(
        true, // src B
        true, // halo
        ((halo_mask >> 0) & 0x1) ? ((last_mask >> 0) & 0x1) ? DEF_A0_fconv_last_instr : DEF_A0_fconv_instr : SKIP_A0_instr,
        ((halo_mask >> 1) & 0x1) ? ((last_mask >> 1) & 0x1) ? DEF_A1_fconv_last_instr : DEF_A1_fconv_instr : SKIP_A1_instr,
        ((halo_mask >> 2) & 0x1) ? ((last_mask >> 2) & 0x1) ? DEF_A2_fconv_last_instr : DEF_A2_fconv_instr : SKIP_A2_instr,
        ((halo_mask >> 3) & 0x1) ? DEF_A3_fconv_instr : SKIP_A3_instr,
        TT_OP_NOP,
        DEF_B_cntx_ovrd_no_z_inc_instr,
        DEF_SKIP_B);
}

inline ckernel_unpack_template ckernel_unpack_template::lBA(
    uint A_instr,
    uint skipA_instr,

    uint B_instr,
    uint skipB_instr)
{
    return ckernel_unpack_template(
        true,  // src B
        false, // halo
        A_instr,
        0,
        0,
        0,
        skipA_instr,
        B_instr,
        skipB_instr);
}

inline ckernel_unpack_template ckernel_unpack_template::loopx1instr(uint instr0, uint skip0)
{
    return ckernel_unpack_template::lA(instr0, skip0);
}

inline ckernel_unpack_template ckernel_unpack_template::loopx2instr(uint instr0, uint instr1, uint skip0, uint skip1)
{
    // Note - 2 instr loop so we will hijack B_instr slot for 2nd instruction via lBA.
    return ckernel_unpack_template::lBA(instr0, skip0, instr1, skip1);
}

} // namespace ckernel
