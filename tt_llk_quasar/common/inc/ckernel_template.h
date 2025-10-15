// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ckernel.h"
#include "ckernel_addrmod.h"

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

    uint m_outer_loop_len;
    uint m_inner_loop_len;
    uint m_loop_op0;
    uint m_loop_op1;
    uint m_end_op0, m_end_op1, m_start_op0;
    uint m_loop0_last_instr; // In the last iteration of the outer loop, this instruction replaces the inner loop instruction, if constructed with one inner
                             // loop instruction or the second inner loop instruction, if constructed with two inner loop instructions.
    uint m_loop1_last_instr; // In the last iteration of the inner loop, this instruction replaces the inner loop instruction, if constructed with one inner
                             // loop instruction or the second inner loop instruction, if constructed with two inner loop instructions.

    // Note: The last iteration of inner loop will also be the last iteration of the outer loop when outer loop length = 1.
    // This means that in this case, last_inner_loop_instr will be replaced by the last_outer_loop_instr

public:
    ckernel_template(uint outer_loop_len, uint inner_loop_len, uint loop_op);
    ckernel_template(uint outer_loop_len, uint inner_loop_len, uint loop_op0, uint loop_op1);
    void set_end_ops(uint end_op0, uint end_op1);
    void set_end_op(uint end_op0);
    void set_start_op(uint start_op0);
    void set_last_inner_loop_instr(uint op);
    void set_last_outer_loop_instr(uint op);
    void set_outer_loop_len(uint len);
    void set_inner_loop_len(uint len);
    void set_loop_instr(uint loop_op0, uint loop_op1);

    void program(volatile uint *instrn_buffer);                    // just programs the registers
    void program_bank0_sw_cntl(volatile uint *instrn_buffer);      // programs BANK0 in software control mode
    void program_bank1_sw_cntl(volatile uint *instrn_buffer);      // programs BANK1 in software control mode
    static void run(volatile uint *instrn_buffer);                 // runs - assumes that registers were already programmed
    static void run_and_finish(volatile uint *instrn_buffer);      // runs and switches mop_config bank - assumes that registers were already programmed
    static void run_bank0_sw_cntl(volatile uint *instrn_buffer);   // run bank 0 in SW control mode
    static void run_bank1_sw_cntl(volatile uint *instrn_buffer);   // run bank 1 in SW control mode
    void program_and_run(volatile uint *instrn_buffer);            // calls program, then run
    void program_and_run_and_finish(volatile uint *instrn_buffer); // calls program, then runs and switches the mop_config bank
};

#if 0
class ckernel_unpack_template {

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
            uint skipB_instr
     ) : m_unpackB(unpackB), m_unpack_halo(unpackHalo),
         m_A0_instr(A0_instr),
         m_A1_instr(A1_instr),
         m_A2_instr(A2_instr),
         m_A3_instr(A3_instr),
         m_B_instr(B_instr),
         m_skipA_instr(skipA_instr),
         m_skipB_instr(skipB_instr) {};

public:

    // Default ZeroSrcA UNPACR_NOP
    static constexpr uint DEF_ZEROSRCA = TT_OP_UNPACR_NOP(0, 0,0,0,0,0,0,p_unpacr::UNP_CLRSRC_ZERO, p_unpacr::UNP_CLRSRC);
    static constexpr uint DEF_NINFSRCA = TT_OP_UNPACR_NOP(0, 0,0,0,0,0,0,p_unpacr::UNP_CLRSRC_NEGINF, p_unpacr::UNP_CLRSRC);
    static constexpr uint DEF_UNPACR_NOP = TT_OP_UNPACR_NOP(0, 0,0,0,0,0,0,0,p_unpacr::UNP_NOP);


    // Default skip A/B instructions that increment Z counters by 1
    static constexpr uint DEF_SKIP_A = TT_OP_INCADCZW(0b001, 0, 0, 0, 1);
    static constexpr uint DEF_SKIP_B = TT_OP_INCADCZW(0b010, 0, 0, 0, 1);

    // Default non-halo A instruction
    //EMC1 static constexpr uint DEF_A_instr  = TT_OP_UNPACR(0,0b1, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    //EMC1 static constexpr uint DEF_A_cntx_ovrd_instr  = TT_OP_UNPACR(0,0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_A_instr  = TT_OP_UNPACR0(0,0b1, 0, 0, 0, 1, 0, 0, 0, 1);
    //EMC1 static constexpr uint DEF_A_cntx_ovrd_instr  = TT_OP_UNPACR0(0,0b1, 0, 0, 1, 1, 0, 0, 0, 1);
    static constexpr uint DEF_A_cntx_ovrd_instr  = TT_OP_UNPACR0(0,0b1, 0, 0, 0, 0, 0, 0, 0, 1);

    // Default B instruction with rarefy
    //EMC1 static constexpr uint DEF_B_rarefy_instr = TT_OP_UNPACR(1, 0b01, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_ENABLE, 0, 0, 0, 0, 1);
    //EMC1 static constexpr uint DEF_B_rarefy_instr = TT_OP_UNPACR1(1, 0b01, 0, 0, 0, 1, 0, 0, 0, 1);

    // Default B instruction with rarefy and context override, coupled with halo on A
    //EMC1 static constexpr uint DEF_B_rarefy_cntx_ovrd_instr = TT_OP_UNPACR(1, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_ENABLE, 0, 0, 0, 0, 1);
    //EMC1 static constexpr uint DEF_B_rarefy_cntx_ovrd_instr = TT_OP_UNPACR1(1, 0b01, 0, 0, 1, 1, 0, 0, 0, 0, 1);

    // Default B instruction without rarefy, no z increment and with context override, coupled with halo on A for factored conv
    //EMC1 static constexpr uint DEF_B_cntx_ovrd_no_z_inc_instr = TT_OP_UNPACR(1, 0b00, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    // Default B instruction without rarefy and context override
    //static constexpr uint DEF_B_cntx_ovrd_instr = TT_OP_UNPACR(1, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    // Default B instruction without rarefy
    //EMC1 static constexpr uint DEF_B_instr = TT_OP_UNPACR(1, 0b01, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint DEF_B_instr = TT_OP_UNPACR1(0, 0b01, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    // MM Dec 28 2023 stupid compiler workaround until we decide what to do with the TDMA kernels
    static constexpr uint DEF_B_cntx_ovrd_instr = DEF_B_instr;
    static constexpr uint DEF_B_cntx_ovrd_no_z_inc_instr = DEF_B_instr;
    static constexpr uint DEF_B_rarefy_cntx_ovrd_instr = DEF_B_instr;

    // Default halo A instructions
    /*EMC1 static constexpr uint DEF_A0_instr = TT_OP_UNPACR(0,0b00,0,p_unpacr::TILE0_CFG_CONTEXT,p_unpacr::TILE0_ADDRCNT_CONTEXT,1,0,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    static constexpr uint DEF_A1_instr = TT_OP_UNPACR(0,0b00,0,p_unpacr::TILE1_CFG_CONTEXT,p_unpacr::TILE1_ADDRCNT_CONTEXT,1,0,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    static constexpr uint DEF_A2_instr = TT_OP_UNPACR(0,0b00,0,p_unpacr::TILE2_CFG_CONTEXT,p_unpacr::TILE2_ADDRCNT_CONTEXT,1,0,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    static constexpr uint DEF_A3_instr = TT_OP_UNPACR(0,0b01,0,p_unpacr::TILE3_CFG_CONTEXT,p_unpacr::TILE3_ADDRCNT_CONTEXT,1,1,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1); */
    static constexpr uint DEF_A0_instr = TT_OP_UNPACR0(0,0b00,0,0,0,0,0,0,0,1);
    static constexpr uint DEF_A1_instr = TT_OP_UNPACR0(0,0b00,0,0,0,0,0,0,0,1);
    static constexpr uint DEF_A2_instr = TT_OP_UNPACR0(0,0b00,0,0,0,0,0,0,0,1);
    static constexpr uint DEF_A3_instr = TT_OP_UNPACR0(0,0b01,0,0,0,0,1,0,0,1);

    // Special case where all later strips are skipped, so this one has to set DVALID because it is last, and increment Z
    /* EMC1
    static constexpr uint DEF_A0_last_instr = TT_OP_UNPACR(0,0b01,0,p_unpacr::TILE0_CFG_CONTEXT,p_unpacr::TILE0_ADDRCNT_CONTEXT,1,1,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    static constexpr uint DEF_A1_last_instr = TT_OP_UNPACR(0,0b01,0,p_unpacr::TILE1_CFG_CONTEXT,p_unpacr::TILE1_ADDRCNT_CONTEXT,1,1,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    static constexpr uint DEF_A2_last_instr = TT_OP_UNPACR(0,0b01,0,p_unpacr::TILE2_CFG_CONTEXT,p_unpacr::TILE2_ADDRCNT_CONTEXT,1,1,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    */
    static constexpr uint DEF_A0_last_instr = TT_OP_UNPACR0(0,0b01,0,0,0,1,0,0,0,1);
    static constexpr uint DEF_A1_last_instr = TT_OP_UNPACR0(0,0b01,0,0,0,1,0,0,0,1);
    static constexpr uint DEF_A2_last_instr = TT_OP_UNPACR0(0,0b01,0,0,0,1,0,0,0,1);

    // Halo A instructions that skip actual unpacking, but increment context
    /* EMC1
    static constexpr uint SKIP_A0_instr = TT_OP_UNPACR(0,0b00,1,p_unpacr::TILE0_CFG_CONTEXT,p_unpacr::TILE0_ADDRCNT_CONTEXT,1,0,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    static constexpr uint SKIP_A1_instr = TT_OP_UNPACR(0,0b00,1,p_unpacr::TILE1_CFG_CONTEXT,p_unpacr::TILE1_ADDRCNT_CONTEXT,1,0,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    static constexpr uint SKIP_A2_instr = TT_OP_UNPACR(0,0b00,1,p_unpacr::TILE2_CFG_CONTEXT,p_unpacr::TILE2_ADDRCNT_CONTEXT,1,0,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    static constexpr uint SKIP_A3_instr = TT_OP_UNPACR(0,0b00,1,p_unpacr::TILE3_CFG_CONTEXT,p_unpacr::TILE3_ADDRCNT_CONTEXT,1,0,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    */
    static constexpr uint SKIP_A0_instr = TT_OP_UNPACR0(0,0b00,0,0,0,0,0,0,0,1);
    static constexpr uint SKIP_A1_instr = TT_OP_UNPACR0(0,0b00,0,0,0,0,0,0,0,1);
    static constexpr uint SKIP_A2_instr = TT_OP_UNPACR0(0,0b00,0,0,0,0,0,0,0,1);
    static constexpr uint SKIP_A3_instr = TT_OP_UNPACR0(0,0b00,0,0,0,0,0,0,0,1);

    // Factored conv halo A instructions
    /* EMC1
    static constexpr uint DEF_A0_fconv_instr = TT_OP_UNPACR(0,0b00,0,p_unpacr::TILE0_CFG_CONTEXT,p_unpacr::TILE0_ADDRCNT_CONTEXT,1,0,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    static constexpr uint DEF_A1_fconv_instr = TT_OP_UNPACR(0,0b00,0,p_unpacr::TILE1_CFG_CONTEXT,p_unpacr::TILE1_ADDRCNT_CONTEXT,1,0,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    static constexpr uint DEF_A2_fconv_instr = TT_OP_UNPACR(0,0b00,0,p_unpacr::TILE2_CFG_CONTEXT,p_unpacr::TILE2_ADDRCNT_CONTEXT,1,0,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    static constexpr uint DEF_A3_fconv_instr = TT_OP_UNPACR(0,0b00,0,p_unpacr::TILE3_CFG_CONTEXT,p_unpacr::TILE3_ADDRCNT_CONTEXT,1,1,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    */
    static constexpr uint DEF_A0_fconv_instr = TT_OP_UNPACR0(0,0b00,0,0,0,0,0,0,0,1);
    static constexpr uint DEF_A1_fconv_instr = TT_OP_UNPACR0(0,0b00,0,0,0,0,0,0,0,1);
    static constexpr uint DEF_A2_fconv_instr = TT_OP_UNPACR0(0,0b00,0,0,0,0,0,0,0,1);
    static constexpr uint DEF_A3_fconv_instr = TT_OP_UNPACR0(0,0b00,0,1,0,0,0,0,0,1);

    // Special case where all later strips are skipped, so this one has to set DVALID because it is last, and increment Z (factored conv)
    /* EMC1
    static constexpr uint DEF_A0_fconv_last_instr = TT_OP_UNPACR(0,0b00,0,p_unpacr::TILE0_CFG_CONTEXT,p_unpacr::TILE0_ADDRCNT_CONTEXT,1,1,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    static constexpr uint DEF_A1_fconv_last_instr = TT_OP_UNPACR(0,0b00,0,p_unpacr::TILE1_CFG_CONTEXT,p_unpacr::TILE1_ADDRCNT_CONTEXT,1,1,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    static constexpr uint DEF_A2_fconv_last_instr = TT_OP_UNPACR(0,0b00,0,p_unpacr::TILE2_CFG_CONTEXT,p_unpacr::TILE2_ADDRCNT_CONTEXT,1,1,p_unpacr::RAREFYB_DISABLE, 0,p_unpacr::AUTO_INC_CONTEXT,1,0,1);
    */
    static constexpr uint DEF_A0_fconv_last_instr = TT_OP_UNPACR0(0,0b00,0,0,0,1,0,0,0,1);
    static constexpr uint DEF_A1_fconv_last_instr = TT_OP_UNPACR0(0,0b00,0,0,0,1,0,0,0,1);
    static constexpr uint DEF_A2_fconv_last_instr = TT_OP_UNPACR0(0,0b00,0,0,0,1,0,0,0,1);

    // Default non-halo D (unpack-to-dest) instruction with context override
    //EMC1 static constexpr uint DEF_D_instr  = TT_OP_UNPACR(0, 0b10001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // unpack0 to dest, and increment Z for src-tile and dest
    static constexpr uint DEF_D_instr  = TT_OP_UNPACR0(0, 0b10001, 0, 0, 0, 0, 0,  0, 0, 1); // unpack0 to dest, and increment Z for src-tile and dest

    //EMC1 static constexpr uint DEF_D_cntx_ovrd_instr  = TT_OP_UNPACR(0, 0b10000, 0, 1, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // unpack0 to dest using context 1
    static constexpr uint DEF_D_cntx_ovrd_instr  = TT_OP_UNPACR0(0, 0b10000, 0, 0, 0, 0, 0,  0, 0, 1); // unpack0 to dest using context 1

    // Unpack-to-dest instruction with context override and context auto increment
    //EMC1 static constexpr uint DEF_D_auto_inc_cntx_instr0  = TT_OP_UNPACR(0, 0b00000, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 0, 0, 1); // unpack0 to dest, and increment Z for src-tile and dest
    //EMC1 static constexpr uint DEF_D_auto_inc_cntx_instr1  = TT_OP_UNPACR(0, 0b10001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, p_unpacr::AUTO_INC_CONTEXT, 0, 0, 1); // unpack0 to dest, and increment Z for src-tile and dest
    static constexpr uint DEF_D_auto_inc_cntx_instr0  = TT_OP_UNPACR0(0, 0b00000, 0, 0, 0, 0, 0, 0, 0, 1); // unpack0 to dest, and increment Z for src-tile and dest
    static constexpr uint DEF_D_auto_inc_cntx_instr1  = TT_OP_UNPACR0(0, 0b10001, 0, 0, 0, 0, 0, 0, 0, 1); // unpack0 to dest, and increment Z for src-tile and dest


    //
    // Convenience factory methods
    //
    static ckernel_unpack_template lzA(
            bool neginf,
            uint A_instr = DEF_A_cntx_ovrd_instr,
            uint skipA_instr = DEF_SKIP_A);

    static ckernel_unpack_template lA(
            uint A_instr = DEF_A_cntx_ovrd_instr,
            uint skipA_instr = DEF_SKIP_A);

    static ckernel_unpack_template lhA(const uint32_t halo_mask);

    static ckernel_unpack_template flhA(const uint32_t halo_mask);

    static ckernel_unpack_template lBhA(const uint32_t halo_mask, const bool rarefy=true);

    static ckernel_unpack_template flBhA(const uint32_t halo_mask);

    static ckernel_unpack_template lBA(
            uint A_instr = DEF_A_instr,
            uint skipA_instr = DEF_SKIP_A,

            uint B_instr = DEF_B_instr,
            uint skipB_instr = DEF_SKIP_B
    );

    static ckernel_unpack_template lBAD();

    void program(volatile uint *instrn_buffer) const;         // just programs the registers
    static void run(volatile uint *instrn_buffer, const uint8_t count, const uint32_t zmask = 0);      // runs - assumes that registers were already programmed
    void program_and_run(volatile uint *instrn_buffer, const uint8_t count, const uint32_t zmask = 0); // calls program, then run

};
#endif

ckernel_template::ckernel_template(uint outer_loop_len, uint inner_loop_len, uint loop_op) :
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
    // MM Oct 14 2022: Seems like this assertion is unwanted, zero OUTER_LOOP_LEN is supported in HW
    // FWASSERT("MOP OUTER LOOP LENGTH should be non-zero", (m_outer_loop_len > 0));
}

ckernel_template::ckernel_template(uint outer_loop_len, uint inner_loop_len, uint loop_op0, uint loop_op1) :
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
    // MM Oct 14 2022: Seems like this assertion is unwanted, zero OUTER_LOOP_LEN is supported in HW
    // FWASSERT("MOP OUTER LOOP LENGTH should be non-zero", (m_outer_loop_len > 0));
}

void ckernel_template::set_outer_loop_len(uint len)
{
    m_outer_loop_len = len;
}

void ckernel_template::set_inner_loop_len(uint len)
{
    m_inner_loop_len = len;
}

void ckernel_template::set_loop_instr(uint loop_op0, uint loop_op1)
{
    m_loop_op0 = loop_op0;
    m_loop_op1 = loop_op1;
}

void ckernel_template::set_end_ops(uint end_op0, uint end_op1)
{
    m_end_op0 = end_op0;
    m_end_op1 = end_op1;
}

void ckernel_template::set_end_op(uint end_op0)
{
    set_end_ops(end_op0, TT_OP_NOP);
}

void ckernel_template::set_start_op(uint start_op0)
{
    m_start_op0 = start_op0;
}

void ckernel_template::set_last_inner_loop_instr(uint op)
{
    m_loop1_last_instr = op;
}

void ckernel_template::set_last_outer_loop_instr(uint op)
{
    m_loop0_last_instr = op;
}

void ckernel_template::program_and_run(volatile uint *instrn_buffer)
{
    program(instrn_buffer);
    run(instrn_buffer);
}

void ckernel_template::program_and_run_and_finish(volatile uint *instrn_buffer)
{
    program(instrn_buffer);
    run_and_finish(instrn_buffer);
}

void ckernel_template::run([[maybe_unused]] volatile uint *instrn_buffer)
{
    TTI_MOP(1, 0, 0, 0); // run the double-loop template
}

void ckernel_template::run_bank0_sw_cntl([[maybe_unused]] volatile uint *instrn_buffer)
{
    TTI_MOP(1, 0, 0, 0); // run the double-loop template
}

void ckernel_template::run_and_finish([[maybe_unused]] volatile uint *instrn_buffer)
{
    TTI_MOP(1, 1, 0, 0); // run the double-loop template
}

void ckernel_template::run_bank1_sw_cntl([[maybe_unused]] volatile uint *instrn_buffer)
{
    TTI_MOP(1, 1, 0, 0); // run the double-loop template
}

void ckernel_template::program([[maybe_unused]] volatile uint *instrn_buffer)
{
    volatile mop_config_regs_t *mop_cfg = reinterpret_cast<volatile mop_config_regs_t *>(MOP_CFG_BASE);

    //  In hardware control mode, a write to a MOP Config bank that is
    //  in use should block, so no mop_sync() is needed.
    //    mop_sync();

    mop_cfg->BANK0_LOOP0_LEN         = m_outer_loop_len;
    mop_cfg->BANK0_LOOP1_LEN         = m_inner_loop_len;
    mop_cfg->BANK0_LOOP_START_INSTR0 = m_start_op0;
    mop_cfg->BANK0_LOOP_END_INSTR0   = m_end_op0;
    mop_cfg->BANK0_LOOP_END_INSTR1   = m_end_op1;
    mop_cfg->BANK0_LOOP_INSTR0       = m_loop_op0;
    mop_cfg->BANK0_LOOP_INSTR1       = m_loop_op1;
    mop_cfg->BANK0_LOOP0_LAST_INSTR  = m_loop0_last_instr;
    mop_cfg->BANK0_LOOP1_LAST_INSTR  = m_loop1_last_instr;
    mop_cfg->MOP_CONFIG              = 1;
}

void ckernel_template::program_bank0_sw_cntl([[maybe_unused]] volatile uint *instrn_buffer)
{
    volatile mop_config_regs_t *mop_cfg = reinterpret_cast<volatile mop_config_regs_t *>(MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg->MOP_CONFIG              = 2;
    mop_cfg->BANK0_LOOP0_LEN         = m_outer_loop_len;
    mop_cfg->BANK0_LOOP1_LEN         = m_inner_loop_len;
    mop_cfg->BANK0_LOOP_START_INSTR0 = m_start_op0;
    mop_cfg->BANK0_LOOP_END_INSTR0   = m_end_op0;
    mop_cfg->BANK0_LOOP_END_INSTR1   = m_end_op1;
    mop_cfg->BANK0_LOOP_INSTR0       = m_loop_op0;
    mop_cfg->BANK0_LOOP_INSTR1       = m_loop_op1;
    mop_cfg->BANK0_LOOP0_LAST_INSTR  = m_loop0_last_instr;
    mop_cfg->BANK0_LOOP1_LAST_INSTR  = m_loop1_last_instr;
    //    mop_cfg->MOP_CONFIG               = 1;
}

void ckernel_template::program_bank1_sw_cntl([[maybe_unused]] volatile uint *instrn_buffer)
{
    volatile mop_config_regs_t *mop_cfg = reinterpret_cast<volatile mop_config_regs_t *>(MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg->MOP_CONFIG              = 2;
    mop_cfg->BANK1_LOOP0_LEN         = m_outer_loop_len;
    mop_cfg->BANK1_LOOP1_LEN         = m_inner_loop_len;
    mop_cfg->BANK1_LOOP_START_INSTR0 = m_start_op0;
    mop_cfg->BANK1_LOOP_END_INSTR0   = m_end_op0;
    mop_cfg->BANK1_LOOP_END_INSTR1   = m_end_op1;
    mop_cfg->BANK1_LOOP_INSTR0       = m_loop_op0;
    mop_cfg->BANK1_LOOP_INSTR1       = m_loop_op1;
    mop_cfg->BANK1_LOOP0_LAST_INSTR  = m_loop0_last_instr;
    mop_cfg->BANK1_LOOP1_LAST_INSTR  = m_loop1_last_instr;
    //    mop_cfg->MOP_CONFIG               = 1;
}

#if 0
void ckernel_unpack_template::program_and_run(volatile uint *instrn_buffer, const uint8_t count, const uint32_t zmask)
{
  program(instrn_buffer);
  run(instrn_buffer, count, zmask);
}

void ckernel_unpack_template::run(volatile uint *instrn_buffer, const uint8_t count, const uint32_t zmask)
{
  // FWASSERT("Unpack template only supports loops up to 128", count <= 128);
  TT_MOP_CFG(zmask >> 16);                // Set the top 16 bits of zmask - we could skip this for count <= 16
  TT_MOP(0, count-1, zmask & 0xFFFF);     // Run the template
}

void ckernel_unpack_template::program(volatile uint *instrn_buffer) const
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

ckernel_unpack_template ckernel_unpack_template::lA(
        uint A_instr,
        uint skipA_instr)
{
  return ckernel_unpack_template(
      false, // src B
      false, // halo
      A_instr,
      0, 0, 0,
      skipA_instr,
      0, 0);
}

ckernel_unpack_template ckernel_unpack_template::lzA(
        bool neginf,
        uint A_instr,
        uint skipA_instr)
{
  return ckernel_unpack_template(
      false, // src B
      true, // halo
      neginf? DEF_NINFSRCA : DEF_ZEROSRCA,
      A_instr, DEF_UNPACR_NOP, DEF_UNPACR_NOP,
      skipA_instr,
      0, 0);
}

ckernel_unpack_template ckernel_unpack_template::lhA(const uint32_t halo_mask)
{
  // Figure out which unpack is last
  const uint last_mask = (halo_mask == 0x1) ? 0x1 :
                         (halo_mask <= 0x3) ? 0x2 :
                         (halo_mask <= 0x7) ? 0x4 : 0;

  return ckernel_unpack_template(
          false, // src B
          true,  // halo
          ((halo_mask >> 0) & 0x1) ? ((last_mask >> 0) & 0x1) ? DEF_A0_last_instr : DEF_A0_instr : SKIP_A0_instr,
          ((halo_mask >> 1) & 0x1) ? ((last_mask >> 1) & 0x1) ? DEF_A1_last_instr : DEF_A1_instr : SKIP_A1_instr,
          ((halo_mask >> 2) & 0x1) ? ((last_mask >> 2) & 0x1) ? DEF_A2_last_instr : DEF_A2_instr : SKIP_A2_instr,
          ((halo_mask >> 3) & 0x1) ? DEF_A3_instr : SKIP_A3_instr,
          DEF_SKIP_A,
          0, 0);
}

ckernel_unpack_template ckernel_unpack_template::flhA(const uint32_t halo_mask)
{
  // Figure out which unpack is last
  const uint last_mask = (halo_mask == 0x1) ? 0x1 :
                         (halo_mask <= 0x3) ? 0x2 :
                         (halo_mask <= 0x7) ? 0x4 : 0;

  return ckernel_unpack_template(
          false, // src B
          true,  // halo
          ((halo_mask >> 0) & 0x1) ? ((last_mask >> 0) & 0x1) ? DEF_A0_fconv_last_instr : DEF_A0_fconv_instr : SKIP_A0_instr,
          ((halo_mask >> 1) & 0x1) ? ((last_mask >> 1) & 0x1) ? DEF_A1_fconv_last_instr : DEF_A1_fconv_instr : SKIP_A1_instr,
          ((halo_mask >> 2) & 0x1) ? ((last_mask >> 2) & 0x1) ? DEF_A2_fconv_last_instr : DEF_A2_fconv_instr : SKIP_A2_instr,
          ((halo_mask >> 3) & 0x1) ? DEF_A3_fconv_instr : SKIP_A3_instr,
          TT_OP_NOP,
          0, 0);
}

ckernel_unpack_template ckernel_unpack_template::lBhA(const uint32_t halo_mask, const bool rarefy)
{
  // Figure out which unpack is last
  const uint last_mask = (halo_mask == 0x1) ? 0x1 :
                         (halo_mask <= 0x3) ? 0x2 :
                         (halo_mask <= 0x7) ? 0x4 : 0;

  return ckernel_unpack_template(
          true, // src B
          true,  // halo
          ((halo_mask >> 0) & 0x1) ? ((last_mask >> 0) & 0x1) ? DEF_A0_last_instr : DEF_A0_instr : SKIP_A0_instr,
          ((halo_mask >> 1) & 0x1) ? ((last_mask >> 1) & 0x1) ? DEF_A1_last_instr : DEF_A1_instr : SKIP_A1_instr,
          ((halo_mask >> 2) & 0x1) ? ((last_mask >> 2) & 0x1) ? DEF_A2_last_instr : DEF_A2_instr : SKIP_A2_instr,
          ((halo_mask >> 3) & 0x1) ? DEF_A3_instr : SKIP_A3_instr,
          DEF_SKIP_A,
          rarefy ? DEF_B_rarefy_cntx_ovrd_instr : DEF_B_cntx_ovrd_instr,
          DEF_SKIP_B);
}

ckernel_unpack_template ckernel_unpack_template::flBhA(const uint32_t halo_mask)
{
  // Figure out which unpack is last
  const uint last_mask = (halo_mask == 0x1) ? 0x1 :
                         (halo_mask <= 0x3) ? 0x2 :
                         (halo_mask <= 0x7) ? 0x4 : 0;

  return ckernel_unpack_template(
          true, // src B
          true,  // halo
          ((halo_mask >> 0) & 0x1) ? ((last_mask >> 0) & 0x1) ? DEF_A0_fconv_last_instr : DEF_A0_fconv_instr : SKIP_A0_instr,
          ((halo_mask >> 1) & 0x1) ? ((last_mask >> 1) & 0x1) ? DEF_A1_fconv_last_instr : DEF_A1_fconv_instr : SKIP_A1_instr,
          ((halo_mask >> 2) & 0x1) ? ((last_mask >> 2) & 0x1) ? DEF_A2_fconv_last_instr : DEF_A2_fconv_instr : SKIP_A2_instr,
          ((halo_mask >> 3) & 0x1) ? DEF_A3_fconv_instr : SKIP_A3_instr,
          TT_OP_NOP,
          DEF_B_cntx_ovrd_no_z_inc_instr,
          DEF_SKIP_B);
}

ckernel_unpack_template ckernel_unpack_template::lBA(
        uint A_instr,
        uint skipA_instr,

        uint B_instr,
        uint skipB_instr)
{
  return ckernel_unpack_template(
          true, // src B
          false,  // halo
          A_instr,
          0, 0, 0,
          skipA_instr,
          B_instr,
          skipB_instr);
}

ckernel_unpack_template ckernel_unpack_template::lBAD()
{
  return ckernel_unpack_template(
          true, // src B
          true, // halo
          DEF_D_cntx_ovrd_instr,
          DEF_A_cntx_ovrd_instr,
          SKIP_A2_instr,
          SKIP_A3_instr,
          DEF_SKIP_A,
          DEF_B_cntx_ovrd_instr,
          DEF_SKIP_B);
}
#endif

} // namespace ckernel
