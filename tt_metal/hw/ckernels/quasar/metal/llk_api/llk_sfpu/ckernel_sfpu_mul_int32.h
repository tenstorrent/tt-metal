// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_instr_params.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel {
namespace sfpu {

// SFPMUL24 multiplies the low 23 bits of each operand; mod1 bit 0 selects which half of the
// 46-bit product reaches the destination.
constexpr std::uint32_t MUL_INT32_MUL24_LOW23 = 0x0;   // product bits [22:0]
constexpr std::uint32_t MUL_INT32_MUL24_HIGH23 = 0x1;  // product bits [45:23]
// SFPSHFT: logical shift, data from lreg_c, amount from imm12 (negative imm shifts right).
constexpr std::uint32_t MUL_INT32_SHFT_IMM_FROM_C = 0x5;
constexpr std::uint32_t MUL_INT32_SHFT_RIGHT_23 = (-23) & 0xFFF;
// SFPSETSGN: mod 1 takes the new sign from imm12_math[0]; mod 0 takes it from src_b, which the
// encoding aliases to lreg_dest, so the destination keeps its own sign bit.
constexpr std::uint32_t MUL_INT32_SETSGN_FROM_IMM = 0x1;
constexpr std::uint32_t MUL_INT32_SETSGN_FROM_DST = 0x0;
constexpr std::uint32_t MUL_INT32_SIGN_POSITIVE = 0x0;

#ifndef DISABLE_SFPLOADMACRO
// SFPLOADMACRO fast path for the fully-constrained shape of _mul_int32_sm_shift23_ (see that
// kernel's comment for what the ensure_* flags promise). With both operands non-negative and both
// below 2^23 the whole row is one multiply, which folds into the MAD unit of the load that fetches
// operand B, taking the row from four issue slots to three.
//
// Three Quasar properties shape this and are worth knowing before changing it:
//
//   1. A scheduled slot may only write the LREG its own load targeted, or the staging register.
//      It cannot write a third register, so each macro spends its load target as a result register.
//   2. Template operands are fixed when the template is captured, and only instruction registers
//      4-7 are programmable. Anything that has to alternate between rows therefore has to be the
//      *loaded* register, which the sequencer rewrites per call, not a template operand.
//   3. SFPMUL24 has two cycles of latency, which is longer than a three-slot row. The store of a
//      row is issued during the next row instead, and the two operand-B registers alternate so the
//      next load cannot land on a product the store has not read yet.
//
// Those same three properties are why the ENSURE_A_23BIT == false shape is left on the generic
// path: it needs a second multiply whose multiplicand is operand B, but B's own macro has already
// spent B's register on the first product, so B would need a second inline load. That lands at six
// issue slots against the generic path's seven, which does not pay for the scheduling risk.
constexpr std::uint32_t MUL_INT32_MACRO_SEQ_ID = 0;                         // SFPLOADMACRO seq_id field
constexpr std::uint32_t MUL_INT32_MACRO_SEQ_CFG = p_sfpconfig::MACRO_SEQ0;  // matching SFPCONFIG dest
constexpr std::uint32_t MUL_INT32_MACRO_IR_MUL24_HIGH = 4;                  // instruction register holding the multiply
// Operand A is a template operand, so it is fixed for every row; the two alternating operand-B
// registers are named by the SFPLOADMACRO itself.
constexpr std::uint32_t MUL_INT32_MACRO_A = p_sfpu::LREG0;
constexpr std::uint32_t MUL_INT32_MACRO_B_EVEN = p_sfpu::LREG1;
constexpr std::uint32_t MUL_INT32_MACRO_B_ODD = p_sfpu::LREG2;

/// One Load Macro slot byte. Bit 7 substitutes the loaded LREG as source B rather than source C,
/// bit 6 routes the result through the staging register, bits 5:3 are the delay in SFPU issue
/// cycles after the SFPLOADMACRO (0 meaning the cycle immediately after it), and bits 2:0 select
/// the instruction register. Register 0 disables the slot, so a zero byte is an unused unit.
constexpr std::uint32_t mul_int32_macro_slot(
    const std::uint32_t instr_reg, const std::uint32_t delay, const bool srcb_override) {
    return (srcb_override ? 0x80u : 0u) | ((delay & 0x7u) << 3) | (instr_reg & 0x7u);
}
constexpr std::uint32_t MUL_INT32_MACRO_SLOT_UNUSED = 0;
#endif

// Int32 multiply ported from the BH DISABLE_SFPLOADMACRO path.
// Uses SFPMUL24 (24-bit partial products) + shifts to produce a full 32-bit
// result.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void _mul_int32_(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    constexpr std::uint32_t tile_stride = 1U << trisc::get_dest_tile_size_log2(TILE_SHAPE);
    const std::uint32_t in0_offset = dst_index_in0 * tile_stride;
    const std::uint32_t in1_offset = dst_index_in1 * tile_stride;
    const std::uint32_t out_offset = dst_index_out * tile_stride;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, in0_offset + (d << 1));
        TT_SFPLOAD(p_sfpu::LREG2, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, in1_offset + (d << 1));

        // Dest layout depends on how operands reached dest:
        //   UNP_DEST / Int32 L1 with 2's-comp tiles → 2's-comp Int32
        //   copy_tile Int8 + fp32_dest_acc FPU → sign-mag Int32 (SIGN_MAGNITUDE_FORMAT=true)
        if constexpr (SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
            TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
        }

        TTI_SFPSHFT((-23) & 0xFFF, p_sfpu::LREG0, p_sfpu::LREG1, 5);  // lreg1 = lreg0 >> 23
        TTI_SFPSHFT((-23) & 0xFFF, p_sfpu::LREG2, p_sfpu::LREG3, 5);  // lreg3 = lreg2 >> 23

        TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG4, 0);  // low(a_lo * b_lo)
        TTI_SFPMUL24(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG5, 1);  // high(a_lo * b_lo)
        TTI_SFPMUL24(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG6, 0);  // low(a_hi * b_lo)
        TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG7, 0);  // low(a_lo * b_hi)

        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);  // lreg5 += lreg6
        TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);  // lreg5 += lreg7
        TTI_SFPSHFT(23, p_sfpu::LREG5, p_sfpu::LREG5, 5);                                          // lreg5 <<= 23
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);  // lreg4 += lreg5

        if constexpr (SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);
        }

        TT_SFPSTORE(p_sfpu::LREG4, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, out_offset + (d << 1));
    }
}

// Programs the SFPLOADMACRO sequence used by _mul_int32_sm_shift23_. Must be called with the same
// template arguments as the kernel, and after _llk_math_eltwise_sfpu_init_. Programming is skipped
// for shapes that run the generic path, so this is safe to call unconditionally.
template <
    bool SIGN_MAGNITUDE_FORMAT = true,
    bool ENSURE_POS_A = false,
    bool ENSURE_POS_B = false,
    bool ENSURE_A_23BIT = false,
    bool ENSURE_B_23BIT = false>
inline void _mul_int32_sm_shift23_init_() {
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (SIGN_MAGNITUDE_FORMAT && ENSURE_POS_A && ENSURE_POS_B && ENSURE_A_23BIT && ENSURE_B_23BIT) {
        // Instruction register 4: VD = high23(MUL24(A, VB)). Capturing through a VD of
        // MACRO_CAPTURE_INSTR4 records the instruction instead of executing it. The sequencer
        // rewrites VD and one source with the LREG the load landed in; bit 7 of the slot sends that
        // substitution to source B, which leaves operand A in place. SFPMUL24 reads VA and VB only,
        // so routing the substitution to source C instead would drop the loaded operand entirely.
        TTI_SFPMUL24(MUL_INT32_MACRO_A, 0, p_sfpu::MACRO_CAPTURE_INSTR4, MUL_INT32_MUL24_HIGH23);

        // Only the MAD unit is scheduled. SFPLOADI stages the 32-bit sequence word in LREG0 for
        // SFPCONFIG, which is why this has to run before the loop claims that register.
        constexpr std::uint32_t mad_slot =
            mul_int32_macro_slot(MUL_INT32_MACRO_IR_MUL24_HIGH, 0 /*delay*/, true /*srcb_override*/);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, (mad_slot << 8) | MUL_INT32_MACRO_SLOT_UNUSED);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0);
        TTI_SFPCONFIG(0, MUL_INT32_MACRO_SEQ_CFG, 0);
        TTI_SFPNOP(0, 0, 0);  // SFPCONFIG hazard: nothing may issue the cycle after SFPCONFIG

        // MACRO_CTRL[11:8] marks a unit dependent, one bit per unit in Simple, MAD, Round, Store
        // order. This sequence is not self-contained -- the multiply reads a register an inline
        // SFPLOAD wrote -- so the MAD unit is marked, which freezes the sequencer on an issue bubble
        // rather than letting the multiply run ahead of its operand. The store fields stay clear
        // because no store is scheduled here. Mod 1 assigns the whole register from the immediate.
        constexpr std::uint32_t macro_ctrl = 1u << 9;
        TTI_SFPCONFIG(macro_ctrl, p_sfpconfig::MACRO_CTRL, 1);
        TTI_SFPNOP(0, 0, 0);  // SFPCONFIG hazard: nothing may issue the cycle after SFPCONFIG
    }
#endif
}

// Sign-magnitude fixed-point multiply: out = sign * ((|in0| * |in1|) >> 23).
//
// in1 is a Q0.23 fixed-point scale, so this rescales in0 by in1 / 2^23 - the quantization
// requant primitive. It shares the 23-bit split with _mul_int32_ above but keeps the *middle*
// slice of the 46-bit product instead of reassembling the low 32 bits, so the two cannot be
// composed: a full multiply discards exactly the bits this one keeps.
//
// Operands are sign-magnitude (Quasar's native Int32 dest layout, SMAG32), so a magnitude is
// the value with bit 31 cleared and the result sign is a bit-31 XOR. A two's-complement dest
// is converted in and back out; INT32_MIN has no sign-magnitude encoding and is not supported.
//
// Exact when each retained cross product fits in 23 bits:
//   (|a| >> 23) * (|b| & 0x7FFFFF) < 2^23  and  (|a| & 0x7FFFFF) * (|b| >> 23) < 2^23,
// and when at most one operand reaches 2^23, because the a_hi * b_hi term is never formed.
// For requant that holds whenever the result itself fits in 23 bits. SFPMUL24 masks each
// operand to 23 bits and SFPIADD wraps at 32 bits; neither saturates.
//
// ENSURE_POS_A / ENSURE_POS_B drop the sign handling for operands the caller guarantees are
// non-negative. ENSURE_A_23BIT / ENSURE_B_23BIT drop the matching cross product for operands
// the caller guarantees are below 2^23.
// TILE_SHAPE stays ahead of the ensure_* flags because SFPU_BINARY_CALL_TINY_TILE appends it as
// the last template argument after whatever the call site supplies.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = true,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32,
    bool ENSURE_POS_A = false,
    bool ENSURE_POS_B = false,
    bool ENSURE_A_23BIT = false,
    bool ENSURE_B_23BIT = false>
inline void _mul_int32_sm_shift23_(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    constexpr std::uint32_t tile_stride = 1U << trisc::get_dest_tile_size_log2(TILE_SHAPE);
    const std::uint32_t in0_offset = dst_index_in0 * tile_stride;
    const std::uint32_t in1_offset = dst_index_in1 * tile_stride;
    const std::uint32_t out_offset = dst_index_out * tile_stride;

#ifndef DISABLE_SFPLOADMACRO
    if constexpr (SIGN_MAGNITUDE_FORMAT && ENSURE_POS_A && ENSURE_POS_B && ENSURE_A_23BIT && ENSURE_B_23BIT) {
        // Three issue slots per row: load A, load B as a macro whose MAD unit multiplies, and store
        // the *previous* row's product. Deferring the store by one row is what gives SFPMUL24 its two
        // cycles, and is why the two operand-B registers alternate. Requires
        // _mul_int32_sm_shift23_init_ with these same template arguments.
        //
        // Addresses: SFPLOADMACRO carries the Dest address in bits [10:1], so it takes addr >> 1,
        // unlike SFPLOAD and SFPSTORE. The destination LREG index is split across lreg_ind_lo (its
        // low two bits) and lreg_ind_hi (bit 2).
        TT_SFPLOAD(MUL_INT32_MACRO_A, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, in0_offset);
        TT_SFPLOADMACRO(
            MUL_INT32_MACRO_SEQ_ID,
            MUL_INT32_MACRO_B_EVEN & 0x3,
            p_sfpu::sfpmem::INT32,
            ADDR_MOD_7,
            0,
            in1_offset >> 1,
            MUL_INT32_MACRO_B_EVEN >> 2);

#pragma GCC unroll 8
        for (int d = 1; d < ITERATIONS; d++) {
            const std::uint32_t b_cur = (d & 1) ? MUL_INT32_MACRO_B_ODD : MUL_INT32_MACRO_B_EVEN;
            const std::uint32_t b_prev = (d & 1) ? MUL_INT32_MACRO_B_EVEN : MUL_INT32_MACRO_B_ODD;
            TT_SFPLOAD(MUL_INT32_MACRO_A, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, in0_offset + (d << 1));
            TT_SFPLOADMACRO(
                MUL_INT32_MACRO_SEQ_ID,
                b_cur & 0x3,
                p_sfpu::sfpmem::INT32,
                ADDR_MOD_7,
                0,
                (in1_offset + (d << 1)) >> 1,
                b_cur >> 2);
            TT_SFPSTORE(b_prev, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, out_offset + ((d - 1) << 1));
        }

        // Drain the last row's scheduled multiply before reading its result. The MAD slot runs one
        // cycle after its SFPLOADMACRO and takes two more to land, and leaving it in flight would let
        // it write into whatever kernel runs next.
        TTI_SFPNOP(0, 0, 0);
        TTI_SFPNOP(0, 0, 0);
        constexpr std::uint32_t b_last = ((ITERATIONS - 1) & 1) ? MUL_INT32_MACRO_B_ODD : MUL_INT32_MACRO_B_EVEN;
        TT_SFPSTORE(b_last, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, out_offset + ((ITERATIONS - 1) << 1));
        return;
    }
#endif

    constexpr std::uint32_t A_RAW = p_sfpu::LREG0;
    constexpr std::uint32_t B_RAW = p_sfpu::LREG2;
    // A magnitude never shares a register with the value carrying the result sign: clearing a
    // sign bit in place would destroy the magnitude the multiply still needs.
    constexpr std::uint32_t A_MAG = ENSURE_POS_A ? A_RAW : p_sfpu::LREG1;
    constexpr std::uint32_t B_MAG = ENSURE_POS_B ? B_RAW : p_sfpu::LREG3;
    // sign(A), or sign(A) ^ sign(B) accumulated into A_RAW. With A known non-negative the
    // result sign is sign(B) alone, so B_RAW carries it and no XOR is needed.
    constexpr std::uint32_t SIGN = ENSURE_POS_A ? B_RAW : A_RAW;
    constexpr std::uint32_t ACC = p_sfpu::LREG4;
    constexpr std::uint32_t TMP = p_sfpu::LREG5;
    // The two high halves are consumed by their own cross product before the next is formed,
    // so they share one register and the live set stays at seven of the eight LRegs.
    constexpr std::uint32_t HI = p_sfpu::LREG6;

    constexpr bool RESULT_IS_POSITIVE = ENSURE_POS_A && ENSURE_POS_B;
    constexpr std::uint32_t RESULT = RESULT_IS_POSITIVE ? ACC : SIGN;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(A_RAW, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, in0_offset + (d << 1));
        TT_SFPLOAD(B_RAW, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, in1_offset + (d << 1));

        if constexpr (!SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPCAST(A_RAW, A_RAW, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);
            TTI_SFPCAST(B_RAW, B_RAW, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);
        }

        if constexpr (!ENSURE_POS_A) {
            TTI_SFPSETSGN(MUL_INT32_SIGN_POSITIVE, A_RAW, A_MAG, MUL_INT32_SETSGN_FROM_IMM);
        }
        if constexpr (!ENSURE_POS_B) {
            TTI_SFPSETSGN(MUL_INT32_SIGN_POSITIVE, B_RAW, B_MAG, MUL_INT32_SETSGN_FROM_IMM);
            if constexpr (!ENSURE_POS_A) {
                // Only bit 31 of SIGN is ever read; the low bits it picks up here are dead.
                TTI_SFPXOR(B_RAW, A_RAW);
            }
        }

        // The low-half product is the one term both operands always contribute to.
        TTI_SFPMUL24(B_MAG, A_MAG, ACC, MUL_INT32_MUL24_HIGH23);

        if constexpr (!ENSURE_A_23BIT) {
            TTI_SFPSHFT(MUL_INT32_SHFT_RIGHT_23, A_MAG, HI, MUL_INT32_SHFT_IMM_FROM_C);
            TTI_SFPMUL24(HI, B_MAG, TMP, MUL_INT32_MUL24_LOW23);
            TTI_SFPIADD(0, TMP, ACC, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
        }
        if constexpr (!ENSURE_B_23BIT) {
            TTI_SFPSHFT(MUL_INT32_SHFT_RIGHT_23, B_MAG, HI, MUL_INT32_SHFT_IMM_FROM_C);
            TTI_SFPMUL24(A_MAG, HI, TMP, MUL_INT32_MUL24_LOW23);
            TTI_SFPIADD(0, TMP, ACC, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
        }

        if constexpr (!RESULT_IS_POSITIVE) {
            TTI_SFPSETSGN(0, ACC, SIGN, MUL_INT32_SETSGN_FROM_DST);
        }
        if constexpr (!SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPCAST(RESULT, RESULT, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
        }
        TT_SFPSTORE(RESULT, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, out_offset + (d << 1));
    }
}

}  // namespace sfpu
}  // namespace ckernel
