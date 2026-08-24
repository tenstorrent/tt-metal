// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic bodies for the shift op file (storm contract,
// fresh_cpp/README.md).  Production: metal ckernel_sfpu_shift.h — all three
// binary Int32 shifts are entirely raw TT_SFPLOAD/TTI_SFP* streams over fixed
// LREG0..4 with magic immediates 0xFE0 (-32) and 0x020 (32); the arithmetic
// right shift synthesizes sign extension by ORing ~0 << (32 - amount) into
// lanes whose value is negative.  The left shift's semantic body predates the
// storm (calculate_left_shift_fresh_cpp, leftshift-fresh row); this header
// states the two RIGHT shifts: shift right by the per-lane amount, zero where
// the amount is outside [0, 32); LOGICAL fills the vacated bits with zero,
// arithmetic (LOGICAL = false) with the sign bit — the golden's
// torch.bitwise_right_shift contract with the same out-of-range guard.  Same
// INT32_2S_COMP load/store contract as the production dispatch (typed
// DataLayout::SM32 — the fresh add/sub/mul/left-shift precedent).
//
// Lane GH 2026-08-24 rewrite (previous bodies preserved in
// fresh_cpp/shift_legacy.h, unwired).  Two facts remove most of the legacy
// 26-slot row:
//  1. Blackhole SFPSHFT has a NATIVE arithmetic mode
//     (SFPSHFT_MOD1_ARITHMETIC; BlackholeA0 SFPSHFT.md functional model:
//     negative shift amount + arithmetic mod = (int32_t)x >> ((-amt) & 31);
//     the pinned craq sim implements the identical branch) — the production
//     kernel's manual sign-fill choreography (save/setcc/setcc/32-amt/NOT/
//     SHFT/OR, and the legacy fresh transcription of it) is a Wormhole-era
//     assumption ("SFPSHFT is logical-only" is true only of WH; sfpi's own
//     ShiftMode::Arithmetic is compiled out under __riscv_xtttensixwh).
//  2. The compound range guard (amount < 0 || amount >= 32) is the single
//     unsigned test (uint32)amount >= 32, decided by one logical shift:
//     shft(amount, -5) != 0 — one SFPSETCC instead of the legacy
//     setcc/iadd-setcc/compc chain.  Exhaustive over all 2^32 amounts:
//     laneGH-evidence-20260824/shift_equiv_cert.c (which also proves the
//     end-to-end old-vs-new row equivalence per lane).
// Both SFPSHFT paths wrap the effective count mod 32 (the & 31 in the
// functional model); every wrapped lane is out-of-range and overwritten by
// the guard, so the wrap is never observable.
#include <cstdint>

namespace ckernel::sfpu
{

template <bool LOGICAL, int ITERATIONS>
__attribute__((noinline)) void calculate_right_shift_fresh_cpp()
{
    constexpr std::uint32_t tile_rows = 32;

#pragma GCC unroll 4
    for (int face = 0; face < 4; ++face)
    {
#pragma GCC unroll 8
        for (int row = 0; row < ITERATIONS; ++row)
        {
            const sfpi::vInt value  = sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>();
            const sfpi::vInt amount = sfpi::dst_reg[tile_rows].mode<sfpi::DataLayout::SM32>();
            // Out-of-range test, one word: any bit above bit 4 set (sign bit
            // included) <=> amount < 0 || amount >= 32.
            const sfpi::vUInt oob = sfpi::shft(sfpi::as<sfpi::vUInt>(amount), -5);
            // SFPSHFT shifts right for negative counts; mode picks the fill.
            sfpi::vInt result;
            if constexpr (LOGICAL)
            {
                result = sfpi::as<sfpi::vInt>(sfpi::shft(sfpi::as<sfpi::vUInt>(value), -amount));
            }
            else
            {
                result = sfpi::shft(value, -amount, sfpi::ShiftMode::Arithmetic);
            }
            v_if (oob != sfpi::vUInt(0u))
            {
                result = 0;
            }
            v_endif;
            sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>() = result;
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, bool LOGICAL, int ITERATIONS>
inline void call_right_shift_fresh_cpp(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh right shift expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh right shift expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh right shift expects full-tile vector mode");

    // Anchor the dynamic tile once in the wrapper so the isolated semantic body
    // contains only constant relative Dst addresses (the fresh max/min, add/sub,
    // mul, and left-shift precedent).
    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_right_shift_fresh_cpp<LOGICAL, ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu
