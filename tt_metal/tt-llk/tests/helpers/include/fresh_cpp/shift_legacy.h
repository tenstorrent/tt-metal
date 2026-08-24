// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// shift — LEGACY semantic bodies, preserved verbatim (symbols renamed
// *_legacy) when lane GH rewrote fresh_cpp/shift.h (2026-08-24): the
// logical-shift + manual sign-fill + compound range-guard composition below
// compiles to a 26-slot replay row and measured +43.01% vs hand
// (weekly-e2e2-weekly-20260821); the live body uses BH SFPSHFT's native
// ARITHMETIC mode (SFPSHFT_MOD1_ARITHMETIC — the manual fill dance is a
// Wormhole-era assumption; BlackholeA0 SFPSHFT.md documents the mode) and a
// one-word unsigned out-of-range test.  Kept for A/B archaeology; not wired
// to any test node.  The live bodies are fresh_cpp/shift.h.
//
// Original header:
// Canonical semantic bodies for the shift op file (storm contract,
// fresh_cpp/README.md).  Production: metal ckernel_sfpu_shift.h — all three
// binary Int32 shifts are entirely raw TT_SFPLOAD/TTI_SFP* streams over fixed
// LREG0..4 with magic immediates 0xFE0 (-32) and 0x020 (32); SFPSHFT is
// logical-only, so the arithmetic right shift synthesizes sign extension by
// ORing ~0 << (32 - amount) into lanes whose value is negative.  The left
// shift's semantic body predates the storm (calculate_left_shift_fresh_cpp,
// leftshift-fresh row); this header states the two RIGHT shifts: shift right
// by the per-lane amount, zero where the amount is outside [0, 32); LOGICAL
// fills the vacated bits with zero, arithmetic (LOGICAL = false) with the
// sign bit — the golden's torch.bitwise_right_shift contract with the same
// out-of-range guard.  Same INT32_2S_COMP load/store contract as the
// production dispatch (typed DataLayout::SM32 — the fresh add/sub/mul/
// left-shift precedent).
#include <cstdint>

namespace ckernel::sfpu
{

template <bool LOGICAL, int ITERATIONS>
__attribute__((noinline)) void calculate_right_shift_fresh_cpp_legacy()
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
            // Logical right shift (SFPSHFT shifts right for negative counts).
            sfpi::vInt result = sfpi::as<sfpi::vInt>(sfpi::shft(sfpi::as<sfpi::vUInt>(value), -amount));
            if constexpr (!LOGICAL)
            {
                // Arithmetic contract: fill the vacated high bits with the
                // sign bit (all-ones mask shifted left by 32 - amount; the
                // amount > 0 guard keeps the in-range amount == 0 identity).
                v_if (value < 0 && amount > 0)
                {
                    const sfpi::vInt fill = sfpi::as<sfpi::vInt>(sfpi::shft(sfpi::vUInt(0xFFFFFFFFu), sfpi::vInt(32) - amount));
                    result                = result | fill;
                }
                v_endif;
            }
            v_if (amount < 0 || amount >= 32)
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
inline void call_right_shift_fresh_cpp_legacy(
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
    calculate_right_shift_fresh_cpp_legacy<LOGICAL, ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu
