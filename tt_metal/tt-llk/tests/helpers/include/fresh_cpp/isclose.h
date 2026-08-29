// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the isclose corpus row (storm contract — see
// README.md in this directory).  Include from an LLK_TRISC_MATH kernel after
// sfpu_operations.h; plain typed C++ only.

#include <cstdint>

namespace ckernel::sfpu
{

// Fresh typed-C++ isclose stating the torch.isclose golden contract
// independently, one datum per row over the binary harness's adjacent-tile
// Dst layout:
//
//   isclose(a, b) = |a - b| <= atol + rtol*|b|          (finite lanes)
//   any inf/NaN operand is decided by IEEE-754 class, not by the tolerance:
//   equal infinities match; NaN matches nothing (both-NaN matches only when
//   EQUAL_NAN).
//
// The production kernel (metal ckernel_sfpu_isclose.h) states the same math
// but parks the sign-clear mask in vConstIntPrgm0 at init and folds its
// special-case predicates for predicate-stack economy; here every constant is
// a plain local and each rule is its own plain predicate region.  Class tests
// use the encoding order directly: the magnitude encoding is |x|'s bit
// pattern (float abs IS the sign-bit clear on the sign-magnitude register
// file, NaN payload preserved), which orders every lane by magnitude class:
// |bits| == inf-bits <=> +-inf and |bits| > inf-bits <=> NaN.
template <bool EQUAL_NAN, int ITERATIONS>
__attribute__((noinline)) void calculate_isclose_fresh_cpp(const std::uint32_t rtol_bits, const std::uint32_t atol_bits)
{
    constexpr std::uint32_t tile_rows = 32;
    constexpr int INF_BITS            = 0x7F800000;

    const sfpi::vFloat rtol = Converter::as_float(rtol_bits);
    const sfpi::vFloat atol = Converter::as_float(atol_bits);

    for (int face = 0; face < 4; ++face)
    {
        for (int row = 0; row < ITERATIONS; ++row)
        {
            const sfpi::vFloat a = sfpi::dst_reg[0];
            const sfpi::vFloat b = sfpi::dst_reg[tile_rows];

            // Finite lanes: the tolerance inequality itself.  NaN operands
            // compare false here, so they already sit at 0 before the class
            // fix-up below.
            const sfpi::vFloat abs_a = sfpi::abs(a);
            const sfpi::vFloat abs_b = sfpi::abs(b);
            sfpi::vFloat result      = 0.0f;
            v_if (sfpi::abs(a - b) <= atol + rtol * abs_b)
            {
                result = 1.0f;
            }
            v_endif;

            // Special lanes (any inf/NaN operand): rebuild the answer from
            // class rules.  The tolerance formula is wrong there (inf <= inf
            // holds even for mismatched signs).
            const sfpi::vInt a_bits = sfpi::as<sfpi::vInt>(a);
            const sfpi::vInt b_bits = sfpi::as<sfpi::vInt>(b);
            const sfpi::vInt a_mag  = sfpi::as<sfpi::vInt>(abs_a);
            const sfpi::vInt b_mag  = sfpi::as<sfpi::vInt>(abs_b);
            v_if (a_mag >= INF_BITS || b_mag >= INF_BITS)
            {
                result = 0.0f;
                v_if (a_mag == INF_BITS && a_bits == b_bits)
                {
                    // Same-signed infinities.
                    result = 1.0f;
                }
                v_endif;
                if constexpr (EQUAL_NAN)
                {
                    v_if (a_mag > INF_BITS && b_mag > INF_BITS)
                    {
                        result = 1.0f;
                    }
                    v_endif;
                }
            }
            v_endif;

            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, bool EQUAL_NAN, int ITERATIONS>
inline void call_isclose_fresh_cpp(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_out,
    const VectorMode vector_mode,
    const std::uint32_t rtol_bits,
    const std::uint32_t atol_bits)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh isclose expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh isclose expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh isclose expects full-tile vector mode");

    // Anchor the dynamic tile once in the wrapper so the isolated semantic
    // body contains only constant relative Dst addresses (the fresh binary
    // max/min, add/sub, and mul precedent).
    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_isclose_fresh_cpp<EQUAL_NAN, ITERATIONS>(rtol_bits, atol_bits);
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu
