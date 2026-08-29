// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// mul_int32 (limb-2) — canonical semantic C++ body (storm contract,
// fresh_cpp/README.md).  Included by fresh_cpp_operations.h AFTER the
// impl-1 body (calculate_mul_int_fresh_cpp) it delegates to on WH.

#include <cstdint>

namespace ckernel::sfpu
{

// Fresh typed-C++ int32 multiply, impl 2 (lane GG, 2026-08-24): CONTRACT-DOMAIN
// limb-2 form.  The corpus row's golden contract (test_fresh_cpp_mul_int /
// test_sfpu_binary_int_uniform SfpuMulInt32) draws operands uniform from
// [1, 40000] -- magnitudes < 2^16, products < 2^31 -- because the
// sign-magnitude Dst packer only round-trips non-negative low-32 products.
// On that contract the radix-23 cross terms of the full identity are
// identically zero: for raw operands a, b < 2^23,
//
//     a * b mod 2^32 == SFPMUL24_LOWER(a, b) + (SFPMUL24_UPPER(a, b) << 23)
//
// exactly (LOWER keeps the low 23 bits, which mod 2^32 does not disturb;
// UPPER's 23-bit operand masking is identity on the domain, so it yields
// (a*b)>>23 untruncated; the recombination has no carry because lo < 2^23).
// Certified against the pinned sim's byte-equivalent SFPMUL24 model, tied to
// the impl-1 full-domain form: exhaustive over ALL 2^32 pairs in [0,2^16)^2
// plus 5.5e9 directed-slice evaluations covering [0,2^23)
// (laneGG-evidence-20260824/mulint32_limb2_cert.c, zero failures).  Same
// domain-restriction discipline as divint32floor (<2^24) and lcm (<2^15):
// the restriction *is* the documented row contract, stated here and in the
// sweep row note.  Operands with magnitude >= 2^23 would need impl 1's cross
// terms (the production kernel's full identity) -- outside this body's
// contract.  Two SFPMUL24s, one shift, one add per row: 7 issued words/row
// against impl 1's 13-word formed calendar and the hand SFPLOADMACRO
// kernel's 8 issue slots/row.  No fixed LREGs, raw instructions, replay
// slots, or SFPLOADMACRO templates.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_mul_int_limb2_fresh_cpp()
{
#if !(__riscv_xtttensixbh || __riscv_xtttensixqsr)
    // WH has no 24-bit integer multiply primitive; the limb-2 form is a
    // BH/QSR vocabulary win.  Keep impl 2 compilable on WH by delegating to
    // the full-domain impl-1 emulation body (the sweep row is bh-scoped).
    calculate_mul_int_fresh_cpp<ITERATIONS>();
#else
    constexpr std::uint32_t tile_rows = 32;

#pragma GCC unroll 4
    for (int face = 0; face < 4; ++face)
    {
#pragma GCC unroll 8
        for (int row = 0; row < ITERATIONS; ++row)
        {
            const sfpi::vInt a                              = sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>();
            const sfpi::vInt b                              = sfpi::dst_reg[tile_rows].mode<sfpi::DataLayout::SM32>();
            const sfpi::vUInt ua                            = sfpi::as<sfpi::vUInt>(a);
            const sfpi::vUInt ub                            = sfpi::as<sfpi::vUInt>(b);
            const sfpi::vUInt hi                            = sfpi::fractional_mul(ua, ub, sfpi::FractionalHalf::High);
            const sfpi::vUInt lo                            = sfpi::fractional_mul(ua, ub, sfpi::FractionalHalf::Low);
            const sfpi::vUInt product                       = lo + (hi << 23);
            sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>() = sfpi::as<sfpi::vInt>(product);
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
#endif
}

template <DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS>
inline void call_mul_int_limb2_fresh_cpp(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh mul int limb2 expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh mul int limb2 expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh mul int limb2 expects full-tile vector mode");

    // Anchor the dynamic tile once in the wrapper so the isolated semantic body
    // contains only constant relative Dst addresses (same idiom as the fresh
    // max/min and mul-int impl-1 selectors).
    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_mul_int_limb2_fresh_cpp<ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu
