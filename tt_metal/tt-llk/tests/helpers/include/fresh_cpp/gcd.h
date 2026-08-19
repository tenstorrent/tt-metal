// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// gcd — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// Binary GCD (Stein's algorithm, published formulation): strip the common
// power of two, keep a odd, then repeatedly strip b's factors of two,
// order (a, b), and subtract; gcd = a << common_shift.  Stated in plain
// typed vInt/vUInt with per-lane predication; ctz(v) = 31 - lz(v & -v).
// The subtract-shift loop runs a fixed 34 rounds: each round leaves b even
// and the following strip halves it at least once, so 17-bit operands
// (stimulus domain [1, 1e5], test_sfpu_binary._INT_BINARY_STIMULI) terminate
// in <= 34 rounds (measured worst case 17 over 2e5 samples:
// laneS2-evidence-20260819/fit_s2.py); finished lanes are predicated off on
// b == 0.  Contract is the node's positive Int32 domain (gcd(a, 0) = a holds
// structurally).  Golden: torch.gcd (golden_generators._gcd), exact Int32.
// Production: metal ckernel_sfpu_gcd.h (hand-issued REPLAY-loop kernel).
#include <cstdint>

namespace ckernel::sfpu
{

// ctz for a nonzero, non-negative lane value.
sfpi_inline sfpi::vInt fresh_gcd_ctz(const sfpi::vInt v)
{
    const sfpi::vInt isolated = v & (sfpi::vInt(0) - v);
    return sfpi::vInt(31) - sfpi::as<sfpi::vInt>(sfpi::lz(isolated));
}

sfpi_inline sfpi::vInt fresh_gcd_shift_right(const sfpi::vInt v, const sfpi::vInt amount)
{
    return sfpi::as<sfpi::vInt>(sfpi::shft(sfpi::as<sfpi::vUInt>(v), sfpi::vInt(0) - amount));
}

template <int ITERATIONS>
__attribute__((noinline)) void calculate_gcd_fresh_cpp()
{
    constexpr std::uint32_t tile_rows = 32;
    // Fixed round bound: see header comment (17-bit stimulus domain).
    constexpr int GCD_ROUNDS = 34;

    for (int face = 0; face < 4; ++face)
    {
        for (int row = 0; row < ITERATIONS; ++row)
        {
            sfpi::vInt a = sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>();
            sfpi::vInt b = sfpi::dst_reg[tile_rows].mode<sfpi::DataLayout::SM32>();

            // Common power of two, then keep a odd.
            const sfpi::vInt common = fresh_gcd_ctz(a | b);
            a                       = fresh_gcd_shift_right(a, fresh_gcd_ctz(a));

            for (int round = 0; round < GCD_ROUNDS; ++round)
            {
                v_if (b != 0)
                {
                    b = fresh_gcd_shift_right(b, fresh_gcd_ctz(b));
                    // Order so a <= b, then subtract (keeps a odd, b even).
                    const sfpi::vInt t = a;
                    v_if (b < a)
                    {
                        a = b;
                        b = t;
                    }
                    v_endif;
                    b = b - a;
                }
                v_endif;
            }

            const sfpi::vInt g                              = sfpi::as<sfpi::vInt>(sfpi::shft(sfpi::as<sfpi::vUInt>(a), common));
            sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>() = g;
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS>
inline void call_gcd_fresh_cpp(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh gcd expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh gcd expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh gcd expects full-tile vector mode");

    // Anchor the dynamic tile once in the wrapper so the isolated semantic body
    // contains only constant relative Dst addresses (the fresh add/sub/mul
    // precedent).
    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_gcd_fresh_cpp<ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu
