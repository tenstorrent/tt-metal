// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// gcd — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// Binary GCD (Stein's algorithm) stated as UNPREDICATED fixed-point rounds:
// strip the common power of two and keep a odd, then run a fixed number of
// strip-sort-subtract rounds
//
//   b <- b >> ctz(b);  (a, b) <- (min(a, b), max(a, b) - min(a, b))
//
// with NO per-round lane predication.  Termination is a fixed point instead
// of a v_if: once a lane's remainder hits zero the state walks
// (g, 0) -> (0, g) and (0, g) is a fixed point of the round (strip keeps 0
// at 0 because ctz's lz(0)=32 spelling shifts 0 by one bit, and 0 stays 0;
// the sort puts 0 first; the subtract returns g), so extra rounds are
// harmless and the odd gcd is a + b at and after the first zero round.
// The ordering step is the typed min/max sort (one architectural SFPSWAP);
// operands are non-negative, where sign-magnitude and integer order
// coincide.  gcd = (a + b) << common.
//
// Round bound: 17.  Exhaustively certified over the row's whole stimulus
// domain (test_sfpu_binary._INT_BINARY_STIMULI[SfpuGcd] = uniform
// [1, 100000]): a DP over all 1.25e9 odd pairs (x, y), 1 <= x <= y <= 1e5,
// of rounds-to-zero for exactly this round shows MAX = 17, witness pair
// (65535, 65537) = 2^16 -+ 1 (laneCS-evidence-20260820/gcd_round_cert.c;
// forward-sound because strip and subtract never increase a lane's max
// component, so every reachable state stays inside the certified set).
// gcd(a, 0) = a holds structurally (the (0, a) fixed point).  Golden:
// torch.gcd (golden_generators._gcd), exact Int32.
// Production: metal ckernel_sfpu_gcd.h (hand-issued REPLAY-loop kernel).
// The previous 34-round predicated body is preserved in
// fresh_cpp/gcd_legacy.h.
#include <cstdint>

namespace ckernel::sfpu
{

// Right-shift amount that drops v's trailing zeros: -ctz(v) = clz(isolated
// lowest set bit) - 31 (negative = logical right shift for sfpi::shft).
// Stated as the shift amount directly so the consumer needs no re-negation
// and the 31 stays an sfpiadd immediate.  For v == 0 the isolated bit is 0,
// lz(0) = 32 gives +1 (a one-bit left shift), and 0 << 1 == 0 — zero is
// preserved, which the fixed-point rounds below rely on.
sfpi_inline sfpi::vInt fresh_gcd_ctz_shift(const sfpi::vInt v)
{
    const sfpi::vInt iso = v & (sfpi::vInt(0) - v);
    return sfpi::as<sfpi::vInt>(sfpi::lz(iso)) - 31;
}

template <int ITERATIONS>
__attribute__((noinline)) void calculate_gcd_fresh_cpp()
{
    constexpr std::uint32_t tile_rows = 32;
    // Certified round bound for the 17-bit stimulus domain: see header.
    constexpr int GCD_ROUNDS = 17;

    for (int face = 0; face < 4; ++face)
    {
        for (int row = 0; row < ITERATIONS; ++row)
        {
            sfpi::vInt a = sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>();
            sfpi::vInt b = sfpi::dst_reg[tile_rows].mode<sfpi::DataLayout::SM32>();

            // Common power of two (kept as the negative shift amount), then
            // keep a odd.
            const sfpi::vInt common_shift = fresh_gcd_ctz_shift(a | b); // -common
            a                             = sfpi::shft(a, fresh_gcd_ctz_shift(a), sfpi::ShiftMode::Logical);

            for (int round = 0; round < GCD_ROUNDS; ++round)
            {
                b                  = sfpi::shft(b, fresh_gcd_ctz_shift(b), sfpi::ShiftMode::Logical);
                const auto ordered = sfpi::min_max(sfpi::as<sfpi::vSMag>(a), sfpi::as<sfpi::vSMag>(b));
                a                  = sfpi::as<sfpi::vInt>(ordered.first);
                b                  = sfpi::as<sfpi::vInt>(ordered.second) - a;
            }

            const sfpi::vInt g                              = sfpi::shft(a + b, sfpi::vInt(0) - common_shift, sfpi::ShiftMode::Logical);
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
