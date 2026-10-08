// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include "sfpi.h"

// Opt-in EMA: caller-owned state, input Dst tile 0, output tile 1.
// Silicon validation currently covers Blackhole; see the scope below.
// Same transpose/layout and recurrence as the raw-register implementation.
// Contract 1 is the FMA candidate checked bit-exact against the legacy path;
// contract 2 is a distinct arithmetic ordering, NOT a bit-exact substitute.
// Validation scope and compiler options: tests/corpus/RAW_LREG_EXPERIMENT.md.
namespace ckernel::sfpu
{
// Caller-held EMA state quad (scrambled space between calls).
struct EmaState
{
    sfpi::vUInt s0, s1, s2, s3; // carry, alpha, beta, spare (natural order)
};

// Program the state quad DIRECTLY IN SCRAMBLED SPACE.  Call ONCE before the
// tile loop (alpha/beta as raw fp32 bit patterns; carry starts at 0).
//
// The scrambled image of the natural state (s0=carry0, s1=alpha, s2=beta,
// s3=spare) is the same vector in every slot: transposing maps
// scrambled_k row_j = natural_j row_k, and carry/alpha/beta are
// row-uniform. Constructing that image directly keeps initialization in typed
// values without needing a transpose of four otherwise unused data inputs.
sfpi_inline void ema_state_init(EmaState& st, const std::uint32_t alpha_bits, const std::uint32_t beta_bits)
{
    sfpi::vFloat s = 0.0f;
    v_if (sfpi::lane_row() == 1) {
        s = sfpi::as<sfpi::vFloat>(sfpi::vInt(static_cast<int>(alpha_bits)));
    }
    v_endif;
    v_if (sfpi::lane_row() == 2) {
        s = sfpi::as<sfpi::vFloat>(sfpi::vInt(static_cast<int>(beta_bits)));
    }
    v_endif;
    st.s0 = sfpi::as<sfpi::vUInt>(s);
    st.s1 = st.s0;
    st.s2 = st.s0;
    st.s3 = st.s0;
}

// One 4-row quad: rows (4*quad .. 4*quad+3) of the input tile at dst addr
// base, EMA'd against the running carry, written to the output tile
// (dst addr base + out_offset).  Address groups per the hand kernel:
// {base, base+2, base+16, base+18} = the quad's four tile rows across all
// 32 columns (faces 0/1 or 2/3, even/odd column halves).
template <int Contract>
sfpi_inline void ema_explicit_quad(EmaState& st, const std::uint32_t base, const std::uint32_t out_offset)
{
    using namespace sfpi;
    const std::uint32_t i = base / 2;
    vFloat x0 = dst_reg[i + 0];
    vFloat x1 = dst_reg[i + 1];
    vFloat x2 = dst_reg[i + 8];
    vFloat x3 = dst_reg[i + 9];

    // Entry transpose: data to row-per-register space, state to natural.
    transp8(x0, x1, x2, x3, st.s0, st.s1, st.s2, st.s3);

    vFloat carry = as<vFloat>(st.s0);
    vFloat alpha = as<vFloat>(st.s1);
    vFloat beta  = as<vFloat>(st.s2);
    vFloat t;
    if constexpr (Contract == 1)
    {
        // fma: t single-rounded, then one single-rounded MAD per step.
        t  = alpha * carry;
        x0 = beta * x0 + t;
        t  = alpha * x0;
        x1 = beta * x1 + t;
        t  = alpha * x1;
        x2 = beta * x2 + t;
        t  = alpha * x2;
        x3 = beta * x3 + t;
    }
    else
    {
        // mul_add: every product and sum individually rounded.
        t  = beta * x0;
        x0 = alpha * carry + t;
        t  = beta * x1;
        x1 = alpha * x0 + t;
        t  = beta * x2;
        x2 = alpha * x1 + t;
        t  = beta * x3;
        x3 = alpha * x2 + t;
    }
    st.s0 = as<vUInt>(x3); // new carry (the hand kernel's SFPMOV L3->L4)
    vUInt sp = as<vUInt>(t);

    // Exit transpose: data back to store layout, state back to scrambled.
    transp8(x0, x1, x2, x3, st.s0, st.s1, st.s2, sp);
    st.s3 = sp;

    dst_reg[out_offset / 2 + i + 0] = x0;
    dst_reg[out_offset / 2 + i + 1] = x1;
    dst_reg[out_offset / 2 + i + 8] = x2;
    dst_reg[out_offset / 2 + i + 9] = x3;
}

// One full 32x32 tile: input at dst tile 0, output at dst tile 1 (addr +64),
// carry continued through st.  Quad walk identical to the hand kernel's
// _process_ema_block_ order (rows 0..31).
template <int Contract = 1>
inline void _calculate_ema_explicit_tile_(EmaState& st)
{
    ema_explicit_quad<Contract>(st, 0, 64);
    ema_explicit_quad<Contract>(st, 4, 64);
    ema_explicit_quad<Contract>(st, 8, 64);
    ema_explicit_quad<Contract>(st, 12, 64);
    ema_explicit_quad<Contract>(st, 32, 64);
    ema_explicit_quad<Contract>(st, 36, 64);
    ema_explicit_quad<Contract>(st, 40, 64);
    ema_explicit_quad<Contract>(st, 44, 64);
}

} // namespace ckernel::sfpu
