// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// FRESH SEMANTIC EMA (lane FK, 2026-08-22) — the first typed version of the
// EMA mechanism (ema-fresh was SKIP_NOT_FEASIBLE before the typed cross-lane
// surface existed: laneED audit named the SFPTRANSP bracketing and the
// raw-LREG alpha/beta/carry ABI as the blockers; sfpi::transp8 (lane FA X1)
// plus caller-held typed state dissolve both).
//
// CONTRACT (identical to the hand kernel ckernel_sfpu_ema.h and the
// crosslane_fixtures/ema.json register-chain golden):
//   y_t = alpha * y_{t-1} + beta * x_t     down the 32 tile rows (t = time),
//   32 columns in parallel; input tile at dst index 0, output tile at dst
//   index 1; the carry y_{-1} continues across tiles.
// TWO arithmetic contracts (lane FB demand golden, both recorded):
//   Contract 1 "fma":     t = fp32(alpha*y);  y' = fp32(beta*x + t)
//                         (two SFPMADs, each single-rounded — the hand
//                         kernel's exact dataflow: TTI_SFPMAD pairs)
//   Contract 2 "mul_add": t1 = fp32(alpha*y); t2 = fp32(beta*x);
//                         y' = fp32(t1 + t2)  (three roundings)
//
// MECHANISM (the hand kernel's, typed): per 4-row quad the data is
// SFPTRANSP'd so each register holds one tile row; the serial MAD chain then
// runs along the register axis; a second SFPTRANSP restores the store
// layout.  Because SFPTRANSP always permutes BOTH register banks
// (SFPTRANSP.md), the alpha/beta/carry state rides the companion bank in a
// PRE-SCRAMBLED form: one transpose at kernel start puts it in the scrambled
// space, and every quad's entry transpose lands it back in natural space
// exactly while the MADs read it (the hand kernel's outer-transpose trick,
// stated in types).  The caller owns the state quad (typed cross-call
// persistence: plain locals outside the tile loop — the typed spelling of
// the hand kernel's LREG4/5/6 cross-call ABI).
//
// Registers: 4 data + 4 state = the full LREG file through every transp8;
// the MAD temp takes the state quad's spare slot, exactly like the hand
// kernel's LREG7.

#include "fresh_common.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

// Caller-held EMA state quad (scrambled space between calls).
struct EmaFreshState
{
    sfpi::vUInt s0, s1, s2, s3; // carry, alpha, beta, spare (natural order)
};

// Program the state quad DIRECTLY IN SCRAMBLED SPACE.  Call ONCE before the
// tile loop (alpha/beta as raw fp32 bit patterns; carry starts at 0).
//
// The scrambled image of the natural state (s0=carry0, s1=alpha, s2=beta,
// s3=spare) is the same vector in every slot: transposing maps
// scrambled_k row_j = natural_j row_k, and carry/alpha/beta are
// row-uniform, so every scrambled register holds (0, alpha, beta, 0) down
// its four subvector rows.  Building that image with two lane-row
// predicates avoids a transpose whose four VALUE outputs would all be dead
// — a shape on which the compiler loses the transp8's companion-bank
// writes (all-dead-value-outputs transp8, lane-FK finding: the readlreg
// collection then reads pre-transpose garbage; witness archived in the
// lane evidence).
sfpi_inline void ema_fresh_state_init(EmaFreshState& st, const std::uint32_t alpha_bits, const std::uint32_t beta_bits)
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
sfpi_inline void ema_fresh_quad(EmaFreshState& st, const std::uint32_t base, const std::uint32_t out_offset)
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
inline void _calculate_ema_fresh_tile_(EmaFreshState& st)
{
    ema_fresh_quad<Contract>(st, 0, 64);
    ema_fresh_quad<Contract>(st, 4, 64);
    ema_fresh_quad<Contract>(st, 8, 64);
    ema_fresh_quad<Contract>(st, 12, 64);
    ema_fresh_quad<Contract>(st, 32, 64);
    ema_fresh_quad<Contract>(st, 36, 64);
    ema_fresh_quad<Contract>(st, 40, 64);
    ema_fresh_quad<Contract>(st, 44, 64);
}

// ---------------------------------------------------------------------------
// Register-chain probe (lane FB crosslane_fixtures/ema.json): the fixture's
// scan-free reformulation, y_i = alpha*x_i + beta*y_{i-1} along dst_reg[0..7]
// with y_{-1} = dst_reg[8], results in place.  fp32 Dst (dest_acc on); pure
// serial vector chain, no cross-lane movement — this pins the ARITHMETIC
// contract on the pinned simulator (a third value = a finding).
// ---------------------------------------------------------------------------
template <int Contract>
inline void _calculate_ema_fresh_rowchain8_(const std::uint32_t alpha_bits, const std::uint32_t beta_bits)
{
    using namespace sfpi;
    vFloat alpha = as<vFloat>(vInt(static_cast<int>(alpha_bits)));
    vFloat beta  = as<vFloat>(vInt(static_cast<int>(beta_bits)));
    vFloat y     = dst_reg[8];
#pragma GCC unroll 8
    for (int r = 0; r < 8; ++r)
    {
        vFloat x = dst_reg[r];
        if constexpr (Contract == 1)
        {
            vFloat t = beta * y;
            y        = alpha * x + t;
        }
        else
        {
            vFloat t1 = alpha * x;
            vFloat t2 = beta * y;
            y         = t1 + t2;
        }
        dst_reg[r] = y;
    }
}

} // namespace ckernel::sfpu
