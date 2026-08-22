// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// FRESH SEMANTIC CUMSUM (lane FK, 2026-08-22) — the first typed version of
// the cumsum mechanism (cumsum-fresh was SKIP_NOT_FEASIBLE before the typed
// cross-lane surface existed: laneEU census named the SFPTRANSP-bracketed
// replay blocks and the LREG4-7 running-state ABI; sfpi::transp8 plus
// caller-held typed state dissolve both).
//
// CONTRACT (identical to the production hand kernel ckernel_sfpu_cumsum.h
// and the crosslane_fixtures/cumsum.json register-chain golden): inclusive
// prefix sum down the 32 tile rows, per column, in place (tile at dst index
// 0); the running prefix continues across tiles.  The fold is SERIAL low
// row -> high row, ONE rounding per add (the order is the contract).
//
// MECHANISM (the production kernel's, typed): per 4-row quad the data is
// SFPTRANSP'd to row-per-register space, chained with serial adds against
// the carry, and SFPTRANSP'd back.  The carry rides the OTHER register bank
// exactly as in the production kernel (its LREG0-3 / LREG4-7 ping-pong):
// the two quad variable sets keep FIXED bank slots across every transp8
// call — the DATA SIDE alternates instead.  (Alternating the argument banks
// would demand a full 8-value cross-bank swap between quads, which needs a
// ninth register — compile-proven lreg-pressure-exceeded, first draft of
// this file.)  The previous quad's row-3 register after the entry transpose
// IS the carry.  The caller owns both quads across tiles (typed cross-call
// persistence; the first tile's other-side quad starts at zero = prefix 0).
//
// Two element contracts:
//   fp32/bf16: SFPADD per step (one fp rounding per add);
//   int32:     SFPIADD per step (exact mod 2^32).

#include "fresh_common.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

// Caller-held quad pair with fixed bank slots (a = value bank, b =
// companion bank; b carries bit-pattern views on the fp arm).
struct CumsumFreshState
{
    sfpi::vFloat a0, a1, a2, a3;
    sfpi::vUInt b0, b1, b2, b3;
};

struct CumsumFreshStateInt
{
    sfpi::vUInt a0, a1, a2, a3;
    sfpi::vUInt b0, b1, b2, b3;
};

sfpi_inline void cumsum_fresh_state_init(CumsumFreshState& st)
{
    st.a0 = 0.0f;
    st.a1 = 0.0f;
    st.a2 = 0.0f;
    st.a3 = 0.0f;
    st.b0 = 0;
    st.b1 = 0;
    st.b2 = 0;
    st.b3 = 0;
}

sfpi_inline void cumsum_fresh_state_init(CumsumFreshStateInt& st)
{
    st.a0 = 0;
    st.a1 = 0;
    st.a2 = 0;
    st.a3 = 0;
    st.b0 = 0;
    st.b1 = 0;
    st.b2 = 0;
    st.b3 = 0;
}

// ---- fp arm ---------------------------------------------------------------
// Address groups per the production kernel: {base, base+2, base+16,
// base+18} = the quad's four tile rows across all 32 columns.

// Data on the a (value-bank) side; carry = b3 after the entry transpose.
sfpi_inline void cumsum_fresh_quad_fp_a(CumsumFreshState& st, const std::uint32_t base)
{
    using namespace sfpi;
    const std::uint32_t i = base / 2;
    st.a0                 = dst_reg[i + 0];
    st.a1                 = dst_reg[i + 1];
    st.a2                 = dst_reg[i + 8];
    st.a3                 = dst_reg[i + 9];
    transp8(st.a0, st.a1, st.a2, st.a3, st.b0, st.b1, st.b2, st.b3);
    st.a0 = as<vFloat>(st.b3) + st.a0;
    st.a1 = st.a0 + st.a1;
    st.a2 = st.a1 + st.a2;
    st.a3 = st.a2 + st.a3;
    transp8(st.a0, st.a1, st.a2, st.a3, st.b0, st.b1, st.b2, st.b3);
    dst_reg[i + 0] = st.a0;
    dst_reg[i + 1] = st.a1;
    dst_reg[i + 8] = st.a2;
    dst_reg[i + 9] = st.a3;
}

// Data on the b (companion-bank) side; carry = a3 after the entry transpose.
sfpi_inline void cumsum_fresh_quad_fp_b(CumsumFreshState& st, const std::uint32_t base)
{
    using namespace sfpi;
    const std::uint32_t i = base / 2;
    st.b0                 = as<vUInt>(vFloat(dst_reg[i + 0]));
    st.b1                 = as<vUInt>(vFloat(dst_reg[i + 1]));
    st.b2                 = as<vUInt>(vFloat(dst_reg[i + 8]));
    st.b3                 = as<vUInt>(vFloat(dst_reg[i + 9]));
    transp8(st.a0, st.a1, st.a2, st.a3, st.b0, st.b1, st.b2, st.b3);
    st.b0 = as<vUInt>(st.a3 + as<vFloat>(st.b0));
    st.b1 = as<vUInt>(as<vFloat>(st.b0) + as<vFloat>(st.b1));
    st.b2 = as<vUInt>(as<vFloat>(st.b1) + as<vFloat>(st.b2));
    st.b3 = as<vUInt>(as<vFloat>(st.b2) + as<vFloat>(st.b3));
    transp8(st.a0, st.a1, st.a2, st.a3, st.b0, st.b1, st.b2, st.b3);
    dst_reg[i + 0] = as<vFloat>(st.b0);
    dst_reg[i + 1] = as<vFloat>(st.b1);
    dst_reg[i + 8] = as<vFloat>(st.b2);
    dst_reg[i + 9] = as<vFloat>(st.b3);
}

// ---- int arm ---------------------------------------------------------------

sfpi_inline void cumsum_fresh_quad_int_a(CumsumFreshStateInt& st, const std::uint32_t base)
{
    using namespace sfpi;
    const std::uint32_t i = base / 2;
    st.a0                 = dst_reg[i + 0];
    st.a1                 = dst_reg[i + 1];
    st.a2                 = dst_reg[i + 8];
    st.a3                 = dst_reg[i + 9];
    transp8(st.a0, st.a1, st.a2, st.a3, st.b0, st.b1, st.b2, st.b3);
    st.a0 = as<vUInt>(as<vInt>(st.b3) + as<vInt>(st.a0));
    st.a1 = as<vUInt>(as<vInt>(st.a0) + as<vInt>(st.a1));
    st.a2 = as<vUInt>(as<vInt>(st.a1) + as<vInt>(st.a2));
    st.a3 = as<vUInt>(as<vInt>(st.a2) + as<vInt>(st.a3));
    transp8(st.a0, st.a1, st.a2, st.a3, st.b0, st.b1, st.b2, st.b3);
    dst_reg[i + 0] = st.a0;
    dst_reg[i + 1] = st.a1;
    dst_reg[i + 8] = st.a2;
    dst_reg[i + 9] = st.a3;
}

sfpi_inline void cumsum_fresh_quad_int_b(CumsumFreshStateInt& st, const std::uint32_t base)
{
    using namespace sfpi;
    const std::uint32_t i = base / 2;
    st.b0                 = dst_reg[i + 0];
    st.b1                 = dst_reg[i + 1];
    st.b2                 = dst_reg[i + 8];
    st.b3                 = dst_reg[i + 9];
    transp8(st.a0, st.a1, st.a2, st.a3, st.b0, st.b1, st.b2, st.b3);
    st.b0 = as<vUInt>(as<vInt>(st.a3) + as<vInt>(st.b0));
    st.b1 = as<vUInt>(as<vInt>(st.b0) + as<vInt>(st.b1));
    st.b2 = as<vUInt>(as<vInt>(st.b1) + as<vInt>(st.b2));
    st.b3 = as<vUInt>(as<vInt>(st.b2) + as<vInt>(st.b3));
    transp8(st.a0, st.a1, st.a2, st.a3, st.b0, st.b1, st.b2, st.b3);
    dst_reg[i + 0] = st.b0;
    dst_reg[i + 1] = st.b1;
    dst_reg[i + 8] = st.b2;
    dst_reg[i + 9] = st.b3;
}

// One full 32x32 tile in place at dst tile 0, prefix continued through st.
// Quad walk = the production kernel's block order (rows 0..31: faces 0/1
// then faces 2/3), sides alternating exactly as the production kernel
// alternates LREG0-3 / LREG4-7.
inline void _calculate_cumsum_fresh_tile_(CumsumFreshState& st)
{
    cumsum_fresh_quad_fp_a(st, 0);
    cumsum_fresh_quad_fp_b(st, 4);
    cumsum_fresh_quad_fp_a(st, 8);
    cumsum_fresh_quad_fp_b(st, 12);
    cumsum_fresh_quad_fp_a(st, 32);
    cumsum_fresh_quad_fp_b(st, 36);
    cumsum_fresh_quad_fp_a(st, 40);
    cumsum_fresh_quad_fp_b(st, 44);
}

inline void _calculate_cumsum_fresh_tile_int_(CumsumFreshStateInt& st)
{
    cumsum_fresh_quad_int_a(st, 0);
    cumsum_fresh_quad_int_b(st, 4);
    cumsum_fresh_quad_int_a(st, 8);
    cumsum_fresh_quad_int_b(st, 12);
    cumsum_fresh_quad_int_a(st, 32);
    cumsum_fresh_quad_int_b(st, 36);
    cumsum_fresh_quad_int_a(st, 40);
    cumsum_fresh_quad_int_b(st, 44);
}

// ---------------------------------------------------------------------------
// Register-chain probes (lane FB crosslane_fixtures/cumsum.json): inclusive
// prefix along dst_reg[0..7], results in place, serial low->high.  fp arm on
// fp32 Dst; int arm on Int32 Dst.  Bit-exact against the fixture goldens on
// the pinned simulator.
// ---------------------------------------------------------------------------
inline void _calculate_cumsum_fresh_rowchain8_fp_()
{
    using namespace sfpi;
    vFloat s = dst_reg[0];
#pragma GCC unroll 8
    for (int r = 1; r < 8; ++r)
    {
        s          = s + vFloat(dst_reg[r]);
        dst_reg[r] = s;
    }
}

inline void _calculate_cumsum_fresh_rowchain8_int_()
{
    using namespace sfpi;
    vInt s = dst_reg[0];
#pragma GCC unroll 8
    for (int r = 1; r < 8; ++r)
    {
        s          = s + vInt(dst_reg[r]);
        dst_reg[r] = s;
    }
}

} // namespace ckernel::sfpu
