// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Compatibility names for existing corpus rows. The tile implementation now
// lives in the production header; only the register-chain probe stays here.
#include "fresh_common.h"
#include "../../../../common/ckernel_sfpu_ema_explicit.h"

namespace ckernel::sfpu
{
using EmaFreshState = EmaState;
sfpi_inline void ema_fresh_state_init(EmaFreshState& st, std::uint32_t alpha, std::uint32_t beta)
{
    ema_state_init(st, alpha, beta);
}
template <int Contract>
sfpi_inline void ema_fresh_quad(EmaFreshState& st, std::uint32_t base, std::uint32_t offset)
{
    ema_explicit_quad<Contract>(st, base, offset);
}
template <int Contract = 1>
inline void _calculate_ema_fresh_tile_(EmaFreshState& st)
{
    _calculate_ema_explicit_tile_<Contract>(st);
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
