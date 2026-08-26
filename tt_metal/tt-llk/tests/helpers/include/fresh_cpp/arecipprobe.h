// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Lane GW ISA-unlock certification probes for the two SFPARECIP modes the
// compiler surface ships but nothing exercised (GS-3): Mod1=2 EXP
// (sign-preserved e^|x| seed) and Mod1=1 COND_RECIP (recip of |x| where the
// source is negative, sign NOT rejoined).  Each body is the bare mode applied
// row-at-a-time — the row's golden is the ISA functional model transcription
// (tt-isa-documentation BlackholeA0 SFPARECIP.md ApproxExp/ApproxRecip,
// golden_generators.py UnarySFPUGolden._approx_exp_probe/_approx_cond_recip_probe)
// compared EXACTLY (atol=rtol=0) on the Float32/dest_acc=Yes pipeline (the
// lane-CX reachability discipline: the only pipeline that delivers and
// returns full fp32 bit patterns).
//
//   * a CRAQ leg (extended craq-sim) proves the sim transcription,
//   * a device leg adjudicates doc-vs-silicon (where-adjudication precedent).
#include <cstdint>

#include "fresh_common.h"

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_approx_exp_probe_cpp()
{
#if __riscv_xtttensixwh
    static_assert(fresh_hwseed_supported_on_wh<ITERATIONS>::value, "SFPARECIP probes require BH SFPARECIP");
#else
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat v   = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = sfpi::approx_exp(v);
        sfpi::dst_reg++;
    }
#endif
}

template <int ITERATIONS>
__attribute__((noinline)) void calculate_approx_cond_recip_probe_cpp()
{
#if __riscv_xtttensixwh
    static_assert(fresh_hwseed_supported_on_wh<ITERATIONS>::value, "SFPARECIP probes require BH SFPARECIP");
#else
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat v   = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = sfpi::approx_recip(v, sfpi::RecipMode::IfNegative);
        sfpi::dst_reg++;
    }
#endif
}

} // namespace ckernel::sfpu
