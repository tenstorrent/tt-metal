// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"
#include "sfpu_mode.h"

namespace ckernel::sfpu {

// softsign(x) = x / (1 + |x|), one instruction sequence per SfpuMode:
//
//   Mode                 reciprocal          special-value guard
//   PreciseStrict        2 Newton steps      yes
//   ApproximateStrict    0 Newton steps      yes
//   PreciseRelaxed       2 Newton steps      no   (the legacy APPROXIMATION_MODE=false sequence)
//   ApproximateRelaxed   0 Newton steps      no   (the legacy APPROXIMATION_MODE=true sequence)
//
// Strict contract, matching torch.nn.functional.softsign in fp32:
//   +-0 -> +-0;  NaN -> NaN;  +-inf -> NaN (inf / inf);  finite |x| >= 2**126 -> +-1.
// NaN results are the canonical +NaN SFPMAD produces.
// The last row is the guard: sfpi::approx_recip returns 0 once 1 + |x| >= 2**126, so the unguarded
// sequence returns +-0 there instead of +-1. Relaxed callers vouch that |x| < 2**126.
template <SfpuMode MODE, int ITERATIONS>
inline void calculate_softsign() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat tmp = sfpi::abs(v) + 1.0f;
        tmp = sfpu_reciprocal<is_approximate(MODE)>(tmp);
        sfpi::vFloat result;
        if constexpr (is_strict(MODE)) {
            // A plain product is SFPMAD with a +0 addend, which turns softsign(-0) into +0; a -0 addend keeps
            // the product's zero sign and changes nothing else. Same instruction count.
            result = v * tmp - 0.0f;
            // Unbiased exponent 126 or 127: the finite band the reciprocal flushes. Exponent 128 (inf, NaN)
            // is left alone, so those lanes keep the canonical NaN SFPMAD produced.
            sfpi::vInt exponent = sfpi::exexp(v);
            v_if(exponent >= 126 && exponent < 128) { result = sfpi::copysgn(sfpi::vFloat(1.0f), v); }
            v_endif;
        } else {
            result = v * tmp;
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <SfpuMode MODE>
void init_softsign() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<is_approximate(MODE)>();
}

// Legacy entry points. APPROXIMATION_MODE alone never guarded the special values, so they map to the
// Relaxed modes and emit exactly the sequence they always did.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softsign() {
    calculate_softsign<make_sfpu_mode(APPROXIMATION_MODE, false /*strict*/), ITERATIONS>();
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    init_softsign<make_sfpu_mode(APPROXIMATION_MODE, false /*strict*/)>();
}

}  // namespace ckernel::sfpu
