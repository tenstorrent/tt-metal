// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// SEMANTIC twin of blaze/kernels/sfpu/logit_softcap_sfpu.hpp.
//
// The original body is ALREADY plain typed SFPI (y = cap * tanh(x)); the only
// obstruction to racing/testing it as an ordinary math-thread SFPU kernel is
// its `#ifdef TRISC_PACK` gate (blaze drives it from the pack thread).  This
// twin carries the byte-equivalent body under a MATH-or-PACK gate so
// tt-metal-style math-thread harnesses can compile and race it.  The original
// is byte-untouched; the underlying _sfpu_tanh_fp32_accurate_ helper is
// thread-agnostic.

#pragma once

#if defined(TRISC_PACK) || defined(TRISC_MATH)
#include "ckernel_sfpu_tanh.h"

namespace ckernel {
namespace sfpu {
namespace semantic {

template <int ITERATIONS>
inline void calculate_logit_softcap(uint32_t cap_bits, uint32_t) {
    for (int d = 0; d < ITERATIONS; d++) {
        // ponytail: compute tanh first, load cap after -> cap not live across the
        // register-heavy tanh (was an SFPI reload ICE at -O3). Algebraically identical.
        sfpi::vFloat t = _sfpu_tanh_fp32_accurate_(sfpi::dst_reg[0]);
        sfpi::vFloat cap = sfpi::as<sfpi::vFloat>(sfpi::vInt(cap_bits));
        sfpi::dst_reg[0] = cap * t;
        sfpi::dst_reg++;
    }
}

}  // namespace semantic
}  // namespace sfpu
}  // namespace ckernel
#endif  // TRISC_PACK || TRISC_MATH
