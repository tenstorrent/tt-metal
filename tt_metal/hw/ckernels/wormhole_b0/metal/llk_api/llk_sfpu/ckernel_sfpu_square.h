// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_row_pair.h"

namespace ckernel::sfpu {

inline void square_init() {
    // The two-row body advances Dest from its second store (see sfpu/ckernel_sfpu_row_pair.h).
    _init_sfpu_row_pair_();
    math::reset_counters(p_setrwc::SET_ABD_F);

    sfpi::vConstIntPrgm0 = 1;
    sfpi::vConstIntPrgm1 = 0x7fff;
    sfpi::vConstIntPrgm2 = 0xffff0000;
}

sfpi_inline sfpi::vFloat float32_to_bf16_rne_prgm(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(in);
    sfpi::vUInt lsb = (bits >> 16) & sfpi::as<sfpi::vUInt>(sfpi::vConstIntPrgm0);
    bits = bits + sfpi::as<sfpi::vUInt>(sfpi::vConstIntPrgm1) + lsb;
    bits = bits & sfpi::as<sfpi::vUInt>(sfpi::vConstIntPrgm2);
    return sfpi::as<sfpi::vFloat>(bits);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_square() {
    static_assert(ITERATIONS % 2 == 0, "calculate_square() processes dest rows in pairs.");

#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat v0 = sfpi::dst_reg[0];
        sfpi::vFloat v1 = sfpi::dst_reg[1];
        sfpi::vFloat r0 = v0 * v0;
        sfpi::vFloat r1 = v1 * v1;
        if constexpr (!is_fp32_dest_acc_en) {
            r0 = float32_to_bf16_rne_prgm(r0);
            r1 = float32_to_bf16_rne_prgm(r1);
        }
        sfpi::dst_reg[0] = r0;
        _sfpu_row_pair_store_second_(r1);
        _sfpu_row_pair_end_();
    }
}

}  // namespace ckernel::sfpu
