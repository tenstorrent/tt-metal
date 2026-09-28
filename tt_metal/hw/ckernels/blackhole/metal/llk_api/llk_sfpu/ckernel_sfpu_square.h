// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu {

inline void square_init() {
    // The paired store walks dest through ADDR_MOD_6, which advances by the two rows
    // the loop just wrote (one sfpi row is two dest counter steps), so the loop body
    // needs no separate increment.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);

    sfpi::vConstIntPrgm0 = 1;
    sfpi::vConstIntPrgm1 = 0x7fff;
}

// fp32 -> bf16 round-to-nearest-even ("bits + 0x7fff + lsb") for a value that is about to be
// stored to a 16-bit (Float16_b) Dest. The low 16 bits of the returned pattern are left
// unspecified rather than cleared: the 16-bit SFPSTORE keeps only the high half, so the
// stored bf16 is the same as with a trailing `& 0xffff0000`, one instruction per row cheaper.
// Do not use it for a value that stays in fp32 or that is read back before the store.
// (SFPSTOCHRND FP32_TO_FP16B "nearest" is not a substitute: on Blackhole silicon it rounds
// ties away from zero, maps NaN to +/-inf and flushes denormals.)
sfpi_inline sfpi::vFloat float32_to_bf16_rne_prgm(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(in);
    sfpi::vUInt lsb = (bits >> 16) & sfpi::as<sfpi::vUInt>(sfpi::vConstIntPrgm0);
    bits = bits + sfpi::as<sfpi::vUInt>(sfpi::vConstIntPrgm1) + lsb;
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
        sfpi::dst_reg[1].mode(ADDR_MOD_6) = r1;
    }
}

}  // namespace ckernel::sfpu
