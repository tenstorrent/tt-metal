// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <int max_input_bits = 31>
inline void calculate_sfpu_gcd_body() {
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0); // c = a
    TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG2, 0); // c |= b

    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 0); // d = c
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // d = -d
    TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG3, 0); // d &= c (isolate LSB)
    TTI_SFPLZ(0, p_sfpu::LREG3, p_sfpu::LREG3, 0); // d = clz(d)

    // Ensure that b is odd: if LSB is zero, then swap with a.
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG2, SFPSHFT2_MOD1_SHFT_LREG); // c = b << d
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, 6); // if c == 0 then b is even
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 0); // swap(a, b)
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0); // a = abs(a)
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 0); // b = abs(b)

    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // a = -a
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // d = -d

    int iterations = max_input_bits - 1;

    #pragma GCC unroll 7
    while (iterations / 4 > 0) {
        TTI_REPLAY(0, 7 * 4, 0, 0);
        iterations -= 4;
    }

    // Replay 2 more iterations, making a total of 30 iterations.
    // The worst case for 31-bit inputs is 31 iterations, but we can skip the final iteration as it only affects a.
    // In addition, we can skip the final operation of the 30th iteration as it only affects a.
    TTI_REPLAY(0, 7 * iterations - 1, 0, 0);

    TTI_SFPENCC(0, 0, 0, 0);
}

template <int ITERATIONS = 8>
inline void calculate_sfpu_gcd(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Binary GCD algorithm.
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in the dest is 64 rows
        constexpr uint dst_tile_size = 64;

        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, 3, dst_index_in0 * dst_tile_size);  // a
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, 3, dst_index_in1 * dst_tile_size);  // b

        calculate_sfpu_gcd_body<31>();

        TT_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, 3, dst_index_out * dst_tile_size);
        dst_reg++;
    }
}

// A/B arms C and D: scope the gathering CSR bit to the replay-buffer LOAD only.
// The hazard the HW team acknowledges for gathering is replay-buffer load, and
// load_replay_buf()'s own bracket compiles out here (it is guarded by the
// ENABLE_GATHERING *build* macro while the hazard is a *runtime* CSR bit), and
// this site bypasses load_replay_buf with a raw TTI_REPLAY anyway.
// Both use the no-NOP CSR form so the Tensix instruction stream is unchanged.
#if defined(LCM_AB_NARROW_ENABLE_GATHERING)
// Mirror of disable_gathering<false>(), but CLEARING bit 18 (= gathering on).
// Same bit-1 serialise + fence the vendor helper uses, so the write lands before
// the replay record -- gathering is resolved early in the pipeline.
#define LCM_AB_GATHERING_ON()                                  \
    do {                                                       \
        asm("csrrs zero, 0x7c0, %0" : : "r"(1 << 1));          \
        asm("fence");                                          \
        asm("csrrc zero, 0x7c0, %0" : : "r"(1 << 18));         \
        asm("csrrc zero, 0x7c0, %0" : : "r"(1 << 1));          \
        asm("fence");                                          \
    } while (0)
#endif

inline void calculate_sfpu_gcd_init() {
#if defined(LCM_AB_NARROW_DISABLE_GATHERING)
    ckernel::disable_gathering<false>();
#elif defined(LCM_AB_NARROW_ENABLE_GATHERING)
    LCM_AB_GATHERING_ON();
#endif
    TTI_REPLAY(0, 7 * 4, 0, 1);
    #pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        // We store {-a, a} in {LREG0, LREG2}, which is convenient for isolating the LSB of a.
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG2, 0); // LREG2 = +a
        TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG0, 0); // LREG0 &= a (isolate LSB and overwrite -a)
        TTI_SFPLZ(0, p_sfpu::LREG0, p_sfpu::LREG0, SFPLZ_MOD1_CC_NE0); // LREG0 = clz(LREG0), disable lanes where a == 0
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE); // LREG0 += d
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG0, SFPSHFT2_MOD1_SHFT_LREG); // LREG0 = a >> -LREG0, making a definitely odd (now both a and b are odd)
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, SFPSWAP_MOD1_VEC_MIN_MAX); // ensure b < a
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // a = b - a (now a is even)
    }
#if defined(LCM_AB_NARROW_DISABLE_GATHERING)
    ckernel::enable_gathering();
#elif defined(LCM_AB_NARROW_ENABLE_GATHERING)
    ckernel::disable_gathering<false>();
#endif
}

}  // namespace sfpu
}  // namespace ckernel
