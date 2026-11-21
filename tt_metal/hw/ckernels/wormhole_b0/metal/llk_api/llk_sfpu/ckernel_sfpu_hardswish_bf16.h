// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

namespace ckernel::sfpu {

// The kernel's instructions in a stall-filling order, constants held in LRegs.
inline void calculate_hardswish_bf16_scheduled() {
    TTI_SFPLOADI(7, 8, 0x3e2a);
    TTI_SFPLOADI(7, 0xa, 0xaaab);
    TTI_SFPLOADI(6, 2, 0x80ff);
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 16; ++pass) {
        TTI_SFPLOAD(0, 0, 3, 0);
        TTI_SFPMUL(0, 7, 9, 2, 0);
        TTI_SFPSETSGN(0, 0, 4, 1);
        TTI_SFPADDI(0x3f00, 2, 0);
        TTI_SFPDIVP2(0x7e, 4, 4, 1);
        TTI_SFPSWAP(0, 9, 2, 9);
        TTI_SFPSWAP(0, 10, 2, 1);
        TTI_SFPADDI(0xbfff, 4, 0);
        TTI_SFPMUL(0, 2, 9, 2, 0);
        TTI_SFPLOAD(1, 6, 3, 0);
        TTI_SFPSETCC(0, 4, 0, 6);
        TTI_SFPSETMAN(0, 0, 2, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPAND(0, 6, 1, 0);
        TTI_SFPXOR(0, 6, 1, 0);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPLOADI(2, 0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 2, 2, 1);
        TTI_SFPSTORE(2, 0, 3, 0);
        TTI_SFPLOAD(0, 0, 3, 2);
        TTI_SFPMUL(0, 7, 9, 2, 0);
        TTI_SFPSETSGN(0, 0, 4, 1);
        TTI_SFPADDI(0x3f00, 2, 0);
        TTI_SFPDIVP2(0x7e, 4, 4, 1);
        TTI_SFPSWAP(0, 9, 2, 9);
        TTI_SFPSWAP(0, 10, 2, 1);
        TTI_SFPADDI(0xbfff, 4, 0);
        TTI_SFPMUL(0, 2, 9, 2, 0);
        TTI_SFPLOAD(1, 6, 3, 2);
        TTI_SFPSETCC(0, 4, 0, 6);
        TTI_SFPSETMAN(0, 0, 2, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPAND(0, 6, 1, 0);
        TTI_SFPXOR(0, 6, 1, 0);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPLOADI(2, 0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 2, 2, 1);
        TTI_SFPSTORE(2, 0, 3, 2);
        TTI_INCRWC(0, 4, 0, 0);
    }
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_hardswish_bf16() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_hardswish_bf16_scheduled();
}
inline void init_hardswish_bf16() {}

}  // namespace ckernel::sfpu
