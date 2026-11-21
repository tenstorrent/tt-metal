// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <array>
#include <cstdint>

namespace ckernel::sfpu {

// The kernel's instructions in a stall-filling order, constants held in LRegs.
inline void calculate_logit_bf16_scheduled() {
    TTI_SFPLOADI(7, 0, 0x3f00);
    TTI_SFPLOADI(6, 2, 0x5579);
    TTI_SFPLOADI(5, 2, 0xb5cc);
    TTI_SFPLOADI(6, 8, 0xbf78);
    TTI_SFPLOADI(5, 8, 0xbfec);
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 16; ++pass) {
        TTI_SFPLOAD(0, 0, 3, 0);
        TTI_SFPMAD(11, 7, 0, 2, 0);
        TTI_SFPMAD(0, 11, 10, 1, 0);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMOV(0, 0, 1, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPSETEXP(0x7f, 1, 4, 1);
        TTI_SFPEXEXP(0, 1, 0, 1);
        TTI_SFPADD(10, 4, 11, 4, 0);
        TTI_SFPADDI(0xbf00, 1, 0);
        TTI_SFPLOADI(2, 2, 0x7218);
        TTI_SFPIADD(0xf81, 0, 0, 1);
        TTI_SFPNOT(0, 0, 3, 0);
        TTI_SFPIADD(1, 3, 3, 5);
        TTI_SFPSETSGN(1, 3, 0, 1);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(6, 1, 5, 3, 0);
        TTI_SFPCAST(0, 0, 0);
        TTI_SFPLOADI(2, 8, 0xbf31);
        TTI_SFPMAD(1, 3, 2, 1, 0);
        TTI_SFPLOADI(2, 2, 0xa23c);
        TTI_SFPLOADI(3, 2, 0x1922);
        TTI_SFPLOADI(2, 8, 0x3de5);
        TTI_SFPLOADI(3, 8, 0xbec2);
        TTI_SFPMAD(2, 4, 3, 2, 0);
        TTI_SFPLOADI(3, 2, 0xc8a6);
        TTI_SFPLOADI(3, 8, 0x3f75);
        TTI_SFPMAD(2, 4, 3, 2, 0);
        TTI_SFPLOADI(3, 2, 0x7218);
        TTI_SFPMUL(4, 2, 9, 2, 0);
        TTI_SFPLOADI(3, 8, 0x3f31);
        TTI_SFPMAD(0, 3, 2, 0, 0);
        TTI_SFPLOAD(2, 0, 3, 0);
        TTI_SFPMAD(11, 7, 2, 3, 0);
        TTI_SFPMAD(0, 11, 1, 0, 0);
        TTI_SFPSETCC(0, 3, 0, 0);
        TTI_SFPMOV(0, 0, 0, 1);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPSETCC(0, 2, 0, 4);
        TTI_SFPMAD(11, 10, 2, 1, 0);
        TTI_SFPNOP;
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPCOMPC(0, 0, 0, 0);
        TTI_SFPLOADI(0, 0, 0x7f80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPSETSGN(0, 2, 2, 1);
        TTI_SFPLOADI(3, 0, 0x7e81);
        TTI_SFPLOAD(1, 6, 3, 0);
        TTI_SFPLOADI(4, 2, 0xff);
        TTI_SFPMAD(11, 3, 2, 2, 0);
        TTI_SFPAND(0, 4, 1, 0);
        TTI_SFPSETCC(0, 2, 0, 4);
        TTI_SFPIADD(0xf01, 1, 2, 5);
        TTI_SFPSETCC(0, 2, 0, 2);
        TTI_SFPLOADI(0, 0, 0xff80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPLOADI(0, 0, 0xff80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 3, 0);
        TTI_SFPLOADI(2, 2, 0x7218);
        TTI_SFPLOAD(0, 0, 3, 2);
        TTI_SFPMAD(11, 7, 0, 1, 0);
        TTI_SFPMAD(0, 11, 10, 3, 0);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPMOV(0, 0, 3, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPSETEXP(0x7f, 3, 4, 1);
        TTI_SFPEXEXP(0, 3, 0, 1);
        TTI_SFPADD(10, 4, 11, 4, 0);
        TTI_SFPIADD(0xf81, 0, 0, 1);
        TTI_SFPNOT(0, 0, 1, 0);
        TTI_SFPIADD(1, 1, 1, 5);
        TTI_SFPSETSGN(1, 1, 0, 1);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPADDI(0xbf00, 3, 0);
        TTI_SFPCAST(0, 0, 0);
        TTI_SFPMAD(6, 3, 5, 1, 0);
        TTI_SFPLOADI(2, 8, 0xbf31);
        TTI_SFPMAD(3, 1, 2, 1, 0);
        TTI_SFPLOADI(2, 2, 0xa23c);
        TTI_SFPLOADI(3, 2, 0x1922);
        TTI_SFPLOADI(2, 8, 0x3de5);
        TTI_SFPLOADI(3, 8, 0xbec2);
        TTI_SFPMAD(2, 4, 3, 2, 0);
        TTI_SFPLOADI(3, 2, 0xc8a6);
        TTI_SFPLOADI(3, 8, 0x3f75);
        TTI_SFPMAD(2, 4, 3, 2, 0);
        TTI_SFPLOADI(3, 2, 0x7218);
        TTI_SFPMUL(4, 2, 9, 2, 0);
        TTI_SFPLOADI(3, 8, 0x3f31);
        TTI_SFPMAD(0, 3, 2, 0, 0);
        TTI_SFPLOAD(2, 0, 3, 2);
        TTI_SFPMAD(11, 7, 2, 3, 0);
        TTI_SFPMAD(0, 11, 1, 0, 0);
        TTI_SFPSETCC(0, 3, 0, 0);
        TTI_SFPMOV(0, 0, 0, 1);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPSETCC(0, 2, 0, 4);
        TTI_SFPMAD(11, 10, 2, 1, 0);
        TTI_SFPNOP;
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPCOMPC(0, 0, 0, 0);
        TTI_SFPLOADI(0, 0, 0x7f80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPSETSGN(0, 2, 2, 1);
        TTI_SFPLOADI(3, 0, 0x7e81);
        TTI_SFPLOAD(1, 6, 3, 2);
        TTI_SFPLOADI(4, 2, 0xff);
        TTI_SFPMAD(11, 3, 2, 2, 0);
        TTI_SFPAND(0, 4, 1, 0);
        TTI_SFPSETCC(0, 2, 0, 4);
        TTI_SFPIADD(0xf01, 1, 2, 5);
        TTI_SFPSETCC(0, 2, 0, 2);
        TTI_SFPLOADI(0, 0, 0xff80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPLOADI(0, 0, 0xff80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 3, 2);
        TTI_INCRWC(0, 4, 0, 0);
    }
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_logit_bf16() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_logit_bf16_scheduled();
}

}  // namespace ckernel::sfpu
