// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <array>

namespace ckernel::sfpu {

// The kernel's instructions in a stall-filling order, constants held in LRegs.
inline void calculate_logsigmoid_bf16_scheduled() {
    TTI_SFPLOADI(7, 2, 0xaa3b);
    TTI_SFPLOADI(7, 8, 0xbfb8);
    TTI_SFPLOADI(6, 2, 0x40f8);
    TTI_SFPLOADI(5, 2, 0xf76);
    TTI_SFPLOADI(6, 8, 0xbd98);
    TTI_SFPLOADI(5, 8, 0x3e82);
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 16; ++pass) {
        TTI_SFPLOAD(2, 0, 3, 0);
        TTI_SFPSETSGN(0, 2, 0, 1);
        TTI_SFPLOADI(1, 0, 0x42b0);
        TTI_SFPSWAP(0, 1, 0, 1);
        TTI_SFPLOADI(3, 0, 0x42fe);
        TTI_SFPMAD(0, 7, 3, 0, 0);
        TTI_SFPLOADI(3, 2, 0xa410);
        TTI_SFPEXEXP(0, 0, 1, 0);
        TTI_SFPEXMAN(0, 0, 0, 0);
        TTI_SFPSHFT(0, 1, 0, 0);
        TTI_SFPEXMAN(0, 0, 1, 1);
        TTI_SFPLOADI(4, 2, 0x5ade);
        TTI_SFPCAST(1, 1, 0);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPLOADI(4, 8, 0x33a8);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x3884);
        TTI_SFPLOADI(4, 8, 0x3f80);
        TTI_SFPMAD(3, 1, 4, 1, 0);
        TTI_SFPEXEXP(0, 0, 0, 1);
        TTI_SFPSETEXP(0, 1, 0, 0);
        TTI_SFPLOADI(3, 2, 0xb38);
        TTI_SFPMAD(6, 0, 5, 1, 0);
        TTI_SFPLOADI(3, 8, 0xbef9);
        TTI_SFPLOADI(4, 2, 0xe63e);
        TTI_SFPMAD(1, 0, 3, 1, 0);
        TTI_SFPLOADI(4, 8, 0x3f7f);
        TTI_SFPMAD(1, 0, 4, 1, 0);
        TTI_SFPMOV(0, 2, 3, 2);
        TTI_SFPMUL(0, 1, 9, 0, 0);
        TTI_SFPLOADI(1, 0, 0x42ba);
        TTI_SFPSWAP(0, 9, 3, 1);
        TTI_SFPMAD(11, 1, 2, 1, 0);
        TTI_SFPMAD(0, 11, 3, 0, 0);
        TTI_SFPSETCC(0, 1, 0, 4);
        TTI_SFPMOV(0, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(1, 0, 0xc120);
        TTI_SFPMAD(11, 1, 2, 1, 0);
        TTI_SFPLOADI(3, 0, 0x7f7f);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPMOV(0, 2, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(11, 2, 3, 2, 0);
        TTI_SFPLOAD(1, 6, 3, 0);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMOV(0, 1, 2, 2);
        TTI_SFPMOV(0, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(3, 2, 0xff);
        TTI_SFPXOR(0, 3, 2, 0);
        TTI_SFPLOADI(3, 2, 0xff);
        TTI_SFPAND(0, 3, 2, 0);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPLOADI(2, 2, 0x7f00);
        TTI_SFPAND(0, 2, 1, 0);
        TTI_SFPSETCC(0, 1, 0, 2);
        TTI_SFPLOADI(0, 0, 0x7f80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 3, 0);
        TTI_SFPLOAD(2, 0, 3, 2);
        TTI_SFPSETSGN(0, 2, 0, 1);
        TTI_SFPLOADI(1, 0, 0x42b0);
        TTI_SFPSWAP(0, 1, 0, 1);
        TTI_SFPLOADI(3, 0, 0x42fe);
        TTI_SFPMAD(0, 7, 3, 0, 0);
        TTI_SFPLOADI(3, 2, 0xa410);
        TTI_SFPEXEXP(0, 0, 1, 0);
        TTI_SFPEXMAN(0, 0, 0, 0);
        TTI_SFPSHFT(0, 1, 0, 0);
        TTI_SFPEXMAN(0, 0, 1, 1);
        TTI_SFPLOADI(4, 2, 0x5ade);
        TTI_SFPCAST(1, 1, 0);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPLOADI(4, 8, 0x33a8);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x3884);
        TTI_SFPLOADI(4, 8, 0x3f80);
        TTI_SFPMAD(3, 1, 4, 1, 0);
        TTI_SFPEXEXP(0, 0, 0, 1);
        TTI_SFPSETEXP(0, 1, 0, 0);
        TTI_SFPLOADI(3, 2, 0xb38);
        TTI_SFPMAD(6, 0, 5, 1, 0);
        TTI_SFPLOADI(3, 8, 0xbef9);
        TTI_SFPLOADI(4, 2, 0xe63e);
        TTI_SFPMAD(1, 0, 3, 1, 0);
        TTI_SFPLOADI(4, 8, 0x3f7f);
        TTI_SFPMAD(1, 0, 4, 1, 0);
        TTI_SFPMOV(0, 2, 3, 2);
        TTI_SFPMUL(0, 1, 9, 0, 0);
        TTI_SFPLOADI(1, 0, 0x42ba);
        TTI_SFPSWAP(0, 9, 3, 1);
        TTI_SFPMAD(11, 1, 2, 1, 0);
        TTI_SFPMAD(0, 11, 3, 0, 0);
        TTI_SFPSETCC(0, 1, 0, 4);
        TTI_SFPMOV(0, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(1, 0, 0xc120);
        TTI_SFPMAD(11, 1, 2, 1, 0);
        TTI_SFPLOADI(3, 0, 0x7f7f);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPMOV(0, 2, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(11, 2, 3, 2, 0);
        TTI_SFPLOAD(1, 6, 3, 2);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMOV(0, 1, 2, 2);
        TTI_SFPMOV(0, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(3, 2, 0xff);
        TTI_SFPXOR(0, 3, 2, 0);
        TTI_SFPLOADI(3, 2, 0xff);
        TTI_SFPAND(0, 3, 2, 0);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPLOADI(2, 2, 0x7f00);
        TTI_SFPAND(0, 2, 1, 0);
        TTI_SFPSETCC(0, 1, 0, 2);
        TTI_SFPLOADI(0, 0, 0x7f80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 3, 2);
        TTI_INCRWC(0, 4, 0, 0);
    }
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_logsigmoid_bf16() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_logsigmoid_bf16_scheduled();
}
inline void init_logsigmoid_bf16() {}

}  // namespace ckernel::sfpu
