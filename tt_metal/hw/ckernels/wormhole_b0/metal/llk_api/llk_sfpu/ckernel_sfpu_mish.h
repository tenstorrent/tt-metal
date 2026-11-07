// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

/**
 * Mish activation function:  mish(x) = x * tanh(softplus(x))
 *
 * Reducing to algebraic identity:
 *
 *     tanh(softplus(x)) = u (u + 2) / (u^2 + 2u + 2),  where u = exp(x)
 *
 * so that mish becomes:
 *
 *     mish(x) = x * u (u + 2) / (u^2 + 2u + 2)
 *
 * Note: BH uses rearranged form (x - 2x/denom) for x >= 0 in order to avoid
 * cancellation through its lower-precision approx_recip.
 *
 * Saturation: For x >= 8.0, mish(x) is approximated as x.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_mish() {
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        static_assert(ITERATIONS % 2 == 0, "calculate_mish processes rows in pairs");
        TTI_REPLAY(0, 16, 1, 1);
        TTI_SFPLOADI(6, 0, 16640);
        TTI_SFPLOAD(4, 0, 3, 0);
        TTI_SFPLOADI(3, 2, 43579);
        TTI_SFPLOADI(1, 0, 17150);
        TTI_SFPLOADI(3, 8, 16312);
        TTI_SFPSWAP(0, 6, 4, 1);
        TTI_SFPMAD(4, 3, 1, 0, 0);
        TTI_SFPLOADI(7, 0, 16640);
        TTI_SFPSWAP(0, 0, 9, 1);
        TTI_SFPEXEXP(0, 0, 4, 0);
        TTI_SFPLOAD(6, 0, 3, 2);
        TTI_SFPSWAP(0, 7, 6, 1);
        TTI_SFPEXMAN(0, 0, 2, 0);
        TTI_SFPSHFT(0, 4, 2, 0);
        TTI_SFPLOADI(0, 2, 42008);
        TTI_SFPLOADI(0, 8, 10156);
        TTI_SFPLOADI(5, 2, 23258);
        TTI_SFPMAD(6, 3, 1, 6, 0);
        TTI_SFPEXMAN(0, 2, 1, 1);
        TTI_SFPCAST(1, 7, 0);
        TTI_SFPLOADI(5, 8, 13224);
        TTI_SFPSWAP(0, 6, 9, 1);
        TTI_SFPMAD(7, 0, 5, 3, 0);
        TTI_SFPLOADI(1, 2, 14469);
        TTI_SFPEXMAN(0, 6, 4, 0);
        TTI_SFPEXEXP(0, 6, 6, 0);
        TTI_SFPLOADI(1, 8, 16256);
        TTI_SFPSHFT(0, 6, 4, 0);
        TTI_SFPMAD(3, 7, 1, 3, 0);
        TTI_SFPEXMAN(0, 4, 6, 1);
        TTI_SFPCAST(6, 7, 0);
        TTI_SFPSETEXP(0, 3, 2, 2);
        TTI_SFP_STOCH_RND(0, 0, 9, 2, 6, 1);
        TTI_SFPMAD(7, 0, 5, 2, 0);
        TTI_SFPADD(10, 6, 10, 0, 0);
        TTI_SFPMAD(2, 7, 1, 5, 0);
        TTI_SFPMAD(0, 0, 10, 2, 0);
        TTI_SFPSETEXP(0, 5, 4, 2);
        TTI_SFP_STOCH_RND(0, 0, 9, 4, 1, 1);
        TTI_SFPADD(10, 1, 10, 5, 0);
        TTI_SFPMOV(0, 2, 3, 2);
        TTI_SFPSETMAN(0, 11, 3, 0);
        TTI_SFPMAD(5, 5, 10, 0, 0);
        TTI_SFPMAD(12, 3, 13, 5, 0);
        TTI_SFPNOT(0, 2, 4, 0);
        TTI_SFPMAD(5, 3, 14, 5, 0);
        TTI_SFPMOV(0, 0, 7, 2);
        TTI_SFPSETMAN(0, 11, 7, 0);
        TTI_SFPSETMAN(0, 4, 4, 1);
        TTI_SFPMAD(3, 5, 10, 3, 0);
        TTI_SFPMULI(16128, 4, 0);
        TTI_SFPMAD(5, 3, 5, 5, 0);
        TTI_SFPMAD(12, 7, 13, 3, 0);
        TTI_SFPMUL(5, 4, 9, 4, 0);
        TTI_SFPMOV(0, 6, 5, 2);
        TTI_SFPMAD(3, 7, 14, 3, 0);
        TTI_SFPADDI(16384, 5, 0);
        TTI_SFPMAD(7, 3, 10, 7, 0);
        TTI_SFPMUL(6, 5, 9, 6, 0);
        TTI_SFPNOT(0, 0, 5, 0);
        TTI_SFPSETSGN(0, 4, 2, 0);
        TTI_SFPSETMAN(0, 5, 5, 1);
        TTI_SFPMAD(3, 7, 3, 4, 0);
        TTI_SFPMUL(6, 2, 9, 2, 0);
        TTI_SFPLOAD(6, 0, 3, 0);
        TTI_SFPMOV(0, 1, 3, 2);
        TTI_SFPMULI(16128, 5, 0);
        TTI_SFPADDI(16384, 3, 0);
        TTI_SFPMUL(6, 2, 9, 7, 0);
        TTI_SFPMUL(4, 5, 9, 5, 0);
        TTI_SFPMUL(1, 3, 9, 1, 0);
        TTI_SFP_STOCH_RND(0, 0, 9, 7, 2, 1);
        TTI_SFPSETSGN(0, 5, 0, 0);
        TTI_SFPSTORE(2, 0, 3, 0);
        TTI_SFPMUL(1, 0, 9, 7, 0);
        TTI_SFPLOAD(1, 0, 3, 2);
        TTI_SFPMUL(1, 7, 9, 3, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 9, 3, 7, 1);
        TTI_SFPSTORE(7, 0, 3, 2);
        TTI_INCRWC(0, 4, 0, 0);
#pragma GCC unroll 1
        for (int d = 2; d < ITERATIONS; d += 2) {
            TTI_REPLAY(0, 16, 0, 0);
            TTI_SFPLOADI(5, 2, 23258);
            TTI_SFPMAD(6, 3, 1, 6, 0);
            TTI_SFPEXMAN(0, 2, 1, 1);
            TTI_SFPCAST(1, 7, 0);
            TTI_SFPLOADI(5, 8, 13224);
            TTI_SFPSWAP(0, 6, 9, 1);
            TTI_SFPMAD(7, 0, 5, 3, 0);
            TTI_SFPLOADI(1, 2, 14469);
            TTI_SFPEXMAN(0, 6, 4, 0);
            TTI_SFPEXEXP(0, 6, 6, 0);
            TTI_SFPLOADI(1, 8, 16256);
            TTI_SFPSHFT(0, 6, 4, 0);
            TTI_SFPMAD(3, 7, 1, 3, 0);
            TTI_SFPEXMAN(0, 4, 6, 1);
            TTI_SFPCAST(6, 7, 0);
            TTI_SFPSETEXP(0, 3, 2, 2);
            TTI_SFP_STOCH_RND(0, 0, 9, 2, 6, 1);
            TTI_SFPMAD(7, 0, 5, 2, 0);
            TTI_SFPADD(10, 6, 10, 0, 0);
            TTI_SFPMAD(2, 7, 1, 5, 0);
            TTI_SFPMAD(0, 0, 10, 2, 0);
            TTI_SFPSETEXP(0, 5, 4, 2);
            TTI_SFP_STOCH_RND(0, 0, 9, 4, 1, 1);
            TTI_SFPADD(10, 1, 10, 5, 0);
            TTI_SFPMOV(0, 2, 3, 2);
            TTI_SFPSETMAN(0, 11, 3, 0);
            TTI_SFPMAD(5, 5, 10, 0, 0);
            TTI_SFPMAD(12, 3, 13, 5, 0);
            TTI_SFPNOT(0, 2, 4, 0);
            TTI_SFPMAD(5, 3, 14, 5, 0);
            TTI_SFPMOV(0, 0, 7, 2);
            TTI_SFPSETMAN(0, 11, 7, 0);
            TTI_SFPSETMAN(0, 4, 4, 1);
            TTI_SFPMAD(3, 5, 10, 3, 0);
            TTI_SFPMULI(16128, 4, 0);
            TTI_SFPMAD(5, 3, 5, 5, 0);
            TTI_SFPMAD(12, 7, 13, 3, 0);
            TTI_SFPMUL(5, 4, 9, 4, 0);
            TTI_SFPMOV(0, 6, 5, 2);
            TTI_SFPMAD(3, 7, 14, 3, 0);
            TTI_SFPADDI(16384, 5, 0);
            TTI_SFPMAD(7, 3, 10, 7, 0);
            TTI_SFPMUL(6, 5, 9, 6, 0);
            TTI_SFPNOT(0, 0, 5, 0);
            TTI_SFPSETSGN(0, 4, 2, 0);
            TTI_SFPSETMAN(0, 5, 5, 1);
            TTI_SFPMAD(3, 7, 3, 4, 0);
            TTI_SFPMUL(6, 2, 9, 2, 0);
            TTI_SFPLOAD(6, 0, 3, 0);
            TTI_SFPMOV(0, 1, 3, 2);
            TTI_SFPMULI(16128, 5, 0);
            TTI_SFPADDI(16384, 3, 0);
            TTI_SFPMUL(6, 2, 9, 7, 0);
            TTI_SFPMUL(4, 5, 9, 5, 0);
            TTI_SFPMUL(1, 3, 9, 1, 0);
            TTI_SFP_STOCH_RND(0, 0, 9, 7, 2, 1);
            TTI_SFPSETSGN(0, 5, 0, 0);
            TTI_SFPSTORE(2, 0, 3, 0);
            TTI_SFPMUL(1, 0, 9, 7, 0);
            TTI_SFPLOAD(1, 0, 3, 2);
            TTI_SFPMUL(1, 7, 9, 3, 0);
            TTI_SFPNOP;
            TTI_SFP_STOCH_RND(0, 0, 9, 3, 7, 1);
            TTI_SFPSTORE(7, 0, 3, 2);
            TTI_INCRWC(0, 4, 0, 0);
        }
        return;
    }
    if constexpr (!APPROXIMATION_MODE && is_fp32_dest_acc_en) {
        static_assert(ITERATIONS % 2 == 0, "calculate_mish processes rows in pairs");
        TTI_REPLAY(0, 16, 1, 1);
        TTI_SFPLOAD(3, 0, 3, 0);
        TTI_SFPLOAD(2, 0, 3, 2);
        TTI_SFPLOADI(0, 2, 43579);
        TTI_SFPLOADI(0, 8, 16312);
        TTI_SFPMUL(0, 3, 9, 1, 0);
        TTI_SFPMUL(0, 2, 9, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 9, 1, 1, 7);
        TTI_SFPCAST(1, 4, 0);
        TTI_SFPABS(0, 1, 1, 0);
        TTI_SFP_STOCH_RND(0, 0, 9, 0, 0, 7);
        TTI_SFPCAST(0, 5, 0);
        TTI_SFPABS(0, 0, 0, 0);
        TTI_SFPLOADI(6, 2, 29184);
        TTI_SFPLOADI(6, 8, 48945);
        TTI_SFPMAD(4, 6, 3, 3, 0);
        TTI_SFPMAD(5, 6, 2, 6, 0);
        TTI_SFPLOADI(2, 2, 48782);
        TTI_SFPLOADI(2, 8, 46527);
        TTI_SFPMAD(4, 2, 3, 3, 0);
        TTI_SFPMAD(5, 2, 6, 2, 0);
        TTI_SFPSETSGN(0, 1, 4, 0);
        TTI_SFPSETSGN(0, 0, 5, 0);
        TTI_SFPLOADI(6, 1, 5541);
        TTI_SFPLOADI(0, 2, 12142);
        TTI_SFPLOADI(0, 8, 15369);
        TTI_SFPMAD(6, 3, 0, 1, 0);
        TTI_SFPMAD(6, 2, 0, 6, 0);
        TTI_SFPLOADI(0, 2, 44461);
        TTI_SFPLOADI(0, 8, 15658);
        TTI_SFPMAD(1, 3, 0, 1, 0);
        TTI_SFPMAD(6, 2, 0, 6, 0);
        TTI_SFPLOADI(0, 2, 43560);
        TTI_SFPLOADI(0, 8, 15914);
        TTI_SFPMAD(1, 3, 0, 1, 0);
        TTI_SFPMAD(6, 2, 0, 0, 0);
        TTI_SFPLOADI(6, 2, 65531);
        TTI_SFPLOADI(6, 8, 16127);
        TTI_SFPMAD(1, 3, 6, 1, 0);
        TTI_SFPMAD(0, 2, 6, 0, 0);
        TTI_SFPMAD(1, 3, 10, 1, 0);
        TTI_SFPMAD(0, 2, 10, 0, 0);
        TTI_SFPMAD(1, 3, 10, 3, 0);
        TTI_SFPMAD(0, 2, 10, 2, 0);
        TTI_SFPIADD(4095, 1, 1, 5);
        TTI_SFPMUL(1, 9, 9, 1, 0);
        TTI_SFPIADD(4095, 0, 0, 5);
        TTI_SFPMUL(0, 9, 9, 0, 0);
        TTI_SFPEXEXP(0, 3, 6, 1);
        TTI_SFPIADD(0, 4, 6, 4);
        TTI_SFPIADD(4095, 6, 4, 9);
        TTI_SFPMOV(0, 6, 1, 0);
        TTI_SFPSETEXP(0, 3, 1, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPEXEXP(0, 2, 3, 1);
        TTI_SFPIADD(0, 5, 3, 4);
        TTI_SFPIADD(4095, 3, 4, 9);
        TTI_SFPMOV(0, 3, 0, 0);
        TTI_SFPSETEXP(0, 2, 0, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPADD(10, 1, 10, 2, 0);
        TTI_SFPMOV(0, 1, 6, 2);
        TTI_SFPMAD(2, 2, 10, 5, 0);
        TTI_SFPADDI(16384, 6, 0);
        TTI_SFPMOV(0, 5, 4, 2);
        TTI_SFPSETMAN(0, 11, 4, 0);
        TTI_SFPMAD(12, 4, 13, 2, 0);
        TTI_SFPNOT(0, 5, 7, 0);
        TTI_SFPMAD(2, 4, 14, 2, 0);
        TTI_SFPSETMAN(0, 7, 7, 1);
        TTI_SFPMAD(4, 2, 10, 3, 0);
        TTI_SFPMULI(16128, 7, 0);
        TTI_SFPMAD(2, 3, 2, 2, 0);
        TTI_SFPMUL(1, 6, 9, 6, 0);
        TTI_SFPMAD(4, 2, 10, 4, 0);
        TTI_SFPLOADI(1, 0, 16640);
        TTI_SFPMAD(2, 4, 2, 4, 0);
        TTI_SFPLOAD(2, 0, 3, 0);
        TTI_SFPMUL(4, 7, 9, 4, 0);
        TTI_SFPMAD(11, 1, 2, 1, 0);
        TTI_SFPSETSGN(0, 4, 5, 0);
        TTI_SFPMUL(6, 5, 9, 4, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(2, 4, 9, 3, 0);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPSTORE(3, 0, 3, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPADD(10, 0, 10, 1, 0);
        TTI_SFPMOV(0, 0, 5, 2);
        TTI_SFPMAD(1, 1, 10, 1, 0);
        TTI_SFPADDI(16384, 5, 0);
        TTI_SFPMOV(0, 1, 7, 2);
        TTI_SFPSETMAN(0, 11, 7, 0);
        TTI_SFPMAD(12, 7, 13, 6, 0);
        TTI_SFPNOT(0, 1, 4, 0);
        TTI_SFPMAD(6, 7, 14, 6, 0);
        TTI_SFPSETMAN(0, 4, 4, 1);
        TTI_SFPMAD(7, 6, 10, 3, 0);
        TTI_SFPMULI(16128, 4, 0);
        TTI_SFPMAD(6, 3, 6, 6, 0);
        TTI_SFPMUL(0, 5, 9, 5, 0);
        TTI_SFPMAD(7, 6, 10, 3, 0);
        TTI_SFPLOAD(7, 0, 3, 2);
        TTI_SFPMAD(6, 3, 6, 3, 0);
        TTI_SFPLOADI(6, 0, 16640);
        TTI_SFPMUL(3, 4, 9, 3, 0);
        TTI_SFPMAD(11, 6, 7, 0, 0);
        TTI_SFPSETSGN(0, 3, 1, 0);
        TTI_SFPMUL(5, 1, 9, 1, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(7, 1, 9, 2, 0);
        TTI_SFPSETCC(0, 0, 0, 0);
        TTI_SFPSTORE(2, 0, 3, 2);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_INCRWC(0, 4, 0, 0);
#pragma GCC unroll 1
        for (int d = 2; d < ITERATIONS; d += 2) {
            TTI_REPLAY(0, 16, 0, 0);
            TTI_SFPLOADI(2, 2, 48782);
            TTI_SFPLOADI(2, 8, 46527);
            TTI_SFPMAD(4, 2, 3, 3, 0);
            TTI_SFPMAD(5, 2, 6, 2, 0);
            TTI_SFPSETSGN(0, 1, 4, 0);
            TTI_SFPSETSGN(0, 0, 5, 0);
            TTI_SFPLOADI(6, 1, 5541);
            TTI_SFPLOADI(0, 2, 12142);
            TTI_SFPLOADI(0, 8, 15369);
            TTI_SFPMAD(6, 3, 0, 1, 0);
            TTI_SFPMAD(6, 2, 0, 6, 0);
            TTI_SFPLOADI(0, 2, 44461);
            TTI_SFPLOADI(0, 8, 15658);
            TTI_SFPMAD(1, 3, 0, 1, 0);
            TTI_SFPMAD(6, 2, 0, 6, 0);
            TTI_SFPLOADI(0, 2, 43560);
            TTI_SFPLOADI(0, 8, 15914);
            TTI_SFPMAD(1, 3, 0, 1, 0);
            TTI_SFPMAD(6, 2, 0, 0, 0);
            TTI_SFPLOADI(6, 2, 65531);
            TTI_SFPLOADI(6, 8, 16127);
            TTI_SFPMAD(1, 3, 6, 1, 0);
            TTI_SFPMAD(0, 2, 6, 0, 0);
            TTI_SFPMAD(1, 3, 10, 1, 0);
            TTI_SFPMAD(0, 2, 10, 0, 0);
            TTI_SFPMAD(1, 3, 10, 3, 0);
            TTI_SFPMAD(0, 2, 10, 2, 0);
            TTI_SFPIADD(4095, 1, 1, 5);
            TTI_SFPMUL(1, 9, 9, 1, 0);
            TTI_SFPIADD(4095, 0, 0, 5);
            TTI_SFPMUL(0, 9, 9, 0, 0);
            TTI_SFPEXEXP(0, 3, 6, 1);
            TTI_SFPIADD(0, 4, 6, 4);
            TTI_SFPIADD(4095, 6, 4, 9);
            TTI_SFPMOV(0, 6, 1, 0);
            TTI_SFPSETEXP(0, 3, 1, 0);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPEXEXP(0, 2, 3, 1);
            TTI_SFPIADD(0, 5, 3, 4);
            TTI_SFPIADD(4095, 3, 4, 9);
            TTI_SFPMOV(0, 3, 0, 0);
            TTI_SFPSETEXP(0, 2, 0, 0);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPADD(10, 1, 10, 2, 0);
            TTI_SFPMOV(0, 1, 6, 2);
            TTI_SFPMAD(2, 2, 10, 5, 0);
            TTI_SFPADDI(16384, 6, 0);
            TTI_SFPMOV(0, 5, 4, 2);
            TTI_SFPSETMAN(0, 11, 4, 0);
            TTI_SFPMAD(12, 4, 13, 2, 0);
            TTI_SFPNOT(0, 5, 7, 0);
            TTI_SFPMAD(2, 4, 14, 2, 0);
            TTI_SFPSETMAN(0, 7, 7, 1);
            TTI_SFPMAD(4, 2, 10, 3, 0);
            TTI_SFPMULI(16128, 7, 0);
            TTI_SFPMAD(2, 3, 2, 2, 0);
            TTI_SFPMUL(1, 6, 9, 6, 0);
            TTI_SFPMAD(4, 2, 10, 4, 0);
            TTI_SFPLOADI(1, 0, 16640);
            TTI_SFPMAD(2, 4, 2, 4, 0);
            TTI_SFPLOAD(2, 0, 3, 0);
            TTI_SFPMUL(4, 7, 9, 4, 0);
            TTI_SFPMAD(11, 1, 2, 1, 0);
            TTI_SFPSETSGN(0, 4, 5, 0);
            TTI_SFPMUL(6, 5, 9, 4, 0);
            TTI_SFPNOP;
            TTI_SFPMUL(2, 4, 9, 3, 0);
            TTI_SFPSETCC(0, 1, 0, 0);
            TTI_SFPSTORE(3, 0, 3, 0);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPADD(10, 0, 10, 1, 0);
            TTI_SFPMOV(0, 0, 5, 2);
            TTI_SFPMAD(1, 1, 10, 1, 0);
            TTI_SFPADDI(16384, 5, 0);
            TTI_SFPMOV(0, 1, 7, 2);
            TTI_SFPSETMAN(0, 11, 7, 0);
            TTI_SFPMAD(12, 7, 13, 6, 0);
            TTI_SFPNOT(0, 1, 4, 0);
            TTI_SFPMAD(6, 7, 14, 6, 0);
            TTI_SFPSETMAN(0, 4, 4, 1);
            TTI_SFPMAD(7, 6, 10, 3, 0);
            TTI_SFPMULI(16128, 4, 0);
            TTI_SFPMAD(6, 3, 6, 6, 0);
            TTI_SFPMUL(0, 5, 9, 5, 0);
            TTI_SFPMAD(7, 6, 10, 3, 0);
            TTI_SFPLOAD(7, 0, 3, 2);
            TTI_SFPMAD(6, 3, 6, 3, 0);
            TTI_SFPLOADI(6, 0, 16640);
            TTI_SFPMUL(3, 4, 9, 3, 0);
            TTI_SFPMAD(11, 6, 7, 0, 0);
            TTI_SFPSETSGN(0, 3, 1, 0);
            TTI_SFPMUL(5, 1, 9, 1, 0);
            TTI_SFPNOP;
            TTI_SFPMUL(7, 1, 9, 2, 0);
            TTI_SFPSETCC(0, 0, 0, 0);
            TTI_SFPSTORE(2, 0, 3, 2);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_INCRWC(0, 4, 0, 0);
        }
        return;
    }
    constexpr float SAT_HI = 8.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // x >= SAT_HI: mish(x) ≈ x
        sfpi::vFloat result = x;

        v_if(x < SAT_HI) {
            sfpi::vFloat u = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(x);

            // numerator = u * (u + 2)
            sfpi::vFloat numer = u * (u + 2.0f);

            // denominator = (1 + u)^2 + 1 = u^2 + 2u + 2
            sfpi::vFloat one_plus_u = u + 1.0f;
            sfpi::vFloat denom = one_plus_u * one_plus_u + 1.0f;

            sfpi::vFloat inv_denom = sfpu_reciprocal_iter<0>(denom);

            result = x * (numer * inv_denom);
        }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void mish_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // exp does not need an init
    recip_init<APPROXIMATION_MODE, false>();
}

}  // namespace ckernel::sfpu
