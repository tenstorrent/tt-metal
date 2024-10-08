// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <climits>

inline __attribute__((always_inline)) unsigned int mulsi3 (unsigned int a, unsigned int b)
{
    unsigned int r = 0;

    #ifdef ARCH_GRAYSKULL
    while (a) {
        if (a & 1) { r += b; }
        a >>= 1;
        b <<= 1;
    }
    #else
        //Wormhole b0 has native multipliers
        r = a * b;
    #endif

    return r;
}

inline __attribute__((always_inline)) uint32_t fast_udiv_12(uint32_t n)
{
    // Uses embedding style magic number
    // * fixed point 1/12 then shifting.
    // https://web.archive.org/web/20190703172151/http://www.hackersdelight.org/magic.htm
    return (((uint64_t) n * 0xAAAAAAAB) >> 32) >> 3;
}

inline __attribute__((always_inline)) uint32_t fast_udiv_56(uint32_t n)
{
    // Uses embedding style magic number
    // * fixed point 1/12 then shifting.
    // https://web.archive.org/web/20190703172151/http://www.hackersdelight.org/magic.htm
    return (((uint64_t) n * 0x24924925) >> 32) >> 3;
}

inline __attribute__((always_inline)) uint32_t fast_udiv_70(uint32_t n)
{
    // Uses embedding style magic number
    // * fixed point 1/70 then shifting.
    // https://web.archive.org/web/20190703172151/http://www.hackersdelight.org/magic.htm
    return (((uint64_t) n * 0xEA0EA0EB) >> 32) >> 6;
}

inline __attribute__((always_inline)) uint32_t fast_udiv_80(uint32_t n)
{
    // Uses embedding style magic number
    // * fixed point 1/80 then shifting.
    // https://web.archive.org/web/20190703172151/http://www.hackersdelight.org/magic.htm
    return (((uint64_t) n * 0xCCCCCCCD) >> 32) >> 6;
}

inline __attribute__((always_inline)) uint32_t fast_udiv_94(uint32_t n)
{
    // Uses embedding style magic number
    // * fixed point 1/12 then shifting.
    // https://web.archive.org/web/20190703172151/http://www.hackersdelight.org/magic.htm
    return (((uint64_t) n * 0xAE4C415D) >> 32) >> 6;
}

inline __attribute__((always_inline)) uint32_t fast_udiv_124(uint32_t n)
{
    return (((uint64_t) n * 0x08421085) >> 32) >> 2;
}

inline __attribute__((always_inline)) uint32_t fast_udiv_130(uint32_t n)
{
    // Uses embedding style magic number
    // * fixed point 1/12 then shifting.
    // https://web.archive.org/web/20190703172151/http://www.hackersdelight.org/magic.htm
    return (((uint64_t) n * 0xFC0FC0FD) >> 32) >> 7;
}

inline __attribute__((always_inline)) uint32_t fast_udiv_140(uint32_t n)
{
    // Uses embedding style magic number
    // * fixed point 1/12 then shifting.
    // https://web.archive.org/web/20190703172151/http://www.hackersdelight.org/magic.htm
    return (((uint64_t) n * 0xEA0EA0EB) >> 32) >> 7;
}

template <uint32_t d>
inline __attribute__((always_inline)) uint32_t udivsi3_const_divisor(uint32_t n)
{
    if constexpr (d == 12) {
        // fast divide for 12 divisor
        return fast_udiv_12(n);
    } else if constexpr (d == 56) {
        // fast divide for 56 divisor. Handles Banked L1 address generation for N300
        return fast_udiv_56(n);
    } else if constexpr (d == 70) {
        return fast_udiv_70(n);
    } else if constexpr (d == 80) {
        return fast_udiv_80(n);
    } else if constexpr (d == 94) {
        // fast divide for 94 divisor. Handles Banked L1 address generation for E75
        return fast_udiv_94(n);
    } else if constexpr (d == 124) {
        return fast_udiv_124(n);
    } else if constexpr (d == 130) {
        return fast_udiv_130(n);
    } else if constexpr (d == 140) {
        return fast_udiv_140(n);
    } else {
        // generic divide from llvm
        const unsigned n_uword_bits = sizeof(uint32_t) * CHAR_BIT;
        unsigned int q;
        unsigned int r;
        unsigned sr;
        /* special cases */
        if (d == 0)
            return 0; /* ?! */
        if (n == 0)
            return 0;
        sr = __builtin_clz(d) - __builtin_clz(n);
        /* 0 <= sr <= n_uword_bits - 1 or sr large */
        if (sr > n_uword_bits - 1)  /* d > r */
            return 0;
        if (sr == n_uword_bits - 1)  /* d == 1 */
            return n;
        ++sr;
        /* 1 <= sr <= n_uword_bits - 1 */
        /* Not a special case */
        q = n << (n_uword_bits - sr);
        r = n >> sr;
        unsigned int  carry = 0;
        for (; sr > 0; --sr)
        {
            /* r:q = ((r:q)  << 1) | carry */
            r = (r << 1) | (q >> (n_uword_bits - 1));
            q = (q << 1) | carry;
            /* carry = 0;
             * if (r.all >= d.all)
             * {
             *      r.all -= d.all;
             *      carry = 1;
             * }
             */
            const int s = (unsigned int)(d - r - 1) >> (n_uword_bits - 1);
            carry = s & 1;
            r -= d & s;
        }
        q = (q << 1) | carry;
        return q;
    }
}
template <uint32_t d>
inline __attribute__((always_inline)) uint32_t umodsi3_const_divisor(uint32_t a)
{
    return a - udivsi3_const_divisor<d>(a) * d;
}
