// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"

using namespace sfpi;

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE>
inline void rand_init(uint32_t seed) {
    math::reset_counters(p_setrwc::SET_ABD_F);
    
    // Avoid seed lock state
    if (seed == 0xffffffff) {
        seed = 0x12345678;
    }
    
    // Incorporate physical core coordinates to separate streams per core
    uint32_t core_id = (static_cast<uint32_t>(my_y[0]) << 6) | static_cast<uint32_t>(my_x[0]);
    uint32_t mixed_seed = seed ^ (core_id * 2654435761U);
    
    // Seed decorrelation using MurmurHash3 32-bit finalizer
    uint32_t h = mixed_seed;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    
    if (h == 0xffffffff) {
        h = 0x12345678;
    }
    
    init_prng_seed(h);
}

template <bool APPROXIMATION_MODE>
inline void rand(uint32_t from, uint32_t scale) {
    // Load scale param to lreg1
    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);

    // Load from param to lreg2
    TT_SFPLOADI(p_sfpu::LREG2, 10, from & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, from >> 16);

    // Load constant 0xFFFF to lreg4 for bitwise AND masking
    TT_SFPLOADI(p_sfpu::LREG4, 10, 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG4, 8, 0x0000);

    // Load scale factor constant 1.52587890625e-5f (2^-16) to lreg6.
    // 1.52587890625e-5f in IEEE-754 binary representation is 0x37800000.
    TT_SFPLOADI(p_sfpu::LREG6, 10, 0x0000);
    TT_SFPLOADI(p_sfpu::LREG6, 8, 0x3780);

#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        // 1. Get 32-bit random integer from hardware PRNG register 9 to lreg0
        TTI_SFPMOV(0, 9, p_sfpu::LREG0, 8);

        // 2. Thomas Wang 32-bit integer hash to break LFSR linearity and decorrelate lanes
        // key = ~key
        TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        // tmp (lreg3) = key << 15
        TTI_SFPSHFT(15, p_sfpu::LREG0, p_sfpu::LREG3, 5);
        // key = key + tmp
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);

        // tmp (lreg3) = key >> 12
        TTI_SFPSHFT((-12) & 0xFFF, p_sfpu::LREG0, p_sfpu::LREG3, 5);
        // key = key ^ tmp
        TTI_SFPXOR(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);

        // tmp (lreg3) = key << 2
        TTI_SFPSHFT(2, p_sfpu::LREG0, p_sfpu::LREG3, 5);
        // key = key + tmp
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);

        // tmp (lreg3) = key >> 4
        TTI_SFPSHFT((-4) & 0xFFF, p_sfpu::LREG0, p_sfpu::LREG3, 5);
        // key = key ^ tmp
        TTI_SFPXOR(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);

        // tmp (lreg3) = key << 3
        TTI_SFPSHFT(3, p_sfpu::LREG0, p_sfpu::LREG3, 5);
        // tmp2 (lreg5) = key << 11
        TTI_SFPSHFT(11, p_sfpu::LREG0, p_sfpu::LREG5, 5);
        // key = key + tmp
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);
        // key = key + tmp2
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG0, 0);

        // tmp (lreg3) = key >> 16
        TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG0, p_sfpu::LREG3, 5);
        // key = key ^ tmp
        TTI_SFPXOR(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);

        // 3. Convert 32-bit random int to high-precision float using two 16-bit splits
        // h_high (lreg3) = key >> 16
        TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG0, p_sfpu::LREG3, 5);
        // Copy key to lreg5
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG5, 0);
        // h_low (lreg5) = key & 0xFFFF
        TTI_SFPAND(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);

        // Cast h_high (lreg3) and h_low (lreg5) to floats
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);
        TTI_SFPCAST(p_sfpu::LREG5, p_sfpu::LREG5, 0);

        // Reconstruct float in [0, 1): rand_float (lreg5) = (h_low * 2^-16 + h_high) * 2^-16
        // lreg5 = lreg5 * lreg6 + lreg3 (f_low * 2^-16 + f_high)
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG3, p_sfpu::LREG5, 0);
        // lreg5 = lreg5 * lreg6 + 0.0 (rand_float in [0, 1))
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);

        // 4. Scale and shift to target range: rand_float = rand_float * scale + from
        // lreg5 = lreg5 * lreg1 + lreg2
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG5, 0);

        // Store result in DST register
        TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
        dst_reg++;
    }
}
}  // namespace ckernel::sfpu
