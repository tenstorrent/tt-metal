// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_typecast.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint16() {
    _calculate_typecast_fp32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        // result = 0 (default for zero and subnormals)
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);
        // exponent = exexp(in); LaneEnabled = (|in| >= 1.0)
        // Disables zero/subnormal lanes (avoids SFPEXEXP quirk: biased_exp=0 → returns 1).
        // Negative lanes with |in| >= 1.0 stay enabled for the wrapping arithmetic below.
        TTI_SFPEXEXP(
            0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // mantissa = exman8(in)  [disabled lanes keep result = 0]
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        // shift_amount = exponent - 23
        TTI_SFPIADD(-23 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // mantissa >>= shift_amount  →  truncated integer part of |in|
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // LaneEnabled = in < 0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        // mantissa = ~mantissa + 1  (two's complement — wraps negative float to uint8)
        TTI_SFPIADD(
            0, p_sfpu::LCONST_0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        // mantissa += 256
        TTI_SFPIADD(256, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);
        // result += 256 (uint8 packer format requirement)
        TTI_SFPIADD(256, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // result &= 0xFF
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool u16 = false>
inline void calculate_typecast_uint_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        if constexpr (u16) {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_3, 0);
        } else {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        }
        TTI_SFPIADD(256, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp16b() {
    _calculate_typecast_uint16_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp16b() {
    _calculate_typecast_int32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_int32() {
    _calculate_typecast_fp32_to_int32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_fp16b() {
    _calculate_typecast_fp32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp32() {
    _calculate_typecast_uint16_to_fp32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp32() {
    _calculate_typecast_int32_to_fp32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint32() {
    _calculate_typecast_fp32_to_uint32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp16b() {
    _calculate_typecast_uint32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp32() {
    _calculate_typecast_uint32_to_fp32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_uint32() {
    _calculate_typecast_uint16_to_uint32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_uint16() {
    _calculate_typecast_uint32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_uint16() {
    _calculate_typecast_int32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_fp16b() {
    _init_typecast_fp32_to_fp16b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_uint32() {
    _init_typecast_uint16_to_uint32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_fp32() {
    _init_typecast_uint32_to_fp32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_fp32() {
    _init_typecast_int32_to_fp32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_fp32() {
    _init_typecast_uint16_to_fp32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_fp16b() {
    _init_typecast_uint16_to_fp16b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_fp16b() {
    _init_typecast_int32_to_fp16b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_fp16b() {
    _init_typecast_uint32_to_fp16b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint16() {
    _init_typecast_fp32_to_uint16_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint8() {
    sfpi::vConstIntPrgm0 = 0xFF;
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint_to_uint8() {
    sfpi::vConstIntPrgm0 = 0xFF;
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_uint16() {
    _init_typecast_uint32_to_uint16_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_uint16() {
    _init_typecast_int32_to_uint16_<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
