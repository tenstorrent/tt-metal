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
inline void calculate_typecast_fp32_to_uint16(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    _calculate_typecast_fp32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint8(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_in * SFP_DST_TILE_ROWS);
        // result = 0 (default for zero, subnormals, and |in| < 1.0)
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);
        // exponent = exexp(in); LaneEnabled = |in| >= 1.0
        // (CC flags avoid SFPEXEXP quirk: zero/subnormal biased_exp=0 returns wrong value)
        TTI_SFPEXEXP(
            0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // mantissa = exman8(in)
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        // shift_amount = exponent - 23
        TTI_SFPIADD(-23 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // result = floor(|in|)
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // LaneEnabled = in < 0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        // result = -result  (two's complement negate)
        TTI_SFPIADD(
            0, p_sfpu::LCONST_0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);
        // result += 256 (packer format; for negatives: −|v|+256 gives correct uint8 wrap)
        TTI_SFPIADD(256, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // result &= 0xFF
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TT_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_2, dst_index_out * SFP_DST_TILE_ROWS);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool u16 = false>
inline void calculate_typecast_uint_to_uint8(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        if constexpr (u16) {
            TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_3, dst_index_in * SFP_DST_TILE_ROWS);
        } else {
            TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, dst_index_in * SFP_DST_TILE_ROWS);
        }
        TTI_SFPIADD(256, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_2, dst_index_out * SFP_DST_TILE_ROWS);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp16b(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    _calculate_typecast_uint16_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp16b(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    _calculate_typecast_int32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_int32(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    _calculate_typecast_fp32_to_int32_<APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_fp16b(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    _calculate_typecast_fp32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp32(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    _calculate_typecast_uint16_to_fp32_<APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp32(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    _calculate_typecast_int32_to_fp32_<APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint32(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    _calculate_typecast_fp32_to_uint32_<APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp16b(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    _calculate_typecast_uint32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp32(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    _calculate_typecast_uint32_to_fp32_<APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_uint32(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    _calculate_typecast_uint16_to_uint32_<APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_uint16(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    _calculate_typecast_uint32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_uint16(std::uint32_t dst_index_in, std::uint32_t dst_index_out) {
    _calculate_typecast_int32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out);
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
