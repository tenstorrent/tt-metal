// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu/ckernel_sfpu_is_fp16_zero.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "sfpu/ckernel_sfpu_comp.h"
#include "llk_math_eltwise_sfpu_op.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, ZeroCompMode COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp() {
    constexpr std::uint32_t V = p_sfpu::LREG0;
    constexpr std::uint32_t ABS_V = p_sfpu::LREG2;
    constexpr std::uint32_t INF = p_sfpu::LREG5;
    constexpr std::uint32_t BFLOAT16_INF = 0x7f80;

    if constexpr (
        COMP_MODE == ZeroCompMode::LtZ || COMP_MODE == ZeroCompMode::GeZ || COMP_MODE == ZeroCompMode::GtZ ||
        COMP_MODE == ZeroCompMode::LeZ) {
        TTI_SFPLOADI(INF, sfpi::SFPLOADI_MOD0_FLOATB, BFLOAT16_INF);
    }

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(V, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSETSGN(0, V, ABS_V, 1);

        // eqz: default 0, set 1 where |v| == 0 (handles ±0; NaN has |v|!=0 → stays 0)
        if constexpr (COMP_MODE == ZeroCompMode::EqZ) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // nez: default 1, set 0 where |v| == 0 (handles ±0; NaN has |v|!=0 → stays 1)
        if constexpr (COMP_MODE == ZeroCompMode::NeZ) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // ltz: default 0; chain: (v < 0) AND (|v| != 0) AND (|v| <= inf) → 1
        if constexpr (COMP_MODE == ZeroCompMode::LtZ) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPSETCC(0, V, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPIADD(0, INF, ABS_V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // gtz: default 0; chain: (v >= 0) AND (|v| != 0) AND (|v| <= inf) → 1
        if constexpr (COMP_MODE == ZeroCompMode::GtZ) {
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPSETCC(0, V, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPIADD(0, INF, ABS_V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // gez: default 1; chain1: (v<0) AND (|v|!=0) → 0 (negatives excl. -0); chain2: |v|>inf → 0 (NaN)
        if constexpr (COMP_MODE == ZeroCompMode::GeZ) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPSETCC(0, V, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPENCC(0, 0, 0, 0);
            TTI_SFPIADD(0, INF, ABS_V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }

        // lez: default 1; chain1: (v>=0) AND (|v|!=0) → 0 (positives excl. +0); chain2: |v|>inf → 0 (NaN)
        if constexpr (COMP_MODE == ZeroCompMode::LeZ) {
            TTI_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPSETCC(0, V, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
            TTI_SFPSETCC(0, ABS_V, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
            TTI_SFPENCC(0, 0, 0, 0);
            TTI_SFPIADD(0, INF, ABS_V, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
            TTI_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
            TTI_SFPENCC(0, 0, 0, 0);
        }
    }
}

template <bool APPROXIMATION_MODE, ZeroCompMode COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_int() {
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0];
        vInt zero = 0;

        // a[i] == 0
        if constexpr (COMP_MODE == ZeroCompMode::EqZ) {
            v_if(v == zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] != 0
        if constexpr (COMP_MODE == ZeroCompMode::NeZ) {
            v_if(v == zero) { v = zero; }
            v_else { v = 1; }
            v_endif;
        }

        // a[i] < 0
        if constexpr (COMP_MODE == ZeroCompMode::LtZ) {
            v_if(v < zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] > 0
        if constexpr (COMP_MODE == ZeroCompMode::GtZ) {
            v_if(v > zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] <= 0
        if constexpr (COMP_MODE == ZeroCompMode::LeZ) {
            v_if(v <= zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] >= 0
        if constexpr (COMP_MODE == ZeroCompMode::GeZ) {
            v_if(v >= zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, ZeroCompMode COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_uint16() {
    static_assert((COMP_MODE == ZeroCompMode::EqZ) or (COMP_MODE == ZeroCompMode::NeZ));
    // UInt16 values live in the low 16 bits of the dest word; DataLayout::U16 loads/stores them
    // directly (SFPLOAD/SFPSTORE mod = UINT16), matching the InstrModLoadStore::LO16 path.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vUInt v = dst_reg[0].mode<sfpi::DataLayout::U16>();
        if constexpr (COMP_MODE == ZeroCompMode::EqZ) {
            vUInt r = 0;
            v_if(v == 0) { r = 1; }
            v_endif;
            dst_reg[0].mode<sfpi::DataLayout::U16>() = r;
        } else {
            vUInt r = 1;
            v_if(v == 0) { r = 0; }
            v_endif;
            dst_reg[0].mode<sfpi::DataLayout::U16>() = r;
        }
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_eqz_uint32() {
    // UInt32 values occupy the full dest word; DataLayout::U32 loads/stores them
    // directly (SFPLOAD/SFPSTORE mod = UINT32). eqz/nez are representation-agnostic
    // (only a compare against the all-zero word), so a plain unsigned compare works.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vUInt v = dst_reg[0].mode<sfpi::DataLayout::U32>();
        vUInt r = 0;
        v_if(v == 0) { r = 1; }
        v_endif;
        dst_reg[0].mode<sfpi::DataLayout::U32>() = r;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_nez_uint32() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vUInt v = dst_reg[0].mode<sfpi::DataLayout::U32>();
        vUInt r = 0;
        v_if(v != 0) { r = 1; }
        v_endif;
        dst_reg[0].mode<sfpi::DataLayout::U32>() = r;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, UnaryCompMode COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_unary_int(int scalar) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0];
        vInt val = 0;

        // a[i] != scalar
        if constexpr (COMP_MODE == UnaryCompMode::Ne) {
            v_if(v != scalar) { val = 1; }
            v_endif;
        }
        // a[i] == scalar
        else if constexpr (COMP_MODE == UnaryCompMode::Eq) {
            v_if(v == scalar) { val = 1; }
            v_endif;
        }
        dst_reg[0] = val;
        dst_reg++;
    }
}

// ---------------------------------------------------------------------------------------------------
// UnaryCompInt<APPROX, COMP_OP, DST_SYNC, DST_ACCUM, ITERATIONS>::calculate(dst_index, vector_mode, scalar)
//   backs unary_ne/eq/gt/ge/lt/le_tile_int32 (int32 compare of each element against `scalar`).
//   ne/eq use this header's calculate_comp_unary_int, gt/ge/lt/le the tt-llk _calculate_comp_unary_int_.
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, UnaryCompMode COMP_OP, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct UnaryCompInt
    : SfpuUnaryOp<UnaryCompInt<APPROXIMATION_MODE, COMP_OP, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static_assert(
        COMP_OP == UnaryCompMode::Ne || COMP_OP == UnaryCompMode::Eq || COMP_OP == UnaryCompMode::Gt ||
            COMP_OP == UnaryCompMode::Ge || COMP_OP == UnaryCompMode::Lt || COMP_OP == UnaryCompMode::Le,
        "UnaryCompInt supports only UnaryCompMode::Ne/Eq/Gt/Ge/Lt/Le");

    static void kernel(int scalar) {
        if constexpr (COMP_OP == UnaryCompMode::Ne || COMP_OP == UnaryCompMode::Eq) {
            calculate_comp_unary_int<APPROXIMATION_MODE, COMP_OP, ITERATIONS>(scalar);
        } else {
            _calculate_comp_unary_int_<APPROXIMATION_MODE, COMP_OP, ITERATIONS>(scalar);
        }
    }
};

// ---------------------------------------------------------------------------------------------------
// ZeroComp<APPROX, FORMAT, COMP_OP, DST_SYNC, DST_ACCUM, ITERATIONS>::calculate(dst_index, vector_mode)
//   backs eqz/nez/ltz/gtz/lez/gez_tile (float formats), *_tile_int32 (Int32), eqz/nez_tile_uint16 (UInt16),
//   eqz/nez_tile_uint32 (UInt32) and the *_tile_init entry points (init_kernel programs ADDR_MOD_6, dest incr 2).
//   COMP_OP is one of ZeroCompMode::EqZ/NeZ/LtZ/GtZ/LeZ/GeZ.
// ---------------------------------------------------------------------------------------------------
template <
    bool APPROXIMATION_MODE,
    DataFormat FORMAT,
    ZeroCompMode COMP_OP,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    int ITERATIONS = 8>
struct ZeroComp
    : SfpuUnaryOp<ZeroComp<APPROXIMATION_MODE, FORMAT, COMP_OP, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() {
        if constexpr (FORMAT == DataFormat::Int32) {
            calculate_comp_int<APPROXIMATION_MODE, COMP_OP, ITERATIONS>();
        } else if constexpr (FORMAT == DataFormat::UInt16) {
            calculate_comp_uint16<APPROXIMATION_MODE, COMP_OP, ITERATIONS>();
        } else if constexpr (FORMAT == DataFormat::UInt32) {
            static_assert(
                COMP_OP == ZeroCompMode::EqZ || COMP_OP == ZeroCompMode::NeZ,
                "UInt32 comparison-to-zero supports only EqZ / NeZ");
            if constexpr (COMP_OP == ZeroCompMode::EqZ) {
                calculate_eqz_uint32<APPROXIMATION_MODE, ITERATIONS>();
            } else {
                calculate_nez_uint32<APPROXIMATION_MODE, ITERATIONS>();
            }
        } else {
            calculate_comp<APPROXIMATION_MODE, COMP_OP, ITERATIONS>();
        }
    }

    // Zero-comparisons store through ADDR_MOD_6 with dest auto-increment.
    static void init_kernel() {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    }
};

}  // namespace sfpu
}  // namespace ckernel
