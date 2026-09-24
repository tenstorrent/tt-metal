// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel {
namespace sfpu {
// Calculates RELU for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_relu_sfp_rows_() {
    TTI_SFPLOAD(
        p_sfpu::LREG0,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        0,
        0);  // load from dest into lreg[0], uses ADDR_MOD_7 (set to all zeroes)

    // SFPARECIP with RELU_MODE does relu eltwise
    TTI_SFPNONLINEAR(
        p_sfpu::LREG0,
        p_sfpu::LREG1,
        p_sfpnonlinear::RELU_MODE);  // Read value from lreg[0], get relu value, load back into lreg[1]

    // Store from lreg[1] into dest register
    TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);
}

// Implements standard relu which does max(0, x)
template <int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_relu_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_relu_sfp_rows_();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();  // does the dest_reg++ (increments by
                                                                                   // 2 rows)
    }
}

inline void _relu_load_threshold_(const std::uint32_t threshold) {
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, threshold & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, threshold >> 16);
}

template <typename T>
inline std::uint32_t _relu_threshold_bits_(T threshold) {
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    if constexpr (std::is_same_v<T, float>) {
        return __builtin_bit_cast(std::uint32_t, threshold);
    } else {
        return threshold;
    }
}

// Calculates Leaky RELU for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_lrelu_sfp_rows_() {
    TTI_SFPLOAD(
        p_sfpu::LREG0,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        0,
        0);  // load from dest into lreg[0], uses ADDR_MOD_7 (set to all zeroes)

    TTI_SFPSETCC(0, p_sfpu::LREG0, 0);  // condition - if value in LREG0 is negative //will set cc result reg

    TTI_SFPMAD(
        p_sfpu::LREG0,
        p_sfpu::LREG2,
        p_sfpu::LCONST_0,
        p_sfpu::LREG0,
        0);  // Multiply and add - LREG0 * LREG2 + LCONST_0 (x * slope + 0)

    TTI_SFPENCC(0, 0);  // clear cc result reg

    // Store from lreg0 into dest register
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);
}

// Implements leaky relu which return x when x > 0 and x*slope when x < 0.
template <int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_lrelu_(const std::uint32_t slope) {
    _relu_load_threshold_(slope);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_lrelu_sfp_rows_();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();  // does the dest_reg++ (increments by
                                                                                   // 2 rows)
    }
}

// Op class for leaky relu. Same name and leading template parameters as on Wormhole/Blackhole.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct Lrelu : SfpuUnaryOp<Lrelu<APPROXIMATION_MODE, ITERATIONS, SLOT>, SLOT> {
    static constexpr auto& calculate = _calculate_lrelu_<ITERATIONS>;
};

// Calculates RELU MIN for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_relu_min_sfp_rows_() {
    TTI_SFPLOAD(
        p_sfpu::LREG0,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        0,
        0);  // load from dest into lreg[0], uses ADDR_MOD_7 (set to all zeroes)

    TTI_SFPGT(p_sfpgt::IMM12_FP32, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpgt::MOD1_SET_CC);
    TTI_SFPMOV(p_sfpu::LREG2 /*src*/, p_sfpu::LREG0 /*dest*/, 0);
    TTI_SFPENCC(0, 0);

    // Store from lreg0 into dest register
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);
}

template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(T threshold) {
    static_assert(ITERATIONS == SFPU_ITERATIONS);
    static_assert(std::is_same_v<VectorType, sfpi::vFloat>, "Quasar relu_min is float-only");
    _relu_load_threshold_(_relu_threshold_bits_(threshold));
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_relu_min_sfp_rows_();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();  // does the dest_reg++ (increments by
                                                                                   // 2 rows)
    }
}

// Op class for relu_min: max(x, threshold). Same name and leading template parameters as on Wormhole/Blackhole.
template <
    typename VectorType,
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    typename T = std::uint32_t,
    trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct ReluMin : SfpuUnaryOp<ReluMin<VectorType, APPROXIMATION_MODE, ITERATIONS, T, SLOT>, SLOT> {
    static constexpr auto& calculate = _relu_min_<VectorType, APPROXIMATION_MODE, ITERATIONS, T>;
};

// Calculates RELU MAX for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_relu_max_sfp_rows_() {
    TTI_SFPLOAD(
        p_sfpu::LREG0,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        0,
        0);  // load from dest into lreg[0], uses ADDR_MOD_7 (set to all zeroes)

    TTI_SFPGT(p_sfpgt::IMM12_FP32, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpgt::MOD1_SET_CC);
    TTI_SFPMOV(p_sfpu::LREG2 /*src*/, p_sfpu::LREG0 /*dest*/, 0);
    TTI_SFPENCC(0, 0);

    TTI_SFPNONLINEAR(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpnonlinear::RELU_MODE);

    TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);
}

template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_max_(T threshold) {
    static_assert(ITERATIONS == SFPU_ITERATIONS);
    static_assert(std::is_same_v<VectorType, sfpi::vFloat>, "Quasar relu_max is float-only");
    _relu_load_threshold_(_relu_threshold_bits_(threshold));
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_relu_max_sfp_rows_();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();  // does the dest_reg++ (increments by
                                                                                   // 2 rows)
    }
}

// Op class for relu_max: max(0, min(x, threshold)). Same name and leading template parameters as on Wormhole/Blackhole.
template <
    typename VectorType,
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    typename T = std::uint32_t,
    trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct ReluMax : SfpuUnaryOp<ReluMax<VectorType, APPROXIMATION_MODE, ITERATIONS, T, SLOT>, SLOT> {
    static constexpr auto& calculate = _relu_max_<VectorType, APPROXIMATION_MODE, ITERATIONS, T>;
};

}  // namespace sfpu
}  // namespace ckernel
