// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

/**
 * @brief Floating-point class test selected by the SFPU isinf/isnan kernels.
 */
enum class FiniteCheck : std::uint8_t
{
    isinf,
    isposinf,
    isneginf,
    isnan,
    isfinite,
};

template <FiniteCheck CHECK, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sfpu_isinf_isnan_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in  = sfpi::dst_reg[0];
        sfpi::vFloat res = 0.0f;
        // Quasar: sfpi::is_inf / is_nan / is_finite spelled out. SFPI lowers their nearby() compare to a CC-only
        // SFPIADD into read-only LREG8, which leaves the lane mask unchanged on Quasar.
        sfpi::vInt exp = sfpi::exexp(in, sfpi::ExponentMode::Biased);

        if constexpr (CHECK == FiniteCheck::isinf)
        {
            v_if (exp >= 255 && sfpi::exman(in) == 0)
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (CHECK == FiniteCheck::isposinf)
        {
            v_if (sfpi::is_pos(in) && exp >= 255 && sfpi::exman(in) == 0)
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (CHECK == FiniteCheck::isneginf)
        {
            v_if (sfpi::is_neg(in) && exp >= 255 && sfpi::exman(in) == 0)
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (CHECK == FiniteCheck::isnan)
        {
            v_if (exp >= 255 && sfpi::exman(in) != 0)
            {
                res = 1.0f;
            }
            v_endif;
        }
        else if constexpr (CHECK == FiniteCheck::isfinite)
        {
            v_if (exp < 255)
            {
                res = 1.0f;
            }
            v_endif;
        }

        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

template <SfpuType operation>
constexpr FiniteCheck _sfpu_type_to_finite_check_()
{
    if constexpr (operation == SfpuType::isinf)
    {
        return FiniteCheck::isinf;
    }
    else if constexpr (operation == SfpuType::isposinf)
    {
        return FiniteCheck::isposinf;
    }
    else if constexpr (operation == SfpuType::isneginf)
    {
        return FiniteCheck::isneginf;
    }
    else if constexpr (operation == SfpuType::isnan)
    {
        return FiniteCheck::isnan;
    }
    else
    {
        static_assert(operation == SfpuType::isfinite, "SfpuType is not an isinf/isnan operation");
        return FiniteCheck::isfinite;
    }
}

// SfpuType-selected entry point, kept for existing callers. It forwards to the FiniteCheck version above.
template <SfpuType operation, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sfpu_isinf_isnan_()
{
    _calculate_sfpu_isinf_isnan_<_sfpu_type_to_finite_check_<operation>(), APPROXIMATION_MODE, ITERATIONS>();
}

// Op class for the floating-point class tests (isinf, isposinf, isneginf, isnan, isfinite): 1.0 where the
// test holds, else 0.0. Same name and leading template parameters as on Wormhole/Blackhole.
template <bool APPROXIMATION_MODE, FiniteCheck CHECK, int ITERATIONS = 8, trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct IsinfIsnan : SfpuUnaryOp<IsinfIsnan<APPROXIMATION_MODE, CHECK, ITERATIONS, SLOT>, SLOT>
{
    static inline __attribute__((always_inline)) void calculate()
    {
        _calculate_sfpu_isinf_isnan_<CHECK, APPROXIMATION_MODE, ITERATIONS>();
    }
};

} // namespace ckernel::sfpu
