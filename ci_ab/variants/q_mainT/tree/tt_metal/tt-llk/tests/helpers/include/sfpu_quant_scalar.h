// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Math-thread dispatch shared by the sfpu_quant_scalar test and perf kernels.

#include <cstdint>

#include "ckernel_sfpu.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "llk_sfpu/ckernel_sfpu_quant.h"

constexpr int QUANT_ITERATIONS = 32;

inline void quant_scalar_op_init()
{
    using namespace ckernel;
    constexpr std::uint32_t neg_zp_bits = QUANT_ZP_BITS ^ 0x80000000u; // dequant adds the negated zero point
    if constexpr (QUANT_OP == 0)
    {
        _llk_math_eltwise_binary_sfpu_init_<SfpuType::quant_int32>();
        if constexpr (QUANT_SCALE_FORM == 0)
        {
            sfpu::quant_init<false, false, DataFormat::Int32>(QUANT_ZP_BITS);
        }
        else
        {
            static_assert(QUANT_SCALE_FORM == 0, "main builds the tile form only");
        }
    }
    else if constexpr (QUANT_OP == 1)
    {
        _llk_math_eltwise_binary_sfpu_init_<SfpuType::requant_int32>();
        if constexpr (QUANT_SCALE_FORM == 0)
        {
            sfpu::requant_init<false, false, DataFormat::Int32, false>(QUANT_ZP_BITS);
        }
        else
        {
            static_assert(QUANT_SCALE_FORM == 0, "main builds the tile form only");
        }
    }
    else
    {
        _llk_math_eltwise_binary_sfpu_init_<SfpuType::dequant_int32>();
        if constexpr (QUANT_SCALE_FORM == 0)
        {
            sfpu::dequant_init<false, false, false>(neg_zp_bits);
        }
        else
        {
            static_assert(QUANT_SCALE_FORM == 0, "main builds the tile form only");
        }
    }
}

inline void quant_scalar_op_tile(std::uint32_t data_tile, std::uint32_t scale_tile, std::uint32_t result_tile)
{
    using namespace ckernel;
    static_assert(QUANT_SCALE_FORM == 0, "main builds the tile form only");
    if constexpr (QUANT_OP == 0)
    {
        _llk_math_eltwise_binary_sfpu_params_(
            sfpu::calculate_quant_int32<false, QUANT_ITERATIONS, false>, data_tile, scale_tile, result_tile, VectorMode::None);
    }
    else if constexpr (QUANT_OP == 1)
    {
        _llk_math_eltwise_binary_sfpu_params_(
            sfpu::calculate_requant_int32<false, QUANT_ITERATIONS, false, false>, data_tile, scale_tile, result_tile, VectorMode::None);
    }
    else
    {
        _llk_math_eltwise_binary_sfpu_params_(
            sfpu::calculate_dequant_int32<false, QUANT_ITERATIONS, false, false>, data_tile, scale_tile, result_tile, VectorMode::None);
    }
}
