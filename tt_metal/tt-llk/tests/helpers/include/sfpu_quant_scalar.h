// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Math-thread dispatch of the sfpu_quant_scalar test and perf kernels: quant, requant and dequant (QUANT_OP 0, 1, 2)
// with the scale as a DEST tile (QUANT_SCALE_FORM 0) or loaded once by the init (QUANT_SCALE_FORM 1), from
// QUANT_SCALAR_CFG; QUANT_ZP_BITS and QUANT_SCALE_BITS are fp32 bits.

#include <cstdint>

#include "ckernel_sfpu.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "llk_sfpu/ckernel_sfpu_quant.h"

// One 32-row call per tile, as the compute API issues it on Blackhole.
constexpr int QUANT_ITERATIONS = 32;

inline void quant_scalar_op_init()
{
    using namespace ckernel;
    constexpr std::uint32_t neg_zp_bits = QUANT_ZP_BITS ^ 0x80000000u; // dequant takes the negated zero point
    if constexpr (QUANT_OP == 0)
    {
        _llk_math_eltwise_binary_sfpu_init_<SfpuType::quant_int32>();
        if constexpr (QUANT_SCALE_FORM == 0)
        {
            sfpu::quant_init<false, false, DataFormat::Int32>(QUANT_ZP_BITS);
        }
        else
        {
            sfpu::quant_init_scalar_scale<false, false, DataFormat::Int32>(QUANT_ZP_BITS, QUANT_SCALE_BITS);
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
            sfpu::requant_init_scalar_scale<false, false, DataFormat::Int32, false>(QUANT_ZP_BITS, QUANT_SCALE_BITS);
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
            sfpu::dequant_init_scalar_scale<false, false, false>(neg_zp_bits, QUANT_SCALE_BITS);
        }
    }
}

inline void quant_scalar_op_tile(std::uint32_t data_tile, std::uint32_t scale_tile, std::uint32_t result_tile)
{
    using namespace ckernel;
    constexpr bool scalar = (QUANT_SCALE_FORM == 1);
    if constexpr (QUANT_OP == 0)
    {
        _llk_math_eltwise_binary_sfpu_params_(
            sfpu::calculate_quant_int32<false, QUANT_ITERATIONS, false, scalar>, data_tile, scale_tile, result_tile, VectorMode::None);
    }
    else if constexpr (QUANT_OP == 1)
    {
        _llk_math_eltwise_binary_sfpu_params_(
            sfpu::calculate_requant_int32<false, QUANT_ITERATIONS, false, false, scalar>, data_tile, scale_tile, result_tile, VectorMode::None);
    }
    else
    {
        _llk_math_eltwise_binary_sfpu_params_(
            sfpu::calculate_dequant_int32<false, QUANT_ITERATIONS, false, false, scalar>, data_tile, scale_tile, result_tile, VectorMode::None);
    }
}
