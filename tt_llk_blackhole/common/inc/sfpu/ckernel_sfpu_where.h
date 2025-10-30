// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_where_fp16_b_(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_in2, const uint dst_index_out)
{
    // size of each tile in Dest is 64 rows
    constexpr uint dst_tile_size = 64;
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    // both are needed since this kernel mixes the use of sfpi and TT calls for load/store

    for (int i = 0; i < ITERATIONS; i++)
    {
        sfpi::vFloat cond = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];

        v_if (cond == 0.0f)
        {
            // output_tensor = false_tensor;
            TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::LO16, ADDR_MOD_7, dst_index_in2 * dst_tile_size);
        }
        v_else
        {
            // output_tensor = true_tensor;
            TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::LO16, ADDR_MOD_7, dst_index_in1 * dst_tile_size);
        }
        v_endif;
        // sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = output_tensor;
        TT_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::LO16, ADDR_MOD_7, dst_index_out * dst_tile_size);

        sfpi::dst_reg++;
    }
}

template <typename T, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_where_impl_(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_in2, const uint dst_index_out)
{
    constexpr uint dst_tile_size_sfpi = 32;

    for (int i = 0; i < ITERATIONS; i++)
    {
        T cond          = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        T output_tensor = 0;

        v_if (cond != 0)
        {
            output_tensor = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        }
        v_else
        {
            output_tensor = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi];
        }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = output_tensor;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, DataFormat data_format, int ITERATIONS>
inline void _calculate_where_(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_in2, const uint dst_index_out)
{
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b || data_format == DataFormat::Int32 || data_format == DataFormat::UInt32,
        "Unsupported data format for _calculate_where_(). Only Float32, Int32, UInt32, and Float16_b are allowed.");

    if constexpr (data_format == DataFormat::Float16_b)
    {
        _calculate_where_fp16_b_<APPROXIMATION_MODE, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out);
    }
    else if constexpr (data_format == DataFormat::Float32)
    {
        _calculate_where_impl_<sfpi::vFloat, APPROXIMATION_MODE, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out);
    }
    else if constexpr (data_format == DataFormat::Int32)
    {
        _calculate_where_impl_<sfpi::vInt, APPROXIMATION_MODE, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out);
    }
    else if constexpr (data_format == DataFormat::UInt32)
    {
        _calculate_where_impl_<sfpi::vUInt, APPROXIMATION_MODE, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out);
    }
    else
    {
        static_assert(false, "Unsupported data format for _calculate_where_(). Only Float32, Int32, UInt32, and Float16_b are allowed.");
    }
}

} // namespace ckernel::sfpu
