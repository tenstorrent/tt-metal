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

    for (int i = 0; i < ITERATIONS; i++)
    {
        // load conditional value
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_in0 * dst_tile_size);

        // if (cond != 0): load true value
        TTI_SFPSETCC(0 /*imm12_math*/, p_sfpu::LREG0, 0 /*unused*/, 2 /*if non-zero*/);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::LO16, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        // else: load false value
        TTI_SFPCOMPC(0 /*unused*/, 0 /*unused*/, 0 /*unused*/, 0 /*unused*/);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::LO16, ADDR_MOD_3, dst_index_in2 * dst_tile_size);
        // end if
        TTI_SFPENCC(0 /*imm12_math*/, 0 /*unused*/, 0 /*unused*/, 0 /*reset cc*/);

        // store result
        TT_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::LO16, ADDR_MOD_3, dst_index_out * dst_tile_size);

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
