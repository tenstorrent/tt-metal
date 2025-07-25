// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

/*

Expects following input in Dst register:
Index 0 ( Tile 0 ) -> condition tensor
Index 32 ( Tile 1 ) -> true tensor
Index 64 ( Tile 2 ) -> false tensor

*/

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_where_fp16_b_()
{
    constexpr uint dst_tile_size_rows = 64;

    sfpi::vFloat cond = sfpi::dst_reg[0];

    for (int i = 0; i < ITERATIONS; i++)
    {
        cond = sfpi::dst_reg[0];

        v_if (cond == 0.0f)
        {
            // output_tensor = false_tensor;
            TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::LO16, ADDR_MOD_3, 2 * dst_tile_size_rows);
        }
        v_else
        {
            // output_tensor = true_tensor;
            TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::LO16, ADDR_MOD_3, dst_tile_size_rows);
            v_endif;
        }
        // sfpi::dst_reg[0] = output_tensor;
        TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::LO16, ADDR_MOD_3, 0);

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_where_fp32_()
{
    constexpr uint dst_tile_size = 32;

    sfpi::vFloat output_tensor = 0;
    sfpi::vFloat true_tensor   = 0;
    sfpi::vFloat false_tensor  = 0;
    sfpi::vFloat cond          = sfpi::dst_reg[0];

    for (int i = 0; i < ITERATIONS; i++)
    {
        cond         = sfpi::dst_reg[0];
        true_tensor  = sfpi::dst_reg[dst_tile_size];
        false_tensor = sfpi::dst_reg[dst_tile_size * 2];

        v_if (cond != 0.0f)
        {
            output_tensor = true_tensor;
        }
        v_else
        {
            output_tensor = false_tensor;
        }
        v_endif;

        sfpi::dst_reg[0] = output_tensor;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, DataFormat data_format, int ITERATIONS>
inline void _calculate_where_()
{
    // Add a compile-time check to ensure only supported formats are used.
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for _calculate_where_(). Only Float32 and Float16_b are allowed.");
    if constexpr (data_format == DataFormat::Float32)
    {
        _calculate_where_fp32_<APPROXIMATION_MODE, ITERATIONS>();
    }
    else
    {
        _calculate_where_fp16_b_<APPROXIMATION_MODE, ITERATIONS>();
    }
}

} // namespace ckernel::sfpu
