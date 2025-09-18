// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_reshuffle_rows_(const uint idx_addr)
{
    constexpr uint output_tile_offset = 64;

    // clr DEST tile 1
    // TODO (Radomir): Add optional clear that is more optimal using tile copy
    // for (uint row=0; row < 32; row+=4) {
    //     TT_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_3, output_tile_offset + row);
    //     TT_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_3, output_tile_offset + row + 2);
    //     TT_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_3, output_tile_offset + row + 32);
    //     TT_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD_3, output_tile_offset + row + 34);
    // }

    volatile tt_l1_ptr uint8_t *idx_ptr  = reinterpret_cast<volatile tt_l1_ptr uint8_t *>(idx_addr + (1 << 4));
    static constexpr uint input_lreg[4]  = {p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3};
    static constexpr uint output_lreg[4] = {p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG7};

    for (uint row = 0; row < 32; row++)
    {
        uint input_row_addr = (row < 16) ? ((row / 4) * 4) : ((row / 4) * 4 + 16);
        uint input_row_lreg = input_lreg[row % 4];

        uint dst_row = (uint)idx_ptr[row];
        if (dst_row >= 32)
        {
            continue;
        }
        uint output_row_addr = (dst_row < 16) ? ((dst_row / 4) * 4) : ((dst_row / 4) * 4 + 16);
        uint output_row_lreg = output_lreg[dst_row % 4];

        // load in the input row and output row
        TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, input_row_addr);
        TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, input_row_addr + 2);
        TT_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_3, input_row_addr + 16);
        TT_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_3, input_row_addr + 18);
        TT_SFPLOAD(p_sfpu::LREG4, 0, ADDR_MOD_3, output_tile_offset + output_row_addr);
        TT_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_3, output_tile_offset + output_row_addr + 2);
        TT_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_3, output_tile_offset + output_row_addr + 16);
        TT_SFPLOAD(p_sfpu::LREG7, 0, ADDR_MOD_3, output_tile_offset + output_row_addr + 18);
        TTI_SFPTRANSP(0, 0, 0, 0); // puts desired input row into LREG "input_row_lreg" and output row into "output_row_lreg"

        // reduce the values in the input row and output row by addition
        TT_SFPADD(input_row_lreg, p_sfpu::LCONST_1, output_row_lreg, output_row_lreg, 0);
        TTI_SFPNOP;

        // store back the reduce rows to output
        TTI_SFPTRANSP(0, 0, 0, 0); // puts desired output row back into into LREG4
        TT_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_3, output_tile_offset + output_row_addr);
        TT_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_3, output_tile_offset + output_row_addr + 2);
        TT_SFPSTORE(p_sfpu::LREG6, 0, ADDR_MOD_3, output_tile_offset + output_row_addr + 16);
        TT_SFPSTORE(p_sfpu::LREG7, 0, ADDR_MOD_3, output_tile_offset + output_row_addr + 18);
    }
}

} // namespace sfpu
} // namespace ckernel
