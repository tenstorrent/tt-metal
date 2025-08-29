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

/**
 * @brief Calculates column-wise MaxPool of a tile, placing output values into the first row.
 *        Also places the index of the max value into the first row of the indices tile.
 *        Supports {FP32, FP16_B} for values, and {UINT16, INT32, UINT32} for indices, inferred from the Dest mode used.
 * @tparam APPROXIMATION_MODE Whether to use the approximation mode (unused).
 * @tparam is_fp32_dest_acc_en Whether Dest is in 32bit mode (true) or 16bit mode (false).
 * @tparam num_rows The number of rows in the tile, must be one of: {9}
 * @tparam ITERATIONS The number of iterations to use for the MaxPool operation (unused).
 * @param values_tile_idx The index of the tile in the Dest register containing the data to be reduced.
 * @param indices_tile_idx The index of the tile in the Dest register containing the indices of the data.
 * @param tile_idx Unused param, needed to conform with format in _llk_math_eltwise_binary_sfpu_params_.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int num_rows, int ITERATIONS>
inline void _calculate_max_pool_with_indices_(const uint values_tile_idx, const uint indices_tile_idx, const uint tile_idx /* unused */)
{
    static_assert(num_rows == 9, "num_rows must be one of: {9}"); // add others as support is added

    // size of each tile in Dest is 64 rows
    constexpr uint dst_tile_size   = 64;
    const uint values_tile_offset  = values_tile_idx * dst_tile_size;
    const uint indices_tile_offset = indices_tile_idx * dst_tile_size;
    // each face is 16 rows
    constexpr uint face_offset        = 16;
    constexpr uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    // F0
    // data
    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 0); // even cols
    TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 4);
    TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 2); // odd cols
    TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 6);
    // index
    TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 0); // even cols
    TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 4);
    TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 2); // odd cols
    TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 6);

    // sort 4 rows
    lltt::replay(0, 7);

    // data
    TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 8);  // even cols
    TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 10); // odd cols
    // index
    TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 8);  // even cols
    TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 10); // odd cols

    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

    TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 0);
    TT_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 2);
    TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 0);
    TT_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 2);

    // F1
    // data
    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + face_offset); // even cols
    TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + face_offset + 4);
    TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + face_offset + 2); // odd cols
    TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + face_offset + 6);
    // index
    TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_tile_offset + face_offset); // even cols
    TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_tile_offset + face_offset + 4);
    TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, indices_tile_offset + face_offset + 2); // odd cols
    TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, indices_tile_offset + face_offset + 6);

    // sort 4 rows
    lltt::replay(0, 7);

    // data
    TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + face_offset + 8);  // even cols
    TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + face_offset + 10); // odd cols
    // index
    TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_tile_offset + face_offset + 8);  // even cols
    TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, indices_tile_offset + face_offset + 10); // odd cols

    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

    TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + face_offset);
    TT_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + face_offset + 2);
    TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_tile_offset + face_offset);
    TT_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, indices_tile_offset + face_offset + 2);
}

inline void _init_max_pool_with_indices_()
{
    // Set bit [2] of the SFPU_CONTROL_REG to enable Destination Index Tracking Mode:
    // LREGs 4-7 will be treated as indices corresponding to the values in LREGs 0-3,
    // and LREGs 4-7 will mirror the movement of the values in LREGs 0-3;
    _sfpu_load_config32_(0xF, 0x0, 0x4);

    // Program replay buffer
    lltt::record(0, 7);

    // Values have been loaded such that 4 rows of Dest occupy the 4 lanes of each LREG
    // To sort those 4 rows, we transpose the SFPU LREGs to put elements of 4 rows of each column into separate LREGs of each unit of SFPU
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Sort the 4 rows of Dest, placing the max value into LREG0 (index into LREG4)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

    // Transpose the LREGs back
    TTI_SFPTRANSP(0, 0, 0, 0);

    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
}

} // namespace sfpu
} // namespace ckernel
