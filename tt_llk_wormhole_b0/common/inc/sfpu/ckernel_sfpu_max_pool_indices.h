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
 *        Can reduce up to 9 rows of a tile.
 * @tparam APPROXIMATION_MODE Whether to use the approximation mode (unused).
 * @tparam is_fp32_dest_acc_en Whether Dest is in 32bit mode (true) or 16bit mode (false).
 * @tparam ITERATIONS The number of iterations to perform (unused).
 * @tparam layout Data layout format, either TILE (default) or ROW_MAJOR.
 * @param values_tile_idx The index of the tile in the Dest register containing the data to be reduced.
 * @param indices_tile_idx The index of the tile in the Dest register containing the indices of the data.
 * @param tile_idx Unused param, needed to conform with format in _llk_math_eltwise_binary_sfpu_params_.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8, ckernel::DataLayout layout = ckernel::DataLayout::TILE>
inline void _calculate_max_pool_with_indices_(const uint values_tile_idx, const uint indices_tile_idx, const uint tile_idx /* unused */)
{
    // size of each tile in Dest is 64 rows
    constexpr uint dst_tile_size   = 64;
    const uint values_tile_offset  = values_tile_idx * dst_tile_size;
    const uint indices_tile_offset = indices_tile_idx * dst_tile_size;
    // each face is 16 rows
    constexpr uint face_offset        = 16;
    constexpr uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    if constexpr (layout == ckernel::DataLayout::ROW_MAJOR)
    {
        // ROW MAJOR DATA VERSION OF MPWI
        // DATA IS EXPECTED TO BE IN THE FOLLOWING ORDER IN DEST:
        // Face 0 Row 0
        // Face 1 Row 0
        // Face 0 Row 1
        // Face 1 Row 1
        // Face 0 Row 2
        // Face 1 Row 2
        // Face 0 Row 3
        // Face 1 Row 3
        // Face 0 Row 4
        // Face 1 Row 4
        // Face 0 Row 5
        // Face 1 Row 5
        // Face 0 Row 6
        // Face 1 Row 6
        // Face 0 Row 7
        // Face 1 Row 7
        // Face 0 Row 8
        // Face 1 Row 8

        auto process_columns = [values_tile_offset, indices_tile_offset](const uint col_offset) __attribute__((always_inline))
        {
            // data
            TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 0 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 4 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 8 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 12 + col_offset);
            // index
            TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 0 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 4 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 8 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 12 + col_offset);

            // sort 4 rows
            lltt::replay(0, 7);

            // data
            TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 16 + col_offset);
            // index
            TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 16 + col_offset);

            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

            TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 0 + col_offset);
            TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 0 + col_offset);
        };

        // F0 + F1 even and odd cols
        constexpr int even_column_offset = 0;
        constexpr int odd_column_offset  = 2;
        process_columns(even_column_offset);
        process_columns(odd_column_offset);
    }
    else
    {
        // TILE (ORIGINAL) VERSION OF MPWI
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
}

/**
 * @brief Calculates column-wise MaxPool of a tile, placing output values into the first row.
 *        Also places the index of the max value into the first row of the indices tile.
 *        Supports {FP32, FP16_B} for values, and {UINT16, INT32, UINT32} for indices, inferred from the Dest mode used.
 *        Can reduce up to 32 rows of a tile.
 * @tparam APPROXIMATION_MODE Whether to use the approximation mode (unused).
 * @tparam is_fp32_dest_acc_en Whether Dest is in 32bit mode (true) or 16bit mode (false).
 * @tparam ITERATIONS The number of iterations to use for the MaxPool operation (unused).
 * @param values_tile_idx The index of the tile in the Dest register containing the data to be reduced.
 * @param indices_tile_idx The index of the tile in the Dest register containing the indices of the data.
 * @param tile_idx Unused param, needed to conform with format in _llk_math_eltwise_binary_sfpu_params_.
 *
 * Note this function is only implemented for ROW_MAJOR data layout, so when _init_max_pool_with_indices_ is called
 * it must be called with layout=DataLayout::ROW_MAJOR.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void _calculate_max_pool_with_indices_generic_(const uint values_tile_idx, const uint indices_tile_idx, const uint tile_idx /* unused */)
{
    // size of each tile in Dest is 64 rows
    constexpr uint dst_tile_size   = 64;
    const uint values_tile_offset  = values_tile_idx * dst_tile_size;
    const uint indices_tile_offset = indices_tile_idx * dst_tile_size;
    // each face is 16 rows
    constexpr uint eight_row_offset   = 16;
    constexpr uint sixteen_row_offset = 32;
    constexpr uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    // ROW MAJOR DATA VERSION OF MPWI
    // DATA IS EXPECTED TO BE IN THE FOLLOWING ORDER IN DEST:
    // Face 0 Row 0
    // Face 1 Row 0
    // Face 0 Row 1
    // Face 1 Row 1
    // .
    // .
    // .
    // Face 0 Row 31
    // Face 1 Row 31

    auto process_16_rows = [values_tile_offset, indices_tile_offset, eight_row_offset, instr_mod_index](const uint base_offset, const uint col_offset)
                               __attribute__((always_inline))
    {
        // Nested lambda to handle load, sort, and store for a face
        auto load_sort_store = [values_tile_offset, indices_tile_offset, base_offset, col_offset, instr_mod_index](const uint eight_row_offset_val)
                                   __attribute__((always_inline))
        {
            // data
            TT_SFPLOAD(
                p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + eight_row_offset_val + base_offset + 0 + col_offset); // Row 0 and 1
            TT_SFPLOAD(
                p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + eight_row_offset_val + base_offset + 4 + col_offset); // Row 2 and 3
            TT_SFPLOAD(
                p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + eight_row_offset_val + base_offset + 8 + col_offset); // Row 4 and 5
            TT_SFPLOAD(
                p_sfpu::LREG3,
                InstrModLoadStore::DEFAULT,
                ADDR_MOD_3,
                values_tile_offset + eight_row_offset_val + base_offset + 12 + col_offset); // Row 6 and 7
            // index
            TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_tile_offset + eight_row_offset_val + base_offset + 0 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_tile_offset + eight_row_offset_val + base_offset + 4 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, indices_tile_offset + eight_row_offset_val + base_offset + 8 + col_offset);
            TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, indices_tile_offset + eight_row_offset_val + base_offset + 12 + col_offset);

            // data is loaded in this format:
            // LREG0          LREG1          LREG2          LREG3
            // Face 0 Row 0   Face 0 Row 2   Face 0 Row 4   Face 0 Row 6
            // Face 1 Row 0   Face 1 Row 2   Face 1 Row 4   Face 1 Row 6
            // Face 0 Row 1   Face 0 Row 3   Face 0 Row 5   Face 0 Row 7
            // Face 1 Row 1   Face 1 Row 3   Face 1 Row 5   Face 1 Row 7

            // then we sort 4 rows, replay does:
            // max between LREG0 and LREG1
            // max between LREG2 and LREG3
            // max between LREG0 and LREG2
            // -
            // now we have these maxes:
            // LREG0
            // Max(F0, R0,2,4,6)
            // Max(F1, R0,2,4,6)
            // Max(F0, R1,3,5,7)
            // Max(F1, R1,3,5,7)
            // -
            // transpose
            // max between LREG0 and LREG2
            // max between LREG1 and LREG3
            // transpose
            // -
            // now we have in LREG0:
            // Max(F0, R0-7)
            // Max(F1, R0-7)
            // Max(F0, R1,3,5,7)
            // Max(F1, R1,3,5,7)
            lltt::replay(0, 7);

            TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + eight_row_offset_val + base_offset + 0 + col_offset);
            TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_tile_offset + eight_row_offset_val + base_offset + 0 + col_offset);
        };

        // Process first 8 rows and second 8 rows for F0 and F1
        load_sort_store(0);
        load_sort_store(eight_row_offset);

        // swap between the two sets of 8 rows
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + base_offset + 0 + col_offset); // Max(R0-7) for F0,1
        TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_tile_offset + base_offset + 0 + col_offset);
        TT_SFPLOAD(
            p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + eight_row_offset + base_offset + 0 + col_offset); // Max(R8-15) for F0,1
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_tile_offset + eight_row_offset + base_offset + 0 + col_offset);

        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // LREG0 contains Max(R0-15) (or Max(R16-31)) for F0,1

        TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_tile_offset + base_offset + col_offset);
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + base_offset + col_offset);
    };

    // First 16 rows
    constexpr int even_column_offset = 0;
    constexpr int odd_column_offset  = 2;
    process_16_rows(0, even_column_offset);
    process_16_rows(0, odd_column_offset);

    // Second 16 rows
    process_16_rows(sixteen_row_offset, even_column_offset);
    process_16_rows(sixteen_row_offset, odd_column_offset);

    // Final swap
    auto final_swap = [values_tile_offset, indices_tile_offset, sixteen_row_offset, instr_mod_index](const uint col_offset) __attribute__((always_inline))
    {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + col_offset); // Max(R0-15) for F0,1
        TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_tile_offset + col_offset);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + sixteen_row_offset + col_offset); // Max(R16-31) for F0,1
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_tile_offset + sixteen_row_offset + col_offset);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // LREG0 contains Max(R0-31) for F0,1
        TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_tile_offset + col_offset);
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + col_offset);
    };

    final_swap(even_column_offset);
    final_swap(odd_column_offset);
}

template <ckernel::DataLayout layout = ckernel::DataLayout::TILE>
inline void _init_max_pool_with_indices_()
{
    // Set bit [2] of the SFPU_CONTROL_REG to enable Destination Index Tracking Mode:
    // LREGs 4-7 will be treated as indices corresponding to the values in LREGs 0-3,
    // and LREGs 4-7 will mirror the movement of the values in LREGs 0-3;
    _sfpu_load_config32_(0xF, 0x0, 0x4);

    if constexpr (layout == ckernel::DataLayout::ROW_MAJOR)
    {
        // Program replay buffer for row major layout
        lltt::record(0, 7);

        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);

        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);
    }
    else
    {
        // Program replay buffer for tiled layout (original)
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
}

} // namespace sfpu
} // namespace ckernel
