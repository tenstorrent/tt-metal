// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// The Dest tile indices are template parameters throughout this file so that every SFPLOAD / SFPSTORE encodes as a
// single TTI_ instruction word (one RISC instruction) instead of a runtime TT_ encode (address arithmetic plus a
// store into the instruction buffer per word). The SFPU instruction stream is identical either way; only the RISC
// issue stream shrinks. That matters on the 32-row path, where the RISC, not the SFPU, otherwise sets the pace.

// size of each tile in Dest is 64 rows
constexpr std::uint32_t max_pool_dst_tile_size = 64;

template <bool is_fp32_dest_acc_en>
constexpr InstrModLoadStore max_pool_instr_mod_index =
    is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

/**
 * @brief ROW_MAJOR, up to 9 rows: reduces one column parity (even or odd columns of both faces) in place.
 * @tparam values_tile_offset Dest row of the values tile.
 * @tparam indices_tile_offset Dest row of the indices tile.
 * @tparam col_offset 0 for the even columns, 2 for the odd columns.
 */
template <
    bool is_fp32_dest_acc_en,
    std::uint32_t values_tile_offset,
    std::uint32_t indices_tile_offset,
    std::uint32_t col_offset>
inline void _max_pool_with_indices_row_major_columns_() {
    constexpr InstrModLoadStore instr_mod_index = max_pool_instr_mod_index<is_fp32_dest_acc_en>;

    // data
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 0 + col_offset);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 4 + col_offset);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 8 + col_offset);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 12 + col_offset);
    // index
    TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 0 + col_offset);
    TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 4 + col_offset);
    TTI_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 8 + col_offset);
    TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 12 + col_offset);

    // sort 4 rows
    lltt::replay(0, 7);

    // data
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 16 + col_offset);
    // index
    TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 16 + col_offset);

    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_tile_offset + 0 + col_offset);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_tile_offset + 0 + col_offset);
}

/**
 * @brief TILE, up to 9 rows: reduces one face (even and odd columns) in place.
 * @tparam values_face_offset Dest row of the face within the values tile.
 * @tparam indices_face_offset Dest row of the face within the indices tile.
 */
template <bool is_fp32_dest_acc_en, std::uint32_t values_face_offset, std::uint32_t indices_face_offset>
inline void _max_pool_with_indices_tile_face_() {
    constexpr InstrModLoadStore instr_mod_index = max_pool_instr_mod_index<is_fp32_dest_acc_en>;

    // data
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_face_offset + 0);  // even cols
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_face_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_face_offset + 2);  // odd cols
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_face_offset + 6);
    // index
    TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_face_offset + 0);  // even cols
    TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_face_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, indices_face_offset + 2);  // odd cols
    TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, indices_face_offset + 6);

    // sort 4 rows
    lltt::replay(0, 7);

    // data
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_face_offset + 8);   // even cols
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_face_offset + 10);  // odd cols
    // index
    TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, indices_face_offset + 8);   // even cols
    TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, indices_face_offset + 10);  // odd cols

    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_face_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, values_face_offset + 2);
    TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, indices_face_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, indices_face_offset + 2);
}

/**
 * @brief Calculates column-wise MaxPool of a tile, placing output values into the first row.
 *        Also places the index of the max value into the first row of the indices tile.
 *        Supports {FP32, FP16_B} for values, and {UINT16, INT32, UINT32} for indices, inferred from the Dest mode used.
 *        Can reduce up to 9 rows of a tile.
 * @tparam APPROXIMATION_MODE Whether to use the approximation mode (unused).
 * @tparam is_fp32_dest_acc_en Whether Dest is in 32bit mode (true) or 16bit mode (false).
 * @tparam ITERATIONS The number of iterations to perform (unused).
 * @tparam layout Data layout format, either TILE (default) or ROW_MAJOR.
 * @tparam accumulate Whether to accumulate results for large kernels (default is false).
 * @tparam values_tile_idx The index of the tile in the Dest register containing the data to be reduced.
 * @tparam indices_tile_idx The index of the tile in the Dest register containing the indices of the data.
 * @param chunk The chunk index for large kernel accumulation.
 */
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    int ITERATIONS,
    DataLayout layout,
    bool accumulate,
    std::uint32_t values_tile_idx,
    std::uint32_t indices_tile_idx>
inline void _calculate_max_pool_with_indices_([[maybe_unused]] const std::uint32_t chunk) {
    constexpr std::uint32_t values_tile_offset = values_tile_idx * max_pool_dst_tile_size;
    constexpr std::uint32_t indices_tile_offset = indices_tile_idx * max_pool_dst_tile_size;
    // each face is 16 rows
    constexpr std::uint32_t face_offset = 16;

    if constexpr (layout == ckernel::DataLayout::ROW_MAJOR) {
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

        // F0 + F1 even and odd cols
        constexpr std::uint32_t even_column_offset = 0;
        constexpr std::uint32_t odd_column_offset = 2;
        _max_pool_with_indices_row_major_columns_<
            is_fp32_dest_acc_en,
            values_tile_offset,
            indices_tile_offset,
            even_column_offset>();
        _max_pool_with_indices_row_major_columns_<
            is_fp32_dest_acc_en,
            values_tile_offset,
            indices_tile_offset,
            odd_column_offset>();
    } else {
        static_assert(!accumulate, "accumulate mode is not supported for TILE layout");
        // TILE (ORIGINAL) VERSION OF MPWI
        // F0
        _max_pool_with_indices_tile_face_<is_fp32_dest_acc_en, values_tile_offset, indices_tile_offset>();
        // F1
        _max_pool_with_indices_tile_face_<
            is_fp32_dest_acc_en,
            values_tile_offset + face_offset,
            indices_tile_offset + face_offset>();
    }
}

/**
 * @brief ROW_MAJOR, 32 rows: loads 3 further LREG pairs (6 rows) behind the running max in LREG0/LREG4 and folds
 *        them in with the replay buffer's first 3 slots (SWAP01, SWAP23, SWAP02).
 * @tparam val_base Dest row of the first value row pair.
 * @tparam idx_base Dest row of the first index row pair.
 */
template <bool is_fp32_dest_acc_en, std::uint32_t val_base, std::uint32_t idx_base>
inline void _max_pool_with_indices_fold_3_pairs_() {
    constexpr InstrModLoadStore instr_mod_index = max_pool_instr_mod_index<is_fp32_dest_acc_en>;

    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 0);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 4);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 8);
    TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, idx_base + 0);
    TTI_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, idx_base + 4);
    TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, idx_base + 8);
    lltt::replay(0, 3);
}

/**
 * @brief ROW_MAJOR, 32 rows: reduces one column parity (even or odd columns of both faces) into row 0 of the values and
 *        indices tiles, folding in the running max of the previous chunk when accumulating.
 *
 * Each SFPLOAD of 4 Dest rows holds 2 logical rows (both faces) of one column parity, so the 32 rows of a parity are
 * 16 LREG pairs. They are folded lane-wise into LREG0/LREG4 with the replay buffer's first 3 slots (SWAP01, SWAP23,
 * SWAP02: 4 pairs -> 1), loading 3 new pairs behind the running max for each further step. The 4 lane-rows (row r:
 * F0, F1, row r+1: F0, F1) are folded only once at the end with the replay buffer's last 4 slots (TRANSP, SWAP02,
 * SWAP13, TRANSP), leaving the per-face maxima in lane-rows 0/1, i.e. Dest rows 0/1 of the store.
 *
 * @tparam col_offset 0 for the even columns, 2 for the odd columns.
 * @param chunk The chunk index for large kernel accumulation.
 */
template <
    bool is_fp32_dest_acc_en,
    bool accumulate,
    std::uint32_t values_tile_offset,
    std::uint32_t indices_tile_offset,
    std::uint32_t col_offset>
inline void _max_pool_with_indices_generic_columns_([[maybe_unused]] const std::uint32_t chunk) {
    constexpr InstrModLoadStore instr_mod_index = max_pool_instr_mod_index<is_fp32_dest_acc_en>;
    constexpr std::uint32_t val_base = values_tile_offset + col_offset;
    constexpr std::uint32_t idx_base = indices_tile_offset + col_offset;

    // Rows 0-7: 4 pairs -> running max in LREG0/LREG4
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 0);   // Row 0 and 1
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 4);   // Row 2 and 3
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 8);   // Row 4 and 5
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 12);  // Row 6 and 7
    TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, idx_base + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, idx_base + 4);
    TTI_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, idx_base + 8);
    TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, idx_base + 12);
    lltt::replay(0, 3);

    // Rows 8-31: 3 pairs at a time behind the running max
    _max_pool_with_indices_fold_3_pairs_<is_fp32_dest_acc_en, val_base + 16, idx_base + 16>();  // Rows 8-13
    _max_pool_with_indices_fold_3_pairs_<is_fp32_dest_acc_en, val_base + 28, idx_base + 28>();  // Rows 14-19
    _max_pool_with_indices_fold_3_pairs_<is_fp32_dest_acc_en, val_base + 40, idx_base + 40>();  // Rows 20-25
    _max_pool_with_indices_fold_3_pairs_<is_fp32_dest_acc_en, val_base + 52, idx_base + 52>();  // Rows 26-31

    // Fold the 4 lane-rows: LREG0/LREG4 lane-rows 0/1 hold Max(R0-31) of face 0 / face 1
    lltt::replay(3, 4);

    if constexpr (accumulate) {
        // the running result lives in the tiles right above the operands
        constexpr std::uint32_t val_accum = val_base + max_pool_dst_tile_size;
        constexpr std::uint32_t idx_accum = idx_base + max_pool_dst_tile_size;
        if (chunk > 0) {  // for all but the first chunk we need to load the previous result from DST 1 and 3 and do
                          // a max with the current result in DST 0 and 2
            TTI_SFPLOAD(
                p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_accum);  // previous accumulated value
            TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, idx_accum);     // previous accumulated index
            TTI_SFPSWAP(
                0,
                p_sfpu::LREG0,
                p_sfpu::LREG1,
                p_sfpswap::ALL_ROWS_MAX);  // LREG0 contains max of current and previous value
        }
        // for each chunk we store the running result to DST 1 and 3
        TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, idx_accum);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_accum);
    }

    // store the final result to DST 0 (data) and DST 2 (indices)
    TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, idx_base);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base);
}

/**
 * @brief Calculates column-wise MaxPool of a tile, placing output values into the first row.
 *        Also places the index of the max value into the first row of the indices tile.
 *        Supports {FP32, FP16_B} for values, and {UINT16, INT32, UINT32} for indices, inferred from the Dest mode used.
 *        Can reduce up to 32 rows of a tile.
 * @tparam APPROXIMATION_MODE Whether to use the approximation mode (unused).
 * @tparam is_fp32_dest_acc_en Whether Dest is in 32bit mode (true) or 16bit mode (false).
 * @tparam ITERATIONS The number of iterations to use for the MaxPool operation (unused).
 * @tparam accumulate Whether to accumulate results for large kernels (default is false).
 * @tparam values_tile_idx The index of the tile in the Dest register containing the data to be reduced.
 * @tparam indices_tile_idx The index of the tile in the Dest register containing the indices of the data.
 * @param chunk The chunk index for large kernel accumulation.
 *
 * Note this function is only implemented for ROW_MAJOR data layout, so when _init_max_pool_with_indices_ is called
 * it must be called with layout=DataLayout::ROW_MAJOR.
 */
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    int ITERATIONS,
    bool accumulate,
    std::uint32_t values_tile_idx,
    std::uint32_t indices_tile_idx>
inline void _calculate_max_pool_with_indices_generic_(const std::uint32_t chunk) {
    constexpr std::uint32_t values_tile_offset = values_tile_idx * max_pool_dst_tile_size;
    constexpr std::uint32_t indices_tile_offset = indices_tile_idx * max_pool_dst_tile_size;

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

    // F0 + F1 even and odd cols
    constexpr std::uint32_t even_column_offset = 0;
    constexpr std::uint32_t odd_column_offset = 2;
    _max_pool_with_indices_generic_columns_<
        is_fp32_dest_acc_en,
        accumulate,
        values_tile_offset,
        indices_tile_offset,
        even_column_offset>(chunk);
    _max_pool_with_indices_generic_columns_<
        is_fp32_dest_acc_en,
        accumulate,
        values_tile_offset,
        indices_tile_offset,
        odd_column_offset>(chunk);
}

/**
 * @brief Entry point used by the compute API. The Dest tile indices are template parameters (see the note at the
 *        top of this file); the three leading runtime arguments only exist for the SFPU_BINARY_CALL calling
 *        convention, which passes and range-checks the same indices at runtime.
 */
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    int num_rows,
    int ITERATIONS,
    DataLayout layout,
    bool accumulate,
    std::uint32_t values_tile_idx,
    std::uint32_t indices_tile_idx>
inline void calculate_max_pool_with_indices(
    [[maybe_unused]] uint values_tile_idx_rt,
    [[maybe_unused]] uint indices_tile_idx_rt,
    [[maybe_unused]] uint unused_tile_idx,
    uint chunk) {
    if constexpr (num_rows <= 9) {
        _calculate_max_pool_with_indices_<
            APPROXIMATION_MODE,
            is_fp32_dest_acc_en,
            ITERATIONS,
            layout,
            accumulate,
            values_tile_idx,
            indices_tile_idx>(chunk);
    } else {
        static_assert(num_rows <= 32, "num_rows must be <= 32");
        static_assert(
            layout == DataLayout::ROW_MAJOR, "generic max pool with indices is only implemented for ROW_MAJOR layout");
        _calculate_max_pool_with_indices_generic_<
            APPROXIMATION_MODE,
            is_fp32_dest_acc_en,
            ITERATIONS,
            accumulate,
            values_tile_idx,
            indices_tile_idx>(chunk);
    }
}

template <bool APPROXIMATION_MODE, DataLayout layout = DataLayout::TILE>
inline void init_max_pool_with_indices() {
    // Set bit [2] of the SFPU_CONTROL_REG to enable Destination Index Tracking Mode:
    // LREGs 4-7 will be treated as indices corresponding to the values in LREGs 0-3,
    // and LREGs 4-7 will mirror the movement of the values in LREGs 0-3;
    _sfpu_load_config32_(0xF, 0x0, 0x4);

    if constexpr (layout == ckernel::DataLayout::ROW_MAJOR) {
        // Program replay buffer for row major layout
        lltt::record(0, 7);

        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);

        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);
    } else {
        // Program replay buffer for tiled layout (original)
        lltt::record(0, 7);

        // Values have been loaded such that 4 rows of Dest occupy the 4 lanes of each LREG
        // To sort those 4 rows, we transpose the SFPU LREGs to put elements of 4 rows of each column into separate
        // LREGs of each unit of SFPU
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

}  // namespace sfpu
}  // namespace ckernel
