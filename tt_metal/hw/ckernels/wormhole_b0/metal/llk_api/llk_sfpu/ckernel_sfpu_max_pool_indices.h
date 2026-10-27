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
// issue stream shrinks (9-row: 87 -> 29, 32-row: 283 -> 107 static instructions per call).

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
 * @brief ROW_MAJOR, 32 rows: reduces 8 rows (4 LREG pairs) to max in LREG0/LREG4, optionally stores the result.
 * @tparam val_base Dest row of the first value row pair.
 * @tparam idx_base Dest row of the first index row pair.
 * @tparam store_result Whether to store LREG0/LREG4 back to val_base/idx_base.
 */
template <bool is_fp32_dest_acc_en, std::uint32_t val_base, std::uint32_t idx_base, bool store_result>
inline void _max_pool_with_indices_reduce_8_rows_() {
    constexpr InstrModLoadStore instr_mod_index = max_pool_instr_mod_index<is_fp32_dest_acc_en>;

    // data
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 0);   // Row 0 and 1
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 4);   // Row 2 and 3
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 8);   // Row 4 and 5
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 12);  // Row 6 and 7
    // index
    TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, idx_base + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, idx_base + 4);
    TTI_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, idx_base + 8);
    TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, idx_base + 12);

    // Reduce 8 rows to max in LREG0/LREG4 via replay buffer
    lltt::replay(0, 7);

    // Only store when necessary - caller controls this
    if constexpr (store_result) {
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base + 0);
        TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, idx_base + 0);
    }
    // Result: Max of 8 rows in LREG0 (values) and LREG4 (indices)
}

/**
 * @brief ROW_MAJOR, 32 rows: reduces 16 rows of one column parity to max in LREG0/LREG4.
 *        After reducing the second 8-row block, the result is already in LREG0/LREG4; only the first block's
 *        result is reloaded into LREG1/LREG5 for the final swap.
 * @tparam base_offset Dest row of the first of the 16 rows within the tile (0 or 32).
 * @tparam col_offset 0 for the even columns, 2 for the odd columns.
 * @tparam store_result If false, the result stays in LREG0/LREG4 for the caller to use directly.
 */
template <
    bool is_fp32_dest_acc_en,
    std::uint32_t values_tile_offset,
    std::uint32_t indices_tile_offset,
    std::uint32_t base_offset,
    std::uint32_t col_offset,
    bool store_result>
inline void _max_pool_with_indices_process_16_rows_() {
    constexpr InstrModLoadStore instr_mod_index = max_pool_instr_mod_index<is_fp32_dest_acc_en>;
    constexpr std::uint32_t eight_row_offset = 16;

    constexpr std::uint32_t val_base_first = values_tile_offset + base_offset + col_offset;
    constexpr std::uint32_t idx_base_first = indices_tile_offset + base_offset + col_offset;
    constexpr std::uint32_t val_base_second = values_tile_offset + eight_row_offset + base_offset + col_offset;
    constexpr std::uint32_t idx_base_second = indices_tile_offset + eight_row_offset + base_offset + col_offset;

    // First 8 rows: reduce and STORE (we need to free registers for second block)
    _max_pool_with_indices_reduce_8_rows_<is_fp32_dest_acc_en, val_base_first, idx_base_first, true>();

    // Second 8 rows: reduce but DON'T STORE - keep result in LREG0/LREG4
    _max_pool_with_indices_reduce_8_rows_<is_fp32_dest_acc_en, val_base_second, idx_base_second, false>();

    // Now: LREG0/LREG4 contains Max(R8-15), need to load Max(R0-7) into LREG1/LREG5
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base_first);
    TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, idx_base_first);

    // Swap to get Max(R0-15) in LREG0/LREG4
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

    // Only store if needed - caller may use LREG0/LREG4 directly
    if constexpr (store_result) {
        TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, idx_base_first);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_base_first);
    }
}

/**
 * @brief ROW_MAJOR, 32 rows: combines Max(R16-31), already in LREG0/LREG4, with the stored Max(R0-15) of one
 *        column parity, folds in the running max of the previous chunk when accumulating, and stores the result.
 * @tparam col_offset 0 for the even columns, 2 for the odd columns.
 * @param chunk The chunk index for large kernel accumulation.
 */
template <
    bool is_fp32_dest_acc_en,
    bool accumulate,
    std::uint32_t values_tile_offset,
    std::uint32_t indices_tile_offset,
    std::uint32_t col_offset>
inline void _max_pool_with_indices_final_swap_([[maybe_unused]] const std::uint32_t chunk) {
    constexpr InstrModLoadStore instr_mod_index = max_pool_instr_mod_index<is_fp32_dest_acc_en>;

    constexpr std::uint32_t val_first = values_tile_offset + col_offset;
    constexpr std::uint32_t idx_first = indices_tile_offset + col_offset;

    // LREG0/LREG4 already contains Max(R16-31) from the previous process_16_rows call
    // Only need to load Max(R0-15) into LREG1/LREG5
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_first);  // Max(R0-15) for F0,1
    TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, idx_first);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);  // LREG0 contains Max(R0-31) for F0,1

    if constexpr (accumulate) {
        // the running result lives in the tiles right above the operands
        constexpr std::uint32_t val_accum = values_tile_offset + max_pool_dst_tile_size + col_offset;
        constexpr std::uint32_t idx_accum = indices_tile_offset + max_pool_dst_tile_size + col_offset;
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
    TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, idx_first);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, val_first);
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
    constexpr std::uint32_t sixteen_row_offset = 32;

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

    // OPTIMIZATION: Process each column completely before moving to the next.
    // This allows the second process_16_rows to leave Max(R16-31) in LREG0/LREG4,
    // which final_swap can use directly without reloading.
    // Saves 2 stores + 2 loads per column = 4 stores + 4 loads total.
    constexpr std::uint32_t even_column_offset = 0;
    constexpr std::uint32_t odd_column_offset = 2;

    // Even columns: process rows 0-15, then 16-31, then final swap
    _max_pool_with_indices_process_16_rows_<
        is_fp32_dest_acc_en,
        values_tile_offset,
        indices_tile_offset,
        0,
        even_column_offset,
        true>();  // Store Max(R0-15) for final_swap to load
    _max_pool_with_indices_process_16_rows_<
        is_fp32_dest_acc_en,
        values_tile_offset,
        indices_tile_offset,
        sixteen_row_offset,
        even_column_offset,
        false>();  // Keep Max(R16-31) in LREG0/LREG4
    _max_pool_with_indices_final_swap_<
        is_fp32_dest_acc_en,
        accumulate,
        values_tile_offset,
        indices_tile_offset,
        even_column_offset>(chunk);  // Uses LREG0/LREG4 directly

    // Odd columns: process rows 0-15, then 16-31, then final swap
    _max_pool_with_indices_process_16_rows_<
        is_fp32_dest_acc_en,
        values_tile_offset,
        indices_tile_offset,
        0,
        odd_column_offset,
        true>();  // Store Max(R0-15) for final_swap to load
    _max_pool_with_indices_process_16_rows_<
        is_fp32_dest_acc_en,
        values_tile_offset,
        indices_tile_offset,
        sixteen_row_offset,
        odd_column_offset,
        false>();  // Keep Max(R16-31) in LREG0/LREG4
    _max_pool_with_indices_final_swap_<
        is_fp32_dest_acc_en,
        accumulate,
        values_tile_offset,
        indices_tile_offset,
        odd_column_offset>(chunk);  // Uses LREG0/LREG4 directly
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
