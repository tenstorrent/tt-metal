// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel {
namespace sfpu {

// Dest geometry (address units): one SFPLOAD covers 4 units x 8 columns, and address bit 1 -
// p_sfpu::col_offset::ODD_COL - picks the odd 8 columns of each. Face f holds units 16f to 16f+15.
constexpr std::uint32_t MAX_POOL_DEST_TILE_SIZE = TILE_NUM_FACES * FACE_R_DIM;  // units per 32x32 tile
constexpr std::uint32_t MAX_POOL_FACE_OFFSET = FACE_R_DIM;                      // units per 16x16 face

// ROW_MAJOR: a logical tile row spans one unit of face 0 and one of face 1, so a row block of the
// walk is that many units long.
constexpr std::uint32_t MAX_POOL_UNITS_PER_ROW = 2;
constexpr std::uint32_t MAX_POOL_EIGHT_ROW_OFFSET = 8 * MAX_POOL_UNITS_PER_ROW;
constexpr std::uint32_t MAX_POOL_SIXTEEN_ROW_OFFSET = 16 * MAX_POOL_UNITS_PER_ROW;

constexpr std::uint32_t SFPU_CTRL_INDEX_TRACKING = 0x4;  // Control Register bit 2 = INDEX_TRACKING_ENABLE
constexpr std::uint32_t MAX_POOL_SWAP_IMM12_FP32 = 0x1;  // SFPSWAP imm12 bit 0 = FP32 compare

// Replay slots for the sort network init_max_pool_with_indices() records for its layout:
// max_pool_sort_tile_() or max_pool_sort_row_major_(), both 2 x SFPTRANSP + 5 x SFPSWAP.
constexpr std::uint32_t MAX_POOL_SORT_START = 0;
constexpr std::uint32_t MAX_POOL_SORT_LEN = 7;
constexpr std::uint32_t MAX_POOL_FOLD_TILE_START = 5;  // TILE only: its last two swaps, LREG0/LREG1 and LREG2/LREG3
constexpr std::uint32_t MAX_POOL_FOLD_TILE_LEN = 2;

/**
 * @brief Compare-exchange one LREG pair so the larger value ends up in VC, the smaller in VD.
 *
 * @tparam VC: Value LREG that receives the larger of the pair, values = <LREG0-LREG3>
 * @tparam VD: Value LREG that receives the smaller of the pair, values = <LREG0-LREG3>
 * @note Index tracking makes LREG[VC+4] / LREG[VD+4] follow the exchange, which is what carries the
 *       indices alongside the values. Enable it with @ref init_max_pool_with_indices first.
 * @note No SFPNOP follows: SFPSWAP always stalls the pipeline on the next cycle, so the next
 *       instruction cannot observe a half-written result (TEN-4581 lists SFPSWAP as safe).
 */
template <std::uint32_t VC, std::uint32_t VD>
inline __attribute__((always_inline)) void max_pool_swap_() {
    TTI_SFPSWAP(MAX_POOL_SWAP_IMM12_FP32, VC, VD, p_sfpswap::ALL_ROWS_MAX);
}

/**
 * @brief Reduce the 8 TILE-layout rows held in LREG0-3 down to LREG0 (even cols) and LREG2 (odd cols).
 *
 * Each of LREG0-3 arrives holding 4 tile rows of 8 columns. SFPTRANSP redistributes a bank so those
 * 4 rows sit in 4 separate registers, a 3-swap network maxes them, and the second SFPTRANSP restores
 * store order - after which two more swaps fold the 4-row groups together.
 *
 * @note This is the TILE-layout body @ref init_max_pool_with_indices records into the replay buffer,
 *       so its instruction count must stay MAX_POOL_SORT_LEN and its last 2 instructions must stay
 *       the two swaps MAX_POOL_FOLD_TILE_START names.
 */
inline __attribute__((always_inline)) void max_pool_sort_tile_() {
    TTI_SFPTRANSP;  // 4 Dest rows of each column -> 4 separate LREGs
    max_pool_swap_<p_sfpu::LREG0, p_sfpu::LREG1>();
    max_pool_swap_<p_sfpu::LREG2, p_sfpu::LREG3>();
    max_pool_swap_<p_sfpu::LREG0, p_sfpu::LREG2>();  // max of the 4 rows -> LREG0 / LREG4
    TTI_SFPTRANSP;                                   // transpose back
    max_pool_swap_<p_sfpu::LREG0, p_sfpu::LREG1>();
    max_pool_swap_<p_sfpu::LREG2, p_sfpu::LREG3>();
}

/**
 * @brief Reduce the 8 ROW_MAJOR rows held in LREG0-3 down to LREG0 / LREG4.
 *
 * Each of LREG0-3 arrives holding one pair of logical rows. Three swaps max the four registers
 * against each other, and the transpose-swap-transpose tail then folds the two rows that share a
 * register.
 *
 * @note This is the ROW_MAJOR body @ref init_max_pool_with_indices records into the replay buffer,
 *       so its instruction count must stay MAX_POOL_SORT_LEN.
 */
inline __attribute__((always_inline)) void max_pool_sort_row_major_() {
    max_pool_swap_<p_sfpu::LREG0, p_sfpu::LREG1>();
    max_pool_swap_<p_sfpu::LREG2, p_sfpu::LREG3>();
    max_pool_swap_<p_sfpu::LREG0, p_sfpu::LREG2>();
    TTI_SFPTRANSP;
    max_pool_swap_<p_sfpu::LREG0, p_sfpu::LREG2>();
    max_pool_swap_<p_sfpu::LREG1, p_sfpu::LREG3>();
    TTI_SFPTRANSP;
}

/**
 * @brief Reduce rows 0-8 of one TILE-layout face into its row 0, values and indices in place.
 *
 * @tparam is_fp32_dest_acc_en: Whether Dest holds 32-bit datums, which selects the load/store modes
 * @param v: Dest address of the face in the values tile
 * @param i: Dest address of the face in the indices tile
 * @note Replays the TILE sort network, so @ref init_max_pool_with_indices<TILE> must have recorded it.
 */
template <bool is_fp32_dest_acc_en>
inline __attribute__((always_inline)) void max_pool_reduce_tile_face_(const std::uint32_t v, const std::uint32_t i) {
    constexpr std::uint32_t val_mode = is_fp32_dest_acc_en ? p_sfpu::sfpmem::FP32 : p_sfpu::sfpmem::FP16B;
    constexpr std::uint32_t idx_mode = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::UINT16;

    TT_SFPLOAD(p_sfpu::LREG0, val_mode, ADDR_MOD_7, 0 /* done */, v + 0);  // rows 0-3, even cols
    TT_SFPLOAD(p_sfpu::LREG1, val_mode, ADDR_MOD_7, 0 /* done */, v + 4);  // rows 4-7, even cols
    TT_SFPLOAD(p_sfpu::LREG2, val_mode, ADDR_MOD_7, 0 /* done */, v + 2);  // rows 0-3, odd cols
    TT_SFPLOAD(p_sfpu::LREG3, val_mode, ADDR_MOD_7, 0 /* done */, v + 6);  // rows 4-7, odd cols
    TT_SFPLOAD(p_sfpu::LREG4, idx_mode, ADDR_MOD_7, 0 /* done */, i + 0);
    TT_SFPLOAD(p_sfpu::LREG5, idx_mode, ADDR_MOD_7, 0 /* done */, i + 4);
    TT_SFPLOAD(p_sfpu::LREG6, idx_mode, ADDR_MOD_7, 0 /* done */, i + 2);
    TT_SFPLOAD(p_sfpu::LREG7, idx_mode, ADDR_MOD_7, 0 /* done */, i + 6);

    // max of rows 0-7: even cols in LREG0, odd cols in LREG2
    TTI_REPLAY(
        MAX_POOL_SORT_START,
        MAX_POOL_SORT_LEN,
        0 /* last */,
        0 /* set_mutex */,
        0 /* execute_while_loading */,
        0 /* load_mode */);

    TT_SFPLOAD(p_sfpu::LREG1, val_mode, ADDR_MOD_7, 0 /* done */, v + 8);   // row 8, even cols
    TT_SFPLOAD(p_sfpu::LREG3, val_mode, ADDR_MOD_7, 0 /* done */, v + 10);  // row 8, odd cols
    TT_SFPLOAD(p_sfpu::LREG5, idx_mode, ADDR_MOD_7, 0 /* done */, i + 8);
    TT_SFPLOAD(p_sfpu::LREG7, idx_mode, ADDR_MOD_7, 0 /* done */, i + 10);

    // fold in row 8
    TTI_REPLAY(
        MAX_POOL_FOLD_TILE_START,
        MAX_POOL_FOLD_TILE_LEN,
        0 /* last */,
        0 /* set_mutex */,
        0 /* execute_while_loading */,
        0 /* load_mode */);

    TT_SFPSTORE(p_sfpu::LREG0, val_mode, ADDR_MOD_7, 0 /* done */, v + 0);
    TT_SFPSTORE(p_sfpu::LREG2, val_mode, ADDR_MOD_7, 0 /* done */, v + 2);
    TT_SFPSTORE(p_sfpu::LREG4, idx_mode, ADDR_MOD_7, 0 /* done */, i + 0);
    TT_SFPSTORE(p_sfpu::LREG6, idx_mode, ADDR_MOD_7, 0 /* done */, i + 2);
}

/**
 * @brief Reduce 9 ROW_MAJOR rows into row 0 of one column parity, values and indices in place.
 *
 * @tparam is_fp32_dest_acc_en: Whether Dest holds 32-bit datums, which selects the load/store modes
 * @param v: Dest address of the values tile, already offset to this column parity
 * @param i: Dest address of the indices tile, already offset to this column parity
 * @note Replays the ROW_MAJOR sort network, so @ref init_max_pool_with_indices<ROW_MAJOR> must have
 *       recorded it.
 */
template <bool is_fp32_dest_acc_en>
inline __attribute__((always_inline)) void max_pool_reduce_row_major_9_(const std::uint32_t v, const std::uint32_t i) {
    constexpr std::uint32_t val_mode = is_fp32_dest_acc_en ? p_sfpu::sfpmem::FP32 : p_sfpu::sfpmem::FP16B;
    constexpr std::uint32_t idx_mode = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::UINT16;

    TT_SFPLOAD(p_sfpu::LREG0, val_mode, ADDR_MOD_7, 0 /* done */, v + 0);   // rows 0-1
    TT_SFPLOAD(p_sfpu::LREG1, val_mode, ADDR_MOD_7, 0 /* done */, v + 4);   // rows 2-3
    TT_SFPLOAD(p_sfpu::LREG2, val_mode, ADDR_MOD_7, 0 /* done */, v + 8);   // rows 4-5
    TT_SFPLOAD(p_sfpu::LREG3, val_mode, ADDR_MOD_7, 0 /* done */, v + 12);  // rows 6-7
    TT_SFPLOAD(p_sfpu::LREG4, idx_mode, ADDR_MOD_7, 0 /* done */, i + 0);
    TT_SFPLOAD(p_sfpu::LREG5, idx_mode, ADDR_MOD_7, 0 /* done */, i + 4);
    TT_SFPLOAD(p_sfpu::LREG6, idx_mode, ADDR_MOD_7, 0 /* done */, i + 8);
    TT_SFPLOAD(p_sfpu::LREG7, idx_mode, ADDR_MOD_7, 0 /* done */, i + 12);

    // max of the 8 rows -> LREG0 / LREG4
    TTI_REPLAY(
        MAX_POOL_SORT_START,
        MAX_POOL_SORT_LEN,
        0 /* last */,
        0 /* set_mutex */,
        0 /* execute_while_loading */,
        0 /* load_mode */);

    TT_SFPLOAD(p_sfpu::LREG1, val_mode, ADDR_MOD_7, 0 /* done */, v + 16);  // row 8
    TT_SFPLOAD(p_sfpu::LREG5, idx_mode, ADDR_MOD_7, 0 /* done */, i + 16);

    max_pool_swap_<p_sfpu::LREG0, p_sfpu::LREG1>();  // fold in row 8

    TT_SFPSTORE(p_sfpu::LREG4, idx_mode, ADDR_MOD_7, 0 /* done */, i + 0);
    TT_SFPSTORE(p_sfpu::LREG0, val_mode, ADDR_MOD_7, 0 /* done */, v + 0);
}

/**
 * @brief Reduce 8 ROW_MAJOR rows into LREG0 / LREG4, optionally storing the result at the block base.
 *
 * @tparam is_fp32_dest_acc_en: Whether Dest holds 32-bit datums, which selects the load/store modes
 * @tparam store: Whether to write the block max back to Dest; leave it false to keep the result in
 *         LREG0 / LREG4 for the caller to fold into the next block
 * @param vb: Dest address of this row block in the values tile
 * @param ib: Dest address of this row block in the indices tile
 * @note Replays the ROW_MAJOR sort network, so @ref init_max_pool_with_indices<ROW_MAJOR> must have
 *       recorded it.
 */
template <bool is_fp32_dest_acc_en, bool store>
inline __attribute__((always_inline)) void max_pool_reduce_8_rows_(const std::uint32_t vb, const std::uint32_t ib) {
    constexpr std::uint32_t val_mode = is_fp32_dest_acc_en ? p_sfpu::sfpmem::FP32 : p_sfpu::sfpmem::FP16B;
    constexpr std::uint32_t idx_mode = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::UINT16;

    TT_SFPLOAD(p_sfpu::LREG0, val_mode, ADDR_MOD_7, 0 /* done */, vb + 0);   // rows 0-1
    TT_SFPLOAD(p_sfpu::LREG1, val_mode, ADDR_MOD_7, 0 /* done */, vb + 4);   // rows 2-3
    TT_SFPLOAD(p_sfpu::LREG2, val_mode, ADDR_MOD_7, 0 /* done */, vb + 8);   // rows 4-5
    TT_SFPLOAD(p_sfpu::LREG3, val_mode, ADDR_MOD_7, 0 /* done */, vb + 12);  // rows 6-7
    TT_SFPLOAD(p_sfpu::LREG4, idx_mode, ADDR_MOD_7, 0 /* done */, ib + 0);
    TT_SFPLOAD(p_sfpu::LREG5, idx_mode, ADDR_MOD_7, 0 /* done */, ib + 4);
    TT_SFPLOAD(p_sfpu::LREG6, idx_mode, ADDR_MOD_7, 0 /* done */, ib + 8);
    TT_SFPLOAD(p_sfpu::LREG7, idx_mode, ADDR_MOD_7, 0 /* done */, ib + 12);

    // max of the 8 rows -> LREG0 / LREG4
    TTI_REPLAY(
        MAX_POOL_SORT_START,
        MAX_POOL_SORT_LEN,
        0 /* last */,
        0 /* set_mutex */,
        0 /* execute_while_loading */,
        0 /* load_mode */);

    if constexpr (store) {
        TT_SFPSTORE(p_sfpu::LREG0, val_mode, ADDR_MOD_7, 0 /* done */, vb);
        TT_SFPSTORE(p_sfpu::LREG4, idx_mode, ADDR_MOD_7, 0 /* done */, ib);
    }
}

/**
 * @brief Reduce the 16 ROW_MAJOR rows at `base` into LREG0 / LREG4 as two 8-row blocks.
 *
 * @tparam is_fp32_dest_acc_en: Whether Dest holds 32-bit datums, which selects the load/store modes
 * @tparam store: Whether to write the 16-row max back to the first block; leave it false to keep the
 *         result in LREG0 / LREG4 for the caller to fold into
 * @param v: Dest address of the values tile
 * @param i: Dest address of the indices tile
 * @param base: Offset of the 16-row span within the tile
 * @param col: Column-parity offset, values = <p_sfpu::col_offset::EVEN_COL/ODD_COL>
 * @note The first block's max is parked in Dest rather than an LREG because the second block's
 *       reduction needs every one of LREG0-7.
 */
template <bool is_fp32_dest_acc_en, bool store>
inline __attribute__((always_inline)) void max_pool_process_16_rows_(
    const std::uint32_t v, const std::uint32_t i, const std::uint32_t base, const std::uint32_t col) {
    constexpr std::uint32_t val_mode = is_fp32_dest_acc_en ? p_sfpu::sfpmem::FP32 : p_sfpu::sfpmem::FP16B;
    constexpr std::uint32_t idx_mode = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::UINT16;

    const std::uint32_t b1 = base + col;
    const std::uint32_t b2 = base + MAX_POOL_EIGHT_ROW_OFFSET + col;

    max_pool_reduce_8_rows_<is_fp32_dest_acc_en, true /* store */>(v + b1, i + b1);
    // max of second block stays in LREG0 / LREG4
    max_pool_reduce_8_rows_<is_fp32_dest_acc_en, false /* store */>(v + b2, i + b2);

    TT_SFPLOAD(p_sfpu::LREG1, val_mode, ADDR_MOD_7, 0 /* done */, v + b1);  // max of first block
    TT_SFPLOAD(p_sfpu::LREG5, idx_mode, ADDR_MOD_7, 0 /* done */, i + b1);
    max_pool_swap_<p_sfpu::LREG0, p_sfpu::LREG1>();

    if constexpr (store) {
        TT_SFPSTORE(p_sfpu::LREG4, idx_mode, ADDR_MOD_7, 0 /* done */, i + b1);
        TT_SFPSTORE(p_sfpu::LREG0, val_mode, ADDR_MOD_7, 0 /* done */, v + b1);
    }
}

/**
 * @brief Combine rows 0-15 (parked at row 0) with rows 16-31 (in LREG0 / LREG4) and store row 0.
 *
 * @tparam is_fp32_dest_acc_en: Whether Dest holds 32-bit datums, which selects the load/store modes
 * @tparam accumulate: Whether to also carry a running max across calls in the tile pair above the
 *         operands - Dest tiles values_tile_idx + 1 and indices_tile_idx + 1
 * @param values_tile_idx: Dest tile index of the values operand
 * @param indices_tile_idx: Dest tile index of the indices operand
 * @param v: Dest address of the values tile
 * @param i: Dest address of the indices tile
 * @param col: Column-parity offset, values = <p_sfpu::col_offset::EVEN_COL/ODD_COL>
 * @param chunk: Index of this call in the accumulation chain; chunk 0 seeds the running max instead
 *         of folding into it. Unused unless accumulate is set.
 */
template <bool is_fp32_dest_acc_en, bool accumulate>
inline __attribute__((always_inline)) void max_pool_final_swap_(
    const std::uint32_t values_tile_idx,
    const std::uint32_t indices_tile_idx,
    const std::uint32_t v,
    const std::uint32_t i,
    const std::uint32_t col,
    [[maybe_unused]] const std::uint32_t chunk) {
    constexpr std::uint32_t val_mode = is_fp32_dest_acc_en ? p_sfpu::sfpmem::FP32 : p_sfpu::sfpmem::FP16B;
    constexpr std::uint32_t idx_mode = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::UINT16;

    TT_SFPLOAD(p_sfpu::LREG1, val_mode, ADDR_MOD_7, 0 /* done */, v + col);  // max of rows 0-15
    TT_SFPLOAD(p_sfpu::LREG5, idx_mode, ADDR_MOD_7, 0 /* done */, i + col);
    max_pool_swap_<p_sfpu::LREG0, p_sfpu::LREG1>();

    if constexpr (accumulate) {
        const std::uint32_t va = (values_tile_idx + 1) * MAX_POOL_DEST_TILE_SIZE + col;
        const std::uint32_t ia = (indices_tile_idx + 1) * MAX_POOL_DEST_TILE_SIZE + col;
        if (chunk > 0) {
            TT_SFPLOAD(p_sfpu::LREG1, val_mode, ADDR_MOD_7, 0 /* done */, va);  // previous running max
            TT_SFPLOAD(p_sfpu::LREG5, idx_mode, ADDR_MOD_7, 0 /* done */, ia);
            max_pool_swap_<p_sfpu::LREG0, p_sfpu::LREG1>();
        }
        TT_SFPSTORE(p_sfpu::LREG4, idx_mode, ADDR_MOD_7, 0 /* done */, ia);  // running result
        TT_SFPSTORE(p_sfpu::LREG0, val_mode, ADDR_MOD_7, 0 /* done */, va);
    }

    TT_SFPSTORE(p_sfpu::LREG4, idx_mode, ADDR_MOD_7, 0 /* done */, i + col);
    TT_SFPSTORE(p_sfpu::LREG0, val_mode, ADDR_MOD_7, 0 /* done */, v + col);
}

/**
 * @brief Column-wise arg-max over rows 0-8 of a Dest tile pair, reduced in place into row 0.
 *
 * @tparam is_fp32_dest_acc_en: Whether Dest holds 32-bit datums, which selects the load/store modes
 * @tparam layout: How the tile's rows sit in Dest, values = <TILE/ROW_MAJOR>
 * @tparam accumulate: Unsupported on the 9-row path, so it must be false
 * @param values_tile_idx: Dest tile index of the values operand
 * @param indices_tile_idx: Dest tile index of the indices operand
 * @param chunk: Index of this call in the accumulation chain. Unused on this path.
 * @note Faces 2 and 3 of the TILE-layout tiles are neither read nor written, and every row but row 0
 *       is scratch on return.
 */
template <bool is_fp32_dest_acc_en, ckernel::DataLayout layout, bool accumulate>
inline void _calculate_max_pool_with_indices_(
    const std::uint32_t values_tile_idx,
    const std::uint32_t indices_tile_idx,
    [[maybe_unused]] const std::uint32_t chunk) {
    const std::uint32_t v = values_tile_idx * MAX_POOL_DEST_TILE_SIZE;
    const std::uint32_t i = indices_tile_idx * MAX_POOL_DEST_TILE_SIZE;

    static_assert(!accumulate, "accumulate is only implemented for the 32-row ROW_MAJOR path (num_rows > 9)");

    if constexpr (layout == ckernel::DataLayout::ROW_MAJOR) {
        max_pool_reduce_row_major_9_<is_fp32_dest_acc_en>(
            v + p_sfpu::col_offset::EVEN_COL, i + p_sfpu::col_offset::EVEN_COL);
        max_pool_reduce_row_major_9_<is_fp32_dest_acc_en>(
            v + p_sfpu::col_offset::ODD_COL, i + p_sfpu::col_offset::ODD_COL);
    } else {
        max_pool_reduce_tile_face_<is_fp32_dest_acc_en>(v, i);
        max_pool_reduce_tile_face_<is_fp32_dest_acc_en>(v + MAX_POOL_FACE_OFFSET, i + MAX_POOL_FACE_OFFSET);
    }
}

/**
 * @brief Column-wise arg-max over all 32 ROW_MAJOR rows of a Dest tile pair, reduced into row 0.
 *
 * Runs the 16-row walk twice per column parity - rows 0-15, then rows 16-31 - and folds the two
 * halves together.
 *
 * @tparam is_fp32_dest_acc_en: Whether Dest holds 32-bit datums, which selects the load/store modes
 * @tparam accumulate: Whether to carry a running max across calls in the tile pair above the
 *         operands - Dest tiles values_tile_idx + 1 and indices_tile_idx + 1
 * @param values_tile_idx: Dest tile index of the values operand
 * @param indices_tile_idx: Dest tile index of the indices operand
 * @param chunk: Index of this call in the accumulation chain; chunk 0 seeds the running max instead
 *         of folding into it. Unused unless accumulate is set.
 * @note Every row but row 0 of both operand tiles is scratch on return.
 */
template <bool is_fp32_dest_acc_en, bool accumulate>
inline void _calculate_max_pool_with_indices_generic_(
    const std::uint32_t values_tile_idx, const std::uint32_t indices_tile_idx, const std::uint32_t chunk) {
    const std::uint32_t v = values_tile_idx * MAX_POOL_DEST_TILE_SIZE;
    const std::uint32_t i = indices_tile_idx * MAX_POOL_DEST_TILE_SIZE;

    constexpr std::uint32_t EVEN = p_sfpu::col_offset::EVEN_COL;
    constexpr std::uint32_t ODD = p_sfpu::col_offset::ODD_COL;
    constexpr std::uint32_t FIRST_16_ROWS = 0;

    // even columns; the max of rows 0-15 is parked at row 0 for the second half to fold in
    max_pool_process_16_rows_<is_fp32_dest_acc_en, true /* store */>(v, i, FIRST_16_ROWS, EVEN);
    max_pool_process_16_rows_<is_fp32_dest_acc_en, false /* store */>(v, i, MAX_POOL_SIXTEEN_ROW_OFFSET, EVEN);
    max_pool_final_swap_<is_fp32_dest_acc_en, accumulate>(values_tile_idx, indices_tile_idx, v, i, EVEN, chunk);

    // odd columns
    max_pool_process_16_rows_<is_fp32_dest_acc_en, true /* store */>(v, i, FIRST_16_ROWS, ODD);
    max_pool_process_16_rows_<is_fp32_dest_acc_en, false /* store */>(v, i, MAX_POOL_SIXTEEN_ROW_OFFSET, ODD);
    max_pool_final_swap_<is_fp32_dest_acc_en, accumulate>(values_tile_idx, indices_tile_idx, v, i, ODD, chunk);
}

/**
 * @brief Enable SFPU index tracking and record the layout's sort network into the replay buffer.
 *
 * @tparam APPROXIMATION_MODE: Unused; the reduction is exact.
 * @tparam layout: Layout whose sort network to record, values = <TILE/ROW_MAJOR>; must match the
 *         layout passed to @ref calculate_max_pool_with_indices
 * @note Call this before @ref calculate_max_pool_with_indices, and after
 *       @ref _llk_math_eltwise_sfpu_init_ - that rewrites the whole SFPU Control Register, so
 *       running it afterwards would clear index tracking again.
 * @note Claims math-thread replay slots MAX_POOL_SORT_START to MAX_POOL_SORT_START +
 *       MAX_POOL_SORT_LEN - 1. Call it again after any other math-thread op that records into the
 *       replay buffer (eltwise binary, reduce, matmul, transpose, binary max/min, ...) and before the
 *       next @ref calculate_max_pool_with_indices.
 */
template <bool APPROXIMATION_MODE, ckernel::DataLayout layout = ckernel::DataLayout::TILE>
inline void init_max_pool_with_indices() {
    // LREG4-7 become the indices of LREG0-3 and follow every SFPSWAP exchange
    ckernel::math::_sfpu_load_config32_(p_sfpconfig::SFPU_CTRL, 0x0 /* upper16 */, SFPU_CTRL_INDEX_TRACKING);
    // Let the SFPU_CTRL write land before the first SFPSWAP relies on index tracking
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);

    if constexpr (layout == ckernel::DataLayout::ROW_MAJOR) {
        load_replay_buf<MAX_POOL_SORT_START, MAX_POOL_SORT_LEN>([] { max_pool_sort_row_major_(); });
    } else {
        load_replay_buf<MAX_POOL_SORT_START, MAX_POOL_SORT_LEN>([] { max_pool_sort_tile_(); });
    }
}

/**
 * @brief Column-wise arg-max over rows 0-8 or rows 0-31 of a Dest tile pair, in place into row 0.
 *
 * For every column, writes the maximum value into row 0 of the values tile and the entry that
 * travelled with it into row 0 of the indices tile. num_rows only selects between two fixed
 * networks: up to 9 reduces exactly rows 0-8, 10 to 32 reduces all rows 0-31. Rows past the
 * requested count but inside the selected network are reduced too, so pad them with a value no
 * larger than any real one (e.g. -inf).
 *
 * @tparam APPROXIMATION_MODE: Unused; the reduction is exact.
 * @tparam is_fp32_dest_acc_en: Whether Dest holds 32-bit datums, which selects the load/store modes
 * @tparam num_rows: 9-versus-32 network selector, values = <1-9 (rows 0-8), 10-32 (rows 0-31)>.
 *         The 32-row network requires ROW_MAJOR layout.
 * @tparam ITERATIONS: Unused; one call covers the whole tile.
 * @tparam layout: How the tile's rows sit in Dest, values = <TILE/ROW_MAJOR>
 * @tparam accumulate: Whether to carry a running max across calls in the tile pair above the
 *         operands; ROW_MAJOR and num_rows > 9 only (static_assert)
 * @param values_tile_idx: Dest tile index of the values operand
 * @param indices_tile_idx: Dest tile index of the indices operand
 * @param unused_tile_idx: Unused; the reduction writes back over its operands.
 * @param chunk: Index of this call in the accumulation chain; chunk 0 seeds the running max instead
 *         of folding into it. Unused unless accumulate is set.
 * @note Call @ref init_max_pool_with_indices with the same layout before this - it enables index
 *       tracking (without it the swaps move values without their indices) and records the sort
 *       network this replays.
 * @note Run this once per tile under VectorMode::None, not once per face: it addresses the whole
 *       tile itself, and every row but row 0 is scratch on return.
 */
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    int num_rows,
    int ITERATIONS = 8,
    ckernel::DataLayout layout = ckernel::DataLayout::TILE,
    bool accumulate = false>
inline void calculate_max_pool_with_indices(
    const std::uint32_t values_tile_idx,
    const std::uint32_t indices_tile_idx,
    [[maybe_unused]] const std::uint32_t unused_tile_idx,
    const std::uint32_t chunk) {
    if constexpr (num_rows <= 9) {
        _calculate_max_pool_with_indices_<is_fp32_dest_acc_en, layout, accumulate>(
            values_tile_idx, indices_tile_idx, chunk);
    } else {
        static_assert(num_rows <= 32, "num_rows must be <= 32");
        static_assert(
            layout == ckernel::DataLayout::ROW_MAJOR,
            "generic max pool with indices is only implemented for ROW_MAJOR layout");
        _calculate_max_pool_with_indices_generic_<is_fp32_dest_acc_en, accumulate>(
            values_tile_idx, indices_tile_idx, chunk);
    }
}

}  // namespace sfpu
}  // namespace ckernel
