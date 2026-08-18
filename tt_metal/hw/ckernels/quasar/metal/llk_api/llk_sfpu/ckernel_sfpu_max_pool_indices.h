// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// LaneConfig bit [2]: SFPSWAP mirrors every value swap in LREG0-3 onto LREG4-7 (argmax).
// Bit [3] (CAPTURE_DEFAULT_DEST_INDEX) must stay clear, or SFPLOAD would overwrite the loaded indices.
constexpr std::uint32_t LANECFG_ENABLE_DEST_INDEX = 0x4;
// SFPCONFIG.config_dest value selecting the LaneConfig register.
constexpr std::uint32_t SFPCFG_DEST_LANECONFIG = 0xF;
// imm12 bit 11 makes SFPSETCC test the sign bit as INT32. Float values in an LREG
// use sign-magnitude representation, so this identifies negative FP16B/FP32 lanes.
constexpr std::uint32_t MAX_POOL_SFPSETCC_SIGNBIT = 0x800;

// SFPLOAD address bit 1 selects the Dest column parity.
constexpr std::uint32_t EVEN_COL_OFFSET = p_sfpu::col_offset::EVEN_COL;  // columns 0, 2, ..., 14
constexpr std::uint32_t ODD_COL_OFFSET = p_sfpu::col_offset::ODD_COL;    // columns 1, 3, ..., 15
// Dest rows covered by one SFPLOAD (4 rows x 8 columns of one parity).
constexpr std::uint32_t ROW_GROUP = 4;
// Generic path: 8 logical ROW_MAJOR rows occupy 16 Dest rows.
constexpr std::uint32_t EIGHT_ROW_OFFSET = 16;
// Generic path: 16 logical ROW_MAJOR rows occupy 32 Dest rows.
constexpr std::uint32_t SIXTEEN_ROW_OFFSET = 32;

// TILE-layout replay map. Both faces issue identical SFPLOAD/SFPSTORE words once the Dest counter
// is advanced by one face, so each memory group is recorded once and replayed per face. Slots 0-15
// are free on the math thread: the FPU datacopy that stages the tiles programs no replay buffer.
// The SFPSWAP/SFPTRANSP network deliberately stays inline — TEN-4690 forbids replaying it, and a
// replayed swap network loses the instruction scheduling the compare chain depends on.
constexpr std::uint32_t TILE_STAGE_LOAD_COUNT = 8;  // 4 value + 4 index loads, rows 0-7
constexpr std::uint32_t TILE_ROW8_LOAD_COUNT = 4;   // 2 value + 2 index loads, row 8
constexpr std::uint32_t TILE_STORE_COUNT = 4;       // 2 value + 2 index stores, row 0
constexpr std::uint32_t TILE_SLOT_STAGE_LOADS = 0;
constexpr std::uint32_t TILE_SLOT_ROW8_LOADS = TILE_SLOT_STAGE_LOADS + TILE_STAGE_LOAD_COUNT;
constexpr std::uint32_t TILE_SLOT_STORES = TILE_SLOT_ROW8_LOADS + TILE_ROW8_LOAD_COUNT;

/**
 * @brief Set LaneConfig bit [2] (ENABLE_DEST_INDEX) so SFPSWAP swaps LREG4-7 alongside LREG0-3.
 *
 * @tparam APPROXIMATION_MODE: Kept for SFPU call-shape parity; unused, values = <true/false>
 * @tparam layout: Dest row arrangement the execute stage expects, values = <TILE/ROW_MAJOR>
 * @note Call once before @ref calculate_max_pool_with_indices; the bit persists on Quasar.
 */
template <bool APPROXIMATION_MODE, DataLayout layout = DataLayout::TILE>
inline void init_max_pool_with_indices() {
    ckernel::math::_sfpu_load_config32_(SFPCFG_DEST_LANECONFIG, 0x0 /* upper16 */, LANECFG_ENABLE_DEST_INDEX);
    // SFPCONFIG is 2-cycle; errata TEN-4581 requires SFPNOP padding before the first SFPSWAP.
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
    // No replay-buffer programming on Quasar: TEN-4690 makes a replayed SFPSWAP network unsafe.
}

/**
 * @brief Keep the maximum in VALUE_LREG_A and its paired index in LREG(4 + A).
 *
 * Quasar ALL_ROWS_MAX compares sign-magnitude bits as unsigned values. Its first
 * swap is therefore reversed when both operands are negative. The condition-code
 * guarded second swap corrects exactly those lanes. LaneConfig index tracking
 * mirrors both swaps into the corresponding index registers.
 */
template <std::uint32_t VALUE_LREG_A, std::uint32_t VALUE_LREG_B>
inline void _max_pool_swap_max_() {
    TTI_SFPSWAP(0 /* imm12 */, VALUE_LREG_A, VALUE_LREG_B, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
    TTI_SFPSETCC(MAX_POOL_SFPSETCC_SIGNBIT, VALUE_LREG_A, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPSETCC(MAX_POOL_SFPSETCC_SIGNBIT, VALUE_LREG_B, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPSWAP(0 /* imm12 */, VALUE_LREG_A, VALUE_LREG_B, sfpi::SFPSWAP_MOD1_SWAP);
    TTI_SFPENCC(0 /* imm12 */, 0 /* mod1: clear CC */);
}

/**
 * @brief TILE-layout tournament: max of LREG0..LREG3 lands in LREG0 (even cols) and LREG2 (odd cols).
 *
 * @note Both SFPTRANSPs bulk-update LREG4-7 as well, which keeps every index paired with its value.
 */
inline void _max_pool_sort4_tile_() {
    TTI_SFPTRANSP;  // move the 4 Dest rows held per LREG into the lane dimension
    _max_pool_swap_max_<p_sfpu::LREG0, p_sfpu::LREG1>();
    _max_pool_swap_max_<p_sfpu::LREG2, p_sfpu::LREG3>();
    _max_pool_swap_max_<p_sfpu::LREG0, p_sfpu::LREG2>();
    TTI_SFPTRANSP;                                        // back; row 0 of LREG0..LREG3 now holds the 4 partials
    _max_pool_swap_max_<p_sfpu::LREG0, p_sfpu::LREG1>();  // even-column partials
    _max_pool_swap_max_<p_sfpu::LREG2, p_sfpu::LREG3>();  // odd-column partials
}

/**
 * @brief ROW_MAJOR tournament: max of the 8 logical rows in LREG0..LREG3 lands in LREG0, its index in LREG4.
 *
 * @note Ends on SFPTRANSP, which already drains the preceding SFPSWAP — a following SFPSTORE of
 *       LREG0-3 needs no SFPNOP.
 */
inline void _max_pool_sort4_row_major_() {
    _max_pool_swap_max_<p_sfpu::LREG0, p_sfpu::LREG1>();
    _max_pool_swap_max_<p_sfpu::LREG2, p_sfpu::LREG3>();
    _max_pool_swap_max_<p_sfpu::LREG0, p_sfpu::LREG2>();
    TTI_SFPTRANSP;
    _max_pool_swap_max_<p_sfpu::LREG0, p_sfpu::LREG2>();
    _max_pool_swap_max_<p_sfpu::LREG1, p_sfpu::LREG3>();
    TTI_SFPTRANSP;
}

/**
 * @brief Up-to-9-row column max with argmax. Row 0 of each column holds the result; the rest is scratch.
 *
 * @tparam APPROXIMATION_MODE: Kept for SFPU call-shape parity; unused, values = <true/false>
 * @tparam is_fp32_dest_acc_en: Dest register is in 32-bit mode; picks the index sfpmem mode, values = <true/false>
 * @tparam num_rows: Rows reduced per column, values = <4/8/9>
 * @tparam ITERATIONS: Kept for SFPU call-shape parity; this path has no per-row loop
 * @tparam layout: Dest row arrangement, values = <TILE/ROW_MAJOR>
 * @tparam accumulate: Must be false; only @ref _calculate_max_pool_with_indices_generic_ accumulates
 * @tparam TILE_SHAPE: Dest tile shape used to convert a tile index to a Dest row offset
 * @param values_tile_idx: Dest tile index of the values tile.
 * @param indices_tile_idx: Dest tile index of the paired index tile.
 * @param chunk: Unused on this path; only the accumulate fold reads it.
 * @note Run @ref init_max_pool_with_indices first.
 * @note The TILE path advances the Dest counter itself to reach face 1, so it must be invoked with
 *       VectorMode::None; a vector-mode face walk would advance it a second time.
 */
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    int num_rows,
    int ITERATIONS,
    DataLayout layout,
    bool accumulate,
    trisc::DstTileShape TILE_SHAPE>
inline void _calculate_max_pool_with_indices_(
    const std::uint32_t values_tile_idx,
    const std::uint32_t indices_tile_idx,
    [[maybe_unused]] const std::uint32_t chunk) {
    static_assert(num_rows == 4 || num_rows == 8 || num_rows == 9, "short max pool supports 4, 8, or 9 rows");
    constexpr std::uint32_t VAL_MODE = is_fp32_dest_acc_en ? p_sfpu::sfpmem::FP32 : p_sfpu::sfpmem::FP16B;
    constexpr std::uint32_t IDX_MODE = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::UINT16;
    constexpr std::uint32_t DST_TILE_ROWS = 1U << trisc::get_dest_tile_size_log2(TILE_SHAPE);

    const std::uint32_t values_tile_offset = values_tile_idx * DST_TILE_ROWS;
    const std::uint32_t indices_tile_offset = indices_tile_idx * DST_TILE_ROWS;

    if constexpr (layout == DataLayout::ROW_MAJOR) {
        // Dest is expected to hold F0R0, F1R0, F0R1, F1R1, ... F0R8, F1R8. One pass per column
        // parity; the parity offset is added to every address instead of using a second LREG pair.
        auto process_columns = [values_tile_offset,
                                indices_tile_offset](const std::uint32_t col_offset) __attribute__((always_inline)) {
            const std::uint32_t val_base = values_tile_offset + col_offset;
            const std::uint32_t idx_base = indices_tile_offset + col_offset;

            TT_SFPLOAD(p_sfpu::LREG0, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base + 0 * ROW_GROUP);
            TT_SFPLOAD(p_sfpu::LREG1, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base + 1 * ROW_GROUP);
            TT_SFPLOAD(
                p_sfpu::LREG2, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base + (num_rows == 4 ? 0 : 2) * ROW_GROUP);
            TT_SFPLOAD(
                p_sfpu::LREG3, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base + (num_rows == 4 ? 1 : 3) * ROW_GROUP);
            TT_SFPLOAD(p_sfpu::LREG4, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base + 0 * ROW_GROUP);
            TT_SFPLOAD(p_sfpu::LREG5, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base + 1 * ROW_GROUP);
            TT_SFPLOAD(
                p_sfpu::LREG6, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base + (num_rows == 4 ? 0 : 2) * ROW_GROUP);
            TT_SFPLOAD(
                p_sfpu::LREG7, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base + (num_rows == 4 ? 1 : 3) * ROW_GROUP);

            _max_pool_sort4_row_major_();  // max of logical rows 0-7 in LREG0 / LREG4

            if constexpr (num_rows == 9) {
                // Fold in logical row 8.
                TT_SFPLOAD(p_sfpu::LREG1, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base + 4 * ROW_GROUP);
                TT_SFPLOAD(p_sfpu::LREG5, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base + 4 * ROW_GROUP);
                _max_pool_swap_max_<p_sfpu::LREG0, p_sfpu::LREG1>();
            }
            // Keep an explicit store separation for the num_rows=9 correction swap.
            TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);

            TT_SFPSTORE(p_sfpu::LREG4, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base);
            TT_SFPSTORE(p_sfpu::LREG0, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base);
        };

        process_columns(EVEN_COL_OFFSET);
        process_columns(ODD_COL_OFFSET);
    } else {
        static_assert(!accumulate, "accumulate mode is not supported for TILE layout");

        // Natural face layout: each face is reduced over the requested 4, 8, or 9 rows. Addresses are face-relative
        // and face 1 is reached by advancing the Dest counter, so both faces issue the same words
        // and the memory groups can be replayed.
        const std::uint32_t val_base = values_tile_offset;
        const std::uint32_t idx_base = indices_tile_offset;

        lltt::record(TILE_SLOT_STAGE_LOADS, TILE_STAGE_LOAD_COUNT);
        TT_SFPLOAD(p_sfpu::LREG0, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base + 0 * ROW_GROUP + EVEN_COL_OFFSET);
        TT_SFPLOAD(
            p_sfpu::LREG1,
            VAL_MODE,
            ADDR_MOD_7,
            0 /* done */,
            val_base + (num_rows == 4 ? 0 : 1) * ROW_GROUP + EVEN_COL_OFFSET);
        TT_SFPLOAD(p_sfpu::LREG2, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base + 0 * ROW_GROUP + ODD_COL_OFFSET);
        TT_SFPLOAD(
            p_sfpu::LREG3,
            VAL_MODE,
            ADDR_MOD_7,
            0 /* done */,
            val_base + (num_rows == 4 ? 0 : 1) * ROW_GROUP + ODD_COL_OFFSET);
        TT_SFPLOAD(p_sfpu::LREG4, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base + 0 * ROW_GROUP + EVEN_COL_OFFSET);
        TT_SFPLOAD(
            p_sfpu::LREG5,
            IDX_MODE,
            ADDR_MOD_7,
            0 /* done */,
            idx_base + (num_rows == 4 ? 0 : 1) * ROW_GROUP + EVEN_COL_OFFSET);
        TT_SFPLOAD(p_sfpu::LREG6, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base + 0 * ROW_GROUP + ODD_COL_OFFSET);
        TT_SFPLOAD(
            p_sfpu::LREG7,
            IDX_MODE,
            ADDR_MOD_7,
            0 /* done */,
            idx_base + (num_rows == 4 ? 0 : 1) * ROW_GROUP + ODD_COL_OFFSET);

        if constexpr (num_rows == 9) {
            // Row 8 arrives in lane row 0 of the third row group.
            lltt::record(TILE_SLOT_ROW8_LOADS, TILE_ROW8_LOAD_COUNT);
            TT_SFPLOAD(p_sfpu::LREG1, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base + 2 * ROW_GROUP + EVEN_COL_OFFSET);
            TT_SFPLOAD(p_sfpu::LREG3, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base + 2 * ROW_GROUP + ODD_COL_OFFSET);
            TT_SFPLOAD(p_sfpu::LREG5, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base + 2 * ROW_GROUP + EVEN_COL_OFFSET);
            TT_SFPLOAD(p_sfpu::LREG7, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base + 2 * ROW_GROUP + ODD_COL_OFFSET);
        }

        lltt::record(TILE_SLOT_STORES, TILE_STORE_COUNT);
        TT_SFPSTORE(p_sfpu::LREG0, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base + EVEN_COL_OFFSET);
        TT_SFPSTORE(p_sfpu::LREG2, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base + ODD_COL_OFFSET);
        TT_SFPSTORE(p_sfpu::LREG4, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base + EVEN_COL_OFFSET);
        TT_SFPSTORE(p_sfpu::LREG6, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base + ODD_COL_OFFSET);

        auto process_face = []() __attribute__((always_inline)) {
            lltt::replay(TILE_SLOT_STAGE_LOADS, TILE_STAGE_LOAD_COUNT);

            _max_pool_sort4_tile_();  // max of rows 0-7 in LREG0 (even cols) / LREG2 (odd cols)

            if constexpr (num_rows == 9) {
                lltt::replay(TILE_SLOT_ROW8_LOADS, TILE_ROW8_LOAD_COUNT);
                _max_pool_swap_max_<p_sfpu::LREG0, p_sfpu::LREG1>();  // fold row 8, even cols
                _max_pool_swap_max_<p_sfpu::LREG2, p_sfpu::LREG3>();  // fold row 8, odd cols
            }
            // Keep an explicit store separation for the num_rows=9 correction swap.
            TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);

            lltt::replay(TILE_SLOT_STORES, TILE_STORE_COUNT);
        };

        process_face();
        // Advance the Dest counter one face; the same recorded words then address face 1.
        // _llk_math_eltwise_sfpu_done_ resets the counter once the op finishes.
        _llk_math_sfpu_inc_dst_face_addr_();
        process_face();
    }
}

/**
 * @brief Up-to-32-row ROW_MAJOR column max with argmax, optionally folded with a running result.
 *
 * @tparam APPROXIMATION_MODE: Kept for SFPU call-shape parity; unused, values = <true/false>
 * @tparam is_fp32_dest_acc_en: Dest register is in 32-bit mode; picks the index sfpmem mode, values = <true/false>
 * @tparam num_rows: Rows reduced per column, values = <16/20/32>
 * @tparam ITERATIONS: Kept for SFPU call-shape parity; this path has no per-row loop
 * @tparam accumulate: Fold this chunk's result into the running result, values = <true/false>
 * @tparam TILE_SHAPE: Dest tile shape used to convert a tile index to a Dest row offset
 * @param values_tile_idx: Dest tile index of the values tile.
 * @param indices_tile_idx: Dest tile index of the paired index tile.
 * @param chunk: Chunk counter; chunk 0 seeds the running result, later chunks fold into it.
 * @note Run @ref init_max_pool_with_indices first.
 * @note With accumulate=true the caller must also reserve Dest tiles values_tile_idx + 1 and
 *       indices_tile_idx + 1 — the running result lives there and the kernel cannot check it.
 */
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    int num_rows,
    int ITERATIONS,
    bool accumulate,
    trisc::DstTileShape TILE_SHAPE>
inline void _calculate_max_pool_with_indices_generic_(
    const std::uint32_t values_tile_idx,
    const std::uint32_t indices_tile_idx,
    [[maybe_unused]] const std::uint32_t chunk) {
    static_assert(num_rows == 16 || num_rows == 20 || num_rows == 32, "generic max pool supports 16, 20, or 32 rows");
    constexpr std::uint32_t VAL_MODE = is_fp32_dest_acc_en ? p_sfpu::sfpmem::FP32 : p_sfpu::sfpmem::FP16B;
    constexpr std::uint32_t IDX_MODE = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::UINT16;
    constexpr std::uint32_t DST_TILE_ROWS = 1U << trisc::get_dest_tile_size_log2(TILE_SHAPE);

    const std::uint32_t values_tile_offset = values_tile_idx * DST_TILE_ROWS;
    const std::uint32_t indices_tile_offset = indices_tile_idx * DST_TILE_ROWS;
    const std::uint32_t values_accum_tile_offset = (values_tile_idx + 1) * DST_TILE_ROWS;
    const std::uint32_t indices_accum_tile_offset = (indices_tile_idx + 1) * DST_TILE_ROWS;

    // Reduce either 4 or 8 logical rows into LREG0 / LREG4. For 4 rows, duplicate
    // the first two load groups into LREG2/3 so no value outside the requested
    // window participates in the fixed 8-row tournament.
    auto reduce_rows =
        [](const std::uint32_t val_base, const std::uint32_t idx_base, const bool four_rows, const bool store_result)
            __attribute__((always_inline)) {
                const std::uint32_t third_group = four_rows ? 0 : 2 * ROW_GROUP;
                const std::uint32_t fourth_group = four_rows ? ROW_GROUP : 3 * ROW_GROUP;

                TT_SFPLOAD(p_sfpu::LREG0, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base + 0 * ROW_GROUP);
                TT_SFPLOAD(p_sfpu::LREG1, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base + 1 * ROW_GROUP);
                TT_SFPLOAD(p_sfpu::LREG2, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base + third_group);
                TT_SFPLOAD(p_sfpu::LREG3, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base + fourth_group);
                TT_SFPLOAD(p_sfpu::LREG4, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base + 0 * ROW_GROUP);
                TT_SFPLOAD(p_sfpu::LREG5, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base + 1 * ROW_GROUP);
                TT_SFPLOAD(p_sfpu::LREG6, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base + third_group);
                TT_SFPLOAD(p_sfpu::LREG7, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base + fourth_group);

                _max_pool_sort4_row_major_();

                if (store_result) {
                    // No SFPNOP: _max_pool_sort4_row_major_ ends on SFPTRANSP.
                    TT_SFPSTORE(p_sfpu::LREG0, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base);
                    TT_SFPSTORE(p_sfpu::LREG4, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base);
                }
            };

    // Reduce 16 logical rows. The first 8-row block is spilled so its registers can be reused;
    // the second block's result stays in LREG0 / LREG4 for the closing swap.
    auto process_16_rows =
        [&reduce_rows, values_tile_offset, indices_tile_offset](
            const std::uint32_t base_offset, const std::uint32_t col_offset, const bool store_result)
            __attribute__((always_inline)) {
                const std::uint32_t val_base_first = values_tile_offset + base_offset + col_offset;
                const std::uint32_t idx_base_first = indices_tile_offset + base_offset + col_offset;
                const std::uint32_t val_base_second = val_base_first + EIGHT_ROW_OFFSET;
                const std::uint32_t idx_base_second = idx_base_first + EIGHT_ROW_OFFSET;

                reduce_rows(val_base_first, idx_base_first, false /* four_rows */, true /* store_result */);
                reduce_rows(val_base_second, idx_base_second, false /* four_rows */, false /* store_result */);

                // LREG0 / LREG4 hold Max(R8-15); reload Max(R0-7) to finish the 16-row reduction.
                TT_SFPLOAD(p_sfpu::LREG1, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base_first);
                TT_SFPLOAD(p_sfpu::LREG5, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base_first);
                _max_pool_swap_max_<p_sfpu::LREG0, p_sfpu::LREG1>();

                if (store_result) {
                    // Drain the SFPSWAP before a store that reads LREG0-3.
                    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
                    TT_SFPSTORE(p_sfpu::LREG4, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_base_first);
                    TT_SFPSTORE(p_sfpu::LREG0, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_base_first);
                }
            };

    // Store the completed reduction and optionally fold it into the running result.
    auto store_result =
        [values_tile_offset, indices_tile_offset, values_accum_tile_offset, indices_accum_tile_offset, chunk](
            const std::uint32_t col_offset) __attribute__((always_inline)) {
            const std::uint32_t val_first = values_tile_offset + col_offset;
            const std::uint32_t idx_first = indices_tile_offset + col_offset;

            if constexpr (accumulate) {
                const std::uint32_t val_accum = values_accum_tile_offset + col_offset;
                const std::uint32_t idx_accum = indices_accum_tile_offset + col_offset;
                if (chunk > 0) {
                    // Fold in the running result held in the accumulator tiles.
                    TT_SFPLOAD(p_sfpu::LREG1, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_accum);
                    TT_SFPLOAD(p_sfpu::LREG5, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_accum);
                    _max_pool_swap_max_<p_sfpu::LREG0, p_sfpu::LREG1>();
                }
                TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
                TT_SFPSTORE(p_sfpu::LREG4, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_accum);
                TT_SFPSTORE(p_sfpu::LREG0, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_accum);
            }

            TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
            TT_SFPSTORE(p_sfpu::LREG4, IDX_MODE, ADDR_MOD_7, 0 /* done */, idx_first);
            TT_SFPSTORE(p_sfpu::LREG0, VAL_MODE, ADDR_MOD_7, 0 /* done */, val_first);
        };

    // Finish one column parity completely while its current partial remains in LREG0/LREG4.
    auto process_column = [&process_16_rows, &reduce_rows, &store_result, values_tile_offset, indices_tile_offset](
                              const std::uint32_t col_offset) __attribute__((always_inline)) {
        if constexpr (num_rows == 16) {
            process_16_rows(0 /* base_offset */, col_offset, false /* store_result */);
        } else {
            // Spill Max(R0-15), then reduce exactly the requested remaining rows.
            process_16_rows(0 /* base_offset */, col_offset, true /* store_result */);
            if constexpr (num_rows == 20) {
                reduce_rows(
                    values_tile_offset + SIXTEEN_ROW_OFFSET + col_offset,
                    indices_tile_offset + SIXTEEN_ROW_OFFSET + col_offset,
                    true /* four_rows */,
                    false /* store_result */);
            } else {
                process_16_rows(SIXTEEN_ROW_OFFSET, col_offset, false /* store_result */);
            }

            // Combine the R0-15 partial with R16-19 or R16-31.
            TT_SFPLOAD(p_sfpu::LREG1, VAL_MODE, ADDR_MOD_7, 0 /* done */, values_tile_offset + col_offset);
            TT_SFPLOAD(p_sfpu::LREG5, IDX_MODE, ADDR_MOD_7, 0 /* done */, indices_tile_offset + col_offset);
            _max_pool_swap_max_<p_sfpu::LREG0, p_sfpu::LREG1>();
        }
        store_result(col_offset);
    };

    process_column(EVEN_COL_OFFSET);
    process_column(ODD_COL_OFFSET);
}

/**
 * @brief Column-wise max with argmax over a Dest tile pair; dispatches on num_rows.
 *
 * Row 0 of every column of both tiles receives the result; the remaining rows are scratch.
 *
 * @tparam APPROXIMATION_MODE: Kept for SFPU call-shape parity; unused, values = <true/false>
 * @tparam is_fp32_dest_acc_en: Dest register is in 32-bit mode; picks the index sfpmem mode, values = <true/false>
 * @tparam num_rows: Rows reduced per column, values = <4/8/9/16/20/32>
 * @tparam ITERATIONS: Kept for SFPU call-shape parity; neither path has a per-row loop
 * @tparam layout: Dest row arrangement, values = <TILE/ROW_MAJOR>. num_rows > 9 requires ROW_MAJOR.
 * @tparam accumulate: Fold this chunk's result into the running result; generic path only, values = <true/false>
 * @tparam TILE_SHAPE: Dest tile shape used to convert a tile index to a Dest row offset
 * @param values_tile_idx: Dest tile index of the values tile.
 * @param indices_tile_idx: Dest tile index of the paired index tile.
 * @param unused_tile_idx: Ignored; present because the binary SFPU dispatch passes an output slot.
 * @param chunk: Chunk counter, read only when accumulate is true.
 * @note Call @ref init_max_pool_with_indices before this function.
 * @note Invoke with VectorMode::None — the kernel walks both faces itself.
 */
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    int num_rows,
    int ITERATIONS = SFPU_ITERATIONS,
    DataLayout layout = DataLayout::TILE,
    bool accumulate = false,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void calculate_max_pool_with_indices(
    std::uint32_t values_tile_idx,
    std::uint32_t indices_tile_idx,
    [[maybe_unused]] std::uint32_t unused_tile_idx,
    std::uint32_t chunk) {
    static_assert(
        num_rows == 4 || num_rows == 8 || num_rows == 9 || num_rows == 16 || num_rows == 20 || num_rows == 32,
        "num_rows must be 4, 8, 9, 16, 20, or 32");

    if constexpr (num_rows <= 9) {
        _calculate_max_pool_with_indices_<
            APPROXIMATION_MODE,
            is_fp32_dest_acc_en,
            num_rows,
            ITERATIONS,
            layout,
            accumulate,
            TILE_SHAPE>(values_tile_idx, indices_tile_idx, chunk);
    } else {
        static_assert(
            layout == DataLayout::ROW_MAJOR, "generic max pool with indices is only implemented for ROW_MAJOR layout");
        _calculate_max_pool_with_indices_generic_<
            APPROXIMATION_MODE,
            is_fp32_dest_acc_en,
            num_rows,
            ITERATIONS,
            accumulate,
            TILE_SHAPE>(values_tile_idx, indices_tile_idx, chunk);
    }
}

}  // namespace sfpu
}  // namespace ckernel
