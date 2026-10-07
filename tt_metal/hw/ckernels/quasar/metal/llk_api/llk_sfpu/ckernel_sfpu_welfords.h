// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Running state lives in the LREG4-7 bank; LREG0-3 each hold one tile row after a quad load.
constexpr std::uint32_t WELFORDS_MEAN_REG = p_sfpu::LREG4;      // running mean_N
constexpr std::uint32_t WELFORDS_M2_REG = p_sfpu::LREG5;        // running M2_N
constexpr std::uint32_t WELFORDS_ALPHA_REG = p_sfpu::LREG6;     // scratch alpha = x - mean_N
constexpr std::uint32_t WELFORDS_RECIP_REG = p_sfpu::LREG7;     // 1/(N+1), or 1/N_scale in finalize

// Dest geometry of a 32x32 tile: 16-datum face rows, faces at 0/16/32/48, next tile at +64.
constexpr std::uint32_t WELFORDS_FACE_STRIDE = FACE_R_DIM;
constexpr std::uint32_t WELFORDS_FACE_PAIR_STRIDE = 2 * FACE_R_DIM;
constexpr std::uint32_t WELFORDS_QUAD_ROWS = 4;
constexpr std::uint32_t WELFORDS_QUADS_PER_FACE_PAIR = FACE_R_DIM / WELFORDS_QUAD_ROWS;
constexpr std::uint32_t WELFORDS_FACE_PAIRS = TILE_NUM_FACES / 2;
constexpr std::uint32_t WELFORDS_QUADS_PER_TILE = WELFORDS_FACE_PAIRS * WELFORDS_QUADS_PER_FACE_PAIR;
constexpr std::uint32_t WELFORDS_TILE_STRIDE = 1U << trisc::get_dest_tile_size_log2(trisc::DstTileShape::Tile32x32);
constexpr std::uint32_t WELFORDS_GROUP_SHIFT = 2;  // one group slot = 4 Dest units
constexpr std::uint32_t WELFORDS_NUM_GROUPS = WELFORDS_TILE_STRIDE >> WELFORDS_GROUP_SHIFT;  // group slots per tile

// Quad column offsets: left-even, left-odd, right-even, right-odd.
constexpr std::uint32_t WELFORDS_LEFT_EVEN = p_sfpu::col_offset::EVEN_COL;
constexpr std::uint32_t WELFORDS_LEFT_ODD = p_sfpu::col_offset::ODD_COL;
constexpr std::uint32_t WELFORDS_RIGHT_EVEN = WELFORDS_FACE_STRIDE + p_sfpu::col_offset::EVEN_COL;
constexpr std::uint32_t WELFORDS_RIGHT_ODD = WELFORDS_FACE_STRIDE + p_sfpu::col_offset::ODD_COL;

// One 4-instruction row body per input LREG0-3, recorded back to back into replay slot 0.
constexpr std::uint32_t WELFORDS_INSTR_PER_ROW = 4;  // keep in sync with _welfords_row_
constexpr std::uint32_t WELFORDS_REPLAY_SLOT = 0;
constexpr std::uint32_t WELFORDS_REPLAY_LEN = WELFORDS_QUAD_ROWS * WELFORDS_INSTR_PER_ROW;
constexpr std::uint32_t WELFORDS_REPLAY_DEPTH = 32;
static_assert(WELFORDS_REPLAY_LEN <= WELFORDS_REPLAY_DEPTH, "the recorded row bodies must fit the replay buffer");

// SFPLOADI immediates.
constexpr std::uint32_t FP16B_ZERO = 0x0000;      // 0.0
constexpr std::uint32_t FP32_HI16_SHIFT = 16;     // fp32 bits [31:16] -> MOD0_UPPER immediate
constexpr std::uint32_t FP32_LO16_MASK = 0xFFFF;  // fp32 bits [15:0]  -> MOD0_LOWER immediate

// Finalize output layout: one tile row (Row) or the raw face layout (Face).
enum class WelfordsOutputLayout : std::uint8_t { Row, Face };

/**
 * @brief Load 1/(idx+1) into WELFORDS_RECIP_REG, broadcast to every lane.
 *
 * @tparam RECIPROCAL_SIZE: Size of reciprocal_lut; 0 divides on the RISC-V instead of indexing it.
 * @param idx: Sample index whose reciprocal is wanted; must be < RECIPROCAL_SIZE when the LUT is used.
 * @param reciprocal_lut: fp32 bit patterns of 1/(i+1).
 */
template <std::size_t RECIPROCAL_SIZE>
inline void _welfords_load_recip_(
    const std::uint32_t idx, [[maybe_unused]] const std::array<std::uint32_t, RECIPROCAL_SIZE>& reciprocal_lut) {
    std::uint32_t bits;
    if constexpr (RECIPROCAL_SIZE > 0) {
        LLK_ASSERT(idx < RECIPROCAL_SIZE, "welfords: reciprocal LUT is shorter than the sample index");
        bits = reciprocal_lut[idx];
    } else {
        bits = __builtin_bit_cast(std::uint32_t, 1.0f / static_cast<float>(idx + 1));
    }
    TT_SFPLOADI(
        WELFORDS_RECIP_REG, sfpi::SFPLOADI_MOD0_UPPER, bits >> FP32_HI16_SHIFT /* imm16: high half of 1/(idx+1) */);
    TT_SFPLOADI(
        WELFORDS_RECIP_REG,
        sfpi::SFPLOADI_MOD0_LOWER,
        bits & FP32_LO16_MASK /* imm16: low half of 1/(idx+1), high half preserved */);
}

/**
 * @brief Fold one tile row into the running mean and M2, per column.
 *
 * alpha stays in WELFORDS_ALPHA_REG while the new mean goes straight into WELFORDS_MEAN_REG, so no
 * second alpha or mean copy is needed. Dependent MAD -> MAD consumers are interlocked by hardware,
 * so no SFPNOP sits between them.
 *
 * @tparam INPUT_LREG: LREG holding the row, values = <LREG0/LREG1/LREG2/LREG3>
 * @note Clobbers INPUT_LREG and WELFORDS_ALPHA_REG, and expects WELFORDS_RECIP_REG to already hold
 *       1/(N+1) for this row — load it with @ref _welfords_load_recip_ first.
 */
template <std::uint32_t INPUT_LREG>
inline void _welfords_row_() {
    TTI_SFPMAD(
        p_sfpu::LCONST_neg1,
        WELFORDS_MEAN_REG,
        INPUT_LREG,
        WELFORDS_ALPHA_REG,
        0 /* instr_mod1 */);  // alpha = x - mean_N
    TTI_SFPMAD(
        WELFORDS_ALPHA_REG,
        WELFORDS_RECIP_REG,
        WELFORDS_MEAN_REG,
        WELFORDS_MEAN_REG,
        0 /* instr_mod1 */);  // mean_{N+1} = alpha/(N+1) + mean_N
    TTI_SFPMAD(
        p_sfpu::LCONST_neg1,
        WELFORDS_MEAN_REG,
        INPUT_LREG,
        INPUT_LREG,
        0 /* instr_mod1 */);  // beta = x - mean_{N+1}
    TTI_SFPMAD(
        WELFORDS_ALPHA_REG, INPUT_LREG, WELFORDS_M2_REG, WELFORDS_M2_REG, 0 /* instr_mod1 */);  // M2 += alpha * beta
}

/**
 * @brief Issue the recorded @ref _welfords_row_ body for one input LREG.
 *
 * @tparam INPUT_LREG: LREG holding the row, values = <LREG0/LREG1/LREG2/LREG3>
 * @note Call @ref welfords_init first — it records the body this replays.
 */
template <std::uint32_t INPUT_LREG>
inline void _welfords_replay_row_() {
    TTI_REPLAY(
        WELFORDS_REPLAY_SLOT + (WELFORDS_INSTR_PER_ROW * INPUT_LREG) /* start_idx */,
        WELFORDS_INSTR_PER_ROW /* len */,
        0 /* last */,
        0 /* set_mutex */,
        0 /* execute_while_loading */,
        0 /* load_mode */);
}

/**
 * @brief Load four whole tile rows of one face pair into LREG0-3.
 *
 * @tparam I: Face pair, values = <0/1>
 * @tparam J: Quad within the face pair, values = <0..WELFORDS_QUADS_PER_FACE_PAIR-1>
 * @note Neither SFPTRANSP is removable. SFPTRANSP permutes LREG4-7 as well as LREG0-3, so it is the
 *       second one that puts the running state back where @ref _welfords_row_ expects it.
 */
template <std::uint32_t I, std::uint32_t J>
inline void _welfords_load_quad_() {
    constexpr std::uint32_t BASE = (I * WELFORDS_FACE_PAIR_STRIDE) + (J * WELFORDS_QUAD_ROWS);

    TTI_SFPTRANSP;  // scrambles LREG4-7; the second transpose restores them
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, BASE + WELFORDS_LEFT_EVEN);
    TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, BASE + WELFORDS_LEFT_ODD);
    TTI_SFPLOAD(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, BASE + WELFORDS_RIGHT_EVEN);
    TTI_SFPLOAD(p_sfpu::LREG3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, BASE + WELFORDS_RIGHT_ODD);
    TTI_SFPTRANSP;  // LREGk = tile row 4J+k of face pair I, all 32 columns; LREG4-7 restored
}

/**
 * @brief Fold all four rows of one quad into the running state.
 *
 * @tparam RECIPROCAL_SIZE: Size of reciprocal_lut; 0 computes 1/(N+1) on the RISC-V.
 * @tparam I: Face pair, values = <0/1>
 * @tparam J: Quad within the face pair, values = <0..WELFORDS_QUADS_PER_FACE_PAIR-1>
 * @param start_idx: Sample index of tile row 0.
 * @param reciprocal_lut: fp32 bit patterns of 1/(i+1).
 */
template <std::size_t RECIPROCAL_SIZE, std::uint32_t I, std::uint32_t J>
inline void _calculate_welfords_quad_(
    const std::uint32_t start_idx, const std::array<std::uint32_t, RECIPROCAL_SIZE>& reciprocal_lut) {
    constexpr std::uint32_t QUAD_IDX = (I * FACE_R_DIM) + (J * WELFORDS_QUAD_ROWS);

    _welfords_load_quad_<I, J>();
    _welfords_load_recip_<RECIPROCAL_SIZE>(start_idx + QUAD_IDX + 0, reciprocal_lut);
    _welfords_replay_row_<p_sfpu::LREG0 /* INPUT_LREG */>();
    _welfords_load_recip_<RECIPROCAL_SIZE>(start_idx + QUAD_IDX + 1, reciprocal_lut);
    _welfords_replay_row_<p_sfpu::LREG1 /* INPUT_LREG */>();
    _welfords_load_recip_<RECIPROCAL_SIZE>(start_idx + QUAD_IDX + 2, reciprocal_lut);
    _welfords_replay_row_<p_sfpu::LREG2 /* INPUT_LREG */>();
    _welfords_load_recip_<RECIPROCAL_SIZE>(start_idx + QUAD_IDX + 3, reciprocal_lut);
    _welfords_replay_row_<p_sfpu::LREG3 /* INPUT_LREG */>();
}

/**
 * @brief Fold row K of an already-loaded quad, if the caller's row window covers it.
 *
 * @tparam RECIPROCAL_SIZE: Size of reciprocal_lut; 0 computes 1/(N+1) on the RISC-V.
 * @tparam K: Row within the quad, values = <0..WELFORDS_QUAD_ROWS-1>
 * @param idx: Sample index of the next processed row; advanced when row K is folded.
 * @param s: First quad-relative row of the window.
 * @param e: One past the last quad-relative row of the window.
 * @param reciprocal_lut: fp32 bit patterns of 1/(i+1).
 */
template <std::size_t RECIPROCAL_SIZE, std::uint32_t K>
inline void _calculate_welfords_quad_row_(
    std::uint32_t& idx,
    const std::uint32_t s,
    const std::uint32_t e,
    const std::array<std::uint32_t, RECIPROCAL_SIZE>& reciprocal_lut) {
    if (s <= K && e > K) {
        _welfords_load_recip_<RECIPROCAL_SIZE>(idx, reciprocal_lut);
        _welfords_replay_row_<p_sfpu::LREG0 + K>();
        ++idx;
    }
}

/**
 * @brief Fold the rows of one quad that fall inside [start_row, end_row), skipping the quad entirely
 *        when the window misses it.
 *
 * @tparam RECIPROCAL_SIZE: Size of reciprocal_lut; 0 computes 1/(N+1) on the RISC-V.
 * @tparam I: Face pair, values = <0/1>
 * @tparam J: Quad within the face pair, values = <0..WELFORDS_QUADS_PER_FACE_PAIR-1>
 * @param idx: Sample index of the next processed row; advanced once per folded row.
 * @param start_row: First tile row of the window.
 * @param end_row: One past the last tile row of the window.
 * @param reciprocal_lut: fp32 bit patterns of 1/(i+1).
 */
template <std::size_t RECIPROCAL_SIZE, std::uint32_t I, std::uint32_t J>
inline void _calculate_welfords_quad_rows_(
    std::uint32_t& idx,
    const std::uint32_t start_row,
    const std::uint32_t end_row,
    const std::array<std::uint32_t, RECIPROCAL_SIZE>& reciprocal_lut) {
    constexpr std::uint32_t LO = (I * FACE_R_DIM) + (J * WELFORDS_QUAD_ROWS);
    constexpr std::uint32_t HI = LO + WELFORDS_QUAD_ROWS;

    if (start_row >= HI || end_row <= LO) {
        return;
    }
    const std::uint32_t s = std::max(LO, start_row) - LO;
    const std::uint32_t e = std::min(HI, end_row) - LO;

    _welfords_load_quad_<I, J>();
    _calculate_welfords_quad_row_<RECIPROCAL_SIZE, 0 /* K */>(idx, s, e, reciprocal_lut);
    _calculate_welfords_quad_row_<RECIPROCAL_SIZE, 1 /* K */>(idx, s, e, reciprocal_lut);
    _calculate_welfords_quad_row_<RECIPROCAL_SIZE, 2 /* K */>(idx, s, e, reciprocal_lut);
    _calculate_welfords_quad_row_<RECIPROCAL_SIZE, 3 /* K */>(idx, s, e, reciprocal_lut);
}

/**
 * @brief Fold all 32 rows of the tile, quad by quad, into the running state.
 *
 * Flat quad index Q walks the tile top to bottom: face pair Q / WELFORDS_QUADS_PER_FACE_PAIR, then
 * quad Q % WELFORDS_QUADS_PER_FACE_PAIR within it. Unrolled at compile time because every quad
 * addresses Dest with its own immediate.
 *
 * @tparam RECIPROCAL_SIZE: Size of reciprocal_lut; 0 computes 1/(N+1) on the RISC-V.
 * @tparam Q: Flat quad index this step folds, values = <0..WELFORDS_QUADS_PER_TILE>; the recursion
 *            stops at WELFORDS_QUADS_PER_TILE.
 * @param start_idx: Sample index of tile row 0.
 * @param reciprocal_lut: fp32 bit patterns of 1/(i+1).
 */
template <std::size_t RECIPROCAL_SIZE, std::uint32_t Q = 0>
inline void _calculate_welfords_all_quads_(
    const std::uint32_t start_idx, const std::array<std::uint32_t, RECIPROCAL_SIZE>& reciprocal_lut) {
    if constexpr (Q < WELFORDS_QUADS_PER_TILE) {
        _calculate_welfords_quad_<
            RECIPROCAL_SIZE,
            Q / WELFORDS_QUADS_PER_FACE_PAIR /* I */,
            Q % WELFORDS_QUADS_PER_FACE_PAIR /* J */>(start_idx, reciprocal_lut);
        _calculate_welfords_all_quads_<RECIPROCAL_SIZE, Q + 1>(start_idx, reciprocal_lut);
    }
}

/**
 * @brief Fold the tile rows inside [start_row, end_row), quad by quad, into the running state.
 *
 * Same compile-time quad walk as @ref _calculate_welfords_all_quads_; quads outside the window are
 * skipped without loading them.
 *
 * @tparam RECIPROCAL_SIZE: Size of reciprocal_lut; 0 computes 1/(N+1) on the RISC-V.
 * @tparam Q: Flat quad index this step folds, values = <0..WELFORDS_QUADS_PER_TILE>; the recursion
 *            stops at WELFORDS_QUADS_PER_TILE.
 * @param idx: Sample index of the next processed row; advanced once per folded row.
 * @param start_row: First tile row of the window.
 * @param end_row: One past the last tile row of the window, values = <start_row..TILE_R_DIM>
 * @param reciprocal_lut: fp32 bit patterns of 1/(i+1).
 */
template <std::size_t RECIPROCAL_SIZE, std::uint32_t Q = 0>
inline void _calculate_welfords_all_quad_rows_(
    std::uint32_t& idx,
    const std::uint32_t start_row,
    const std::uint32_t end_row,
    const std::array<std::uint32_t, RECIPROCAL_SIZE>& reciprocal_lut) {
    if constexpr (Q < WELFORDS_QUADS_PER_TILE) {
        _calculate_welfords_quad_rows_<
            RECIPROCAL_SIZE,
            Q / WELFORDS_QUADS_PER_FACE_PAIR /* I */,
            Q % WELFORDS_QUADS_PER_FACE_PAIR /* J */>(idx, start_row, end_row, reciprocal_lut);
        _calculate_welfords_all_quad_rows_<RECIPROCAL_SIZE, Q + 1>(idx, start_row, end_row, reciprocal_lut);
    }
}

/**
 * @brief Record the four per-input-LREG Welford row bodies into replay slot 0.
 *
 * @note Call after @ref _llk_math_eltwise_sfpu_init_ and before @ref calculate_welfords, and again
 *       after any other SFPU op records its own replay on this thread. Does not touch LREG4/LREG5.
 */
inline void welfords_init() {
    load_replay_buf<WELFORDS_REPLAY_SLOT, WELFORDS_REPLAY_LEN, false /* exec_while_loading */>([] {
        _welfords_row_<p_sfpu::LREG0 /* INPUT_LREG */>();
        _welfords_row_<p_sfpu::LREG1 /* INPUT_LREG */>();
        _welfords_row_<p_sfpu::LREG2 /* INPUT_LREG */>();
        _welfords_row_<p_sfpu::LREG3 /* INPUT_LREG */>();
    });
}

/**
 * @brief Zero the running mean (LREG4) and M2 (LREG5).
 */
inline void welfords_clear_previous_mean_and_m2() {
    TTI_SFPLOADI(WELFORDS_MEAN_REG, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ZERO);
    TTI_SFPLOADI(WELFORDS_M2_REG, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ZERO);
}

/**
 * @brief Fold the rows of one 32x32 Dest tile into the per-column running mean and M2.
 *
 * @tparam PARTIAL_TILE: Whether only rows [start_row, start_row + num_rows) are processed.
 * @tparam RECIPROCAL_SIZE: Size of reciprocal_lut; 0 computes 1/(N+1) on the RISC-V.
 * @param start_idx: Sample index of the first processed row (LUT index of its 1/(N+1)).
 * @param reciprocal_lut: fp32 bit patterns of 1/(i+1).
 * @param start_row: First tile row to process; used only when PARTIAL_TILE.
 * @param num_rows: Number of tile rows to process; used only when PARTIAL_TILE. Requires
 *        start_row + num_rows <= TILE_R_DIM.
 * @note Run once per tile under VectorMode::RC_custom. Call @ref welfords_init first; the state
 *       carries across calls in LREG4/LREG5, so write nothing to LREG4-7 between tiles.
 */
template <bool PARTIAL_TILE, std::size_t RECIPROCAL_SIZE>
inline void calculate_welfords(
    const std::uint32_t start_idx,
    const std::array<std::uint32_t, RECIPROCAL_SIZE>& reciprocal_lut,
    [[maybe_unused]] const std::uint32_t start_row = 0,
    [[maybe_unused]] const std::uint32_t num_rows = TILE_R_DIM) {
    if constexpr (!PARTIAL_TILE) {
        _calculate_welfords_all_quads_<RECIPROCAL_SIZE>(start_idx, reciprocal_lut);
    } else {
        if (num_rows == 0) {
            return;
        }
        LLK_ASSERT(
            num_rows <= TILE_R_DIM && start_row <= TILE_R_DIM - num_rows,
            "welfords: partial row window runs past the tile");  // overflow-safe start_row + num_rows <= TILE_R_DIM
        const std::uint32_t end_row = start_row + num_rows;
        std::uint32_t idx = start_idx;
        _calculate_welfords_all_quad_rows_<RECIPROCAL_SIZE>(idx, start_row, end_row, reciprocal_lut);
    }
}

/**
 * @brief Save the running mean to the Dest tile and M2 to the tile after it.
 *
 * @tparam GROUPED: Whether the state goes to group slot group_id (Dest unit offset group_id << 2).
 * @param group_id: Group slot, values = <0..WELFORDS_NUM_GROUPS-1> (0..15); used only when GROUPED.
 * @note Run under VectorMode::RC_custom. Pair with @ref welfords_load_mean_m2_from_dst.
 */
template <bool GROUPED = false>
inline void welfords_store_mean_m2_to_dst([[maybe_unused]] const std::uint32_t group_id = 0) {
    if constexpr (GROUPED) {
        LLK_ASSERT(group_id < WELFORDS_NUM_GROUPS, "welfords: group_id past the last group slot of the tile");
        const std::uint32_t group_offset = group_id << WELFORDS_GROUP_SHIFT;
        TT_SFPSTORE(WELFORDS_MEAN_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, group_offset);
        TT_SFPSTORE(
            WELFORDS_M2_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_TILE_STRIDE + group_offset);
    } else {
        TTI_SFPSTORE(WELFORDS_MEAN_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg_addr */);
        TTI_SFPSTORE(WELFORDS_M2_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_TILE_STRIDE);
    }
}

/**
 * @brief Restore the running mean from the Dest tile and M2 from the tile after it.
 *
 * @tparam GROUPED: Whether the state comes from group slot group_id (Dest unit offset group_id << 2).
 * @param group_id: Group slot, values = <0..WELFORDS_NUM_GROUPS-1> (0..15); used only when GROUPED.
 * @note Run under VectorMode::RC_custom. Pair with @ref welfords_store_mean_m2_to_dst.
 */
template <bool GROUPED = false>
inline void welfords_load_mean_m2_from_dst([[maybe_unused]] const std::uint32_t group_id = 0) {
    if constexpr (GROUPED) {
        LLK_ASSERT(group_id < WELFORDS_NUM_GROUPS, "welfords: group_id past the last group slot of the tile");
        const std::uint32_t group_offset = group_id << WELFORDS_GROUP_SHIFT;
        TT_SFPLOAD(WELFORDS_MEAN_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, group_offset);
        TT_SFPLOAD(
            WELFORDS_M2_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_TILE_STRIDE + group_offset);
    } else {
        TTI_SFPLOAD(WELFORDS_MEAN_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg_addr */);
        TTI_SFPLOAD(WELFORDS_M2_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_TILE_STRIDE);
    }
}

/**
 * @brief Convert M2 to variance and store mean to the Dest tile and variance to the tile after it.
 *
 * @tparam LAYOUT: Row writes tile row 0 in natural column order; Face writes the raw state layout.
 * @tparam GROUPED: Whether the Face output goes to group slot group_id; not supported for Row.
 * @tparam RECIPROCAL_SIZE: Size of reciprocal_lut; 0 computes 1/(scale_idx+1) on the RISC-V.
 * @param scale_idx: Variance divisor index: variance = M2 / (scale_idx + 1).
 * @param reciprocal_lut: fp32 bit patterns of 1/(i+1).
 * @param group_id: Group slot, values = <0..WELFORDS_NUM_GROUPS-1> (0..15); used only when GROUPED.
 * @note Run under VectorMode::RC_custom. The running state is invalid afterwards for either layout:
 *       Row scrambles LREG0-7, and Face overwrites M2 (LREG5) with the variance in place. Do not keep
 *       accumulating or call @ref welfords_store_mean_m2_to_dst after a finalize; reload the state with
 *       @ref welfords_load_mean_m2_from_dst or clear it with @ref welfords_clear_previous_mean_and_m2.
 */
template <WelfordsOutputLayout LAYOUT, bool GROUPED, std::size_t RECIPROCAL_SIZE>
inline void welfords_store_mean_var_to_dst(
    const std::uint32_t scale_idx,
    const std::array<std::uint32_t, RECIPROCAL_SIZE>& reciprocal_lut,
    [[maybe_unused]] const std::uint32_t group_id = 0) {
    static_assert(!(LAYOUT == WelfordsOutputLayout::Row && GROUPED), "the Row layout has no grouped variant");

    _welfords_load_recip_<RECIPROCAL_SIZE>(scale_idx, reciprocal_lut);  // RECIP = 1/(scale_idx+1)

    if constexpr (LAYOUT == WelfordsOutputLayout::Row) {
        TTI_SFPMOV(WELFORDS_MEAN_REG, p_sfpu::LREG0, 0 /* instr_mod1: plain copy */);  // mean -> LREG0
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ZERO);
        TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ZERO);
        TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ZERO);
        TTI_SFPMAD(
            WELFORDS_RECIP_REG,
            WELFORDS_M2_REG,
            p_sfpu::LCONST_0,
            p_sfpu::LREG4,
            0 /* instr_mod1 */);  // variance -> LREG4
        TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ZERO);
        TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ZERO);
        TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ZERO);
        TTI_SFPTRANSP;  // every lane's mean lands in lane-row 0 of LREG0-3, variance in LREG4-7

        // Mean tile row 0 (rows 1-3 zero)
        TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_LEFT_EVEN);
        TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_LEFT_ODD);
        TTI_SFPSTORE(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_RIGHT_EVEN);
        TTI_SFPSTORE(p_sfpu::LREG3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_RIGHT_ODD);

        // Variance tile row 0 (rows 1-3 zero)
        TTI_SFPSTORE(
            p_sfpu::LREG4,
            p_sfpu::sfpmem::DEFAULT,
            ADDR_MOD_7,
            0 /* done */,
            WELFORDS_TILE_STRIDE + WELFORDS_LEFT_EVEN);
        TTI_SFPSTORE(
            p_sfpu::LREG5, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_TILE_STRIDE + WELFORDS_LEFT_ODD);
        TTI_SFPSTORE(
            p_sfpu::LREG6,
            p_sfpu::sfpmem::DEFAULT,
            ADDR_MOD_7,
            0 /* done */,
            WELFORDS_TILE_STRIDE + WELFORDS_RIGHT_EVEN);
        TTI_SFPSTORE(
            p_sfpu::LREG7,
            p_sfpu::sfpmem::DEFAULT,
            ADDR_MOD_7,
            0 /* done */,
            WELFORDS_TILE_STRIDE + WELFORDS_RIGHT_ODD);
    } else {
        TTI_SFPMAD(
            WELFORDS_RECIP_REG,
            WELFORDS_M2_REG,
            p_sfpu::LCONST_0,
            WELFORDS_M2_REG,
            0 /* instr_mod1 */);  // M2 := variance
        // The raw layout is the state layout, so the variance goes out exactly where M2 would.
        welfords_store_mean_m2_to_dst<GROUPED>(group_id);
    }
}

// ---------------------------------------------------------------------------------------------
// Shifted two-pass column statistics (port of tt-llk/common/ckernel_sfpu_welfords_common.h).
//
// Same register protocol, function names and signatures as the Blackhole/Wormhole helpers, so the
// compute-API wiring maps one to one:
//   - Pass one keeps a common anchor in LREG4 and accumulates (x - anchor) in LREG5, alternating
//     with LREG6 when the dual accumulator is on. _two_pass_finish_shifted_mean_ turns the sum into
//     the mean in LREG4 and clears LREG5/LREG6.
//   - Pass two accumulates (x - mean)^2 in LREG5 (and LREG6 when dual).
//   - The optional retained anchor lives in LREG7; the paired SFPTRANSPs of every quad load
//     preserve it.
//   - State spills use two consecutive Dest tiles: mean at raw offset 0, M2 at WELFORDS_TILE_STRIDE.
//     The retained anchor can be parked in the unused group slot 1 (raw offset 4) of the mean tile.
//   - Finalisers take reciprocals as fp32 bit patterns so the RISC-V never divides.
//
// Quasar deviations from Blackhole: SFPMOV/SFPNOP/SFPTRANSP encodings; -1 is LCONST_neg1 (LREG11,
// as in the online path above); the column rotate is SFPSHFT2 mode 3 (the same rotate the Quasar
// row reduce uses) with an SFPNOP before its reader; and the first-lane broadcast isolates SFPU
// column 7 with zero-fill shifts (SFPSHFT2 mode 4) instead of masking with Blackhole's LTILEID.
// The SFPNOPs Blackhole emits before an SFPTRANSP that reads a just-updated accumulator are kept.
// ---------------------------------------------------------------------------------------------

constexpr std::uint32_t TWO_PASS_MEAN_REG = p_sfpu::LREG4;                          // anchor in pass one, mean after it
constexpr std::uint32_t TWO_PASS_ACC_REG = p_sfpu::LREG5;                           // shifted sum / M2 (even rows)
constexpr std::uint32_t TWO_PASS_ACC2_REG = p_sfpu::LREG6;                          // second accumulator (odd rows)
constexpr std::uint32_t TWO_PASS_ANCHOR_REG = p_sfpu::LREG7;                        // retained anchor
constexpr std::uint32_t TWO_PASS_ANCHOR_STATE_OFFSET = 1U << WELFORDS_GROUP_SHIFT;  // group slot 1 of the mean tile
constexpr std::uint32_t TWO_PASS_SFPU_COLUMNS = 8;                                  // SFPU column instances a row spans
constexpr std::uint32_t TWO_PASS_SFPSHFT2_ROTATE = 3;             // rotate one LREG across columns, X -> X+1
constexpr std::uint32_t TWO_PASS_SFPSHFT2_SHIFT_ZERO_FILL = 4;    // shift one LREG across columns, column 0 <- 0
constexpr std::uint32_t TWO_PASS_LANE_RECIPROCAL_FP16B = 0x3D00;  // 1/32, one per lane population
constexpr std::uint32_t FP32_SIGN_BIT = 0x80000000U;

/** @brief Wait out the latency of the previous SFPU result before an SFPTRANSP/SFPSTORE reads it. */
inline void _two_pass_drain_() { TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */); }

/**
 * @brief Load an fp32 bit pattern into every lane of LREG.
 *
 * @tparam LREG: Destination LREG.
 * @param bits: fp32 bit pattern.
 */
template <std::uint32_t LREG>
inline void _two_pass_load_fp32_(const std::uint32_t bits) {
    TT_SFPLOADI(LREG, sfpi::SFPLOADI_MOD0_UPPER, bits >> FP32_HI16_SHIFT /* imm16: high half */);
    TT_SFPLOADI(LREG, sfpi::SFPLOADI_MOD0_LOWER, bits & FP32_LO16_MASK /* imm16: low half, high kept */);
}

/** @brief Zero LREG. */
template <std::uint32_t LREG>
inline void _two_pass_zero_() {
    TTI_SFPLOADI(LREG, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ZERO);
}

/** @brief LREG5 += LREG6: fold the dual accumulator into the primary one. */
inline void _two_pass_fold_dual_() {
    TTI_SFPADD(TWO_PASS_ACC_REG, p_sfpu::LCONST_1, TWO_PASS_ACC2_REG, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
}

/**
 * @brief Accumulate one shifted input row: ACC += x - anchor.
 *
 * @tparam INPUT_LREG: LREG holding the row, values = <LREG0/LREG1/LREG2/LREG3>; clobbered.
 * @tparam ACC_LREG: Accumulator, values = <LREG5/LREG6>
 */
template <std::uint32_t INPUT_LREG, std::uint32_t ACC_LREG>
inline void _two_pass_accumulate_shifted_sum_row_() {
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, INPUT_LREG, INPUT_LREG, 0 /* instr_mod1 */);  // x - anchor
    TTI_SFPADD(ACC_LREG, p_sfpu::LCONST_1, INPUT_LREG, ACC_LREG, 0 /* instr_mod1 */);                // ACC += it
}

/** @brief Accumulate a whole loaded quad through the two independent LREG5/LREG6 chains. */
inline void _two_pass_accumulate_shifted_sum_block_() {
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, p_sfpu::LREG0, p_sfpu::LREG0, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, p_sfpu::LREG1, p_sfpu::LREG1, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, p_sfpu::LREG2, p_sfpu::LREG2, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, p_sfpu::LREG3, p_sfpu::LREG3, 0 /* instr_mod1 */);
    TTI_SFPADD(TWO_PASS_ACC_REG, p_sfpu::LCONST_1, p_sfpu::LREG0, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
    TTI_SFPADD(TWO_PASS_ACC2_REG, p_sfpu::LCONST_1, p_sfpu::LREG1, TWO_PASS_ACC2_REG, 0 /* instr_mod1 */);
    TTI_SFPADD(TWO_PASS_ACC_REG, p_sfpu::LCONST_1, p_sfpu::LREG2, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
    TTI_SFPADD(TWO_PASS_ACC2_REG, p_sfpu::LCONST_1, p_sfpu::LREG3, TWO_PASS_ACC2_REG, 0 /* instr_mod1 */);
    _two_pass_drain_();  // the next quad's SFPTRANSP reads LREG6
}

/**
 * @brief Accumulate row K of a loaded quad when it lies in [first, last).
 *
 * @tparam DUAL: Odd rows go to LREG6 instead of LREG5.
 * @tparam K: Row within the quad, values = <0..3>
 */
template <bool DUAL, std::uint32_t K>
inline void _two_pass_accumulate_shifted_sum_row_if_(const std::uint32_t first, const std::uint32_t last) {
    if (first <= K && last > K) {
        constexpr std::uint32_t ACC_LREG = (DUAL && (K & 1U) != 0) ? TWO_PASS_ACC2_REG : TWO_PASS_ACC_REG;
        _two_pass_accumulate_shifted_sum_row_<p_sfpu::LREG0 + K /* INPUT_LREG */, ACC_LREG>();
    }
}

/**
 * @brief Accumulate rows [first, last) of an already-loaded quad.
 *
 * @tparam DUAL: Whether odd rows use LREG6.
 * @param first: First selected row within the quad.
 * @param last: One past the last selected row within the quad.
 */
template <bool DUAL>
inline void _two_pass_accumulate_shifted_sum_loaded_block_(const std::uint32_t first, const std::uint32_t last) {
    if constexpr (DUAL) {
        if (first == 0 && last == WELFORDS_QUAD_ROWS) {
            _two_pass_accumulate_shifted_sum_block_();
            return;
        }
    }
    _two_pass_accumulate_shifted_sum_row_if_<DUAL, 0 /* K */>(first, last);
    _two_pass_accumulate_shifted_sum_row_if_<DUAL, 1 /* K */>(first, last);
    _two_pass_accumulate_shifted_sum_row_if_<DUAL, 2 /* K */>(first, last);
    _two_pass_accumulate_shifted_sum_row_if_<DUAL, 3 /* K */>(first, last);
    _two_pass_drain_();  // the next quad's SFPTRANSP reads the accumulator
}

/** @brief Accumulate one centred square into the serial LREG5 chain (LREG6 is the residual scratch). */
template <std::uint32_t INPUT_LREG>
inline void _two_pass_accumulate_m2_single_() {
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, INPUT_LREG, TWO_PASS_ACC2_REG, 0 /* instr_mod1 */);
    TTI_SFPMAD(TWO_PASS_ACC2_REG, TWO_PASS_ACC2_REG, TWO_PASS_ACC_REG, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
}

/**
 * @brief Accumulate a whole loaded quad into the serial LREG5 chain, in row order.
 *
 * Residuals alternate between LREG6 and the dead first input LREG0, so a retained anchor in LREG7
 * survives and the M2 updates stay in row order.
 */
inline void _two_pass_accumulate_m2_block_() {
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, p_sfpu::LREG0, TWO_PASS_ACC2_REG, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, p_sfpu::LREG1, p_sfpu::LREG0, 0 /* instr_mod1 */);
    TTI_SFPMAD(TWO_PASS_ACC2_REG, TWO_PASS_ACC2_REG, TWO_PASS_ACC_REG, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, p_sfpu::LREG2, TWO_PASS_ACC2_REG, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, TWO_PASS_ACC_REG, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, p_sfpu::LREG3, p_sfpu::LREG0, 0 /* instr_mod1 */);
    TTI_SFPMAD(TWO_PASS_ACC2_REG, TWO_PASS_ACC2_REG, TWO_PASS_ACC_REG, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, TWO_PASS_ACC_REG, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
    _two_pass_drain_();  // the next quad's SFPTRANSP reads LREG5
}

/** @brief Accumulate one centred square into the selected dual chain; the input is formed in place. */
template <std::uint32_t INPUT_LREG, std::uint32_t ACC_LREG>
inline void _two_pass_accumulate_m2_dual_single_() {
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, INPUT_LREG, INPUT_LREG, 0 /* instr_mod1 */);
    TTI_SFPMAD(INPUT_LREG, INPUT_LREG, ACC_LREG, ACC_LREG, 0 /* instr_mod1 */);
}

/** @brief Accumulate a whole loaded quad, alternating the LREG5/LREG6 M2 chains. */
inline void _two_pass_accumulate_m2_dual_block_() {
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, p_sfpu::LREG0, p_sfpu::LREG0, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, p_sfpu::LREG1, p_sfpu::LREG1, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, p_sfpu::LREG2, p_sfpu::LREG2, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, p_sfpu::LREG3, p_sfpu::LREG3, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, TWO_PASS_ACC_REG, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG1, TWO_PASS_ACC2_REG, TWO_PASS_ACC2_REG, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG2, TWO_PASS_ACC_REG, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG3, TWO_PASS_ACC2_REG, TWO_PASS_ACC2_REG, 0 /* instr_mod1 */);
    _two_pass_drain_();  // the next quad's SFPTRANSP reads LREG6
}

/** @brief Accumulate the centred square of row K of a loaded quad when it lies in [first, last). */
template <bool DUAL, std::uint32_t K>
inline void _two_pass_accumulate_m2_row_if_(const std::uint32_t first, const std::uint32_t last) {
    if (first <= K && last > K) {
        if constexpr (DUAL) {
            constexpr std::uint32_t ACC_LREG = (K & 1U) != 0 ? TWO_PASS_ACC2_REG : TWO_PASS_ACC_REG;
            _two_pass_accumulate_m2_dual_single_<p_sfpu::LREG0 + K /* INPUT_LREG */, ACC_LREG>();
        } else {
            _two_pass_accumulate_m2_single_<p_sfpu::LREG0 + K /* INPUT_LREG */>();
        }
    }
}

/**
 * @brief Load quad (I, J) and process its intersection with rows [start_row, end_row).
 *
 * @tparam ACCUMULATE_M2: Pass two (centred squares) when true, pass one (shifted sums) otherwise.
 * @tparam DUAL: Enables the second LREG6 dependency chain.
 * @tparam I: Face pair, values = <0/1>
 * @tparam J: Quad within the face pair, values = <0..WELFORDS_QUADS_PER_FACE_PAIR-1>
 */
template <bool ACCUMULATE_M2, bool DUAL, std::uint32_t I, std::uint32_t J>
inline void _two_pass_block_rows_(const std::uint32_t start_row, const std::uint32_t end_row) {
    constexpr std::uint32_t LO = (I * FACE_R_DIM) + (J * WELFORDS_QUAD_ROWS);
    constexpr std::uint32_t HI = LO + WELFORDS_QUAD_ROWS;
    if (start_row >= HI || end_row <= LO) {
        return;
    }
    const std::uint32_t first = std::max(LO, start_row) - LO;
    const std::uint32_t last = std::min(HI, end_row) - LO;

    _welfords_load_quad_<I, J>();
    if constexpr (ACCUMULATE_M2) {
        if (first == 0 && last == WELFORDS_QUAD_ROWS) {
            if constexpr (DUAL) {
                _two_pass_accumulate_m2_dual_block_();
            } else {
                _two_pass_accumulate_m2_block_();
            }
            return;
        }
        _two_pass_accumulate_m2_row_if_<DUAL, 0 /* K */>(first, last);
        _two_pass_accumulate_m2_row_if_<DUAL, 1 /* K */>(first, last);
        _two_pass_accumulate_m2_row_if_<DUAL, 2 /* K */>(first, last);
        _two_pass_accumulate_m2_row_if_<DUAL, 3 /* K */>(first, last);
        _two_pass_drain_();  // the next quad's SFPTRANSP reads the accumulator
    } else {
        _two_pass_accumulate_shifted_sum_loaded_block_<DUAL>(first, last);
    }
}

/**
 * @brief Load quad (I, J), copy its first selected row into LREG4 as the anchor, zero the sums and
 *        accumulate the quad's selected rows.
 */
template <bool DUAL, std::uint32_t I, std::uint32_t J>
inline void _two_pass_initialize_anchor_and_accumulate_block_(
    const std::uint32_t start_row, const std::uint32_t end_row) {
    constexpr std::uint32_t LO = (I * FACE_R_DIM) + (J * WELFORDS_QUAD_ROWS);
    constexpr std::uint32_t HI = LO + WELFORDS_QUAD_ROWS;
    if (start_row >= HI || end_row <= LO) {
        return;
    }
    const std::uint32_t first = std::max(LO, start_row) - LO;
    const std::uint32_t last = std::min(HI, end_row) - LO;

    _welfords_load_quad_<I, J>();
    if (first == 0) {
        TTI_SFPMOV(p_sfpu::LREG0, TWO_PASS_MEAN_REG, 0 /* instr_mod1: plain copy */);
    } else if (first == 1) {
        TTI_SFPMOV(p_sfpu::LREG1, TWO_PASS_MEAN_REG, 0 /* instr_mod1: plain copy */);
    } else if (first == 2) {
        TTI_SFPMOV(p_sfpu::LREG2, TWO_PASS_MEAN_REG, 0 /* instr_mod1: plain copy */);
    } else {
        TTI_SFPMOV(p_sfpu::LREG3, TWO_PASS_MEAN_REG, 0 /* instr_mod1: plain copy */);
    }
    _two_pass_zero_<TWO_PASS_ACC_REG>();
    if constexpr (DUAL) {
        _two_pass_zero_<TWO_PASS_ACC2_REG>();
    }
    _two_pass_accumulate_shifted_sum_loaded_block_<DUAL>(first, last);
}

/** @brief Pass-one quad dispatch: the quad holding start_row initialises the anchor when asked to. */
template <bool INITIALIZE_ANCHOR, bool DUAL, std::uint32_t I, std::uint32_t J>
inline void _two_pass_shifted_block_rows_(const std::uint32_t start_row, const std::uint32_t end_row) {
    constexpr std::uint32_t LO = (I * FACE_R_DIM) + (J * WELFORDS_QUAD_ROWS);
    constexpr std::uint32_t HI = LO + WELFORDS_QUAD_ROWS;
    if constexpr (INITIALIZE_ANCHOR) {
        if (start_row >= LO && start_row < HI) {
            _two_pass_initialize_anchor_and_accumulate_block_<DUAL, I, J>(start_row, end_row);
            return;
        }
    }
    _two_pass_block_rows_<false /* ACCUMULATE_M2 */, DUAL, I, J>(start_row, end_row);
}

/**
 * @brief Walk every quad of the tile, top to bottom, for pass one (Q is the flat quad index).
 */
template <bool INITIALIZE_ANCHOR, bool DUAL, std::uint32_t Q = 0>
inline void _two_pass_all_shifted_blocks_(const std::uint32_t start_row, const std::uint32_t end_row) {
    if constexpr (Q < WELFORDS_QUADS_PER_TILE) {
        _two_pass_shifted_block_rows_<
            INITIALIZE_ANCHOR,
            DUAL,
            Q / WELFORDS_QUADS_PER_FACE_PAIR /* I */,
            Q % WELFORDS_QUADS_PER_FACE_PAIR /* J */>(start_row, end_row);
        _two_pass_all_shifted_blocks_<INITIALIZE_ANCHOR, DUAL, Q + 1>(start_row, end_row);
    }
}

/** @brief Walk every quad of the tile, top to bottom, for pass two (Q is the flat quad index). */
template <bool DUAL, std::uint32_t Q = 0>
inline void _two_pass_all_m2_blocks_(const std::uint32_t start_row, const std::uint32_t end_row) {
    if constexpr (Q < WELFORDS_QUADS_PER_TILE) {
        _two_pass_block_rows_<
            true /* ACCUMULATE_M2 */,
            DUAL,
            Q / WELFORDS_QUADS_PER_FACE_PAIR /* I */,
            Q % WELFORDS_QUADS_PER_FACE_PAIR /* J */>(start_row, end_row);
        _two_pass_all_m2_blocks_<DUAL, Q + 1>(start_row, end_row);
    }
}

/**
 * @brief Pass two: accumulate centred M2 over rows [start_row, start_row + num_rows) of the tile.
 *
 * @tparam dual_m2: Odd quad rows accumulate into LREG6, even ones into LREG5.
 * @param start_row: First tile row.
 * @param num_rows: Rows to process; requires start_row + num_rows <= TILE_R_DIM.
 * @note Run once per tile under VectorMode::RC_custom, after @ref _two_pass_finish_shifted_mean_.
 */
template <bool dual_m2>
inline void _two_pass_update_rows_(const std::uint32_t start_row, const std::uint32_t num_rows) {
    if (num_rows == 0) {
        return;
    }
    LLK_ASSERT(num_rows <= TILE_R_DIM && start_row <= TILE_R_DIM - num_rows, "two-pass: row window runs past the tile");
    if (start_row == 0 && num_rows == TILE_R_DIM) {
        // Constant bounds let the compiler drop the per-quad intersection on the full-tile path.
        _two_pass_all_m2_blocks_<dual_m2>(0, TILE_R_DIM);
        return;
    }
    _two_pass_all_m2_blocks_<dual_m2>(start_row, start_row + num_rows);
}

/**
 * @brief Pass one (shifted sums), or pass two when accumulate_m2, over a row window of the tile.
 *
 * @tparam accumulate_m2: Dispatch to @ref _two_pass_update_rows_ when true.
 * @tparam initialize_anchor: Copy the first selected row into LREG4 and zero the sums first; set it
 *         on the first call of a population only.
 * @tparam dual_accumulator: Use LREG6 as a second accumulator.
 * @param start_row: First tile row.
 * @param num_rows: Rows to process; requires start_row + num_rows <= TILE_R_DIM.
 * @note Run once per tile under VectorMode::RC_custom.
 */
template <bool accumulate_m2, bool initialize_anchor, bool dual_accumulator>
inline void _two_pass_update_shifted_rows_(const std::uint32_t start_row, const std::uint32_t num_rows) {
    if constexpr (accumulate_m2) {
        _two_pass_update_rows_<dual_accumulator>(start_row, num_rows);
        return;
    }
    if (num_rows == 0) {
        return;
    }
    LLK_ASSERT(num_rows <= TILE_R_DIM && start_row <= TILE_R_DIM - num_rows, "two-pass: row window runs past the tile");
    if (start_row == 0 && num_rows == TILE_R_DIM) {
        _two_pass_all_shifted_blocks_<initialize_anchor, dual_accumulator>(0, TILE_R_DIM);
        return;
    }
    _two_pass_all_shifted_blocks_<initialize_anchor, dual_accumulator>(start_row, start_row + num_rows);
}

/**
 * @brief mean = anchor + shifted_sum * reciprocal into LREG4, then clear the M2 accumulators.
 *
 * @tparam dual_sum: Fold LREG6 into LREG5 first.
 * @tparam retain_anchor: Keep the anchor in LREG7 for a compensated finaliser; requires dual_sum.
 * @param reciprocal_bits: fp32 bits of 1/population.
 */
template <bool dual_sum, bool retain_anchor = false>
inline void _two_pass_finish_shifted_mean_(const std::uint32_t reciprocal_bits) {
    static_assert(!retain_anchor || dual_sum, "anchor retention requires the dual-accumulator statistics path");
    if constexpr (retain_anchor) {
        TTI_SFPMOV(TWO_PASS_MEAN_REG, p_sfpu::LREG0, 0 /* instr_mod1: plain copy */);
    }
    _two_pass_load_fp32_<p_sfpu::LREG7>(reciprocal_bits);
    if constexpr (dual_sum) {
        _two_pass_fold_dual_();
    }
    TTI_SFPMAD(TWO_PASS_ACC_REG, p_sfpu::LREG7, TWO_PASS_MEAN_REG, TWO_PASS_MEAN_REG, 0 /* instr_mod1 */);
    if constexpr (retain_anchor) {
        TTI_SFPMOV(p_sfpu::LREG0, TWO_PASS_ANCHOR_REG, 0 /* instr_mod1: plain copy */);
    }
    _two_pass_zero_<TWO_PASS_ACC_REG>();
    _two_pass_zero_<TWO_PASS_ACC2_REG>();
}

/** @brief Zero the mean / shifted-sum / M2 state in LREG4-LREG6. */
inline void _two_pass_clear_stats_() {
    _two_pass_zero_<TWO_PASS_MEAN_REG>();
    _two_pass_zero_<TWO_PASS_ACC_REG>();
    _two_pass_zero_<TWO_PASS_ACC2_REG>();
}

/** @brief Store the retained LREG7 anchor at raw offset 0 of the current Dest tile. */
inline void _two_pass_store_anchor_to_dst_() {
    TTI_SFPSTORE(TWO_PASS_ANCHOR_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg_addr */);
}

/** @brief Restore LREG7 from raw offset 0 of the current Dest tile. */
inline void _two_pass_load_anchor_from_dst_() {
    TTI_SFPLOAD(TWO_PASS_ANCHOR_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg_addr */);
}

/** @brief Park the retained anchor in the mean-state tile's unused group slot 1 (raw offset 4). */
inline void _two_pass_store_anchor_to_state_dst_() {
    TTI_SFPSTORE(TWO_PASS_ANCHOR_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, TWO_PASS_ANCHOR_STATE_OFFSET);
}

/** @brief Restore LREG7 from group slot 1 (raw offset 4) of the mean-state tile. */
inline void _two_pass_load_anchor_from_state_dst_() {
    TTI_SFPLOAD(TWO_PASS_ANCHOR_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, TWO_PASS_ANCHOR_STATE_OFFSET);
}

/**
 * @brief Spill mean (LREG4) and M2 (LREG5) to the current tile and the one after it, raw layout.
 *
 * @tparam dual_m2: Fold LREG6 into LREG5 first. LREG6 is cleared afterwards.
 */
template <bool dual_m2>
inline void _two_pass_store_mean_m2_to_dst_() {
    if constexpr (dual_m2) {
        _two_pass_fold_dual_();
        _two_pass_drain_();
    }
    TTI_SFPSTORE(TWO_PASS_MEAN_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg_addr */);
    TTI_SFPSTORE(TWO_PASS_ACC_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_TILE_STRIDE);
    _two_pass_zero_<TWO_PASS_ACC2_REG>();
}

/**
 * @brief Chan-combine the block state in LREG4/LREG5 with the (mean, M2) spilled at the current tile.
 *
 *   mean = mean_a + delta * n_b / (n_a + n_b)
 *   M2   = M2_a + M2_b + delta^2 * n_a * n_b / (n_a + n_b),   delta = mean_b - mean_a
 *
 * The combined state is written back over the spill and left in LREG4/LREG5.
 *
 * @tparam dual_m2: Fold LREG6 into LREG5 first.
 * @param total_reciprocal_bits: fp32 bits of 1 / (n_a + n_b).
 * @param block_n_bits: fp32 bits of n_b, the current block's population.
 */
template <bool dual_m2>
inline void _two_pass_combine_block_to_dst_(
    const std::uint32_t total_reciprocal_bits, const std::uint32_t block_n_bits) {
    if constexpr (dual_m2) {
        _two_pass_fold_dual_();
    }

    // LREG2 = n_b / (n_a + n_b), LREG3 = n_b * (1 - LREG2) = n_a * n_b / (n_a + n_b).
    _two_pass_load_fp32_<p_sfpu::LREG2>(total_reciprocal_bits);
    _two_pass_load_fp32_<p_sfpu::LREG3>(block_n_bits);
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG2, 0 /* instr_mod1 */);
    _two_pass_load_fp32_<p_sfpu::LREG7>(block_n_bits ^ FP32_SIGN_BIT);  // -n_b
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG3, p_sfpu::LREG3, 0 /* instr_mod1 */);

    // The preceding blocks' (mean_a, M2_a) from Dest.
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg_addr */);
    TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_TILE_STRIDE);

    TTI_SFPMAD(p_sfpu::LCONST_neg1, p_sfpu::LREG0, TWO_PASS_MEAN_REG, p_sfpu::LREG7, 0 /* instr_mod1 */);  // delta
    TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, TWO_PASS_ACC_REG, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);  // M2_a + M2_b
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG2, p_sfpu::LREG0, TWO_PASS_MEAN_REG, 0 /* instr_mod1 */);       // mean
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG7, 0 /* instr_mod1 */);        // delta^2
    _two_pass_drain_();
    TTI_SFPSTORE(TWO_PASS_MEAN_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg_addr */);
    TTI_SFPMUL(p_sfpu::LREG7, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG7, 0 /* instr_mod1 */);
    _two_pass_zero_<TWO_PASS_ACC2_REG>();
    TTI_SFPADD(TWO_PASS_ACC_REG, p_sfpu::LCONST_1, p_sfpu::LREG7, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
    _two_pass_drain_();
    TTI_SFPSTORE(TWO_PASS_ACC_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_TILE_STRIDE);
}

/**
 * @brief Store the 4x8 lane vectors of LREG0-3 (after an SFPTRANSP) as one tile row at base.
 */
template <std::uint32_t FIRST_LREG>
inline void _two_pass_store_row_(const std::uint32_t base) {
    TT_SFPSTORE(FIRST_LREG + 0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, base + WELFORDS_LEFT_EVEN);
    TT_SFPSTORE(FIRST_LREG + 1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, base + WELFORDS_LEFT_ODD);
    TT_SFPSTORE(FIRST_LREG + 2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, base + WELFORDS_RIGHT_EVEN);
    TT_SFPSTORE(FIRST_LREG + 3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, base + WELFORDS_RIGHT_ODD);
}

/**
 * @brief Variance = M2 * reciprocal; write mean as tile row 0 of the current tile and variance as
 *        tile row 0 of the next one (rows 1-3 of each face pair's first quad are zero).
 *
 * @tparam dual_m2: Fold LREG6 into LREG5 first.
 * @tparam store_mean: Also write the mean tile; the variance tile is always written.
 * @param reciprocal_bits: fp32 bits of the variance divisor's reciprocal.
 * @note Scrambles LREG0-7; the running state is invalid afterwards.
 */
template <bool dual_m2, bool store_mean = true>
inline void _two_pass_store_mean_var_to_dst_row_(const std::uint32_t reciprocal_bits) {
    if constexpr (dual_m2) {
        _two_pass_fold_dual_();
    }
    if constexpr (store_mean) {
        TTI_SFPMOV(
            TWO_PASS_MEAN_REG, p_sfpu::LREG0, 0 /* instr_mod1: plain copy */);  // save mean before LREG4 is reused
    }
    _two_pass_load_fp32_<p_sfpu::LREG6>(reciprocal_bits);
    TTI_SFPMUL(TWO_PASS_ACC_REG, p_sfpu::LREG6, p_sfpu::LCONST_0, p_sfpu::LREG4, 0 /* instr_mod1 */);  // variance
    if constexpr (store_mean) {
        _two_pass_zero_<p_sfpu::LREG1>();
        _two_pass_zero_<p_sfpu::LREG2>();
        _two_pass_zero_<p_sfpu::LREG3>();
    }
    _two_pass_zero_<p_sfpu::LREG5>();
    _two_pass_zero_<p_sfpu::LREG6>();
    _two_pass_zero_<p_sfpu::LREG7>();
    TTI_SFPTRANSP;  // lane values -> tile row 0 of LREG0-3 (mean) and LREG4-7 (variance)

    if constexpr (store_mean) {
        _two_pass_store_row_<p_sfpu::LREG0>(0);
    }
    _two_pass_store_row_<p_sfpu::LREG4>(WELFORDS_TILE_STRIDE);
}

/**
 * @brief Store the retained anchor, anchor - mean and the variance for a compensated finaliser.
 *
 * Mean tile row 0 is the anchor and row 16 is anchor - mean; the next tile's row 0 is the variance.
 *
 * @tparam dual_m2: Fold LREG6 into LREG5 first.
 * @param reciprocal_bits: fp32 bits of the variance divisor's reciprocal.
 * @note Needs the anchor retained in LREG7 (@ref _two_pass_finish_shifted_mean_ with retain_anchor).
 */
template <bool dual_m2>
inline void _two_pass_store_split_mean_var_to_dst_row_(const std::uint32_t reciprocal_bits) {
    if constexpr (dual_m2) {
        _two_pass_fold_dual_();
    }
    TTI_SFPMOV(TWO_PASS_ANCHOR_REG, p_sfpu::LREG0, 0 /* instr_mod1: plain copy */);  // anchor
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, TWO_PASS_ANCHOR_REG, TWO_PASS_MEAN_REG, 0 /* instr_mod1 */);
    _two_pass_drain_();
    TTI_SFPMOV(TWO_PASS_MEAN_REG, p_sfpu::LREG1, 0 /* instr_mod1: plain copy */);  // anchor - mean
    _two_pass_load_fp32_<p_sfpu::LREG6>(reciprocal_bits);
    TTI_SFPMUL(TWO_PASS_ACC_REG, p_sfpu::LREG6, p_sfpu::LCONST_0, TWO_PASS_MEAN_REG, 0 /* instr_mod1 */);  // variance
    _two_pass_drain_();

    // Park the variance in the next tile while the two transposes form rows 0 and 16 of this one.
    TTI_SFPSTORE(TWO_PASS_MEAN_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_TILE_STRIDE);
    TTI_SFPMOV(p_sfpu::LREG1, p_sfpu::LREG4, 0 /* instr_mod1: plain copy */);
    _two_pass_zero_<p_sfpu::LREG1>();
    _two_pass_zero_<p_sfpu::LREG2>();
    _two_pass_zero_<p_sfpu::LREG3>();
    _two_pass_zero_<p_sfpu::LREG5>();
    _two_pass_zero_<p_sfpu::LREG6>();
    _two_pass_zero_<p_sfpu::LREG7>();
    TTI_SFPTRANSP;
    _two_pass_store_row_<p_sfpu::LREG0>(0);                          // row 0: anchor
    _two_pass_store_row_<p_sfpu::LREG4>(WELFORDS_FACE_PAIR_STRIDE);  // row 16: anchor - mean

    // Expand the parked variance into row 0 of its own tile.
    TTI_SFPLOAD(p_sfpu::LREG4, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_TILE_STRIDE);
    _two_pass_zero_<p_sfpu::LREG5>();
    _two_pass_zero_<p_sfpu::LREG6>();
    _two_pass_zero_<p_sfpu::LREG7>();
    TTI_SFPTRANSP;
    _two_pass_store_row_<p_sfpu::LREG4>(WELFORDS_TILE_STRIDE);
}

/**
 * @brief Variance = M2 * reciprocal; store one group's mean and variance in its raw-face slots.
 *
 * @tparam dual_m2: Fold LREG6 into LREG5 first.
 * @param group_id: Group slot, values = <0..WELFORDS_NUM_GROUPS-1>
 * @param reciprocal_bits: fp32 bits of the variance divisor's reciprocal.
 */
template <bool dual_m2>
inline void _two_pass_store_mean_var_to_dst_raw_group_(
    const std::uint32_t group_id, const std::uint32_t reciprocal_bits) {
    LLK_ASSERT(group_id < WELFORDS_NUM_GROUPS, "two-pass: group_id past the last group slot of the tile");
    if constexpr (dual_m2) {
        _two_pass_fold_dual_();
        _two_pass_drain_();
    }
    _two_pass_load_fp32_<p_sfpu::LREG6>(reciprocal_bits);
    TTI_SFPMUL(TWO_PASS_ACC_REG, p_sfpu::LREG6, p_sfpu::LCONST_0, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
    _two_pass_drain_();
    const std::uint32_t group_offset = group_id << WELFORDS_GROUP_SHIFT;
    TT_SFPSTORE(TWO_PASS_MEAN_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, group_offset);
    TT_SFPSTORE(
        TWO_PASS_ACC_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_TILE_STRIDE + group_offset);
    _two_pass_zero_<TWO_PASS_ACC2_REG>();
}

/**
 * @brief Save the single-accumulator state to one group slot and restore another into LREG4/LREG5.
 *
 * @tparam dual_accumulator: Must be false; LREG6 and a retained LREG7 are not part of a slot.
 */
template <bool dual_accumulator>
inline void _two_pass_switch_group_(const std::uint32_t save_group_id, const std::uint32_t restore_group_id) {
    static_assert(!dual_accumulator, "group switching only preserves the single-accumulator LREG4/LREG5 state");
    welfords_store_mean_m2_to_dst<true /* GROUPED */>(save_group_id);
    welfords_load_mean_m2_from_dst<true /* GROUPED */>(restore_group_id);
}

/** @brief Rotate LREG one SFPU column (X -> X+1, wrapping) and wait for the result. */
template <std::uint32_t LREG>
inline void _two_pass_rotate_() {
    TTI_SFPSHFT2(0 /* imm12 */, LREG, LREG, TWO_PASS_SFPSHFT2_ROTATE);
    _two_pass_drain_();  // SFPSHFT2 takes two cycles
}

/** @brief Rotate LREG by DISTANCE SFPU columns. */
template <std::uint32_t LREG, std::uint32_t DISTANCE>
inline void _two_pass_rotate_by_() {
    if constexpr (DISTANCE > 0) {
        _two_pass_rotate_<LREG>();
        _two_pass_rotate_by_<LREG, DISTANCE - 1>();
    }
}

/** @brief Shift LREG one SFPU column (X -> X+1), zero-filling column 0, and wait for the result. */
template <std::uint32_t LREG, std::uint32_t DISTANCE>
inline void _two_pass_shift_zero_fill_by_() {
    if constexpr (DISTANCE > 0) {
        TTI_SFPSHFT2(0 /* imm12 */, LREG, LREG, TWO_PASS_SFPSHFT2_SHIFT_ZERO_FILL);
        _two_pass_drain_();  // SFPSHFT2 takes two cycles
        _two_pass_shift_zero_fill_by_<LREG, DISTANCE - 1>();
    }
}

/**
 * @brief One stage of the column fold: SUM += SUM rotated by DISTANCE (copy kept in SCRATCH).
 */
template <std::uint32_t SUM, std::uint32_t SCRATCH, std::uint32_t DISTANCE>
inline void _two_pass_fold_stage_() {
    TTI_SFPMOV(SUM, SCRATCH, 0 /* instr_mod1: plain copy */);
    _two_pass_rotate_by_<SCRATCH, DISTANCE>();
    TTI_SFPADD(SUM, p_sfpu::LCONST_1, SCRATCH, SUM, 0 /* instr_mod1 */);
    _two_pass_drain_();
}

/**
 * @brief Fold the 8 SFPU columns of SUM together: afterwards every column holds its row's total.
 */
template <std::uint32_t SUM, std::uint32_t SCRATCH>
inline void _two_pass_fold_columns_() {
    _two_pass_fold_stage_<SUM, SCRATCH, TWO_PASS_SFPU_COLUMNS / 2>();
    _two_pass_fold_stage_<SUM, SCRATCH, TWO_PASS_SFPU_COLUMNS / 4>();
    _two_pass_fold_stage_<SUM, SCRATCH, TWO_PASS_SFPU_COLUMNS / 8>();
}

/**
 * @brief Horizontally sum LREG0 and LREG4 over all 32 lanes.
 *
 * @tparam broadcast_result: Broadcast each total to every lane; otherwise the totals are valid in
 *         sub-row 0 only (LREG0-3 and LREG5-7 are clobbered either way).
 */
template <bool broadcast_result>
inline void _two_pass_horizontal_sum_pair_() {
    _two_pass_fold_columns_<p_sfpu::LREG0, p_sfpu::LREG1>();
    _two_pass_fold_columns_<p_sfpu::LREG4, p_sfpu::LREG5>();

    // Transpose the four row totals into LREG0-3 / LREG4-7 sub-row 0, then add them.
    _two_pass_zero_<p_sfpu::LREG1>();
    _two_pass_zero_<p_sfpu::LREG2>();
    _two_pass_zero_<p_sfpu::LREG3>();
    _two_pass_zero_<p_sfpu::LREG5>();
    _two_pass_zero_<p_sfpu::LREG6>();
    _two_pass_zero_<p_sfpu::LREG7>();
    TTI_SFPTRANSP;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0 /* instr_mod1 */);
    TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0 /* instr_mod1 */);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /* instr_mod1 */);
    TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG4, 0 /* instr_mod1 */);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG0, 0 /* instr_mod1 */);
    TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG4, 0 /* instr_mod1 */);
    _two_pass_drain_();

    if constexpr (broadcast_result) {
        TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG1, 0 /* instr_mod1: plain copy */);
        TTI_SFPMOV(p_sfpu::LREG4, p_sfpu::LREG5, 0 /* instr_mod1: plain copy */);
        TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG2, 0 /* instr_mod1: plain copy */);
        TTI_SFPMOV(p_sfpu::LREG4, p_sfpu::LREG6, 0 /* instr_mod1: plain copy */);
        TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG3, 0 /* instr_mod1: plain copy */);
        TTI_SFPMOV(p_sfpu::LREG4, p_sfpu::LREG7, 0 /* instr_mod1: plain copy */);
        TTI_SFPTRANSP;
    }
}

/**
 * @brief Horizontally sum LREG0 over all 32 lanes; the total is valid in sub-row 0 only.
 *
 * The transpose also reduces a broadcast LREG4 to the same sub-row-0 layout (LREG5-7 are zeroed).
 */
inline void _two_pass_horizontal_sum_mean_() {
    _two_pass_fold_columns_<p_sfpu::LREG0, p_sfpu::LREG1>();

    _two_pass_zero_<p_sfpu::LREG1>();
    _two_pass_zero_<p_sfpu::LREG2>();
    _two_pass_zero_<p_sfpu::LREG3>();
    _two_pass_zero_<p_sfpu::LREG5>();
    _two_pass_zero_<p_sfpu::LREG6>();
    _two_pass_zero_<p_sfpu::LREG7>();
    TTI_SFPTRANSP;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0 /* instr_mod1 */);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /* instr_mod1 */);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG0, 0 /* instr_mod1 */);
    _two_pass_drain_();
}

/**
 * @brief Broadcast lane (sub-row 0, SFPU column 0) of LREG0 to all 32 lanes; LREG4-7 are kept.
 *
 * Seven zero-fill column shifts move column 0 to column 7 and leave every other column zero, in
 * place of Blackhole's LTILEID mask.
 */
inline void _two_pass_broadcast_one_lane_() {
    _two_pass_shift_zero_fill_by_<p_sfpu::LREG0, TWO_PASS_SFPU_COLUMNS - 1>();  // column 0 -> column 7, rest 0

    // The first transpose isolates sub-row 0 in LREG0; replicate it before transposing back. The
    // second transpose also restores LREG4-7.
    _two_pass_zero_<p_sfpu::LREG1>();
    _two_pass_zero_<p_sfpu::LREG2>();
    _two_pass_zero_<p_sfpu::LREG3>();
    TTI_SFPTRANSP;
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG1, 0 /* instr_mod1: plain copy */);
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG2, 0 /* instr_mod1: plain copy */);
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG3, 0 /* instr_mod1: plain copy */);
    TTI_SFPTRANSP;

    // Spread the single non-zero column: the columns stay disjoint, so no two anchors are added.
    _two_pass_fold_columns_<p_sfpu::LREG0, p_sfpu::LREG1>();
}

/**
 * @brief Finalise lane-local statistics, combine the 32 equal lane populations and store one group.
 *
 * Total variance = mean lane variance + variance of the lane means. The lane means are centred on
 * one lane first, so the cross-lane sums do not cancel. The mean is stored broadcast to every lane
 * of the group slot; the variance is valid in sub-row 0 (the first 8 lanes) of the slot.
 *
 * @tparam dual_m2: Fold LREG6 into LREG5 first.
 * @tparam average_variance: Store the total variance; otherwise store 32 times it (the lane sum).
 * @param group_id: Group slot, values = <0..WELFORDS_NUM_GROUPS-1>
 * @param reciprocal_bits: fp32 bits of 1 / (per-lane population).
 * @note Writes mean to the current tile and variance to the next one, and clobbers the tile after
 *       that as scratch.
 */
template <bool dual_m2, bool average_variance = true>
inline void _two_pass_store_combined_mean_var_to_dst_raw_group_(
    const std::uint32_t group_id, const std::uint32_t reciprocal_bits) {
    LLK_ASSERT(group_id < WELFORDS_NUM_GROUPS, "two-pass: group_id past the last group slot of the tile");
    if constexpr (dual_m2) {
        _two_pass_fold_dual_();
    }

    // Each lane's M2 -> variance, then centre each lane's mean on one lane's mean.
    _two_pass_load_fp32_<p_sfpu::LREG6>(reciprocal_bits);
    TTI_SFPMUL(TWO_PASS_ACC_REG, p_sfpu::LREG6, p_sfpu::LCONST_0, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
    constexpr std::uint32_t SCRATCH = 2 * WELFORDS_TILE_STRIDE;
    constexpr std::uint32_t SCRATCH_ANCHOR = SCRATCH + TWO_PASS_ANCHOR_STATE_OFFSET;

    TTI_SFPMOV(TWO_PASS_MEAN_REG, p_sfpu::LREG0, 0 /* instr_mod1: plain copy */);
    _two_pass_broadcast_one_lane_();
    // Write both column parities, so every lane of the slot is defined.
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, SCRATCH_ANCHOR + WELFORDS_LEFT_EVEN);
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, SCRATCH_ANCHOR + WELFORDS_LEFT_ODD);
    TTI_SFPMAD(p_sfpu::LCONST_neg1, p_sfpu::LREG0, TWO_PASS_MEAN_REG, p_sfpu::LREG0, 0 /* instr_mod1 */);
    _two_pass_drain_();
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, SCRATCH + WELFORDS_LEFT_EVEN);
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, SCRATCH + WELFORDS_LEFT_ODD);
    // Keep the lane-variance sum unscaled, so the final 1/32 applies once to both variance terms.
    TTI_SFPMOV(TWO_PASS_ACC_REG, p_sfpu::LREG4, 0 /* instr_mod1: plain copy */);
    _two_pass_horizontal_sum_pair_<true /* broadcast_result */>();

    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, TWO_PASS_LANE_RECIPROCAL_FP16B);
    TTI_SFPMUL(
        p_sfpu::LREG0, p_sfpu::LREG6, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /* instr_mod1 */);  // mean of centred means

    // Variance of the centred lane means, then the absolute mean.
    TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, SCRATCH + WELFORDS_LEFT_EVEN);
    TTI_SFPLOAD(p_sfpu::LREG7, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, SCRATCH_ANCHOR + WELFORDS_LEFT_EVEN);
    TTI_SFPMAD(p_sfpu::LCONST_neg1, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG1, 0 /* instr_mod1 */);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG7, 0 /* instr_mod1 */);
    _two_pass_drain_();
    const std::uint32_t group_offset = group_id << WELFORDS_GROUP_SHIFT;
    TT_SFPSTORE(p_sfpu::LREG7, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, group_offset + WELFORDS_LEFT_EVEN);
    TT_SFPSTORE(p_sfpu::LREG7, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, group_offset + WELFORDS_LEFT_ODD);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /* instr_mod1 */);
    _two_pass_drain_();
    _two_pass_horizontal_sum_mean_();

    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
    if constexpr (average_variance) {
        TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, TWO_PASS_LANE_RECIPROCAL_FP16B);
        TTI_SFPMUL(TWO_PASS_ACC_REG, p_sfpu::LREG6, p_sfpu::LCONST_0, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
    }
    _two_pass_drain_();
    TT_SFPSTORE(
        TWO_PASS_ACC_REG,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        0 /* done */,
        WELFORDS_TILE_STRIDE + group_offset + WELFORDS_LEFT_EVEN);
    TT_SFPSTORE(
        TWO_PASS_ACC_REG,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        0 /* done */,
        WELFORDS_TILE_STRIDE + group_offset + WELFORDS_LEFT_ODD);
}

}  // namespace sfpu
}  // namespace ckernel
