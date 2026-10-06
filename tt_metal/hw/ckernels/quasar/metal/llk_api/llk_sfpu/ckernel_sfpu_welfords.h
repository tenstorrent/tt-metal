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

}  // namespace sfpu
}  // namespace ckernel
