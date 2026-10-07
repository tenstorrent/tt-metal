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
constexpr std::uint32_t WELFORDS_MEAN_REG = p_sfpu::LREG4;
constexpr std::uint32_t WELFORDS_M2_REG = p_sfpu::LREG5;
constexpr std::uint32_t WELFORDS_ALPHA_REG = p_sfpu::LREG6;     // alpha = x - mean_N
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
constexpr std::uint32_t WELFORDS_NUM_GROUPS = WELFORDS_TILE_STRIDE >> WELFORDS_GROUP_SHIFT;

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

constexpr std::uint32_t FP16B_ZERO = 0x0000;
constexpr std::uint32_t FP32_HI16_SHIFT = 16;     // fp32 bits [31:16] -> MOD0_UPPER immediate
constexpr std::uint32_t FP32_LO16_MASK = 0xFFFF;  // fp32 bits [15:0]  -> MOD0_LOWER immediate

enum class WelfordsOutputLayout : std::uint8_t { Row, Face };

/**
 * @brief Load 1/(idx+1) into WELFORDS_RECIP_REG.
 * @tparam RECIPROCAL_SIZE: Length of reciprocal_lut; 0 divides on the RISC-V instead.
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
 * @brief Fold one tile row into the running mean and M2; expects 1/(N+1) in WELFORDS_RECIP_REG.
 * @note Dependent MADs are interlocked by hardware, so no SFPNOP sits between them.
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

/** @brief Replay the row body that @ref welfords_init recorded for INPUT_LREG. */
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
 * @brief Load tile rows 4J..4J+3 of face pair I into LREG0-3.
 * @note SFPTRANSP also permutes LREG4-7; the second one restores the running state.
 */
template <std::uint32_t I, std::uint32_t J>
inline void _welfords_load_quad_() {
    constexpr std::uint32_t BASE = (I * WELFORDS_FACE_PAIR_STRIDE) + (J * WELFORDS_QUAD_ROWS);

    TTI_SFPTRANSP;
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, BASE + WELFORDS_LEFT_EVEN);
    TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, BASE + WELFORDS_LEFT_ODD);
    TTI_SFPLOAD(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, BASE + WELFORDS_RIGHT_EVEN);
    TTI_SFPLOAD(p_sfpu::LREG3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, BASE + WELFORDS_RIGHT_ODD);
    TTI_SFPTRANSP;
}

/** @brief Fold all four rows of quad (I, J); start_idx is the sample index of tile row 0. */
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

/** @brief Fold row K of the loaded quad if it lies in [s, e); advances idx when it does. */
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

/** @brief Fold the rows of quad (I, J) inside [start_row, end_row); skips the load when none are. */
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
 * @brief Fold every quad of the tile, top to bottom (Q is the flat quad index).
 * @note Unrolled at compile time because each quad addresses Dest with its own immediate.
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

/** @brief Windowed @ref _calculate_welfords_all_quads_: folds only rows [start_row, end_row). */
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
 * @brief Record the four row bodies into replay slot 0.
 * @note Re-run after any other SFPU op records its own replay on this thread.
 */
inline void welfords_init() {
    load_replay_buf<WELFORDS_REPLAY_SLOT, WELFORDS_REPLAY_LEN, false /* exec_while_loading */>([] {
        _welfords_row_<p_sfpu::LREG0 /* INPUT_LREG */>();
        _welfords_row_<p_sfpu::LREG1 /* INPUT_LREG */>();
        _welfords_row_<p_sfpu::LREG2 /* INPUT_LREG */>();
        _welfords_row_<p_sfpu::LREG3 /* INPUT_LREG */>();
    });
}

/** @brief Zero the running mean (LREG4) and M2 (LREG5). */
inline void welfords_clear_previous_mean_and_m2() {
    TTI_SFPLOADI(WELFORDS_MEAN_REG, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ZERO);
    TTI_SFPLOADI(WELFORDS_M2_REG, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ZERO);
}

/**
 * @brief Fold one tile (or, when PARTIAL_TILE, rows [start_row, start_row + num_rows)) into the state.
 * @note The state carries across calls in LREG4/LREG5; write nothing to LREG4-7 between tiles.
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
            "welfords: partial row window runs past the tile");  // overflow-safe form
        const std::uint32_t end_row = start_row + num_rows;
        std::uint32_t idx = start_idx;
        _calculate_welfords_all_quad_rows_<RECIPROCAL_SIZE>(idx, start_row, end_row, reciprocal_lut);
    }
}

/** @brief Save mean to this tile and M2 to the next; GROUPED uses group slot group_id (0..15). */
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

/** @brief Restore the state saved by @ref welfords_store_mean_m2_to_dst. */
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
 * @brief Write mean to this tile and variance = M2 / (scale_idx + 1) to the next (Row or raw Face).
 * @note Invalidates the running state either way (Row scrambles LREG0-7, Face overwrites M2).
 */
template <WelfordsOutputLayout LAYOUT, bool GROUPED, std::size_t RECIPROCAL_SIZE>
inline void welfords_store_mean_var_to_dst(
    const std::uint32_t scale_idx,
    const std::array<std::uint32_t, RECIPROCAL_SIZE>& reciprocal_lut,
    [[maybe_unused]] const std::uint32_t group_id = 0) {
    static_assert(!(LAYOUT == WelfordsOutputLayout::Row && GROUPED), "the Row layout has no grouped variant");

    _welfords_load_recip_<RECIPROCAL_SIZE>(scale_idx, reciprocal_lut);

    if constexpr (LAYOUT == WelfordsOutputLayout::Row) {
        TTI_SFPMOV(WELFORDS_MEAN_REG, p_sfpu::LREG0, 0 /* instr_mod1: plain copy */);
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
        TTI_SFPTRANSP;  // lane values -> tile row 0 (mean in LREG0-3, variance in LREG4-7)

        TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_LEFT_EVEN);
        TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_LEFT_ODD);
        TTI_SFPSTORE(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_RIGHT_EVEN);
        TTI_SFPSTORE(p_sfpu::LREG3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_RIGHT_ODD);

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

// Shifted two-pass column statistics, ported from tt-llk/common/ckernel_sfpu_welfords_common.h with
// the same names, signatures and register protocol so the compute-API wiring maps one to one.

constexpr std::uint32_t TWO_PASS_MEAN_REG = p_sfpu::LREG4;                          // anchor in pass one, then mean
constexpr std::uint32_t TWO_PASS_ACC_REG = p_sfpu::LREG5;                           // shifted sum / M2 (even rows)
constexpr std::uint32_t TWO_PASS_ACC2_REG = p_sfpu::LREG6;                          // dual accumulator (odd rows)
constexpr std::uint32_t TWO_PASS_ANCHOR_REG = p_sfpu::LREG7;                        // retained anchor
constexpr std::uint32_t TWO_PASS_ANCHOR_STATE_OFFSET = 1U << WELFORDS_GROUP_SHIFT;  // unused slot 1 of the mean tile
constexpr std::uint32_t TWO_PASS_SFPU_COLUMNS = 8;
constexpr std::uint32_t TWO_PASS_SFPSHFT2_ROTATE = 3;             // column X -> X+1, wrapping
constexpr std::uint32_t TWO_PASS_SFPSHFT2_SHIFT_ZERO_FILL = 4;    // column X -> X+1, column 0 <- 0
constexpr std::uint32_t TWO_PASS_LANE_RECIPROCAL_FP16B = 0x3D00;  // 1/32
constexpr std::uint32_t FP32_SIGN_BIT = 0x80000000U;

/** @brief Wait out the previous result's latency before an SFPTRANSP, SFPSTORE or SFPSHFT2 reader. */
inline void _two_pass_drain_() { TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */); }

/** @brief Load fp32 bit pattern bits into LREG. */
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

/** @brief LREG5 += LREG6. */
inline void _two_pass_fold_dual_() {
    TTI_SFPADD(TWO_PASS_ACC_REG, p_sfpu::LCONST_1, TWO_PASS_ACC2_REG, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
}

/** @brief ACC_LREG += INPUT_LREG - anchor (clobbers INPUT_LREG). */
template <std::uint32_t INPUT_LREG, std::uint32_t ACC_LREG>
inline void _two_pass_accumulate_shifted_sum_row_() {
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, INPUT_LREG, INPUT_LREG, 0 /* instr_mod1 */);
    TTI_SFPADD(ACC_LREG, p_sfpu::LCONST_1, INPUT_LREG, ACC_LREG, 0 /* instr_mod1 */);
}

/** @brief Pass one over a whole loaded quad, alternating the LREG5/LREG6 chains. */
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

/** @brief Pass one for row K of the loaded quad if it lies in [first, last); DUAL sends odd rows to LREG6. */
template <bool DUAL, std::uint32_t K>
inline void _two_pass_accumulate_shifted_sum_row_if_(const std::uint32_t first, const std::uint32_t last) {
    if (first <= K && last > K) {
        constexpr std::uint32_t ACC_LREG = (DUAL && (K & 1U) != 0) ? TWO_PASS_ACC2_REG : TWO_PASS_ACC_REG;
        _two_pass_accumulate_shifted_sum_row_<p_sfpu::LREG0 + K /* INPUT_LREG */, ACC_LREG>();
    }
}

/** @brief Pass one over rows [first, last) of the loaded quad. */
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

/** @brief LREG5 += (INPUT_LREG - mean)^2, with LREG6 as the residual scratch. */
template <std::uint32_t INPUT_LREG>
inline void _two_pass_accumulate_m2_single_() {
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, INPUT_LREG, TWO_PASS_ACC2_REG, 0 /* instr_mod1 */);
    TTI_SFPMAD(TWO_PASS_ACC2_REG, TWO_PASS_ACC2_REG, TWO_PASS_ACC_REG, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
}

/**
 * @brief Pass two over a whole loaded quad into LREG5, in row order.
 * @note Residuals alternate LREG6 and the dead LREG0 so a retained anchor in LREG7 survives.
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

/** @brief ACC_LREG += (INPUT_LREG - mean)^2 (clobbers INPUT_LREG). */
template <std::uint32_t INPUT_LREG, std::uint32_t ACC_LREG>
inline void _two_pass_accumulate_m2_dual_single_() {
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, INPUT_LREG, INPUT_LREG, 0 /* instr_mod1 */);
    TTI_SFPMAD(INPUT_LREG, INPUT_LREG, ACC_LREG, ACC_LREG, 0 /* instr_mod1 */);
}

/** @brief Pass two over a whole loaded quad, alternating the LREG5/LREG6 chains. */
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

/** @brief Pass two for row K of the loaded quad if it lies in [first, last). */
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

/** @brief Load quad (I, J) and run pass one, or pass two when ACCUMULATE_M2, over its rows in the window. */
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

/** @brief Pass one for quad (I, J), first taking its first selected row as the anchor and zeroing the sums. */
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

/** @brief Pass-one quad dispatch; the quad holding start_row initialises the anchor when asked to. */
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

/** @brief Pass one over every quad of the tile (Q is the flat quad index). */
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

/** @brief Pass two over every quad of the tile (Q is the flat quad index). */
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

/** @brief Pass two (centred M2) over rows [start_row, start_row + num_rows) of the tile. */
template <bool dual_m2>
inline void _two_pass_update_rows_(const std::uint32_t start_row, const std::uint32_t num_rows) {
    if (num_rows == 0) {
        return;
    }
    LLK_ASSERT(num_rows <= TILE_R_DIM && start_row <= TILE_R_DIM - num_rows, "two-pass: row window runs past the tile");
    if (start_row == 0 && num_rows == TILE_R_DIM) {
        // Constant bounds let the compiler drop the per-quad window checks.
        _two_pass_all_m2_blocks_<dual_m2>(0, TILE_R_DIM);
        return;
    }
    _two_pass_all_m2_blocks_<dual_m2>(start_row, start_row + num_rows);
}

/**
 * @brief Pass one (shifted sums), or pass two when accumulate_m2, over a row window of the tile.
 * @tparam initialize_anchor: Set on the first call of a population only.
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
 * @brief LREG4 = anchor + shifted_sum / N, then clear the M2 accumulators.
 * @tparam retain_anchor: Keep the anchor in LREG7 for the split finaliser.
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

/** @brief Zero LREG4-LREG6. */
inline void _two_pass_clear_stats_() {
    _two_pass_zero_<TWO_PASS_MEAN_REG>();
    _two_pass_zero_<TWO_PASS_ACC_REG>();
    _two_pass_zero_<TWO_PASS_ACC2_REG>();
}

/** @brief Store the retained anchor at raw offset 0 of this tile. */
inline void _two_pass_store_anchor_to_dst_() {
    TTI_SFPSTORE(TWO_PASS_ANCHOR_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg_addr */);
}

/** @brief Reload the retained anchor from raw offset 0 of this tile. */
inline void _two_pass_load_anchor_from_dst_() {
    TTI_SFPLOAD(TWO_PASS_ANCHOR_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg_addr */);
}

/** @brief Store the retained anchor in the mean-state tile's unused slot 1. */
inline void _two_pass_store_anchor_to_state_dst_() {
    TTI_SFPSTORE(TWO_PASS_ANCHOR_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, TWO_PASS_ANCHOR_STATE_OFFSET);
}

/** @brief Reload the retained anchor from the mean-state tile's slot 1. */
inline void _two_pass_load_anchor_from_state_dst_() {
    TTI_SFPLOAD(TWO_PASS_ANCHOR_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, TWO_PASS_ANCHOR_STATE_OFFSET);
}

/** @brief Spill mean to this tile and M2 to the next, raw layout. */
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
 * @brief Chan-combine the block state in LREG4/LREG5 with the spill at this tile, in place and in LREG4/5.
 * @note Takes fp32 bits of 1 / (n_a + n_b) and of n_b, so the RISC-V never divides.
 */
template <bool dual_m2>
inline void _two_pass_combine_block_to_dst_(
    const std::uint32_t total_reciprocal_bits, const std::uint32_t block_n_bits) {
    if constexpr (dual_m2) {
        _two_pass_fold_dual_();
    }

    // LREG2 = n_b / (n_a + n_b), LREG3 = n_a * n_b / (n_a + n_b).
    _two_pass_load_fp32_<p_sfpu::LREG2>(total_reciprocal_bits);
    _two_pass_load_fp32_<p_sfpu::LREG3>(block_n_bits);
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG2, 0 /* instr_mod1 */);
    _two_pass_load_fp32_<p_sfpu::LREG7>(block_n_bits ^ FP32_SIGN_BIT);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG3, p_sfpu::LREG3, 0 /* instr_mod1 */);

    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg_addr */);
    TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_TILE_STRIDE);

    TTI_SFPMAD(p_sfpu::LCONST_neg1, p_sfpu::LREG0, TWO_PASS_MEAN_REG, p_sfpu::LREG7, 0 /* instr_mod1 */);  // delta
    TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, TWO_PASS_ACC_REG, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG2, p_sfpu::LREG0, TWO_PASS_MEAN_REG, 0 /* instr_mod1 */);
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG7, 0 /* instr_mod1 */);
    _two_pass_drain_();
    TTI_SFPSTORE(TWO_PASS_MEAN_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg_addr */);
    TTI_SFPMUL(p_sfpu::LREG7, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG7, 0 /* instr_mod1 */);
    _two_pass_zero_<TWO_PASS_ACC2_REG>();
    TTI_SFPADD(TWO_PASS_ACC_REG, p_sfpu::LCONST_1, p_sfpu::LREG7, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
    _two_pass_drain_();
    TTI_SFPSTORE(TWO_PASS_ACC_REG, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_TILE_STRIDE);
}

/** @brief Store transposed LREGs FIRST_LREG..+3 as one tile row at base. */
template <std::uint32_t FIRST_LREG>
inline void _two_pass_store_row_(const std::uint32_t base) {
    TT_SFPSTORE(FIRST_LREG + 0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, base + WELFORDS_LEFT_EVEN);
    TT_SFPSTORE(FIRST_LREG + 1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, base + WELFORDS_LEFT_ODD);
    TT_SFPSTORE(FIRST_LREG + 2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, base + WELFORDS_RIGHT_EVEN);
    TT_SFPSTORE(FIRST_LREG + 3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, base + WELFORDS_RIGHT_ODD);
}

/**
 * @brief Write mean as row 0 of this tile (when store_mean) and M2 * reciprocal as row 0 of the next.
 * @note Scrambles LREG0-7.
 */
template <bool dual_m2, bool store_mean = true>
inline void _two_pass_store_mean_var_to_dst_row_(const std::uint32_t reciprocal_bits) {
    if constexpr (dual_m2) {
        _two_pass_fold_dual_();
    }
    if constexpr (store_mean) {
        TTI_SFPMOV(TWO_PASS_MEAN_REG, p_sfpu::LREG0, 0 /* instr_mod1: plain copy */);
    }
    _two_pass_load_fp32_<p_sfpu::LREG6>(reciprocal_bits);
    TTI_SFPMUL(TWO_PASS_ACC_REG, p_sfpu::LREG6, p_sfpu::LCONST_0, p_sfpu::LREG4, 0 /* instr_mod1 */);
    if constexpr (store_mean) {
        _two_pass_zero_<p_sfpu::LREG1>();
        _two_pass_zero_<p_sfpu::LREG2>();
        _two_pass_zero_<p_sfpu::LREG3>();
    }
    _two_pass_zero_<p_sfpu::LREG5>();
    _two_pass_zero_<p_sfpu::LREG6>();
    _two_pass_zero_<p_sfpu::LREG7>();
    TTI_SFPTRANSP;

    if constexpr (store_mean) {
        _two_pass_store_row_<p_sfpu::LREG0>(0);
    }
    _two_pass_store_row_<p_sfpu::LREG4>(WELFORDS_TILE_STRIDE);
}

/**
 * @brief Write row 0 = anchor and row 16 = anchor - mean to this tile, variance as row 0 of the next.
 * @note Needs the anchor retained in LREG7.
 */
template <bool dual_m2>
inline void _two_pass_store_split_mean_var_to_dst_row_(const std::uint32_t reciprocal_bits) {
    if constexpr (dual_m2) {
        _two_pass_fold_dual_();
    }
    TTI_SFPMOV(TWO_PASS_ANCHOR_REG, p_sfpu::LREG0, 0 /* instr_mod1: plain copy */);
    TTI_SFPMAD(p_sfpu::LCONST_neg1, TWO_PASS_MEAN_REG, TWO_PASS_ANCHOR_REG, TWO_PASS_MEAN_REG, 0 /* instr_mod1 */);
    _two_pass_drain_();
    TTI_SFPMOV(TWO_PASS_MEAN_REG, p_sfpu::LREG1, 0 /* instr_mod1: plain copy */);
    _two_pass_load_fp32_<p_sfpu::LREG6>(reciprocal_bits);
    TTI_SFPMUL(TWO_PASS_ACC_REG, p_sfpu::LREG6, p_sfpu::LCONST_0, TWO_PASS_MEAN_REG, 0 /* instr_mod1 */);
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
    _two_pass_store_row_<p_sfpu::LREG0>(0);
    _two_pass_store_row_<p_sfpu::LREG4>(WELFORDS_FACE_PAIR_STRIDE);

    TTI_SFPLOAD(p_sfpu::LREG4, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, WELFORDS_TILE_STRIDE);
    _two_pass_zero_<p_sfpu::LREG5>();
    _two_pass_zero_<p_sfpu::LREG6>();
    _two_pass_zero_<p_sfpu::LREG7>();
    TTI_SFPTRANSP;
    _two_pass_store_row_<p_sfpu::LREG4>(WELFORDS_TILE_STRIDE);
}

/** @brief Store mean and M2 * reciprocal in group slot group_id (0..15) of this tile and the next. */
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

/** @brief Save LREG4/LREG5 to one group slot and restore another; single accumulator only. */
template <bool dual_accumulator>
inline void _two_pass_switch_group_(const std::uint32_t save_group_id, const std::uint32_t restore_group_id) {
    static_assert(!dual_accumulator, "group switching only preserves the single-accumulator LREG4/LREG5 state");
    welfords_store_mean_m2_to_dst<true /* GROUPED */>(save_group_id);
    welfords_load_mean_m2_from_dst<true /* GROUPED */>(restore_group_id);
}

/** @brief Rotate LREG by one SFPU column. */
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

/** @brief Shift LREG by DISTANCE SFPU columns, zero-filling. */
template <std::uint32_t LREG, std::uint32_t DISTANCE>
inline void _two_pass_shift_zero_fill_by_() {
    if constexpr (DISTANCE > 0) {
        TTI_SFPSHFT2(0 /* imm12 */, LREG, LREG, TWO_PASS_SFPSHFT2_SHIFT_ZERO_FILL);
        _two_pass_drain_();  // SFPSHFT2 takes two cycles
        _two_pass_shift_zero_fill_by_<LREG, DISTANCE - 1>();
    }
}

/** @brief SUM += SUM rotated by DISTANCE columns. */
template <std::uint32_t SUM, std::uint32_t SCRATCH, std::uint32_t DISTANCE>
inline void _two_pass_fold_stage_() {
    TTI_SFPMOV(SUM, SCRATCH, 0 /* instr_mod1: plain copy */);
    _two_pass_rotate_by_<SCRATCH, DISTANCE>();
    TTI_SFPADD(SUM, p_sfpu::LCONST_1, SCRATCH, SUM, 0 /* instr_mod1 */);
    _two_pass_drain_();
}

/** @brief Afterwards every SFPU column of SUM holds its row's total. */
template <std::uint32_t SUM, std::uint32_t SCRATCH>
inline void _two_pass_fold_columns_() {
    _two_pass_fold_stage_<SUM, SCRATCH, TWO_PASS_SFPU_COLUMNS / 2>();
    _two_pass_fold_stage_<SUM, SCRATCH, TWO_PASS_SFPU_COLUMNS / 4>();
    _two_pass_fold_stage_<SUM, SCRATCH, TWO_PASS_SFPU_COLUMNS / 8>();
}

/**
 * @brief Sum LREG0 and LREG4 over all 32 lanes; valid in sub-row 0 unless broadcast_result.
 */
template <bool broadcast_result>
inline void _two_pass_horizontal_sum_pair_() {
    _two_pass_fold_columns_<p_sfpu::LREG0, p_sfpu::LREG1>();
    _two_pass_fold_columns_<p_sfpu::LREG4, p_sfpu::LREG5>();

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
 * @brief Sum LREG0 over all 32 lanes into sub-row 0.
 * @note The transpose also leaves a broadcast LREG4 valid in sub-row 0 only.
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
 * @brief Broadcast one lane of LREG0 to all 32 lanes, keeping LREG4-7.
 * @note Quasar has no documented lane-id register, so zero-fill shifts replace Blackhole's LTILEID mask.
 */
inline void _two_pass_broadcast_one_lane_() {
    _two_pass_shift_zero_fill_by_<p_sfpu::LREG0, TWO_PASS_SFPU_COLUMNS - 1>();  // only column 7 non-zero

    // Replicate sub-row 0 across the transposed LREG0-3, then transpose back.
    _two_pass_zero_<p_sfpu::LREG1>();
    _two_pass_zero_<p_sfpu::LREG2>();
    _two_pass_zero_<p_sfpu::LREG3>();
    TTI_SFPTRANSP;
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG1, 0 /* instr_mod1: plain copy */);
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG2, 0 /* instr_mod1: plain copy */);
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG3, 0 /* instr_mod1: plain copy */);
    TTI_SFPTRANSP;

    // The columns stay disjoint while spreading, so no two values are ever added.
    _two_pass_fold_columns_<p_sfpu::LREG0, p_sfpu::LREG1>();
}

/**
 * @brief Combine the 32 lane populations and store one group's mean (all lanes) and variance (sub-row 0).
 * @note Lane means are centred on one lane first; the tile after the variance tile is scratch.
 */
template <bool dual_m2, bool average_variance = true>
inline void _two_pass_store_combined_mean_var_to_dst_raw_group_(
    const std::uint32_t group_id, const std::uint32_t reciprocal_bits) {
    LLK_ASSERT(group_id < WELFORDS_NUM_GROUPS, "two-pass: group_id past the last group slot of the tile");
    if constexpr (dual_m2) {
        _two_pass_fold_dual_();
    }

    _two_pass_load_fp32_<p_sfpu::LREG6>(reciprocal_bits);
    TTI_SFPMUL(TWO_PASS_ACC_REG, p_sfpu::LREG6, p_sfpu::LCONST_0, TWO_PASS_ACC_REG, 0 /* instr_mod1 */);
    constexpr std::uint32_t SCRATCH = 2 * WELFORDS_TILE_STRIDE;
    constexpr std::uint32_t SCRATCH_ANCHOR = SCRATCH + TWO_PASS_ANCHOR_STATE_OFFSET;

    TTI_SFPMOV(TWO_PASS_MEAN_REG, p_sfpu::LREG0, 0 /* instr_mod1: plain copy */);
    _two_pass_broadcast_one_lane_();
    // Both column parities, so every lane of the slot is defined.
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, SCRATCH_ANCHOR + WELFORDS_LEFT_EVEN);
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, SCRATCH_ANCHOR + WELFORDS_LEFT_ODD);
    TTI_SFPMAD(p_sfpu::LCONST_neg1, p_sfpu::LREG0, TWO_PASS_MEAN_REG, p_sfpu::LREG0, 0 /* instr_mod1 */);
    _two_pass_drain_();
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, SCRATCH + WELFORDS_LEFT_EVEN);
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, SCRATCH + WELFORDS_LEFT_ODD);
    // Unscaled, so the final 1/32 applies once to both variance terms.
    TTI_SFPMOV(TWO_PASS_ACC_REG, p_sfpu::LREG4, 0 /* instr_mod1: plain copy */);
    _two_pass_horizontal_sum_pair_<true /* broadcast_result */>();

    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, TWO_PASS_LANE_RECIPROCAL_FP16B);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG6, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /* instr_mod1 */);

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
