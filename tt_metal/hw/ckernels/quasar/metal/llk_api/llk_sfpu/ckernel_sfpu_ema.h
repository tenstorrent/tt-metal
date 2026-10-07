// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "ckernel_sfpu_replay_bank1.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Dest addr units are face rows; immediates {0, 2, 16, 18} load one quad of four whole tile rows.
constexpr std::uint32_t EMA_QUAD_ROWS = 4;
constexpr std::uint32_t EMA_FACE_STRIDE = FACE_R_DIM;
constexpr std::uint32_t EMA_DEST_TILE_ROWS = FACE_R_DIM * TILE_NUM_FACES;
constexpr std::uint32_t EMA_QUADS_PER_FACE_PAIR = FACE_R_DIM / EMA_QUAD_ROWS;
constexpr std::uint32_t EMA_FACE_PAIRS = TILE_NUM_FACES / 2;
constexpr std::uint32_t EMA_FACE_PAIR_JUMP = EMA_FACE_STRIDE;

// Production contract: output at dst_index + 1.
constexpr std::uint32_t EMA_OUTPUT_TILE_DELTA = 1;

// Rides the quad's last store to step Dest one quad.
constexpr std::uint32_t EMA_ADDR_MOD = ADDR_MOD_6;

// Resident across calls; LREG7 is unused.
constexpr std::uint32_t EMA_CARRY_REG = p_sfpu::LREG4;
constexpr std::uint32_t EMA_ALPHA_REG = p_sfpu::LREG5;
constexpr std::uint32_t EMA_BETA_REG = p_sfpu::LREG6;

constexpr std::uint32_t FP32_LO16_MASK = 0xFFFF;
constexpr std::uint32_t FP32_HI16_SHIFT = 16;

constexpr std::uint32_t FP16B_ZERO = 0x0000;

constexpr std::uint32_t EMA_NO_DONE = 0;
constexpr std::uint32_t EMA_PLAIN_MOD1 = 0;

// Replay length of _calculate_ema_row_quad_; update with the body.
constexpr std::uint32_t EMA_QUAD_LOADS = 4;
constexpr std::uint32_t EMA_QUAD_TRANSPOSES = 2;
constexpr std::uint32_t EMA_QUAD_MADS = 2 * EMA_QUAD_ROWS;
constexpr std::uint32_t EMA_QUAD_MOVS = 1;
constexpr std::uint32_t EMA_QUAD_STORES = EMA_QUAD_LOADS;
constexpr std::uint32_t EMA_REPLAY_LEN =
    EMA_QUAD_LOADS + EMA_QUAD_TRANSPOSES + EMA_QUAD_MADS + EMA_QUAD_MOVS + EMA_QUAD_STORES;

// Replay bank 1, so bank-0 FPU ops can run between EMA tiles without a re-init.
constexpr std::uint32_t EMA_REPLAY_SLOT = 0;
static_assert(EMA_REPLAY_SLOT + EMA_REPLAY_LEN <= SFPU_REPLAY_BANK_DEPTH, "the recorded body must fit one replay bank");

/**
 * @brief Load a runtime fp32 bit pattern into an LREG (SFPLOADI writes 16 bits per issue).
 */
inline void _ema_load_fp32_(const std::uint32_t lreg, const std::uint32_t bits) {
    TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_LOWER, bits & FP32_LO16_MASK);
    TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_UPPER, bits >> FP32_HI16_SHIFT);
}

/**
 * @brief EMA recurrence down one quad of four tile rows; recorded for replay by @ref init_ema.
 *
 * @tparam OUT_TILE_DELTA: Tiles from input to output; 0 writes in place.
 * @note SFPTRANSP also swizzles LREG4-7; the pair leaves the carry and weights in place.
 */
template <std::uint32_t OUT_TILE_DELTA>
inline void _calculate_ema_row_quad_() {
    constexpr std::uint32_t OUT = OUT_TILE_DELTA * EMA_DEST_TILE_ROWS;

    constexpr std::uint32_t LEFT_EVEN = p_sfpu::col_offset::EVEN_COL;
    constexpr std::uint32_t LEFT_ODD = p_sfpu::col_offset::ODD_COL;
    constexpr std::uint32_t RIGHT_EVEN = EMA_FACE_STRIDE + p_sfpu::col_offset::EVEN_COL;
    constexpr std::uint32_t RIGHT_ODD = EMA_FACE_STRIDE + p_sfpu::col_offset::ODD_COL;

    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, EMA_NO_DONE, LEFT_EVEN);
    TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, EMA_NO_DONE, LEFT_ODD);
    TTI_SFPLOAD(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, EMA_NO_DONE, RIGHT_EVEN);
    TTI_SFPLOAD(p_sfpu::LREG3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, EMA_NO_DONE, RIGHT_ODD);

    // LREG_j = tile row j
    TTI_SFPTRANSP;

    // Independent beta * x first, so only one MAD per row sits on the serial carry chain
    TTI_SFPMAD(EMA_BETA_REG, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, EMA_PLAIN_MOD1);
    TTI_SFPMAD(EMA_BETA_REG, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG1, EMA_PLAIN_MOD1);
    TTI_SFPMAD(EMA_BETA_REG, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG2, EMA_PLAIN_MOD1);
    TTI_SFPMAD(EMA_BETA_REG, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG3, EMA_PLAIN_MOD1);

    TTI_SFPMAD(EMA_ALPHA_REG, EMA_CARRY_REG, p_sfpu::LREG0, p_sfpu::LREG0, EMA_PLAIN_MOD1);
    TTI_SFPMAD(EMA_ALPHA_REG, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG1, EMA_PLAIN_MOD1);
    TTI_SFPMAD(EMA_ALPHA_REG, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG2, EMA_PLAIN_MOD1);
    TTI_SFPMAD(EMA_ALPHA_REG, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG3, EMA_PLAIN_MOD1);

    TTI_SFPMOV(p_sfpu::LREG3, EMA_CARRY_REG, EMA_PLAIN_MOD1);

    TTI_SFPTRANSP;

    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, EMA_NO_DONE, OUT + LEFT_EVEN);
    TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, EMA_NO_DONE, OUT + LEFT_ODD);
    TTI_SFPSTORE(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, EMA_NO_DONE, OUT + RIGHT_EVEN);
    TTI_SFPSTORE(p_sfpu::LREG3, p_sfpu::sfpmem::DEFAULT, EMA_ADDR_MOD, EMA_NO_DONE, OUT + RIGHT_ODD);
}

/**
 * @brief Program the address mode and record the quad body into replay bank 1.
 *
 * @tparam OUT_TILE_DELTA: Tiles from input to output, baked into the recording; 0 writes in place.
 * @note Writes no LREG: re-running it alone resumes a chain after an op that only touched ADDR_MOD_6
 *       or replay bank 1. An op that writes LREG4-6 ends the chain (init, load weights, clear).
 */
template <std::uint32_t OUT_TILE_DELTA = EMA_OUTPUT_TILE_DELTA>
inline void init_ema() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = EMA_QUAD_ROWS},
    }
        .set(EMA_ADDR_MOD);

    _sfpu_record_replay_bank1_<EMA_REPLAY_SLOT, EMA_REPLAY_LEN>([] { _calculate_ema_row_quad_<OUT_TILE_DELTA>(); });
}

/**
 * @brief Load the fp32 weights: alpha on the carry, beta on the input. They stay resident.
 */
inline void ema_load_alpha_beta(const std::uint32_t alpha, const std::uint32_t beta) {
    _ema_load_fp32_(EMA_ALPHA_REG, alpha);
    _ema_load_fp32_(EMA_BETA_REG, beta);
}

/**
 * @brief Zero the carry, starting a new chain.
 */
inline void ema_clear_previous_output() { TTI_SFPLOADI(EMA_CARRY_REG, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ZERO); }

/**
 * @brief Column-wise EMA (out = alpha * out_prev + beta * x) of one 32x32 Dest tile.
 *
 * @note 32x32 tiles only, once per tile under VectorMode::RC_custom.
 * @note The carry stays in LREG4, so consecutive calls continue one chain top to bottom.
 * @note Overwrites bank 0's last replay slot (see SFPU_REPLAY_BANK_SWITCH_SLOT_FREE).
 */
inline void calculate_ema() {
    _sfpu_enter_replay_bank1_();

    // Paired with the exit transpose: keeps LREG4-7 math-ready between tiles. Do not remove.
    TTI_SFPTRANSP;

    for (std::uint32_t face_pair = 0; face_pair < EMA_FACE_PAIRS; face_pair++) {
        if (face_pair != 0) {
            // Skip the right face to reach the next face pair
            math::_incr_counters_<0 /*srca*/, 0 /*srcb*/, EMA_FACE_PAIR_JUMP, 0 /*cr*/>();
        }

        for (std::uint32_t quad = 0; quad < EMA_QUADS_PER_FACE_PAIR; quad++) {
            // `last` on the final replay returns the read ID to bank 0
            const bool last_quad = (face_pair == EMA_FACE_PAIRS - 1) && (quad == EMA_QUADS_PER_FACE_PAIR - 1);
            if (last_quad) {
                TTI_REPLAY(
                    EMA_REPLAY_SLOT, EMA_REPLAY_LEN, 1 /*last*/, 0 /*set_mutex*/, 0 /*exec_while_loading*/, 0 /*load*/);
            } else {
                TTI_REPLAY(
                    EMA_REPLAY_SLOT, EMA_REPLAY_LEN, 0 /*last*/, 0 /*set_mutex*/, 0 /*exec_while_loading*/, 0 /*load*/);
            }
        }
    }

    TTI_SFPTRANSP;
}

}  // namespace sfpu
}  // namespace ckernel
