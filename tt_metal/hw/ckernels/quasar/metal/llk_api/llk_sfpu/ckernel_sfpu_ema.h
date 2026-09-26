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
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Dest geometry of a 32x32 tile as the SFPU addresses it: 64 addr units of 16 datums, one per face
// row, with face f holding units 16f to 16f+15. An SFPLOAD covers the four units of [addr & ~3, +3]
// and, by address bit 1, either the even or the odd 8 datums of each. So the addr / addr+2 pair
// brings in four whole rows of one face, and the same pair one face away (+16) brings in the four
// rows of the face beside it - together, four whole tile rows, 32 columns. That row quad is the
// unit this kernel walks, addressed by the four immediates 0, 2, 16, 18.
constexpr std::uint32_t EMA_FACE_STRIDE = FACE_R_DIM;
constexpr std::uint32_t EMA_QUAD_ROWS = 4;
constexpr std::uint32_t EMA_ROW_QUADS = TILE_R_DIM / EMA_QUAD_ROWS;

// Quads walk a face pair - faces 0/1, then faces 2/3 - so after the last quad of a pair Dest sits at
// the end of the left face, one whole face short of where the next pair begins.
constexpr std::uint32_t EMA_QUADS_PER_FACE_PAIR = FACE_R_DIM / EMA_QUAD_ROWS;
constexpr std::uint32_t EMA_FACE_PAIRS = EMA_ROW_QUADS / EMA_QUADS_PER_FACE_PAIR;
constexpr std::uint32_t EMA_FACE_PAIR_JUMP = EMA_FACE_STRIDE;

// Addressing units per 32x32 Dest tile; the result lands OUT_TILE_DELTA of these past the input.
constexpr std::uint32_t EMA_TILE_STRIDE = 1u << trisc::get_dest_tile_size_log2(trisc::DstTileShape::Tile32x32);

// Bank A (LREG0-3) takes the quad's four tile rows; bank B carries the recurrence state.
constexpr std::uint32_t EMA_ROW_REG_BASE = p_sfpu::LREG0;
constexpr std::uint32_t EMA_CARRY_REG = p_sfpu::LREG4;  // running EMA, out[t-1]
constexpr std::uint32_t EMA_ALPHA_REG = p_sfpu::LREG5;  // weight on the carry
constexpr std::uint32_t EMA_BETA_REG = p_sfpu::LREG6;   // weight on the input
constexpr std::uint32_t EMA_TEMP_REG = p_sfpu::LREG7;   // alpha * out[t-1], dead between quads

// SFPLOADI writes one 16-bit half of an LREG per issue, so an fp32 weight takes two of them.
constexpr std::uint32_t EMA_WEIGHT_HALF_SHIFT = 16;
constexpr std::uint32_t EMA_WEIGHT_LOW_HALF_MASK = 0xFFFFu;

// Advances Dest by one quad on the last store of every quad, so every quad addresses Dest with the
// same four quad-relative immediates - which is what lets one recording cover every quad. A quad's
// other memory ops use ADDR_MOD_7, the all-zeroes mod the SFPU framework programs.
constexpr std::uint32_t EMA_ADDR_MOD = ADDR_MOD_6;

// The quad body is byte-identical across the whole tile, so the shortest sequence that repeats is
// one quad. Recorded once into replay slot 0 by init_ema and run over the 8 quads, that leaves a
// tile costing 11 instruction issues instead of the 154 the SFPU executes.
constexpr std::uint32_t EMA_QUAD_INSTRS = 19;  // 4 SFPLOAD + SFPTRANSP + 8 SFPMAD + SFPMOV + SFPTRANSP + 4 SFPSTORE
constexpr std::uint32_t EMA_REPLAY_SLOT = 0;
constexpr std::uint32_t EMA_REPLAY_LEN = EMA_QUAD_INSTRS;
constexpr std::uint32_t EMA_REPLAY_DEPTH = 32;
static_assert(EMA_REPLAY_LEN <= EMA_REPLAY_DEPTH, "the recorded body must fit the replay buffer");

/**
 * @brief Zero the running EMA carry, so the next tile starts a fresh chain.
 */
inline void clear_ema_carry() {
    // LCONST_0 reads 0.0 in every lane, so this zeroes the carry whatever the transpose parity is.
    TTI_SFPMOV(p_sfpu::LCONST_0, EMA_CARRY_REG, 0 /* instr_mod1: plain copy */);
}

/**
 * @brief Run the EMA recurrence over one tile row quad, addressed relative to wherever the Dest
 *        counter currently stands.
 *
 * Four SFPLOADs bring in the quad's 128 datums, one register per (face, column parity). SFPTRANSP
 * then redistributes bank A so each register holds one whole tile row, which four MAD pairs chain:
 * carry -> row 0 -> row 1 -> row 2 -> row 3. The quad's last row stays in the carry for the next
 * quad to pick up.
 *
 * @tparam OUT_TILE_DELTA: Tiles from the input tile to the result, values = <0 (in place)/1 (the
 *         neighbouring tile, leaving the input intact)>
 * @note The last SFPSTORE rides EMA_ADDR_MOD to step Dest to the next quad, so this only runs
 *       correctly once @ref init_ema has programmed that mod.
 */
template <std::uint32_t OUT_TILE_DELTA>
inline void _calculate_ema_row_quad_() {
    constexpr std::uint32_t OUT = OUT_TILE_DELTA * EMA_TILE_STRIDE;

    constexpr std::uint32_t LEFT_EVEN = p_sfpu::col_offset::EVEN_COL;
    constexpr std::uint32_t LEFT_ODD = p_sfpu::col_offset::ODD_COL;
    constexpr std::uint32_t RIGHT_EVEN = EMA_FACE_STRIDE + p_sfpu::col_offset::EVEN_COL;
    constexpr std::uint32_t RIGHT_ODD = EMA_FACE_STRIDE + p_sfpu::col_offset::ODD_COL;

    // The quad's 128 datums: both column parities of the left face, then of the face beside it
    TTI_SFPLOAD(EMA_ROW_REG_BASE + 0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, LEFT_EVEN);
    TTI_SFPLOAD(EMA_ROW_REG_BASE + 1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, LEFT_ODD);
    TTI_SFPLOAD(EMA_ROW_REG_BASE + 2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, RIGHT_EVEN);
    TTI_SFPLOAD(EMA_ROW_REG_BASE + 3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, RIGHT_ODD);

    // Both banks transpose; bank A now holds one whole tile row per register, all 32 columns
    TTI_SFPTRANSP;

    // Serial recurrence down the quad's four rows, two MADs per row:
    //   temp   = alpha * out[r-1]
    //   out[r] = beta * x[r] + temp
    // Every MAD feeds the next - no SFPNOP, the hardware interlocks a dependent consumer of a
    // 2-cycle MAD.
    TTI_SFPMAD(EMA_ALPHA_REG, EMA_CARRY_REG, p_sfpu::LCONST_0, EMA_TEMP_REG, 0 /* instr_mod1: no negation */);
    TTI_SFPMAD(EMA_BETA_REG, EMA_ROW_REG_BASE + 0, EMA_TEMP_REG, EMA_ROW_REG_BASE + 0, 0 /* instr_mod1: no negation */);
    TTI_SFPMAD(EMA_ALPHA_REG, EMA_ROW_REG_BASE + 0, p_sfpu::LCONST_0, EMA_TEMP_REG, 0 /* instr_mod1: no negation */);
    TTI_SFPMAD(EMA_BETA_REG, EMA_ROW_REG_BASE + 1, EMA_TEMP_REG, EMA_ROW_REG_BASE + 1, 0 /* instr_mod1: no negation */);
    TTI_SFPMAD(EMA_ALPHA_REG, EMA_ROW_REG_BASE + 1, p_sfpu::LCONST_0, EMA_TEMP_REG, 0 /* instr_mod1: no negation */);
    TTI_SFPMAD(EMA_BETA_REG, EMA_ROW_REG_BASE + 2, EMA_TEMP_REG, EMA_ROW_REG_BASE + 2, 0 /* instr_mod1: no negation */);
    TTI_SFPMAD(EMA_ALPHA_REG, EMA_ROW_REG_BASE + 2, p_sfpu::LCONST_0, EMA_TEMP_REG, 0 /* instr_mod1: no negation */);
    TTI_SFPMAD(EMA_BETA_REG, EMA_ROW_REG_BASE + 3, EMA_TEMP_REG, EMA_ROW_REG_BASE + 3, 0 /* instr_mod1: no negation */);

    // The quad's last row becomes the next quad's carry
    TTI_SFPMOV(EMA_ROW_REG_BASE + 3, EMA_CARRY_REG, 0 /* instr_mod1: plain copy */);

    // Involution: bank A returns to store order, bank B to its pre-transpose lane layout
    TTI_SFPTRANSP;

    // Write the quad back OUT_TILE_DELTA tiles over, then step Dest to the next quad
    TTI_SFPSTORE(EMA_ROW_REG_BASE + 0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, OUT + LEFT_EVEN);
    TTI_SFPSTORE(EMA_ROW_REG_BASE + 1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, OUT + LEFT_ODD);
    TTI_SFPSTORE(EMA_ROW_REG_BASE + 2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, OUT + RIGHT_EVEN);
    TTI_SFPSTORE(EMA_ROW_REG_BASE + 3, p_sfpu::sfpmem::DEFAULT, EMA_ADDR_MOD, 0 /* done */, OUT + RIGHT_ODD);
}

/**
 * @brief Load the two smoothing weights, start a fresh chain, and configure the state the tile walk
 *        depends on.
 *
 * Resets the RWC counters so the quad-relative Dest immediates start from 0, programs ADDR_MOD_6
 * with the per-quad Dest advance the replayed body rides on, and records that body into the replay
 * buffer. Every instruction the recording captures is an immediate, so recording it here rather than
 * per call leaves each tile costing only its replays.
 *
 * @tparam OUT_TILE_DELTA: Tiles from the input tile to the result, values = <0 (in place)/1 (the
 *         neighbouring tile, leaving the input intact)>
 * @param alpha_bits: Weight on the carry, as a raw fp32 bit pattern.
 * @param beta_bits: Weight on the input, as a raw fp32 bit pattern.
 * @note Pass the same OUT_TILE_DELTA to @ref calculate_ema - it is baked into the store immediates
 *       of the body recorded here.
 * @note Call this before @ref calculate_ema, and again before resuming EMA after any other SFPU op
 *       has run on this thread - that op's own init overwrites both the recording and the weights.
 */
template <std::uint32_t OUT_TILE_DELTA = 1>
inline void init_ema(const std::uint32_t alpha_bits, const std::uint32_t beta_bits) {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = EMA_QUAD_ROWS},
    }
        .set(EMA_ADDR_MOD);

    // Runtime weights, hence the TT_ (buffer) form rather than TTI_
    TT_SFPLOADI(EMA_ALPHA_REG, sfpi::SFPLOADI_MOD0_UPPER, alpha_bits >> EMA_WEIGHT_HALF_SHIFT);
    TT_SFPLOADI(EMA_ALPHA_REG, sfpi::SFPLOADI_MOD0_LOWER, alpha_bits & EMA_WEIGHT_LOW_HALF_MASK);
    TT_SFPLOADI(EMA_BETA_REG, sfpi::SFPLOADI_MOD0_UPPER, beta_bits >> EMA_WEIGHT_HALF_SHIFT);
    TT_SFPLOADI(EMA_BETA_REG, sfpi::SFPLOADI_MOD0_LOWER, beta_bits & EMA_WEIGHT_LOW_HALF_MASK);
    clear_ema_carry();

    // Record only; the tile the walk would touch is not this call's to write.
    load_replay_buf<EMA_REPLAY_SLOT, EMA_REPLAY_LEN, false /* exec_while_loading */>(
        [] { _calculate_ema_row_quad_<OUT_TILE_DELTA>(); });
}

/**
 * @brief Column-wise (top-to-bottom) exponential moving average of one whole 32x32 Dest tile:
 *        out[t] = alpha * out[t-1] + beta * in[t], 32 columns in parallel.
 *
 * Walks the tile as EMA_ROW_QUADS quads issued as replays of the body @ref init_ema recorded, one
 * face pair at a time.
 *
 * @tparam OUT_TILE_DELTA: Tiles from the input tile to the result, values = <0 (in place)/1 (the
 *         neighbouring tile, leaving the input intact)>
 * @param first: Whether this tile starts a fresh chain; zeroes the carry.
 * @note Run this once per tile under VectorMode::RC_custom, not once per face - the chain spans the
 *       whole tile. It leaves the Dest RWC counter advanced part way into the tile, which
 *       @ref _llk_math_eltwise_sfpu_done_ resets.
 * @note The carry and the two weights live in LREG4-6 across calls. Feed tiles top to bottom with
 *       first = false to continue the chain, and run no other SFPU op in between - any other op
 *       overwrites those registers.
 * @note With OUT_TILE_DELTA = 1 the store lands one tile past dst_index, which the harness bounds
 *       check does not cover: keep dst_index below max_dest_tiles - OUT_TILE_DELTA.
 * @note Uses replay slot 0 on the math thread.
 * @note Call @ref init_ema with the same OUT_TILE_DELTA before this - it programs the address mode
 *       and records the body this replays.
 */
template <std::uint32_t OUT_TILE_DELTA = 1>
inline void calculate_ema(const bool first = false) {
    if (first) {
        clear_ema_carry();
    }

    // Transpose parity: SFPTRANSP permutes bank B as well, so the carry and the weights are only in
    // their intended lane layout after an even number of transposes. Each quad does two, and these
    // two guards keep the whole call even - which is what carries the state into the next tile.
    TTI_SFPTRANSP;

    for (std::uint32_t face_pair = 0; face_pair < EMA_FACE_PAIRS; face_pair++) {
        if (face_pair != 0) {
            // The quad advance only walks the left face of a pair, so the right face still separates
            // where the previous pair ended from where this one begins.
            math::_incr_counters_<0 /* srca */, 0 /* srcb */, EMA_FACE_PAIR_JUMP, 0 /* cr */>();
        }

        for (std::uint32_t quad = 0; quad < EMA_QUADS_PER_FACE_PAIR; quad++) {
            TTI_REPLAY(
                EMA_REPLAY_SLOT,
                EMA_REPLAY_LEN,
                0 /* last */,
                0 /* set_mutex */,
                0 /* execute_while_loading */,
                0 /* load_mode */);
        }
    }

    TTI_SFPTRANSP;
}

}  // namespace sfpu
}  // namespace ckernel
