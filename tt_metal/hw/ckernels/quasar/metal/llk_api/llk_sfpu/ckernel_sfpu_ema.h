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

// Dest geometry of a 32x32 tile as the SFPU addresses it: 64 addr units, one per face row, face f at
// units 16f..16f+15. The immediates {0, 2, 16, 18} bring in one quad of four whole tile rows.
constexpr std::uint32_t EMA_QUAD_ROWS = 4;
constexpr std::uint32_t EMA_FACE_STRIDE = FACE_R_DIM;
constexpr std::uint32_t EMA_DEST_TILE_ROWS = FACE_R_DIM * TILE_NUM_FACES;
constexpr std::uint32_t EMA_QUADS_PER_FACE_PAIR = FACE_R_DIM / EMA_QUAD_ROWS;
constexpr std::uint32_t EMA_FACE_PAIRS = TILE_NUM_FACES / 2;
constexpr std::uint32_t EMA_FACE_PAIR_JUMP = EMA_FACE_STRIDE;

// Production contract: input in Dest tile dst_index, output in dst_index + 1.
constexpr std::uint32_t EMA_OUTPUT_TILE_DELTA = 1;

// Last store of every quad rides this mod to step Dest one quad; the other memory ops use ADDR_MOD_7.
constexpr std::uint32_t EMA_ADDR_MOD = ADDR_MOD_6;

// Register roles, resident across calls: carry = EMA_old per column, alpha/beta lane-uniform weights.
constexpr std::uint32_t EMA_CARRY_REG = p_sfpu::LREG4;
constexpr std::uint32_t EMA_ALPHA_REG = p_sfpu::LREG5;
constexpr std::uint32_t EMA_BETA_REG = p_sfpu::LREG6;
constexpr std::uint32_t EMA_SCRATCH_REG = p_sfpu::LREG7;

// Splits a runtime fp32 bit pattern into the two SFPLOADI halves.
constexpr std::uint32_t FP32_LO16_MASK = 0xFFFF;
constexpr std::uint32_t FP32_HI16_SHIFT = 16;

constexpr std::uint32_t FP16B_ZERO = 0x0000;

// Immediates every memory op / MAD / MOV of the quad body shares.
constexpr std::uint32_t EMA_NO_DONE = 0;     // SFPLOAD / SFPSTORE: no Dest done signal
constexpr std::uint32_t EMA_PLAIN_MOD1 = 0;  // SFPMAD: no negation; SFPMOV: plain copy

// One row-quad body, composed from its instructions so the replay length follows the body.
constexpr std::uint32_t EMA_QUAD_LOADS = 4;                 // one per (face, column parity)
constexpr std::uint32_t EMA_QUAD_TRANSPOSES = 2;            // into rows and back
constexpr std::uint32_t EMA_QUAD_MADS = 2 * EMA_QUAD_ROWS;  // alpha * old, then + beta * x, per row
constexpr std::uint32_t EMA_QUAD_MOVS = 1;                  // last row -> carry
constexpr std::uint32_t EMA_QUAD_STORES = EMA_QUAD_LOADS;
constexpr std::uint32_t EMA_REPLAY_LEN =
    EMA_QUAD_LOADS + EMA_QUAD_TRANSPOSES + EMA_QUAD_MADS + EMA_QUAD_MOVS + EMA_QUAD_STORES;

// The math thread's replay buffer is double-banked: 64 entries, two banks of 32. A REPLAY's start
// index addresses within a bank, and which bank it hits comes from a write ID (loads) and a read ID
// (executes), each flipped by a REPLAY with `last` set once it completes. Every math-thread FPU op
// records into and replays from bank 0 with both IDs at 0, so EMA keeps its body in bank 1 - as
// cumsum does - and an FPU op such as transpose_dest can run between EMA tiles without a re-init.
// Both IDs are back at 0 whenever EMA is not running.
constexpr std::uint32_t EMA_REPLAY_SLOT = 0;
constexpr std::uint32_t EMA_REPLAY_BANK_DEPTH = 32;
static_assert(EMA_REPLAY_LEN <= EMA_REPLAY_BANK_DEPTH, "the recorded body must fit one replay bank");

// The read ID only flips after an executed REPLAY, so entering bank 1 costs one instruction replayed
// out of bank 0: an SFPNOP kept in bank 0's last slot. No math-thread FPU recording reaches that slot
// (the longest, 32-bit transpose_dest, stops at 23), and the SFPNOP is refreshed every tile anyway.
constexpr std::uint32_t EMA_BANK_SWITCH_SLOT = EMA_REPLAY_BANK_DEPTH - 1;

/**
 * @brief Write a runtime fp32 bit pattern into one LREG, lane-uniform.
 *
 * SFPLOADI writes one 16-bit half per issue, so an fp32 value takes two.
 *
 * @param lreg: Destination LREG.
 * @param bits: fp32 bit pattern.
 */
inline void _ema_load_fp32_(const std::uint32_t lreg, const std::uint32_t bits) {
    TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_LOWER, bits & FP32_LO16_MASK);
    TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_UPPER, bits >> FP32_HI16_SHIFT);
}

/**
 * @brief Run the EMA recurrence down one quad of four tile rows.
 *
 * Four SFPLOADs bring in the quad's 128 datums, one register per (face, column parity). SFPTRANSP
 * then redistributes the bank so LREG0-3 each hold one whole tile row, which the MAD pairs chain:
 * carry -> row 0 -> row 1 -> row 2 -> row 3. The second SFPTRANSP returns the rows to store order,
 * and the quad's last row stays in the carry for the next quad.
 *
 * @tparam OUT_TILE_DELTA: Tiles from the input tile to the output tile; 0 writes in place.
 * @note This is recorded into the replay buffer rather than executed directly, so every Dest
 *       address it uses is a quad-relative immediate and the last SFPSTORE rides EMA_ADDR_MOD to
 *       step Dest one quad. It is only correct once @ref init_ema has programmed that mod.
 * @note Both SFPTRANSP calls swizzle both LREG banks, which is what carries EMA_CARRY_REG into and
 *       out of the recurrence untouched.
 */
template <std::uint32_t OUT_TILE_DELTA>
inline void _calculate_ema_row_quad_() {
    constexpr std::uint32_t OUT = OUT_TILE_DELTA * EMA_DEST_TILE_ROWS;

    constexpr std::uint32_t LEFT_EVEN = p_sfpu::col_offset::EVEN_COL;
    constexpr std::uint32_t LEFT_ODD = p_sfpu::col_offset::ODD_COL;
    constexpr std::uint32_t RIGHT_EVEN = EMA_FACE_STRIDE + p_sfpu::col_offset::EVEN_COL;
    constexpr std::uint32_t RIGHT_ODD = EMA_FACE_STRIDE + p_sfpu::col_offset::ODD_COL;

    // Load the quad: both column parities of the left face, then of the face beside it
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, EMA_NO_DONE, LEFT_EVEN);
    TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, EMA_NO_DONE, LEFT_ODD);
    TTI_SFPLOAD(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, EMA_NO_DONE, RIGHT_EVEN);
    TTI_SFPLOAD(p_sfpu::LREG3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, EMA_NO_DONE, RIGHT_ODD);

    // LREG_j = tile row j; LREG4-7 back to math layout
    TTI_SFPTRANSP;

    // EMA chain down the quad: tmp = alpha * old; row = beta * x + tmp. MAD consumers are interlocked.
    TTI_SFPMAD(EMA_ALPHA_REG, EMA_CARRY_REG, p_sfpu::LCONST_0, EMA_SCRATCH_REG, EMA_PLAIN_MOD1);
    TTI_SFPMAD(EMA_BETA_REG, p_sfpu::LREG0, EMA_SCRATCH_REG, p_sfpu::LREG0, EMA_PLAIN_MOD1);
    TTI_SFPMAD(EMA_ALPHA_REG, p_sfpu::LREG0, p_sfpu::LCONST_0, EMA_SCRATCH_REG, EMA_PLAIN_MOD1);
    TTI_SFPMAD(EMA_BETA_REG, p_sfpu::LREG1, EMA_SCRATCH_REG, p_sfpu::LREG1, EMA_PLAIN_MOD1);
    TTI_SFPMAD(EMA_ALPHA_REG, p_sfpu::LREG1, p_sfpu::LCONST_0, EMA_SCRATCH_REG, EMA_PLAIN_MOD1);
    TTI_SFPMAD(EMA_BETA_REG, p_sfpu::LREG2, EMA_SCRATCH_REG, p_sfpu::LREG2, EMA_PLAIN_MOD1);
    TTI_SFPMAD(EMA_ALPHA_REG, p_sfpu::LREG2, p_sfpu::LCONST_0, EMA_SCRATCH_REG, EMA_PLAIN_MOD1);
    TTI_SFPMAD(EMA_BETA_REG, p_sfpu::LREG3, EMA_SCRATCH_REG, p_sfpu::LREG3, EMA_PLAIN_MOD1);

    TTI_SFPMOV(p_sfpu::LREG3, EMA_CARRY_REG, EMA_PLAIN_MOD1);  // carry = last row

    // Rows back to store order; LREG4-7 back to between-quad layout
    TTI_SFPTRANSP;

    // Store the quad to the output tile, then step Dest to the next quad
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, EMA_NO_DONE, OUT + LEFT_EVEN);
    TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, EMA_NO_DONE, OUT + LEFT_ODD);
    TTI_SFPSTORE(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, EMA_NO_DONE, OUT + RIGHT_EVEN);
    TTI_SFPSTORE(p_sfpu::LREG3, p_sfpu::sfpmem::DEFAULT, EMA_ADDR_MOD, EMA_NO_DONE, OUT + RIGHT_ODD);
}

/**
 * @brief Program the SFPU state the EMA tile walk depends on and record its body.
 *
 * Resets the RWC counters so the quad-relative Dest immediates start from 0, programs
 * EMA_ADDR_MOD with the per-quad Dest advance the replayed body rides on, and records that body
 * into replay bank 1. Every instruction the recording captures is an immediate, so recording it
 * once here leaves each tile costing only its replays.
 *
 * @tparam OUT_TILE_DELTA: Tiles from the input tile to the output tile, baked into the recorded
 *         stores; the production contract is EMA_OUTPUT_TILE_DELTA, 0 writes in place.
 * @note @ref calculate_ema replays what this recorded, so this decides where its output lands.
 * @note Call this before @ref calculate_ema, and again before resuming EMA after any op that
 *       reprograms ADDR_MOD_6 (several SFPU ops do), records into replay bank 1 (cumsum does), or
 *       writes LREG4-7. Follow it with @ref ema_load_alpha_beta to install the weights and
 *       @ref ema_clear_previous_output to start a chain.
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

    // Loading the bank switch with `last` flips the write ID to bank 1 for the body below.
    TTI_REPLAY(EMA_BANK_SWITCH_SLOT, 1 /*len*/, 1 /*last*/, 0 /*set_mutex*/, 0 /*exec_while_loading*/, 1 /*load*/);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);

    // Record only; the tile the walk would touch is not this call's to write. `last` flips the write
    // ID back to bank 0 for everyone else.
    load_replay_buf<EMA_REPLAY_SLOT, EMA_REPLAY_LEN, false /*exec_while_loading*/, 0 /*set_mutex*/, 1 /*last*/>(
        [] { _calculate_ema_row_quad_<OUT_TILE_DELTA>(); });
}

/**
 * @brief Install the EMA smoothing weights, lane-uniform, for every later tile.
 *
 * @param alpha: fp32 bit pattern of the weight on the carry (EMA_old).
 * @param beta: fp32 bit pattern of the weight on the incoming datum.
 * @note The weights stay resident in EMA_ALPHA_REG / EMA_BETA_REG across calls, so load them once
 *       after @ref init_ema and write nothing to those LREGs while EMA tiles are in flight.
 */
inline void ema_load_alpha_beta(const std::uint32_t alpha, const std::uint32_t beta) {
    _ema_load_fp32_(EMA_ALPHA_REG, alpha);
    _ema_load_fp32_(EMA_BETA_REG, beta);
}

/**
 * @brief Zero the running EMA_old carry, starting a fresh top-to-bottom chain.
 *
 * @note Clears EMA_CARRY_REG only - the neighbouring LREGs hold the weights and the scratch, and
 *       @ref calculate_ema's tile-entry transpose leaves the carry where the chain expects it.
 */
inline void ema_clear_previous_output() { TTI_SFPLOADI(EMA_CARRY_REG, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ZERO); }

/**
 * @brief Column-wise (top-to-bottom) exponential moving average of one whole 32x32 Dest tile.
 *
 * Computes EMA_new = alpha * EMA_old + beta * x for each of the tile's 32 columns independently,
 * walking the tile as row quads issued as replays of the body @ref init_ema recorded into replay
 * bank 1, one face pair at a time.
 *
 * @note Where the output lands is fixed by the OUT_TILE_DELTA @ref init_ema recorded the body with:
 *       the production contract is EMA_OUTPUT_TILE_DELTA (input at dst_index, output at
 *       dst_index + 1), 0 writes in place.
 * @note Run this once per tile under VectorMode::RC_custom, not once per face - the chain spans the
 *       whole tile. It leaves the Dest RWC counter advanced part way into the tile, which
 *       @ref _llk_math_eltwise_sfpu_done_ resets.
 * @note The carry survives in EMA_CARRY_REG on return, so consecutive calls continue one time
 *       sequence. Feed tiles top-to-bottom, call @ref ema_clear_previous_output to start a chain,
 *       and write nothing to EMA_CARRY_REG / EMA_ALPHA_REG / EMA_BETA_REG / EMA_SCRATCH_REG in
 *       between.
 * @note The transposes surrounding the tile walk are what keep that register bank math-ready
 *       between tiles; they pair up and must not be removed.
 * @note Replays from bank 1 of the math thread's replay buffer, and overwrites bank 0's last slot
 *       with the SFPNOP that switches banks. Both replay bank IDs are back at 0 on return.
 * @note Call @ref init_ema and @ref ema_load_alpha_beta before this.
 */
inline void calculate_ema() {
    // Refresh the bank switch in bank 0, then replay it with `last` so the read ID flips to bank 1.
    TTI_REPLAY(EMA_BANK_SWITCH_SLOT, 1 /*len*/, 0 /*last*/, 0 /*set_mutex*/, 0 /*exec_while_loading*/, 1 /*load*/);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);
    TTI_REPLAY(EMA_BANK_SWITCH_SLOT, 1 /*len*/, 1 /*last*/, 0 /*set_mutex*/, 0 /*exec_while_loading*/, 0 /*load*/);

    // Tile-entry bracket: keeps the LREG4-7 bank math-ready between tiles
    TTI_SFPTRANSP;

    for (std::uint32_t face_pair = 0; face_pair < EMA_FACE_PAIRS; face_pair++) {
        if (face_pair != 0) {
            // Quads walked only the left face; skip the right face to reach the next pair
            math::_incr_counters_<0 /*srca*/, 0 /*srcb*/, EMA_FACE_PAIR_JUMP, 0 /*cr*/>();
        }

        for (std::uint32_t quad = 0; quad < EMA_QUADS_PER_FACE_PAIR; quad++) {
            // `last` on the tile's final replay flips the read ID back to bank 0 for everyone else.
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

    // Tile-exit bracket
    TTI_SFPTRANSP;
}

}  // namespace sfpu
}  // namespace ckernel
