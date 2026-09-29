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

// Dest geometry of a 32x32 tile as the SFPU addresses it: 64 addr units of 16 datums, one per face
// row, with face f holding units 16f to 16f+15. An SFPLOAD covers the four units of [addr & ~3, +3]
// and, by address bit 1, either the even or the odd 8 datums of each. So the addr / addr+2 pair
// brings in four whole rows of one face, and the same pair one face away (+16) brings in the four
// rows of the face beside it - together, four whole tile rows. That row quad is the unit this kernel
// walks in, addressed by the four immediates 0, 2, 16, 18.
constexpr std::uint32_t CUMSUM_FACE_STRIDE = FACE_R_DIM;
constexpr std::uint32_t CUMSUM_QUAD_ROWS = 4;
constexpr std::uint32_t CUMSUM_ROW_QUADS = TILE_R_DIM / CUMSUM_QUAD_ROWS;

// Quads walk a face pair - faces 0/1, then faces 2/3 - so after the last quad of a pair Dest sits at
// the end of the left face, one whole face short of where the next pair begins.
constexpr std::uint32_t CUMSUM_QUADS_PER_FACE_PAIR = FACE_R_DIM / CUMSUM_QUAD_ROWS;
constexpr std::uint32_t CUMSUM_FACE_PAIRS = TILE_NUM_FACES / 2;
constexpr std::uint32_t CUMSUM_FACE_PAIR_JUMP = CUMSUM_FACE_STRIDE;

// The two LREG banks the transpose network swizzles: LREG0-3 and LREG4-7. Quads alternate between
// them so each quad's running totals survive in the bank the next quad does not load into.
constexpr std::uint32_t CUMSUM_LREG_BANK_A = p_sfpu::LREG0;
constexpr std::uint32_t CUMSUM_LREG_BANK_B = p_sfpu::LREG4;

// Post-transpose, bank register j holds tile row j of the quad, all 32 columns. The last one is
// therefore the quad's running totals, which the next quad picks up as its carry.
constexpr std::uint32_t CUMSUM_CARRY_REG = CUMSUM_QUAD_ROWS - 1;

// Advances Dest by one quad on the last store of every quad, so every quad addresses Dest with the
// same four quad-relative immediates - which is what lets one recording cover every quad. A quad's
// other memory ops use ADDR_MOD_7, the all-zeroes mod the SFPU framework programs.
constexpr std::uint32_t CUMSUM_ADDR_MOD = ADDR_MOD_6;

// Quads alternate banks, so the shortest sequence that repeats byte-for-byte is two quads, one per
// bank. Recorded once into replay slot 0 by cumsum_init and run over the 4 bank pairs, that leaves a
// tile costing 5 instruction issues instead of the 112 the SFPU executes.
constexpr std::uint32_t CUMSUM_QUAD_INSTRS = 14;  // 4 SFPLOAD + SFPTRANSP + 4 SFPADD + SFPTRANSP + 4 SFPSTORE
constexpr std::uint32_t CUMSUM_QUADS_PER_RECORDING = 2;
constexpr std::uint32_t CUMSUM_REPLAY_SLOT = 0;
constexpr std::uint32_t CUMSUM_REPLAY_LEN = CUMSUM_QUADS_PER_RECORDING * CUMSUM_QUAD_INSTRS;
constexpr std::uint32_t CUMSUM_REPLAYS_PER_FACE_PAIR = CUMSUM_QUADS_PER_FACE_PAIR / CUMSUM_QUADS_PER_RECORDING;
constexpr std::uint32_t CUMSUM_REPLAY_DEPTH = 32;
static_assert(CUMSUM_ROW_QUADS % CUMSUM_QUADS_PER_RECORDING == 0, "the recorded body must tile the Dest tile exactly");
static_assert(CUMSUM_REPLAY_LEN <= CUMSUM_REPLAY_DEPTH, "the recorded body must fit the replay buffer");

/**
 * @brief Accumulate one tile row quad on top of the previous quad's running totals.
 *
 * Four SFPLOADs bring in the quad's 128 datums, one register per (face, column parity). SFPTRANSP
 * then redistributes this bank so each register holds one whole tile row, which four SFPADDs chain:
 * carry -> row 0 -> row 1 -> row 2 -> row 3. The quad's own totals stay in the bank's last register
 * for the next quad to pick up as its carry.
 *
 * @tparam LREG_BASE: First LREG of this quad's bank, values = <CUMSUM_LREG_BANK_A/CUMSUM_LREG_BANK_B>
 * @note Alternate LREG_BASE between quads. The carry only survives because it sits in the bank this
 *       quad's SFPLOADs do not overwrite, so SFPTRANSP's involution restores it untouched.
 * @note The last SFPSTORE rides CUMSUM_ADDR_MOD to step Dest to the next quad, so this only runs
 *       correctly once @ref cumsum_init has programmed that mod.
 */
template <std::uint32_t LREG_BASE>
inline void _calculate_cumsum_row_quad_() {
    // The previous quad ran on the other bank and left its totals in that bank's last register.
    constexpr std::uint32_t OTHER_BANK = (LREG_BASE == CUMSUM_LREG_BANK_A) ? CUMSUM_LREG_BANK_B : CUMSUM_LREG_BANK_A;
    constexpr std::uint32_t CARRY = OTHER_BANK + CUMSUM_CARRY_REG;

    constexpr std::uint32_t LEFT_EVEN = p_sfpu::col_offset::EVEN_COL;
    constexpr std::uint32_t LEFT_ODD = p_sfpu::col_offset::ODD_COL;
    constexpr std::uint32_t RIGHT_EVEN = CUMSUM_FACE_STRIDE + p_sfpu::col_offset::EVEN_COL;
    constexpr std::uint32_t RIGHT_ODD = CUMSUM_FACE_STRIDE + p_sfpu::col_offset::ODD_COL;

    // The quad's 128 datums: both column parities of the left face, then of the face beside it
    TTI_SFPLOAD(LREG_BASE + 0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, LEFT_EVEN);
    TTI_SFPLOAD(LREG_BASE + 1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, LEFT_ODD);
    TTI_SFPLOAD(LREG_BASE + 2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, RIGHT_EVEN);
    TTI_SFPLOAD(LREG_BASE + 3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, RIGHT_ODD);

    // Both banks transpose; this bank's registers now hold one whole tile row each
    TTI_SFPTRANSP;

    // Serial column-wise prefix sum down the quad's four rows. SFPADD is dest = a*b + c, so a = 1.0
    // makes it dest = prev + cur. No SFPNOP between them - the hardware interlocks a dependent
    // consumer of a 2-cycle MAD.
    TTI_SFPADD(p_sfpu::LCONST_1, CARRY, LREG_BASE + 0, LREG_BASE + 0, 0 /* instr_mod1: no negation */);
    TTI_SFPADD(p_sfpu::LCONST_1, LREG_BASE + 0, LREG_BASE + 1, LREG_BASE + 1, 0 /* instr_mod1: no negation */);
    TTI_SFPADD(p_sfpu::LCONST_1, LREG_BASE + 1, LREG_BASE + 2, LREG_BASE + 2, 0 /* instr_mod1: no negation */);
    TTI_SFPADD(p_sfpu::LCONST_1, LREG_BASE + 2, LREG_BASE + 3, LREG_BASE + 3, 0 /* instr_mod1: no negation */);

    // Involution: this bank returns to store order, the other bank to its pre-transpose contents -
    // that is what keeps the previous quad's carry intact.
    TTI_SFPTRANSP;

    // Write the quad back to the same four slots, then step Dest to the next quad
    TTI_SFPSTORE(LREG_BASE + 0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, LEFT_EVEN);
    TTI_SFPSTORE(LREG_BASE + 1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, LEFT_ODD);
    TTI_SFPSTORE(LREG_BASE + 2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, RIGHT_EVEN);
    TTI_SFPSTORE(LREG_BASE + 3, p_sfpu::sfpmem::DEFAULT, CUMSUM_ADDR_MOD, 0 /* done */, RIGHT_ODD);
}

/**
 * @brief Configure the SFPU state the cumsum tile walk depends on.
 *
 * Resets the RWC counters so the quad-relative Dest immediates start from 0, programs ADDR_MOD_6
 * with the per-quad Dest advance the replayed body rides on, and records that body into the replay
 * buffer. Every instruction the recording captures is an immediate, so it needs no runtime state and
 * recording it here rather than per call leaves each tile costing only its replays.
 *
 * @note Call this before @ref calculate_cumsum, and again before resuming cumsum after any other
 *       SFPU op has run on this thread - the recording is what the other op's own init overwrites.
 */
template <bool APPROXIMATION_MODE /*unused*/>
inline void cumsum_init() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = CUMSUM_QUAD_ROWS},
    }
        .set(CUMSUM_ADDR_MOD);

    // Record only; the tile the walk would touch is not this call's to write.
    load_replay_buf<CUMSUM_REPLAY_SLOT, CUMSUM_REPLAY_LEN, false /* exec_while_loading */>([] {
        _calculate_cumsum_row_quad_<CUMSUM_LREG_BANK_A>();
        _calculate_cumsum_row_quad_<CUMSUM_LREG_BANK_B>();
    });
}

/**
 * @brief Column-wise (top-to-bottom) cumulative sum of one whole 32x32 Dest tile, in place.
 *
 * Walks the tile as CUMSUM_ROW_QUADS quads on alternating LREG banks, issued as replays of the
 * two-quad body @ref cumsum_init recorded, one face pair at a time.
 *
 * @param first: Whether this is the first tile of a top-to-bottom chain; zeroes the carry bank.
 * @note Run this once per tile under VectorMode::RC_custom, not once per face - the chain spans the
 *       whole tile. It leaves the Dest RWC counter advanced part way into the tile, which
 *       @ref _llk_math_eltwise_sfpu_done_ resets.
 * @note On return CUMSUM_LREG_BANK_B (LREG4-7) collectively holds this tile's 32 column totals in
 *       store order. The next call's first transpose reconstructs the carry in LREG7. Feed tiles
 *       top-to-bottom and write nothing to LREG4-7 in between.
 * @note Uses replay slot 0 on the math thread.
 * @note Call @ref cumsum_init before this - it programs the address mode and records the body this
 *       replays.
 */
template <bool APPROXIMATION_MODE /*unused*/, int ITERATIONS = 8 /*unused*/>
inline void calculate_cumsum(const bool first) {
    if (first) {
        // Zero the whole carry bank, not just its last register: SFPTRANSP swizzles lanes across the
        // entire bank, so a non-zero register anywhere in it would surface in the carry's lanes.
        TTI_SFPMOV(p_sfpu::LCONST_0, CUMSUM_LREG_BANK_B + 0, 0 /* instr_mod1: plain copy */);
        TTI_SFPMOV(p_sfpu::LCONST_0, CUMSUM_LREG_BANK_B + 1, 0 /* instr_mod1: plain copy */);
        TTI_SFPMOV(p_sfpu::LCONST_0, CUMSUM_LREG_BANK_B + 2, 0 /* instr_mod1: plain copy */);
        TTI_SFPMOV(p_sfpu::LCONST_0, CUMSUM_LREG_BANK_B + 3, 0 /* instr_mod1: plain copy */);
    }

    for (std::uint32_t face_pair = 0; face_pair < CUMSUM_FACE_PAIRS; face_pair++) {
        if (face_pair != 0) {
            // The quad advance only walks the left face of a pair, so the right face still separates
            // where the previous pair ended from where this one begins.
            math::_incr_counters_<0 /* srca */, 0 /* srcb */, CUMSUM_FACE_PAIR_JUMP, 0 /* cr */>();
        }

        // Each replay runs two quads, one per LREG bank; Dest auto-increment walks them down the face.
        for (std::uint32_t replay = 0; replay < CUMSUM_REPLAYS_PER_FACE_PAIR; replay++) {
            TTI_REPLAY(
                CUMSUM_REPLAY_SLOT,
                CUMSUM_REPLAY_LEN,
                0 /* last */,
                0 /* set_mutex */,
                0 /* execute_while_loading */,
                0 /* load_mode */);
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
