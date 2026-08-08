// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel {
namespace sfpu {

// Dest geometry of a 32x32 tile as the SFPU addresses it: 64 addr units of 16 datums, laid out
// row-interleaved - addr unit 2*row covers tile columns 0-15 of a tile row, 2*row+1 covers columns
// 16-31 of the same row. One access therefore reaches two tile rows, and that row pair is the unit
// this kernel walks the tile in.
constexpr std::uint32_t CUMSUM_ADDR_UNITS_PER_ROW = 2;
constexpr std::uint32_t CUMSUM_ROWS_PER_PAIR = 2;
constexpr std::uint32_t CUMSUM_ROW_PAIRS = TILE_R_DIM / CUMSUM_ROWS_PER_PAIR;

// Addr units one pass advances Dest by.
constexpr std::uint32_t CUMSUM_ROW_PAIR_STRIDE = CUMSUM_ADDR_UNITS_PER_ROW * CUMSUM_ROWS_PER_PAIR;

// Address offset of a pass's second access: address bit 1 selects which half of each addr unit's 16
// datums the 32 lanes cover, so the pair base and base + this offset cover the pair's 64 datums.
constexpr std::uint32_t CUMSUM_SECOND_ACCESS_OFF = 2;

// The two LREG banks the transpose network swizzles: LREGS1 = LREG0-3, LREGS2 = LREG4-7. Passes
// alternate between them so each pass's running totals survive in the bank the next pass does not
// load into.
constexpr std::uint32_t CUMSUM_LREG_BANK_A = p_sfpu::LREG0;
constexpr std::uint32_t CUMSUM_LREG_BANK_B = p_sfpu::LREG4;

// Bank-relative register roles, in post-transpose order. The ROW1 pair is also the pass's carry out.
constexpr std::uint32_t CUMSUM_ROW0_LO = 0;  // even tile row, columns 0-15
constexpr std::uint32_t CUMSUM_ROW0_HI = 1;  // even tile row, columns 16-31
constexpr std::uint32_t CUMSUM_ROW1_LO = 2;  // odd tile row, columns 0-15
constexpr std::uint32_t CUMSUM_ROW1_HI = 3;  // odd tile row, columns 16-31

// Advances Dest by one row pair on the last store of every pass, so every pass addresses Dest with
// the same two pair-relative immediates - which is what lets one recording cover every row pair. The
// pass's other three memory ops use ADDR_MOD_7, the all-zeroes mod the SFPU framework programs.
constexpr std::uint32_t CUMSUM_ADDR_MOD = ADDR_MOD_6;

// Passes alternate banks, so the shortest sequence that repeats byte-for-byte is two passes, one per
// bank. Recorded once into replay slot 0 and run over the 8 bank pairs, that turns 160 instruction
// issues into 28.
constexpr std::uint32_t CUMSUM_PASS_INSTRS = 10;  // 2 SFPLOAD + SFPTRANSP + 4 SFPADD + SFPTRANSP + 2 SFPSTORE
constexpr std::uint32_t CUMSUM_PASSES_PER_RECORDING = 2;
constexpr std::uint32_t CUMSUM_REPLAY_SLOT = 0;
constexpr std::uint32_t CUMSUM_REPLAY_LEN = CUMSUM_PASSES_PER_RECORDING * CUMSUM_PASS_INSTRS;
constexpr std::uint32_t CUMSUM_REPLAY_ITERS = CUMSUM_ROW_PAIRS / CUMSUM_PASSES_PER_RECORDING;
static_assert(CUMSUM_ROW_PAIRS % CUMSUM_PASSES_PER_RECORDING == 0, "the recorded body must tile the Dest tile exactly");

/**
 * @brief Configure the SFPU state the cumsum tile walk depends on.
 *
 * Resets the RWC counters so the pair-relative Dest immediates start from 0, and programs
 * ADDR_MOD_6 with the per-pass Dest advance the replayed body rides on.
 *
 * @note Call this before @ref calculate_cumsum.
 */
inline void cumsum_init() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = CUMSUM_ROW_PAIR_STRIDE},
    }
        .set(CUMSUM_ADDR_MOD);
}

/**
 * @brief Accumulate one tile row pair on top of the previous pair's running totals.
 *
 * Two SFPLOADs bring in the pair's 64 datums. SFPTRANSP then redistributes this bank's four
 * registers so each holds one (tile row, tile column half) of the pair, so the pair carries two
 * running totals - one per column half - which four SFPADDs chain: carry -> even row -> odd row.
 * The pass's own totals stay in the ROW1 pair for the next pass to pick up as its carry.
 *
 * @tparam LREG_BASE: First LREG of this pass's bank, values = <CUMSUM_LREG_BANK_A/CUMSUM_LREG_BANK_B>
 * @note Alternate LREG_BASE between passes. The carry only survives because it sits in the bank
 *       this pass's SFPLOADs do not overwrite, so SFPTRANSP's involution restores it untouched.
 * @note Call @ref cumsum_init first - the last SFPSTORE uses the address mode it programs to step
 *       Dest to the next row pair.
 */
template <std::uint32_t LREG_BASE>
inline void _calculate_cumsum_row_pair_() {
    constexpr std::uint32_t ROW0_LO = LREG_BASE + CUMSUM_ROW0_LO;
    constexpr std::uint32_t ROW0_HI = LREG_BASE + CUMSUM_ROW0_HI;
    constexpr std::uint32_t ROW1_LO = LREG_BASE + CUMSUM_ROW1_LO;
    constexpr std::uint32_t ROW1_HI = LREG_BASE + CUMSUM_ROW1_HI;

    // The previous pass ran on the other bank and left its totals in that bank's ROW1 pair.
    constexpr std::uint32_t OTHER_BANK = (LREG_BASE == CUMSUM_LREG_BANK_A) ? CUMSUM_LREG_BANK_B : CUMSUM_LREG_BANK_A;
    constexpr std::uint32_t CARRY_LO = OTHER_BANK + CUMSUM_ROW1_LO;
    constexpr std::uint32_t CARRY_HI = OTHER_BANK + CUMSUM_ROW1_HI;

    // The row pair's 64 datums into the first two LREGs of this bank
    TTI_SFPLOAD(ROW0_LO, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg_addr */);
    TTI_SFPLOAD(ROW0_HI, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, CUMSUM_SECOND_ACCESS_OFF);

    // Both banks transpose; this bank's registers now split by (tile row, tile column half)
    TTI_SFPTRANSP;

    // Serial column-wise prefix sum, one chain per tile column half. SFPADD is dest = a*b + c, so
    // a = 1.0 makes it dest = prev + cur. No SFPNOP between them - the hardware interlocks a
    // dependent consumer of a 2-cycle MAD.
    TTI_SFPADD(p_sfpu::LCONST_1, CARRY_LO, ROW0_LO, ROW0_LO, 0 /* instr_mod1 */);  // even row = carry + row
    TTI_SFPADD(p_sfpu::LCONST_1, CARRY_HI, ROW0_HI, ROW0_HI, 0 /* instr_mod1 */);
    TTI_SFPADD(p_sfpu::LCONST_1, ROW0_LO, ROW1_LO, ROW1_LO, 0 /* instr_mod1 */);  // odd row = even row + row
    TTI_SFPADD(p_sfpu::LCONST_1, ROW0_HI, ROW1_HI, ROW1_HI, 0 /* instr_mod1 */);  // the new carry pair

    // Involution: this bank returns to store order, the other bank to its pre-transpose contents -
    // that is what keeps the previous pair's carry intact.
    TTI_SFPTRANSP;

    // Write the row pair back to the same two slots, then step Dest to the next row pair
    TTI_SFPSTORE(ROW0_LO, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg_addr */);
    TTI_SFPSTORE(ROW0_HI, p_sfpu::sfpmem::DEFAULT, CUMSUM_ADDR_MOD, 0 /* done */, CUMSUM_SECOND_ACCESS_OFF);
}

/**
 * @brief Column-wise (top-to-bottom) cumulative sum of one whole 32x32 Dest tile, in place.
 *
 * Walks the tile as CUMSUM_ROW_PAIRS passes on alternating LREG banks, issued as one recorded
 * two-pass replay body run CUMSUM_REPLAY_ITERS times.
 *
 * @param first: Whether this is the first tile of a top-to-bottom chain; zeroes the carry bank.
 * @note Run this once per tile under VectorMode::RC_custom, not once per face - the chain spans the
 *       whole tile. It leaves the Dest RWC counter advanced by a whole tile, which
 *       @ref _llk_math_eltwise_sfpu_done_ resets.
 * @note On return CUMSUM_LREG_BANK_B's ROW1 pair (LREG6/LREG7) holds this tile's 32 column totals -
 *       the carry the next call consumes with first == false. Feed tiles top-to-bottom and write
 *       nothing to LREG4-7 in between.
 * @note Uses replay slot 0 on the math thread.
 * @note Call @ref cumsum_init before this to program the address mode it depends on.
 */
inline void calculate_cumsum(const bool first) {
    if (first) {
        // Zero the whole carry bank, not just its ROW1 pair: SFPTRANSP swizzles lanes across the
        // entire bank, so a non-zero ROW0 pair would surface in the carry pair's lanes.
        TTI_SFPMOV(p_sfpu::LCONST_0, CUMSUM_LREG_BANK_B + CUMSUM_ROW0_LO, 0 /* instr_mod1: plain copy */);
        TTI_SFPMOV(p_sfpu::LCONST_0, CUMSUM_LREG_BANK_B + CUMSUM_ROW0_HI, 0 /* instr_mod1: plain copy */);
        TTI_SFPMOV(p_sfpu::LCONST_0, CUMSUM_LREG_BANK_B + CUMSUM_ROW1_LO, 0 /* instr_mod1: plain copy */);
        TTI_SFPMOV(p_sfpu::LCONST_0, CUMSUM_LREG_BANK_B + CUMSUM_ROW1_HI, 0 /* instr_mod1: plain copy */);
    }

    // Record the bank-A + bank-B pass pair; exec_while_loading runs row pairs 0-1 while recording.
    load_replay_buf<CUMSUM_REPLAY_SLOT, CUMSUM_REPLAY_LEN, true /* exec_while_loading */>([] {
        _calculate_cumsum_row_pair_<CUMSUM_LREG_BANK_A>();
        _calculate_cumsum_row_pair_<CUMSUM_LREG_BANK_B>();
    });

    // Replay the remaining bank pairs; Dest auto-increment carries them down the tile.
    for (std::uint32_t iter = 1; iter < CUMSUM_REPLAY_ITERS; iter++) {
        TTI_REPLAY(
            CUMSUM_REPLAY_SLOT,
            CUMSUM_REPLAY_LEN,
            0 /* last */,
            0 /* set_mutex */,
            0 /* execute_while_loading */,
            0 /* load_mode */);
    }
}

}  // namespace sfpu
}  // namespace ckernel
