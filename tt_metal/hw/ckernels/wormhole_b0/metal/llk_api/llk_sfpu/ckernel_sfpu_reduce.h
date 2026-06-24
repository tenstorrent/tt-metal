// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "llk_defs.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// ============================================================================
// Constants for 32x32 Tile Layout
// ============================================================================
// Each face is 16 rows, tile has 4 faces arranged as:
// Face 0 (rows 0-15)  | Face 1 (rows 0-15)
// Face 2 (rows 16-31) | Face 3 (rows 16-31)

constexpr std::uint32_t NUM_FACES = 4;
constexpr std::uint32_t UPPER_FACE_ADDRS[NUM_FACES] = {0, 0, 16, 16};    // Face 0, 0, 1, 1
constexpr std::uint32_t LOWER_FACE_ADDRS[NUM_FACES] = {32, 32, 48, 48};  // Face 2, 2, 3, 3
constexpr std::uint32_t COLUMN_OFFSETS[NUM_FACES] = {0, 2, 0, 2};        // even, odd, even, odd

constexpr std::uint32_t ROWS_PER_LOAD = 4;

// Constants for averaging (division by 32)
constexpr std::uint32_t AVG_SHIFT_AMOUNT = 5;    // 2^5 = 32
constexpr std::uint32_t AVG_SHIFT_MASK = 0xfff;  // Mask for shift instruction encoding

// Constants for MAX/MIN reduction
constexpr std::uint32_t ROWS_PER_TILE = 64;
constexpr std::uint32_t ROWS_PER_FACE = 16;

// Register holding the 0x0000FFFF mask used to clear garbage high bits when loading UInt16 data
// from a 32-bit (fp32 dest accumulation) dest word. Maps to sfpi::vConstIntPrgm0 on Wormhole B0.
constexpr std::uint32_t CLEAR_REG = p_sfpu::LREG12;

// SFPLOAD wrapper that optionally clears the high 16 bits of the loaded value. Needed for UInt16
// reduce with fp32 dest accumulation: the datum lives in the low 16 bits of a 32-bit dest word and
// the high 16 bits are garbage, which would otherwise corrupt integer compare/add operations.
template <bool clear_high_bits>
inline void load_and_clear_high_bits(
    const std::uint32_t lreg_ind,
    const InstrModLoadStore instr_mod0,
    const std::uint32_t sfpu_addr_mode,
    const std::uint32_t dest_reg_addr) {
    TT_SFPLOAD(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr);
    if constexpr (clear_high_bits) {
        TT_SFPAND(0, CLEAR_REG, lreg_ind, 0);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Load data from upper and lower faces into LREG0-7
 * @tparam INSTRUCTION_MODE The instruction mode for load operations
 * @param upper_face_addr Base address of upper face (Face 0 or Face 1)
 * @param lower_face_addr Base address of lower face (Face 2 or Face 3)
 * @param column_offset Column offset for the current iteration
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits>
inline void load_face_data(std::uint32_t upper_face_addr, std::uint32_t lower_face_addr, std::uint32_t column_offset) {
    // Load upper face data (Face 0 or Face 1) into LREG0-3
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, upper_face_addr + column_offset);  // rows 0-3
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, upper_face_addr + column_offset + ROWS_PER_LOAD);  // rows 4-7
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_3, upper_face_addr + column_offset + 2 * ROWS_PER_LOAD);  // rows 8-11
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG3,
        INSTRUCTION_MODE,
        ADDR_MOD_3,
        upper_face_addr + column_offset + 3 * ROWS_PER_LOAD);  // rows 12-15

    // Load lower face data (Face 2 or Face 3) into LREG4-7
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, lower_face_addr + column_offset);  // rows 0-3
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, lower_face_addr + column_offset + ROWS_PER_LOAD);  // rows 4-7
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_3, lower_face_addr + column_offset + 2 * ROWS_PER_LOAD);  // rows 8-11
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG7,
        INSTRUCTION_MODE,
        ADDR_MOD_3,
        lower_face_addr + column_offset + 3 * ROWS_PER_LOAD);  // rows 12-15
}

/**
 * @brief Load all 16 rows of a single face into LREG0-3 (4 rows per LREG)
 * @tparam INSTRUCTION_MODE The instruction mode for load operations
 * @tparam clear_high_bits Whether to mask the garbage high bits on load (true for UInt16 in 32-bit dest)
 * @param face_addr Base address of face
 * @param column_offset Column offset for the current iteration: even columns (0) or odd columns (2) of the face
 * @tparam DST_LREG_BASE First of the four consecutive LREGs to load into (defaults to LREG0).
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, std::uint32_t DST_LREG_BASE = p_sfpu::LREG0>
inline void load_face_data(std::uint32_t face_addr, std::uint32_t column_offset) {
    // Load the 4 row-groups into DST_LREG_BASE..DST_LREG_BASE+3. DST_LREG_BASE defaults to LREG0, but
    // callers can target LREG4 to feed a recorded swap buffer that operates on LREG4-7 directly, avoiding
    // a redundant LREG0-3 -> LREG4-7 shuffle.
    load_and_clear_high_bits<clear_high_bits>(
        DST_LREG_BASE + 0, INSTRUCTION_MODE, ADDR_MOD_3, face_addr + column_offset);  // rows 0-3
    load_and_clear_high_bits<clear_high_bits>(
        DST_LREG_BASE + 1, INSTRUCTION_MODE, ADDR_MOD_3, face_addr + column_offset + ROWS_PER_LOAD);  // rows 4-7
    load_and_clear_high_bits<clear_high_bits>(
        DST_LREG_BASE + 2, INSTRUCTION_MODE, ADDR_MOD_3, face_addr + column_offset + 2 * ROWS_PER_LOAD);  // rows 8-11
    load_and_clear_high_bits<clear_high_bits>(
        DST_LREG_BASE + 3, INSTRUCTION_MODE, ADDR_MOD_3, face_addr + column_offset + 3 * ROWS_PER_LOAD);  // rows 12-15
}

/**
 * @brief Perform integer averaging with proper handling of negative numbers
 * @tparam INSTRUCTION_MODE The instruction mode (determines signed vs unsigned)
 *
 * For integer formats, we need to handle negative numbers properly for division by 32.
 * Since Wormhole B0 only supports logical shift (not arithmetic), we need to:
 * 1. Check if the number is negative using condition codes (only for signed formats)
 * 2. If negative, negate it, shift right by 5 bits, then negate back
 * 3. If positive, just shift right by 5 bits
 */
template <InstrModLoadStore INSTRUCTION_MODE>
inline void perform_int_average() {
    if constexpr (INSTRUCTION_MODE == InstrModLoadStore::INT32) {
        // For signed Int32 format, use absolute value approach for proper division by 32
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);  // Save original value for sign check
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);  // Get absolute value of LREG0
        TTI_SFPSHFT(
            -AVG_SHIFT_AMOUNT & AVG_SHIFT_MASK,
            p_sfpu::LREG0,
            p_sfpu::LREG0,
            0b01);  // Perform logical right shift by 5 bits (divide by 32)

        // Restore sign if original value was negative
        // Check if original value was negative (sign bit set)
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 4);  // Set condition code if original sign bit is 0 (positive)
        TTI_SFPCOMPC(0, 0, 0, 0);              // Invert condition code (now true if original was negative)
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 6);  // Negate LREG0 if condition is true
        TTI_SFPENCC(0, 0, 0, 0);                             // Clear condition codes
    } else if constexpr (INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP) {
        // Two's-complement signed divide-by-32 (round toward zero). SFPABS clears the sign bit and is
        // only correct for sign-magnitude, so for 2's-complement we take the magnitude via a conditional
        // negate (0 - x), logical-shift, then restore the sign with a second conditional negate.
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);      // Save original (2's-complement) value for sign check
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 4);                // cc if sign bit == 0 (non-negative)
        TTI_SFPCOMPC(0, 0, 0, 0);                            // Invert -> cc if negative
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 6);  // LREG0 = -LREG0 (magnitude) when negative
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSHFT(-AVG_SHIFT_AMOUNT & AVG_SHIFT_MASK, p_sfpu::LREG0, p_sfpu::LREG0, 0b01);  // |x| >> 5 (divide by 32)
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 4);                // cc if original sign bit == 0 (non-negative)
        TTI_SFPCOMPC(0, 0, 0, 0);                            // Invert -> cc if original was negative
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 6);  // Restore sign (2's-complement negate) when negative
        TTI_SFPENCC(0, 0, 0, 0);
    } else {
        // For unsigned formats (UInt32), just use logical shift directly since they can't be negative
        TTI_SFPSHFT(-AVG_SHIFT_AMOUNT & AVG_SHIFT_MASK, p_sfpu::LREG0, p_sfpu::LREG0, 0b01);
    }
}

// Programmable float constant register holding 1/32 (0.03125) for float AVG. Preloaded once by
// init_reduce_sum_avg (only when pool_type == AVG and the format is float), so perform_float_average
// avoids rebuilding the constant via two SFPLOADI on every column group. Maps to LREG12 on Wormhole B0
// (sfpi::vConstFloatPrgm0); the float reduce path does not use the UInt16 high-bit mask, so this
// register is free to hold the constant across the whole reduce.
constexpr std::uint32_t AVG_RECIP_REG = p_sfpu::LREG12;

/**
 * @brief Perform floating-point averaging (multiply by 1/32)
 *
 * For a 32x32 tile, each column sum represents the sum of exactly 32 values (one per row).
 * This function divides by 32 by multiplying by the constant 1/32 (0.03125), which
 * init_reduce_sum_avg() preloaded into AVG_RECIP_REG (sfpi::vConstFloatPrgm0).
 */
inline void perform_float_average() {
    // Multiply by 1/32 (divide by 32) using the preloaded constant register.
    TTI_SFPMUL(p_sfpu::LREG0, AVG_RECIP_REG, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_NOP;  // Required after SFPMUL due to 2-cycle latency
}

template <PoolType pool_type, InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16>
inline void perform_reduce_col_sum_avg() {
    // Determine if integer or float mode at compile time
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP ||
         INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Optimized column reduction: Reduce → Add → Transpose → HalfReduce
    // Instead of the naive Transpose → Reduce → Transpose → Reduce → Add approach, we first reduce
    // across registers, then add upper+lower faces (all 4 positions carry meaningful partial sums),
    // then transpose, then do a final half-reduce on LREG0-3 only. This eliminates one transpose
    // and halves the second reduction pass.
    for (std::uint32_t i = 0; i < NUM_FACES; i++) {
        const std::uint32_t upper_face_addr = UPPER_FACE_ADDRS[i];
        const std::uint32_t lower_face_addr = LOWER_FACE_ADDRS[i];
        const std::uint32_t column_offset = COLUMN_OFFSETS[i];
        load_face_data<INSTRUCTION_MODE, clear_high_bits>(upper_face_addr, lower_face_addr, column_offset);

        // Step 1: Tree-reduce across registers (LREG0-3→LREG0, LREG4-7→LREG4) without transpose.
        lltt::replay(0, 6);

        // Step 2: Cross-face addition — all 4 positions carry meaningful partial sums.
        if constexpr (is_integer_mode) {
            TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, 4);  // LREG0 = upper + lower (int)
        } else {
            TTI_SFPNOP;  // Wait for LREG4 (2-cycle SFPADD latency)
            TTI_SFPADD(
                p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);  // LREG0 = upper + lower (float)
            TTI_SFPNOP;                                                             // Wait for LREG0 before SFPTRANSP
        }

        // Step 3: Transpose to rearrange the 4 partial sums for final reduction
        TTI_SFPTRANSP(0, 0, 0, 0);

        // Step 4: Final tree-reduce across LREG0-3 only (LREG4-7 no longer needed).
        if constexpr (is_integer_mode) {
            lltt::replay(0, 3);  // Reuse first 3 positions of sequential integer buffer (A1,A2,A3)
        } else {
            lltt::replay(6, 4);  // Half reduce with NOP for 2-cycle SFPADD latency
            // The half reduce ends with an SFPADD producing LREG0 (2-cycle latency). The next consumer of
            // LREG0 is either SFPSTORE (SUM path) or the SFPMUL in perform_float_average() (AVG path, which
            // now reads the preloaded 1/32 constant register instead of rebuilding it with two independent
            // SFPLOADI). Both consume LREG0 immediately, so a NOP is required in either case.
            TTI_SFPNOP;
        }

        // Perform averaging if requested (different for int vs float)
        if constexpr (pool_type == PoolType::AVG) {
            if constexpr (is_integer_mode) {
                perform_int_average<INSTRUCTION_MODE>();
            } else {
                perform_float_average();
            }
        }
        // Store the final column sum/average to the first row.
        // Mode 9 (SFPSTORE_MOD0_FMT_LO16) is only needed when the packer-visible OUTPUT is UInt16 in a
        // 32-bit dest: there the reduced value sits in the low 16 bits but the packer reads the high 16,
        // so we move low->high. When the output is a full 32-bit format (e.g. UInt32) the packer reads
        // the whole dest word, so we use the plain INSTRUCTION_MODE store even for UInt16 input.
        constexpr std::uint32_t STORE_MODE =
            pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
        TTI_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_3, upper_face_addr + column_offset);
    }
}

/**
 * @brief Performs two horizontal reductions in parallel (LREG0/LREG1 and LREG4/LREG5), interleaving
 *        instructions to hide SFPSHFT2 latency.
 *
 * SFPU hardware operates on 8 column slices in parallel but independently; column slices cannot
 * directly communicate. SFPSHFT2 is the only instruction that moves data across columns.
 * This function reduces 8 partial sums (one per column) in each of two LREG pairs down to a single
 * total sum in column 0 of 8 rows (LREG0 and LREG4).
 *
 * Interleaving: SFPSHFT2 has 2-cycle latency and would normally require SFPNOP after each use.
 * We run two reductions in lockstep (LREG0/LREG1 and LREG4/LREG5) so that the second SFPSHFT2
 * fills the latency slot of the first, avoiding extra NOPs and reducing total cycle count.
 *
 * Algorithm (log2(8) = 3 reduction stages + 1 rotate). Each stage: copy to temp, shift right
 * by the appropriate amount so columns align, then add (SFPIADD or SFPADD) to fold halves together.
 *
 *   Phase 1: Shift by 4 and add -> 8 partial sums (cols 0-7) become 4 sums (cols 4-7).
 *   Phase 2: Shift by 2 and add -> 4 sums become 2 sums (cols 6-7).
 *   Phase 3: Shift by 1 and add -> 2 sums become 1 sum (col 7).
 *   Phase 4: Rotate right by 1  -> move the single sum from column 7 to column 0.
 *
 * @tparam is_integer_mode True for integer types (uses SFPIADD), false for float (uses SFPADD)
 */
template <bool is_integer_mode>
inline void horizontal_reduce() {
    // Phase 1: Shift by 4 and add -> 8 partial sums (cols 0-7) become 4 sums (cols 4-7).
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);

    // Four right-shifts-by-4 in lockstep for both pairs; second SFPSHFT2 hides first's latency
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);

    if constexpr (is_integer_mode) {
        TTI_SFPNOP;
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);
    } else {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    }

    // Phase 2: Shift by 2 and add -> 4 sums become 2 sums (cols 6-7).
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    if constexpr (is_integer_mode) {
        TTI_SFPNOP;
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);
    } else {
        TTI_SFPADD(
            p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // lreg0 = lreg0 * 1.0 + lreg1 (float)
        TTI_SFPADD(
            p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);  // lreg4 = lreg4 * 1.0 + lreg5 (float)
    }

    // Phase 3: Shift by 1 and add -> 2 sums become 1 sum (col 7).
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    if constexpr (is_integer_mode) {
        TTI_SFPNOP;
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);
    } else {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    }

    // Phase 4: Rotate right by 1 -> move single sum from col 7 to col 0 for store.
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG0, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG4, 3);
    // LREG0[0 column slice] = sum of all elements in the first 4 rows of this 8-row block (first half)
    // LREG4[0 column slice] = sum of all elements in the next 4 rows of this 8-row block (second half)
}

constexpr std::uint32_t HORIZONTAL_REDUCE_MAX_REPLAY_LEN = 16;

/**
 * @brief Records phases 2-4 of horizontal max reduction into replay buffer at slot 0 (16 instructions).
 *
 * The full horizontal max reduction has 28 instructions across 4 phases:
 *   Phase 1: 2 MOV + 8 SHFT2 + 2 SWAP = 12 (inline, shift-by-4)
 *   Phase 2: 2 MOV + 4 SHFT2 + 2 SWAP =  8 (replay, shift-by-2)
 *   Phase 3: 2 MOV + 2 SHFT2 + 2 SWAP =  6 (replay, shift-by-1)
 *   Phase 4: 2 SHFT2                   =  2 (replay, rotate to col 0)
 *
 * Phase 1 (12 instr) stays inline; phases 2+3+4 (16 instr) fit exactly in one replay buffer.
 * Must be called once before perform_reduce_row_max_tile.
 */
inline void record_horizontal_reduce_max() {
    // Record phases 2, 3, and 4 into replay buffer (16 instructions).
    // Phase 1 (12 instructions) is too large to combine, so it stays inline.
    lltt::record(0, HORIZONTAL_REDUCE_MAX_REPLAY_LEN);

    // Phase 2: Shift by 2 and max -> 4 maxes become 2 maxes (cols 6-7).
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);

    // Phase 3: Shift by 1 and max -> 2 maxes become 1 max (col 7).
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);

    // Phase 4: Rotate right by 1 -> move single max from col 7 to col 0 for store.
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG0, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG4, 3);
}

/**
 * @brief Executes horizontal max reduction: phase 1 inline, phases 2-4 via replay buffer.
 *        record_horizontal_reduce_max() must have been called before first use.
 */
inline void horizontal_reduce_max() {
    // Phase 1 (inline): Shift by 4 and max -> 8 values become 4 maxes (cols 4-7).
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);

    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 4);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 4);

    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);

    // Phases 2, 3, 4 via replay buffer
    lltt::replay(0, HORIZONTAL_REDUCE_MAX_REPLAY_LEN);
}

/**
 * @brief Row-wise maximum reduction for a single 32x32 tile.
 *
 * Processes the tile in 2 face-pairs: (f0+f1) for tile rows 0-15, (f2+f3) for tile rows 16-31.
 * Each face-pair iteration processes 8 rows (two groups of 4 rows each).
 *
 * For each 8-row group:
 * 1. Load 4 rows from left face (even cols) and 4 rows from right face (odd cols) into LREG0-3
 * 2. Load the next 4 rows into LREG4-7
 * 3. Use vertical SFPSWAP to reduce LREG pairs down (keeping max between left/right face columns)
 * 4. Use horizontal_reduce_max to consolidate 8 SFPU columns into column 0
 * 5. Store the per-row max into column 0
 *
 * @tparam INSTRUCTION_MODE Load/store instruction mode (FP32, FP16B, or INT32 for sign-magnitude int max)
 * @param tile_row_offset Base row offset for this tile in the dest register
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits>
inline void perform_reduce_row_max_tile(std::uint32_t tile_row_offset, std::uint32_t result_store_mode) {
#pragma GCC unroll 2
    for (std::uint32_t face_pair = 0; face_pair < 2; face_pair++) {
        std::uint32_t face_pair_base = face_pair * 2 * ROWS_PER_FACE;

#pragma GCC unroll 2
        for (std::uint32_t row_group = 0; row_group < 2; row_group++) {
            std::uint32_t row_offset_first = row_group * 8;
            std::uint32_t row_offset_second = row_offset_first + 4;

            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_first);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_first + 2);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG2,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG3,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first + 2);

            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_second);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_second + 2);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG6,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG7,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second + 2);

            // Vertical max: reduce left/right face pairs via compare-and-swap.
            // Two groups (LREG0-3 and LREG4-7) are interleaved so that each SFPSWAP
            // on independent registers hides the 2-cycle latency of the previous one.
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, 1);  // hides LREG0/2 latency
            TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, 1);  // hides LREG4/6 latency
            TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, 1);  // hides LREG1/3 latency
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);  // hides LREG5/7 latency; LREG1 ready
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);  // hides LREG0/1 latency; LREG5 ready

            // Horizontal max: consolidate 8 SFPU columns into column 0
            horizontal_reduce_max();

            // result_store_mode is mode 9 only when this per-tile store is the final packer-visible
            // result (single column tile); otherwise it is intermediate and stays in the low 16 bits.
            TT_SFPSTORE(
                p_sfpu::LREG0, result_store_mode, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_first);
            TT_SFPSTORE(
                p_sfpu::LREG4, result_store_mode, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_second);
        }
    }
}

template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits>
inline void perform_reduce_row_sum_tile(std::uint32_t tile_row_offset, std::uint32_t result_store_mode) {
    // Determine if integer or float mode at compile time
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP ||
         INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Process tile in 2 face-pairs: (f0+f1) for tile rows 0-15, (f2+f3) for tile rows 16-31
    // Each face-pair iteration processes 8 rows (two groups of 4 rows each)
    for (std::uint32_t face_pair = 0; face_pair < 2; face_pair++) {
        // Base offset for this face pair:
        // face_pair 0: faces 0+1 (dest rows 0-31)
        // face_pair 1: faces 2+3 (dest rows 32-63)
        std::uint32_t face_pair_base = face_pair * 2 * ROWS_PER_FACE;

        for (std::uint32_t row_group = 0; row_group < 2; row_group++) {
            // Within each face, process rows in groups of 8 (two sub-groups of 4)
            std::uint32_t row_offset_first = row_group * 8;          // 0 or 8
            std::uint32_t row_offset_second = row_offset_first + 4;  // 4 or 12

            // Load 4 rows from face 0 (or 2) and face 1 (or 3)
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_first);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_first + 2);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG2,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG3,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first + 2);

            // Load next 4 rows from face 0 (or 2) and face 1 (or 3)
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_second);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_second + 2);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG6,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG7,
                INSTRUCTION_MODE,
                ADDR_MOD_3,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second + 2);

            // Replay the full tree-reduce buffer (both int and float use 6 instructions)
            lltt::replay(0, 6);

            // Horizontal reduction: consolidate all 8 SFPU columns into column 0 (interleaved for latency hiding)
            horizontal_reduce<is_integer_mode>();

            // result_store_mode is mode 9 (SFPSTORE_MOD0_FMT_LO16) only when this per-tile store is the
            // final, packer-visible result (single column tile); otherwise it is an intermediate store
            // re-loaded by the cross-tile accumulation and must stay in the low 16 bits.
            TT_SFPSTORE(
                p_sfpu::LREG0, result_store_mode, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_first);
            TT_SFPSTORE(
                p_sfpu::LREG4, result_store_mode, ADDR_MOD_3, tile_row_offset + face_pair_base + row_offset_second);
        }
    }
}

/**
 * @brief Accumulates partial row sums from all tiles in a row of tiles in tensor into tile 0 of that row.
 *
 * After per-tile row reduction, each tile has partial row sums in its column 0.
 * This function accumulates those row sums across all tiles into tile_row_base (first tile in this row of tiles in
 * tensor). Each tile already has 8 partial row-sum results in column 0 (written by perform_reduce_row_sum_tile): 2
 * face-pairs × 2 row-groups × 2 sums per group (first 4 rows and next 4 rows of each 8-row group) = 8. They are stored
 * at row offsets 0, 4, 8, 12, 32, 36, 40, 44. Each result occupies 4 rows (one LREG).
 *
 * We process these 8 results in two batches of 4. For each batch:
 * - Load tile 0's four LREGs into LREG0-3 from the first batch's offsets.
 * - For each other tile, load its four LREGs at the same offsets into LREG4-7 and add into LREG0-3.
 * - Store LREG0-3 back to tile 0.
 * LREG4-7 hold the other tiles' data so loads and adds can be pipelined without NOPs.
 *
 * @tparam INSTRUCTION_MODE The load/store instruction mode
 * @param tile_row_base Base address of the first tile in this row of tiles
 * @param block_ct_dim Number of tiles along x axis of tensor (column tiles)
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16>
inline void sum_first_columns_across_tiles(std::uint32_t tile_row_base, std::uint32_t block_ct_dim) {
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP ||
         INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Row offset for each of the 8 partial row-sum results (face 0: 0,4,8,12; face 2: 32,36,40,44)
    constexpr std::uint32_t RESULT_ROWS[8] = {0, 4, 8, 12, 32, 36, 40, 44};

    for (std::uint32_t batch = 0; batch < 2; batch++) {
        std::uint32_t base_idx = batch * 4;

        // Load tile 0's four LREGs at this batch's offsets (0,4,8,12 or 32,36,40,44) into LREG0-3
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 0]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 1]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 2]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 3]);

        // Accumulate from remaining tiles
        for (std::uint32_t t = 1; t < block_ct_dim; t++) {
            std::uint32_t tile_offset = tile_row_base + t * ROWS_PER_TILE;

            // Load tile t's four LREGs at the same offsets into LREG4-7
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 0]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 1]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 2]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 3]);

            // Add LREG4-7 into LREG0-3
            if constexpr (is_integer_mode) {
                TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, 4);
                TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG1, 4);
                TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG2, 4);
                TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG3, 4);
            } else {
                TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
                TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG1, 0);
                TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                TTI_SFPADD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG3, 0);
            }
        }

        // Store LREG0-3 back to tile 0. This is the final, packer-visible result, so it uses mode 9
        // (SFPSTORE_MOD0_FMT_LO16) only when the OUTPUT is UInt16 in a 32-bit dest (packer reads the
        // high 16 bits); a 32-bit output (e.g. UInt32) is stored with the plain INSTRUCTION_MODE.
        constexpr std::uint32_t STORE_MODE =
            pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
        TT_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 0]);
        TT_SFPSTORE(p_sfpu::LREG1, STORE_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 1]);
        TT_SFPSTORE(p_sfpu::LREG2, STORE_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 2]);
        TT_SFPSTORE(p_sfpu::LREG3, STORE_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 3]);
    }
}

/**
 * @brief Accumulates partial row maxima from all tiles in a row of tiles into tile 0.
 *
 * Mirrors sum_first_columns_across_tiles but uses SFPSWAP instead of SFPADD
 * to keep maximum values rather than sums.
 *
 * @tparam INSTRUCTION_MODE Load/store instruction mode (FP32 or INT32_2S_COMP for signed int max)
 * @param tile_row_base Base address of the first tile in this row of tiles
 * @param block_ct_dim Number of tiles along x axis of tensor (column tiles)
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16>
inline void max_first_columns_across_tiles(std::uint32_t tile_row_base, std::uint32_t block_ct_dim) {
    constexpr std::uint32_t RESULT_ROWS[8] = {0, 4, 8, 12, 32, 36, 40, 44};

    for (std::uint32_t batch = 0; batch < 2; batch++) {
        std::uint32_t base_idx = batch * 4;

        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 0]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 1]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 2]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 3]);

        for (std::uint32_t t = 1; t < block_ct_dim; t++) {
            std::uint32_t tile_offset = tile_row_base + t * ROWS_PER_TILE;

            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 0]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 1]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 2]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_3, tile_offset + RESULT_ROWS[base_idx + 3]);

            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG4, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG5, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG7, 1);
        }

        // Final, packer-visible store: use mode 9 (SFPSTORE_MOD0_FMT_LO16) only when the OUTPUT is
        // UInt16 in a 32-bit dest (pack_low16); a 32-bit output keeps the plain INSTRUCTION_MODE store.
        constexpr std::uint32_t STORE_MODE =
            pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
        TT_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 0]);
        TT_SFPSTORE(p_sfpu::LREG1, STORE_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 1]);
        TT_SFPSTORE(p_sfpu::LREG2, STORE_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 2]);
        TT_SFPSTORE(p_sfpu::LREG3, STORE_MODE, ADDR_MOD_3, tile_row_base + RESULT_ROWS[base_idx + 3]);
    }
}

template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16>
inline void perform_reduce_row_sum(std::uint32_t block_ct_dim, std::uint32_t block_rt_dim) {
    // When there is a single column tile, the per-tile store is the final packer-visible result and must
    // use mode 9 only when the OUTPUT is UInt16 in a 32-bit dest (pack_low16). With multiple column tiles
    // the per-tile store is intermediate (re-loaded by sum_first_columns_across_tiles) and must stay in the
    // low 16 bits via INSTRUCTION_MODE; the final mode-9 (if any) is applied by the cross-tile store.
    const std::uint32_t tile_store_mode = (pack_low16 && block_ct_dim == 1)
                                              ? 9u /* SFPSTORE_MOD0_FMT_LO16 */
                                              : static_cast<std::uint32_t>(INSTRUCTION_MODE);

    for (std::uint32_t i = 0; i < block_rt_dim; i++) {
        std::uint32_t tile_row_offset = ROWS_PER_TILE * block_ct_dim * i;

        // Step 1: Reduce each tile individually (horizontal reduction within each tile)
        for (std::uint32_t j = 0; j < block_ct_dim; j++) {
            std::uint32_t tile_offset = tile_row_offset + (ROWS_PER_TILE * j);
            perform_reduce_row_sum_tile<INSTRUCTION_MODE, clear_high_bits>(tile_offset, tile_store_mode);
        }

        // Step 2: Sum column 0 from all tiles in this row into tile 0's column 0 of this row
        if (block_ct_dim > 1) {
            sum_first_columns_across_tiles<INSTRUCTION_MODE, clear_high_bits, pack_low16>(
                tile_row_offset, block_ct_dim);
        }
    }
}

/**
 * @brief Row-wise maximum reduction across a block of tiles.
 *
 * For each row of tiles, reduces every tile individually via perform_reduce_row_max_tile,
 * then (if block_ct_dim > 1) accumulates the per-tile column-0 maxima across tiles using
 * compare-and-swap into tile 0's column 0.
 *
 * @tparam INSTRUCTION_MODE Load/store instruction mode (FP32, FP16B, or INT32 for sign-magnitude int max)
 * @param block_ct_dim Number of tiles along x axis of tensor (column tiles)
 * @param block_rt_dim Number of tiles along y axis of tensor (row tiles)
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16>
inline void perform_reduce_row_max(std::uint32_t block_ct_dim, std::uint32_t block_rt_dim) {
    record_horizontal_reduce_max();

    // Single column tile => per-tile store is the final packer-visible result, which uses mode 9 only
    // when the OUTPUT is UInt16 in a 32-bit dest (pack_low16); otherwise it is intermediate and stays
    // in the low 16 bits via INSTRUCTION_MODE.
    const std::uint32_t tile_store_mode = (pack_low16 && block_ct_dim == 1)
                                              ? 9u /* SFPSTORE_MOD0_FMT_LO16 */
                                              : static_cast<std::uint32_t>(INSTRUCTION_MODE);

    for (std::uint32_t i = 0; i < block_rt_dim; i++) {
        std::uint32_t tile_row_offset = ROWS_PER_TILE * block_ct_dim * i;

        for (std::uint32_t j = 0; j < block_ct_dim; j++) {
            std::uint32_t tile_offset = tile_row_offset + (ROWS_PER_TILE * j);
            perform_reduce_row_max_tile<INSTRUCTION_MODE, clear_high_bits>(tile_offset, tile_store_mode);
        }

        if (block_ct_dim > 1) {
            max_first_columns_across_tiles<INSTRUCTION_MODE, clear_high_bits, pack_low16>(
                tile_row_offset, block_ct_dim);
        }
    }
}

/**
 * @brief Runtime validation helper for supported data formats for reduce sfpu kernel
 */
constexpr bool is_supported_reduce_format(DataFormat format) {
    return format == DataFormat::Int32 || format == DataFormat::UInt32 || format == DataFormat::Float32 ||
           format == DataFormat::Float16_b || format == DataFormat::UInt16;
}

/**
 * @brief Configure address mode for SFPU reduce Max/Min kernel.
 * @param num_cols The number of columns in the tensor block of multiple tiles
 * @note One tile is 64 rows in dest
 */
inline void configure_addrmod_max_min(std::uint32_t num_cols) {
    // Reduction done on first tile before looping through the rest, so we look at num_cols - 1 tile
    std::uint32_t skip_rows = (num_cols - 1) * ROWS_PER_TILE;

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 16},
    }
        .set(ADDR_MOD_6);

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = static_cast<std::int16_t>(skip_rows)},
    }
        .set(ADDR_MOD_5);
}

// ============================================================================
// Init Reduce Kernels
// ============================================================================

/**
 * @brief Initialization for the manual (non-LOADMACRO) SFPU reduce MAX/MIN compare-and-swap path.
 *        Records a 3-instruction replay buffer that reduces LREG4-7 down to LREG4 via vertical
 *        compare-and-swaps, and sets the swap-direction config. Used by the Int32 path and by the
 *        UInt16-in-32-bit-dest path (calculate_reduce_max_min_uint16), where the load and the swap
 *        cannot be fused into a LOADMACRO (the garbage high bits must be masked in between).
 *
 * @tparam INSTRUCTION_MODE The instruction mode (INT32 for UInt16 in 32-bit dest)
 * @tparam pool_type The PoolType enum value (MAX or MIN). MAX inverts the swap direction here.
 */
template <InstrModLoadStore INSTRUCTION_MODE, PoolType pool_type>
inline void init_reduce_max_min_int32() {
    // Initialize SFPU config and set swap direction
    _init_sfpu_config_reg();
    if constexpr (pool_type == PoolType::MAX) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0100);  // Load lower 16 bits (bit 8)
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);  // Load upper 16 bits
        TTI_SFPCONFIG(0, 0xF, 0);
    }

    lltt::record(0, 3);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, 1);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG5, 1);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, 1);
}

/**
 * @brief Initialization for SFPU reduce MAX/MIN kernel.
 *        Sets up LOADMACRO sequences for compare-and-swap operations, configures address modifiers,
 *        and records replay buffers for efficient column-wise maximum/minimum reduction.
 *        For MIN operations, inverts the swap direction by configuring the SFPU control register.
 *
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, INT32_2S_COMP, LO16, DEFAULT
 * (FP32, FP16B)
 * @tparam pool_type The PoolType enum value (MAX or MIN). MIN inverts the swap direction for minimum reduction.
 * @param num_cols The number of columns to process (typically 32 for a single tile, or multiple of 32 for block
 * operations)
 */
template <InstrModLoadStore INSTRUCTION_MODE, PoolType pool_type, bool clear_high_bits>
inline void init_reduce_max_min(std::uint32_t num_cols) {
    // Initialize SFPU config and set swap direction before defining LOADMACRO sequences
    _init_sfpu_config_reg();

    // Invert swap direction for MIN operations, set 8th bit in SFPU config register
    if constexpr (pool_type == PoolType::MIN) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0100);  // Load lower 16 bits (bit 8)
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);  // Load upper 16 bits
        TTI_SFPCONFIG(0, 0xF, 0);
    }

    // Setup LOADMACRO sequence 0
    TTI_SFPSWAP(0, p_sfpu::LREG4, (0xC | p_sfpu::LREG0), 1);
    TTI_SFPLOADI(0, 0xA, 0x0084);
    TTI_SFPLOADI(0, 0x8, 0x0000);
    TTI_SFPCONFIG(0, 4, 0);

    // Setup LOADMACRO sequence 1
    TTI_SFPSWAP(0, p_sfpu::LREG5, (0xD | p_sfpu::LREG4), 1);
    TTI_SFPLOADI(0, 0xA, 0x0085);
    TTI_SFPLOADI(0, 0x8, 0x0000);
    TTI_SFPCONFIG(0, 5, 0);

    configure_addrmod_max_min(num_cols);

    // Record replay buffer for compare-and-swap operations
    // MAX uses LOADMACRO mechanism.
    // When clearing high bits (UInt16 in 32-bit dest), two extra SFPAND ops mask the garbage high bits
    // of the manually loaded face-1 columns (LREG0/LREG1) before the compare-and-swap, so the buffer
    // grows from 9 to 11 instructions.
    constexpr std::uint32_t buffer_len = clear_high_bits ? 11 : 9;
    lltt::record<lltt::NoExec>(0, buffer_len);
    TTI_INCRWC(0, 4, 0, 0);
    TTI_SFPLOADMACRO(5, INSTRUCTION_MODE, ADDR_MOD_3, 2);
    TTI_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, 16);
    TTI_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, 18);
    if constexpr (clear_high_bits) {
        TTI_SFPAND(0, CLEAR_REG, p_sfpu::LREG1, 0);
        TTI_SFPAND(0, CLEAR_REG, p_sfpu::LREG0, 0);
    }
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG1, 1);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG0, 1);
    TTI_SFPLOADMACRO(0, INSTRUCTION_MODE, ADDR_MOD_3, 0);

    // Dummy loads to increment dest counters
    TTI_SFPLOAD(8, INSTRUCTION_MODE, ADDR_MOD_6, 0);
    TTI_SFPLOAD(8, INSTRUCTION_MODE, ADDR_MOD_5, 0);
}

/**
 * @brief Initialization for SFPU reduce SUM and AVG kernels.
 *        Records replay buffers for column-wise summation using tree reduction.
 *        Integer: 6 sequential instructions (positions 0-5), first 3 reusable as half-reduce.
 *        Float: 6 interleaved instructions (positions 0-5) + 5 half-reduce with NOPs (positions 6-10).
 *        Interleaving eliminates NOPs in the full reduce by using B operations to hide A's 2-cycle latency.
 *
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, INT32_2S_COMP, LO16, DEFAULT
 * (FP32, FP16B)
 * @tparam pool_type The PoolType enum value (SUM or AVG). AVG with a float format preloads the 1/32 constant.
 */
template <InstrModLoadStore INSTRUCTION_MODE, PoolType pool_type>
inline void init_reduce_sum_avg() {
    _init_sfpu_config_reg();

    // Determine if integer or float mode based on INSTRUCTION_MODE
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP ||
         INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Float AVG divides by 32 by multiplying by 1/32. Preload that constant once here (only when it is
    // actually needed: float AVG) into the programmable float const register AVG_RECIP_REG so
    // perform_float_average() collapses to a single SFPMUL instead of rebuilding the constant with two
    // SFPLOADI on every column group. Integer AVG uses perform_int_average (shift) and pays nothing; SUM
    // never averages, so it pays nothing either.
    if constexpr (pool_type == PoolType::AVG && !is_integer_mode) {
        sfpi::vConstFloatPrgm0 = 0.03125f;
    }

    if constexpr (is_integer_mode) {
        // Integer replay buffer: 6 sequential instructions (1-cycle SFPIADD latency, no NOPs needed)
        // Positions 0-2 form a natural half-reduce for LREG0-3 (reusable via replay(0, 3))
        lltt::record(0, 6);

        // Upper face column summation (LREG0-3)
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, 4);  // LREG2 = LREG2 + LREG3
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG1, 4);  // LREG1 = LREG1 + LREG2
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);  // LREG0 = LREG0 + LREG1

        // Lower face column summation (LREG4-7)
        TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG6, 4);  // LREG6 = LREG6 + LREG7
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, 4);  // LREG5 = LREG5 + LREG6
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);  // LREG4 = LREG4 + LREG5
    } else {
        // Float replay buffer: 10 instructions total
        // Positions 0-5: Full reduce, interleaved to eliminate NOPs (B hides A's 2-cycle latency)
        // Positions 6-9: Half reduce for LREG0-3 only, with NOP for dependent SFPADD chain
        lltt::record(0, 10);

        // Full reduce (positions 0-5): interleaved upper/lower face summation
        TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);  // A1
        TTI_SFPADD(p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG6, 0);  // B1 (hides A1 latency)
        TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG1, 0);  // A2
        TTI_SFPADD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG5, 0);  // B2 (hides A2 latency)
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // A3
        TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);  // B3 (hides A3 latency)

        // Half reduce (positions 6-9): upper face only, NOP required for 2-cycle SFPADD latency
        // Reorders A1/A3/A2 so A1 and A3 (disjoint outputs) issue back-to-back; only A3's latency before A2 needs a
        // NOP.
        TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);  // A1
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // A3
        TTI_SFPNOP;  // Cover A3 latency (A1 is already available)
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);  // A2
    }
}

// ============================================================================
// Calculate Functions
// ============================================================================

/**
 * @brief Column-wise MAX/MIN reduction for UInt16 data in a 32-bit (fp32) dest, single 32x32 tile.
 *
 * The UInt16 datum lives in the low 16 bits of a 32-bit dest word with arbitrary garbage in the high
 * 16 bits, so every value must be masked (AND 0x0000FFFF) before it participates in a compare-and-swap.
 * The LOADMACRO pipeline used by calculate_reduce_max_min() fuses the load and the swap into a single
 * macro, leaving no place to clear the high bits in between; the garbage then dominates the integer
 * comparison and produces wrong minima/maxima for any non-constant input.
 *
 * Instead we use the same manual load/mask/swap structure as the Int32 path. Masked UInt16 values are
 * always non-negative, so a plain integer SFPSWAP orders them correctly (no sign-magnitude cast needed);
 * we move LREG0-3 -> LREG4-7 to feed the recorded swap buffer. Final, packer-visible results are written
 * with SFPSTORE mode 9 (SFPSTORE_MOD0_FMT_LO16) so the packer reads the low bits.
 *
 * Reuses init_reduce_max_min_int32()'s 3-swap replay buffer and swap-direction config. Only a single
 * 32x32 tile is processed per call, matching the column-reduce driver which invokes the kernel per tile.
 *
 * @tparam INSTRUCTION_MODE The instruction mode (INT32 for UInt16 in 32-bit dest)
 * @tparam pool_type The pool type (MAX or MIN) to determine swap direction
 * @tparam reduce_dim The reduction dimension (currently only REDUCE_COL is supported)
 * @tparam clear_high_bits Whether to mask the garbage high bits on load (true for UInt16 in 32-bit dest)
 * @tparam pack_low16 Whether the final packer-visible store uses mode 9 (true for UInt16 OUTPUT in 32-bit dest)
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    InstrModLoadStore INSTRUCTION_MODE,
    bool clear_high_bits,
    bool pack_low16>
inline void calculate_reduce_max_min_uint16() {
    static_assert(reduce_dim == ReduceDim::REDUCE_COL, "Only column reduction (REDUCE_COL) is currently supported");
    static_assert(
        pool_type == PoolType::MAX || pool_type == PoolType::MIN,
        "Only MAX and MIN pool types are supported for this function");

    constexpr std::uint32_t ODD_COLUMNS = 2;
    constexpr std::uint32_t COLUMN_OFFSETS[4] = {0, 2, 0, 2};  // even, odd, even, odd
    constexpr std::uint32_t FACE_ADDRS[2][4] = {
        {0, 0, 32, 32},   // j=0: Face 0 and Face 2
        {16, 16, 48, 48}  // j=1: Face 1 and Face 3
    };
    constexpr std::uint32_t FINAL_REDUCE_ADDRS[2][2] = {
        {0, 32},  // j=0: Face 0 and Face 2
        {16, 48}  // j=1: Face 1 and Face 3
    };

    // The intermediate stores below land in non-row-0 dest slots that we reload, so they use the
    // plain (full 32-bit) instruction mode. Only the final row-0 stores are packer-visible and use
    // mode 9 (SFPSTORE_MOD0_FMT_LO16) when the OUTPUT is UInt16 in a 32-bit dest (pack_low16), so the
    // packer reads the low 16 bits of the dest word.
    constexpr std::uint32_t STORE_MODE =
        pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);

    for (std::uint32_t j = 0; j < 2; j++) {
        std::uint32_t top_face_addr = FINAL_REDUCE_ADDRS[j][0];     // face 0 & 1 dst indices
        std::uint32_t bottom_face_addr = FINAL_REDUCE_ADDRS[j][1];  // face 2 & 3 dst indices

        // Reduce each of the four vertically adjacent faces (f0,f2 then f1,f3) within itself; the
        // max/min of its 16 rows is left in the top 4 rows.
        for (std::uint32_t i = 0; i < NUM_FACES; i++) {
            // Masked load straight into LREG4-7, where the recorded swap buffer reduces them. Loading
            // directly into the target registers eliminates the four LREG0-3 -> LREG4-7 moves (and the
            // post-reduce LREG4 -> LREG0 move) that the previous LREG0-3 load required.
            load_face_data<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG4>(FACE_ADDRS[j][i], COLUMN_OFFSETS[i]);

            lltt::replay(0, 3);  // compare-and-swap reduce LREG4-7 -> LREG4

            TT_SFPSTORE(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, FACE_ADDRS[j][i] + COLUMN_OFFSETS[i]);
        }

        // Load the partial max/min (top 4 rows) of the two vertically adjacent faces into LREG0-3.
        load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, top_face_addr);
        load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, bottom_face_addr);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_3, top_face_addr + ODD_COLUMNS);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_3, bottom_face_addr + ODD_COLUMNS);

        // Move into LREG4-7 for the transpose + cross-row reduction.
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
        TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG5, 0);
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG6, 0);
        TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG7, 0);

        // Transpose so the 4 partial results of each column sit in one register, reduce, transpose back.
        TTI_SFPTRANSP(0, 0, 0, 0);
        lltt::replay(0, 3);
        TTI_SFPTRANSP(0, 0, 0, 0);

        // Swap to combine the two vertically adjacent faces (even and odd columns).
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, 1);  // odd columns of face pair
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, 1);  // even columns of face pair

        TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPMOV(0, p_sfpu::LREG6, p_sfpu::LREG1, 0);

        TT_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_3, top_face_addr);
        TT_SFPSTORE(p_sfpu::LREG1, STORE_MODE, ADDR_MOD_3, top_face_addr + ODD_COLUMNS);
    }
}

/**
 * @brief Column-wise maximum/minimum reduction kernel for SFPU reduce MAX/MIN operation.
 *        Processes a block of tiles vertically (block_height tiles stacked) and computes the maximum or minimum value
 *        for each of the columns across all rows in the block. The maximum/minimum values are placed into
 *        the first row of the output tile (row 0 of faces 0 and 1) in tilized format for each tile in the top row of
 * tiles in the block.
 *
 *        Algorithm:
 *        - Initializes LREG4-7 with the first face pair's data (even/odd columns from faces 0 and 1)
 *        - For each tile in the block, performs compare-and-swap operations using replay buffers to find maxima/minima
 *        - Uses SFPSWAP instruction for comparisons to determine maximum/minimum between two lregs storing column data
 *        - For MIN operations, swap direction is inverted (configured during init via SFPU control register)
 *        - Transposes and sorts results to align maxima/minima correctly across LREG4-7
 *        - Stores final maximum/minimum values to row 0 (32 datums across faces 0 and 1)
 *
 * @tparam pool_type The PoolType enum value (MAX or MIN). MIN uses inverted swap direction for minimum reduction.
 * @tparam reduce_dim The reduction dimension: REDUCE_COL for column-wise, REDUCE_ROW for row-wise (MAX only,
 * FP32/Int32).
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32_2S_COMP, LO16, DEFAULT (FP32,
 * FP16B)
 * @param block_ct_dim Number of tiles along x axis (column tiles, default 1).
 * @param block_rt_dim Number of tiles along y axis (row tiles, default 1).
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    InstrModLoadStore INSTRUCTION_MODE,
    bool clear_high_bits,
    bool pack_low16>
inline void calculate_reduce_max_min(const std::uint32_t block_ct_dim = 1, const std::uint32_t block_rt_dim = 1) {
    static_assert(
        reduce_dim == ReduceDim::REDUCE_COL || (pool_type == PoolType::MAX && reduce_dim == ReduceDim::REDUCE_ROW),
        "Only column reduction (REDUCE_COL) and row MAX reduction (REDUCE_ROW with MAX) are supported");

    if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
        static_assert(
            pool_type == PoolType::MAX || pool_type == PoolType::SUM,
            "Row reduction (REDUCE_ROW) currently only supports MAX and SUM pool types");
        perform_reduce_row_max<INSTRUCTION_MODE, clear_high_bits, pack_low16>(block_ct_dim, block_rt_dim);
    } else {
        // The recorded replay buffer in init_reduce_max_min has two extra SFPAND ops when clearing
        // high bits, so the per-face-pair replay window grows accordingly.
        constexpr std::uint32_t replay_buffer_offset = clear_high_bits ? 9 : 7;
        constexpr std::uint32_t replay_buffer_next_face = clear_high_bits ? 10 : 8;

        // Initial loads: LREG4-7 will hold maximum values across F0 and F1
        TTI_SFPLOAD(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_3, 0);
        TTI_SFPLOAD(p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_3, 2);
        TTI_SFPLOAD(p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_3, 16);
        TTI_SFPLOAD(p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_3, 18);

        // First tile processing (F0, F1, F2, F3)
        lltt::replay(0, replay_buffer_offset);
        lltt::replay(0, replay_buffer_offset);
        lltt::replay(0, replay_buffer_next_face);

        lltt::replay(0, replay_buffer_offset);
        lltt::replay(0, replay_buffer_offset);
        lltt::replay(0, replay_buffer_offset);
        lltt::replay(0, replay_buffer_next_face + 1);

        // Remaining tiles
        for (std::uint32_t i = 0; i < block_rt_dim - 1; i++) {
            lltt::replay(0, replay_buffer_offset);
            lltt::replay(0, replay_buffer_offset);
            lltt::replay(0, replay_buffer_offset);
            lltt::replay(0, replay_buffer_next_face);

            lltt::replay(0, replay_buffer_offset);
            lltt::replay(0, replay_buffer_offset);
            lltt::replay(0, replay_buffer_offset);
            lltt::replay(0, replay_buffer_next_face + 1);
        }

        // Reset dest RWC counter
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

        // Finalize: Sort and store maximum/minimum values to row 0
        TTI_SFPTRANSP(0, 0, 0, 0);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, 1);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG6, 1);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);
        TTI_SFPTRANSP(0, 0, 0, 0);

        // Store results to first row.
        // For UInt16 OUTPUT in a 32-bit dest the reduced value lives in the low 16 bits of the LREG, but
        // the packer reads the high 16 bits of the dest word. SFPSTORE mode 9 (SFPSTORE_MOD0_FMT_LO16)
        // writes the low 16 bits into the half the packer consumes; a 32-bit output keeps the plain store.
        constexpr std::uint32_t STORE_MODE =
            pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
        TTI_SFPSTORE(p_sfpu::LREG4, STORE_MODE, ADDR_MOD_3, 0);
        TTI_SFPSTORE(p_sfpu::LREG5, STORE_MODE, ADDR_MOD_3, 2);
        TTI_SFPSTORE(p_sfpu::LREG6, STORE_MODE, ADDR_MOD_3, 16);
        TTI_SFPSTORE(p_sfpu::LREG7, STORE_MODE, ADDR_MOD_3, 18);
    }
}

/**
 * @brief Column-wise sum/average reduction kernel for SFPU reduce SUM and AVG operations.
 *        Computes the sum or average of each column, placing the 32 output values into the first row
 *        of the output tile (row 0 of faces 0 and 1).
 *
 *        Uses a 4-iteration approach that processes vertically aligned face pairs (0+2, 1+3) to optimize
 *        column operations and minimize load/store operations. Each iteration handles 8 columns using
 *        transpose operations and replay buffers for tree reduction.
 *
 *        For AVG mode: Integer formats use arithmetic shift with condition codes to handle negative numbers;
 *        float formats multiply by 1/32 constant.
 *
 * @tparam pool_type The reduction operation, currently supported: (SUM, AVG)
 * @tparam reduce_dim The reduction dimension (currently only REDUCE_COL is supported)
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, INT32_2S_COMP, LO16, DEFAULT
 * (FP32, FP16B)
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    InstrModLoadStore INSTRUCTION_MODE,
    bool clear_high_bits,
    bool pack_low16>
inline void calculate_reduce_sum_avg(std::uint32_t block_ct_dim, std::uint32_t block_rt_dim) {
    // Compile-time assertions to restrict to currently supported operations
    static_assert(
        reduce_dim == ReduceDim::REDUCE_COL || (pool_type == PoolType::SUM && reduce_dim == ReduceDim::REDUCE_ROW),
        "Only column reduction (REDUCE_COL) is supported, except row reduction (REDUCE_ROW) is allowed only for SUM");
    static_assert(
        pool_type == PoolType::SUM || pool_type == PoolType::AVG,
        "Only SUM and AVG pool types are currently supported on SFPU");

    // Supported instruction modes for SFPU reduce sum/avg (integer and float)
    constexpr bool is_supported_reduce_instr_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP ||
         INSTRUCTION_MODE == InstrModLoadStore::LO16 || INSTRUCTION_MODE == InstrModLoadStore::DEFAULT ||
         INSTRUCTION_MODE == InstrModLoadStore::FP32 || INSTRUCTION_MODE == InstrModLoadStore::FP16B);
    static_assert(
        is_supported_reduce_instr_mode,
        "INSTRUCTION_MODE must be one of: INT32, INT32_2S_COMP, LO16, DEFAULT, FP32, FP16B");

    if constexpr (reduce_dim == ReduceDim::REDUCE_COL) {
        perform_reduce_col_sum_avg<pool_type, INSTRUCTION_MODE, clear_high_bits, pack_low16>();
    } else {
        static_assert(pool_type == PoolType::SUM, "Row reduction (REDUCE_ROW) is allowed only for SUM");
        perform_reduce_row_sum<INSTRUCTION_MODE, clear_high_bits, pack_low16>(block_ct_dim, block_rt_dim);
    }
    // For column reductions: sums are stored horizontally in the first row of tensor in dest reg
    // For row reductions: sums are stored vertically in the first column of tensor in dest reg
}

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Unified reduction init kernel wrapper for SFPU reduce kernel.
 *        Determines the instruction mode from format, then dispatches to the appropriate init kernel.
 * @tparam pool_type The reduction operation, currently supported: (SUM, AVG, MAX)
 * @tparam format The data format, currently supported: (Int32, UInt32, UInt16, Float32, Float16_b)
 * @param block_ct_dim Block dimension (used for MAX reduction to specify number of columns, default is 1 for single
 * tile)
 */
template <PoolType pool_type, DataFormat format, bool is_fp32_dest_accum_en>
inline void init_reduce(std::uint32_t block_ct_dim = 1) {
    static_assert(
        is_supported_reduce_format(format),
        "Unsupported data format. Supported formats: Int32, UInt32, UInt16, Float32, Float16_b");

    // Determine InstrModLoadStore from llk_defs. Int32 SUM/AVG use INT32_2S_COMP so SFPIADD operates
    // on two's-complement values. Int32 MAX/MIN keep plain INT32 (sign-magnitude): SFPSWAP(VEC_MIN_MAX)
    // is a float/sign-magnitude comparator and orders sign-magnitude integers correctly, whereas
    // two's-complement negatives would be mis-ordered.
    constexpr InstrModLoadStore INSTRUCTION_MODE =
        (format == DataFormat::Int32 && (pool_type == PoolType::SUM || pool_type == PoolType::AVG))
            ? InstrModLoadStore::INT32_2S_COMP
        : (format == DataFormat::Float16_b) ? InstrModLoadStore::DEFAULT
                                            : GetSfpLoadStoreInstrMod<format, is_fp32_dest_accum_en>();

    // Garbage high bits need to be cleared when loading UInt16 data from a 32-bit (fp32) dest word.
    constexpr bool clear_high_bits = (is_fp32_dest_accum_en && format == DataFormat::UInt16);

    if constexpr (clear_high_bits) {
        // CLEAR_REG (sfpi::vConstIntPrgm0 / LREG12) holds the mask used by SFPLOAD_EXT to clear high bits.
        sfpi::vConstIntPrgm0 = 0x0000FFFF;
    }

    // Dispatch to appropriate PoolType init
    if constexpr (pool_type == PoolType::MAX || pool_type == PoolType::MIN) {
        if constexpr (clear_high_bits) {
            // UInt16 in 32-bit dest uses the manual (non-LOADMACRO) compare-and-swap path so the
            // garbage high bits can be masked before each swap. It reuses the Int32 path's 3-swap
            // replay buffer and swap-direction config (the body is format-agnostic).
            init_reduce_max_min_int32<INSTRUCTION_MODE, pool_type>();
        } else {
            init_reduce_max_min<INSTRUCTION_MODE, pool_type, false>(block_ct_dim);
        }
    } else if constexpr (pool_type == PoolType::SUM || pool_type == PoolType::AVG) {
        init_reduce_sum_avg<INSTRUCTION_MODE, pool_type>();
    } else {
        static_assert(
            pool_type == PoolType::SUM || pool_type == PoolType::AVG || pool_type == PoolType::MAX,
            "Unsupported pool_type. Currently supported: SUM, AVG, MAX");
    }
}

/**
 * @brief Unified reduction kernel wrapper for a 32x32 tile.
 *        Determines the instruction mode from format, then dispatches to the appropriate reduction kernel.
 * @tparam pool_type The reduction operation, currently supported: (SUM, AVG, MAX, MIN)
 * @tparam reduce_dim The reduction dimension (currently only REDUCE_COL is supported)
 * @tparam format The INPUT data format, currently supported: (Int32, UInt32, UInt16, Float32, Float16_b). Drives the
 *         instruction mode and load-time high-bit masking.
 * @tparam output_format The packer-visible OUTPUT data format (defaults to @p format). Drives the final store mode:
 *         UInt16 output in a 32-bit dest is stored via mode 9 (low->high 16-bit swap), while a 32-bit output (e.g.
 *         UInt32) is stored with the plain instruction mode. This lets UInt16 input be summed into a UInt32 output
 *         without overflow.
 * @param block_ct_dim Block dimension (used for SUM/AVG column reduction to specify number of columns, default is 1 for
 * single tile)
 * @param block_rt_dim Block dimension (used for MAX/MIN reduction to specify block height, or SUM row reduction;
 * default is 1 for single tile)
 *
 * @note Constraints (unable to static assert for block_rt_dim runtime parameter)
 *       - MAX/MIN with Int32 format only supports block_rt_dim == 1 (single tile)
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    DataFormat format,
    bool is_fp32_dest_accum_en,
    DataFormat output_format = format>
inline void calculate_reduce(std::uint32_t block_ct_dim = 1, std::uint32_t block_rt_dim = 1) {
    static_assert(
        reduce_dim == ReduceDim::REDUCE_COL || (pool_type == PoolType::SUM && reduce_dim == ReduceDim::REDUCE_ROW) ||
            (pool_type == PoolType::MAX && reduce_dim == ReduceDim::REDUCE_ROW),
        "Row reduction (REDUCE_ROW) is supported for SUM and MAX pool types");
    static_assert(
        is_supported_reduce_format(format),
        "Unsupported data format. Supported formats: Int32, UInt32, UInt16, Float32, Float16_b");

    // Determine InstrModLoadStore from llk_defs.
    // Int32 SUM/AVG use INT32_2S_COMP so SFPIADD operates on two's-complement values. Int32 MAX/MIN
    // (both dims) keep plain INT32 (sign-magnitude): SFPSWAP(VEC_MIN_MAX) is a float/sign-magnitude
    // comparator that orders sign-magnitude integers correctly; two's-complement negatives are
    // mis-ordered (max of negatives would return the most-negative value).
    constexpr bool int32_sum_avg =
        (format == DataFormat::Int32 && (pool_type == PoolType::SUM || pool_type == PoolType::AVG));
    constexpr InstrModLoadStore INSTRUCTION_MODE = int32_sum_avg ? InstrModLoadStore::INT32_2S_COMP
                                                   : (format == DataFormat::Float16_b)
                                                       ? InstrModLoadStore::DEFAULT
                                                       : GetSfpLoadStoreInstrMod<format, is_fp32_dest_accum_en>();

    // Garbage high bits need to be cleared when loading UInt16 data from a 32-bit (fp32) dest word
    // (driven by INPUT format).
    constexpr bool clear_high_bits = (is_fp32_dest_accum_en && format == DataFormat::UInt16);

    // The packer-visible result must go through mode-9 (low->high 16-bit) store only when the OUTPUT is UInt16
    // in a 32-bit dest. A 32-bit output (e.g. UInt32) keeps the full word, so it uses the plain store. This is
    // driven by the OUTPUT format and is independent of the load-time masking above.
    constexpr bool pack_low16 = (is_fp32_dest_accum_en && output_format == DataFormat::UInt16);

    // Dispatch to appropriate reduction kernel based on PoolType
    if constexpr (pool_type == PoolType::MAX || pool_type == PoolType::MIN) {
        if constexpr (clear_high_bits) {
            // UInt16 in 32-bit dest: manual load/mask/swap path (LOADMACRO cannot mask between load and swap).
            calculate_reduce_max_min_uint16<pool_type, reduce_dim, INSTRUCTION_MODE, clear_high_bits, pack_low16>();
        } else {
            calculate_reduce_max_min<pool_type, reduce_dim, INSTRUCTION_MODE, false, pack_low16>(
                block_ct_dim, block_rt_dim);
        }
    } else if constexpr (pool_type == PoolType::SUM || pool_type == PoolType::AVG) {
        calculate_reduce_sum_avg<pool_type, reduce_dim, INSTRUCTION_MODE, clear_high_bits, pack_low16>(
            block_ct_dim, block_rt_dim);
    } else {
        static_assert(
            pool_type == PoolType::SUM || pool_type == PoolType::AVG || pool_type == PoolType::MAX ||
                pool_type == PoolType::MIN,
            "Unsupported pool_type. Currently supported: SUM, AVG, MAX, MIN");
    }
}

}  // namespace sfpu
}  // namespace ckernel
