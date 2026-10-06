// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/numeric/bfloat16.h"
#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"
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
constexpr std::uint32_t ROWS_PER_LOAD = 4;

// Constants for averaging (division by 32)
constexpr std::uint32_t AVG_SHIFT_AMOUNT = 5;    // 2^5 = 32
constexpr std::uint32_t AVG_SHIFT_MASK = 0xfff;  // Mask for shift instruction encoding

// Constants for MAX reduction
constexpr std::uint32_t ROWS_PER_TILE = 64;
constexpr std::uint32_t ROWS_PER_FACE = 16;

// Tile-layout address tables for the manual column MAX/MIN reduce path (calculate_reduce_max_min_uint16).
// Hoisted to file scope alongside the other reduce kernels for consistency.
//   COL_REDUCE_ODD_COLUMNS:    dest-word offset between the even- and odd-column halves of a face.
//   COL_REDUCE_COLUMN_OFFSETS: even/odd column-half selector per face index (even, odd, even, odd).
//   COL_REDUCE_FACE_ADDRS:     per-face load addresses for the two vertically adjacent face pairs.
//   COL_REDUCE_FINAL_ADDRS:    top/bottom face dst indices used by the cross-face reduce.
constexpr std::uint32_t COL_REDUCE_ODD_COLUMNS = 2;
constexpr std::uint32_t COL_REDUCE_COLUMN_OFFSETS[NUM_FACES] = {0, 2, 0, 2};  // even, odd, even, odd
constexpr std::uint32_t COL_REDUCE_FACE_ADDRS[2][NUM_FACES] = {
    {0, 0, 32, 32},   // j=0: Face 0 and Face 2
    {16, 16, 48, 48}  // j=1: Face 1 and Face 3
};
constexpr std::uint32_t COL_REDUCE_FINAL_ADDRS[2][2] = {
    {0, 32},  // j=0: Face 0 and Face 2
    {16, 48}  // j=1: Face 1 and Face 3
};

// Register holding the 0x0000FFFF mask used to clear garbage high bits when loading UInt16 data
// from a 32-bit (fp32 dest accumulation) dest word. Maps to sfpi::vConstIntPrgm0 on Blackhole.
constexpr std::uint32_t CLEAR_REG = p_sfpu::LREG12;

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

// SFPIADD adds in two's-complement while SFPSWAP compares in sign-magnitude, so the Int32 reduce paths
// have to move operands between the two encodings. Wormhole does this in the INT32_2S_COMP load/store
// mode; on Blackhole that mode is deprecated and does not convert (BlackholeA0 SFPLOAD.md), so the reduce
// code converts explicitly with this helper.
//
// It uses the same SFPCAST+SFPSETSGN primitive as the element-wise int kernels (see _add_int_): the cast
// is direction-neutral (one mode converts both ways) and uses no condition codes, so it is robust against
// SFPABS/cc interaction. It needs one free GPR (LREG0-7) as scratch; LREG8-15 are constant/config
// registers that SFPCAST cannot write.
// "Representation swap": converts an integer between sign-magnitude and two's-complement.
constexpr InstrModCast REDUCE_INT_REPRESENTATION_SWAP_CAST = InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP;

// Convert one int operand between sign-magnitude and two's-complement, in place. `scratch_gpr` must be a free
// GPR (LREG0-7) distinct from `reg`; it is clobbered. The result is left in `reg`.
inline void convert_int_representation_inplace(std::uint32_t reg, std::uint32_t scratch_gpr) {
    apply_sign_magnitude_conversion(reg, scratch_gpr, REDUCE_INT_REPRESENTATION_SWAP_CAST);
}

/**
 * @brief Load data from face into LREG0-3
 * @tparam INSTRUCTION_MODE The instruction mode for load operations
 * @param face_addr Base address of face
 * @param column_offset Column offset for the current iteration, load all rows for even columns (0) or odd columns (2)
 * of the face
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, std::uint32_t DST_LREG_BASE = p_sfpu::LREG0>
inline void load_face_data(std::uint32_t face_addr, std::uint32_t column_offset) {
    // Load the 4 row-groups into DST_LREG_BASE..DST_LREG_BASE+3. DST_LREG_BASE defaults to LREG0, but
    // callers can target LREG4 to feed a recorded swap buffer that operates on LREG4-7 directly, avoiding
    // a redundant LREG0-3 -> LREG4-7 shuffle.
    load_and_clear_high_bits<clear_high_bits>(
        DST_LREG_BASE + 0, INSTRUCTION_MODE, ADDR_MOD_7, face_addr + column_offset);  // rows 0-3
    load_and_clear_high_bits<clear_high_bits>(
        DST_LREG_BASE + 1, INSTRUCTION_MODE, ADDR_MOD_7, face_addr + column_offset + ROWS_PER_LOAD);  // rows 4-7
    load_and_clear_high_bits<clear_high_bits>(
        DST_LREG_BASE + 2, INSTRUCTION_MODE, ADDR_MOD_7, face_addr + column_offset + 2 * ROWS_PER_LOAD);  // rows 8-11
    load_and_clear_high_bits<clear_high_bits>(
        DST_LREG_BASE + 3, INSTRUCTION_MODE, ADDR_MOD_7, face_addr + column_offset + 3 * ROWS_PER_LOAD);  // rows 12-15
}

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
        p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, upper_face_addr + column_offset);  // rows 0-3
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, upper_face_addr + column_offset + ROWS_PER_LOAD);  // rows 4-7
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, upper_face_addr + column_offset + 2 * ROWS_PER_LOAD);  // rows 8-11
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG3,
        INSTRUCTION_MODE,
        ADDR_MOD_7,
        upper_face_addr + column_offset + 3 * ROWS_PER_LOAD);  // rows 12-15

    // Load lower face data (Face 2 or Face 3) into LREG4-7
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, lower_face_addr + column_offset);  // rows 0-3
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, lower_face_addr + column_offset + ROWS_PER_LOAD);  // rows 4-7
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, lower_face_addr + column_offset + 2 * ROWS_PER_LOAD);  // rows 8-11
    load_and_clear_high_bits<clear_high_bits>(
        p_sfpu::LREG7,
        INSTRUCTION_MODE,
        ADDR_MOD_7,
        lower_face_addr + column_offset + 3 * ROWS_PER_LOAD);  // rows 12-15
}

/**
 * @brief Divide the column-AVG accumulator in LREG0 by the 32-row column extent, in place, rounding toward zero.
 *
 * @tparam INSTRUCTION_MODE The load/store mode the column path is running under.
 * @tparam is_signed_int    Whether the *data* is signed, i.e. the reduce format is Int32. LREG0 then holds a
 *                          two's-complement sum; otherwise (UInt32, UInt16) it holds an unsigned 32-bit sum.
 *
 * @p is_signed_int is a property of the format, not of @p INSTRUCTION_MODE, and must be passed in
 * separately: llk_defs.h GetSfpLoadStoreInstrMod maps UInt32 - and UInt16 in a 32-bit Dest - to
 * InstrModLoadStore::INT32 just like signed Int32, so the mode cannot tell signed data from
 * unsigned. Branching on it sent unsigned column sums down the signed path, which read bit 31 of
 * the sum as a sign, so every column summing to 2^31 or more came back as the *signed* quotient of
 * the same bits rather than the unsigned one: 0xBEBC2000 divided to 0xFDF5E100, i.e. int32(sum)/32
 * (tenstorrent/tt-metal#57509).
 *
 * A bare arithmetic right shift would be wrong here, even though Blackhole's SFPSHFT has one
 * (sfpi::ShiftMode::Arithmetic, used by ckernel_sfpu_shift.h): it rounds toward negative infinity
 * where the golden truncates toward zero (-33/32 is -1, not -2). So the signed case shifts the
 * magnitude logically and then restores the sign. Unsigned data has no sign to restore and is a
 * bare logical shift.
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool is_signed_int>
inline void perform_int_average() {
    if constexpr (is_signed_int) {
        // Two's-complement signed divide-by-32 (round toward zero): shift the magnitude, then negate
        // the lanes whose sum was negative. Integer SFPABS is a two's-complement abs (tt-isa SFPABS.md)
        // and leaves INT32_MIN as 0x80000000, which the logical shift still maps to 2^26 and the
        // negate to INT32_MIN / 32.
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);                      // Save the signed sum for the sign check
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPABS_MOD1_INT);  // |x|
        TTI_SFPSHFT(-AVG_SHIFT_AMOUNT & AVG_SHIFT_MASK, p_sfpu::LREG0, p_sfpu::LREG0, 0b01);  // |x| >> 5 (divide by 32)
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);  // cc if the sum was negative
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 6);               // Restore sign (2's-complement negate)
        TTI_SFPENCC(0, 0, 0, 0);
    } else {
        // Unsigned formats (UInt32, and UInt16 in either Dest width): the sum is a magnitude over
        // the full 32 bits, so the quotient is a bare logical shift.
        TTI_SFPSHFT(-AVG_SHIFT_AMOUNT & AVG_SHIFT_MASK, p_sfpu::LREG0, p_sfpu::LREG0, 0b01);
    }
}

// Programmable float constant register holding 1/32 (0.03125) for float AVG. Preloaded once by
// init_reduce_sum_avg (only when pool_type == AVG and the format is float), so perform_float_average
// avoids rebuilding the constant via two SFPLOADI on every column group. Maps to LREG12 on Blackhole
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
}

// Dest addresses of the four column SUM/AVG groups.
constexpr std::uint32_t COL_SUM_UPPER_FACE_ADDRS[NUM_FACES] = {0, 0, 16, 16};    // Face 0, 0, 1, 1
constexpr std::uint32_t COL_SUM_LOWER_FACE_ADDRS[NUM_FACES] = {32, 32, 48, 48};  // Face 2, 2, 3, 3
constexpr std::uint32_t COL_SUM_COLUMN_OFFSETS[NUM_FACES] = {0, 2, 0, 2};        // even, odd, even, odd

/**
 * @brief Load one 4-row group of a face into LREG, masking the high bits of a UInt16 datum in a 32-bit dest.
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, std::uint32_t LREG, std::uint32_t ADDR>
inline void col_sum_load() {
    TTI_SFPLOAD(LREG, INSTRUCTION_MODE, ADDR_MOD_7, ADDR);
    if constexpr (clear_high_bits) {
        TTI_SFPAND(0, CLEAR_REG, LREG, 0);
    }
}

/**
 * @brief One add of the column half-reduce: DST = DST + SRC (SFPIADD for integer modes, SFPADD otherwise).
 */
template <bool is_integer_mode, std::uint32_t DST, std::uint32_t SRC>
inline void half_reduce_add() {
    if constexpr (is_integer_mode) {
        TTI_SFPIADD(0, SRC, DST, 4);
    } else {
        TTI_SFPADD(DST, p_sfpu::LCONST_1, SRC, DST, 0);
    }
}

/**
 * @brief One column group of perform_reduce_col_sum_avg. The lower face is already in LREG4-7; the upper face is
 *        loaded here and the next group's lower face between the dependent adds of the half tree.
 */
template <
    PoolType pool_type,
    InstrModLoadStore INSTRUCTION_MODE,
    bool clear_high_bits,
    bool pack_low16,
    bool is_signed_int,
    std::uint32_t GROUP>
inline void perform_reduce_col_sum_avg_group() {
    static_assert(GROUP < NUM_FACES, "four column groups per tile");
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::LO16);
    constexpr std::uint32_t UPPER = COL_SUM_UPPER_FACE_ADDRS[GROUP] + COL_SUM_COLUMN_OFFSETS[GROUP];
    constexpr bool HAS_NEXT = (GROUP + 1 < NUM_FACES);
    constexpr std::uint32_t NEXT_LOWER =
        HAS_NEXT ? COL_SUM_LOWER_FACE_ADDRS[GROUP + 1] + COL_SUM_COLUMN_OFFSETS[GROUP + 1] : 0;
    // Mode 9 (SFPSTORE_MOD0_FMT_LO16) only for a UInt16 output in a 32-bit dest: the packer reads the high 16 bits of
    // the word, so the low half is moved up; a 32-bit output takes the plain store.
    constexpr std::uint32_t STORE_MODE =
        pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);

    // Step 1: tree-reduce LREG0-3 -> LREG0 and LREG4-7 -> LREG4 (the replay); each position holds a 4-row partial sum.
    col_sum_load<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG0, UPPER>();
    col_sum_load<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG1, UPPER + ROWS_PER_LOAD>();
    col_sum_load<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG2, UPPER + 2 * ROWS_PER_LOAD>();
    col_sum_load<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG3, UPPER + 3 * ROWS_PER_LOAD>();
    lltt::replay(0, 6);

    // Step 2: cross-face addition; all four positions carry partial sums.
    if constexpr (is_integer_mode) {
        TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, 4);  // LREG0 = upper + lower (int)
    } else {
        TTI_SFPADD(
            p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);  // LREG0 = upper + lower (float)
    }

    // Step 3: transpose the four partial sums for the final reduction.
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 4: Final tree-reduce across LREG0-3 only (LREG4-7 no longer needed), issued inline so that the next
    // group's lower-face loads can sit between the dependent adds; LREG0[0] = total column sum.
    half_reduce_add<is_integer_mode, p_sfpu::LREG2, p_sfpu::LREG3>();
    if constexpr (HAS_NEXT) {
        col_sum_load<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG4, NEXT_LOWER>();
    }
    half_reduce_add<is_integer_mode, p_sfpu::LREG1, p_sfpu::LREG2>();
    if constexpr (HAS_NEXT) {
        col_sum_load<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG5, NEXT_LOWER + ROWS_PER_LOAD>();
    }
    half_reduce_add<is_integer_mode, p_sfpu::LREG0, p_sfpu::LREG1>();
    if constexpr (HAS_NEXT) {
        col_sum_load<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG6, NEXT_LOWER + 2 * ROWS_PER_LOAD>();
    }

    // Perform averaging if requested (different for int vs float)
    if constexpr (pool_type == PoolType::AVG) {
        if constexpr (is_integer_mode) {
            perform_int_average<INSTRUCTION_MODE, is_signed_int>();
        } else {
            perform_float_average();
        }
    }
    if constexpr (HAS_NEXT) {
        col_sum_load<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG7, NEXT_LOWER + 3 * ROWS_PER_LOAD>();
    }

    // Store the final column sum/average to the first row.
    TTI_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_7, UPPER);
}

template <
    PoolType pool_type,
    InstrModLoadStore INSTRUCTION_MODE,
    bool clear_high_bits,
    bool pack_low16,
    bool is_signed_int>
inline void perform_reduce_col_sum_avg() {
    // Reduce across registers, add the faces, transpose, half-reduce LREG0-3; the next group's lower-face loads sit
    // between the half tree's dependent adds (a multiply-add result read next stalls a cycle), every push a TTI_ immediate.
    constexpr std::uint32_t LOWER0 = COL_SUM_LOWER_FACE_ADDRS[0] + COL_SUM_COLUMN_OFFSETS[0];
    col_sum_load<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG4, LOWER0>();
    col_sum_load<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG5, LOWER0 + ROWS_PER_LOAD>();
    col_sum_load<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG6, LOWER0 + 2 * ROWS_PER_LOAD>();
    col_sum_load<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG7, LOWER0 + 3 * ROWS_PER_LOAD>();

    perform_reduce_col_sum_avg_group<pool_type, INSTRUCTION_MODE, clear_high_bits, pack_low16, is_signed_int, 0>();
    perform_reduce_col_sum_avg_group<pool_type, INSTRUCTION_MODE, clear_high_bits, pack_low16, is_signed_int, 1>();
    perform_reduce_col_sum_avg_group<pool_type, INSTRUCTION_MODE, clear_high_bits, pack_low16, is_signed_int, 2>();
    perform_reduce_col_sum_avg_group<pool_type, INSTRUCTION_MODE, clear_high_bits, pack_low16, is_signed_int, 3>();
}

// ============================================================================
// Horizontal (cross-column) reduction of the two 4-row accumulators
// ============================================================================
// The 8 SFPU column slices exchange data only through SFPSHFT2 (SUBVEC_SHFLROR1), which rotates VC right by
// one lane within each 8-lane sub-vector and writes VD; VD may differ from VC. horizontal_reduce (the float sum,
// whose pairing is kept) folds the per-column partials in LREG0 / LREG4 into the full 8-column result, replicated
// in every column, with a 3-stage butterfly (rotate by 4, 2, 1; fold after each stage) that uses LREG1 / LREG5 as
// rotate registers. horizontal_reduce_merged (integer sum, MAX, MIN) shares the last two stages between the two
// accumulators and leaves the result in column 0 only, the column every store and cross-tile pass reads.
//
// The two pairs are interleaved instruction by instruction. On Blackhole an SFPSHFT2 or SFPSWAP auto-stalls
// the next cycle (only SFPNOP issues), so this hides no latency, but it keeps every consumer two
// instructions behind its producer and the sequences need no explicit SFPNOPs.
//
// All horizontal reduces are issued inline; none is recorded. Replaying them gives the same TILE_LOOP (the
// row bodies are SFPU-bound, not issue-bound) while recording costs ~25-28 cycles per init, and ttnn re-inits
// before every single-tile sfpu_reduce. Inline also means the row path writes no replay slots, so the column
// windows below survive a row reduce under one shared init (sfpu_reduce_multidim_test.cpp REDUCE_ORDER).
//
// Replay slots (REPLAY_BUF_SIZE = 32 per thread; the FPU ops' windows start at ckernel::math::replay_buf_offset = 16):
//   init_reduce_sum_avg:                [0, 6)   tree-add window (column and row SUM/AVG; the column half tree is inline)
//   init_reduce_max_min:                [0, 11)  LOADMACRO column window (float, UInt32)
//   init_reduce_max_min_int32:          [0, 3)   manual 3-swap window (UInt16 in 32-bit Dest)
//   init_reduce_max_min_int32_signed:   [0, 15)  signed Int32 column window
//
// SFPSWAP direction (SFPCONFIG bit 8): the row paths and the manual column paths read opposite SFPSWAP
// operands, so the same pool type needs opposite directions. Each of them sets its own on entry
// (set_sfpswap_direction / set_manual_col_swap_direction). Only the LOADMACRO column path trusts its init;
// the row paths share that convention.

/**
 * @brief Rotate LREG1 and LREG5 right by one column, in place (one butterfly step for both pairs).
 */
inline void horizontal_reduce_rotate_step() {
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
}

/**
 * @brief LREG1 = ror<shift>(LREG0) and LREG5 = ror<shift>(LREG4), 2 * shift instructions.
 *        The first rotate of each pair reads the accumulator directly; the rest rotate in place.
 *
 * @tparam shift Butterfly stage distance: 4, 2 or 1 columns.
 */
template <std::uint32_t shift>
inline void horizontal_reduce_rotate() {
    static_assert(shift == 4 || shift == 2 || shift == 1, "butterfly stages rotate by 4, 2 or 1 columns");
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    if constexpr (shift >= 2) {
        horizontal_reduce_rotate_step();
    }
    if constexpr (shift >= 4) {
        horizontal_reduce_rotate_step();
        horizontal_reduce_rotate_step();
    }
}

/**
 * @brief Fold the rotated copies into the accumulators: LREG0 += LREG1, LREG4 += LREG5.
 *
 * @tparam is_integer_mode True for integer types (uses SFPIADD), false for float (uses SFPADD)
 */
template <bool is_integer_mode>
inline void horizontal_reduce_add() {
    if constexpr (is_integer_mode) {
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);
    } else {
        TTI_SFPADD(
            p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // lreg0 = lreg0 * 1.0 + lreg1 (float)
        TTI_SFPADD(
            p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);  // lreg4 = lreg4 * 1.0 + lreg5 (float)
    }
}


/**
 * @brief Horizontal SUM of the two accumulators: 20 instructions (14 SFPSHFT2 + 6 adds), inline, once per
 *        8-row group. Callers: the row SUM/AVG kernels and _softmax_k_ (experimental/ckernel_sfpu_softmax_k.h).
 *
 * In: LREG0 / LREG4 = per-column partials of the two 4-row groups. Out: LREG0 / LREG4 = the 8-column sum in
 * every column. Clobbers LREG1 / LREG5 only.
 *
 *   Stage 1: rotate by 4, add -> 4 duplicated pair sums.
 *   Stage 2: rotate by 2, add -> 2 duplicated quad sums.
 *   Stage 3: rotate by 1, add -> the full sum in every column.
 *
 * @tparam is_integer_mode True for integer types (SFPIADD), false for float (SFPADD).
 */
template <bool is_integer_mode>
inline void horizontal_reduce() {
    horizontal_reduce_rotate<4 /*shift*/>();
    horizontal_reduce_add<is_integer_mode>();
    horizontal_reduce_rotate<2 /*shift*/>();
    horizontal_reduce_add<is_integer_mode>();
    horizontal_reduce_rotate<1 /*shift*/>();
    horizontal_reduce_add<is_integer_mode>();
}

/**
 * @brief Moves the odd lanes of SRC into DST. LReg[15] holds twice the lane index, so its bit 1 shifted to the sign
 *        marks the odd lanes of every 8-lane row. Clobbers TMP; leaves every lane enabled.
 */
template <std::uint32_t SRC, std::uint32_t DST, std::uint32_t TMP>
inline void merge_odd_lanes() {
    TTI_SFPSHFT(30, p_sfpu::LTILEID, TMP, 5 /* ARG_IMM | ARG_IMM_USE_VC: TMP = LReg[15] << 30 */);
    TTI_SFPSETCC(0, TMP, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPMOV(0, SRC, DST, 0);
    TTI_SFPENCC(0, 0, 0, 0);
}

struct RowFoldIntAdd {
    template <std::uint32_t ACC, std::uint32_t SRC>
    static inline void apply() {
        TTI_SFPIADD(0, SRC, ACC, 4);
    }
};

// SFPSWAP leaves the extreme selected by SFPCONFIG bit 8 (MAX by default, MIN when set) in ACC.
struct RowFoldSwap {
    template <std::uint32_t ACC, std::uint32_t SRC>
    static inline void apply() {
        TTI_SFPSWAP(0, ACC, SRC, 1);
    }
};

/**
 * @brief Horizontal reduce of the two accumulators of an 8-row group with one shared butterfly: 9 SFPSHFT2, 4 folds.
 *
 * After the first rotate and fold, the odd lanes of an accumulator hold the pairs (0,1), (2,3), (4,5), (6,7) and the
 * even lanes the pairs (7,0), (1,2), (3,4), (5,6). LREG4 takes the odd lanes of LREG0, and the rotate-by-2 and
 * rotate-by-4 stages run once for both: LREG4 ends with group B's total in its even lanes and group A's in its odd
 * lanes, and a last rotate brings A's total to lane 0 of LREG0. Only column 0 of each row holds the result.
 *
 * In: LREG0 / LREG4 = per-column partials of the two 4-row groups. Out: column 0 of LREG0 / LREG4. Clobbers
 * LREG1, LREG2, LREG5. The fold must be associative and commutative (integer add, extreme), so the result equals
 * horizontal_reduce's whatever the pairing; the float sum keeps horizontal_reduce.
 */
template <typename Fold>
inline void horizontal_reduce_merged() {
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    Fold::template apply<p_sfpu::LREG0, p_sfpu::LREG1>();
    Fold::template apply<p_sfpu::LREG4, p_sfpu::LREG5>();
    merge_odd_lanes<p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LREG2>();

    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    Fold::template apply<p_sfpu::LREG4, p_sfpu::LREG5>();

    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    Fold::template apply<p_sfpu::LREG4, p_sfpu::LREG5>();

    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
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
 * 4. Use horizontal_reduce_merged to fold the 8 SFPU columns; column 0 then holds the row max
 * 5. Store the per-row max, reading column 0
 *
 * On the LOADMACRO path the four compare-and-swaps with a freshly loaded register run inside SFPLOADMACRO sequences
 * 0 to 3 (SFPLOADMACRO loads LREG0-3 only); a scheduled SFPSWAP holds the simple sub-unit for two cycles.
 *
 * @tparam INSTRUCTION_MODE Load/store instruction mode (FP32, FP16B, or INT32 for sign-magnitude int max)
 * @param tile_row_offset Base row offset for this tile in the dest register
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits>
inline void perform_reduce_row_max_tile(std::uint32_t tile_row_offset, std::uint32_t result_store_mode) {
    // The fused path cannot mask the high bits of a UInt16 operand between its load and its compare.
#ifdef DISABLE_SFPLOADMACRO
    constexpr bool fused_vertical_swap = false;
#else
    constexpr bool fused_vertical_swap = !clear_high_bits;
#endif

#pragma GCC unroll 2
    for (std::uint32_t face_pair = 0; face_pair < 2; face_pair++) {
        std::uint32_t face_pair_base = face_pair * 2 * ROWS_PER_FACE;

#pragma GCC unroll 2
        for (std::uint32_t row_group = 0; row_group < 2; row_group++) {
            std::uint32_t row_offset_first = row_group * 8;
            std::uint32_t row_offset_second = row_offset_first + 4;

            if constexpr (fused_vertical_swap) {
                const std::uint32_t first = tile_row_offset + face_pair_base + row_offset_first;
                const std::uint32_t second = tile_row_offset + face_pair_base + row_offset_second;

                // Rows r..r+3: sequence 2 leaves the extreme of LREG0 and LREG2 in LREG0, sequence 3 that of LREG1
                // and LREG3 in LREG1.
                TT_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, first);
                TT_SFPLOADMACRO((2 << 2) | p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, first + ROWS_PER_FACE);
                TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, first + 2);
                TT_SFPLOADMACRO((3 << 2) | p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, first + ROWS_PER_FACE + 2);

                // Rows r+4..r+7: sequences 0 and 1 with the accumulators LREG4 and LREG5.
                TT_SFPLOAD(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, second);
                TT_SFPLOADMACRO((0 << 2) | p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, second + ROWS_PER_FACE);
                TT_SFPLOAD(p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, second + 2);
                TT_SFPLOADMACRO((1 << 2) | p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, second + ROWS_PER_FACE + 2);

                // The last scheduled swap runs on the two cycles after its load.
                TTI_SFPNOP;
                TTI_SFPNOP;

                // Combine the even and odd column extremes of each 4-row group.
                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
                TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);
            } else {
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_first);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG1,
                    INSTRUCTION_MODE,
                    ADDR_MOD_7,
                    tile_row_offset + face_pair_base + row_offset_first + 2);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG2,
                    INSTRUCTION_MODE,
                    ADDR_MOD_7,
                    tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG3,
                    INSTRUCTION_MODE,
                    ADDR_MOD_7,
                    tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first + 2);

                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_second);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG5,
                    INSTRUCTION_MODE,
                    ADDR_MOD_7,
                    tile_row_offset + face_pair_base + row_offset_second + 2);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG6,
                    INSTRUCTION_MODE,
                    ADDR_MOD_7,
                    tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second);
                load_and_clear_high_bits<clear_high_bits>(
                    p_sfpu::LREG7,
                    INSTRUCTION_MODE,
                    ADDR_MOD_7,
                    tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second + 2);

                // Vertical max: reduce left/right face pairs via compare-and-swap.
                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, 1);
                TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, 1);
                TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, 1);
                TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, 1);
                TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
                TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);
            }

            horizontal_reduce_merged<RowFoldSwap>();

            // result_store_mode is mode 9 (SFPSTORE_MOD0_FMT_LO16) only when this per-tile store is the
            // final, packer-visible result (single column tile); otherwise it is intermediate and stays
            // in the low 16 bits.
            TT_SFPSTORE(
                p_sfpu::LREG0, result_store_mode, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_first);
            TT_SFPSTORE(
                p_sfpu::LREG4, result_store_mode, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_second);
        }
    }
}

/**
 * @brief Row-wise maximum reduction for a single 32x32 tile using Int32 values on Blackhole.
 *
 * Int32 operands reach DEST as two's-complement: that is how ttnn feeds the device, and a multi-axis
 * reduce (e.g. ttir.max dim=[1,2]) chains a column reduce then this row reduce over the same DEST, with
 * the column path (calculate_reduce_max_min_int32_col) leaving its result in two's-complement. SFPSWAP
 * compares in sign-magnitude and on Blackhole INT32_2S_COMP load/store is a no-op (it does not convert),
 * so we cast each loaded operand two's-complement -> sign-magnitude before the compare-and-swap reduce.
 *
 * The surviving maxima are cast back to two's-complement only when @p final_store is set — i.e. the
 * single-column-tile case where this per-tile store is the packer-visible result and must match ttnn /
 * the column path. With block_ct_dim > 1 the store is an intermediate that max_first_columns_across_tiles_int32
 * re-reads and keeps comparing in sign-magnitude, so we leave it in sign-magnitude and skip the
 * round-trip cast (the cross-tile step applies the single two's-complement cast on the final result).
 *
 * Register budget: the cast needs a free GPR scratch, but the 8 loaded operands leave none. As in the
 * column / sum Int32 paths we process the two independent 4-row groups in sequence: load+cast+reduce
 * group A into LREG0 (freeing LREG1-3), then use those freed registers as scratch for group B into LREG4.
 *
 * @param tile_row_offset Base row offset for this tile in the dest register
 * @param final_store     Whether this per-tile store is the final, packer-visible result (block_ct_dim == 1).
 */
template <bool clear_high_bits = false>
inline void perform_reduce_row_max_int32_tile(
    std::uint32_t tile_row_offset, std::uint32_t result_store_mode, bool final_store) {
    constexpr InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32;  // raw load/store; cast is explicit

    for (std::uint32_t face_pair = 0; face_pair < 2; face_pair++) {
        std::uint32_t face_pair_base = face_pair * 2 * ROWS_PER_FACE;

        for (std::uint32_t row_group = 0; row_group < 2; row_group++) {
            std::uint32_t row_offset_first = row_group * 8;
            std::uint32_t row_offset_second = row_offset_first + 4;

            const std::uint32_t group_a_base = tile_row_offset + face_pair_base + row_offset_first;
            const std::uint32_t group_b_base = tile_row_offset + face_pair_base + row_offset_second;

            // Group A (first 4 rows) -> LREG0-3, scratch from the still-free LREG4-7. Cast two's-complement
            // -> sign-magnitude, then reduce the left/right face columns into LREG0 via compare-and-swap.
            TT_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, group_a_base);
            TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, group_a_base + 2);
            TT_SFPLOAD(p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, group_a_base + ROWS_PER_FACE);
            TT_SFPLOAD(p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, group_a_base + ROWS_PER_FACE + 2);
            convert_int_representation_inplace(p_sfpu::LREG0, p_sfpu::LREG4);
            convert_int_representation_inplace(p_sfpu::LREG1, p_sfpu::LREG5);
            convert_int_representation_inplace(p_sfpu::LREG2, p_sfpu::LREG6);
            convert_int_representation_inplace(p_sfpu::LREG3, p_sfpu::LREG7);
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);  // group A max in LREG0; LREG1-3 now free

            // Group B (next 4 rows) -> LREG4-7, scratch from the now-free LREG1-3.
            TT_SFPLOAD(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, group_b_base);
            TT_SFPLOAD(p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, group_b_base + 2);
            TT_SFPLOAD(p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, group_b_base + ROWS_PER_FACE);
            TT_SFPLOAD(p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, group_b_base + ROWS_PER_FACE + 2);
            convert_int_representation_inplace(p_sfpu::LREG4, p_sfpu::LREG1);
            convert_int_representation_inplace(p_sfpu::LREG5, p_sfpu::LREG2);
            convert_int_representation_inplace(p_sfpu::LREG6, p_sfpu::LREG3);
            convert_int_representation_inplace(p_sfpu::LREG7, p_sfpu::LREG1);
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);  // group B max in LREG4

            // Consolidate the 8 SFPU columns into column 0 (operates on LREG0/LREG1 and LREG4/LREG5).
            horizontal_reduce_merged<RowFoldSwap>();

            // Cast the sign-magnitude winners back to two's-complement only for the final, packer-visible
            // store (single column tile). Intermediate stores (block_ct_dim > 1) stay in sign-magnitude;
            // the cross-tile step re-reads them as-is and applies the single cast on the final result.
            // Only LREG0 and LREG4 hold results, so LREG1/LREG5 are free GPR scratch.
            if (final_store) {
                convert_int_representation_inplace(p_sfpu::LREG0, p_sfpu::LREG1);
                convert_int_representation_inplace(p_sfpu::LREG4, p_sfpu::LREG5);
            }

            TT_SFPSTORE(p_sfpu::LREG0, result_store_mode, ADDR_MOD_7, group_a_base);
            TT_SFPSTORE(p_sfpu::LREG4, result_store_mode, ADDR_MOD_7, group_b_base);
        }
    }
}

/**
 * @brief Accumulates partial row maxima from all tiles in a row of tiles into tile 0.
 *
 * Mirrors sum_first_columns_across_tiles but uses SFPSWAP instead of SFPADD.
 *
 * @tparam INSTRUCTION_MODE Load/store instruction mode (FP32, FP16B, or INT32 for sign-magnitude int max)
 * @param tile_row_base Base address of the first tile in this row of tiles
 * @param block_ct_dim Number of tiles along x axis of tensor (column tiles)
 */
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16>
inline void max_first_columns_across_tiles(std::uint32_t tile_row_base, std::uint32_t block_ct_dim) {
    constexpr std::uint32_t RESULT_ROWS[8] = {0, 4, 8, 12, 32, 36, 40, 44};

    for (std::uint32_t batch = 0; batch < 2; batch++) {
        std::uint32_t base_idx = batch * 4;

        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 0]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 1]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 2]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 3]);

        for (std::uint32_t t = 1; t < block_ct_dim; t++) {
            std::uint32_t tile_offset = tile_row_base + t * ROWS_PER_TILE;

            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 0]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 1]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 2]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 3]);

            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG4, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG5, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG7, 1);
        }

        // Final, packer-visible store: use mode 9 (SFPSTORE_MOD0_FMT_LO16) only when the OUTPUT is
        // UInt16 in a 32-bit dest (pack_low16); a 32-bit output keeps the plain INSTRUCTION_MODE store.
        constexpr std::uint32_t STORE_MODE =
            pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
        TT_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 0]);
        TT_SFPSTORE(p_sfpu::LREG1, STORE_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 1]);
        TT_SFPSTORE(p_sfpu::LREG2, STORE_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 2]);
        TT_SFPSTORE(p_sfpu::LREG3, STORE_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 3]);
    }
}

/**
 * @brief Accumulates partial row maxima from all tiles in a row of tiles into tile 0 (Int32).
 *
 * The per-tile row maxima written by perform_reduce_row_max_int32_tile are already in sign-magnitude
 * (the representation SFPSWAP compares in) because that path skips the round-trip cast for intermediate
 * stores. So we load and compare them directly — no per-operand cast — and apply a single
 * sign-magnitude -> two's-complement cast on the surviving maxima before the final, packer-visible store
 * (matching ttnn and the column path). This mirrors the float max_first_columns_across_tiles' pipelined
 * load-4 / swap-4 shape, with one trailing cast block added.
 *
 * @param tile_row_base Base address of the first tile in this row of tiles
 * @param block_ct_dim Number of tiles along x axis of tensor (column tiles)
 */
template <bool clear_high_bits = false, bool pack_low16 = false>
inline void max_first_columns_across_tiles_int32(std::uint32_t tile_row_base, std::uint32_t block_ct_dim) {
    constexpr InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32;  // raw load/store; cast is explicit
    constexpr std::uint32_t RESULT_ROWS[8] = {0, 4, 8, 12, 32, 36, 40, 44};

    for (std::uint32_t batch = 0; batch < 2; batch++) {
        std::uint32_t base_idx = batch * 4;

        // Tile 0's intermediates are already sign-magnitude, so load them straight into the accumulators.
        load_and_clear_high_bits<false>(
            p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 0]);
        load_and_clear_high_bits<false>(
            p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 1]);
        load_and_clear_high_bits<false>(
            p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 2]);
        load_and_clear_high_bits<false>(
            p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 3]);

        for (std::uint32_t t = 1; t < block_ct_dim; t++) {
            std::uint32_t tile_offset = tile_row_base + t * ROWS_PER_TILE;

            // Accumulators and operands are all sign-magnitude, so load the four operands into LREG4-7 and
            // compare-and-swap with no cast — the same pipelined shape as the float cross-tile path.
            load_and_clear_high_bits<false>(
                p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 0]);
            load_and_clear_high_bits<false>(
                p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 1]);
            load_and_clear_high_bits<false>(
                p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 2]);
            load_and_clear_high_bits<false>(
                p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 3]);
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG4, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG5, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG7, 1);
        }

        // Cast the surviving sign-magnitude maxima back to two's-complement before the store. LREG4-7 are
        // free after the accumulation loop.
        convert_int_representation_inplace(p_sfpu::LREG0, p_sfpu::LREG4);
        convert_int_representation_inplace(p_sfpu::LREG1, p_sfpu::LREG5);
        convert_int_representation_inplace(p_sfpu::LREG2, p_sfpu::LREG6);
        convert_int_representation_inplace(p_sfpu::LREG3, p_sfpu::LREG7);

        TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 0]);
        TT_SFPSTORE(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 1]);
        TT_SFPSTORE(p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 2]);
        TT_SFPSTORE(p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 3]);
    }
}

/**
 * @brief Sets the SFPSWAP(VEC_MIN_MAX) direction in one instruction. SFPCONFIG with instr_mod1 = 1 writes
 *        LaneConfig from the immediate (the form _init_sfpu_config_reg() uses), so it also resets the rest of
 *        the register and clobbers no LREG. Bit 8 clear: VC gets the maximum, VD the minimum. Bit 8 set inverts.
 *
 * The row MAX/MIN paths and the LOADMACRO column path read VC: MAX is the default, MIN inverts. The manual
 * column paths read VD: MAX inverts (see set_manual_col_swap_direction).
 *
 * @tparam invert True to set SFPCONFIG bit 8.
 */
template <bool invert>
inline void set_sfpswap_direction() {
    TTI_SFPCONFIG(invert ? 0x0100 : 0x0000, 0xF, 1);
}

/**
 * @brief SFPSWAP direction for the manual column MAX/MIN paths, which read the LOW (VD) operand of each
 *        compare-and-swap: MAX inverts (bit 8 set), MIN is the default, the opposite of the row paths.
 *
 * Called on entry by calculate_reduce_max_min_uint16 (UInt16 in a 32-bit Dest, and every non-Int32 format
 * under DISABLE_SFPLOADMACRO) and calculate_reduce_max_min_int32_col, instead of trusting the init: under one
 * shared init a preceding row MAX/MIN leaves the row convention in place.
 *
 * @tparam pool_type MAX or MIN.
 */
template <PoolType pool_type>
inline void set_manual_col_swap_direction() {
    set_sfpswap_direction<pool_type == PoolType::MAX>();
}

/**
 * @brief Row-wise maximum/minimum reduction across a block of tiles.
 *
 * For each row of tiles, reduces every tile individually, then (if block_ct_dim > 1)
 * accumulates the per-tile column-0 extrema across tiles using compare-and-swap into
 * tile 0's column 0.
 *
 * MAX and MIN share the entire compare-and-swap machinery (per-tile reduce, horizontal_reduce_merged,
 * max_first_columns_across_tiles, and the Int32 two's-complement<->sign-magnitude casts); the only
 * difference is the SFPSWAP comparator direction, set once here via SFPCONFIG bit 8 (0 = MAX,
 * 1 = MIN). The representation seen by SFPSWAP (float, or sign-magnitude for Int32 after the explicit
 * cast) is identical for MAX and MIN, so flipping bit 8 cleanly inverts the result for every format.
 *
 * @tparam pool_type MAX or MIN.
 * @tparam INSTRUCTION_MODE Load/store instruction mode (FP32, FP16B, or INT32 for sign-magnitude int extrema)
 * @param block_ct_dim Number of tiles along x axis of tensor (column tiles)
 * @param block_rt_dim Number of tiles along y axis of tensor (row tiles)
 */
template <PoolType pool_type, InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16>
inline void perform_reduce_row_max_min(std::uint32_t block_ct_dim, std::uint32_t block_rt_dim) {
    static_assert(
        pool_type == PoolType::MAX || pool_type == PoolType::MIN,
        "perform_reduce_row_max_min only supports MAX and MIN pool types");

    constexpr bool is_int32 = (INSTRUCTION_MODE == InstrModLoadStore::INT32);

    // Set the SFPSWAP direction on entry (MAX default, MIN inverts) rather than trusting the init: under one
    // shared init (e.g. ttir.max dim=[1,2]) a preceding manual column MAX/MIN leaves the opposite convention.
    set_sfpswap_direction<pool_type == PoolType::MIN>();

    // The horizontal reduce is inline and nothing is recorded here, so the column LOADMACRO window [0, 11)
    // survives a row reduce under one shared init.

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
            if constexpr (is_int32) {
                // Single column tile => this per-tile store is the final, packer-visible result and must
                // be cast back to two's-complement; otherwise it is an intermediate kept in sign-magnitude.
                perform_reduce_row_max_int32_tile(tile_offset, tile_store_mode, /*final_store=*/block_ct_dim == 1);
            } else {
                perform_reduce_row_max_tile<INSTRUCTION_MODE, clear_high_bits>(tile_offset, tile_store_mode);
            }
        }

        if (block_ct_dim > 1) {
            if constexpr (is_int32) {
                max_first_columns_across_tiles_int32<clear_high_bits, pack_low16>(tile_row_offset, block_ct_dim);
            } else {
                max_first_columns_across_tiles<INSTRUCTION_MODE, clear_high_bits, pack_low16>(
                    tile_row_offset, block_ct_dim);
            }
        }
    }
}

/**
 * @brief Reciprocal divisor for a row average, carried as two SFPLOADI halves.
 *
 * A row reduction collapses every column of a row of tiles, so the average divides the row sum by
 * num_cols = 32 * block_ct_dim columns. The column path always divides by the fixed 32 rows of a
 * tile and can use the compile-time 1/32 constant the init preloads into the programmable float
 * const register; the row divisor instead depends on the runtime block_ct_dim. A runtime value
 * cannot be written into that programmable const register with a plain SFPLOADI (it requires a
 * config write), so the row path keeps the reciprocal in an ordinary working LREG and multiplies
 * with it directly. This struct carries the precomputed 16-bit halves so each divide site can
 * reload the reciprocal into a scratch register.
 */
struct RowAvgReciprocal {
    std::uint16_t high16 = 0;
    std::uint16_t low16 = 0;
};

inline RowAvgReciprocal make_row_avg_reciprocal(std::uint32_t num_cols) {
    const FloatBits bits(1.0f / static_cast<float>(num_cols));
    return RowAvgReciprocal{bits.high16, bits.low16};
}

/**
 * @brief Load the row-average reciprocal into @p scratch_lreg (an SFPLOADI pair).
 */
inline void load_row_avg_reciprocal_into(std::uint32_t scratch_lreg, RowAvgReciprocal recip) {
    TT_SFPLOADI(scratch_lreg, sfpi::SFPLOADI_MOD0_UPPER, recip.high16);
    TT_SFPLOADI(scratch_lreg, sfpi::SFPLOADI_MOD0_LOWER, recip.low16);
}

template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool is_avg = false>
inline void perform_reduce_row_sum_tile(
    std::uint32_t tile_row_offset,
    std::uint32_t result_store_mode,
    bool divide_now = false,
    RowAvgReciprocal recip = {}) {
    // Determine if integer or float mode at compile time
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Process tile in 2 face-pairs: (f0+f1) for tile rows 0-15, (f2+f3) for tile rows 16-31
    // Each face-pair iteration processes 8 rows (two groups of 4 rows each)
    for (std::uint32_t face_pair = 0; face_pair < 2; face_pair++) {
        // Base offset for this face pair:
        // face_pair 0: faces 0+1 (dest indices 0-31)
        // face_pair 1: faces 2+3 (dest indices 32-63)
        std::uint32_t face_pair_base = face_pair * 2 * ROWS_PER_FACE;

        for (std::uint32_t row_group = 0; row_group < 2; row_group++) {
            // Within each face, process rows in groups of 8 (two sub-groups of 4)
            std::uint32_t row_offset_first = row_group * 8;          // 0 or 8
            std::uint32_t row_offset_second = row_offset_first + 4;  // 4 or 12

            const std::uint32_t group_a_base = tile_row_offset + face_pair_base + row_offset_first;
            const std::uint32_t group_b_base = tile_row_offset + face_pair_base + row_offset_second;

            // Load 4 rows from face 0 (or 2) and face 1 (or 3)
            load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, group_a_base);
            load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, group_a_base + 2);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, group_a_base + ROWS_PER_FACE);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, group_a_base + ROWS_PER_FACE + 2);

            // Load next 4 rows from face 0 (or 2) and face 1 (or 3)
            load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, group_b_base);
            load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, group_b_base + 2);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, group_b_base + ROWS_PER_FACE);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, group_b_base + ROWS_PER_FACE + 2);

            // Perform vertical sum of loaded rows via replay buffer
            // After this: LREG0 contains sum of first 4 rows, LREG4 contains sum of next 4 rows
            lltt::replay(0, 6);

            // Horizontal reduction, inline (see the horizontal-reduce section): every column of LREG0 / LREG4
            // then holds its 4-row group's full row sum.
            if constexpr (is_integer_mode) {
                horizontal_reduce_merged<RowFoldIntAdd>();
            } else {
                horizontal_reduce<is_integer_mode>();
            }

            // For a single-column-tile AVG the per-tile sum is already the full row sum, so divide it
            // here (float-only path; row AVG is restricted to float formats). When block_ct_dim > 1 the
            // per-tile store is an intermediate partial sum and the division is deferred to
            // sum_first_columns_across_tiles instead. The two results (first/second 4-row group) live in
            // LREG0/LREG4; LREG2 is free after the horizontal reduce and holds the reciprocal.
            if constexpr (is_avg) {
                if (divide_now) {
                    load_row_avg_reciprocal_into(p_sfpu::LREG2, recip);
                    // The two SFPMULs cover each other's 2-cycle latency: the LREG4 multiply sits
                    // between the LREG0 multiply and the LREG0 store, and the first store sits
                    // between the LREG4 multiply and the LREG4 store, so no extra NOP is needed.
                    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
                    TTI_SFPMUL(p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
                }
            }

            // result_store_mode is mode 9 (SFPSTORE_MOD0_FMT_LO16) only when this per-tile store is the
            // final, packer-visible result (single column tile); otherwise it is an intermediate store
            // re-loaded by the cross-tile accumulation and must stay in the low 16 bits.
            TT_SFPSTORE(
                p_sfpu::LREG0, result_store_mode, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_first);
            TT_SFPSTORE(
                p_sfpu::LREG4, result_store_mode, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_second);
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
template <InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16, bool is_avg = false>
inline void sum_first_columns_across_tiles(
    std::uint32_t tile_row_base, std::uint32_t block_ct_dim, RowAvgReciprocal recip = {}) {
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Row offset for each of the 8 partial row-sum results (face 0: 0,4,8,12; face 2: 32,36,40,44)
    constexpr std::uint32_t RESULT_ROWS[8] = {0, 4, 8, 12, 32, 36, 40, 44};

    for (std::uint32_t batch = 0; batch < 2; batch++) {
        std::uint32_t base_idx = batch * 4;

        // Load tile 0's four LREGs at this batch's offsets (0,4,8,12 or 32,36,40,44) into LREG0-3
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 0]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 1]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 2]);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 3]);

        // Accumulate from remaining tiles
        for (std::uint32_t t = 1; t < block_ct_dim; t++) {
            std::uint32_t tile_offset = tile_row_base + t * ROWS_PER_TILE;

            // Load tile t's four LREGs at the same offsets into LREG4-7
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 0]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 1]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 2]);
            load_and_clear_high_bits<clear_high_bits>(
                p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 3]);

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

        // LREG0-3 now hold the full per-row sum across all column tiles. For AVG (float-only path)
        // this is the point to divide by num_cols, before the final packer-visible store. LREG4 is
        // free after the accumulation loop and holds the reciprocal for the four multiplies.
        if constexpr (is_avg) {
            load_row_avg_reciprocal_into(p_sfpu::LREG4, recip);
            // The four SFPMULs cover each other's 2-cycle latency: at least two multiplies (and the
            // earlier stores) separate every multiply from the store that reads its result, so no
            // extra NOP is needed before the stores below.
            TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
            TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
            TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
            TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
        }

        // Store LREG0-3 back to tile 0. This is the final, packer-visible result, so it uses mode 9
        // (SFPSTORE_MOD0_FMT_LO16) only when the OUTPUT is UInt16 in a 32-bit dest (packer reads the
        // high 16 bits); a 32-bit output (e.g. UInt32) is stored with the plain INSTRUCTION_MODE.
        constexpr std::uint32_t STORE_MODE =
            pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
        TT_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 0]);
        TT_SFPSTORE(p_sfpu::LREG1, STORE_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 1]);
        TT_SFPSTORE(p_sfpu::LREG2, STORE_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 2]);
        TT_SFPSTORE(p_sfpu::LREG3, STORE_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 3]);
    }
}

template <PoolType pool_type, InstrModLoadStore INSTRUCTION_MODE, bool clear_high_bits, bool pack_low16>
inline void perform_reduce_row_sum_avg(std::uint32_t block_ct_dim, std::uint32_t block_rt_dim) {
    static_assert(
        pool_type == PoolType::SUM || pool_type == PoolType::AVG,
        "perform_reduce_row_sum_avg only supports SUM and AVG pool types");

    // AVG divides each row sum by the number of reduced columns (32 per tile * block_ct_dim tiles).
    // The reciprocal is computed once and passed down to whichever step performs the final store.
    constexpr bool is_avg = (pool_type == PoolType::AVG);
    const RowAvgReciprocal recip = is_avg ? make_row_avg_reciprocal(32u * block_ct_dim) : RowAvgReciprocal{};

    // When there is a single column tile, the per-tile store is the final packer-visible result and must
    // use mode 9 only when the OUTPUT is UInt16 in a 32-bit dest (pack_low16). With multiple column tiles
    // the per-tile store is intermediate (re-loaded by sum_first_columns_across_tiles) and must stay in the
    // low 16 bits via INSTRUCTION_MODE; the final mode-9 (if any) is applied by the cross-tile store.
    const std::uint32_t tile_store_mode = (pack_low16 && block_ct_dim == 1)
                                              ? 9u /* SFPSTORE_MOD0_FMT_LO16 */
                                              : static_cast<std::uint32_t>(INSTRUCTION_MODE);

    // For AVG, the divide happens at the point the full row sum is known: in the per-tile reducer for a
    // single column tile, or in the cross-tile accumulation step for multiple column tiles.
    const bool divide_in_tile = is_avg && (block_ct_dim == 1);

    for (std::uint32_t i = 0; i < block_rt_dim; i++) {
        std::uint32_t tile_row_offset = ROWS_PER_TILE * block_ct_dim * i;

        // Step 1: Reduce each tile individually (horizontal reduction within each tile)
        for (std::uint32_t j = 0; j < block_ct_dim; j++) {
            std::uint32_t tile_offset = tile_row_offset + (ROWS_PER_TILE * j);
            perform_reduce_row_sum_tile<INSTRUCTION_MODE, clear_high_bits, is_avg>(
                tile_offset, tile_store_mode, divide_in_tile, recip);
        }

        // Step 2: Sum column 0 from all tiles in this row into tile 0's column 0
        if (block_ct_dim > 1) {
            sum_first_columns_across_tiles<INSTRUCTION_MODE, clear_high_bits, pack_low16, is_avg>(
                tile_row_offset, block_ct_dim, recip);
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
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);

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
 * @brief Initialization for SFPU reduce MAX/MIN kernel on 32x32 tile for Int32 format. Due to RTL bug INT32_2S_COMP
 * LOAD/STORE has no effect. Must cast to INT_SIGN_MAGN_TO_INT32_2S_COMP before swapping. Since CAST and SWAP are both
 * SIMPLE instructions, cannot be integrated together in LOADMACRO sequence. Therefore, we need to initialize the kernel
 * with manual loads and stores in order to perform the CAST and SWAP operations.
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, LO16, DEFAULT
 * (FP32, FP16B)
 * @tparam pool_type MAX or MIN; unused here. The direction is set on entry by the calculates
 *         (calculate_reduce_max_min_uint16, the row paths) and read by the recorded swaps at replay time.
 */
template <InstrModLoadStore INSTRUCTION_MODE, PoolType pool_type>
inline void init_reduce_max_min_int32() {
    _init_sfpu_config_reg();

    lltt::record(0, 3);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, 1);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG5, 1);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, 1);
}

/**
 * @brief Initialization for SFPU reduce MAX/MIN kernel.
 *        Sets up LOADMACRO sequences for compare-and-swap operations, configures address modifiers,
 *        and records replay buffers for efficient column-wise maximum/minimum reduction.
 *
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, LO16, DEFAULT
 * (FP32, FP16B)
 * @tparam pool_type The pool type (MAX or MIN) to determine swap direction
 * @param num_cols The number of columns to process (typically 32 for a single tile, or multiple of 32 for block
 * operations)
 */
template <InstrModLoadStore INSTRUCTION_MODE, PoolType pool_type, bool clear_high_bits>
inline void init_reduce_max_min([[maybe_unused]] std::uint32_t num_cols) {
#ifdef DISABLE_SFPLOADMACRO
    init_reduce_max_min_int32<INSTRUCTION_MODE, pool_type>();
    return;
#endif

    // Reset the SFPU config and set the swap direction (MIN inverts). calculate_reduce_max_min trusts it; that
    // is safe under one shared init because the row paths use the same convention and write no replay slot.
    set_sfpswap_direction<pool_type == PoolType::MIN>();

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

    // Setup LOADMACRO sequences 2 and 3 for the row MAX/MIN kernel: the fused load-and-compare of sequences 0 and 1
    // with accumulators LREG0 and LREG1 (0x80: the loaded register is the swap's dest; 0x06 / 0x07: template 2 / 3).
    TTI_SFPSWAP(0, p_sfpu::LREG0, 0xE /* instruction template 2 */, 1);
    TTI_SFPLOADI(0, 0xA, 0x0286);
    TTI_SFPLOADI(0, 0x8, 0x0000);
    TTI_SFPCONFIG(0, 6, 0);

    TTI_SFPSWAP(0, p_sfpu::LREG1, 0xF /* instruction template 3 */, 1);
    TTI_SFPLOADI(0, 0xA, 0x0287);
    TTI_SFPLOADI(0, 0x8, 0x0000);
    TTI_SFPCONFIG(0, 7, 0);

    configure_addrmod_max_min(num_cols);

    // Record replay buffer for compare-and-swap operations.
    // Note: this LOADMACRO-based path is only used for float/UInt32 formats. UInt16 in 32-bit dest
    // cannot use it because the fused load+swap leaves no place to mask the garbage high bits, so it
    // is routed to the manual calculate_reduce_max_min_uint16() path instead.
    constexpr std::uint32_t buffer_len = 11;
    lltt::record<lltt::NoExec>(0, buffer_len);
    TTI_INCRWC(0, 4, 0, 0);
    TTI_SFPLOADMACRO(5, INSTRUCTION_MODE, ADDR_MOD_7, 2);
    TTI_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, 16);
    TTI_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, 18);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG1, 1);
    TTI_SFPNOP;
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG0, 1);
    TTI_SFPNOP;
    TTI_SFPLOADMACRO(0, INSTRUCTION_MODE, ADDR_MOD_7, 0);

    // Dummy loads to increment dest counters
    TTI_SFPLOAD(8, INSTRUCTION_MODE, ADDR_MOD_6, 0);
    TTI_SFPLOAD(8, INSTRUCTION_MODE, ADDR_MOD_5, 0);
}

/**
 * @brief Initialization for SFPU reduce SUM and AVG kernels.
 *        Records the replay buffer for the vertical tree reduction:
 *        - Positions 0-5: Full tree reduce for both LREG groups (used by both col and row reduce)
 *        The column kernel's half tree reduce is issued inline by perform_reduce_col_sum_avg.
 *
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, LO16, DEFAULT
 * (FP32, FP16B)
 */
template <InstrModLoadStore INSTRUCTION_MODE, PoolType pool_type>
inline void init_reduce_sum_avg() {
    _init_sfpu_config_reg();

    // Determine if integer or float mode based on INSTRUCTION_MODE
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Float AVG divides by 32 by multiplying by 1/32. Preload that constant once here (only when it is
    // actually needed: float AVG) into the programmable float const register AVG_RECIP_REG so
    // perform_float_average() collapses to a single SFPMUL instead of rebuilding the constant with two
    // SFPLOADI on every column group. Integer AVG uses perform_int_average (shift) and pays nothing; SUM
    // never averages, so it pays nothing either.
    if constexpr (pool_type == PoolType::AVG && !is_integer_mode) {
        sfpi::vConstFloatPrgm0 = 0.03125f;
    }

    // Record one replay buffer:
    // Positions 0-5: Full tree reduce (both LREG groups, interleaved for latency hiding)
    //   - Used by column reduce (first pass) and row reduce

    if constexpr (is_integer_mode) {
        lltt::record(0, 6);

        // Full reduce (positions 0-5): interleaved upper/lower face summation
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, 4);  // LREG2 = LREG2 + LREG3
        TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG6, 4);  // LREG6 = LREG6 + LREG7
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG1, 4);  // LREG1 = LREG1 + LREG2
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, 4);  // LREG5 = LREG5 + LREG6
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);  // LREG0 = LREG0 + LREG1
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, 4);  // LREG4 = LREG4 + LREG5
    } else {
        lltt::record(0, 6);

        // Full reduce (positions 0-5): interleaved to eliminate read-after-write dependencies
        TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);  // A1
        TTI_SFPADD(p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG6, 0);  // B1
        TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG1, 0);  // A2
        TTI_SFPADD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG5, 0);  // B2
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // A3
        TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);  // B3
    }
}

// ============================================================================
// Calculate Functions
// ============================================================================

/**
 * @brief Column-wise maximum/minimum reduction kernel for UInt16 stored in a 32-bit (fp32 dest acc) dest.
 *
 * UInt16 datums live in the low 16 bits of a 32-bit dest word and carry garbage in the high 16 bits,
 * so every value must be masked (AND 0x0000FFFF) before it participates in a compare-and-swap. The
 * float/UInt16 LOADMACRO pipeline used by calculate_reduce_max_min() fuses the load and the swap into
 * a single macro, leaving no place to clear the high bits in between; the garbage then dominates the
 * integer comparison and produces wrong minima/maxima for any non-constant input.
 *
 * Instead we use the same manual load/mask/swap structure as the Int32 path (which exists for the
 * analogous reason that CAST and SWAP cannot share a LOADMACRO). The sign-magnitude casts of the
 * Int32 path are unnecessary here: masked UInt16 values are always non-negative, so a plain integer
 * SFPSWAP orders them correctly and the cast would be the identity. We therefore replace the casts
 * with plain moves (LREG0-3 -> LREG4-7) that feed the recorded swap buffer. Final, packer-visible
 * results are written with SFPSTORE mode 9 (SFPSTORE_MOD0_FMT_LO16) so the packer reads the low bits.
 *
 * The reduction reuses init_reduce_max_min_int32()'s 3-swap replay buffer and swap-direction config.
 * Only a single 32x32 tile is processed per call (block height 1), matching the column-reduce driver
 * which invokes the kernel once per tile.
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

    // Shared tile-layout address tables live at file scope (COL_REDUCE_*), so the UInt16 and Int32
    // column MAX/MIN paths stay in lockstep.

    // The intermediate stores below land in non-row-0 dest slots that we reload, so they use the
    // plain (full 32-bit) instruction mode. Only the final row-0 stores are packer-visible and use
    // mode 9 (SFPSTORE_MOD0_FMT_LO16) when the OUTPUT is UInt16 in a 32-bit dest (pack_low16), so the
    // packer reads the low 16 bits of the dest word.
    constexpr std::uint32_t STORE_MODE =
        pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);

    // This path reads VD, so MAX needs the inverted direction. Set it on entry: under one shared init a
    // preceding row MAX/MIN leaves the row convention, and under DISABLE_SFPLOADMACRO every non-Int32 column
    // MAX/MIN comes through here.
    set_manual_col_swap_direction<pool_type>();

    for (std::uint32_t j = 0; j < 2; j++) {
        std::uint32_t top_face_addr = COL_REDUCE_FINAL_ADDRS[j][0];     // face 0 & 1 dst indices
        std::uint32_t bottom_face_addr = COL_REDUCE_FINAL_ADDRS[j][1];  // face 2 & 3 dst indices

        // Reduce each of the four vertically adjacent faces (f0,f2 then f1,f3) within itself; the
        // max/min of its 16 rows is left in the top 4 rows.
        for (std::uint32_t i = 0; i < NUM_FACES; i++) {
            // Masked load straight into LREG4-7, where the recorded swap buffer reduces them. Loading
            // directly into the target registers eliminates the four LREG0-3 -> LREG4-7 moves (and the
            // post-reduce LREG4 -> LREG0 move) that the previous LREG0-3 load required.
            load_face_data<INSTRUCTION_MODE, clear_high_bits, p_sfpu::LREG4>(
                COL_REDUCE_FACE_ADDRS[j][i], COL_REDUCE_COLUMN_OFFSETS[i]);

            lltt::replay(0, 3);  // compare-and-swap reduce LREG4-7 -> LREG4

            TT_SFPSTORE(
                p_sfpu::LREG4,
                INSTRUCTION_MODE,
                ADDR_MOD_7,
                COL_REDUCE_FACE_ADDRS[j][i] + COL_REDUCE_COLUMN_OFFSETS[i]);
        }

        // Load the partial max/min (top 4 rows) of the two vertically adjacent faces into LREG0-3.
        load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, top_face_addr);
        load_and_clear_high_bits<clear_high_bits>(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, bottom_face_addr);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, top_face_addr + COL_REDUCE_ODD_COLUMNS);
        load_and_clear_high_bits<clear_high_bits>(
            p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, bottom_face_addr + COL_REDUCE_ODD_COLUMNS);

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

        TT_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_7, top_face_addr);
        TT_SFPSTORE(p_sfpu::LREG1, STORE_MODE, ADDR_MOD_7, top_face_addr + COL_REDUCE_ODD_COLUMNS);
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
 *        - Transposes and sorts results to align maxima/minima correctly across LREG4-7
 *        - Stores final maximum/minimum values to row 0 (32 datums across faces 0 and 1)
 *
 * @tparam pool_type The pool type (MAX or MIN)
 * @tparam reduce_dim The reduction dimension (currently only REDUCE_COL is supported)
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, LO16, DEFAULT
 * (FP32, FP16B)
 * @param block_height The number of tiles in the vertical block to reduce (default is 1 for single tile).
 *                     For example, block_height=4 means reduce across 4 vertically stacked tiles (128 rows total).
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    InstrModLoadStore INSTRUCTION_MODE,
    bool clear_high_bits,
    bool pack_low16>
inline void calculate_reduce_max_min(const std::uint32_t block_height) {
    static_assert(reduce_dim == ReduceDim::REDUCE_COL, "Only column reduction (REDUCE_COL) is currently supported");
    static_assert(
        pool_type == PoolType::MAX || pool_type == PoolType::MIN,
        "Only MAX and MIN pool types are supported for this function");

    // Per-face-pair replay window and the dummy-load tail in the recorded LOADMACRO buffer.
    constexpr std::uint32_t replay_buffer_offset = 9;
    constexpr std::uint32_t replay_buffer_next_face = 10;

    // Initial loads: LREG4-7 will hold maximum values across F0 and F1
    TTI_SFPLOAD(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, 2);
    TTI_SFPLOAD(p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, 16);
    TTI_SFPLOAD(p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, 18);

    // First tile processing (F0, F1, F2, F3)
    lltt::replay(0, replay_buffer_offset);
    lltt::replay(0, replay_buffer_offset);
    lltt::replay(0, replay_buffer_next_face);

    lltt::replay(0, replay_buffer_offset);
    lltt::replay(0, replay_buffer_offset);
    lltt::replay(0, replay_buffer_offset);
    lltt::replay(0, replay_buffer_next_face + 1);

    // Remaining tiles
    for (std::uint32_t i = 0; i < block_height - 1; i++) {
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
    TTI_SFPSWAP(0 /*unused*/, p_sfpu::LREG6 /*lreg_src_c*/, p_sfpu::LREG7 /*lreg_dest*/, 1 /*instr_mod1*/);
    TTI_SFPSWAP(0 /*unused*/, p_sfpu::LREG5 /*lreg_src_c*/, p_sfpu::LREG6 /*lreg_dest*/, 1 /*instr_mod1*/);
    TTI_SFPSWAP(0 /*unused*/, p_sfpu::LREG4 /*lreg_src_c*/, p_sfpu::LREG5 /*lreg_dest*/, 1 /*instr_mod1*/);
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Store results to first row.
    // For UInt16 OUTPUT in a 32-bit dest the reduced value lives in the low 16 bits of the LREG, but
    // the packer reads the high 16 bits of the dest word. SFPSTORE mode 9 (SFPSTORE_MOD0_FMT_LO16)
    // writes the low 16 bits into the half the packer consumes; a 32-bit output keeps the plain store.
    constexpr std::uint32_t STORE_MODE =
        pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
    TTI_SFPSTORE(p_sfpu::LREG4, STORE_MODE, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG5, STORE_MODE, ADDR_MOD_7, 2);
    TTI_SFPSTORE(p_sfpu::LREG6, STORE_MODE, ADDR_MOD_7, 16);
    TTI_SFPSTORE(p_sfpu::LREG7, STORE_MODE, ADDR_MOD_7, 18);
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
 *        For AVG mode: signed Int32 divides by 32 with a magnitude, logical shift and sign restore; the
 *        unsigned formats (UInt32, UInt16) use a plain logical shift; float formats multiply by 1/32.
 *
 * @tparam pool_type The reduction operation, currently supported: (SUM, AVG)
 * @tparam reduce_dim The reduction dimension (currently only REDUCE_COL is supported)
 * @tparam INSTRUCTION_MODE The instruction mode for integer and float formats: INT32, LO16, DEFAULT
 * (FP32, FP16B)
 * @tparam is_signed_int Whether the reduce format is signed (Int32). Passed separately from
 * INSTRUCTION_MODE because unsigned 32-bit formats share the INT32 mode; see perform_int_average().
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    InstrModLoadStore INSTRUCTION_MODE,
    bool clear_high_bits,
    bool pack_low16,
    bool is_signed_int>
inline void calculate_reduce_sum_avg(std::uint32_t block_ct_dim, std::uint32_t block_rt_dim) {
    // Integer vs float is determined by the load/store mode. Row AVG divides the row sum by the
    // (runtime) column count, which is only exact via a reciprocal multiply for float formats; an
    // integer row AVG by an arbitrary column count would need a general integer divide and is not
    // supported (integer AVG stays column-only, where the divisor is the fixed 32 rows of a tile).
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::LO16);

    // Compile-time assertions to restrict to currently supported operations
    static_assert(
        reduce_dim == ReduceDim::REDUCE_COL ||
            (reduce_dim == ReduceDim::REDUCE_ROW &&
             (pool_type == PoolType::SUM || (pool_type == PoolType::AVG && !is_integer_mode))),
        "Row reduction (REDUCE_ROW) supports SUM (all formats) and AVG (float formats only)");
    static_assert(
        pool_type == PoolType::SUM || pool_type == PoolType::AVG,
        "Only SUM and AVG pool types are currently supported on SFPU");

    // Supported instruction modes for SFPU reduce sum/avg (integer and float)
    constexpr bool is_supported_reduce_instr_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::LO16 ||
         INSTRUCTION_MODE == InstrModLoadStore::DEFAULT || INSTRUCTION_MODE == InstrModLoadStore::FP32 ||
         INSTRUCTION_MODE == InstrModLoadStore::FP16B);
    static_assert(is_supported_reduce_instr_mode, "INSTRUCTION_MODE must be one of: INT32, LO16, FP32, FP16B, DEFAULT");

    if constexpr (reduce_dim == ReduceDim::REDUCE_COL) {
        perform_reduce_col_sum_avg<pool_type, INSTRUCTION_MODE, clear_high_bits, pack_low16, is_signed_int>();
    } else {
        perform_reduce_row_sum_avg<pool_type, INSTRUCTION_MODE, clear_high_bits, pack_low16>(
            block_ct_dim, block_rt_dim);
    }
    // For column reductions: sums are stored horizontally in the first row of tensor in dest reg
    // For row reductions: sums are stored vertically in the first column of tensor in dest reg
}

// ============================================================================
// Signed Int32 two's-complement MAX/MIN reduce (issue #49803)
// ============================================================================
// The sign-magnitude paths above (convert_int_representation_inplace / SFPCAST
// INT_SIGN_MAGN_TO_INT32_2S_COMP) collapse INT32_MIN (0x80000000) to sign-magnitude "-0" (magnitude 0),
// so it ranks as 0 and is dropped by SFPSWAP. These functions keep the raw two's-complement bits (load
// plain INT32) and correct the ordering in software: the column and cross-tile paths with the both-negative
// re-swap in _emit_int32_signed_cswap_ (as the Wormhole fix, PR #49085), the per-tile row path with an order
// map on every operand. Correct over the full Int32 range, INT32_MIN included. The sign-magnitude functions
// above are kept for SUM/AVG (which need two's-complement for SFPIADD) and for UInt16/UInt32.

// Number of SFPU instructions emitted by _emit_int32_signed_cswap_ (one two's-complement compare-and-swap).
constexpr std::uint32_t INT32_SIGNED_CSWAP_LEN = 5;

/**
 * @brief Two's-complement signed compare-and-swap of a register pair, matching SFPSWAP(VEC_MIN_MAX)
 *        semantics but correct for the full Int32 range (including INT32_MIN).
 *
 * SFPSWAP(VEC_MIN_MAX) compares in sign-magnitude. For two's-complement operands that is correct except
 * when BOTH operands are negative (order reverses), and INT32_MIN reads as sign-magnitude "-0" (ranked
 * as 0). Loading plain INT32 (bits preserved, not cast to sign-magnitude) and re-exchanging the pair on
 * lanes where both operands are negative cures both problems. The re-swap is SFPSWAP_MOD1_SWAP
 * (unconditional exchange) gated by the condition code, so it is unaffected by the MAX/MIN direction
 * config, and being a symmetric exchange it leaves the corrected extreme in whichever register the
 * caller reads next. Mirrors calculate_binary_max_min_int32 in ckernel_sfpu_binary_max_min.h.
 *
 * @tparam HIGH_LREG The srcC register of the compare-and-swap.
 * @tparam LOW_LREG  The dest register of the compare-and-swap.
 */
template <std::uint32_t HIGH_LREG, std::uint32_t LOW_LREG>
inline void _emit_int32_signed_cswap_() {
    TTI_SFPSWAP(0, HIGH_LREG, LOW_LREG, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    TTI_SFPSETCC(0, LOW_LREG, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);   // cc: LOW_LREG < 0
    TTI_SFPSETCC(0, HIGH_LREG, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);  // cc &= HIGH_LREG < 0 (both negative)
    TTI_SFPSWAP(0, HIGH_LREG, LOW_LREG, sfpi::SFPSWAP_MOD1_SWAP);
    TTI_SFPENCC(0, 0, 0, 0);
}

// The row path maps each operand to x ^ 0x7FFFFFFF on negative lanes, the inverse of SFPSWAP's sign-magnitude remap
// (tt-isa SFPSWAP.md), so a plain SFPSWAP orders two's-complement values; the map is its own inverse.
constexpr std::uint32_t INT32_ORDER_MAP_REG = p_sfpu::LREG12;

// Writes 0x7FFFFFFF to INT32_ORDER_MAP_REG (sfpi::vConstIntPrgm0). Clobbers LREG0.
inline void load_int32_order_map_mask() {
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0xFFFF);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x7FFF);
    TTI_SFPCONFIG(0, INT32_ORDER_MAP_REG, 0);
}

template <std::uint32_t LREG>
inline void int32_order_map() {
    TTI_SFPSETCC(0, LREG, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPXOR(0, INT32_ORDER_MAP_REG, LREG, 0);
    TTI_SFPENCC(0, 0, 0, 0);
}

/**
 * @brief Signed-Int32 per-tile row MAX/MIN reduction. Mirrors perform_reduce_row_max_tile's manual path on
 *        order-mapped operands; the two results are mapped back before the store. Needs INT32_ORDER_MAP_REG.
 */
inline void perform_reduce_row_max_tile_int32(std::uint32_t tile_row_offset, std::uint32_t result_store_mode) {
    constexpr InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32;
#pragma GCC unroll 2
    for (std::uint32_t face_pair = 0; face_pair < 2; face_pair++) {
        std::uint32_t face_pair_base = face_pair * 2 * ROWS_PER_FACE;

#pragma GCC unroll 2
        for (std::uint32_t row_group = 0; row_group < 2; row_group++) {
            std::uint32_t row_offset_first = row_group * 8;
            std::uint32_t row_offset_second = row_offset_first + 4;

            TT_SFPLOAD(
                p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_first);
            TT_SFPLOAD(
                p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_first + 2);
            TT_SFPLOAD(
                p_sfpu::LREG2,
                INSTRUCTION_MODE,
                ADDR_MOD_7,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first);
            TT_SFPLOAD(
                p_sfpu::LREG3,
                INSTRUCTION_MODE,
                ADDR_MOD_7,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_first + 2);

            TT_SFPLOAD(
                p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_second);
            TT_SFPLOAD(
                p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_second + 2);
            TT_SFPLOAD(
                p_sfpu::LREG6,
                INSTRUCTION_MODE,
                ADDR_MOD_7,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second);
            TT_SFPLOAD(
                p_sfpu::LREG7,
                INSTRUCTION_MODE,
                ADDR_MOD_7,
                tile_row_offset + face_pair_base + ROWS_PER_FACE + row_offset_second + 2);

            int32_order_map<p_sfpu::LREG0>();
            int32_order_map<p_sfpu::LREG1>();
            int32_order_map<p_sfpu::LREG2>();
            int32_order_map<p_sfpu::LREG3>();
            int32_order_map<p_sfpu::LREG4>();
            int32_order_map<p_sfpu::LREG5>();
            int32_order_map<p_sfpu::LREG6>();
            int32_order_map<p_sfpu::LREG7>();

            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);

            horizontal_reduce_merged<RowFoldSwap>();

            int32_order_map<p_sfpu::LREG0>();
            int32_order_map<p_sfpu::LREG4>();

            TT_SFPSTORE(
                p_sfpu::LREG0, result_store_mode, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_first);
            TT_SFPSTORE(
                p_sfpu::LREG4, result_store_mode, ADDR_MOD_7, tile_row_offset + face_pair_base + row_offset_second);
        }
    }
}

/**
 * @brief Signed-Int32 cross-tile row MAX/MIN combine. Mirrors max_first_columns_across_tiles but loads
 *        plain INT32 and uses the signed compare-and-swap. Named _signed to avoid colliding with the
 *        sign-magnitude max_first_columns_across_tiles_int32 above (still used by the UInt16/UInt32 path).
 */
inline void max_first_columns_across_tiles_int32_signed(std::uint32_t tile_row_base, std::uint32_t block_ct_dim) {
    constexpr InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32;
    constexpr std::uint32_t RESULT_ROWS[8] = {0, 4, 8, 12, 32, 36, 40, 44};

    for (std::uint32_t batch = 0; batch < 2; batch++) {
        std::uint32_t base_idx = batch * 4;

        TT_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 0]);
        TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 1]);
        TT_SFPLOAD(p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 2]);
        TT_SFPLOAD(p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 3]);

        for (std::uint32_t t = 1; t < block_ct_dim; t++) {
            std::uint32_t tile_offset = tile_row_base + t * ROWS_PER_TILE;

            TT_SFPLOAD(p_sfpu::LREG4, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 0]);
            TT_SFPLOAD(p_sfpu::LREG5, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 1]);
            TT_SFPLOAD(p_sfpu::LREG6, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 2]);
            TT_SFPLOAD(p_sfpu::LREG7, INSTRUCTION_MODE, ADDR_MOD_7, tile_offset + RESULT_ROWS[base_idx + 3]);

            _emit_int32_signed_cswap_<p_sfpu::LREG0, p_sfpu::LREG4>();
            _emit_int32_signed_cswap_<p_sfpu::LREG1, p_sfpu::LREG5>();
            _emit_int32_signed_cswap_<p_sfpu::LREG2, p_sfpu::LREG6>();
            _emit_int32_signed_cswap_<p_sfpu::LREG3, p_sfpu::LREG7>();
        }

        TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 0]);
        TT_SFPSTORE(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 1]);
        TT_SFPSTORE(p_sfpu::LREG2, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 2]);
        TT_SFPSTORE(p_sfpu::LREG3, INSTRUCTION_MODE, ADDR_MOD_7, tile_row_base + RESULT_ROWS[base_idx + 3]);
    }
}

/**
 * @brief Signed-Int32 row MAX/MIN reduction across a block of tiles. Mirrors perform_reduce_row_max_min
 *        but routes through the signed-Int32 per-tile and cross-tile helpers (the horizontal reduce is fully
 *        inline, no recorded buffer). Correct over the full Int32 range.
 */
template <PoolType pool_type>
inline void perform_reduce_row_max_min_int32(std::uint32_t block_ct_dim, std::uint32_t block_rt_dim) {
    static_assert(
        pool_type == PoolType::MAX || pool_type == PoolType::MIN,
        "perform_reduce_row_max_min_int32 only supports MAX and MIN pool types");

    constexpr InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32;

    // Re-establish the SFPSWAP direction (see perform_reduce_row_max_min): MAX is the default (bit 8 = 0),
    // MIN sets bit 8. The cross-tile signed compare-and-swap obeys it too; its both-negative fix is a
    // direction-independent unconditional exchange.
    set_sfpswap_direction<pool_type == PoolType::MIN>();
    load_int32_order_map_mask();

    // Int32 MAX/MIN never widens to a UInt16 output, so the per-tile store always uses the plain mode.
    const std::uint32_t tile_store_mode = static_cast<std::uint32_t>(INSTRUCTION_MODE);

    for (std::uint32_t i = 0; i < block_rt_dim; i++) {
        std::uint32_t tile_row_offset = ROWS_PER_TILE * block_ct_dim * i;

        for (std::uint32_t j = 0; j < block_ct_dim; j++) {
            std::uint32_t tile_offset = tile_row_offset + (ROWS_PER_TILE * j);
            perform_reduce_row_max_tile_int32(tile_offset, tile_store_mode);
        }

        if (block_ct_dim > 1) {
            max_first_columns_across_tiles_int32_signed(tile_row_offset, block_ct_dim);
        }
    }
}

/**
 * @brief Init for the signed Int32 column MAX/MIN reduce: resets the SFPU config and records the LREG4-7 ->
 *        LREG4 window of three two's-complement signed compare-and-swaps (so INT32_MIN is handled).
 *
 * No direction is set here: both Int32 MAX/MIN calculates set their own on entry, and the recorded
 * compare-and-swaps read it at replay time.
 *
 * @tparam pool_type MAX or MIN; unused here.
 */
template <PoolType pool_type>
inline void init_reduce_max_min_int32_signed() {
    _init_sfpu_config_reg();

    lltt::record(0, 3 * INT32_SIGNED_CSWAP_LEN);
    _emit_int32_signed_cswap_<p_sfpu::LREG7, p_sfpu::LREG6>();
    _emit_int32_signed_cswap_<p_sfpu::LREG6, p_sfpu::LREG5>();
    _emit_int32_signed_cswap_<p_sfpu::LREG5, p_sfpu::LREG4>();
}

/**
 * @brief Column-wise MAX/MIN reduction for signed Int32, single 32x32 tile, correct over the full Int32
 *        range (including INT32_MIN). Same manual load/reduce/transpose structure as
 *        calculate_reduce_max_min_uint16, but loads plain INT32 (two's-complement bits preserved) and uses
 *        the signed compare-and-swap. The vertical LREG4-7 -> LREG4 reduction reuses the 3-swap replay
 *        buffer recorded by init_reduce_max_min_int32_signed; the final face-pair combine emits the signed
 *        swap inline. Int32 MAX/MIN only supports a single tile (block_rt_dim == 1).
 *
 * @tparam pool_type MAX or MIN. MAX is the inverted swap direction here; it is set on entry, not taken from
 *         the init.
 * @tparam reduce_dim The reduction dimension; must be REDUCE_COL for this helper.
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void calculate_reduce_max_min_int32_col() {
    static_assert(reduce_dim == ReduceDim::REDUCE_COL, "Only column reduction (REDUCE_COL) is supported here");
    static_assert(
        pool_type == PoolType::MAX || pool_type == PoolType::MIN,
        "Only MAX and MIN pool types are supported for this function");

    constexpr InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32;
    constexpr std::uint32_t REPLAY_LEN = 3 * INT32_SIGNED_CSWAP_LEN;

    constexpr std::uint32_t ODD_COLUMNS = 2;
    constexpr std::uint32_t COLUMN_OFFSETS[4] = {0, 2, 0, 2};  // even, odd, even, odd
    constexpr std::uint32_t FACE_ADDRS[2][4] = {
        {0, 0, 32, 32},   // j=0: Face 0 and Face 2
        {16, 16, 48, 48}  // j=1: Face 1 and Face 3
    };

    // Set this path's direction on entry: under one shared init a preceding row MAX/MIN leaves the row
    // convention, which is inverted here.
    set_manual_col_swap_direction<pool_type>();

    // Where each per-face partial (left in LREG4 by the reduce) is parked so it survives the remaining
    // face loads (which clobber LREG4-7). The order matches the LREG0-3 layout the transpose expects.
    constexpr std::uint32_t PARTIAL_LREG[4] = {p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG3};

    for (std::uint32_t j = 0; j < 2; j++) {
        std::uint32_t top_face_addr = FACE_ADDRS[j][0];  // face 0 & 1 row-0 dst index

        // Reduce each of the four vertically adjacent faces within itself; the max/min of its 16 rows is
        // left in the top 4 rows. The face loads and the reduce replay only touch LREG4-7, so each partial
        // can be parked in LREG0-3 (via PARTIAL_LREG[i]) and survive the remaining faces.
        for (std::uint32_t i = 0; i < NUM_FACES; i++) {
            load_face_data<INSTRUCTION_MODE, false, p_sfpu::LREG4>(FACE_ADDRS[j][i], COLUMN_OFFSETS[i]);

            lltt::replay(0, REPLAY_LEN);  // signed compare-and-swap reduce LREG4-7 -> LREG4

            TT_SFPMOV(0, p_sfpu::LREG4, PARTIAL_LREG[i], 0);
        }

        // Move the four partials (now in LREG0-3) into LREG4-7 for the transpose + cross-row reduction.
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
        TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG5, 0);
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG6, 0);
        TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG7, 0);

        // Transpose so the 4 partial results of each column sit in one register, reduce, transpose back.
        TTI_SFPTRANSP(0, 0, 0, 0);
        lltt::replay(0, REPLAY_LEN);
        TTI_SFPTRANSP(0, 0, 0, 0);

        // Signed compare-and-swap to combine the two vertically adjacent faces (even and odd columns).
        _emit_int32_signed_cswap_<p_sfpu::LREG7, p_sfpu::LREG6>();  // odd columns of face pair
        _emit_int32_signed_cswap_<p_sfpu::LREG5, p_sfpu::LREG4>();  // even columns of face pair

        TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPMOV(0, p_sfpu::LREG6, p_sfpu::LREG1, 0);

        TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, top_face_addr);
        TT_SFPSTORE(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, top_face_addr + ODD_COLUMNS);
    }
}

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Unified reduction init kernel wrapper for SFPU reduce kernel.
 *        Determines the instruction mode from format, then dispatches to the appropriate init kernel.
 * @tparam pool_type The reduction operation, currently supported: (SUM, AVG, MAX, MIN)
 * @tparam format The data format, currently supported: (Int32, UInt32, UInt16, Float32, Float16_b)
 * @param block_ct_dim Block dimension (used for MAX/MIN reduction to specify number of columns, default is 1 for single
 * tile)
 */
template <PoolType pool_type, DataFormat format, bool is_fp32_dest_acc_en>
inline void init_reduce(std::uint32_t block_ct_dim = 1) {
    math::reset_counters(p_setrwc::SET_ABD_F);
    static_assert(
        is_supported_reduce_format(format),
        "Unsupported data format. Supported formats: Int32, UInt32, UInt16, Float32, Float16_b");

    // Int32 reduce operands are two's-complement in DEST for every reduction. SUM and AVG load with plain
    // INT32 so the word reaches SFPIADD (a two's-complement adder) unchanged. MAX/MIN
    // dispatch to init_reduce_max_min_int32_signed below (plain INT32 load + a software signed
    // compare-and-swap, correct over the full Int32 range including INT32_MIN), so they do not use
    // INSTRUCTION_MODE here. init_reduce has no reduce_dim, so the column reduce consumes the LREG4-7 -> LREG4
    // replay buffer recorded by init_reduce_max_min_int32_signed; both calculates set their own SFPSWAP
    // direction on entry.
    constexpr bool int32_max_min =
        (format == DataFormat::Int32 && (pool_type == PoolType::MAX || pool_type == PoolType::MIN));
    constexpr InstrModLoadStore INSTRUCTION_MODE = GetSfpLoadStoreInstrMod<format, is_fp32_dest_acc_en>();

    // Garbage high bits needs to be cleared when loading UInt16 data
    constexpr bool clear_high_bits = (is_fp32_dest_acc_en && format == DataFormat::UInt16);

    if constexpr (clear_high_bits) {
        // CLEAR_REG (sfpi::vConstIntPrgm0 / LREG12) holds the mask used by SFPLOAD_EXT to clear high bits.
        sfpi::vConstIntPrgm0 = 0x0000FFFF;
    }

    // Dispatch to appropriate PoolType init
    if constexpr (pool_type == PoolType::MAX || pool_type == PoolType::MIN) {
        if constexpr (int32_max_min) {
            // Signed Int32 MAX/MIN: records the LREG4-7 -> LREG4 signed compare-and-swap replay buffer
            // (consumed by the column reduce). Handles INT32_MIN correctly.
            init_reduce_max_min_int32_signed<pool_type>();
        } else if constexpr (clear_high_bits) {
            // UInt16 in 32-bit dest uses the manual (non-LOADMACRO) compare-and-swap path so the
            // garbage high bits can be masked before each swap. It reuses the Int32 path's 3-swap
            // replay buffer (the body is format-agnostic); the calculate sets the swap direction itself.
            init_reduce_max_min_int32<INSTRUCTION_MODE, pool_type>();
        } else {
            // Non-Int32 MAX/MIN (Float32, Float16_b, UInt32): the generic LOADMACRO-based init (or its
            // manual-swap fallback under DISABLE_SFPLOADMACRO). Int32 is fully handled by int32_max_min
            // above, for both column and row reduce.
            init_reduce_max_min<INSTRUCTION_MODE, pool_type, false>(block_ct_dim);
        }
    } else if constexpr (pool_type == PoolType::SUM || pool_type == PoolType::AVG) {
        init_reduce_sum_avg<INSTRUCTION_MODE, pool_type>();
    } else {
        static_assert(
            pool_type == PoolType::SUM || pool_type == PoolType::AVG || pool_type == PoolType::MAX ||
                pool_type == PoolType::MIN,
            "Unsupported pool_type. Currently supported: SUM, AVG, MAX, MIN");
    }
}

/**
 * @brief Unified reduction kernel wrapper for a 32x32 tile.
 *        Determines the instruction mode from format, then dispatches to the appropriate reduction kernel.
 * @tparam pool_type The reduction operation, currently supported: (SUM, AVG, MAX, MIN)
 * @tparam reduce_dim The reduction dimension: REDUCE_COL for column-wise, REDUCE_ROW for row-wise (SUM/MAX/MIN all
 * formats; AVG float formats only).
 * @tparam format The INPUT data format, currently supported: (Int32, UInt32, UInt16, Float32, Float16_b). Drives the
 *         instruction mode and load-time high-bit masking.
 * @tparam output_format The packer-visible OUTPUT data format (defaults to @p format). Drives the final store mode:
 *         UInt16 output in a 32-bit dest is stored via mode 9 (low->high 16-bit swap), while a 32-bit output (e.g.
 *         UInt32) is stored with the plain instruction mode. This lets UInt16 input be summed into a UInt32 output
 *         without overflow.
 * @param block_ct_dim Block dimension (used for SUM/AVG column reduction to specify number of columns, default is 1 for
 * single tile)
 * @param block_rt_dim Block dimension (used for MAX/MIN reduction to specify block height, or SUM/MAX row reduction;
 * default is 1 for single tile)
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    DataFormat format,
    bool is_fp32_dest_acc_en,
    DataFormat output_format = format>
inline void calculate_reduce(
    [[maybe_unused]] std::uint32_t block_ct_dim = 1, [[maybe_unused]] std::uint32_t block_rt_dim = 1) {
    // Row reduction supports SUM/MAX/MIN for every supported format; AVG is row-supported only for
    // float formats, because the row divisor is the runtime column count and only the float
    // reciprocal-multiply divides exactly (integer AVG stays column-only with its fixed /32 divisor).
    constexpr bool is_float_format = (format == DataFormat::Float32 || format == DataFormat::Float16_b);
    static_assert(
        reduce_dim == ReduceDim::REDUCE_COL ||
            (reduce_dim == ReduceDim::REDUCE_ROW &&
             (pool_type == PoolType::SUM || pool_type == PoolType::MAX || pool_type == PoolType::MIN ||
              (pool_type == PoolType::AVG && is_float_format))),
        "Row reduction (REDUCE_ROW) supports SUM/MAX/MIN (all formats) and AVG (float formats only)");
    static_assert(
        is_supported_reduce_format(format),
        "Unsupported data format. Supported formats: Int32, UInt32, UInt16, Float32, Float16_b");

    // Int32 DEST representation and load/store mode per reduction. On Blackhole INT32_2S_COMP is a no-op
    // (tt-isa SFPLOAD.md: MOD0_FMT_INT32_SM is deprecated and performs no conversion), so any
    // sign-magnitude<->two's-complement change is done explicitly via SFPCAST. DEST holds two's-complement
    // Int32 for every reduction:
    //   SUM (two's-complement): plain INT32 so SFPIADD (a two's-complement adder) gets the word unchanged.
    //   MAX/MIN (two's-complement): plain INT32 (bits preserved), feeding the software signed
    //        compare-and-swap path (calculate_reduce_max_min_int32_col / perform_reduce_row_max_min_int32),
    //        which is correct over the full Int32 range including INT32_MIN (issue #49803). Applies to BOTH
    //        column and row, so a multi-axis reduce (column-then-row over the same DEST) stays consistent.
    //   AVG (two's-complement): plain INT32, like SUM, so the column sum and perform_int_average's
    //        divide-by-32 see the word unchanged.
    constexpr bool int32_max_min =
        (format == DataFormat::Int32 && (pool_type == PoolType::MAX || pool_type == PoolType::MIN));
    constexpr bool int32_max_min_col = int32_max_min && (reduce_dim == ReduceDim::REDUCE_COL);
    constexpr bool int32_max_min_row = int32_max_min && (reduce_dim == ReduceDim::REDUCE_ROW);
    constexpr InstrModLoadStore INSTRUCTION_MODE =
        int32_max_min ? InstrModLoadStore::INT32 : GetSfpLoadStoreInstrMod<format, is_fp32_dest_acc_en>();

    // Signedness of the DATA, which the instruction mode does not carry: Int32, UInt32 and UInt16 in a
    // 32-bit Dest all load with InstrModLoadStore::INT32. Only the integer column AVG divide needs it;
    // reading it off the mode instead divided every unsigned column sum with bit 31 set as a signed one
    // (tenstorrent/tt-metal#57509); see perform_int_average().
    constexpr bool is_signed_int = (format == DataFormat::Int32);

    // Garbage high bits needs to be cleared when loading UInt16 data (driven by INPUT format).
    constexpr bool clear_high_bits = (is_fp32_dest_acc_en && format == DataFormat::UInt16);

    // The packer-visible result must go through mode-9 (low->high 16-bit) store only when the OUTPUT is UInt16
    // in a 32-bit dest. A 32-bit output (e.g. UInt32) keeps the full word, so it uses the plain store. This is
    // driven by the OUTPUT format and is independent of the load-time masking above.
    constexpr bool pack_low16 = (is_fp32_dest_acc_en && output_format == DataFormat::UInt16);

    // Dispatch to appropriate reduction kernel based on PoolType
    if constexpr (pool_type == PoolType::MAX || pool_type == PoolType::MIN) {
        if constexpr (int32_max_min_col) {
            // Signed Int32 column MAX/MIN: two's-complement compare-and-swap path (handles INT32_MIN,
            // issue #49803). Single-tile (32x32) kernel that ignores block_ct_dim/block_rt_dim, so guard
            // the single-tile contract loudly rather than silently dropping tiles.
            LLK_ASSERT(
                block_ct_dim == 1 && block_rt_dim == 1,
                "Int32 column MAX/MIN reduce only supports a single tile (block_ct_dim == block_rt_dim == 1)");
            calculate_reduce_max_min_int32_col<pool_type, reduce_dim>();
        } else if constexpr (int32_max_min_row) {
            // Signed Int32 row MAX/MIN: two's-complement compare-and-swap path (handles INT32_MIN).
            perform_reduce_row_max_min_int32<pool_type>(block_ct_dim, block_rt_dim);
        } else if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
            static_assert(
                INSTRUCTION_MODE == InstrModLoadStore::FP32 || INSTRUCTION_MODE == InstrModLoadStore::INT32 ||
                    INSTRUCTION_MODE == InstrModLoadStore::FP16B,
                "Row MAX/MIN reduction supports FP32, FP16B, and INT32 (sign-magnitude) instruction modes");
            perform_reduce_row_max_min<pool_type, INSTRUCTION_MODE, clear_high_bits, pack_low16>(
                block_ct_dim, block_rt_dim);
#ifdef DISABLE_SFPLOADMACRO
        } else {
            // LOADMACRO-disabled builds use the manual load/swap column reducer. With
            // clear_high_bits=false it is just the unfused equivalent of the LOADMACRO
            // compare-and-swap pipeline.
            calculate_reduce_max_min_uint16<pool_type, reduce_dim, INSTRUCTION_MODE, clear_high_bits, pack_low16>();
#else
        } else if constexpr (clear_high_bits) {
            // UInt16 in 32-bit dest: manual load/mask/swap path (LOADMACRO cannot mask between load and swap).
            calculate_reduce_max_min_uint16<pool_type, reduce_dim, INSTRUCTION_MODE, clear_high_bits, pack_low16>();
        } else {
            calculate_reduce_max_min<pool_type, reduce_dim, INSTRUCTION_MODE, false, pack_low16>(block_rt_dim);
#endif
        }
    } else if constexpr (pool_type == PoolType::SUM || pool_type == PoolType::AVG) {
        calculate_reduce_sum_avg<pool_type, reduce_dim, INSTRUCTION_MODE, clear_high_bits, pack_low16, is_signed_int>(
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
