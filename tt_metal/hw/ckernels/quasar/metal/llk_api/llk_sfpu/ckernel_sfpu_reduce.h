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
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// ============================================================================
// Dest geometry
// ============================================================================
// A 32x32 tile is not a flat square in Dest but four 16x16 faces: faces 0/1 side by side over
// rows 0-15, faces 2/3 the same over rows 16-31. One address holds one face row (16 datums), so
// face f starts at 16f - the four faces sit at 0, 16, 32 and 48.
//
// One SFPLOAD reads four consecutive face rows, and address bit 1 picks the even or odd 8
// columns of each - 4 rows x 8 columns = 32 datums, exactly the SFPU's 4 rows x 8 columns.
//
// Both axes load four at a time, but group them differently:
//   column group - one face, one column parity: 8 columns, that face's 16 rows.
//   row quad     - four whole tile rows: both parities of two side-by-side faces.

constexpr std::uint32_t REDUCE_FACE_STRIDE = FACE_R_DIM;                   // 16 addr units per face
constexpr std::uint32_t REDUCE_FACE_PAIR_STRIDE = 2 * REDUCE_FACE_STRIDE;  // faces 0/1 then faces 2/3
constexpr std::uint32_t REDUCE_TILE_STRIDE = 1U << trisc::get_dest_tile_size_log2(trisc::DstTileShape::Tile32x32);
constexpr std::uint32_t REDUCE_ROWS_PER_LOAD = 4;  // face rows one SFPLOAD covers
constexpr std::uint32_t REDUCE_QUADS_PER_FACE_PAIR = FACE_R_DIM / REDUCE_ROWS_PER_LOAD;
constexpr std::uint32_t REDUCE_FACE_PAIRS = TILE_NUM_FACES / 2;
constexpr std::uint32_t REDUCE_SFPU_COLUMNS = 8;         // SFPU column instances a row spans
constexpr std::uint32_t REDUCE_COL_EXTENT = TILE_R_DIM;  // rows a column reduce collapses
constexpr std::uint32_t REDUCE_COLS_PER_TILE = TILE_C_DIM;

// Working registers. Only LREG0-7 are usable as destinations here: writing 12-15 captures the
// instruction into a Load Macro register instead of executing it, and 9/10/11 are the read-only
// constants 0.0, 1.0 and -1.0.
constexpr std::uint32_t REDUCE_ACC_REG = p_sfpu::LREG4;    // row reduce: running per-row total
constexpr std::uint32_t REDUCE_ROT_REG = p_sfpu::LREG5;    // row reduce: rotated copy of the accumulator
constexpr std::uint32_t REDUCE_RECIP_REG = p_sfpu::LREG6;  // row AVG: 1/num_cols, live for the whole block

// bfloat16 encoding of 1/REDUCE_COL_EXTENT, the column-AVG divisor. A power of two, so the
// SFPMULI that applies it is exact.
constexpr std::uint32_t REDUCE_RECIP_COL_EXTENT_FP16B = 0x3D00;
static_assert(REDUCE_COL_EXTENT == 32, "REDUCE_RECIP_COL_EXTENT_FP16B is the bfloat16 encoding of 1/32");

// SFPSETCC imm12 bit 11: interpret src_c as two's-complement INT32 rather than FP32/SMAG32,
// which turns "less than zero" into a plain sign-bit test. Same encoding the Quasar
// element-wise max/min kernel uses (@ref calculate_binary_max_min).
constexpr std::uint32_t REDUCE_SFPSETCC_INT32_SIGNBIT = 0x800;

// SFPSHFT2 instr_mod1 3: rotate an LREG sideways across the 8 SFPU column instances, column X
// feeding column X+1 and wrapping around. This is the only instruction that moves data between
// columns at all, so it is the only way a row reduce can bring its 8 columns together.
constexpr std::uint32_t REDUCE_SFPSHFT2_ROTATE = 3;

/**
 * @brief SFPLOAD/SFPSTORE format-select code for a reduce operand format.
 *
 * Floats take DEFAULT, letting the hardware resolve the Dest word format at runtime from
 * ALU_ACC_CTRL_SFPU_Fp32 and the SrcB format register - both already programmed from formats.math.
 * One kernel body therefore covers Float32, Float16_b and Float16 in either Dest width.
 *
 * Int32 names its mode outright: implied formats are unreliable under unpack-to-dest (TEN-4674).
 *
 * @tparam FORMAT: Math-side data format of the reduce operand
 */
template <DataFormat FORMAT>
inline constexpr std::uint32_t reduce_sfpmem_mode() {
    return (FORMAT == DataFormat::Int32) ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::DEFAULT;
}

/** @brief Whether @p FORMAT reduces on the integer instruction path (SFPIADD, no float scaling). */
template <DataFormat FORMAT>
inline constexpr bool reduce_is_int_format() {
    return FORMAT == DataFormat::Int32;
}

/** @brief Formats this kernel implements. */
constexpr bool is_supported_reduce_format(DataFormat format) {
    return format == DataFormat::Float32 || format == DataFormat::Float16_b || format == DataFormat::Float16 ||
           format == DataFormat::Int32;
}

// ============================================================================
// Pairwise combine
// ============================================================================

/**
 * @brief Fold @p SRC into @p DST with the reduction's operator, leaving the result in @p DST.
 *
 * The single step every reduce is built from. SUM and AVG both just add; AVG differs only in the
 * scaling its caller applies at the end. MAX uses SFPSWAP, which writes the larger operand to its
 * first register and the smaller to its second - naming @p DST first is what leaves the max there.
 *
 * SFPSWAP compares as two's-complement int32, which is exactly right for Int32 (Dest already
 * holds two's-complement words). Float bits are sign-magnitude, and comparing those as two's
 * complement orders them correctly except when *both* operands are negative, where the ordering
 * inverts. The second, CC-guarded swap fixes only those lanes - so floats cost a compare plus a
 * correction, Int32 only a compare.
 *
 * @tparam POOL_TYPE: Reduction operator, values = <SUM/AVG/MAX>
 * @tparam IS_INT: Whether the operands are integers (skips the float sign-magnitude correction)
 * @tparam DST: LREG holding the running result; overwritten with the folded value
 * @tparam SRC: LREG holding the incoming value; clobbered
 * @note Every MAX fold ends in an SFPNOP: SFPSWAP takes 2 cycles and its result must not be read
 *       by the next instruction (the SFPSWAP -> SFPSTORE auto-stall bug). That trailing NOP is
 *       what makes this safe back-to-back and directly before a store.
 */
template <PoolType POOL_TYPE, bool IS_INT, std::uint32_t DST, std::uint32_t SRC>
inline void reduce_combine() {
    static_assert(DST <= p_sfpu::LREG7 && SRC <= p_sfpu::LREG7, "reduce may only write LREG0-7");
    static_assert(DST != SRC, "reduce_combine needs two distinct registers");

    if constexpr (POOL_TYPE == PoolType::MAX) {
        // Two's-complement compare: DST <- larger, SRC <- smaller.
        TTI_SFPSWAP(0 /* imm12 */, DST, SRC, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);

        if constexpr (!IS_INT) {
            // Successive SFPSETCCs AND into CC, so the swap below fires only where both
            // operands are negative - the lanes the two's-complement compare ordered backwards.
            TTI_SFPSETCC(REDUCE_SFPSETCC_INT32_SIGNBIT, DST, sfpi::SFPSETCC_MOD1_LREG_LT0);
            TTI_SFPSETCC(REDUCE_SFPSETCC_INT32_SIGNBIT, SRC, sfpi::SFPSETCC_MOD1_LREG_LT0);
            TTI_SFPSWAP(0 /* imm12 */, DST, SRC, p_sfpswap::UNCONDITIONALLY);
            TTI_SFPENCC(0 /* imm12 */, 0 /* mod1: clear CC */);
            TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
        }
    } else if constexpr (IS_INT) {
        // SFPIADD raises CC off the result sign by default; the reduce never predicates on it.
        TTI_SFPIADD(0 /* imm12 */, SRC, DST, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
    } else {
        // SFPADD is dest = a * b + c, so a = DST and b = 1.0 makes it dest = DST + SRC.
        TTI_SFPADD(DST, p_sfpu::LCONST_1, SRC, DST, 0 /* instr_mod1: no negation */);
    }
}

// ============================================================================
// Column reduce (REDUCE_COL): collapse a tile's 32 rows onto row 0
// ============================================================================

/**
 * @brief Reduce one column group of a tile - 8 of its 32 columns, over all 32 rows.
 *
 * A column of 32 values spans two faces vertically, so a group pairs a top face with the one below
 * it at a single column parity. Eight SFPLOADs cover all 32 rows: LREG0-3 top, LREG4-7 bottom.
 *
 * Folding takes two passes, because the SFPU can only combine matching lanes of two registers -
 * never two lanes of the same register. Pass 1 folds each bank to LREG0/LREG4 then folds those
 * together, leaving every LREG0 lane holding a quarter of its column: lane (r, c) covers rows r,
 * r+4, r+8, r+12 at column c. Those quarters are stacked in LREG0's rows, out of reach - which is
 * what SFPTRANSP is for. It spreads them into row 0 of LREG0-3, where the ordinary fold finishes.
 *
 * @tparam POOL_TYPE: Reduction operator, values = <SUM/AVG/MAX>
 * @tparam MODE: SFPLOAD/SFPSTORE format-select code, @ref reduce_sfpmem_mode
 * @tparam IS_INT: Whether operands are integers
 * @tparam TOP_FACE_ADDR: Dest address of the top face of the pair, values = <0/REDUCE_FACE_STRIDE>
 * @tparam COLUMN_OFFSET: Column parity selector, values = <EVEN_COL/ODD_COL>
 * @note Writes the totals to row 0 of the top face; rows 1-3 keep the transpose's leftovers, which
 *       is fine - row 0 is all a column reduce's consumers read.
 */
template <PoolType POOL_TYPE, std::uint32_t MODE, bool IS_INT, std::uint32_t TOP_FACE_ADDR, std::uint32_t COLUMN_OFFSET>
inline void reduce_col_group() {
    constexpr std::uint32_t TOP = TOP_FACE_ADDR + COLUMN_OFFSET;
    constexpr std::uint32_t BOT = TOP + REDUCE_FACE_PAIR_STRIDE;

    // Top face rows 0-15 into LREG0-3, the face below it into LREG4-7.
    TTI_SFPLOAD(p_sfpu::LREG0, MODE, ADDR_MOD_7, 0 /* done */, TOP + 0 * REDUCE_ROWS_PER_LOAD);
    TTI_SFPLOAD(p_sfpu::LREG1, MODE, ADDR_MOD_7, 0 /* done */, TOP + 1 * REDUCE_ROWS_PER_LOAD);
    TTI_SFPLOAD(p_sfpu::LREG2, MODE, ADDR_MOD_7, 0 /* done */, TOP + 2 * REDUCE_ROWS_PER_LOAD);
    TTI_SFPLOAD(p_sfpu::LREG3, MODE, ADDR_MOD_7, 0 /* done */, TOP + 3 * REDUCE_ROWS_PER_LOAD);
    TTI_SFPLOAD(p_sfpu::LREG4, MODE, ADDR_MOD_7, 0 /* done */, BOT + 0 * REDUCE_ROWS_PER_LOAD);
    TTI_SFPLOAD(p_sfpu::LREG5, MODE, ADDR_MOD_7, 0 /* done */, BOT + 1 * REDUCE_ROWS_PER_LOAD);
    TTI_SFPLOAD(p_sfpu::LREG6, MODE, ADDR_MOD_7, 0 /* done */, BOT + 2 * REDUCE_ROWS_PER_LOAD);
    TTI_SFPLOAD(p_sfpu::LREG7, MODE, ADDR_MOD_7, 0 /* done */, BOT + 3 * REDUCE_ROWS_PER_LOAD);

    // Fold each bank down to its first register. The two chains are independent, so interleaving
    // them means no fold ever immediately follows the one it depends on.
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG2, p_sfpu::LREG3>();
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG6, p_sfpu::LREG7>();
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG1, p_sfpu::LREG2>();
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG5, p_sfpu::LREG6>();
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG0, p_sfpu::LREG1>();
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG4, p_sfpu::LREG5>();

    // Both faces together: every lane of LREG0 now carries a quarter of its column.
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG0, p_sfpu::LREG4>();

    // Those four quarters are stacked in LREG0's four rows, and nothing can fold them in place.
    // SFPTRANSP spreads them across row 0 of LREG0-3, where the ordinary fold reaches them.
    TTI_SFPTRANSP;
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG2, p_sfpu::LREG3>();
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG1, p_sfpu::LREG2>();
    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG0, p_sfpu::LREG1>();

    if constexpr (POOL_TYPE == PoolType::AVG) {
        // A column always spans exactly 32 rows, so the divisor is a compile-time power of two.
        // It fits in the SFPMULI immediate, so no constant register is tied up for it.
        TTI_SFPMULI(REDUCE_RECIP_COL_EXTENT_FP16B, p_sfpu::LREG0, 0 /* instr_mod1 */);
    }

    TTI_SFPSTORE(p_sfpu::LREG0, MODE, ADDR_MOD_7, 0 /* done */, TOP);
}

/**
 * @brief Column-reduce one whole 32x32 Dest tile in place.
 *
 * Walks the tile's four column groups - both parities of the two top faces - which between them
 * cover all 32 columns of row 0.
 *
 * @tparam POOL_TYPE: Reduction operator, values = <SUM/AVG/MAX>
 * @tparam FORMAT: Math-side data format of the operand
 */
template <PoolType POOL_TYPE, DataFormat FORMAT>
inline void reduce_col_tile() {
    constexpr std::uint32_t MODE = reduce_sfpmem_mode<FORMAT>();
    constexpr bool IS_INT = reduce_is_int_format<FORMAT>();
    constexpr std::uint32_t FACE_1 = REDUCE_FACE_STRIDE;

    reduce_col_group<POOL_TYPE, MODE, IS_INT, 0, p_sfpu::col_offset::EVEN_COL>();
    reduce_col_group<POOL_TYPE, MODE, IS_INT, 0, p_sfpu::col_offset::ODD_COL>();
    reduce_col_group<POOL_TYPE, MODE, IS_INT, FACE_1, p_sfpu::col_offset::EVEN_COL>();
    reduce_col_group<POOL_TYPE, MODE, IS_INT, FACE_1, p_sfpu::col_offset::ODD_COL>();
}

// ============================================================================
// Row reduce (REDUCE_ROW): collapse a tile row's columns onto column 0
// ============================================================================

/**
 * @brief Fold the accumulator's 8 SFPU column instances together, in place.
 *
 * Each of the 8 columns holds a partial total, but the columns are independent lanes and
 * SFPSHFT2's rotate is the only way to move a value between them. So: rotate a copy by 4 and fold,
 * then by 2, then by 1. After three stages every column holds the total of all eight, so the store
 * that follows is correct whichever lane the packer reads.
 *
 * @tparam POOL_TYPE: Reduction operator, values = <SUM/AVG/MAX>
 * @tparam IS_INT: Whether operands are integers
 * @note An SFPNOP separates each rotate from its reader: SFPSHFT2 takes 2 cycles and this kernel
 *       does not assume the pipeline interlocks a dependent read.
 */
template <PoolType POOL_TYPE, bool IS_INT>
inline void reduce_row_fold_columns() {
#pragma GCC unroll 1
    for (std::uint32_t distance = REDUCE_SFPU_COLUMNS / 2; distance > 0; distance >>= 1) {
        TTI_SFPMOV(REDUCE_ACC_REG, REDUCE_ROT_REG, 0 /* instr_mod1: plain copy */);
        for (std::uint32_t step = 0; step < distance; step++) {
            TTI_SFPSHFT2(0 /* imm12 */, REDUCE_ROT_REG, REDUCE_ROT_REG, REDUCE_SFPSHFT2_ROTATE);
            TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);
        }
        reduce_combine<POOL_TYPE, IS_INT, REDUCE_ACC_REG, REDUCE_ROT_REG>();
    }
}

/**
 * @brief Row-reduce a block of tiles held in Dest, writing each row's total to its column 0.
 *
 * Works through Dest four tile rows at a time (a "row quad"). Per quad, the accumulator sweeps
 * left to right over the tile row's @p block_ct_dim tiles, then the quad's 8 SFPU columns fold
 * together and the total is stored to column 0 of the tile row's first tile.
 *
 * @tparam POOL_TYPE: Reduction operator, values = <SUM/AVG/MAX>
 * @tparam FORMAT: Math-side data format of the operand
 * @param block_ct_dim: Tiles per tile row, i.e. the width being reduced
 * @param block_rt_dim: Tile rows in the block
 * @note The whole block must already be in Dest. A row's total spans every tile of its tile row,
 *       so unlike a column reduce this cannot run one Dest-sized batch at a time - there would be
 *       nowhere to keep the partial totals across the handoff to PACK.
 * @note AVG expects @ref reduce_row_load_reciprocal to have put 1/num_cols in REDUCE_RECIP_REG,
 *       which this leaves untouched.
 */
template <PoolType POOL_TYPE, DataFormat FORMAT>
inline void reduce_row_block(const std::uint32_t block_ct_dim, const std::uint32_t block_rt_dim) {
    constexpr std::uint32_t MODE = reduce_sfpmem_mode<FORMAT>();
    constexpr bool IS_INT = reduce_is_int_format<FORMAT>();

    for (std::uint32_t rt = 0; rt < block_rt_dim; rt++) {
        const std::uint32_t row_base = rt * block_ct_dim * REDUCE_TILE_STRIDE;

        for (std::uint32_t face_pair = 0; face_pair < REDUCE_FACE_PAIRS; face_pair++) {
            for (std::uint32_t quad = 0; quad < REDUCE_QUADS_PER_FACE_PAIR; quad++) {
                // Four tile rows: both parities of the left face and of the face beside it.
                const std::uint32_t quad_offset = (face_pair * REDUCE_FACE_PAIR_STRIDE) + (quad * REDUCE_ROWS_PER_LOAD);

                for (std::uint32_t ct = 0; ct < block_ct_dim; ct++) {
                    const std::uint32_t tile_base = row_base + (ct * REDUCE_TILE_STRIDE) + quad_offset;

                    TT_SFPLOAD(p_sfpu::LREG0, MODE, ADDR_MOD_7, 0 /* done */, tile_base + p_sfpu::col_offset::EVEN_COL);
                    TT_SFPLOAD(p_sfpu::LREG1, MODE, ADDR_MOD_7, 0 /* done */, tile_base + p_sfpu::col_offset::ODD_COL);
                    TT_SFPLOAD(
                        p_sfpu::LREG2,
                        MODE,
                        ADDR_MOD_7,
                        0 /* done */,
                        tile_base + REDUCE_FACE_STRIDE + p_sfpu::col_offset::EVEN_COL);
                    TT_SFPLOAD(
                        p_sfpu::LREG3,
                        MODE,
                        ADDR_MOD_7,
                        0 /* done */,
                        tile_base + REDUCE_FACE_STRIDE + p_sfpu::col_offset::ODD_COL);

                    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG0, p_sfpu::LREG1>();
                    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG2, p_sfpu::LREG3>();
                    reduce_combine<POOL_TYPE, IS_INT, p_sfpu::LREG0, p_sfpu::LREG2>();

                    if (ct == 0) {
                        // Start the accumulator from the first tile instead of an identity value.
                        // MAX would otherwise need a per-format -infinity to start from.
                        TTI_SFPMOV(p_sfpu::LREG0, REDUCE_ACC_REG, 0 /* instr_mod1: plain copy */);
                    } else {
                        reduce_combine<POOL_TYPE, IS_INT, REDUCE_ACC_REG, p_sfpu::LREG0>();
                    }
                }

                reduce_row_fold_columns<POOL_TYPE, IS_INT>();

                if constexpr (POOL_TYPE == PoolType::AVG) {
                    // dest = a * b + c with c = 0.0: scale by the preloaded 1/num_cols.
                    TTI_SFPMUL(
                        REDUCE_ACC_REG,
                        REDUCE_RECIP_REG,
                        p_sfpu::LCONST_0,
                        REDUCE_ACC_REG,
                        0 /* instr_mod1: no negation */);
                }

                // Column 0 of these four tile rows lives in the first tile's left face, even parity.
                TT_SFPSTORE(
                    REDUCE_ACC_REG,
                    MODE,
                    ADDR_MOD_7,
                    0 /* done */,
                    row_base + quad_offset + p_sfpu::col_offset::EVEN_COL);
            }
        }
    }
}

/**
 * @brief Put 1/num_cols into REDUCE_RECIP_REG for the row AVG scaling.
 *
 * A row's divisor is the runtime column count and need not be a power of two, so unlike the column
 * AVG it will not fit an SFPMULI immediate and needs a register. Computed on the RISC-V side and
 * loaded 16 bits at a time, which is all SFPLOADI writes per instruction.
 *
 * @param num_cols: Columns each row total spans
 * @note Call once per @ref calculate_reduce, before @ref reduce_row_block, which keeps the register
 *       intact for the whole block.
 */
inline void reduce_row_load_reciprocal(const std::uint32_t num_cols) {
    const float reciprocal = 1.0f / static_cast<float>(num_cols);
    const std::uint32_t bits = __builtin_bit_cast(std::uint32_t, reciprocal);

    TT_SFPLOADI(REDUCE_RECIP_REG, sfpi::SFPLOADI_MOD0_LOWER, bits & 0xFFFF);
    TT_SFPLOADI(REDUCE_RECIP_REG, sfpi::SFPLOADI_MOD0_UPPER, bits >> 16);
}

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Prepare the math thread's SFPU state for a run of reduce calls.
 *
 * Only one thing to do: zero the RWC counters. The reduce addresses Dest with immediates measured
 * from the base @ref _llk_math_eltwise_sfpu_start_ programs, so they must start from zero.
 *
 * The rest - SFPU config register, ADDR_MOD_7 - is already set up by
 * @ref _llk_math_eltwise_sfpu_init_. No reduce path claims a programmable constant register or a
 * replay slot, so this cannot disturb a neighbouring op's setup, or be disturbed by one.
 *
 * @tparam POOL_TYPE: Reduction operator, values = <SUM/AVG/MAX>
 * @tparam FORMAT: Math-side data format of the operand
 * @tparam IS_FP32_DEST_ACC_EN: Whether Dest holds 32-bit words
 * @param block_ct_dim: Unused; nothing here depends on the block width. Present so the signature
 *        matches Blackhole/Wormhole, which the shared Compute API @c sfpu_reduce_init forwards a
 *        runtime argument to.
 * @note Call after @ref _llk_math_eltwise_sfpu_init_ and before @ref calculate_reduce.
 */
template <PoolType POOL_TYPE, DataFormat FORMAT, bool IS_FP32_DEST_ACC_EN>
inline void init_reduce([[maybe_unused]] const std::uint32_t block_ct_dim = 1) {
    static_assert(
        is_supported_reduce_format(FORMAT), "Unsupported reduce format: expected Float32, Float16_b, Float16 or Int32");
    static_assert(
        POOL_TYPE == PoolType::SUM || POOL_TYPE == PoolType::AVG || POOL_TYPE == PoolType::MAX,
        "Unsupported pool_type: Quasar PoolType provides SUM, AVG and MAX");

    math::_reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Reduce Dest along one axis with the given pooling operator.
 *
 * REDUCE_COL folds a tile's 32 rows onto row 0. A column sits inside one tile, so tiles are
 * independent: call once per tile, and a large reduce can be fed to Dest one batch at a time.
 *
 * REDUCE_ROW folds a tile row's columns onto column 0. A row spans every tile in its tile row, so
 * call once for the whole block - which must be resident in Dest.
 *
 * @tparam POOL_TYPE: Reduction operator, values = <SUM/AVG/MAX>
 * @tparam REDUCE_DIM: Axis to collapse, values = <REDUCE_COL/REDUCE_ROW>
 * @tparam FORMAT: Math-side data format of the operand
 * @tparam IS_FP32_DEST_ACC_EN: Whether Dest holds 32-bit words
 * @param block_ct_dim: Tiles per tile row; the width a row reduce spans (REDUCE_ROW only)
 * @param block_rt_dim: Tile rows in the block (REDUCE_ROW only)
 * @note Run under VectorMode::RC_custom. This kernel walks Dest itself instead of being invoked
 *       once per face, and REDUCE_ROW deliberately reaches past the tile the caller based it on.
 * @note Call @ref init_reduce before this.
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIM, DataFormat FORMAT, bool IS_FP32_DEST_ACC_EN>
inline void calculate_reduce(
    [[maybe_unused]] const std::uint32_t block_ct_dim = 1, [[maybe_unused]] const std::uint32_t block_rt_dim = 1) {
    static_assert(
        is_supported_reduce_format(FORMAT), "Unsupported reduce format: expected Float32, Float16_b, Float16 or Int32");
    static_assert(
        REDUCE_DIM == ReduceDim::REDUCE_COL || REDUCE_DIM == ReduceDim::REDUCE_ROW,
        "Unsupported reduce_dim: expected REDUCE_COL or REDUCE_ROW");
    // Averaging integers has to round the quotient, and how to round is the caller's decision -
    // not one this kernel should make silently.
    static_assert(
        !(POOL_TYPE == PoolType::AVG && reduce_is_int_format<FORMAT>()),
        "Integer AVG reduce is not supported: reduce as a float format and typecast the result");

    if constexpr (REDUCE_DIM == ReduceDim::REDUCE_COL) {
        reduce_col_tile<POOL_TYPE, FORMAT>();
    } else {
        if constexpr (POOL_TYPE == PoolType::AVG) {
            reduce_row_load_reciprocal(block_ct_dim * REDUCE_COLS_PER_TILE);
        }
        reduce_row_block<POOL_TYPE, FORMAT>(block_ct_dim, block_rt_dim);
    }
}

}  // namespace sfpu
}  // namespace ckernel
