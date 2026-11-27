// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// One SFPLOAD covers four face rows x 8 datums; address bit 1 picks the even or odd columns.
constexpr std::uint32_t BCAST_DEST_TILE_STRIDE = 1U << trisc::get_dest_tile_size_log2(trisc::DstTileShape::Tile32x32);
constexpr std::uint32_t BCAST_FACE_STRIDE = FACE_R_DIM;
constexpr std::uint32_t BCAST_FACE_PAIR_STRIDE = 2 * FACE_R_DIM;
constexpr std::uint32_t BCAST_ROW_BAND_STRIDE = 4;  // face rows per SFPLOAD
constexpr std::uint32_t BCAST_BANDS_PER_FACE_PAIR = FACE_R_DIM / BCAST_ROW_BAND_STRIDE;
constexpr std::uint32_t BCAST_FACE_PAIRS = 2;

constexpr std::uint32_t BCAST_LEFT_EVEN = p_sfpu::col_offset::EVEN_COL;
constexpr std::uint32_t BCAST_LEFT_ODD = p_sfpu::col_offset::ODD_COL;
constexpr std::uint32_t BCAST_RIGHT_EVEN = BCAST_FACE_STRIDE + p_sfpu::col_offset::EVEN_COL;
constexpr std::uint32_t BCAST_RIGHT_ODD = BCAST_FACE_STRIDE + p_sfpu::col_offset::ODD_COL;

constexpr std::uint32_t BCAST_SFPSHFT2_MOD1_ROTATE_COLS = 3;           // SFPU col X -> X+1, col 7 wraps to col 0
constexpr std::uint32_t BCAST_SFPSHFT2_MOD1_SHIFT_COLS_ZERO_FILL = 4;  // SFPU col X -> X+1, col 0 <- 0
constexpr std::uint32_t BCAST_MAD_MOD1_NEGATE_C = 0x2;

constexpr std::uint32_t BCAST_COL_LREG_BCAST = p_sfpu::LREG0;
constexpr std::uint32_t BCAST_COL_LREG_TMP = p_sfpu::LREG2;
constexpr std::uint32_t BCAST_COL_LREG_MASK = p_sfpu::LREG6;
constexpr std::uint32_t BCAST_COL_LREG_DATA0 = p_sfpu::LREG1;
constexpr std::uint32_t BCAST_COL_LREG_DATA1 = p_sfpu::LREG3;
constexpr std::uint32_t BCAST_COL_LREG_DATA2 = p_sfpu::LREG4;
constexpr std::uint32_t BCAST_COL_LREG_DATA3 = p_sfpu::LREG5;

// ROW: hoisted bcast rows in LREG4-7; only LREG4 is read after the transpose
constexpr std::uint32_t BCAST_ROW_LREG_DATA = p_sfpu::LREG0;
constexpr std::uint32_t BCAST_ROW_LREG_BCAST = p_sfpu::LREG4;

template <BinaryOp BINOP, std::uint32_t DST, std::uint32_t BCAST>
inline void binary_bcast_op_() {
    static_assert(
        BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL,
        "binary_bcast only supports ADD, SUB and MUL");

    if constexpr (BINOP == BinaryOp::ADD) {
        TTI_SFPADD(p_sfpu::LCONST_1, DST, BCAST, DST, 0 /*instr_mod1*/);
    } else if constexpr (BINOP == BinaryOp::SUB) {
        TTI_SFPADD(p_sfpu::LCONST_1, DST, BCAST, DST, BCAST_MAD_MOD1_NEGATE_C);
    } else {
        TTI_SFPMUL(DST, BCAST, p_sfpu::LCONST_0, DST, 0 /*instr_mod1*/);
    }
}

// Col-0 bit mask. Isolating col 0 bitwise (not by multiplying by 0) keeps Inf/NaN in the other
// columns out of the broadcast (0 * Inf = NaN) and keeps the sign of a -0.0 in col 0.
inline void binary_bcast_build_col0_mask_() {
    TTI_SFPLOADI(BCAST_COL_LREG_TMP, sfpi::SFPLOADI_MOD0_SHORT, 0xFFFF);
    TTI_SFPSHFT2(0 /*imm12*/, BCAST_COL_LREG_TMP, BCAST_COL_LREG_MASK, BCAST_SFPSHFT2_MOD1_SHIFT_COLS_ZERO_FILL);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);  // 2-cycle global shift
    TTI_SFPNOT(BCAST_COL_LREG_MASK, BCAST_COL_LREG_MASK);
}

// Replay slots: COL and ROW never run in the same call, so both start at slot 0
constexpr std::uint32_t BCAST_REPLAY_DEPTH = 32;
constexpr std::uint32_t BCAST_COL_FOLD_REPLAY_SLOT = 0;
constexpr std::uint32_t BCAST_COL_FOLD_REPLAY_LEN = 9;
constexpr std::uint32_t BCAST_COL_OP_REPLAY_SLOT = BCAST_COL_FOLD_REPLAY_SLOT + BCAST_COL_FOLD_REPLAY_LEN;
constexpr std::uint32_t BCAST_COL_OP_REPLAY_LEN = 5;
constexpr std::uint32_t BCAST_ROW_OP_REPLAY_SLOT = 0;
constexpr std::uint32_t BCAST_ROW_OP_REPLAY_LEN = 6;
constexpr std::uint32_t BCAST_COL_REPLAY_SLOTS_USED = BCAST_COL_OP_REPLAY_SLOT + BCAST_COL_OP_REPLAY_LEN;
constexpr std::uint32_t BCAST_ROW_REPLAY_SLOTS_USED = BCAST_ROW_OP_REPLAY_SLOT + BCAST_ROW_OP_REPLAY_LEN;
static_assert(BCAST_COL_REPLAY_SLOTS_USED <= BCAST_REPLAY_DEPTH, "COL replay bodies must fit");
static_assert(BCAST_ROW_REPLAY_SLOTS_USED <= BCAST_REPLAY_DEPTH, "ROW replay body must fit");
static_assert(
    BCAST_COL_REPLAY_SLOTS_USED == 14 && BCAST_ROW_REPLAY_SLOTS_USED == 6,
    "update the replay-slot ranges in the init_binary_bcast doc");

template <std::uint32_t SLOT, std::uint32_t LEN>
inline void binary_bcast_replay_() {
    TTI_REPLAY(SLOT, LEN, 0 /*last*/, 0 /*set_mutex*/, 0 /*execute_while_loading*/, 0 /*load_mode*/);
}

// Mask + fold stages 1-2 (distances 1, 2). Cols 1..7 are bit zero after the AND, so each OR copies
// col 0 bit-exactly.
inline void binary_bcast_col_fold_() {
    constexpr std::uint32_t B = BCAST_COL_LREG_BCAST;
    constexpr std::uint32_t T = BCAST_COL_LREG_TMP;
    constexpr std::uint32_t M = BCAST_COL_LREG_MASK;
    constexpr std::uint32_t ROT = BCAST_SFPSHFT2_MOD1_ROTATE_COLS;

    TTI_SFPAND(M, B);

    TTI_SFPSHFT2(0 /*imm12*/, B, T, ROT);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);
    TTI_SFPOR(T, B);

    TTI_SFPSHFT2(0 /*imm12*/, B, T, ROT);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);
    TTI_SFPSHFT2(0 /*imm12*/, T, T, ROT);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);
    TTI_SFPOR(T, B);
}

// Last fold OR of stage 3, then the binop on all four data slots
template <BinaryOp BINOP>
inline void binary_bcast_col_op_() {
    constexpr std::uint32_t B = BCAST_COL_LREG_BCAST;

    TTI_SFPOR(BCAST_COL_LREG_TMP, B);  // B = bcast[row][0] in all 8 cols

    binary_bcast_op_<BINOP, BCAST_COL_LREG_DATA0, B>();
    binary_bcast_op_<BINOP, BCAST_COL_LREG_DATA1, B>();
    binary_bcast_op_<BINOP, BCAST_COL_LREG_DATA2, B>();
    binary_bcast_op_<BINOP, BCAST_COL_LREG_DATA3, B>();
}

inline void binary_bcast_col_band_(
    const std::uint32_t bcast_addr, const std::uint32_t data_addr, const std::uint32_t out_addr) {
    constexpr std::uint32_t B = BCAST_COL_LREG_BCAST;
    constexpr std::uint32_t T = BCAST_COL_LREG_TMP;
    constexpr std::uint32_t D0 = BCAST_COL_LREG_DATA0;
    constexpr std::uint32_t D1 = BCAST_COL_LREG_DATA1;
    constexpr std::uint32_t D2 = BCAST_COL_LREG_DATA2;
    constexpr std::uint32_t D3 = BCAST_COL_LREG_DATA3;
    constexpr std::uint32_t ROT = BCAST_SFPSHFT2_MOD1_ROTATE_COLS;

    // SFPU col 0 = tile col 0 of the band's 4 rows
    TT_SFPLOAD(B, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, bcast_addr + BCAST_LEFT_EVEN /*dest_reg_addr*/);
    // binary_bcast_col_fold_
    binary_bcast_replay_<BCAST_COL_FOLD_REPLAY_SLOT, BCAST_COL_FOLD_REPLAY_LEN>();

    // Stage 3: fold distance 4; the data loads fill each shift's latency slot
    TTI_SFPSHFT2(0 /*imm12*/, B, T, ROT);
    TT_SFPLOAD(D0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_LEFT_EVEN /*dest_reg_addr*/);
    TTI_SFPSHFT2(0 /*imm12*/, T, T, ROT);
    TT_SFPLOAD(D1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_LEFT_ODD /*dest_reg_addr*/);
    TTI_SFPSHFT2(0 /*imm12*/, T, T, ROT);
    TT_SFPLOAD(D2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_RIGHT_EVEN /*dest_reg_addr*/);
    TTI_SFPSHFT2(0 /*imm12*/, T, T, ROT);
    TT_SFPLOAD(D3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_RIGHT_ODD /*dest_reg_addr*/);
    // binary_bcast_col_op_
    binary_bcast_replay_<BCAST_COL_OP_REPLAY_SLOT, BCAST_COL_OP_REPLAY_LEN>();

    TT_SFPSTORE(D0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_LEFT_EVEN /*dest_reg_addr*/);
    TT_SFPSTORE(D1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_LEFT_ODD /*dest_reg_addr*/);
    TT_SFPSTORE(D2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_RIGHT_EVEN /*dest_reg_addr*/);
    TT_SFPSTORE(D3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_RIGHT_ODD /*dest_reg_addr*/);
}

// Transpose to row order, binop against hoisted bcast row 0, transpose back
template <BinaryOp BINOP>
inline void binary_bcast_row_op_() {
    constexpr std::uint32_t D = BCAST_ROW_LREG_DATA;
    constexpr std::uint32_t B = BCAST_ROW_LREG_BCAST;

    TTI_SFPTRANSP;

    // Never write LREG4-7 so the next transpose restores the hoisted bcast rows
    binary_bcast_op_<BINOP, D + 0 /*DST*/, B>();
    binary_bcast_op_<BINOP, D + 1 /*DST*/, B>();
    binary_bcast_op_<BINOP, D + 2 /*DST*/, B>();
    binary_bcast_op_<BINOP, D + 3 /*DST*/, B>();

    TTI_SFPTRANSP;
}

// Bcast rows 0-3 into LREG4-7, hoisted for the whole tile
inline void binary_bcast_row_hoist_(const std::uint32_t bcast_addr) {
    constexpr std::uint32_t B0 = BCAST_ROW_LREG_BCAST + 0;
    constexpr std::uint32_t B1 = BCAST_ROW_LREG_BCAST + 1;
    constexpr std::uint32_t B2 = BCAST_ROW_LREG_BCAST + 2;
    constexpr std::uint32_t B3 = BCAST_ROW_LREG_BCAST + 3;

    TT_SFPLOAD(B0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, bcast_addr + BCAST_LEFT_EVEN /*dest_reg_addr*/);
    TT_SFPLOAD(B1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, bcast_addr + BCAST_LEFT_ODD /*dest_reg_addr*/);
    TT_SFPLOAD(B2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, bcast_addr + BCAST_RIGHT_EVEN /*dest_reg_addr*/);
    TT_SFPLOAD(B3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, bcast_addr + BCAST_RIGHT_ODD /*dest_reg_addr*/);
}

inline void binary_bcast_row_band_(const std::uint32_t data_addr, const std::uint32_t out_addr) {
    constexpr std::uint32_t D0 = BCAST_ROW_LREG_DATA + 0;
    constexpr std::uint32_t D1 = BCAST_ROW_LREG_DATA + 1;
    constexpr std::uint32_t D2 = BCAST_ROW_LREG_DATA + 2;
    constexpr std::uint32_t D3 = BCAST_ROW_LREG_DATA + 3;

    TT_SFPLOAD(D0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_LEFT_EVEN /*dest_reg_addr*/);
    TT_SFPLOAD(D1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_LEFT_ODD /*dest_reg_addr*/);
    TT_SFPLOAD(D2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_RIGHT_EVEN /*dest_reg_addr*/);
    TT_SFPLOAD(D3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_RIGHT_ODD /*dest_reg_addr*/);

    // binary_bcast_row_op_
    binary_bcast_replay_<BCAST_ROW_OP_REPLAY_SLOT, BCAST_ROW_OP_REPLAY_LEN>();

    TT_SFPSTORE(D0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_LEFT_EVEN /*dest_reg_addr*/);
    TT_SFPSTORE(D1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_LEFT_ODD /*dest_reg_addr*/);
    TT_SFPSTORE(D2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_RIGHT_EVEN /*dest_reg_addr*/);
    TT_SFPSTORE(D3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_RIGHT_ODD /*dest_reg_addr*/);
}

// The col-0 mask in LREG6 and the fold replay (slots 0..8), the state every column broadcast reads.
inline void binary_bcast_col_fold_init_() {
    binary_bcast_build_col0_mask_();
    load_replay_buf<BCAST_COL_FOLD_REPLAY_SLOT, BCAST_COL_FOLD_REPLAY_LEN>([] { binary_bcast_col_fold_(); });
}

/**
 * @brief Records the per-band replay bodies once (and, for COL, builds the col-0 mask in LREG6).
 *
 * @note Re-run after any op that records into math replay bank 0 slots 0..13 (COL) / 0..5 (ROW) or,
 *       for COL, writes LREG6 (a ROW binary_bcast call does both).
 */
template <BinaryOp BINOP, BroadcastType BCAST_DIM>
inline void init_binary_bcast() {
    static_assert(
        BCAST_DIM == BroadcastType::COL || BCAST_DIM == BroadcastType::ROW, "binary_bcast only supports COL and ROW");

    if constexpr (BCAST_DIM == BroadcastType::COL) {
        binary_bcast_col_fold_init_();
        load_replay_buf<BCAST_COL_OP_REPLAY_SLOT, BCAST_COL_OP_REPLAY_LEN>([] { binary_bcast_col_op_<BINOP>(); });
    } else {
        load_replay_buf<BCAST_ROW_OP_REPLAY_SLOT, BCAST_ROW_OP_REPLAY_LEN>([] { binary_bcast_row_op_<BINOP>(); });
    }
}

/**
 * @brief out = data OP bcast[r][0] (COL) or data OP bcast[0][c] (ROW) over one whole 32x32 Dest tile.
 *
 * @tparam BINOP ADD/SUB/MUL, float operands only.
 * @note Call once per tile under VectorMode::None, after @ref init_binary_bcast with the same template
 *       args. Overwrites LREG0-5 (COL) or LREG0-7 (ROW); the output tile may alias either input.
 */
template <BinaryOp BINOP, BroadcastType BCAST_DIM>
inline void calculate_binary_bcast(
    const std::uint32_t dst_index_data, const std::uint32_t dst_index_bcast, const std::uint32_t dst_index_out) {
    static_assert(
        BCAST_DIM == BroadcastType::COL || BCAST_DIM == BroadcastType::ROW, "binary_bcast only supports COL and ROW");

    const std::uint32_t data_base = dst_index_data * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t bcast_base = dst_index_bcast * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t out_base = dst_index_out * BCAST_DEST_TILE_STRIDE;

    if constexpr (BCAST_DIM == BroadcastType::COL) {
        for (std::uint32_t face_pair = 0; face_pair < BCAST_FACE_PAIRS; face_pair++) {
            for (std::uint32_t band = 0; band < BCAST_BANDS_PER_FACE_PAIR; band++) {
                const std::uint32_t offset = face_pair * BCAST_FACE_PAIR_STRIDE + band * BCAST_ROW_BAND_STRIDE;
                binary_bcast_col_band_(
                    bcast_base + offset /*bcast_addr*/,
                    data_base + offset /*data_addr*/,
                    out_base + offset /*out_addr*/);
            }
        }
    } else {
        binary_bcast_row_hoist_(bcast_base);

        for (std::uint32_t face_pair = 0; face_pair < BCAST_FACE_PAIRS; face_pair++) {
            for (std::uint32_t band = 0; band < BCAST_BANDS_PER_FACE_PAIR; band++) {
                const std::uint32_t offset = face_pair * BCAST_FACE_PAIR_STRIDE + band * BCAST_ROW_BAND_STRIDE;
                binary_bcast_row_band_(data_base + offset /*data_addr*/, out_base + offset /*out_addr*/);
            }
        }
    }
}

// Compute API entry points (sfpu_{add,sub,mul}_bcast_{col,row}, as named on Blackhole / Wormhole).
// There one init per broadcast dim serves ADD, SUB and MUL; here the binop is recorded into the
// replay body, so each call records it (init_binary_bcast) before running the tile.

template <BroadcastType BCAST_DIM>
inline void _sfpu_binary_bcast_init_() {
    static_assert(
        BCAST_DIM == BroadcastType::COL || BCAST_DIM == BroadcastType::ROW, "binary_bcast only supports COL and ROW");
}

template <BinaryOp BINOP, BroadcastType BCAST_DIM>
inline void _calculate_sfpu_binary_bcast_full_tile_(
    const std::uint32_t dst_index_data, const std::uint32_t dst_index_bcast, const std::uint32_t dst_index_out) {
    init_binary_bcast<BINOP, BCAST_DIM>();
    calculate_binary_bcast<BINOP, BCAST_DIM>(dst_index_data, dst_index_bcast, dst_index_out);
}

// Partial-tile column broadcasts (the Compute API's sfpu_*_bcast_col with VectorMode::R and sfpu_*_bcast_col_band):
// the full tile's per-band body over fewer bands. A band is 4 tile rows of the upper face pair (faces 0 and 1),
// all 32 columns; nothing outside the named bands is read or written.

/**
 * @brief out = data OP bcast[r][0] over tile rows 0..15 (faces 0 and 1); rows 16..31 are neither read nor written.
 *
 * @tparam BINOP ADD/SUB/MUL, float operands only.
 * @note Call once per tile under VectorMode::None. Records its replay bodies and lane mask first, as the full-tile
 *       entry does. Overwrites LREG0-6; the output tile may alias either input.
 */
template <BinaryOp BINOP>
inline void _calculate_sfpu_binary_bcast_col_rows_0_15_(
    const std::uint32_t dst_index_data, const std::uint32_t dst_index_bcast, const std::uint32_t dst_index_out) {
    init_binary_bcast<BINOP, BroadcastType::COL>();
    const std::uint32_t data_base = dst_index_data * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t bcast_base = dst_index_bcast * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t out_base = dst_index_out * BCAST_DEST_TILE_STRIDE;
    for (std::uint32_t band = 0; band < BCAST_BANDS_PER_FACE_PAIR; band++) {
        const std::uint32_t offset = band * BCAST_ROW_BAND_STRIDE;
        binary_bcast_col_band_(
            bcast_base + offset /*bcast_addr*/, data_base + offset /*data_addr*/, out_base + offset /*out_addr*/);
    }
}

/**
 * @brief out rows 4*data_band..+3 = data rows 4*data_band..+3 OP bcast[4*bcast_band + i][0], all 32 columns.
 *
 * @tparam BINOP ADD/SUB/MUL, float operands only.
 * @param data_band,bcast_band 0..3 (the Compute API checks the range).
 * @note Call once per band under VectorMode::None. Records its replay bodies and lane mask first. Overwrites
 *       LREG0-6.
 */
template <BinaryOp BINOP>
inline void _calculate_sfpu_binary_bcast_col_band_(
    const std::uint32_t dst_index_data,
    const std::uint32_t dst_index_bcast,
    const std::uint32_t dst_index_out,
    const std::uint32_t data_band,
    const std::uint32_t bcast_band) {
    init_binary_bcast<BINOP, BroadcastType::COL>();
    const std::uint32_t data_offset = data_band * BCAST_ROW_BAND_STRIDE;
    binary_bcast_col_band_(
        dst_index_bcast * BCAST_DEST_TILE_STRIDE + bcast_band * BCAST_ROW_BAND_STRIDE /*bcast_addr*/,
        dst_index_data * BCAST_DEST_TILE_STRIDE + data_offset /*data_addr*/,
        dst_index_out * BCAST_DEST_TILE_STRIDE + data_offset /*out_addr*/);
}

// Scalar broadcast from DEST: element [0, 0] spread to every lane of programmable constant LREG14 (sfpi's
// vConstFloatPrgm2). Shared by sfpu_bcast_scalar_load and the scalar normalisation.

// LREG14 as an operand (read-only from instructions; written through SFPCONFIG only).
constexpr std::uint32_t BCAST_SCALAR_LREG_PARKED = 14;
static_assert(BCAST_SCALAR_LREG_PARKED == p_sfpconfig::PROG_CONST_LREG14, "SFPCONFIG index and LREG index agree");

/**
 * @brief Spread element [0, 0] of a DEST tile across the 8 SFPU columns of LREG0 row 0.
 *
 * Bit-exact (-0.0 and NaN payloads survive): the column broadcast's col-0 bit mask and OR fold, with the fold's
 * last distance (4) spelled out. Rows 1..3 end up holding their own column-0 values; only row 0 is the scalar, and
 * row 0 is what SFPCONFIG reads. Clobbers LREG0, LREG2 and LREG6 (the column-broadcast lane mask).
 *
 * @param scalar_addr DEST address of the tile's first vector (tile index * BCAST_DEST_TILE_STRIDE).
 */
inline void binary_bcast_scalar_spread_row0_(const std::uint32_t scalar_addr) {
    constexpr std::uint32_t B = BCAST_COL_LREG_BCAST;
    constexpr std::uint32_t T = BCAST_COL_LREG_TMP;
    constexpr std::uint32_t ROT = BCAST_SFPSHFT2_MOD1_ROTATE_COLS;

    TT_SFPLOAD(B, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, scalar_addr + BCAST_LEFT_EVEN /*dest_reg_addr*/);
    binary_bcast_build_col0_mask_();
    // Mask and fold distances 1 and 2.
    binary_bcast_col_fold_();
    // Fold distance 4: four single-column rotates, each read the cycle after a NOP (2-cycle global shift).
    TTI_SFPSHFT2(0 /*imm12*/, B, T, ROT);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);
    TTI_SFPSHFT2(0 /*imm12*/, T, T, ROT);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);
    TTI_SFPSHFT2(0 /*imm12*/, T, T, ROT);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);
    TTI_SFPSHFT2(0 /*imm12*/, T, T, ROT);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);
    TTI_SFPOR(T, B);
}

/**
 * @brief Write row 0 of LREG0 to every row of programmable constant LREG14.
 *
 * SFPCONFIG with instr_mod1 = 0 takes its data from row 0 of LREG0, per SFPU column, and broadcasts it to the
 * register's four rows.
 */
inline void binary_bcast_park_lreg0_row0_() {
    TTI_SFPCONFIG(0 /*imm16_math*/, p_sfpconfig::PROG_CONST_LREG14 /*config_dest*/, 0 /*instr_mod1*/);
    // SFPCONFIG hazard: no instruction may issue the cycle after it.
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);
}

// Element [0, 0] of the tile at `scalar_addr` -> every lane of LREG14. Clobbers LREG0, LREG2 and LREG6.
inline void binary_bcast_scalar_to_lreg14_(const std::uint32_t scalar_addr) {
    binary_bcast_scalar_spread_row0_(scalar_addr);
    binary_bcast_park_lreg0_row0_();
}

// ============================================================================
// Fused normalisation: (data [+ residual] - mean) * inv_std, statistics broadcast from column 0 (or from [0][0])
// ============================================================================
//
// Per 4-row band, with the column broadcast's lane mask and fold replay (binary_bcast_col_fold_init_):
//   1. mean column -> LREG0 (fold), the data loaded in the stage-3 shift slots, LREG7 = -mean;
//   2. inv_std column -> LREG0 (fold), the data centred (+ LREG7) in the stage-3 shift slots;
//   3. data *= LREG0, stored.
// LREG budget as the column broadcast: 0 broadcast, 2 fold temp, 6 lane mask, 1/3/4/5 data, 7 the negated mean.
// Every value is an add then a multiply, each rounded, as on Blackhole.

constexpr std::uint32_t BCAST_NORM_LREG_NEG_MEAN = p_sfpu::LREG7;
constexpr std::uint32_t BCAST_SFPMOV_MOD1_NEGATE = 1;
constexpr std::uint32_t BCAST_BAND_SLOTS = 4;  // SFPLOADs per band: left / right face x even / odd columns
constexpr std::uint32_t BCAST_COL_DATA_LREG[BCAST_BAND_SLOTS] = {
    BCAST_COL_LREG_DATA0, BCAST_COL_LREG_DATA1, BCAST_COL_LREG_DATA2, BCAST_COL_LREG_DATA3};
constexpr std::uint32_t BCAST_BAND_SLOT_OFFSET[BCAST_BAND_SLOTS] = {
    BCAST_LEFT_EVEN, BCAST_LEFT_ODD, BCAST_RIGHT_EVEN, BCAST_RIGHT_ODD};

template <std::uint32_t K>
using bcast_slot_t = std::integral_constant<std::uint32_t, K>;

// LREG0 = the column broadcast of `col_addr`'s band. `slot(bcast_slot_t<k>)` runs in the k-th stage-3 shift latency
// slot (one instruction each, as the data loads of binary_bcast_col_band_).
template <typename Slot>
inline void binary_bcast_col_value_(const std::uint32_t col_addr, Slot&& slot) {
    constexpr std::uint32_t B = BCAST_COL_LREG_BCAST;
    constexpr std::uint32_t T = BCAST_COL_LREG_TMP;
    constexpr std::uint32_t ROT = BCAST_SFPSHFT2_MOD1_ROTATE_COLS;

    TT_SFPLOAD(B, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, col_addr + BCAST_LEFT_EVEN /*dest_reg_addr*/);
    // binary_bcast_col_fold_: mask and fold distances 1, 2
    binary_bcast_replay_<BCAST_COL_FOLD_REPLAY_SLOT, BCAST_COL_FOLD_REPLAY_LEN>();
    // Stage 3: fold distance 4
    TTI_SFPSHFT2(0 /*imm12*/, B, T, ROT);
    slot(bcast_slot_t<0>{});
    TTI_SFPSHFT2(0 /*imm12*/, T, T, ROT);
    slot(bcast_slot_t<1>{});
    TTI_SFPSHFT2(0 /*imm12*/, T, T, ROT);
    slot(bcast_slot_t<2>{});
    TTI_SFPSHFT2(0 /*imm12*/, T, T, ROT);
    slot(bcast_slot_t<3>{});
    TTI_SFPOR(T, B);  // B = col[row][0] in all 8 cols
}

// Stage-3 slot fillers.
struct bcast_load_data_slot_ {
    std::uint32_t addr;
    template <std::uint32_t K>
    void operator()(bcast_slot_t<K>) const {
        TT_SFPLOAD(
            BCAST_COL_DATA_LREG[K], p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, addr + BCAST_BAND_SLOT_OFFSET[K]);
    }
};

struct bcast_centre_slot_ {
    template <std::uint32_t K>
    void operator()(bcast_slot_t<K>) const {
        TTI_SFPADD(
            p_sfpu::LCONST_1,
            BCAST_COL_DATA_LREG[K],
            BCAST_NORM_LREG_NEG_MEAN,
            BCAST_COL_DATA_LREG[K],
            0 /*instr_mod1*/);
    }
};

struct bcast_nop_slot_ {
    template <std::uint32_t K>
    void operator()(bcast_slot_t<K>) const {
        TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);
    }
};

template <std::uint32_t... K>
inline void binary_bcast_load_band_(const std::uint32_t addr, std::integer_sequence<std::uint32_t, K...>) {
    (bcast_load_data_slot_{addr}(bcast_slot_t<K>{}), ...);
}

template <std::uint32_t... K>
inline void binary_bcast_centre_band_(std::integer_sequence<std::uint32_t, K...>) {
    (bcast_centre_slot_{}(bcast_slot_t<K>{}), ...);
}

// Data *= LREG `SCALE`, stored to the band at `out_addr`. The SFPMUL -> SFPSTORE distance is four, as in the band
// body.
template <std::uint32_t SCALE, std::uint32_t... K>
inline void binary_bcast_scale_store_band_(const std::uint32_t out_addr, std::integer_sequence<std::uint32_t, K...>) {
    (TTI_SFPMUL(BCAST_COL_DATA_LREG[K], SCALE, p_sfpu::LCONST_0, BCAST_COL_DATA_LREG[K], 0 /*instr_mod1*/), ...);
    (TT_SFPSTORE(
         BCAST_COL_DATA_LREG[K], p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_BAND_SLOT_OFFSET[K]),
     ...);
}

using bcast_band_slots_t = std::make_integer_sequence<std::uint32_t, BCAST_BAND_SLOTS>;

// LREG7 = -mean[row][0] of the band; `slot` (the data loads, or NOPs) fills the fold's stage-3 slots.
template <typename Slot>
inline void binary_bcast_neg_mean_(const std::uint32_t mean_addr, Slot&& slot) {
    binary_bcast_col_value_(mean_addr, slot);
    TTI_SFPMOV(BCAST_COL_LREG_BCAST, BCAST_NORM_LREG_NEG_MEAN, BCAST_SFPMOV_MOD1_NEGATE);
}

// With LREG7 = -mean and the band's data in LREG1/3/4/5: broadcast inv_std (centring the data in its shift slots),
// scale and store to `out_addr`. Leaves inv_std in LREG0.
inline void binary_bcast_normalize_finish_(const std::uint32_t inv_std_addr, const std::uint32_t out_addr) {
    binary_bcast_col_value_(inv_std_addr, bcast_centre_slot_{});
    binary_bcast_scale_store_band_<BCAST_COL_LREG_BCAST>(out_addr, bcast_band_slots_t{});
}

// The 8 bands of a tile: fn(offset), the offset of that band from every tile base.
template <typename Fn>
inline void binary_bcast_for_each_band_(Fn&& fn) {
    for (std::uint32_t face_pair = 0; face_pair < BCAST_FACE_PAIRS; face_pair++) {
        for (std::uint32_t band = 0; band < BCAST_BANDS_PER_FACE_PAIR; band++) {
            fn(face_pair * BCAST_FACE_PAIR_STRIDE + band * BCAST_ROW_BAND_STRIDE);
        }
    }
}

/**
 * @brief out = (data - mean[r][0]) * inv_std[r][0] over one whole fp32 tile.
 *
 * Column 0 of the statistics tiles is masked bitwise before the broadcast, so their other columns are never used.
 *
 * @note Call once per tile under VectorMode::None. Builds the lane mask and fold replay itself. Overwrites LREG0-7
 *       and math replay slots 0..8; the output tile may alias any input.
 */
inline void _calculate_sfpu_normalize_bcast_col_full_tile_(
    const std::uint32_t dst_index_data,
    const std::uint32_t dst_index_mean,
    const std::uint32_t dst_index_out,
    const std::uint32_t dst_index_inv_std) {
    binary_bcast_col_fold_init_();
    const std::uint32_t data_base = dst_index_data * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t mean_base = dst_index_mean * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t inv_std_base = dst_index_inv_std * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t out_base = dst_index_out * BCAST_DEST_TILE_STRIDE;
    binary_bcast_for_each_band_([&](const std::uint32_t offset) {
        binary_bcast_neg_mean_(mean_base + offset, bcast_load_data_slot_{data_base + offset});
        binary_bcast_normalize_finish_(inv_std_base + offset, out_base + offset);
    });
}

/**
 * @brief Two data tiles normalised in place with the same statistics: each band's mean and inv_std are broadcast
 *        once and applied to both tiles.
 *
 * @note As @ref _calculate_sfpu_normalize_bcast_col_full_tile_; `dst_index_out` is unused (both tiles in place).
 */
inline void _calculate_sfpu_normalize_bcast_col_two_tiles_(
    const std::uint32_t dst_index_data,
    const std::uint32_t dst_index_second_data,
    [[maybe_unused]] const std::uint32_t dst_index_out,
    const std::uint32_t dst_index_mean,
    const std::uint32_t dst_index_inv_std) {
    binary_bcast_col_fold_init_();
    const std::uint32_t first_base = dst_index_data * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t second_base = dst_index_second_data * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t mean_base = dst_index_mean * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t inv_std_base = dst_index_inv_std * BCAST_DEST_TILE_STRIDE;
    binary_bcast_for_each_band_([&](const std::uint32_t offset) {
        binary_bcast_neg_mean_(mean_base + offset, bcast_load_data_slot_{first_base + offset});
        binary_bcast_normalize_finish_(inv_std_base + offset, first_base + offset);
        // LREG7 (-mean) and LREG0 (inv_std) still hold this band's statistics.
        binary_bcast_load_band_(second_base + offset, bcast_band_slots_t{});
        binary_bcast_centre_band_(bcast_band_slots_t{});
        binary_bcast_scale_store_band_<BCAST_COL_LREG_BCAST>(second_base + offset, bcast_band_slots_t{});
    });
}

/**
 * @brief out = (data + residual - mean[r][0]) * inv_std[r][0] over one whole fp32 tile.
 *
 * @note As @ref _calculate_sfpu_normalize_bcast_col_full_tile_.
 */
inline void _calculate_sfpu_residual_normalize_bcast_col_full_tile_(
    const std::uint32_t dst_index_data,
    const std::uint32_t dst_index_residual,
    const std::uint32_t dst_index_out,
    const std::uint32_t dst_index_mean,
    const std::uint32_t dst_index_inv_std) {
    binary_bcast_col_fold_init_();
    const std::uint32_t data_base = dst_index_data * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t residual_base = dst_index_residual * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t mean_base = dst_index_mean * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t inv_std_base = dst_index_inv_std * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t out_base = dst_index_out * BCAST_DEST_TILE_STRIDE;
    binary_bcast_for_each_band_([&](const std::uint32_t offset) {
        // data += residual, the residual staged through LREG0 / LREG2 / LREG7 (free until the mean broadcast).
        const std::uint32_t residual_addr = residual_base + offset;
        binary_bcast_load_band_(data_base + offset, bcast_band_slots_t{});
        TT_SFPLOAD(
            BCAST_COL_LREG_BCAST, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, residual_addr + BCAST_LEFT_EVEN);
        TT_SFPLOAD(BCAST_COL_LREG_TMP, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, residual_addr + BCAST_LEFT_ODD);
        TT_SFPLOAD(
            BCAST_NORM_LREG_NEG_MEAN,
            p_sfpu::sfpmem::DEFAULT,
            ADDR_MOD_7,
            0 /*done*/,
            residual_addr + BCAST_RIGHT_EVEN);
        TTI_SFPADD(
            p_sfpu::LCONST_1, BCAST_COL_LREG_DATA0, BCAST_COL_LREG_BCAST, BCAST_COL_LREG_DATA0, 0 /*instr_mod1*/);
        TT_SFPLOAD(
            BCAST_COL_LREG_BCAST, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, residual_addr + BCAST_RIGHT_ODD);
        TTI_SFPADD(p_sfpu::LCONST_1, BCAST_COL_LREG_DATA1, BCAST_COL_LREG_TMP, BCAST_COL_LREG_DATA1, 0 /*instr_mod1*/);
        TTI_SFPADD(
            p_sfpu::LCONST_1, BCAST_COL_LREG_DATA2, BCAST_NORM_LREG_NEG_MEAN, BCAST_COL_LREG_DATA2, 0 /*instr_mod1*/);
        TTI_SFPADD(
            p_sfpu::LCONST_1, BCAST_COL_LREG_DATA3, BCAST_COL_LREG_BCAST, BCAST_COL_LREG_DATA3, 0 /*instr_mod1*/);
        binary_bcast_neg_mean_(mean_base + offset, bcast_nop_slot_{});
        binary_bcast_normalize_finish_(inv_std_base + offset, out_base + offset);
    });
}

/**
 * @brief out = (data - mean[0][0]) * inv_std[0][0] over one whole fp32 tile.
 *
 * Each statistic is spread from element [0][0] (bit mask and fold) and parked through SFPCONFIG in LREG14, the
 * only Quasar path that copies one row of an LREG into all four; -mean then sits in LREG6 and inv_std in LREG7 for
 * the whole tile.
 *
 * @note Call once per tile under VectorMode::None. Clobbers LREG0-7 and programmable constant LREG14
 *       (vConstFloatPrgm2): an op that reads it, or a parked sfpu_bcast_scalar_load value, needs its init / load
 *       again.
 */
inline void _calculate_sfpu_normalize_bcast_scalar_full_tile_(
    const std::uint32_t dst_index_data,
    const std::uint32_t dst_index_mean,
    const std::uint32_t dst_index_out,
    const std::uint32_t dst_index_inv_std) {
    constexpr std::uint32_t NEG_MEAN = p_sfpu::LREG6;
    constexpr std::uint32_t INV_STD = p_sfpu::LREG7;
    constexpr std::uint32_t D[BCAST_BAND_SLOTS] = {p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3};
    // inv_std first: the spread rebuilds the lane mask in LREG6, where -mean goes.
    binary_bcast_scalar_to_lreg14_(dst_index_inv_std * BCAST_DEST_TILE_STRIDE);
    TTI_SFPMOV(BCAST_SCALAR_LREG_PARKED, INV_STD, 0 /*instr_mod1*/);
    binary_bcast_scalar_to_lreg14_(dst_index_mean * BCAST_DEST_TILE_STRIDE);
    TTI_SFPMOV(BCAST_SCALAR_LREG_PARKED, NEG_MEAN, BCAST_SFPMOV_MOD1_NEGATE);

    const std::uint32_t data_base = dst_index_data * BCAST_DEST_TILE_STRIDE;
    const std::uint32_t out_base = dst_index_out * BCAST_DEST_TILE_STRIDE;
    binary_bcast_for_each_band_([&](const std::uint32_t offset) {
        const std::uint32_t data_addr = data_base + offset;
        const std::uint32_t out_addr = out_base + offset;
        TT_SFPLOAD(D[0], p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_LEFT_EVEN);
        TT_SFPLOAD(D[1], p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_LEFT_ODD);
        TT_SFPLOAD(D[2], p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_RIGHT_EVEN);
        TT_SFPLOAD(D[3], p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, data_addr + BCAST_RIGHT_ODD);
        TTI_SFPADD(p_sfpu::LCONST_1, D[0], NEG_MEAN, D[0], 0 /*instr_mod1*/);
        TTI_SFPADD(p_sfpu::LCONST_1, D[1], NEG_MEAN, D[1], 0 /*instr_mod1*/);
        TTI_SFPADD(p_sfpu::LCONST_1, D[2], NEG_MEAN, D[2], 0 /*instr_mod1*/);
        TTI_SFPADD(p_sfpu::LCONST_1, D[3], NEG_MEAN, D[3], 0 /*instr_mod1*/);
        TTI_SFPMUL(D[0], INV_STD, p_sfpu::LCONST_0, D[0], 0 /*instr_mod1*/);
        TTI_SFPMUL(D[1], INV_STD, p_sfpu::LCONST_0, D[1], 0 /*instr_mod1*/);
        TTI_SFPMUL(D[2], INV_STD, p_sfpu::LCONST_0, D[2], 0 /*instr_mod1*/);
        TTI_SFPMUL(D[3], INV_STD, p_sfpu::LCONST_0, D[3], 0 /*instr_mod1*/);
        TT_SFPSTORE(D[0], p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_LEFT_EVEN);
        TT_SFPSTORE(D[1], p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_LEFT_ODD);
        TT_SFPSTORE(D[2], p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_RIGHT_EVEN);
        TT_SFPSTORE(D[3], p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, out_addr + BCAST_RIGHT_ODD);
    });
}

}  // namespace sfpu
}  // namespace ckernel
