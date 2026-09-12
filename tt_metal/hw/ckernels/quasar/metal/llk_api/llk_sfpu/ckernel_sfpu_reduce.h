// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "lltt.h"
#include "sfpi.h"
#include "tensor_shape.h"

namespace ckernel {
namespace sfpu {

// How to read this file
// ---------------------
// Public API is init_reduce + calculate_reduce at the bottom. calculate_reduce
// dispatches to one of four paths:
//   COL + SUM/AVG → _calculate_reduce_col_sum_avg_
//   COL + MAX/MIN → _calculate_reduce_col_max_min_
//   ROW           → _calculate_reduce_row_
//
// init_reduce records dest-address-free instruction windows into the 32-slot
// replay buffer (NoExec; TEN-4690). Calculate replays those windows. Dest
// addresses stay outside recorded windows (absolute SFPLOAD/STORE + ADDR_MOD_7
// incr=0). Every helper below is inline; splitting is for structure only.

// =============================================================================
// Dest geometry as SFPLOAD/SFPSTORE address it
// =============================================================================
// dest_reg_addr bits [10:2] pick a group of four consecutive face rows and bit 1
// picks even vs odd columns of that group. A 32x32 tile is four 16x16 faces,
// left-to-right then top-to-bottom:
//   Face 0 rows 0-15 cols 0-15  (addr 0)  | Face 1 rows 0-15 cols 16-31 (addr 16)
//   Face 2 rows 16-31 cols 0-15 (addr 32) | Face 3 rows 16-31 cols 16-31 (addr 48)
static_assert(MAX_FACE_R_DIM == 16 && MAX_TILE_R_DIM == 32 && MAX_NUM_FACES == 4);

constexpr std::uint32_t REDUCE_ROWS_PER_LOAD = 4;  // SFPLOAD 4-row dest group
constexpr std::uint32_t REDUCE_ODD_COLUMNS = p_sfpu::col_offset::ODD_COL;
constexpr std::uint32_t REDUCE_AVG_SHIFT = 5;  // log2(MAX_TILE_C_DIM)
constexpr std::uint32_t REDUCE_AVG_SHIFT_MASK = 0xfff;
// bf16 encoding of 1/MAX_TILE_C_DIM = 0.03125, used by SFPMULI for float column AVG.
constexpr std::uint32_t REDUCE_AVG_RECIP_BF16 = 0x3D00;
// imm12 bit 11: SFPSETCC reads src_c as two's-complement INT32 (plain sign-bit test).
constexpr std::uint32_t REDUCE_SETCC_INT32_SIGNBIT = 0x800;
// vConstIntPrgm0 / LREG12 holds 0x0000FFFF for UInt16 loads. Reduce does not use SFPLUTFP32, so
// programming this CREG is safe (the LUT intercept only applies to table reads).
constexpr std::uint32_t REDUCE_UINT16_MASK_LREG = 12;

// Row-reduce partials land in dest column 0: 4-row groups of Face 0 then Face 2.
constexpr std::uint32_t REDUCE_ROW_RESULT_ADDRS[8] = {0, 4, 8, 12, 32, 36, 40, 44};

// Per-face dest bases for column reduce. Index is the even/odd half of each face pair.
constexpr std::uint32_t REDUCE_COL_UPPER_FACE[MAX_NUM_FACES] = {0, 0, 16, 16};
constexpr std::uint32_t REDUCE_COL_LOWER_FACE[MAX_NUM_FACES] = {32, 32, 48, 48};
constexpr std::uint32_t REDUCE_COL_COLUMN_OFFSET[MAX_NUM_FACES] = {0, 2, 0, 2};

constexpr std::uint32_t REDUCE_COL_FACE_ADDRS[MAX_NUM_FACES_C_DIM][MAX_NUM_FACES] = {
    {0, 0, 32, 32},    // even cols: Face 0, Face 0, Face 2, Face 2
    {16, 16, 48, 48},  // odd cols:  Face 1, Face 1, Face 3, Face 3
};
constexpr std::uint32_t REDUCE_COL_FINAL_ADDRS[MAX_NUM_FACES_C_DIM][MAX_NUM_FACES_R_DIM] = {
    {0, 32},   // even: Face 0, Face 2
    {16, 48},  // odd:  Face 1, Face 3
};

// =============================================================================
// Replay-buffer layout (32 slots)
// =============================================================================
// SUM/AVG (init_reduce):
//   [0, 6)  tree-add LREG0-3 || LREG4-7     REDUCE_TREE_ADD_FULL_LEN
//   [6, 9)  tree-add LREG0-3                REDUCE_TREE_ADD_HALF_LEN
//   [9, 23) horizontal-sum phases 2-3       REDUCE_HADD_TAIL_LEN
//
// MAX/MIN (init_reduce):
//   [0, 3)  tree cswap LREG4-7 → LREG4      REDUCE_TREE_CSWAP_LEN
//   [3, 19) horizontal-max phases 2-4       REDUCE_HMAX_REPLAY_LEN
//
constexpr std::uint32_t REDUCE_REPLAY_BUF_LEN = 32;
constexpr std::uint32_t REDUCE_TREE_ADD_FULL_LEN = 6;
constexpr std::uint32_t REDUCE_TREE_ADD_HALF_LEN = 3;
constexpr std::uint32_t REDUCE_HADD_TAIL_SLOT = REDUCE_TREE_ADD_FULL_LEN + REDUCE_TREE_ADD_HALF_LEN;
constexpr std::uint32_t REDUCE_HADD_TAIL_LEN = 14;  // horizontal-sum phases 2-3
constexpr std::uint32_t REDUCE_SUM_AVG_REPLAY_LEN = REDUCE_HADD_TAIL_SLOT + REDUCE_HADD_TAIL_LEN;
constexpr std::uint32_t REDUCE_TREE_CSWAP_LEN = 3;    // SWAP+SWAP + SWAP
constexpr std::uint32_t REDUCE_HMAX_SLOT = REDUCE_TREE_CSWAP_LEN;
constexpr std::uint32_t REDUCE_HMAX_REPLAY_LEN = 16;  // phases 2+3+4 (8+6+2)
constexpr std::uint32_t REDUCE_MAX_MIN_REPLAY_LEN = REDUCE_HMAX_SLOT + REDUCE_HMAX_REPLAY_LEN;

// =============================================================================
// Format predicates
// =============================================================================

// Quasar Dest has no UInt16 slot; UInt16 stimuli ride the Int16/SMAG16 container
// (formats.math is Int16). Treat both as unsigned 16-bit for reduce.
template <DataFormat FMT>
inline constexpr bool _reduce_is_uint16_() {
    return FMT == DataFormat::UInt16 || FMT == DataFormat::Int16;
}

template <DataFormat FMT>
inline constexpr bool _reduce_is_integer_() {
    return FMT == DataFormat::Int32 || _reduce_is_uint16_<FMT>();
}

template <DataFormat FMT>
inline constexpr bool _reduce_is_float_() {
    return FMT == DataFormat::Float32 || FMT == DataFormat::Float16_b;
}

// Quasar SFPSWAP imm12 bit 0 selects the compare: 0 = int32 2's-complement, 1 = fp32
// (assembly.yaml). Int32 Dest is already 2's-complement. Float uses the fp32 mode so
// IEEE ordering (including both-negative pairs) is correct without a software re-swap.
template <DataFormat FMT>
inline constexpr std::uint32_t _reduce_swap_imm12_() {
    return _reduce_is_float_<FMT>() ? 1u : 0u;
}

template <DataFormat FMT>
inline constexpr std::uint32_t _reduce_sfpmem_() {
    if constexpr (_reduce_is_uint16_<FMT>()) {
        return p_sfpu::sfpmem::UINT16;
    }
    return _sfpu_sfpmem_type_<FMT>();
}

// UInt16 in a 32-bit dest lives in the low 16 bits of the LReg, but the packer reads the high 16
// bits of the dest word. SFPSTORE LO16 writes the low half into the packer-visible half. Intermediate
// stores keep UINT16 so a later UINT16 load still round-trips. 16-bit dest (dest_acc No) already
// holds the value in the packer-visible half — do not AND 0xFFFF there, that zeros the datum.
template <bool PACK_LOW16>
inline constexpr std::uint32_t _reduce_store_sfpmem_(const std::uint32_t sfpmem) {
    return PACK_LOW16 ? p_sfpu::sfpmem::LO16 : sfpmem;
}

inline constexpr std::uint32_t _reduce_tile_stride_() {
    return 1U << trisc::get_dest_tile_size_log2(trisc::DstTileShape::Tile32x32);
}

// =============================================================================
// Load / store / add / compare-swap
// =============================================================================

template <bool CLEAR_HIGH>
inline void _reduce_load_(const std::uint32_t lreg, const std::uint32_t sfpmem, const std::uint32_t dest_addr) {
    TT_SFPLOAD(lreg, sfpmem, ADDR_MOD_7, 0 /* done */, dest_addr);
    if constexpr (CLEAR_HIGH) {
        TT_SFPAND(REDUCE_UINT16_MASK_LREG, lreg);
    }
}

inline void _reduce_store_(const std::uint32_t lreg, const std::uint32_t sfpmem, const std::uint32_t dest_addr) {
    TT_SFPSTORE(lreg, sfpmem, ADDR_MOD_7, 0 /* done */, dest_addr);
}

/**
 * @brief Compare-and-swap keeping the extremum in FIRST.
 *
 * SFPSWAP places the smaller operand in lreg_dest and the larger in lreg_c. MAX therefore issues
 * SWAP(FIRST, SECOND) so the max lands in FIRST; MIN issues SWAP(SECOND, FIRST) so the min lands
 * in FIRST.
 */
template <DataFormat FMT, bool IS_MAX, std::uint32_t FIRST, std::uint32_t SECOND>
inline void _reduce_cswap_() {
    constexpr std::uint32_t IMM12 = _reduce_swap_imm12_<FMT>();
    if constexpr (IS_MAX) {
        TTI_SFPSWAP(IMM12, FIRST, SECOND, p_sfpswap::ALL_ROWS_MAX);
    } else {
        TTI_SFPSWAP(IMM12, SECOND, FIRST, p_sfpswap::ALL_ROWS_MAX);
    }
}

template <DataFormat FMT, bool IS_MAX, std::uint32_t A0, std::uint32_t B0, std::uint32_t A1, std::uint32_t B1>
inline void _reduce_cswap_pair_() {
    constexpr std::uint32_t IMM12 = _reduce_swap_imm12_<FMT>();
    if constexpr (IS_MAX) {
        TTI_SFPSWAP(IMM12, A0, B0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(IMM12, A1, B1, p_sfpswap::ALL_ROWS_MAX);
    } else {
        TTI_SFPSWAP(IMM12, B0, A0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(IMM12, B1, A1, p_sfpswap::ALL_ROWS_MAX);
    }
}

template <bool IS_INTEGER>
inline void _reduce_add_(const std::uint32_t src_c, const std::uint32_t dest) {
    if constexpr (IS_INTEGER) {
        TT_SFPIADD(0, src_c, dest, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
    } else {
        TT_SFPADD(dest, p_sfpu::LCONST_1, src_c, dest, 0);
    }
}

template <std::uint32_t SFPMEM, bool CLEAR_HIGH>
inline void _reduce_load_face_(
    const std::uint32_t dst_lreg_base, const std::uint32_t face_addr, const std::uint32_t column_offset) {
    _reduce_load_<CLEAR_HIGH>(dst_lreg_base + 0, SFPMEM, face_addr + column_offset);
    _reduce_load_<CLEAR_HIGH>(dst_lreg_base + 1, SFPMEM, face_addr + column_offset + REDUCE_ROWS_PER_LOAD);
    _reduce_load_<CLEAR_HIGH>(dst_lreg_base + 2, SFPMEM, face_addr + column_offset + 2 * REDUCE_ROWS_PER_LOAD);
    _reduce_load_<CLEAR_HIGH>(dst_lreg_base + 3, SFPMEM, face_addr + column_offset + 3 * REDUCE_ROWS_PER_LOAD);
}

// =============================================================================
// Vertical trees (recorded by init_reduce)
// =============================================================================

// Tree-reduce LREG0-3 -> LREG0 and LREG4-7 -> LREG4 by addition. The two trees are interleaved so
// each SFPADD/SFPIADD covers the other's 2-cycle latency.
template <bool IS_INTEGER>
inline void _reduce_tree_add_full_() {
    if constexpr (IS_INTEGER) {
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
        TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
    } else {
        TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);
        TTI_SFPADD(p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG6, 0);
        TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        TTI_SFPADD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG5, 0);
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    }
}

template <bool IS_INTEGER>
inline void _reduce_tree_add_half_() {
    if constexpr (IS_INTEGER) {
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
    } else {
        TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);
        TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    }
}

// Tree-reduce LREG4-7 -> LREG4 by compare-and-swap (max or min). Recorded by
// init_reduce for MAX/MIN; the column path replays it (10 times per tile).
template <DataFormat FMT, bool IS_MAX>
inline void _reduce_tree_cswap_lreg4_7_() {
    _reduce_cswap_pair_<FMT, IS_MAX, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG7>();
    _reduce_cswap_<FMT, IS_MAX, p_sfpu::LREG4, p_sfpu::LREG6>();
}

// =============================================================================
// Horizontal fold: 8 SFPU columns → column 0 of LREG0 and LREG4
// =============================================================================
// SFPSHFT2 mode 3 rotates one LREG globally across SFPU columns (with wrap).
// Interleaving LREG0/LREG1 and LREG4/LREG5 hides the 2-cycle SHFT2 latency.
// Three fold stages (rotate-by-4, -2, -1) collapse 8 partials into column 0.
//
// Phase 1 (rotate-by-4) stays inline. Later phases are recorded and replayed.

template <bool IS_INTEGER>
inline void _horizontal_reduce_sum_tail_() {
    // Phase 2: rotate-by-2 and add.
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 3);
    if constexpr (IS_INTEGER) {
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
    } else {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    }

    // Phase 3: rotate-by-1 and add.
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 3);
    if constexpr (IS_INTEGER) {
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);
    } else {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    }
}

template <bool IS_INTEGER>
inline void _horizontal_reduce_sum_() {
    // Phase 1: rotate-by-4 and add (inline).
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 3);
    _reduce_add_<IS_INTEGER>(p_sfpu::LREG1, p_sfpu::LREG0);
    _reduce_add_<IS_INTEGER>(p_sfpu::LREG5, p_sfpu::LREG4);
    lltt::replay(REDUCE_HADD_TAIL_SLOT, REDUCE_HADD_TAIL_LEN);
}

template <DataFormat FMT, bool IS_MAX>
inline void _horizontal_reduce_max_tail_() {
    // Phase 2: rotate-by-2 and keep extremum (2 MOV + 4 SHFT2 + 2 SWAP).
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 3);
    _reduce_cswap_pair_<FMT, IS_MAX, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG5>();

    // Phase 3: rotate-by-1 and keep extremum (2 MOV + 2 SHFT2 + 2 SWAP).
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 3);
    _reduce_cswap_pair_<FMT, IS_MAX, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG5>();

    // Phase 4: rotate the result into column 0 (2 SHFT2).
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG0, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG4, 3);
}

template <DataFormat FMT, bool IS_MAX>
inline void _horizontal_reduce_max_() {
    // Phase 1: rotate-by-4 and keep extremum (inline). Phases 2-4 come from the replay buffer.
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, 3);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, 3);
    _reduce_cswap_pair_<FMT, IS_MAX, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG5>();
    lltt::replay(REDUCE_HMAX_SLOT, REDUCE_HMAX_REPLAY_LEN);
}

// Toward-zero signed divide-by-32 of LREG0 (two's-complement). Logical right shift of the magnitude,
// then restore the original sign. Arithmetic shift would round toward -inf and miss the golden.
inline void _reduce_int_average_() {
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPSETCC(REDUCE_SETCC_INT32_SIGNBIT, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 6);  // 0 - LREG0 when negative (mod 6: sub, no CC update)
    TTI_SFPENCC(0, 0);
    TTI_SFPSHFT((-REDUCE_AVG_SHIFT) & REDUCE_AVG_SHIFT_MASK, p_sfpu::LREG0, p_sfpu::LREG0, 0b01);
    TTI_SFPSETCC(REDUCE_SETCC_INT32_SIGNBIT, p_sfpu::LREG1, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 6);
    TTI_SFPENCC(0, 0);
}

inline void _reduce_float_average_() {
    // SFPMULI: dest *= imm16 in FP16B. 0x3D00 is bf16 1/32. LREG12 is a LUT intercept on Quasar, so
    // the reciprocal is an immediate rather than a programmable const register.
    TTI_SFPMULI(REDUCE_AVG_RECIP_BF16, p_sfpu::LREG0, 0);
}

inline void _reduce_load_reciprocal_(const std::uint32_t lreg, const std::uint32_t num_cols) {
    const float recip = 1.0f / static_cast<float>(num_cols);
    std::uint32_t bits = 0;
    static_assert(sizeof(float) == sizeof(std::uint32_t), "float must be 32-bit");
    __builtin_memcpy(&bits, &recip, sizeof(bits));
    TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_UPPER, bits >> 16);
    TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_LOWER, bits & 0xFFFFu);
}

// =============================================================================
// Column reduce
// =============================================================================

/**
 * @brief Column-wise SUM/AVG of one 32x32 tile. Result is written to dest row 0 of faces 0 and 1.
 *
 * Per even/odd column half of each vertically adjacent face pair: tree-add the 16 rows of each
 * face, add the two faces, transpose so the four partials of a column sit in one register, then
 * tree-add LREG0-3 into LREG0[0].
 */
template <PoolType POOL, DataFormat FMT, bool PACK_LOW16>
inline void _calculate_reduce_col_sum_avg_() {
    constexpr bool IS_INTEGER = _reduce_is_integer_<FMT>();
    constexpr bool CLEAR_HIGH = PACK_LOW16;
    constexpr std::uint32_t SFPMEM = _reduce_sfpmem_<FMT>();
    constexpr std::uint32_t RESULT_SFPMEM = _reduce_store_sfpmem_<PACK_LOW16>(SFPMEM);

    for (std::uint32_t i = 0; i < MAX_NUM_FACES; i++) {
        const std::uint32_t upper = REDUCE_COL_UPPER_FACE[i];
        const std::uint32_t lower = REDUCE_COL_LOWER_FACE[i];
        const std::uint32_t col = REDUCE_COL_COLUMN_OFFSET[i];

        _reduce_load_face_<SFPMEM, CLEAR_HIGH>(p_sfpu::LREG0, upper, col);
        _reduce_load_face_<SFPMEM, CLEAR_HIGH>(p_sfpu::LREG4, lower, col);
        lltt::replay(0, REDUCE_TREE_ADD_FULL_LEN);
        _reduce_add_<IS_INTEGER>(p_sfpu::LREG4, p_sfpu::LREG0);
        TTI_SFPTRANSP;
        lltt::replay(REDUCE_TREE_ADD_FULL_LEN, REDUCE_TREE_ADD_HALF_LEN);

        if constexpr (POOL == PoolType::AVG) {
            if constexpr (IS_INTEGER) {
                _reduce_int_average_();
            } else {
                _reduce_float_average_();
            }
        }
        _reduce_store_(p_sfpu::LREG0, RESULT_SFPMEM, upper + col);
    }
}

/**
 * @brief Column-wise MAX/MIN of one 32x32 tile. Result is written to dest row 0 of faces 0 and 1.
 *
 * Reduce each face's 16 rows into its top 4 rows via compare-and-swap, then fold the two
 * vertically adjacent faces together with a transpose pair so the four lane-rows of a column meet.
 */
template <PoolType POOL, DataFormat FMT, bool PACK_LOW16>
inline void _calculate_reduce_col_max_min_() {
    static_assert(POOL == PoolType::MAX || POOL == PoolType::MIN, "column max/min only");
    constexpr bool IS_MAX = (POOL == PoolType::MAX);
    constexpr bool CLEAR_HIGH = PACK_LOW16;
    constexpr std::uint32_t SFPMEM = _reduce_sfpmem_<FMT>();
    constexpr std::uint32_t RESULT_SFPMEM = _reduce_store_sfpmem_<PACK_LOW16>(SFPMEM);

    for (std::uint32_t j = 0; j < MAX_NUM_FACES_C_DIM; j++) {
        const std::uint32_t top_face = REDUCE_COL_FINAL_ADDRS[j][0];
        const std::uint32_t bot_face = REDUCE_COL_FINAL_ADDRS[j][1];

        for (std::uint32_t i = 0; i < MAX_NUM_FACES; i++) {
            _reduce_load_face_<SFPMEM, CLEAR_HIGH>(
                p_sfpu::LREG4, REDUCE_COL_FACE_ADDRS[j][i], REDUCE_COL_COLUMN_OFFSET[i]);
            lltt::replay(0, REDUCE_TREE_CSWAP_LEN);
            _reduce_store_(p_sfpu::LREG4, SFPMEM, REDUCE_COL_FACE_ADDRS[j][i] + REDUCE_COL_COLUMN_OFFSET[i]);
        }

        _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG0, SFPMEM, top_face);
        _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG1, SFPMEM, bot_face);
        _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG2, SFPMEM, top_face + REDUCE_ODD_COLUMNS);
        _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG3, SFPMEM, bot_face + REDUCE_ODD_COLUMNS);

        TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG4, 0);
        TTI_SFPMOV(p_sfpu::LREG1, p_sfpu::LREG5, 0);
        TTI_SFPMOV(p_sfpu::LREG2, p_sfpu::LREG6, 0);
        TTI_SFPMOV(p_sfpu::LREG3, p_sfpu::LREG7, 0);

        TTI_SFPTRANSP;
        lltt::replay(0, REDUCE_TREE_CSWAP_LEN);
        TTI_SFPTRANSP;

        _reduce_cswap_<FMT, IS_MAX, p_sfpu::LREG4, p_sfpu::LREG5>();
        _reduce_cswap_<FMT, IS_MAX, p_sfpu::LREG6, p_sfpu::LREG7>();

        TTI_SFPMOV(p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPMOV(p_sfpu::LREG6, p_sfpu::LREG1, 0);
        _reduce_store_(p_sfpu::LREG0, RESULT_SFPMEM, top_face);
        _reduce_store_(p_sfpu::LREG1, RESULT_SFPMEM, top_face + REDUCE_ODD_COLUMNS);
    }
}

// =============================================================================
// Row reduce
// =============================================================================

template <std::uint32_t SFPMEM, bool CLEAR_HIGH>
inline void _reduce_load_row_halves_(const std::uint32_t group_a, const std::uint32_t group_b) {
    _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG0, SFPMEM, group_a);
    _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG1, SFPMEM, group_a + REDUCE_ODD_COLUMNS);
    _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG2, SFPMEM, group_a + MAX_FACE_R_DIM);
    _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG3, SFPMEM, group_a + MAX_FACE_R_DIM + REDUCE_ODD_COLUMNS);
    _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG4, SFPMEM, group_b);
    _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG5, SFPMEM, group_b + REDUCE_ODD_COLUMNS);
    _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG6, SFPMEM, group_b + MAX_FACE_R_DIM);
    _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG7, SFPMEM, group_b + MAX_FACE_R_DIM + REDUCE_ODD_COLUMNS);
}

template <PoolType POOL, DataFormat FMT, bool CLEAR_HIGH>
inline void _perform_reduce_row_tile_(const std::uint32_t tile_offset, const std::uint32_t store_sfpmem) {
    constexpr bool IS_MAX_MIN = (POOL == PoolType::MAX || POOL == PoolType::MIN);
    constexpr std::uint32_t SFPMEM = _reduce_sfpmem_<FMT>();

    for (std::uint32_t face_pair = 0; face_pair < MAX_NUM_FACES_R_DIM; face_pair++) {
        const std::uint32_t face_pair_base = face_pair * MAX_TILE_R_DIM;
        for (std::uint32_t row_group = 0; row_group < 2; row_group++) {
            const std::uint32_t group_a = tile_offset + face_pair_base + row_group * 2 * REDUCE_ROWS_PER_LOAD;
            const std::uint32_t group_b = group_a + REDUCE_ROWS_PER_LOAD;
            _reduce_load_row_halves_<SFPMEM, CLEAR_HIGH>(group_a, group_b);

            if constexpr (IS_MAX_MIN) {
                constexpr bool IS_MAX = (POOL == PoolType::MAX);
                // Vertical max/min: left/right face pairs, then even/odd columns of each 4-row group.
                _reduce_cswap_pair_<FMT, IS_MAX, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LREG6>();
                _reduce_cswap_pair_<FMT, IS_MAX, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG5, p_sfpu::LREG7>();
                _reduce_cswap_pair_<FMT, IS_MAX, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG5>();
                _horizontal_reduce_max_<FMT, IS_MAX>();
            } else {
                constexpr bool IS_INTEGER = _reduce_is_integer_<FMT>();
                lltt::replay(0, REDUCE_TREE_ADD_FULL_LEN);
                _horizontal_reduce_sum_<IS_INTEGER>();
            }
            _reduce_store_(p_sfpu::LREG0, store_sfpmem, group_a);
            _reduce_store_(p_sfpu::LREG4, store_sfpmem, group_b);
        }
    }
}

// Single-tile row AVG divides by MAX_TILE_C_DIM in-register before the store. Multi-tile rows
// accumulate first and divide in _combine_first_columns_ instead.
template <DataFormat FMT, bool CLEAR_HIGH>
inline void _perform_reduce_row_avg_tile_(const std::uint32_t tile_offset, const std::uint32_t store_sfpmem) {
    constexpr bool IS_INTEGER = _reduce_is_integer_<FMT>();
    constexpr std::uint32_t SFPMEM = _reduce_sfpmem_<FMT>();

    for (std::uint32_t face_pair = 0; face_pair < MAX_NUM_FACES_R_DIM; face_pair++) {
        const std::uint32_t face_pair_base = face_pair * MAX_TILE_R_DIM;
        for (std::uint32_t row_group = 0; row_group < 2; row_group++) {
            const std::uint32_t group_a = tile_offset + face_pair_base + row_group * 2 * REDUCE_ROWS_PER_LOAD;
            const std::uint32_t group_b = group_a + REDUCE_ROWS_PER_LOAD;
            _reduce_load_row_halves_<SFPMEM, CLEAR_HIGH>(group_a, group_b);

            lltt::replay(0, REDUCE_TREE_ADD_FULL_LEN);
            _horizontal_reduce_sum_<IS_INTEGER>();
            _reduce_load_reciprocal_(p_sfpu::LREG2, MAX_TILE_C_DIM);
            TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
            TTI_SFPMUL(p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);

            _reduce_store_(p_sfpu::LREG0, store_sfpmem, group_a);
            _reduce_store_(p_sfpu::LREG4, store_sfpmem, group_b);
        }
    }
}

template <PoolType POOL, DataFormat FMT, bool PACK_LOW16>
inline void _combine_first_columns_(const std::uint32_t tile_row_base, const std::uint32_t block_ct_dim) {
    constexpr bool IS_INTEGER = _reduce_is_integer_<FMT>();
    constexpr bool IS_MAX_MIN = (POOL == PoolType::MAX || POOL == PoolType::MIN);
    constexpr bool IS_MAX = (POOL == PoolType::MAX);
    constexpr bool IS_AVG = (POOL == PoolType::AVG);
    constexpr bool CLEAR_HIGH = PACK_LOW16;
    constexpr std::uint32_t SFPMEM = _reduce_sfpmem_<FMT>();
    constexpr std::uint32_t RESULT_SFPMEM = _reduce_store_sfpmem_<PACK_LOW16>(SFPMEM);
    constexpr std::uint32_t TILE_STRIDE = _reduce_tile_stride_();

    for (std::uint32_t batch = 0; batch < 2; batch++) {
        const std::uint32_t base_idx = batch * 4;
        _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG0, SFPMEM, tile_row_base + REDUCE_ROW_RESULT_ADDRS[base_idx + 0]);
        _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG1, SFPMEM, tile_row_base + REDUCE_ROW_RESULT_ADDRS[base_idx + 1]);
        _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG2, SFPMEM, tile_row_base + REDUCE_ROW_RESULT_ADDRS[base_idx + 2]);
        _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG3, SFPMEM, tile_row_base + REDUCE_ROW_RESULT_ADDRS[base_idx + 3]);

        for (std::uint32_t t = 1; t < block_ct_dim; t++) {
            const std::uint32_t tile_offset = tile_row_base + t * TILE_STRIDE;
            _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG4, SFPMEM, tile_offset + REDUCE_ROW_RESULT_ADDRS[base_idx + 0]);
            _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG5, SFPMEM, tile_offset + REDUCE_ROW_RESULT_ADDRS[base_idx + 1]);
            _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG6, SFPMEM, tile_offset + REDUCE_ROW_RESULT_ADDRS[base_idx + 2]);
            _reduce_load_<CLEAR_HIGH>(p_sfpu::LREG7, SFPMEM, tile_offset + REDUCE_ROW_RESULT_ADDRS[base_idx + 3]);

            if constexpr (IS_MAX_MIN) {
                _reduce_cswap_pair_<FMT, IS_MAX, p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LREG5>();
                _reduce_cswap_pair_<FMT, IS_MAX, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG3, p_sfpu::LREG7>();
            } else {
                _reduce_add_<IS_INTEGER>(p_sfpu::LREG4, p_sfpu::LREG0);
                _reduce_add_<IS_INTEGER>(p_sfpu::LREG5, p_sfpu::LREG1);
                _reduce_add_<IS_INTEGER>(p_sfpu::LREG6, p_sfpu::LREG2);
                _reduce_add_<IS_INTEGER>(p_sfpu::LREG7, p_sfpu::LREG3);
            }
        }

        if constexpr (IS_AVG) {
            _reduce_load_reciprocal_(p_sfpu::LREG4, MAX_TILE_C_DIM * block_ct_dim);
            TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
            TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
            TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
            TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
        }

        _reduce_store_(p_sfpu::LREG0, RESULT_SFPMEM, tile_row_base + REDUCE_ROW_RESULT_ADDRS[base_idx + 0]);
        _reduce_store_(p_sfpu::LREG1, RESULT_SFPMEM, tile_row_base + REDUCE_ROW_RESULT_ADDRS[base_idx + 1]);
        _reduce_store_(p_sfpu::LREG2, RESULT_SFPMEM, tile_row_base + REDUCE_ROW_RESULT_ADDRS[base_idx + 2]);
        _reduce_store_(p_sfpu::LREG3, RESULT_SFPMEM, tile_row_base + REDUCE_ROW_RESULT_ADDRS[base_idx + 3]);
    }
}

template <PoolType POOL, DataFormat FMT, bool PACK_LOW16>
inline void _calculate_reduce_row_(const std::uint32_t block_ct_dim, const std::uint32_t block_rt_dim) {
    constexpr std::uint32_t TILE_STRIDE = _reduce_tile_stride_();
    constexpr std::uint32_t SFPMEM = _reduce_sfpmem_<FMT>();
    // LO16 only on the packer-visible store (single-tile row, or the combine into tile 0).
    const std::uint32_t tile_store_sfpmem = (PACK_LOW16 && block_ct_dim == 1) ? p_sfpu::sfpmem::LO16 : SFPMEM;

    for (std::uint32_t r = 0; r < block_rt_dim; r++) {
        const std::uint32_t tile_row_base = TILE_STRIDE * block_ct_dim * r;
        for (std::uint32_t c = 0; c < block_ct_dim; c++) {
            const std::uint32_t tile_offset = tile_row_base + TILE_STRIDE * c;
            if constexpr (POOL == PoolType::AVG) {
                if (block_ct_dim == 1) {
                    _perform_reduce_row_avg_tile_<FMT, PACK_LOW16>(tile_offset, tile_store_sfpmem);
                    continue;
                }
            }
            _perform_reduce_row_tile_<POOL, FMT, PACK_LOW16>(tile_offset, tile_store_sfpmem);
        }
        if (block_ct_dim > 1) {
            _combine_first_columns_<POOL, FMT, PACK_LOW16>(tile_row_base, block_ct_dim);
        }
    }
}

// =============================================================================
// Public API
// =============================================================================

/**
 * @brief Initialization for the SFPU reduce kernel.
 *
 * @tparam pool_type: Reduction op, values = <SUM/AVG/MAX/MIN>
 * @tparam format: Dest data format, values = <Int32/UInt16/Float32/Float16_b>
 * @tparam is_fp32_dest_acc_en: Whether Dest is in 32-bit mode
 * @note Pair with @ref calculate_reduce. Call after the shared SFPU init. Records dest-free
 *       tree/tail sequences into the replay buffer (NoExec; TEN-4690).
 */
template <PoolType pool_type, DataFormat format, bool is_fp32_dest_acc_en>
inline void init_reduce([[maybe_unused]] std::uint32_t block_ct_dim = 1) {
    static_assert(
        format == DataFormat::Int32 || _reduce_is_uint16_<format>() || format == DataFormat::Float32 ||
            format == DataFormat::Float16_b,
        "Unsupported reduce format. Supported: Int32, UInt16 (Int16 container), Float32, Float16_b");
    static_assert(
        pool_type == PoolType::SUM || pool_type == PoolType::AVG || pool_type == PoolType::MAX ||
            pool_type == PoolType::MIN,
        "Unsupported pool_type. Supported: SUM, AVG, MAX, MIN");
    ckernel::math::_reset_counters_<p_setrwc::SET_ABD_F>();
    if constexpr (is_fp32_dest_acc_en && _reduce_is_uint16_<format>()) {
        // Mask for SFPAND after UINT16 loads from a 32-bit dest word. 16-bit dest does not need it.
        sfpi::vConstIntPrgm0 = 0x0000FFFF;
    }

    // Dest-address-free trees/tails. lltt::record is NoExec (TEN-4690).
    if constexpr (pool_type == PoolType::MAX || pool_type == PoolType::MIN) {
        constexpr bool IS_MAX = (pool_type == PoolType::MAX);
        static_assert(REDUCE_MAX_MIN_REPLAY_LEN <= REDUCE_REPLAY_BUF_LEN, "MAX/MIN replay exceeds 32-slot buffer");
        lltt::record(0, REDUCE_MAX_MIN_REPLAY_LEN);
        _reduce_tree_cswap_lreg4_7_<format, IS_MAX>();
        _horizontal_reduce_max_tail_<format, IS_MAX>();
    } else {
        constexpr bool IS_INTEGER = _reduce_is_integer_<format>();
        static_assert(REDUCE_SUM_AVG_REPLAY_LEN <= REDUCE_REPLAY_BUF_LEN, "SUM/AVG replay exceeds 32-slot buffer");
        lltt::record(0, REDUCE_SUM_AVG_REPLAY_LEN);
        _reduce_tree_add_full_<IS_INTEGER>();
        _reduce_tree_add_half_<IS_INTEGER>();
        _horizontal_reduce_sum_tail_<IS_INTEGER>();
    }
}

/**
 * @brief SFPU reduce of a 32x32 tile (or a dest block of tiles for REDUCE_ROW).
 *
 * Column reduce writes each column's result into dest row 0 of faces 0 and 1. Row reduce writes
 * each row's result into dest column 0. Only that row/column is defined on return; the rest of the
 * tile is scratch.
 *
 * @tparam pool_type: Reduction op, values = <SUM/AVG/MAX/MIN>
 * @tparam reduce_dim: REDUCE_COL or REDUCE_ROW (AVG row is float formats only)
 * @tparam format: Input Dest format, values = <Int32/UInt16/Float32/Float16_b>
 * @tparam is_fp32_dest_acc_en: Whether Dest is in 32-bit mode
 * @tparam output_format: Packer-visible output format (defaults to format)
 * @param block_ct_dim: Tiles along x, used by REDUCE_ROW (default 1)
 * @param block_rt_dim: Tiles along y, used by REDUCE_ROW (default 1)
 * @note Call @ref init_reduce first. Run under VectorMode::RC_custom — the reduction spans every face.
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    DataFormat format,
    bool is_fp32_dest_acc_en,
    DataFormat output_format = format>
inline void calculate_reduce(
    [[maybe_unused]] std::uint32_t block_ct_dim = 1, [[maybe_unused]] std::uint32_t block_rt_dim = 1) {
    constexpr bool is_float_format = _reduce_is_float_<format>();
    static_assert(
        reduce_dim == ReduceDim::REDUCE_COL ||
            (reduce_dim == ReduceDim::REDUCE_ROW &&
             (pool_type == PoolType::SUM || pool_type == PoolType::MAX || pool_type == PoolType::MIN ||
              (pool_type == PoolType::AVG && is_float_format))),
        "Row reduction supports SUM/MAX/MIN (all formats) and AVG (float formats only)");
    static_assert(
        format == DataFormat::Int32 || _reduce_is_uint16_<format>() || format == DataFormat::Float32 ||
            format == DataFormat::Float16_b,
        "Unsupported reduce format. Supported: Int32, UInt16 (Int16 container), Float32, Float16_b");
    static_assert(
        pool_type == PoolType::SUM || pool_type == PoolType::AVG || pool_type == PoolType::MAX ||
            pool_type == PoolType::MIN,
        "Unsupported pool_type. Supported: SUM, AVG, MAX, MIN");

    constexpr bool pack_low16 = is_fp32_dest_acc_en && _reduce_is_uint16_<output_format>();
    if constexpr (reduce_dim == ReduceDim::REDUCE_COL) {
        if constexpr (pool_type == PoolType::MAX || pool_type == PoolType::MIN) {
            _calculate_reduce_col_max_min_<pool_type, format, pack_low16>();
        } else {
            _calculate_reduce_col_sum_avg_<pool_type, format, pack_low16>();
        }
    } else {
        _calculate_reduce_row_<pool_type, format, pack_low16>(block_ct_dim, block_rt_dim);
    }
}

}  // namespace sfpu
}  // namespace ckernel
