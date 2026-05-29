// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

inline void set_dst_write_addr_offset(std::uint32_t addr) {
    LLK_ASSERT(addr < DEST_REGISTER_HALF_SIZE, "Address overflow in set_dst_write_addr_offset");
    std::uint32_t dst_index = addr + get_dest_buffer_base();
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_top32_load8(std::uint32_t offset, std::uint32_t dist) {
    constexpr std::uint32_t dst_indices_offset = 128;  // 2 tile x 64 rows per tile
    constexpr std::uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset = (offset & 0xF) + face_offset * 32;

    // Load 16 consecutive numbers
    TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, ld_offset);
    TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, ld_offset + dist);

    // Load 16 consecutive indices
    TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, dst_indices_offset + ld_offset);
    TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + ld_offset + dist);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_top32_store8(std::uint32_t offset, std::uint32_t dist) {
    constexpr std::uint32_t dst_indices_offset = 128;  // 2 tile x 64 rows per tile
    constexpr std::uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset = (offset & 0xF) + face_offset * 32;

    // Load 16 consecutive numbers
    TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, ld_offset);
    TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, ld_offset + dist);

    // Load 16 consecutive indices
    TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, dst_indices_offset + ld_offset + 0);
    TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + ld_offset + dist);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_top32_load16(std::uint32_t dist0, std::uint32_t dist1) {
    constexpr std::uint32_t dst_indices_offset = 128;  // 2 tile x 64 rows per tile
    constexpr std::uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    // Load 16 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    if ((dist0 == 4) && (dist1 == 8)) {
        TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 4);
        TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 8);
        TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, 12);
    } else {
        TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 0 + dist0);
        TT_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, dist1);
        TT_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, dist1 + dist0);
    }

    // Load 16 consecutive indices
    TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 0);
    if ((dist0 == 4) && (dist1 == 8)) {
        TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 4);
        TTI_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 8);
        TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 12);
    } else {
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 0 + dist0);
        TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, dst_indices_offset + dist1);
        TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, dst_indices_offset + dist1 + dist0);
    }
}

template <bool is_fp32_dest_acc_en, bool alt_addr_mod = false>
inline void bitonic_top32_store16(std::uint32_t dist0, std::uint32_t dist1) {
    constexpr std::uint32_t dst_indices_offset = 128;  // 2 tile x 64 rows per tile
    constexpr std::uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    // Load 16 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    if ((dist0 == 4) && (dist1 == 8)) {
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 4);
        TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, 8);
        TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, 12);
    } else {
        TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0 + dist0);
        TT_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, dist1);
        TT_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, dist1 + dist0);
    }

    // Load 16 consecutive indices
    TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 0);
    if ((dist0 == 4) && (dist1 == 8)) {
        TTI_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 4);
        TTI_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 8);
        TTI_SFPSTORE(p_sfpu::LREG7, instr_mod_index, alt_addr_mod ? ADDR_MOD_6 : ADDR_MOD_7, dst_indices_offset + 12);
    } else {
        TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 0 + dist0);
        TT_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, dst_indices_offset + dist1);
        TT_SFPSTORE(
            p_sfpu::LREG7, instr_mod_index, alt_addr_mod ? ADDR_MOD_6 : ADDR_MOD_7, dst_indices_offset + dist1 + dist0);
    }
}

inline void bitonic_top32_ph3_st4_to_1(bool dir) {
    if (dir == static_cast<bool>(SortDir::ArgMin)) {
        TTI_SFPCONFIG(0x104, 0xF, 1);  // Reverse the max/min behaviour of SWAP
        TTI_SFPNOP;
        TTI_SFPNOP;
    }

    // Step 4
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 4
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);

    if (dir == static_cast<bool>(SortDir::ArgMin)) {
        TTI_SFPCONFIG(0x004, 0xF, 1);  // Restore the max/min behaviour of SWAP
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
}

inline void bitonic_top32_ph2_st3_to_1() {
    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::UNCONDITIONALLY);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

inline void bitonic_top32_ph1_st2_to_1() {
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

inline void bitonic_top32_ph0_st1_to_1() {
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

inline void bitonic_top32_step_N(bool dir) {
    // Step N
    if (dir == static_cast<bool>(SortDir::ArgMax)) {
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    } else {
        // Min
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    }
}

inline void bitonic_top32_inc_x8_dest(std::uint32_t inc) {
    std::uint32_t inc_grp8 = inc >> 3;
    for (std::uint32_t i = 0; i < inc_grp8; i++) {
        TTI_INCRWC(0, 8, 0, 0);
    }
}

inline void bitonic_top32_inc_x4_dest(std::uint32_t inc, bool cr) {
    std::uint32_t inc_grp4 = inc >> 2;
    if (cr) {
        for (std::uint32_t i = 0; i < inc_grp4; i++) {
            TTI_INCRWC(0b100, 4, 0, 0);
        }
    } else {
        for (std::uint32_t i = 0; i < inc_grp4; i++) {
            TTI_INCRWC(0, 4, 0, 0);
        }
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _bitonic_top32_phases_steps_(const int idir) {
    const int i_end_phase = 4;
    const int i_start_phase = 0;

    bool dir = idir;
    // produce bitonic sequences len=16
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int d = 0; d < 4; d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
        bitonic_top32_ph0_st1_to_1();
        bitonic_top32_ph1_st2_to_1();
        bitonic_top32_ph2_st3_to_1();
        bitonic_top32_ph3_st4_to_1(dir);
        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
        dir = !dir;
    }

    // produce bitonic sequences len=32
    std::uint32_t num_steps = 5;  // log(32)
    std::uint32_t start_step = num_steps;
    std::uint32_t end_step = 4;
    std::uint32_t sorted_seq_length = 1 << num_steps;  // 32
    std::uint32_t datums_compared = 0;
    std::uint32_t total_datums_to_compare = 64;
    for (std::uint32_t ss = start_step; ss > end_step; ss--) {
        // Steps N to 5
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        dir = idir;
        std::uint32_t dist = 16;
        std::uint32_t inner_d =
            dist >> 3;  // How many loops to sort the sequence of length (2^ss / 16). Each loop sorts 16
        for (std::uint32_t d = 0; d < 2; d++) {
            for (std::uint32_t ii = 0; ii < inner_d; ii++) {
                bitonic_top32_load16<is_fp32_dest_acc_en>(4, dist);
                bitonic_top32_step_N(dir);
                bitonic_top32_store16<is_fp32_dest_acc_en, false>(4, dist);
                bitonic_top32_inc_x8_dest(8);
            }
            bitonic_top32_inc_x8_dest(16);
            dir = !dir;
        }
    }
    // steps 4 to 1
    dir = idir;
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int d = 0; d < 2; d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
        bitonic_top32_ph3_st4_to_1(dir);
        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
        bitonic_top32_ph3_st4_to_1(dir);
        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
        dir = !dir;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool top_min>
inline void _bitonic_top32_merge_(const bool across_tiles) {
    std::uint32_t dist = across_tiles ? 64 : 32;

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int d = 0; d < 4; d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, dist);
        bitonic_top32_step_N(top_min);
        bitonic_top32_store16<is_fp32_dest_acc_en, false>(4, dist);
        bitonic_top32_inc_x8_dest(8);
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _bitonic_top32_rebuild_(const bool idir, const bool skip_second) {
    // Step 5
    bool dir = idir;
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    constexpr std::uint32_t dist = 16;
    constexpr std::uint32_t inner_d = dist >> 3;
    for (std::uint32_t d = 0; d < (skip_second ? 1 : 2); d++) {
        for (std::uint32_t ii = 0; ii < inner_d; ii++) {
            bitonic_top32_load16<is_fp32_dest_acc_en>(4, dist);
            bitonic_top32_step_N(dir);
            bitonic_top32_store16<is_fp32_dest_acc_en, false>(4, dist);
            bitonic_top32_inc_x8_dest(8);
        }
        bitonic_top32_inc_x8_dest(16);
        dir = !dir;
    }
    // steps 4 to 1
    dir = idir;
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (std::uint32_t d = 0; d < (skip_second ? 1 : 2); d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
        bitonic_top32_ph3_st4_to_1(dir);
        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
        bitonic_top32_ph3_st4_to_1(dir);
        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
        dir = !dir;
    }
}

// clang-format off
/**
 * Produces bitonic top32 on 16 independent columns of 1024 elements
 * Input data must be in row major (RM) layout and pre-sorted to len 32 sub arrays
 * The data needs to be loaded into the DST register transposed, as the sorting happens on columns
 * The indices need to be loaded into the DST register in the same way, but with offset of 2 tiles
 *
 * Algorithm:
 * 1. Reduild len 32 bitonic sequences from the pre-sorted data
 *    - do on both even and odd cols
 * 2. Merge and rebuild F0/F1 sequences with F2/F3 sequences
 *    - do on both even and odd cols
 *    - even and odd cols alternate in sort direction
 */
// clang-format on
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool top_min>
inline void _bitonic_top32_of_1024_rm_pre_sorted_prep_(std::uint32_t dst_index) {
    constexpr std::uint32_t odd_col_offset = 2;
    constexpr bool decreasing = false;
    const std::uint32_t tile_offset = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];

    /// Step 1
    // Build len 32 bitonic sequences from the pre-sorted data
    bool dir = decreasing;
    for (int col = 0; col < 2; col++) {
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        for (int d = 0; d < 4; d++) {
            bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
            bitonic_top32_ph3_st4_to_1(dir);
            bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
            dir = !dir;
        }
        _bitonic_top32_rebuild_<APPROXIMATION_MODE, is_fp32_dest_acc_en>(
            /* idir */ decreasing, /* skip_second */ false);
        set_dst_write_addr_offset(tile_offset + odd_col_offset);
    }
    set_dst_write_addr_offset(tile_offset);

    /// Step 2
    // Merge and rebuild F0/F1 sequences with F2/F3 sequences
    dir = top_min;
    for (int col = 0; col < 2; col++) {
        _bitonic_top32_merge_<APPROXIMATION_MODE, is_fp32_dest_acc_en, decreasing>(/* across_tiles */ false);
        _bitonic_top32_rebuild_<APPROXIMATION_MODE, is_fp32_dest_acc_en>(/* idir */ dir, /* skip_second */ true);
        dir = !dir;
        set_dst_write_addr_offset(tile_offset + odd_col_offset);
    }
    set_dst_write_addr_offset(tile_offset);
}

// clang-format off
/**
 * Combines top32 sequences across F0/F1 of 2 adjacent tiles independently on 16 columns
 * Implemented with simple merge and rebuild steps
 */
// clang-format on
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _bitonic_top32_of_1024_rm_pre_sorted_combine_(std::uint32_t dst_index) {
    constexpr std::uint32_t odd_col_offset = 2;
    constexpr bool decreasing = false;
    constexpr bool increasing = true;
    const std::uint32_t tile_offset = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];
    _bitonic_top32_merge_<APPROXIMATION_MODE, is_fp32_dest_acc_en, decreasing>(/* across_tiles */ true);
    _bitonic_top32_rebuild_<APPROXIMATION_MODE, is_fp32_dest_acc_en>(/* idir */ decreasing, /* skip_second */ true);
    set_dst_write_addr_offset(tile_offset + odd_col_offset);
    _bitonic_top32_merge_<APPROXIMATION_MODE, is_fp32_dest_acc_en, decreasing>(/* across_tiles */ true);
    _bitonic_top32_rebuild_<APPROXIMATION_MODE, is_fp32_dest_acc_en>(/* idir */ increasing, /* skip_second */ true);
    set_dst_write_addr_offset(tile_offset);
}

// clang-format off
/**
 * Produces final top32 from sequences in F0/F1 with data pre-sorted to len 32 sub arrays
 * Odd cols start with decreasing, even cols increasing
 * Final output is in even col 0 of F0/F1
 *
 * Algorithm:
 * 1. Merge even and odd cols and rebuild, then store to odd cols
 *    - alternate SFPU instances with increasing/decreasing
 *    - after this step, there are 8 cols remaining (all odd cols)
 * 2. Shift odd cols by 1 SFPU instance right, and store to even cols
 * 3. Merge even and odd cols and rebuild, then store to odd cols
 *    - alternate every 2nd SFPU instance with increasing/decreasing
 *    - after this step, there are 4 cols remaining (every 2nd odd col)
 * 4. Shift odd cols by 2 SFPU instances right, and store to even cols
 * 5. Merge even and odd cols and rebuild, then store to odd cols
 *    - alternate every 4th SFPU instance with increasing/decreasing
 *    - after this step, there are 2 cols remaining (every 4th odd col)
 * 6. Shift odd cols by 4 SFPU instances right, and store to even cols
 * 7. Merge even and odd cols and rebuild, then store to even cols
 *    - after this step, final col is produced in even col 0 of F0/F1
 */
// clang-format on
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _bitonic_top32_of_1024_rm_pre_sorted_final_(std::uint32_t dst_index) {
    constexpr bool decreasing = false;
    constexpr bool increasing = true;
    constexpr std::uint32_t odd_col_offset = 2;
    const std::uint32_t tile_offset = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];

    /// Step 1
    // Merge even and odd cols and rebuild, then store to odd cols
    std::uint32_t dist = odd_col_offset;
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int d = 0; d < 4; d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, dist);
        bitonic_top32_step_N(decreasing);
        bitonic_top32_store16<is_fp32_dest_acc_en, false>(4, dist);
        bitonic_top32_inc_x8_dest(8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    // set every SFPU instance to alternate SWAP direction
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x0104);
    TTI_SFPCONFIG(0x4444, 0xF, 8);
    for (int d = 0; d < 2; d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 16);
        bitonic_top32_step_N(decreasing);
        bitonic_top32_store16<is_fp32_dest_acc_en, false>(4, 16);
        bitonic_top32_inc_x8_dest(8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int d = 0; d < 2; d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
        bitonic_top32_ph3_st4_to_1(decreasing);
        set_dst_write_addr_offset(tile_offset + odd_col_offset);
        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
        set_dst_write_addr_offset(tile_offset);
    }

    /// Step 2
    // Shift odd cols by 1 SFPU instance right, and store to even cols
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TTI_SFPCONFIG(0x0000, 0xF, 1);  // disable SFPU config for shifting
    for (int d = 0; d < 2; d++) {
        set_dst_write_addr_offset(tile_offset + odd_col_offset);
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
        TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        set_dst_write_addr_offset(tile_offset);
        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
    }
    TTI_SFPCONFIG(0x0004, 0xF, 1);  // Restore index tracking mode

    /// Step 3
    // Merge even and odd cols and rebuild, then store to odd cols
    dist = odd_col_offset;
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int d = 0; d < 4; d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, dist);
        bitonic_top32_step_N(decreasing);
        bitonic_top32_store16<is_fp32_dest_acc_en, false>(4, dist);
        bitonic_top32_inc_x8_dest(8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    // set every 2 SFPU instances to alternate SWAP direction
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x0104);
    TTI_SFPCONFIG(0x5050, 0xF, 8);
    for (int d = 0; d < 2; d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 16);
        bitonic_top32_step_N(decreasing);
        bitonic_top32_store16<is_fp32_dest_acc_en, false>(4, 16);
        bitonic_top32_inc_x8_dest(8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int d = 0; d < 2; d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
        bitonic_top32_ph3_st4_to_1(decreasing);
        set_dst_write_addr_offset(tile_offset + odd_col_offset);
        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
        set_dst_write_addr_offset(tile_offset);
    }

    /// Step 4
    // Shift odd cols by 2 SFPU instances right, and store to even cols
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TTI_SFPCONFIG(0x0000, 0xF, 1);  // disable SFPU config for shifting
    for (int d = 0; d < 2; d++) {
        set_dst_write_addr_offset(tile_offset + odd_col_offset);
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
        for (int i = 0; i < 2; i++) {
            TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
            TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
            TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
            TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
            TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
            TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
            TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
            TTI_SFPSHFT2(0, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        }
        set_dst_write_addr_offset(tile_offset);
        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
    }
    TTI_SFPCONFIG(0x0004, 0xF, 1);  // Restore index tracking mode

    /// Step 5
    // Merge even and odd cols and rebuild, then store to odd cols
    dist = odd_col_offset;
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int d = 0; d < 4; d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, dist);
        bitonic_top32_step_N(decreasing);
        bitonic_top32_store16<is_fp32_dest_acc_en, false>(4, dist);
        bitonic_top32_inc_x8_dest(8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    // set every 4 SFPU instances to alternate SWAP direction
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x0104);
    TTI_SFPCONFIG(0x5500, 0xF, 8);
    for (int d = 0; d < 2; d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 16);
        bitonic_top32_step_N(decreasing);
        bitonic_top32_store16<is_fp32_dest_acc_en, false>(4, 16);
        bitonic_top32_inc_x8_dest(8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int d = 0; d < 2; d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
        bitonic_top32_ph3_st4_to_1(decreasing);
        set_dst_write_addr_offset(tile_offset + odd_col_offset);
        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
        set_dst_write_addr_offset(tile_offset);
    }

    /// Step 6
    // Shift odd cols by 4 SFPU instances right, and store to even cols
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TTI_SFPCONFIG(0x0000, 0xF, 1);  // disable SFPU config for shifting
    for (int d = 0; d < 2; d++) {
        set_dst_write_addr_offset(tile_offset + odd_col_offset);
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
        for (int i = 0; i < 4; i++) {
            TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
            TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
            TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
            TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
            TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
            TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
            TTI_SFPSHFT2(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
            TTI_SFPSHFT2(0, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        }
        set_dst_write_addr_offset(tile_offset);
        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
    }
    TTI_SFPCONFIG(0x0004, 0xF, 1);  // Restore index tracking mode

    /// Step 7
    // Merge even and odd cols and rebuild, then store to even cols
    dist = odd_col_offset;
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int d = 0; d < 4; d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, dist);
        bitonic_top32_step_N(decreasing);
        bitonic_top32_store16<is_fp32_dest_acc_en, false>(4, dist);
        bitonic_top32_inc_x8_dest(8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int d = 0; d < 2; d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 16);
        bitonic_top32_step_N(decreasing);
        bitonic_top32_store16<is_fp32_dest_acc_en, false>(4, 16);
        bitonic_top32_inc_x8_dest(8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int d = 0; d < 2; d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
        bitonic_top32_ph3_st4_to_1(decreasing);
        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
    }
}

inline void _top32_rm_configure_addrmod_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 16},
    }
        .set(ADDR_MOD_6);
}

inline void _top32_rm_init_() {
    _sfpu_load_config32_(0xF, 0x0, 0x4);  // Set bit [2] of the SFPU_CONTROL_REG to enable index tracking mode
    _top32_rm_configure_addrmod_();
}

}  // namespace sfpu
}  // namespace ckernel
