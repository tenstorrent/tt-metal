// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "sfpu/ckernel_sfpu_topk.h"
#include "lltt.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel {
namespace sfpu {

// TODO: Initial evaluation of using replay buffers here did not show any performance improvement
// Try re-evaluating with latest op sequence and record larger sequences

constexpr uint32_t dst_tile_offset = 64;  // 1 tile x 64 rows per tile
constexpr uint32_t scores_offset = 0;
constexpr uint32_t indices_offset = scores_offset + dst_tile_offset;
constexpr uint32_t bias_offset = indices_offset + dst_tile_offset;
constexpr uint32_t interm_offset = bias_offset + dst_tile_offset;

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load16_single_face() {
    // Load 16 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, bias_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, bias_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, bias_offset + 8);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, bias_offset + 12);

    constexpr uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, indices_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, indices_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, indices_offset + 8);
    TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, indices_offset + 12);
}

template <bool is_fp32_dest_acc_en, uint32_t offset = 0>
inline void bitonic_topk_load16_concat_indices_single_face() {
    // Load 16 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, bias_offset + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, bias_offset + 4 + offset);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, bias_offset + 8 + offset);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, bias_offset + 12 + offset);

    static_assert(!is_fp32_dest_acc_en, "is_fp32_dest_acc_en must be false");
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, indices_offset + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, indices_offset + 4 + offset);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, indices_offset + 8 + offset);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, indices_offset + 12 + offset);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, scores_offset + 0 + offset);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, scores_offset + 4 + offset);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, scores_offset + 8 + offset);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, scores_offset + 12 + offset);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load16_concatted_indices_single_face() {
    // Load 16 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, bias_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, bias_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, bias_offset + 8);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, bias_offset + 12);

    static_assert(!is_fp32_dest_acc_en, "is_fp32_dest_acc_en must be false");
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, interm_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, interm_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, interm_offset + 8);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, interm_offset + 12);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store16_concatted_indices_single_face() {
    // Store 16 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, bias_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, bias_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, bias_offset + 8);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, bias_offset + 12);

    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, interm_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, interm_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, interm_offset + 8);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, interm_offset + 12);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store16_single_face() {
    // Store 16 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, bias_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, bias_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, bias_offset + 8);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, bias_offset + 12);

    constexpr uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, indices_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, indices_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, indices_offset + 8);
    TTI_SFPSTORE(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, indices_offset + 12);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load8_even_cols_single_face() {
    // Load 8 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, bias_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, bias_offset + 4);

    constexpr uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, indices_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, indices_offset + 4);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load8_even_cols_concatted_indices_single_face() {
    // Load 8 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, bias_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, bias_offset + 4);

    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, interm_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, interm_offset + 4);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store8_even_cols_single_face() {
    // Store 8 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, bias_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, bias_offset + 4);

    constexpr uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, indices_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, indices_offset + 4);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store8_even_cols_split_indices_single_face() {
    static_assert(!is_fp32_dest_acc_en, "is_fp32_dest_acc_en must be true");
    // Store 8 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, bias_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, bias_offset + 4);

    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, indices_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, indices_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, scores_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, scores_offset + 4);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store8_even_cols_concatted_indices_single_face() {
    // Store 8 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, bias_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, bias_offset + 4);

    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, interm_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, interm_offset + 4);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load8_odd_cols_concatted_indices_single_face() {
    // Load 8 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, bias_offset + 2);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, bias_offset + 6);

    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, interm_offset + 2);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, interm_offset + 6);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store8_odd_cols_concatted_indices_single_face() {
    // Store 8 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, bias_offset + 2);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, bias_offset + 6);

    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, interm_offset + 2);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, interm_offset + 6);
}

template <bool start_transpose, bool end_transpose>
inline void bitonic_topk_ph0_st1_to_1_single_face() {
    if constexpr (start_transpose) {
        TTI_SFPTRANSP(0, 0, 0, 0);
    }

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

    if constexpr (end_transpose) {
        TTI_SFPTRANSP(0, 0, 0, 0);
    }
}

template <bool start_transpose, bool end_transpose>
inline void bitonic_topk_ph1_st2_to_1_single_face() {
    if constexpr (start_transpose) {
        TTI_SFPTRANSP(0, 0, 0, 0);
    }

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    if constexpr (end_transpose) {
        TTI_SFPTRANSP(0, 0, 0, 0);
    }
}

template <bool end_transpose, bool bitonic = true>
inline void bitonic_topk_ph2_st3_to_1_single_face() {
    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    if constexpr (bitonic) {
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    } else {
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    }

    TTI_SFPTRANSP(0, 0, 0, 0);

    constexpr int swap_mode = bitonic ? p_sfpswap::ROWS_01_MAX : p_sfpswap::ALL_ROWS_MAX;

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, swap_mode);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, swap_mode);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, swap_mode);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, swap_mode);

    if constexpr (end_transpose) {
        TTI_SFPTRANSP(0, 0, 0, 0);
    }
}

template <bool dir, bool end_transpose>
inline void bitonic_top8_ph3_st4_to_1() {
    // TODO: Use replay buffer for these instructions
    if constexpr (dir == (bool)SortDir::ArgMax) {
        // Step 4
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

        // Step 3
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPTRANSP(0, 0, 0, 0);

        // Step 2
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

        // Step 1
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        if constexpr (end_transpose) {
            TTI_SFPTRANSP(0, 0, 0, 0);
        }
    } else {
        // Step 4
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

        // Step 3
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);

        // Step 2
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

        // Step 1
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

        if constexpr (end_transpose) {
            TTI_SFPTRANSP(0, 0, 0, 0);
        }
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool idir>
inline void bitonic_top8_ph0_to_ph3() {
    // Phase 0
    {
        constexpr bool start_transpose = true;
        constexpr bool end_transpose = false;
        constexpr int phase_replay_count = 2 + (int)start_transpose + (int)end_transpose;
        bitonic_topk_ph0_st1_to_1_single_face<start_transpose, end_transpose>();
    }
    // Phase 1
    {
        constexpr bool start_transpose = false;
        constexpr bool end_transpose = true;
        constexpr int phase_replay_count = 4 + (int)start_transpose + (int)end_transpose;
        // Odd Columns
        bitonic_topk_ph1_st2_to_1_single_face<start_transpose, end_transpose>();
    }
    // Phase 2
    {
        constexpr bool end_transpose = true;
        constexpr int phase_replay_count = 7 + (int)end_transpose;
        // Even Columns
        bitonic_topk_ph2_st3_to_1_single_face<end_transpose>();
    }
    // Modified Phase 3 for top8
    {
        constexpr bool end_transpose = true;
        constexpr int phase_replay_count = 8 + (int)end_transpose;
        bitonic_top8_ph3_st4_to_1<idir, end_transpose>();
    }
}

void reverse_sort_order() {
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG3, p_sfpswap::UNCONDITIONALLY);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpswap::UNCONDITIONALLY);
    TTI_SFPTRANSP(0, 0, 0, 0);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _deepseek_moe_gate_sum_top2() {
    constexpr bool idir = false;  // Sort descending order
    constexpr int load_store_replay_count = 8;
    constexpr int load_replay_offset = 0;
    constexpr int store_replay_offset = load_replay_offset + load_store_replay_count;
    constexpr int phase_replay_offset = store_replay_offset + load_store_replay_count;

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Phase 0-3 Even Columns
    bitonic_topk_load16_concat_indices_single_face<is_fp32_dest_acc_en, 0>();
    bitonic_top8_ph0_to_ph3<APPROXIMATION_MODE, is_fp32_dest_acc_en, idir>();
    bitonic_topk_store8_even_cols_concatted_indices_single_face<is_fp32_dest_acc_en>();

    // Phase 0-3 Odd Columns
    bitonic_topk_load16_concat_indices_single_face<is_fp32_dest_acc_en, 2>();
    bitonic_top8_ph0_to_ph3<APPROXIMATION_MODE, is_fp32_dest_acc_en, !idir>();

    // Instead of a full phase 4, we rerun phase 3 since we are only comparing top8 values
    bitonic_topk_load8_even_cols_concatted_indices_single_face<is_fp32_dest_acc_en>();
    bitonic_top8_ph3_st4_to_1<idir, true>();
    bitonic_topk_store8_even_cols_split_indices_single_face<is_fp32_dest_acc_en>();

    // Sum top2
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPNOP;
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Replicate the top2 sum down the column
    // This is needed for the subsequent top4 sort
    // TODO: Evaluate compared to broadcast using MOVB2D in FPU
    TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);
    TTI_SFPMOV(0, p_sfpu::LREG14, p_sfpu::LREG0, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, interm_offset);
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, interm_offset + 4);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _deepseek_moe_gate_sort_top4_groups() {
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    // Sort top4
    // Load the top2 sums and concat indices
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, interm_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, interm_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, indices_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, indices_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, scores_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, scores_offset + 4);

    // Load the top2 sums (again) and bias scores
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
    TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
    TTI_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_7, bias_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG7, 0, ADDR_MOD_7, bias_offset + 4);

    // Sort 8 groups (not bitonic)
    bitonic_topk_ph0_st1_to_1_single_face<true, false>();
    bitonic_topk_ph1_st2_to_1_single_face<false, true>();
    bitonic_topk_ph2_st3_to_1_single_face<true, false>();
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, indices_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, scores_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG6, 0, ADDR_MOD_7, bias_offset + 0);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _deepseek_moe_gate_top8(uint32_t eps, uint32_t scale) {
    constexpr bool idir = false;  // Sort descending order

    // Combine and sort 4 groups of 8 values to 2 groups of 8 values
    // Even Columns sorted Top8 in LREG0 and LREG1
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Reverse order of sort for top8 values in odd columns
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, bias_offset + 6);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, bias_offset + 2);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, indices_offset + 6);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, indices_offset + 2);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, scores_offset + 6);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, scores_offset + 2);
    reverse_sort_order();

    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, bias_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, bias_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, indices_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, indices_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, scores_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, scores_offset + 4);
    bitonic_top8_ph3_st4_to_1<idir, true>();

    bitonic_topk_store8_even_cols_concatted_indices_single_face<is_fp32_dest_acc_en>();

    // Move and reverse the other column of 8 values
    // Disable index tracking while we shift values
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    // There is a hardware bug that affects operations on LREG4-7 while this is enabled
    // Note this overwrites LREG0
    _sfpu_load_config32_(0xF, 0x0, 0x0);
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG6, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);

    // Re-enable index tracking
    _sfpu_load_config32_(0xF, 0x0, 0x4);
    reverse_sort_order();
    bitonic_topk_load8_even_cols_concatted_indices_single_face<is_fp32_dest_acc_en>();

    // Step 4 Only, we need top8 but it doesn't have to be sorted
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, indices_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::LO16_ONLY, ADDR_MOD_7, indices_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, scores_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, scores_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, scores_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, scores_offset + 4);

    // Reduce the top8 values to 1 value
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);

    // Calculate 1 / (sum + eps) * scale
    sfpi::vFloat l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vFloat eps_value = Converter::as_float(eps);
    l0 = l0 + eps_value;
    l0 = sfpu_reciprocal<APPROXIMATION_MODE>(l0);
    sfpi::vFloat scale_value = Converter::as_float(scale);
    l0 = l0 * scale_value;
    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    TTI_SFPNOP;

    // Broadcast to all rows and multiply by the top8 values
    TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, scores_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, scores_offset + 4);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, scores_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, scores_offset + 4);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _init_deepseek_moe_gate_topk() {
    _sfpu_load_config32_(0xF, 0x0, 0x4);  // Set bit [2] of the SFPU_CONTROL_REG to enable index tracking mode
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
