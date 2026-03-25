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

inline void bitonic_top32_inc_x8_dest(std::uint32_t inc, bool cr) {
    std::uint32_t inc_grp8 = inc >> 3;
    if (cr) {
        for (std::uint32_t i = 0; i < inc_grp8; i++) {
            TTI_INCRWC(0b100, 8, 0, 0);
        }
    } else {
        for (std::uint32_t i = 0; i < inc_grp8; i++) {
            TTI_INCRWC(0, 8, 0, 0);
        }
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
        std::uint32_t dist = (ss == 5) ? 16 : 32;
        std::uint32_t inner_d =
            dist >> 3;  // How many loops to sort the sequence of length (2^ss / 16). Each loop sorts 16
        datums_compared = 0;
        while (datums_compared < total_datums_to_compare) {
            for (std::uint32_t ii = 0; ii < inner_d; ii++) {
                bitonic_top32_load16<is_fp32_dest_acc_en>(4, dist);
                bitonic_top32_step_N(dir);
                bitonic_top32_store16<is_fp32_dest_acc_en, false>(4, dist);
                std::uint32_t dst_inc = 8;
                bool dst_cr = false;
                if (ii == (inner_d - 1)) {
                    dst_cr = true;
                    dst_inc = 2 * dist;
                }
                bitonic_top32_inc_x8_dest(dst_inc, dst_cr);
                datums_compared += 16;
            }
            dir = (datums_compared == sorted_seq_length) ? !dir : dir;
        }
    }
    // steps 4 to 1
    dir = idir;
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    datums_compared = 0;
    while (datums_compared < total_datums_to_compare) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
        bitonic_top32_ph3_st4_to_1(dir);
        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
        datums_compared += 16;
        dir = (datums_compared == sorted_seq_length) ? !dir : dir;
    }

    // produce final sequence len=64
    num_steps = 6;  // log(64)
    start_step = num_steps;
    end_step = 4;
    sorted_seq_length = 1 << num_steps;  // 64
    datums_compared = 0;
    total_datums_to_compare = 64;
    for (std::uint32_t ss = start_step; ss > end_step; ss--) {
        // Steps N to 5
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        dir = idir;
        std::uint32_t dist = (ss == 5) ? 16 : 32;
        std::uint32_t inner_d =
            dist >> 3;  // How many loops to sort the sequence of length (2^ss / 16). Each loop sorts 16
        datums_compared = 0;
        while (datums_compared < total_datums_to_compare) {
            for (std::uint32_t ii = 0; ii < inner_d; ii++) {
                bitonic_top32_load16<is_fp32_dest_acc_en>(4, dist);
                bitonic_top32_step_N(dir);
                bitonic_top32_store16<is_fp32_dest_acc_en, false>(4, dist);
                std::uint32_t dst_inc = 8;
                bool dst_cr = false;
                if (ii == (inner_d - 1)) {
                    dst_cr = true;
                    dst_inc = 2 * dist;
                }
                bitonic_top32_inc_x8_dest(dst_inc, dst_cr);
                datums_compared += 16;
            }
            dir = (datums_compared == sorted_seq_length) ? !dir : dir;
        }
    }
    // steps 4 to 1
    dir = idir;
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    datums_compared = 0;
    while (datums_compared < total_datums_to_compare) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
        bitonic_top32_ph3_st4_to_1(dir);
        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
        datums_compared += 16;
        dir = (datums_compared == sorted_seq_length) ? !dir : dir;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool top_min>
inline void _bitonic_top32_merge_() {
    std::uint32_t dist = 64;

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int d = 0; d < 8; d++) {
        bitonic_top32_load16<is_fp32_dest_acc_en>(4, dist);
        bitonic_top32_step_N(top_min);
        bitonic_top32_store16<is_fp32_dest_acc_en, false>(4, dist);
        bitonic_top32_inc_x8_dest(8, false);
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _bitonic_top32_rebuild_(
    const bool idir, const int m_iter, const int k, const int logk, const int skip_second) {
    std::uint32_t dst_addr_offset = 0;
    for (int face = 0; face < 2; face++) {
        for (int col = 0; col < 2; col++) {
            std::uint32_t total_datums_shift = (skip_second & 0x1);
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
            std::uint32_t rebuild_m = m_iter + 1;
            std::uint32_t total_datums_to_compare =
                ((64 >> rebuild_m) < 2 * k)
                    ? 2 * k
                    : (64 >>
                       rebuild_m);  // max(2*k, 64/(2^m)) total datums to compare; there's always at least 2*K datums
            total_datums_to_compare = total_datums_to_compare >> total_datums_shift;  // Reduce by 2 if skipping last
            std::uint32_t dist = (k << rebuild_m) > 32 ? 32 : (k << rebuild_m);       // min(32, k*2^k)
            std::uint32_t ld_offset = (dist >> 4) * 32 + (dist & 0xF);
            std::uint32_t ld_dist;
            int ph = logk - 1;
            bool dir = idir;
            std::uint32_t datums_compared = 0;

            switch (ph) {
                case 0: break;
                case 1:
                    if (m_iter >= 2) {
                        while (datums_compared < total_datums_to_compare) {
                            // Groups of 8 datums being sorted at the same time
                            bitonic_top32_load8<is_fp32_dest_acc_en>(0, ld_offset);
                            bitonic_top32_ph1_st2_to_1();
                            bitonic_top32_store8<is_fp32_dest_acc_en>(0, ld_offset);
                            bitonic_top32_inc_x8_dest(64, false);
                            datums_compared += 16;
                        }
                        break;
                    } else {
                        ld_dist = (ld_offset < 16) ? 4 * ld_offset : 2 * ld_offset;
                        while (datums_compared < total_datums_to_compare) {
                            bitonic_top32_load16<is_fp32_dest_acc_en>(ld_offset, ld_dist);
                            bitonic_top32_ph1_st2_to_1();
                            bitonic_top32_store16<is_fp32_dest_acc_en, true>(ld_offset, ld_dist);
                            TTI_INCRWC(0, 8, 0, 0);
                            TTI_INCRWC(0, 8, 0, 0);
                            TTI_INCRWC(0, 8, 0, 0);
                            TTI_INCRWC(0, 8, 0, 0);
                            datums_compared += 16;
                        }
                        break;
                    }
                case 2:
                    while (datums_compared < total_datums_to_compare) {
                        bitonic_top32_load16<is_fp32_dest_acc_en>(4, ld_offset);
                        bitonic_top32_ph2_st3_to_1();
                        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, ld_offset);
                        TTI_INCRWC(0, 8, 0, 0);
                        TTI_INCRWC(0, 8, 0, 0);
                        TTI_INCRWC(0, 8, 0, 0);
                        TTI_INCRWC(0, 8, 0, 0);
                        datums_compared += 16;
                    }
                    break;
                case 3:
                    while (datums_compared < total_datums_to_compare) {
                        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
                        bitonic_top32_ph3_st4_to_1(dir);
                        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
                        TTI_INCRWC(0, 8, 0, 0);
                        TTI_INCRWC(0, 8, 0, 0);
                        TTI_INCRWC(0, 8, 0, 0);
                        TTI_INCRWC(0, 8, 0, 0);
                        datums_compared += 16;
                        dir = !dir;
                    }
                    break;
                default:
                    std::uint32_t num_steps = ph + 1;
                    std::uint32_t start_step = num_steps;
                    std::uint32_t end_step = 4;
                    std::uint32_t sorted_seq_length = 1 << num_steps;
                    std::uint32_t total_datums_to_compare = 64;
                    for (std::uint32_t ss = start_step; ss > end_step; ss--) {
                        // Steps N to 5
                        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                        dir = idir;
                        datums_compared = 0;
                        std::uint32_t dist = (ss == 5) ? 16 : 32;
                        std::uint32_t inner_d =
                            dist >> 3;  // How many loops to sort the sequence of length (2^ss / 16). Each loop sorts 16
                        std::uint32_t dst_offset = 0;
                        while (datums_compared < total_datums_to_compare) {
                            for (std::uint32_t ii = 0; ii < inner_d; ii++) {
                                bitonic_top32_load16<is_fp32_dest_acc_en>(
                                    4, 2 * dist);  // load/store with offset of face 1 (in row major face layout)
                                bitonic_top32_step_N(dir);
                                bitonic_top32_store16<is_fp32_dest_acc_en, false>(
                                    4, 2 * dist);  // load/store with offset of face 1 (in row major face layout)
                                std::uint32_t dst_inc = 8;
                                dst_offset += dst_inc;
                                bool dst_cr = false;
                                if (ii == (inner_d - 1)) {
                                    dst_cr = true;
                                    dst_inc = 4 * dist;
                                    dst_offset = 2 * dist;
                                } else if (dst_offset == 16) {
                                    dst_cr = true;
                                    dst_inc = 32;
                                }
                                bitonic_top32_inc_x8_dest(dst_inc, dst_cr);
                                datums_compared += 16;
                            }
                            dir = (datums_compared == sorted_seq_length)
                                      ? !dir
                                      : dir;  // total_sorted = total_loops * 16; if total_sorted == sorted_seq_length
                        }
                    }
                    // steps 4 to 1
                    dir = idir;
                    datums_compared = 0;
                    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                    while (datums_compared < total_datums_to_compare) {
                        bitonic_top32_load16<is_fp32_dest_acc_en>(4, 8);
                        bitonic_top32_ph3_st4_to_1(dir);
                        bitonic_top32_store16<is_fp32_dest_acc_en, true>(4, 8);
                        datums_compared += 16;
                        dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                    }
            }

            dst_addr_offset += 2;
            set_dst_write_addr_offset(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr_offset(dst_addr_offset);
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
