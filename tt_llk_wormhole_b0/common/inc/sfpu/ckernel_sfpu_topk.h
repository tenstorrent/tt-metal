// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_load_config.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

static int32_t topk_replay_init = 0;

inline void set_dst_write_addr(uint32_t addr)
{
    uint dst_index = addr + get_dest_buffer_base();
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load8(uint offset, uint dist)
{
    constexpr uint dst_indices_offset = 128; // 2 tile x 64 rows per tile
    constexpr uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    uint face_offset = offset >> 4;
    uint ld_offset   = (offset & 0xF) + face_offset * 32;

    // Load 16 consecutive numbers
    TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, ld_offset);
    TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, ld_offset + dist);

    // Load 16 consecutive indices
    TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, dst_indices_offset + ld_offset);
    TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, dst_indices_offset + ld_offset + dist);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store8(uint offset, uint dist)
{
    constexpr uint dst_indices_offset = 128; // 2 tile x 64 rows per tile
    constexpr uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    uint face_offset = offset >> 4;
    uint ld_offset   = (offset & 0xF) + face_offset * 32;

    // Load 16 consecutive numbers
    TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, ld_offset);
    TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, ld_offset + dist);

    // Load 16 consecutive indices
    TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, dst_indices_offset + ld_offset + 0);
    TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, dst_indices_offset + ld_offset + dist);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load16(uint dist0, uint dist1)
{
    constexpr uint dst_indices_offset = 128; // 2 tile x 64 rows per tile
    constexpr uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    // Load 16 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, 4);
        TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_3, 8);
        TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_3, 12);
    }
    else
    {
        TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, 0 + dist0);
        TT_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_3, dist1);
        TT_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_3, dist1 + dist0);
    }

    // Load 16 consecutive indices
    TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 4);
        TTI_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 8);
        TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 12);
    }
    else
    {
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 0 + dist0);
        TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, dst_indices_offset + dist1);
        TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, dst_indices_offset + dist1 + dist0);
    }
}

template <bool is_fp32_dest_acc_en, bool alt_addr_mod = false>
inline void bitonic_topk_store16(uint dist0, uint dist1)
{
    constexpr uint dst_indices_offset = 128; // 2 tile x 64 rows per tile
    constexpr uint8_t instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;

    // Load 16 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 4);
        TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_3, 8);
        TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_3, 12);
    }
    else
    {
        TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 0 + dist0);
        TT_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_3, dist1);
        TT_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_3, dist1 + dist0);
    }

    // Load 16 consecutive indices
    TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 4);
        TTI_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 8);
        TTI_SFPSTORE(p_sfpu::LREG7, instr_mod_index, alt_addr_mod ? ADDR_MOD_2 : ADDR_MOD_3, dst_indices_offset + 12);
    }
    else
    {
        TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 0 + dist0);
        TT_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, dst_indices_offset + dist1);
        TT_SFPSTORE(p_sfpu::LREG7, instr_mod_index, alt_addr_mod ? ADDR_MOD_2 : ADDR_MOD_3, dst_indices_offset + dist1 + dist0);
    }
}

inline void bitonic_topk_ph3_st4_to_1(bool dir, bool &init_replay, int replay_start)
{
    if (dir == (bool)SortDir::ArgMin)
    {
        TTI_SFPCONFIG(0x104, 0xF, 1); // Reverse the max/min behaviour of SWAP
        TTI_SFPNOP;
        TTI_SFPNOP;
    }

    if (init_replay)
    {
        lltt::record<lltt::Exec>(replay_start, 5);

        // Step 4
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

        // Step 3
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);

        init_replay = false;
    }
    else
    {
        lltt::replay(replay_start, 5);
    }

    lltt::replay(replay_start, 5);

    if (dir == (bool)SortDir::ArgMin)
    {
        TTI_SFPCONFIG(0x004, 0xF, 1); // Restore the max/min behaviour of SWAP
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
}

inline void bitonic_topk_ph2_st3_to_1()
{
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

inline void bitonic_topk_ph1_st2_to_1()
{
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

inline void bitonic_topk_ph0_st1_to_1()
{
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::UNCONDITIONALLY);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

inline void bitonic_topk_step_N(bool dir)
{
    // Step N
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    if (dir == (bool)SortDir::ArgMin)
    {
        // Min
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::UNCONDITIONALLY);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::UNCONDITIONALLY);
    }
    // MT: Maybe there's a way to optimize out unconditional transpose at every step. Think about this
}

inline void bitonic_topk_inc_x8_dest(uint inc, bool cr)
{
    uint inc_grp8 = inc >> 3;
    for (uint i = 0; i < inc_grp8; i++)
    {
        if (cr)
        {
            TTI_INCRWC(0b100, 8, 0, 0);
        }
        else
        {
            TTI_INCRWC(0, 8, 0, 0);
        }
    }
}

inline void bitonic_topk_inc_x4_dest(uint inc, bool cr)
{
    uint inc_grp4 = inc >> 2;
    for (uint i = 0; i < inc_grp4; i++)
    {
        if (cr)
        {
            TTI_INCRWC(0b100, 4, 0, 0);
        }
        else
        {
            TTI_INCRWC(0, 4, 0, 0);
        }
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void _bitonic_topk_phases_steps(const int idir, const int i_end_phase, const int i_start_phase, const int i_end_step, const int i_start_step)
{
    // If more than 1 phase is requested, do all the steps from all phases
    // If 1 phase is requested, use i_start_step/i_end_step parameters

    // init the replay buffer for local sort if uninitialized
    bool init_load  = (topk_replay_init >= 0) ? true : false;
    bool init_store = (topk_replay_init >= 0) ? true : false;
    bool init_phase;

    uint dst_addr_offset = 0;
    for (int face = 0; face < 2; face++)
    {
        for (int col = 0; col < 2; col++)
        {
            bool dir = idir;
            for (int ph = i_start_phase; ph < (i_end_phase + 1); ph++)
            {
                init_phase = true; // init each new phase of local sort in replay buffer

                TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                switch (ph)
                {
                    case 0:
                        for (int d = 0; d < 4; d++)
                        {
                            // Groups of 16 datums being sorted at the same time
                            if (init_load)
                            {
                                lltt::record<lltt::Exec>(0, 8);
                                bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                                init_load = false;
                            }
                            else
                            {
                                lltt::replay(0, 8);
                            }
                            if (init_phase)
                            {
                                lltt::record<lltt::Exec>(16, 5);
                                bitonic_topk_ph0_st1_to_1();
                                init_phase = false;
                            }
                            else
                            {
                                lltt::replay(16, 5);
                            }
                            if (init_store)
                            {
                                lltt::record<lltt::Exec>(8, 8);
                                bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                                init_store = false;
                            }
                            else
                            {
                                lltt::replay(8, 8);
                            }
                        }
                        break;
                    case 1:
                        for (int d = 0; d < 4; d++)
                        {
                            // Groups of 16 datums being sorted at the same time
                            lltt::replay(0, 8);
                            if (init_phase)
                            {
                                lltt::record<lltt::Exec>(16, 6);
                                bitonic_topk_ph1_st2_to_1();
                                init_phase = false;
                            }
                            else
                            {
                                lltt::replay(16, 6);
                            }
                            lltt::replay(8, 8);
                        }
                        break;
                    case 2:
                        for (int d = 0; d < 4; d++)
                        {
                            lltt::replay(0, 8);
                            if (init_phase)
                            {
                                lltt::record<lltt::Exec>(16, 9);
                                bitonic_topk_ph2_st3_to_1();
                                init_phase = false;
                            }
                            else
                            {
                                lltt::replay(16, 9);
                            }
                            lltt::replay(8, 8);
                        }
                        break;
                    case 3:
                        for (int d = 0; d < 4; d++)
                        {
                            lltt::replay(0, 8);
                            bitonic_topk_ph3_st4_to_1(dir, init_phase, 16);
                            lltt::replay(8, 8);
                            dir = !dir;
                        }
                        break;
                    default:
                        uint num_steps               = ph + 1;
                        uint start_step              = (i_start_phase == i_end_phase) ? i_start_step : num_steps;
                        uint end_step                = (i_start_phase == i_end_phase) ? i_end_step : 4;
                        uint sorted_seq_length       = 1 << num_steps;
                        uint datums_compared         = 0;
                        uint total_datums_to_compare = 64;
                        for (uint ss = start_step; ss > end_step; ss--)
                        {
                            // Steps N to 5
                            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                            dir             = idir;
                            uint dist       = (ss == 5) ? 16 : 32;
                            uint inner_d    = dist >> 3; // How many loops to sort the sequence of length (2^ss / 16). Each loop sorts 16
                            datums_compared = 0;
                            uint dst_offset = 0;
                            while (datums_compared < total_datums_to_compare)
                            {
                                for (uint ii = 0; ii < inner_d; ii++)
                                {
                                    bitonic_topk_load16<is_fp32_dest_acc_en>(4, 2 * dist); // load/store with offset of face 1 (in row major face layout)
                                    bitonic_topk_step_N(dir);
                                    bitonic_topk_store16<is_fp32_dest_acc_en, false>(
                                        4, 2 * dist); // load/store with offset of face 1 (in row major face layout)
                                    uint dst_inc = 8;
                                    dst_offset += dst_inc;
                                    bool dst_cr = false;
                                    if (ii == (inner_d - 1))
                                    {
                                        dst_cr     = true;
                                        dst_inc    = 4 * dist;
                                        dst_offset = 2 * dist;
                                    }
                                    else if (dst_offset == 16)
                                    {
                                        dst_cr  = true;
                                        dst_inc = 32;
                                    }
                                    bitonic_topk_inc_x8_dest(dst_inc, dst_cr);
                                    datums_compared += 16;
                                }
                                dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                            }
                        }
                        // steps 4 to 1
                        dir = idir;
                        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                        datums_compared = 0;
                        while (datums_compared < total_datums_to_compare)
                        {
                            lltt::replay(0, 8);
                            bitonic_topk_ph3_st4_to_1(dir, init_phase, 16);
                            lltt::replay(8, 8);
                            datums_compared += 16;
                            dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                        }
                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
    topk_replay_init = -1;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool top_min, int ITERATIONS>
inline void _bitonic_topk_merge(const int m_iter, const int k)
{
    uint dst_addr_offset = 0;
    for (int face = 0; face < 2; face++)
    {
        for (int col = 0; col < 2; col++)
        {
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
            int k_max                    = k > 32 ? 32 : k;
            uint inner_d                 = k_max >> 2; // inner loop comparisons to sort len=K sequence;
            uint total_datums_to_compare = ((64 >> m_iter) < 2 * k_max)
                                               ? 2 * k_max
                                               : (64 >> m_iter); // max(2, max(64, 64/(2^m))) total datums to compare; there's always at least 2*K datums
            uint dist                    = (k_max << m_iter) > 32 ? 32 : (k_max << m_iter); // min(32, k*2^k)
            uint ld_dist                 = (dist < 16) ? dist : 2 * dist;                   // Accounts for face offsets within a tile
            uint datums_compared         = 0;
            uint dst_offset              = 0;
            uint dst_cr                  = 0;

            while (datums_compared < total_datums_to_compare)
            {
                for (uint ii = 0; ii < inner_d; ii++)
                {
                    bitonic_topk_load8<is_fp32_dest_acc_en>(dst_offset, ld_dist);
                    TTI_SFPSWAP(0, top_min ? p_sfpu::LREG1 : p_sfpu::LREG0, top_min ? p_sfpu::LREG0 : p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                    bitonic_topk_store8<is_fp32_dest_acc_en>(dst_offset, ld_dist);
                    datums_compared += 8;
                    if (ii == (inner_d - 1))
                    {
                        dst_cr += 2 * dist;
                        dst_offset = dst_cr;
                    }
                    else
                    {
                        dst_offset += 4;
                    }
                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void _bitonic_topk_rebuild(const bool idir, const int m_iter, const int k, const int logk, const int skip_second)
{
    // init replay buffer for rebuild interation 'm_iter' if uninitialized
    bool init_rebuild = (topk_replay_init != m_iter + 1) ? true : false;

    uint dst_addr_offset = 0;
    for (int face = 0; face < 2; face++)
    {
        for (int col = 0; col < 2; col++)
        {
            uint total_datums_shift = (skip_second & 0x1);
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
            uint rebuild_m = m_iter + 1;
            uint total_datums_to_compare =
                ((64 >> rebuild_m) < 2 * k) ? 2 * k : (64 >> rebuild_m); // max(2*k, 64/(2^m)) total datums to compare; there's always at least 2*K datums
            total_datums_to_compare = total_datums_to_compare >> total_datums_shift; // Reduce by 2 if skipping last
            uint dist               = (k << rebuild_m) > 32 ? 32 : (k << rebuild_m); // min(32, k*2^k)
            uint ld_offset          = (dist >> 4) * 32 + (dist & 0xF);
            uint ld_dist;
            int ph               = logk - 1;
            bool dir             = idir;
            uint datums_compared = 0;

            switch (ph)
            {
                case 0:

                    break;
                case 1:
                    if (m_iter >= 2)
                    {
                        while (datums_compared < total_datums_to_compare)
                        {
                            // Groups of 8 datums being sorted at the same time
                            if (init_rebuild)
                            {
                                lltt::record<lltt::Exec>(0, 22);
                                bitonic_topk_load8<is_fp32_dest_acc_en>(0, ld_offset);
                                bitonic_topk_ph1_st2_to_1();
                                bitonic_topk_store8<is_fp32_dest_acc_en>(0, ld_offset);
                                bitonic_topk_inc_x8_dest(64, false);
                                init_rebuild = false;
                            }
                            else
                            {
                                lltt::replay(0, 22);
                            }
                            datums_compared += 16;
                        }
                        break;
                    }
                    else
                    {
                        ld_dist = (ld_offset < 16) ? 4 * ld_offset : 2 * ld_offset;
                        while (datums_compared < total_datums_to_compare)
                        {
                            // Groups of 16 datums being sorted at the same time
                            if (init_rebuild)
                            {
                                lltt::record<lltt::Exec>(0, 26);
                                bitonic_topk_load16<is_fp32_dest_acc_en>(ld_offset, ld_dist);
                                bitonic_topk_ph1_st2_to_1();
                                bitonic_topk_store16<is_fp32_dest_acc_en, true>(ld_offset, ld_dist);
                                TTI_INCRWC(0, 8, 0, 0);
                                TTI_INCRWC(0, 8, 0, 0);
                                TTI_INCRWC(0, 8, 0, 0);
                                TTI_INCRWC(0, 8, 0, 0);
                                init_rebuild = false;
                            }
                            else
                            {
                                lltt::replay(0, 26);
                            }
                            datums_compared += 16;
                        }
                        break;
                    }
                case 2:
                    while (datums_compared < total_datums_to_compare)
                    {
                        // Groups of 16 datums being sorted at the same time
                        if (init_rebuild)
                        {
                            lltt::record<lltt::Exec>(0, 29);
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, ld_offset);
                            bitonic_topk_ph2_st3_to_1();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, ld_offset);
                            TTI_INCRWC(0, 8, 0, 0);
                            TTI_INCRWC(0, 8, 0, 0);
                            TTI_INCRWC(0, 8, 0, 0);
                            TTI_INCRWC(0, 8, 0, 0);
                            init_rebuild = false;
                        }
                        else
                        {
                            lltt::replay(0, 29);
                        }
                        datums_compared += 16;
                    }
                    break;
                case 3:
                    while (datums_compared < total_datums_to_compare)
                    {
                        // Groups of 16 datums being sorted at the same time
                        if (init_rebuild)
                        {
                            lltt::record<lltt::Exec>(0, 8);
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                            bitonic_topk_ph3_st4_to_1(dir, init_rebuild, 8);
                            lltt::record<lltt::Exec>(13, 12);
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                            TTI_INCRWC(0, 8, 0, 0);
                            TTI_INCRWC(0, 8, 0, 0);
                            TTI_INCRWC(0, 8, 0, 0);
                            TTI_INCRWC(0, 8, 0, 0);
                        }
                        else
                        {
                            lltt::replay(0, 8);
                            bitonic_topk_ph3_st4_to_1(dir, init_rebuild, 8);
                            lltt::replay(13, 12);
                        }
                        datums_compared += 16;
                        dir = !dir;
                    }
                    break;
                default:
                    uint num_steps               = ph + 1;
                    uint start_step              = num_steps;
                    uint end_step                = 4;
                    uint sorted_seq_length       = 1 << num_steps;
                    uint total_datums_to_compare = 64;
                    for (uint ss = start_step; ss > end_step; ss--)
                    {
                        // Steps N to 5
                        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                        dir             = idir;
                        datums_compared = 0;
                        uint dist       = (ss == 5) ? 16 : 32;
                        uint inner_d    = dist >> 3; // How many loops to sort the sequence of length (2^ss / 16). Each loop sorts 16
                        uint dst_offset = 0;
                        while (datums_compared < total_datums_to_compare)
                        {
                            for (uint ii = 0; ii < inner_d; ii++)
                            {
                                bitonic_topk_load16<is_fp32_dest_acc_en>(4, 2 * dist); // load/store with offset of face 1 (in row major face layout)
                                bitonic_topk_step_N(dir);
                                bitonic_topk_store16<is_fp32_dest_acc_en, false>(4, 2 * dist); // load/store with offset of face 1 (in row major face layout)
                                uint dst_inc = 8;
                                dst_offset += dst_inc;
                                bool dst_cr = false;
                                if (ii == (inner_d - 1))
                                {
                                    dst_cr     = true;
                                    dst_inc    = 4 * dist;
                                    dst_offset = 2 * dist;
                                }
                                else if (dst_offset == 16)
                                {
                                    dst_cr  = true;
                                    dst_inc = 32;
                                }
                                bitonic_topk_inc_x8_dest(dst_inc, dst_cr);
                                datums_compared += 16;
                            }
                            dir = (datums_compared == sorted_seq_length) ? !dir : dir; // total_sorted = total_loops * 16; if total_sorted == sorted_seq_length
                        }
                    }
                    // steps 4 to 1
                    dir             = idir;
                    datums_compared = 0;
                    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                    while (datums_compared < total_datums_to_compare)
                    {
                        if (init_rebuild)
                        {
                            lltt::record<lltt::Exec>(0, 8);
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                            bitonic_topk_ph3_st4_to_1(dir, init_rebuild, 8);
                            lltt::record<lltt::Exec>(13, 8);
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                        }
                        else
                        {
                            lltt::replay(0, 8);
                            bitonic_topk_ph3_st4_to_1(dir, init_rebuild, 8);
                            lltt::replay(13, 8);
                        }
                        datums_compared += 16;
                        dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                    }
            }

            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
    topk_replay_init = m_iter + 1;
}

inline void _init_topk()
{
    topk_replay_init = 0;
    _sfpu_load_config32_(0xF, 0x0, 0x4); // Set bit [2] of the SFPU_CONTROL_REG to enable index tracking mode
}

} // namespace sfpu
} // namespace ckernel
