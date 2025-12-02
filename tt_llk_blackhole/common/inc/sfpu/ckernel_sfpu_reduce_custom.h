// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

//**************************************************************
// SFPU REDUCE MAX COL IMPLEMENTATION
//**************************************************************
inline void sfpu_reduce_max_col_subblock_4x2_configure_addrmod()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_3);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 64},
    }
        .set(ADDR_MOD_2);
}

inline void sfpu_reduce_max_col_subblock_4x2_load_initial_values()
{
    constexpr uint16_t neg_inf_fp16b = 0xFF80;

    // F0 - Initialize with negative infinity
    TTI_SFPLOADI(p_sfpu::LREG4, InstrModLoadStore::FP16B, neg_inf_fp16b);
    TTI_SFPLOADI(p_sfpu::LREG5, InstrModLoadStore::FP16B, neg_inf_fp16b);

    // F1 - Initialize with negative infinity
    TTI_SFPLOADI(p_sfpu::LREG6, InstrModLoadStore::FP16B, neg_inf_fp16b);
    TTI_SFPLOADI(p_sfpu::LREG7, InstrModLoadStore::FP16B, neg_inf_fp16b);
}

template <DataFormat format>
inline void _init_reduce_max_col_subblock_4x2_()
{
    static_assert(format == DataFormat::Float16_b, "Unsupported data format. Supported formats: Float16_b");

    _init_sfpu_config_reg();
    sfpu_reduce_max_col_subblock_4x2_configure_addrmod();

    // ***********************************************************
    // Record replay buffer
    lltt::record<lltt::NoExec>(0, 8);

    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);
    TTI_SFPSWAP(0 /*unused*/, p_sfpu::LREG4 /*lreg_src_c*/, p_sfpu::LREG2 /*lreg_dest*/, 1 /*instr_mod1*/);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_3, 2);
    TTI_SFPSWAP(0 /*unused*/, p_sfpu::LREG5 /*lreg_src_c*/, p_sfpu::LREG3 /*lreg_dest*/, 1 /*instr_mod1*/);

    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_3, 16);
    TTI_SFPSWAP(0 /*unused*/, p_sfpu::LREG6 /*lreg_src_c*/, p_sfpu::LREG2 /*lreg_dest*/, 1 /*instr_mod1*/);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_3, 18);
    TTI_SFPSWAP(0 /*unused*/, p_sfpu::LREG7 /*lreg_src_c*/, p_sfpu::LREG3 /*lreg_dest*/, 1 /*instr_mod1*/);

    // ***********************************************************
}

inline void _move_to_next_subblock_4x2_()
{
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_3, -128 & 0x3fff); // wherever
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_3, -126 & 0x3fff); // wherever

    TTI_SFPTRANSP(0, 0, 0, 0); // all arguments are unused

    TTI_SFPSWAP(0 /*unused*/, p_sfpu::LREG6 /*lreg_src_c*/, p_sfpu::LREG7 /*lreg_dest*/, 1 /*instr_mod1*/);
    TTI_SFPSWAP(0 /*unused*/, p_sfpu::LREG5 /*lreg_src_c*/, p_sfpu::LREG6 /*lreg_dest*/, 1 /*instr_mod1*/);
    TTI_SFPSWAP(0 /*unused*/, p_sfpu::LREG4 /*lreg_src_c*/, p_sfpu::LREG5 /*lreg_dest*/, 1 /*instr_mod1*/);

    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_3, -128 & 0x3fff); // wherever
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_3, -126 & 0x3fff); // wherever
}

template <PoolType pool_type, ReduceDim reduce_dim, DataFormat format>
inline void _calculate_reduce_max_col_subblock_4x2_(const uint32_t block_height /*, const uint32_t block_width*/)
{
    static_assert(reduce_dim == REDUCE_COL, "Only column reduction (REDUCE_COL) is currently supported");
    static_assert(pool_type == PoolType::MAX, "Only MAX pool type is currently supported");
    static_assert(format == DataFormat::Float16_b, "SFPU reduce max col only supports Float16_b format");

    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::PACK);

    sfpu_reduce_max_col_subblock_4x2_load_initial_values(); // LREGS 4-7 are initialized with negative infinity
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    for (uint32_t i = 0; i < block_height; i++)
    {
#pragma GCC unroll 8
        for (int j = 0; j < 8; j++)
        {
            lltt::replay(0, 8);

            if (j % 4 == 3)
            {
                TTI_INCRWC(0, 10, 0, 0);
                TTI_INCRWC(0, 10, 0, 0);
            }
            else
            {
                TTI_INCRWC(0, 4, 0, 0);
            }
        }

        // go to next tile in same row
        TTI_SFPLOAD(8, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);
    }

    _move_to_next_subblock_4x2_();

    TTI_SFPSWAP(0 /*unused*/, p_sfpu::LREG0 /*lreg_src_c*/, p_sfpu::LREG4 /*lreg_dest*/, 1 /*instr_mod1*/);

    sfpu_reduce_max_col_subblock_4x2_load_initial_values();
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TTI_SFPLOAD(8, InstrModLoadStore::FP16B, ADDR_MOD_2, 0); // dummy

    for (uint32_t i = 0; i < block_height; i++)
    {
#pragma GCC unroll 8
        for (int j = 0; j < 8; j++)
        {
            lltt::replay(0, 8);

            if (j % 4 == 3)
            {
                TTI_INCRWC(0, 10, 0, 0);
                TTI_INCRWC(0, 10, 0, 0);
            }
            else
            {
                TTI_INCRWC(0, 4, 0, 0);
            }
        }

        TTI_SFPLOAD(8, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);
    }

    _move_to_next_subblock_4x2_();

    TTI_SFPSWAP(0 /*unused*/, p_sfpu::LREG1 /*lreg_src_c*/, p_sfpu::LREG4 /*lreg_dest*/, 1 /*instr_mod1*/);
}

inline void _reduce_max_col_subblock_4x2_prologue_()
{
    constexpr uint16_t neg_inf_fp16b = 0xFF80;

    // F0 - Initialize with negative infinity
    TTI_SFPLOADI(p_sfpu::LREG0, InstrModLoadStore::FP16B, neg_inf_fp16b);
    TTI_SFPLOADI(p_sfpu::LREG1, InstrModLoadStore::FP16B, neg_inf_fp16b);
}

inline void _reduce_max_col_subblock_4x2_epilogue_()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG4, 0); // move result of reduce to LREG4

    TTI_SFPTRANSP(0, 0, 0, 0); // all arguments are unused

    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_3, 2);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_3, 16);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_3, 18);

    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_3, 64 + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_3, 64 + 2);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_3, 64 + 16);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_3, 64 + 18);
}

} // namespace sfpu
} // namespace ckernel
