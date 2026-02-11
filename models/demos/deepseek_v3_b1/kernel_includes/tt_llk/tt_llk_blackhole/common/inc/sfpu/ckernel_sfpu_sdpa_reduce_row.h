// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

//**************************************************************
// SFPU SDPA REDUCE MAX ROW IMPLEMENTATION
//**************************************************************
namespace sdpa_reduce_row {
constexpr std::uint8_t ZERO_ADDR_MOD = ADDR_MOD_7;
constexpr std::uint8_t TILE_OFFSET_ADDR_MOD = ADDR_MOD_5;
}  // namespace sdpa_reduce_row
inline void sfpu_sdpa_reduce_row_subblock_8x32_configure_addrmod() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(sdpa_reduce_row::ZERO_ADDR_MOD);
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 16},
    }
        .set(sdpa_reduce_row::TILE_OFFSET_ADDR_MOD);
}

template <PoolType pool_type, uint lreg_a, uint lreg_b>
inline void reduce_lregs_instr() {
    if constexpr (pool_type == PoolType::MAX) {
        TTI_SFPSWAP(0 /*unused*/, lreg_a /*lreg_src_c*/, lreg_b /*lreg_dest*/, p_sfpswap::ALL_ROWS_MAX /*instr_mod1*/);
    } else {
        TTI_SFPADD(lreg_a, p_sfpu::LCONST_1, lreg_b, lreg_a, 0);
    }
}

template <PoolType pool_type>
inline void reduce_row_8x32_instrs() {
    TTI_SFPLOAD(p_sfpu::LREG1, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 0);
    reduce_lregs_instr<pool_type, p_sfpu::LREG0, p_sfpu::LREG1>();
    TTI_SFPLOAD(p_sfpu::LREG3, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 4);
    reduce_lregs_instr<pool_type, p_sfpu::LREG2, p_sfpu::LREG3>();
    TTI_SFPLOAD(p_sfpu::LREG1, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 2);
    reduce_lregs_instr<pool_type, p_sfpu::LREG0, p_sfpu::LREG1>();
    TTI_SFPLOAD(p_sfpu::LREG3, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 6);
    reduce_lregs_instr<pool_type, p_sfpu::LREG2, p_sfpu::LREG3>();
    TTI_SFPLOAD(p_sfpu::LREG1, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 8);
    reduce_lregs_instr<pool_type, p_sfpu::LREG0, p_sfpu::LREG1>();
    TTI_SFPLOAD(p_sfpu::LREG3, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 12);
    reduce_lregs_instr<pool_type, p_sfpu::LREG2, p_sfpu::LREG3>();
    TTI_SFPLOAD(p_sfpu::LREG1, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 10);
    reduce_lregs_instr<pool_type, p_sfpu::LREG0, p_sfpu::LREG1>();
    TTI_SFPLOAD(p_sfpu::LREG3, 0, sdpa_reduce_row::TILE_OFFSET_ADDR_MOD, 14);
    reduce_lregs_instr<pool_type, p_sfpu::LREG2, p_sfpu::LREG3>();
}

inline void _init_sdpa_reduce_row_8x32_replay_buffers_() {
    // ***********************************************************
    // Record replay buffer
    // LREG0 will contain the first 4 rows, LREG2 will contain the second 4 rows
    // Max will be the first 16 instructions, SUM will be the last 16 instructions
    load_replay_buf<NoExec>(0, 32, [] {
        // Max instructions
        reduce_row_8x32_instrs<PoolType::MAX>();
        // Sum instructions
        reduce_row_8x32_instrs<PoolType::SUM>();
    });
}

template <DataFormat format>
inline void _init_sdpa_reduce_row_8x32_() {
    static_assert(format == DataFormat::Float16_b, "Unsupported data format. Supported formats: Float16_b");

    _init_sfpu_config_reg();
    sfpu_sdpa_reduce_row_subblock_8x32_configure_addrmod();

    _init_sdpa_reduce_row_8x32_replay_buffers_();
}

template <DataFormat format, PoolType pool_type>
inline void _sdpa_reduce_row_8x32_epilogue_() {
    static_assert(format == DataFormat::Float16_b, "Unsupported data format. Supported formats: Float16_b");
    static_assert(
        pool_type == PoolType::MAX || pool_type == PoolType::SUM,
        "Unsupported pool type. Supported pool types: MAX, SUM");

    // Reduce by 1/2 each time
    // 8 -> 4 -> 2 -> 1
    // Shift 4x
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    reduce_lregs_instr<pool_type, p_sfpu::LREG0, p_sfpu::LREG1>();
    reduce_lregs_instr<pool_type, p_sfpu::LREG2, p_sfpu::LREG3>();
    // Shift 2x
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    reduce_lregs_instr<pool_type, p_sfpu::LREG0, p_sfpu::LREG1>();
    reduce_lregs_instr<pool_type, p_sfpu::LREG2, p_sfpu::LREG3>();
    // Shift 1x
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1);
    reduce_lregs_instr<pool_type, p_sfpu::LREG0, p_sfpu::LREG1>();
    reduce_lregs_instr<pool_type, p_sfpu::LREG2, p_sfpu::LREG3>();
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
}

template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    DataFormat format,
    uint32_t block_width,
    bool skip_signalling = false>
inline void _calculate_sdpa_reduce_row_8x32_(uint src_index) {
    static_assert(reduce_dim == REDUCE_ROW, "Only row reduction (REDUCE_ROW) is currently supported");
    static_assert(
        pool_type == PoolType::MAX || pool_type == PoolType::SUM,
        "Unsupported pool type. Supported pool types: MAX, SUM");
    static_assert(format == DataFormat::Float16_b, "SFPU reduce max col only supports Float16_b format");

    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, src_index + get_dest_buffer_base());
    constexpr uint32_t replay_start = pool_type == PoolType::MAX ? 0 : 16;

    // TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    // TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::PACK);

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    if constexpr (!skip_signalling) {
        t6_semaphore_wait_on_zero<p_stall::STALL_SFPU>(semaphore::FPU_SFPU);
    }

    TTI_SFPLOAD(p_sfpu::LREG0, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 0);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 4);
    lltt::replay(replay_start + 4, 12);

    if constexpr (block_width > 1) {
        for (uint32_t i = 0; i < block_width - 1; i++) {
            if constexpr (!skip_signalling) {
                t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU);
                t6_semaphore_wait_on_zero<p_stall::STALL_SFPU>(semaphore::FPU_SFPU);
            }
            lltt::replay(replay_start, 16);
        }
    }
    _sdpa_reduce_row_8x32_epilogue_<format, pool_type>();
}

template <DataFormat format, uint32_t block_width, bool skip_signalling = false>
inline void _calculate_sdpa_reduce_max_row_8x32_(uint src_index, uint dst_index, bool prev_max = false) {
    static_assert(format == DataFormat::Float16_b, "SFPU reduce max col only supports Float16_b format");

    _calculate_sdpa_reduce_row_8x32_<PoolType::MAX, ReduceDim::REDUCE_ROW, format, block_width, skip_signalling>(
        src_index);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + get_dest_buffer_base());
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    // Load Prev Max
    if (prev_max) {
        TTI_SFPLOAD(p_sfpu::LREG1, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 0);
        reduce_lregs_instr<PoolType::MAX, p_sfpu::LREG0, p_sfpu::LREG1>();
        TTI_SFPLOAD(p_sfpu::LREG3, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 4);
        reduce_lregs_instr<PoolType::MAX, p_sfpu::LREG2, p_sfpu::LREG3>();
        // Restore so that prev remains cached
        TTI_SFPLOAD(p_sfpu::LREG1, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 0);
        TTI_SFPLOAD(p_sfpu::LREG3, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 4);
    }
    TTI_SFPSTORE(p_sfpu::LREG0, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 0);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 4);
    if constexpr (!skip_signalling) {
        t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU);
    }
}

template <DataFormat format, uint32_t block_width, bool skip_signalling = false>
inline void _calculate_sdpa_reduce_sum_row_8x32_(uint src_index, uint dst_index, bool prev_sum = false) {
    static_assert(format == DataFormat::Float16_b, "SFPU reduce max col only supports Float16_b format");

    _calculate_sdpa_reduce_row_8x32_<PoolType::SUM, ReduceDim::REDUCE_ROW, format, block_width, skip_signalling>(
        src_index);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + get_dest_buffer_base());
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    // Load Prev Sum
    if (prev_sum) {
        TTI_SFPLOAD(p_sfpu::LREG1, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 0);
        reduce_lregs_instr<PoolType::SUM, p_sfpu::LREG0, p_sfpu::LREG1>();
        TTI_SFPLOAD(p_sfpu::LREG3, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 4);
        reduce_lregs_instr<PoolType::SUM, p_sfpu::LREG2, p_sfpu::LREG3>();
    }
    TTI_SFPSTORE(p_sfpu::LREG0, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 0);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, sdpa_reduce_row::ZERO_ADDR_MOD, 4);
    if constexpr (!skip_signalling) {
        t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU);
    }
}

}  // namespace sfpu
}  // namespace ckernel
