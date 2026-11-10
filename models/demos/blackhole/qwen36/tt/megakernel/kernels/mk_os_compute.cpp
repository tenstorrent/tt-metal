// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Megakernel AR (oneshot_ar.py OneShotARNorm), owner-core compute:
//   1. owners: the all_reduce_async one-shot reduction (kernels/compute/reduction.cpp, same operand order) + residual
//      -> the output shard (this core's 10-tile chunk of the new residual);
//   2. every core, once its chunk is in the output shard (own reduction, or the all-gather via the receiver): partial
//      row statistic sum(r^2) x 2^-12 over its tiles -> CB 25; leader: rstd = rsqrt(sum of the 16 partials + eps)
//      -> CB 27; y = r * rstd (leader's tile, CB 31) -> CB 29 (multicast into CB_X by the tail kernel).
// runtime args: [has_reduce, num_blocks, block_tiles, is_leader]
#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

constexpr uint32_t CB_SQ = 24, CB_PART = 25, CB_STATS = 26, CB_RSTD = 27, CB_EPS = 28, CB_Y = 29, CB_SCALER = 30,
                   CB_RSTD_IN = 31;

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in1 = cb_in0;
    constexpr bool fuse_residual = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t cb_residual = get_compile_time_arg_val(3);
    constexpr uint32_t CT = get_compile_time_arg_val(4);    // chunk tiles (10)
    constexpr uint32_t NSTAT = get_compile_time_arg_val(5); // cores sharing the row statistic (16)
    constexpr bool PART_ONLY = get_compile_time_arg_val(6) != 0;  // OneShotARX (spec v0.4): stop after the partial
    // OneShotARSK BFP8 knobs (only with PART_ONLY): AGQ = owners pack a BFP8 copy of the reduced chunk into CB_OUTQ,
    // and every core's partial is taken from CB_OUTQ (the copy every device receives); RSQ = CB_RED holds BFP8 partials
    // (the bf16 residual is then unpacked with srcA reconfigured)
    constexpr bool AGQ = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t CB_OUTQ = get_compile_time_arg_val(8);
    constexpr bool RSQ = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t CB_STAT_SRC = AGQ ? CB_OUTQ : cb_out0;
    const uint32_t has_reduce = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t block_num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t is_leader = get_arg_val<uint32_t>(3);
    constexpr uint32_t max_dst_tiles = 8;
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out0);

    if (has_reduce) {
        DeviceZoneScopedN("OS-REDUCE");
        const uint32_t copy_first_block = num_blocks % 2 != 0;
        CircularBuffer cb_in(cb_in0);
        CircularBuffer cb_out(cb_out0);
        cb_in.wait_front(num_blocks * block_num_tiles);
        if constexpr (fuse_residual) {
            CircularBuffer cb_res(cb_residual);
            cb_res.wait_front(block_num_tiles);
        }
        cb_out.reserve_back(block_num_tiles);
        add_init(cb_in0, cb_in1, true);
        uint32_t num_pack_iters = (block_num_tiles + max_dst_tiles - 1) / max_dst_tiles;
        uint32_t cnt = 0;
        for (uint32_t p = 0; p < num_pack_iters; ++p) {
            uint32_t n = std::min(max_dst_tiles, block_num_tiles - cnt);
            if constexpr (fuse_residual) {
                if (p > 0) add_init(cb_in0, cb_in1, true);
            }
            tile_regs_acquire();
            for (uint32_t block = 0; block < num_blocks; block += 2) {
                if (!(copy_first_block && block == 0)) {
                    for (uint32_t i = 0; i < n; ++i) {
                        add_tiles(cb_in0, cb_in1, block * block_num_tiles + p * max_dst_tiles + i,
                                  (block + 1) * block_num_tiles + p * max_dst_tiles + i, i);
                    }
                }
            }
            if constexpr (fuse_residual) {
                if constexpr (RSQ) {
                    reconfig_data_format_srca(cb_in0, cb_residual);
                }
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_residual);
                for (uint32_t i = 0; i < n; ++i) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_residual, p * max_dst_tiles + i, i);
                }
                if constexpr (RSQ) {
                    reconfig_data_format_srca(cb_residual, cb_in0);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n; ++i) pack_tile(i, cb_out0, p * max_dst_tiles + i);
            tile_regs_release();
            cnt += n;
        }
        cb_out.push_back(block_num_tiles);
        if constexpr (AGQ) {
            // BFP8 copy of the reduced chunk: the all-gather payload and the source of CB_X / the partials everywhere
            cb_wait_front(cb_out0, block_num_tiles);
            cb_reserve_back(CB_OUTQ, block_num_tiles);
            reconfig_data_format_srca(cb_in0, cb_out0);
            pack_reconfig_data_format(CB_OUTQ);
            copy_tile_to_dst_init_short(cb_out0);
            for (uint32_t t = 0; t < block_num_tiles; ++t) {
                tile_regs_acquire();
                copy_tile(cb_out0, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_OUTQ);
                tile_regs_release();
            }
            cb_push_back(CB_OUTQ, block_num_tiles);
        }
    }

    // ---- norm tail (every core) ----
    cb_wait_front(CB_STAT_SRC, CT);
    {
        DeviceZoneScopedN("OS-STAT");
        cb_reserve_back(CB_SQ, CT);
        reconfig_data_format(CB_STAT_SRC, CB_STAT_SRC);
        pack_reconfig_data_format(CB_SQ);
        mul_init(CB_STAT_SRC, CB_STAT_SRC);
        for (uint32_t t = 0; t < CT; ++t) {
            tile_regs_acquire();
            mul_tiles(CB_STAT_SRC, CB_STAT_SRC, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_SQ);
            tile_regs_release();
        }
        cb_push_back(CB_SQ, CT);
        cb_wait_front(CB_SQ, CT);
        cb_wait_front(CB_SCALER, 1);
        cb_reserve_back(CB_PART, 1);
        reconfig_data_format(CB_SQ, CB_SCALER);
        pack_reconfig_data_format(CB_PART);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQ, CB_SCALER, CB_PART);
        tile_regs_acquire();
        for (uint32_t t = 0; t < CT; ++t) reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQ, CB_SCALER, t, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_PART);
        tile_regs_release();
        reduce_uninit();
        cb_push_back(CB_PART, 1);
        cb_pop_front(CB_SQ, CT);
    }
    if constexpr (PART_ONLY) {
        return;
    }
    cb_wait_front(CB_EPS, 1);
    if (is_leader) {
        DeviceZoneScopedN("OS-RSTD");
        cb_wait_front(CB_STATS, NSTAT);
        cb_reserve_back(CB_RSTD, 1);
        reconfig_data_format(CB_STATS, CB_STATS);
        pack_reconfig_data_format(CB_RSTD);
        tile_regs_acquire();
        add_init(CB_STATS, CB_STATS);
        add_tiles(CB_STATS, CB_STATS, 0, 1, 0);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_STATS);
        for (uint32_t j = 2; j < NSTAT; ++j) add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_STATS, j, 0);
        reconfig_data_format_srcb(CB_STATS, CB_EPS);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_EPS);
        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_EPS, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_RSTD);
        tile_regs_release();
        cb_push_back(CB_RSTD, 1);
        cb_pop_front(CB_STATS, NSTAT);
    }
    cb_pop_front(CB_EPS, 1);
    cb_wait_front(CB_RSTD_IN, 1);
    {
        DeviceZoneScopedN("OS-SCALE");
        cb_reserve_back(CB_Y, CT);
        reconfig_data_format(cb_out0, CB_RSTD_IN);
        pack_reconfig_data_format(CB_Y);
        mul_bcast_cols_init(cb_out0, CB_RSTD_IN);
        for (uint32_t t = 0; t < CT; ++t) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_out0, CB_RSTD_IN, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_Y);
            tile_regs_release();
        }
        cb_push_back(CB_Y, CT);
    }
    cb_pop_front(CB_RSTD_IN, 1);
    cb_pop_front(cb_out0, CT);
}
