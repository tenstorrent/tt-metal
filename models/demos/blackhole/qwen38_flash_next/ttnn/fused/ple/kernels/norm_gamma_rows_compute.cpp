// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE group norm, one row-block of local_tiles tiles on one core: gr_read's norm_compute.cpp unit
// (rmsnorm_post_allgather's stats reduce, + eps, rsqrt, x * rsqrt), packed fp32 as the op's fp32-dest intermediates
// are, then the op's FUSE_GAMMA stage: mul_tiles_bcast_rows with the fp32 gamma through SrcB (its row 0 over every
// row), packed bf16.
// CBs: 0 x (bf16, local_tiles), 1 stats (bf16, stats_tiles), 2 scaler (bf16 or fp32, 1), 3 eps (bf16, 1), 4 gamma
// (fp32, local_tiles), 5 variance (fp32, 1), 6 inverse_root (rsqrt, fp32, 1), 7 unit (fp32, local_tiles),
// 16 normalized (bf16, local_tiles).
// Compile-time args: 0 local_tiles (the builder's LOCAL_TILES), 1 stats_tiles (STATS_TILES), 2 tiles_per_block
// (divides local_tiles), 3 the output cb.
#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// Each pass's dest handshake in one call: the math TRISC acquires and the pack TRISC waits; then the math TRISC
// commits and the pack TRISC releases.
ALWI void dest_acquire_and_wait() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void dest_commit_and_release() {
    tile_regs_commit();
    tile_regs_release();
}

void kernel_main() {
    constexpr uint32_t local_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t stats_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(2);
    constexpr uint32_t cb_input = 0;
    constexpr uint32_t cb_stats = 1;
    constexpr uint32_t cb_scaler = 2;
    constexpr uint32_t cb_eps = 3;
    constexpr uint32_t cb_gamma = 4;
    constexpr uint32_t cb_variance = 5;
    constexpr uint32_t cb_inverse_root = 6;
    constexpr uint32_t cb_unit = 7;
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);

    compute_kernel_hw_startup(cb_input, cb_input, cb_variance);
    DataflowBuffer input(cb_input);
    DataflowBuffer scaler(cb_scaler);
    DataflowBuffer eps(cb_eps);
    DataflowBuffer gamma(cb_gamma);
    DataflowBuffer variance(cb_variance);
    DataflowBuffer inverse_root(cb_inverse_root);
    DataflowBuffer unit(cb_unit);
    DataflowBuffer out(cb_out);

    {
        scaler.wait_front(1);
        eps.wait_front(1);

        compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, cb_stats, cb_scaler, cb_variance>(
            compute_kernel_lib::ReduceInputBlockShape::row(stats_tiles));

        variance.wait_front(1);
        inverse_root.reserve_back(1);
        reconfig_data_format(cb_variance, cb_eps);
        pack_reconfig_data_format(cb_inverse_root);
        add_init(cb_variance, cb_eps);
        dest_acquire_and_wait();
        add_tiles(cb_variance, cb_eps, 0, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        pack_tile(0, cb_inverse_root);
        dest_commit_and_release();
        inverse_root.push_back(1);
        variance.pop_front(1);
    }

    {
        reconfig_data_format(cb_input, cb_inverse_root);
        pack_reconfig_data_format(cb_unit);
        mul_bcast_cols_init(cb_input, cb_inverse_root);
        inverse_root.wait_front(1);
        for (uint32_t block_start = 0; block_start < local_tiles; block_start += tiles_per_block) {
            input.wait_front(tiles_per_block);
            unit.reserve_back(tiles_per_block);
            dest_acquire_and_wait();
            for (uint32_t t = 0; t < tiles_per_block; ++t) {
                mul_tiles_bcast_cols(cb_input, cb_inverse_root, t, 0, t);
                pack_tile(t, cb_unit);
            }
            dest_commit_and_release();
            unit.push_back(tiles_per_block);
            input.pop_front(tiles_per_block);
        }
        inverse_root.pop_front(1);
    }

    {
        // the chain's FUSE_GAMMA stage: unit (bf16, SrcA) x gamma (fp32, SrcB) on the FPU, gamma's row 0 broadcast
        unit.wait_front(local_tiles);
        gamma.wait_front(local_tiles);
        reconfig_data_format(cb_unit, cb_gamma);
        pack_reconfig_data_format(cb_out);
        mul_bcast_rows_init(cb_unit, cb_gamma);
        for (uint32_t block_start = 0; block_start < local_tiles; block_start += tiles_per_block) {
            out.reserve_back(tiles_per_block);
            dest_acquire_and_wait();
            for (uint32_t t = 0; t < tiles_per_block; ++t) {
                mul_tiles_bcast_rows(cb_unit, cb_gamma, block_start + t, block_start + t, t);
                pack_tile(t, cb_out);
            }
            dest_commit_and_release();
            out.push_back(tiles_per_block);
        }
        unit.pop_front(local_tiles);
        gamma.pop_front(local_tiles);
        scaler.pop_front(1);
        eps.pop_front(1);
    }
}
