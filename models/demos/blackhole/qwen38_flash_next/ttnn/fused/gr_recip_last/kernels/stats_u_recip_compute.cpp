// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The norm core of the GR read with the rsqrt applied LAST (ttnn/fused/gr_recip_last): four phases in one kernel.
//   1. stats: gr_read/kernels/stats_compute.cpp's body (x*x on the FPU packed fp32, the row reduce against the 1.0
//      scaler, one bf16 rounding) -- the residual is NOT popped: phase 2 reads the same tiles.
//   2. u = x * gamma/4 on the FPU (bf16 residual, fp32 gamma rows, 32-bit dest, packed bf16 twice: c_u for the
//      multicast to the down workers, c_ukeep for phase 4).  Gamma comes before the rsqrt so the workers' matmul
//      needs no statistics; this moves the chain's rounding points (COMPONENT class).
//   3. recip: norm_compute.cpp's reduce of the gathered stats against the AVG scaler, + eps, rsqrt, verbatim (the
//      same bits as the chain's rsqrt), packed fp32 twice: c_recip_out for the multicast to the workers, c_recip here.
//   4. normalized' = u * recip (norm_compute's mul_tiles_bcast_cols on u instead of x), packed bf16 for the gate.
// CBs: 0 residual (bf16, width_tiles), 1 gathered stats (bf16, stats_tiles), 2 avg scaler (chain bf16, 1), 3 eps
// (bf16, 1), 4 gamma rows (fp32, width_tiles), 5 var (fp32, 1), 6 recip (fp32, 1), 7 u kept (bf16, width_tiles);
// compile-time: 3 c_u (bf16, width_tiles), 4 stats scaler (fp32, 1), 5 x^2 (fp32, width_tiles), 6 stats out (bf16, 1),
// 7 recip out (fp32, 1), 8 normalized' out (bf16, width_tiles).
// Compile-time args: 0 width_tiles, 1 stats_tiles, 2 block_tiles (divides width_tiles; <= 4 fp32 dest tiles per
// block), 3-8 the CBs above.

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

ALWI void acquire_and_wait_dst() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void commit_and_release_dst() {
    tile_regs_commit();
    tile_regs_release();
}

void kernel_main() {
    constexpr uint32_t width_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t stats_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t c_u = get_compile_time_arg_val(3);
    constexpr uint32_t c_sscaler = get_compile_time_arg_val(4);
    constexpr uint32_t c_x2 = get_compile_time_arg_val(5);
    constexpr uint32_t c_sout = get_compile_time_arg_val(6);
    constexpr uint32_t c_recip_out = get_compile_time_arg_val(7);
    constexpr uint32_t c_nout = get_compile_time_arg_val(8);
    constexpr uint32_t c_res = 0;
    constexpr uint32_t c_stats = 1;
    constexpr uint32_t c_scaler = 2;
    constexpr uint32_t c_eps = 3;
    constexpr uint32_t c_gamma = 4;
    constexpr uint32_t c_var = 5;
    constexpr uint32_t c_recip = 6;
    constexpr uint32_t c_ukeep = 7;
    static_assert(
        width_tiles % block_tiles == 0 && block_tiles <= 4, "block_tiles divides width_tiles and fits the fp32 dest");

    // ---- 1. stats: stats_compute.cpp, the residual kept ----
    compute_kernel_hw_startup(c_res, c_sscaler, c_x2);
    {
        {
            DataflowBuffer res(c_res);
            DataflowBuffer x2(c_x2);
            DataflowBuffer scaler(c_sscaler);

            reconfig_data_format(c_res, c_res);
            pack_reconfig_data_format(c_x2);
            mul_init(c_res, c_res);
            for (uint32_t wt = 0; wt < width_tiles; ++wt) {
                res.wait_front(wt + 1);
                x2.reserve_back(1);
                tile_regs_acquire();
                mul_tiles(c_res, c_res, wt, wt, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, c_x2);
                tile_regs_release();
                x2.push_back(1);
            }
            compute_kernel_lib::reduce<
                PoolType::AVG,
                ReduceDim::REDUCE_ROW,
                c_x2,
                c_sscaler,
                c_sout,
                compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
                compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast>(compute_kernel_lib::ReduceInputBlockShape::row(width_tiles));
            scaler.pop_front(1);
        }
    }
    // ---- 2. u = x * gamma/4: the FPU product of the bf16 residual and the fp32 gamma rows, packed bf16 twice ----
    {
        {
            DataflowBuffer res(c_res);
            DataflowBuffer gamma(c_gamma);
            DataflowBuffer u_row(c_u);
            DataflowBuffer ukeep(c_ukeep);

            reconfig_data_format(c_res, c_gamma);
            pack_reconfig_data_format(c_u);
            mul_init(c_res, c_gamma);
            for (uint32_t wt = 0; wt < width_tiles; wt += block_tiles) {
                gamma.wait_front(wt + block_tiles);
                u_row.reserve_back(block_tiles);
                ukeep.reserve_back(block_tiles);
                tile_regs_acquire();
                for (uint32_t i = 0; i < block_tiles; ++i) {
                    mul_tiles(c_res, c_gamma, wt + i, wt + i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < block_tiles; ++i) {
                    pack_tile(i, c_u);
                }
                for (uint32_t i = 0; i < block_tiles; ++i) {
                    pack_tile(i, c_ukeep);
                }
                tile_regs_release();
                u_row.push_back(block_tiles);
                ukeep.push_back(block_tiles);
            }
            res.pop_front(width_tiles);
            gamma.pop_front(width_tiles);
        }
    }
    // ---- 3. recip: norm_compute.cpp's reduce, + eps, rsqrt (verbatim), after its startup formats ----
    reconfig_data_format(c_res, c_res);
    pack_reconfig_data_format(c_var);
    {
        {
            DataflowBuffer scaler(c_scaler);
            DataflowBuffer eps(c_eps);
            DataflowBuffer var(c_var);
            DataflowBuffer recip(c_recip);
            DataflowBuffer recip_out(c_recip_out);

            scaler.wait_front(1);
            eps.wait_front(1);

            compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, c_stats, c_scaler, c_var>(
                compute_kernel_lib::ReduceInputBlockShape::row(stats_tiles));

            var.wait_front(1);
            recip.reserve_back(1);
            recip_out.reserve_back(1);
            reconfig_data_format(c_var, c_eps);
            pack_reconfig_data_format(c_recip);
            add_init(c_var, c_eps);
            acquire_and_wait_dst();
            add_tiles(c_var, c_eps, 0, 0, 0);
            rsqrt_tile_init();
            rsqrt_tile(0);
            pack_tile(0, c_recip);
            pack_tile(0, c_recip_out);
            commit_and_release_dst();
            recip.push_back(1);
            recip_out.push_back(1);
            var.pop_front(1);
            scaler.pop_front(1);
            eps.pop_front(1);
        }
    }
    // ---- 4. normalized' = u * recip: norm_compute's column-broadcast multiply on u, packed bf16 for the gate ----
    {
        {
            DataflowBuffer ukeep(c_ukeep);
            DataflowBuffer recip(c_recip);
            DataflowBuffer nout(c_nout);

            reconfig_data_format(c_ukeep, c_recip);
            pack_reconfig_data_format(c_nout);
            mul_bcast_cols_init(c_ukeep, c_recip);
            recip.wait_front(1);
            for (uint32_t wt = 0; wt < width_tiles; wt += block_tiles) {
                ukeep.wait_front(block_tiles);
                nout.reserve_back(block_tiles);
                acquire_and_wait_dst();
                for (uint32_t wtr = 0; wtr < block_tiles; ++wtr) {
                    mul_tiles_bcast_cols(c_ukeep, c_recip, wtr, 0, wtr);
                    pack_tile(wtr, c_nout);
                }
                commit_and_release_dst();
                nout.push_back(block_tiles);
                ukeep.pop_front(block_tiles);
            }
            recip.pop_front(1);
        }
    }
}
