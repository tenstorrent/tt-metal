// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE group norm, one row-block of Wt tiles on one core: gr_read's norm_compute.cpp (the gathered stats reduced
// against the AVG scaler, + eps, rsqrt, x * rsqrt packed bf16 = rmsnorm_post_allgather_metal2.cpp's unit) followed by
// that kernel's FUSE_GAMMA stage instead of gr_read's separate SFPU multiply: the unit packed fp32 (the op's
// intermediates carry the dest's format: fp32 under fp32_dest_acc_en) times the fp32 gamma tile on the FPU with
// mul_tiles_bcast_rows (gamma through SrcB; the LLK broadcasts the gamma tile's ROW 0 over
// the 32 rows -- the chain applies its weight's first row to every branch row), 32-bit dest, packed bf16.
// CBs: 0 x (bf16, Wt), 1 stats (bf16, S), 2 scaler (bf16 or fp32, 1), 3 eps (bf16, 1), 4 gamma (fp32, Wt),
// 5 var (fp32, 1), 6 rsqrt (fp32, 1), 7 unit (fp32, Wt), 16 normalized (bf16, Wt).
// Compile-time args: 0 Wt, 1 S (stats tiles), 2 blk (tiles per block, divides Wt), 3 the output cb.
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
#include "../../kernels/zones.h"

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t blk = get_compile_time_arg_val(2);
    constexpr uint32_t c_res = 0;
    constexpr uint32_t c_stats = 1;
    constexpr uint32_t c_scaler = 2;
    constexpr uint32_t c_eps = 3;
    constexpr uint32_t c_gamma = 4;
    constexpr uint32_t c_var = 5;
    constexpr uint32_t c_recip = 6;
    constexpr uint32_t c_unit = 7;
    constexpr uint32_t c_out = get_compile_time_arg_val(3);

    compute_kernel_hw_startup(c_res, c_res, c_var);
    DataflowBuffer res(c_res);
    DataflowBuffer scaler(c_scaler);
    DataflowBuffer eps(c_eps);
    DataflowBuffer gamma(c_gamma);
    DataflowBuffer var(c_var);
    DataflowBuffer recip(c_recip);
    DataflowBuffer unit(c_unit);
    DataflowBuffer out(c_out);

    {
        FUSED_ZONE("fz_pl_norm_c_stats");
        scaler.wait_front(1);
        eps.wait_front(1);

        compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, c_stats, c_scaler, c_var>(
            compute_kernel_lib::ReduceInputBlockShape::row(S));

        var.wait_front(1);
        recip.reserve_back(1);
        reconfig_data_format(c_var, c_eps);
        pack_reconfig_data_format(c_recip);
        add_init(c_var, c_eps);
        ACQ();
        add_tiles(c_var, c_eps, 0, 0, 0);
        rsqrt_tile_init<false>();
        rsqrt_tile<false>(0);
        pack_tile(0, c_recip);
        REL();
        recip.push_back(1);
        var.pop_front(1);
    }

    {
        FUSED_ZONE("fz_pl_norm_c_unit");
        reconfig_data_format(c_res, c_recip);
        pack_reconfig_data_format(c_unit);
        mul_bcast_cols_init(c_res, c_recip);
        recip.wait_front(1);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            res.wait_front(blk);
            unit.reserve_back(blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; ++wtr) {
                mul_tiles_bcast_cols(c_res, c_recip, wtr, 0, wtr);
                pack_tile(wtr, c_unit);
            }
            REL();
            unit.push_back(blk);
            res.pop_front(blk);
        }
        recip.pop_front(1);
    }

    {
        FUSED_ZONE("fz_pl_norm_c_gamma");
        // the chain's FUSE_GAMMA stage: unit (bf16, SrcA) x gamma (fp32, SrcB) on the FPU, gamma's row 0 broadcast
        unit.wait_front(Wt);
        gamma.wait_front(Wt);
        reconfig_data_format(c_unit, c_gamma);
        pack_reconfig_data_format(c_out);
        mul_bcast_rows_init(c_unit, c_gamma);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            out.reserve_back(blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; ++wtr) {
                mul_tiles_bcast_rows(c_unit, c_gamma, wt + wtr, wt + wtr, wtr);
                pack_tile(wtr, c_out);
            }
            REL();
            out.push_back(blk);
        }
        unit.pop_front(Wt);
        gamma.pop_front(Wt);
        scaler.pop_front(1);
        eps.pop_front(1);
    }
}
