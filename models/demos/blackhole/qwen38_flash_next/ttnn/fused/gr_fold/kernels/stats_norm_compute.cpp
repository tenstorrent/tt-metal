// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The fused read's norm core: gr_read's stats_compute.cpp body (the sum of squares of the residual row-block as the
// bf16 stats tile) followed by its norm_compute.cpp body (the gathered stats reduced, + eps, rsqrt, x * rsqrt, then
// the SFPU gamma product) in ONE kernel, both bodies verbatim (tests/test_fused_gr_fold_static.py holds them equal
// to the originals): the same LLK sequences, rounding points and CB formats as the two programs they came from.
// Between the phases the unpack/pack formats are set as norm_compute's hardware startup leaves them.  The stats
// phase's CBs are compile-time args so they sit beside the norm phase's 0-7 and 16.
// CBs: stats phase 0 residual (bf16, Wt), c_sscaler (fp32, 1), c_x2 (fp32, Wt), c_sout (bf16, 1); norm phase 0
// residual again (bf16, Wt), 1 gathered stats (bf16, S), 2 scaler, 3 eps, 4 gamma (fp32, Wt), 5 var, 6 rsqrt,
// 7 unit (bf16, Wt), c_out normalized (bf16, Wt).
// Compile-time args: 0 Wt, 1 S, 2 blk, 3 normalized out cb, 4 stats scaler cb, 5 x2 cb, 6 stats out cb.

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
    constexpr uint32_t c_out = get_compile_time_arg_val(3);
    constexpr uint32_t c_sscaler = get_compile_time_arg_val(4);
    constexpr uint32_t c_x2 = get_compile_time_arg_val(5);
    constexpr uint32_t c_sout = get_compile_time_arg_val(6);
    constexpr uint32_t c_res = 0;
    constexpr uint32_t c_stats = 1;
    constexpr uint32_t c_scaler = 2;
    constexpr uint32_t c_eps = 3;
    constexpr uint32_t c_gamma = 4;
    constexpr uint32_t c_var = 5;
    constexpr uint32_t c_recip = 6;
    constexpr uint32_t c_unit = 7;

    // ---- stats phase: stats_compute.cpp ----
    compute_kernel_hw_startup(c_res, c_sscaler, c_x2);
    {
        FUSED_ZONE("fz_gf_sn_c_stats");
        {
            DataflowBuffer res(c_res);
            DataflowBuffer x2(c_x2);
            DataflowBuffer scaler(c_sscaler);

            reconfig_data_format(c_res, c_res);
            pack_reconfig_data_format(c_x2);
            mul_init(c_res, c_res);
            for (uint32_t wt = 0; wt < Wt; ++wt) {
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
                ReduceFp32Mode::Fast>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));
            res.pop_front(Wt);
            scaler.pop_front(1);
        }
    }
    // ---- norm phase: norm_compute.cpp, after its startup formats (residual bf16 into both source registers, the
    // fp32 var packed) ----
    reconfig_data_format(c_res, c_res);
    pack_reconfig_data_format(c_var);
    {
        FUSED_ZONE("fz_gf_sn_c_norm");
        {
            DataflowBuffer res(c_res);
            DataflowBuffer scaler(c_scaler);
            DataflowBuffer eps(c_eps);
            DataflowBuffer gamma(c_gamma);
            DataflowBuffer var(c_var);
            DataflowBuffer recip(c_recip);
            DataflowBuffer unit(c_unit);
            DataflowBuffer out(c_out);

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

            // unit (bf16) * gamma/4 (fp32) on the SFPU, both operands unpacked to the 32-bit dest, packed bf16.
            unit.wait_front(Wt);
            gamma.wait_front(Wt);
            pack_reconfig_data_format(c_out);
            mul_binary_tile_init();
            for (uint32_t i = 0; i < Wt; i += 2) {
                out.reserve_back(2);
                tile_regs_acquire();
                reconfig_data_format_srca(c_gamma, c_unit);
                copy_init(c_unit);
                copy_tile(c_unit, i, 0);
                copy_tile(c_unit, i + 1, 2);
                reconfig_data_format_srca(c_unit, c_gamma);
                copy_init(c_gamma);
                copy_tile(c_gamma, i, 1);
                mul_binary_tile<true>(0, 1, 0);
                copy_tile(c_gamma, i + 1, 3);
                mul_binary_tile<true>(2, 3, 2);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, c_out);
                pack_tile(2, c_out);
                tile_regs_release();
                out.push_back(2);
            }
            unit.pop_front(Wt);
            gamma.pop_front(Wt);
            scaler.pop_front(1);
            eps.pop_front(1);
        }
    }
}
