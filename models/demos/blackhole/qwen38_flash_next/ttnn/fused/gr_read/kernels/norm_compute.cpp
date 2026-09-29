// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GR read stage 2a, one branch row-block per core: the gathered stats tiles reduced against the AVG scaler,
// + eps, rsqrt, x * rsqrt packed bf16 (rmsnorm_post_allgather_metal2.cpp without gamma), then the SFPU product of the
// bf16 unit with the fp32 gamma/4 packed bf16 (eltwise_binary_sfpu_no_bcast.cpp under a 32-bit dest).
// CBs: 0 residual (bf16, Wt), 1 stats (bf16, S), 2 scaler (bf16 or fp32, 1), 3 eps (bf16, 1), 4 gamma (fp32, Wt),
// 5 var (fp32, 1), 6 rsqrt (fp32, 1), 7 unit (bf16, Wt), 16 normalized (bf16, Wt).
// Compile-time args: 0 Wt, 1 S (stats tiles), 2 blk (tiles per multiply block, divides Wt), 3 the output cb.

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
    FUSED_ZONE("fz_gr_norm_c");
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
