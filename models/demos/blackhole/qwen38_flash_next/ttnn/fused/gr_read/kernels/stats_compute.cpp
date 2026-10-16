// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GR read stage 1, one branch row-block per core: sum of squares of the Wt residual tiles as the bf16 stats tile.
// The op sequence of rmsnorm_pre_allgather.cpp for a bf16 input under a 32-bit dest: x*x on the FPU packed to a
// Float32 CB, then the row reduce against the 1.0 scaler tile, one bf16 rounding by the packer.
// CBs: 0 residual (bf16, Wt), 1 scaler (fp32, 1), 2 x^2 (fp32, Wt), 16 stats (bf16, 1).  Compile-time arg 0: Wt.

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "../../kernels/zones.h"

void kernel_main() {
    FUSED_ZONE("fz_gr_stats_c");
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t c_res = 0;
    constexpr uint32_t c_scaler = 1;
    constexpr uint32_t c_x2 = 2;
    constexpr uint32_t c_out = 16;

    compute_kernel_hw_startup(c_res, c_scaler, c_x2);
    DataflowBuffer res(c_res);
    DataflowBuffer x2(c_x2);
    DataflowBuffer scaler(c_scaler);

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
        c_scaler,
        c_out,
        compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));
    res.pop_front(Wt);
    scaler.pop_front(1);
}
