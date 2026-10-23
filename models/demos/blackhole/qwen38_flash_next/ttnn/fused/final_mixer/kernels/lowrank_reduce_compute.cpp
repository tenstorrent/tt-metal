// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Final mixer stage 3a, one core, T low-rank column tiles: the four device partials summed the way the chain's
// all_reduce composite sums them.  local_sum_float32 reshapes the gathered rows to a leading device dim and
// ttnn::sum transposes that dim into H (zero pad) and reduces with ReduceOpDim::H; for an fp32 input with the
// reduce's default fp32_dest_acc_en the generic reduce takes its ACCURATE SFPU path (reduce_op.cpp
// fp32_sfpu_eligible): the input tile unpacked straight into the 32-bit dest (no SrcA tf32 truncation), the rows
// folded by the fp32 SFPU (compute_kernel_lib::reduce<SUM, REDUCE_COL, ..., ReduceFp32Mode::Accurate>, Precise
// mode; the scaler CB is waited on and ignored) -- the column sums land in row 0.  Then as gr_read's low_rank: the
// fp32 sum tile re-read into the dest (the chain's typecast op boundary), the RNE typecast to bf16, staged bf16,
// the silu of each bf16 tile (unary silu on a bf16 tensor).
// CBs: 0 stacked tiles (fp32, T; unpacked to the dest as fp32), 1 reduce scaler (fp32, 1), 2 sums (fp32, T;
// unpacked to the dest as fp32 on the re-read), 3 staged bf16 (T), 16 low rank (bf16, T).  Compile-time args: 0 T.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t c_st = 0, c_scaler = 1, c_sum = 2, c_q = 3, c_lr = 16;
    constexpr uint32_t fp32 = static_cast<uint32_t>(DataFormat::Float32);
    constexpr uint32_t bf16 = static_cast<uint32_t>(DataFormat::Float16_b);
    compute_kernel_hw_startup(c_st, c_scaler, c_sum);
    DataflowBuffer st(c_st);
    DataflowBuffer scaler(c_scaler);
    DataflowBuffer sum(c_sum);
    DataflowBuffer q(c_q);
    DataflowBuffer lr(c_lr);

    {
        FUSED_ZONE("fz_fm_lr_c_reduce");
        // 1. the chain's reduce: the accurate fp32 SFPU fold of each stacked tile's rows (REDUCE_COL: Ht = 1 tile tall,
        //    Wt = T tiles wide, one batch); the helper waits on the scaler tile and pops the input and scaler itself
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            c_st,
            c_scaler,
            c_sum,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
            ReduceFp32Mode::Accurate>(
            compute_kernel_lib::ReduceInputBlockShape::of(1, T, 1),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            compute_kernel_lib::NoOp{});
        scaler.pop_front(1);
    }

    {
        FUSED_ZONE("fz_fm_lr_c_cast");
        // 2. the chain's typecast: the fp32 sum tile exact in the dest, RNE to bf16
        sum.wait_front(T);
        reconfig_data_format_srca(c_st, c_sum);
        copy_init(c_sum);
        for (uint32_t t = 0; t < T; ++t) {
            tile_regs_acquire();
            copy_tile(c_sum, t, 0);
            typecast_tile_init<fp32, bf16>();
            typecast_tile<fp32, bf16>(0);
            tile_regs_commit();
            q.reserve_back(1);
            tile_regs_wait();
            pack_reconfig_data_format(c_q);
            pack_tile(0, c_q);
            tile_regs_release();
            q.push_back(1);
        }
        sum.pop_front(T);
    }

    {
        FUSED_ZONE("fz_fm_lr_c_silu");
        // 3. the chain's silu on the bf16 tensor
        q.wait_front(T);
        reconfig_data_format_srca(c_sum, c_q);
        copy_init(c_q);
        silu_tile_init();
        for (uint32_t t = 0; t < T; ++t) {
            tile_regs_acquire();
            copy_tile(c_q, t, 0);
            silu_tile<false>(0);
            tile_regs_commit();
            lr.reserve_back(1);
            tile_regs_wait();
            pack_reconfig_data_format(c_lr);
            pack_tile(0, c_lr);
            tile_regs_release();
            lr.push_back(1);
        }
        q.pop_front(T);
    }
}
