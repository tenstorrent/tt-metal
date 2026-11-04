// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// index_tail, norm core compute (fp32 dest): rms_norm of the index query tile row (first pair only), then per lane the
// patched ring times the ones tile on the SFPU (matches the one-hot select: bf16 RNE store, zero rule), the 0.25-scaled
// column sum and rms_norm of the pooled row.  Runtime args: 0 lane_count, 1 do_query.

#include "rms_norm_mirror.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "index_tail_cbs.h"

using namespace index_tail;

void kernel_main() {
    const uint32_t lane_count = get_arg_val<uint32_t>(0);
    const uint32_t do_query = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(cb_x, cb_x, cb_xmm2);
    if (do_query) {
        rms_norm_rows<
            head_tiles,
            128,
            cb_x,
            cb_scaler,
            cb_eps,
            cb_gamma_q,
            cb_xmm2,
            cb_ex2,
            cb_ex2pe,
            cb_fusion,
            cb_nq>(1);
    }

    CircularBuffer(cb_ones).wait_front(1);
    for (uint32_t i = 0; i < lane_count; ++i) {
        CircularBuffer(cb_ring).wait_front(head_tiles);
        CircularBuffer(cb_ringc).reserve_back(head_tiles);
        CircularBuffer(cb_ringw).reserve_back(head_tiles);
        reconfig_data_format(cb_ring, cb_ones);
        pack_reconfig_data_format(cb_ringc);
        mul_binary_tile_init();
        for (uint32_t t = 0; t < head_tiles; ++t) {
            tile_regs_acquire();
            copy_init(cb_ring);
            copy_tile(cb_ring, t, 0);
            copy_init(cb_ones);
            copy_tile(cb_ones, 0, 1);
            mul_binary_tile<false>(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_ringc);
            tile_regs_release();
        }
        CircularBuffer(cb_ringc).push_back(head_tiles);
        CircularBuffer(cb_ringw).push_back(head_tiles);
        CircularBuffer(cb_ring).pop_front(head_tiles);

        reconfig_data_format(cb_ringc, cb_scaler_ring);
        pack_reconfig_data_format(cb_pooled);
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            cb_ringc,
            cb_scaler_ring,
            cb_pooled,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
            ReduceFp32Mode::Fast>(
            compute_kernel_lib::ReduceInputBlockShape::of(1, head_tiles, 1),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            compute_kernel_lib::NoOp{});

        rms_norm_rows<
            head_tiles,
            128,
            cb_pooled,
            cb_scaler,
            cb_eps,
            cb_gamma_k,
            cb_xmm2,
            cb_ex2,
            cb_ex2pe,
            cb_fusion,
            cb_nk>(1);
    }
}
