// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// index_tail, norm core compute (fp32 dest): rms_norm of the index query tile row (index_q_norm; the first pair only),
// then per lane of this core the patched ring times the ones tile on the SFPU (the chain's one-hot select arithmetic:
// bf16 RNE store, zero rule), the 0.25-scaled column sum (reduce.cpp's helper call), and rms_norm of the pooled row
// (index_k_norm).  Runtime args: 0 lane_count, 1 do_query.

#include "rms_norm_mirror.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "index_tail_cbs.h"
#include "../../kernels/zones.h"

using namespace index_tail;

void kernel_main() {
    const uint32_t lane_count = get_arg_val<uint32_t>(0);
    const uint32_t do_query = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(CB_X, CB_X, CB_XMM2);
    if (do_query) {
        FUSED_ZONE("fz_qs_it_cn_query");
        rms_norm_rows<
            HEAD_TILES,
            128,
            CB_X,
            CB_SCALER,
            CB_EPS,
            CB_GAMMA_Q,
            CB_XMM2,
            CB_EX2,
            CB_EX2PE,
            CB_FUSION,
            CB_NQ>(1);
    }

    cb_wait_front(CB_ONES, 1);
    for (uint32_t i = 0; i < lane_count; ++i) {
        FUSED_ZONE("fz_qs_it_cn_lane");
        cb_wait_front(CB_RING, HEAD_TILES);
        cb_reserve_back(CB_RINGC, HEAD_TILES);
        cb_reserve_back(CB_RINGW, HEAD_TILES);
        reconfig_data_format(CB_RING, CB_ONES);
        pack_reconfig_data_format(CB_RINGC);
        mul_binary_tile_init();
        for (uint32_t t = 0; t < HEAD_TILES; ++t) {
            tile_regs_acquire();
            copy_init(CB_RING);
            copy_tile(CB_RING, t, 0);
            copy_init(CB_ONES);
            copy_tile(CB_ONES, 0, 1);
            mul_binary_tile<false>(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_RINGC);
            tile_regs_release();
        }
        cb_push_back(CB_RINGC, HEAD_TILES);
        cb_push_back(CB_RINGW, HEAD_TILES);
        cb_pop_front(CB_RING, HEAD_TILES);

        reconfig_data_format(CB_RINGC, CB_SCALER_RING);
        pack_reconfig_data_format(CB_POOLED);
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            CB_RINGC,
            CB_SCALER_RING,
            CB_POOLED,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
            ReduceFp32Mode::Fast>(
            compute_kernel_lib::ReduceInputBlockShape::of(1, HEAD_TILES, 1),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            compute_kernel_lib::NoOp{});

        rms_norm_rows<
            HEAD_TILES,
            128,
            CB_POOLED,
            CB_SCALER,
            CB_EPS,
            CB_GAMMA_K,
            CB_XMM2,
            CB_EX2,
            CB_EX2PE,
            CB_FUSION,
            CB_NK>(1);
    }
}
