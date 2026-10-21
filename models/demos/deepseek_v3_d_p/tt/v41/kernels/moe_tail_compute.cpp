// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the DeepSeek-V4.1 decode MoE tail (tt/v41/moe_tail.py): out[jj] = sum_k act[jj, k] * score[k] with the
// score tile's column 0 broadcast along the rows, the K products accumulated in fp32 in DEST, one pack per output
// tile. After models/demos/blackhole/deepseek_v41_flash/tt/moe_tail_kernels/moe_tail_compute.cpp (ssinghal/dsv4p1).
//
// compile_time_args = [TPC, K, cb_act, cb_score, cb_out]

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"

constexpr uint32_t TPC = get_compile_time_arg_val(0);
constexpr uint32_t K = get_compile_time_arg_val(1);
constexpr uint32_t cb_act = get_compile_time_arg_val(2);
constexpr uint32_t cb_score = get_compile_time_arg_val(3);
constexpr uint32_t cb_out = get_compile_time_arg_val(4);

void kernel_main() {
    DataflowBuffer act(cb_act), score(cb_score), out(cb_out);
    compute_kernel_hw_startup(cb_act, cb_score, cb_out);
    bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_act, cb_score);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
        cb_act, cb_score, 1)));
    reconfig_data_format(cb_act, cb_score);
    score.wait_front(K);
    act.wait_front(K * TPC);
    for (uint32_t jj = 0; jj < TPC; ++jj) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < K; ++k) {
            mul_tiles_bcast_cols(cb_act, cb_score, jj * K + k, k, 0);
        }
        tile_regs_commit();
        out.reserve_back(1);
        pack_reconfig_data_format(cb_out);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        out.push_back(1);
    }
    act.pop_front(K * TPC);
    score.pop_front(K);
}
