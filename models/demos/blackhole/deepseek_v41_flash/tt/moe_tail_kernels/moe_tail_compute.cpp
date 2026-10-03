// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// out_tile[j] = sum_k act_tile[j,k] * score_col[k]  (hardware MAC with column broadcast), same maths as the
// deepseek_moe_fast_reduce_nc_fused compute kernel.
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

using namespace ckernel;

constexpr uint32_t TPC = get_compile_time_arg_val(0);
constexpr uint32_t K = get_compile_time_arg_val(1);
constexpr uint32_t cb_a = get_compile_time_arg_val(2);
constexpr uint32_t cb_s = get_compile_time_arg_val(3);
constexpr uint32_t cb_o = get_compile_time_arg_val(4);

void kernel_main() {
    CircularBuffer ca(cb_a), cs(cb_s), co(cb_o);
    compute_kernel_hw_startup(cb_a, cb_s, cb_o);
    bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_a, cb_s);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(cb_a, cb_s, 1)));
    reconfig_data_format(cb_a, cb_s);
    cs.wait_front(K);
    ca.wait_front(K * TPC);
    for (uint32_t j = 0; j < TPC; ++j) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < K; ++k) {
            mul_tiles_bcast_cols(cb_a, cb_s, j * K + k, k, 0);
        }
        tile_regs_commit();
        co.reserve_back(1);
        pack_reconfig_data_format(cb_o);
        tile_regs_wait();
        pack_tile(0, cb_o);
        tile_regs_release();
        co.push_back(1);
    }
    ca.pop_front(K * TPC);
    cs.pop_front(K);
}
