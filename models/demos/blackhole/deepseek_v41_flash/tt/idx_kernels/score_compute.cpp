// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused verify-indexer score, compute. Per key tile column ct:
//   S_r  = relu(q_r @ K_u^T)  (4 d-tiles accumulated in DEST, relu in the packer, rounded to bf16)     r = row of the
//   verify block (user u) out  = sum_r  Wt_r @ S_r  (Wt_r: row r holds the gate vector, rest 0 -> output row r) same
//   numerics as matmul(w_bd, S)
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_wt = get_compile_time_arg_val(2);
    constexpr uint32_t cb_s = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t R = get_compile_time_arg_val(5);
    constexpr uint32_t U = get_compile_time_arg_val(6);
    constexpr uint32_t NROW = R / U;
    const uint32_t nct = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_s);
    CircularBuffer q(cb_q), k(cb_k), wt(cb_wt), s(cb_s), out(cb_out);
    q.wait_front(R * 4);
    wt.wait_front(R);

    for (uint32_t ct = 0; ct < nct; ++ct) {
        k.wait_front(U * 4);
        // stage 1: S_r (relu in the packer)
        reconfig_data_format(cb_s, cb_k, cb_wt, cb_q);
        pack_reconfig_l1_acc(0);
        matmul_block_init(cb_q, cb_k, 1 /*transpose k*/, 1, 1, 4);
        pack_relu_config(ReluConfig::zero());
        s.reserve_back(R);
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t u = r / NROW;
            tile_regs_acquire();
            for (uint32_t d = 0; d < 4; ++d) {
                matmul_block(cb_q, cb_k, r * 4 + d, u * 4 + d, 0, 1, 1, 1, 4);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_s);
            tile_regs_release();
        }
        s.push_back(R);
        k.pop_front(U * 4);
        // stage 2: gate-weighted head sum of every row into one output tile
        s.wait_front(R);
        reconfig_data_format(cb_k, cb_s, cb_q, cb_wt);
        pack_relu_config(ReluConfig::none());
        matmul_block_init(cb_wt, cb_s, 0, 1, 1, 1);
        out.reserve_back(1);
        tile_regs_acquire();
        for (uint32_t r = 0; r < R; ++r) {
            matmul_block(cb_wt, cb_s, r, r, 0, 0, 1, 1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        out.push_back(1);
        s.pop_front(R);
    }
}
