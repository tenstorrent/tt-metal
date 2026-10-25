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
    constexpr uint32_t CT = get_compile_time_arg_val(7);
    constexpr uint32_t NROW = R / U;
    constexpr uint32_t cb_mail = get_compile_time_arg_val(8);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_s);
    CircularBuffer q(cb_q), k(cb_k), wt(cb_wt), s(cb_s), out(cb_out), mail(cb_mail);
    mail.wait_front(1);
    const uint32_t nct = ckernel::read_tile_value(cb_mail, 0, 0);
    mail.pop_front(1);
    q.wait_front(R * 4);
    wt.wait_front(R);

    for (uint32_t ct = 0; ct < nct; ct += CT) {
        const uint32_t g = (nct - ct < CT) ? (nct - ct) : CT;  // key tile columns of this group
        k.wait_front(g * U * 4);
        // stage 1: S[r][j] = relu(q_r @ K_u(ct + j)^T), g tiles per DEST pass (relu in the packer, bf16)
        reconfig_data_format(cb_s, cb_k, cb_wt, cb_q);
        pack_reconfig_l1_acc(0);
        matmul_block_init(cb_q, cb_k, 1 /*transpose k*/, 1, 1, 4);
        pack_relu_config(ReluConfig::zero());
        s.reserve_back(R * CT);
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t u = r / NROW;
            tile_regs_acquire();
            for (uint32_t j = 0; j < g; ++j) {
                for (uint32_t d = 0; d < 4; ++d) {
                    matmul_block(cb_q, cb_k, r * 4 + d, (j * U + u) * 4 + d, j, 1, 1, 1, 4);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < g; ++j) {
                pack_tile<true>(j, cb_s, r * CT + j);
            }
            tile_regs_release();
        }
        s.push_back(R * CT);
        k.pop_front(g * U * 4);
        // stage 2: gate-weighted head sum of every row into one output tile per key tile column
        s.wait_front(R * CT);
        reconfig_data_format(cb_k, cb_s, cb_q, cb_wt);
        pack_relu_config(ReluConfig::none());
        matmul_block_init(cb_wt, cb_s, 0, 1, 1, 1);
        out.reserve_back(g);
        for (uint32_t j = 0; j < g; ++j) {
            tile_regs_acquire();
            for (uint32_t r = 0; r < R; ++r) {
                matmul_block(cb_wt, cb_s, r, r * CT + j, 0, 0, 1, 1, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        out.push_back(g);
        s.pop_front(R * CT);
    }
}
