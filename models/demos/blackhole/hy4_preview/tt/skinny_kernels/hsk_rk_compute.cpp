// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Hy4-Preview router, K-split over the 8 DRAM-bank cores (no x exchange): core b holds the K tiles [KL b, KL b + KL) of ALL NT expert tile columns,
//   partial_n = x[K slice b] @ W[K slice b, tile column n]   (bf16 operands, HiFi4, fp32 dest: the same LLK matmul as the ttnn.matmul of Hy4Gate),
//   the NT partials of core b are sent to the NT cores (tile n -> core n, slot b of cb_red), core n sums its NC partials in the fixed order c = 0..NC-1
//   (fp32 SFPU adds on UnpackToDestFp32 tiles), score = sigmoid(logits), rank = score + bias (the same SFPU functions as the ttnn chain); fp32 score / rank tiles out.
// CT args: 0 cb_x ([16, 32] bf16 tiles, KL), 1 cb_w (bf16 32x32 ring), 2 cb_pout (fp32 [8, 32], 4 pages), 3 cb_red (fp32 [8, 32], NC pages), 4 cb_bias,
//   5 cb_score, 6 cb_rank, 7 KL, 8 NT (= NC = 8), 9 NPASS_W (dest tiles per pass, 4)

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_pout = get_compile_time_arg_val(2);
    constexpr uint32_t cb_red = get_compile_time_arg_val(3);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(4);
    constexpr uint32_t cb_score = get_compile_time_arg_val(5);
    constexpr uint32_t cb_rank = get_compile_time_arg_val(6);
    constexpr uint32_t KL = get_compile_time_arg_val(7);
    constexpr uint32_t NT = get_compile_time_arg_val(8);
    constexpr uint32_t PW = get_compile_time_arg_val(9);

    compute_kernel_hw_startup(cb_x, cb_w, cb_pout);
    CircularBuffer x(cb_x), w(cb_w), pout(cb_pout), red(cb_red), bias(cb_bias), score(cb_score), rank(cb_rank);

    x.wait_front(KL);
    reconfig_data_format(cb_x, cb_w);
    matmul_init(cb_x, cb_w);
    for (uint32_t pass = 0; pass < NT / PW; ++pass) {
        DeviceZoneScopedN("T_RMM");
        tile_regs_acquire();
        for (uint32_t n = 0; n < PW; ++n) {
            w.wait_front(KL);
            for (uint32_t k = 0; k < KL; ++k) {
                matmul_tiles(cb_x, cb_w, k, k, n);
            }
            w.pop_front(KL);
        }
        tile_regs_commit();
        tile_regs_wait();
        pout.reserve_back(PW);
        for (uint32_t n = 0; n < PW; ++n) {
            pack_tile(n, cb_pout);
        }
        pout.push_back(PW);
        tile_regs_release();
    }
    x.pop_front(KL);
    {
        DeviceZoneScopedN("T_REPI");
        red.wait_front(NT);
        bias.wait_front(1);
        reconfig_data_format_srca(cb_w, cb_red);
        copy_init(cb_red);
        tile_regs_acquire();
        copy_tile(cb_red, 0, 0);
        for (uint32_t c = 1; c < NT; ++c) {
            copy_tile(cb_red, c, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        sigmoid_tile_init();
        sigmoid_tile(0);
        copy_init(cb_bias);
        copy_tile(cb_bias, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 2);
        tile_regs_commit();
        tile_regs_wait();
        score.reserve_back(1);
        rank.reserve_back(1);
        pack_tile(0, cb_score);
        pack_tile(2, cb_rank);
        score.push_back(1);
        rank.push_back(1);
        tile_regs_release();
        red.pop_front(NT);
        bias.pop_front(1);
    }
}
