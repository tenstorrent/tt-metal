// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DSV4.1 router matmul, compute.  part = sum_j x_j @ w_j over the core's K slice (bf16 operands, HiFi4, fp32 dest: the
// same accumulation as the ttnn matmul); the reducer core (R == 1) then sums the S partials (fp32 SFPU adds, fixed
// order), score = sqrt(softplus(logits)) and rank = score + bias, all fp32 in DST and packed as fp32 (same ops as the
// ttnn.add(activations) / ttnn.add chain of the fp32 router).

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    DeviceZoneScopedN("RC_ALL");
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_part = get_compile_time_arg_val(2);
    constexpr uint32_t cb_red = get_compile_time_arg_val(3);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(4);
    constexpr uint32_t cb_rank = get_compile_time_arg_val(5);
    constexpr uint32_t cb_score = get_compile_time_arg_val(6);
    constexpr uint32_t KS = get_compile_time_arg_val(7);
    constexpr uint32_t B = get_compile_time_arg_val(8);
    constexpr uint32_t S = get_compile_time_arg_val(9);
    constexpr uint32_t BETA = get_compile_time_arg_val(10);
    constexpr uint32_t BETA_RCP = get_compile_time_arg_val(11);
    constexpr uint32_t THRESH = get_compile_time_arg_val(12);
    constexpr uint32_t MODE = get_compile_time_arg_val(13);  // 0 normal, 1 copy only, 3 raw matmul out (debug, S == 1)
    const uint32_t reducer = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_x, cb_w, cb_part);
    CircularBuffer x(cb_x), w(cb_w), part(cb_part), red(cb_red), bias(cb_bias), rank(cb_rank), score(cb_score);

    reconfig_data_format(cb_x, cb_w);
    matmul_init(cb_x, cb_w);
    tile_regs_acquire();
    for (uint32_t b = 0; b < KS / B; ++b) {
        x.wait_front((b + 1) * B);
        w.wait_front((b + 1) * B);
        for (uint32_t j = b * B; j < (b + 1) * B; ++j) {
            matmul_tiles(cb_x, cb_w, j, j, 0);
        }
    }
    tile_regs_commit();
    {
        DeviceZoneScopedN("RC_mmdone");
    }
    tile_regs_wait();
    if constexpr (MODE == 3) {
        score.reserve_back(1);
        rank.reserve_back(1);
        pack_tile(0, cb_score);
        pack_tile(0, cb_rank);
        score.push_back(1);
        rank.push_back(1);
        tile_regs_release();
        bias.wait_front(1);
        return;
    }
    part.reserve_back(1);
    pack_tile(0, cb_part);
    part.push_back(1);
    tile_regs_release();
    x.pop_front(KS);
    w.pop_front(KS);

    if (reducer) {
        part.wait_front(1);
        bias.wait_front(1);
        if constexpr (S > 1) {
            red.wait_front(S - 1);
        }
        reconfig_data_format_srca(cb_w, cb_part);
        copy_init(cb_part);
        tile_regs_acquire();
        copy_tile(cb_part, 0, 0);
        if constexpr (S > 1) {
            copy_init(cb_red);
            for (uint32_t q = 0; q < S - 1; ++q) {
                copy_tile(cb_red, q, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
        }
        if constexpr (MODE == 1) {
            copy_tile(cb_part, 0, 2);
        } else if constexpr (MODE == 4) {
            softplus_tile_init();
            softplus_tile(0, BETA, BETA_RCP, THRESH);
        } else if constexpr (MODE == 5) {
            sqrt_tile_init();
            sqrt_tile(0);
        } else {
            softplus_tile_init();
            softplus_tile(0, BETA, BETA_RCP, THRESH);
            sqrt_tile_init();
            sqrt_tile(0);
            copy_init(cb_bias);
            copy_tile(cb_bias, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 2);
        }
        tile_regs_commit();
        tile_regs_wait();
        score.reserve_back(1);
        rank.reserve_back(1);
        pack_tile(0, cb_score);
        pack_tile(2, cb_rank);
        score.push_back(1);
        rank.push_back(1);
        tile_regs_release();
        part.pop_front(1);
        bias.pop_front(1);
        if constexpr (S > 1) {
            red.pop_front(S - 1);
        }
    }
}
