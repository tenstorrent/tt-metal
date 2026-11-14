// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Hy4-Preview router matmul TRISC (8 DRAM-bank cores, core b owns expert tile column b = experts 32 b .. 32 b + 31, K = KT tiles):
//   logits = x @ W[:, 32 b : 32 b + 32] with the SAME LLK matmul / fidelity (HiFi4) / fp32 dest accumulation as the ttnn.matmul of Hy4Gate,
//   score = sigmoid(logits), rank = score + bias (fp32 SFPU, the same functions as the ttnn.sigmoid / ttnn.add chain); score and rank are packed fp32.
// CT args: 0 cb_x (tiny [8, 32] bf16 tiles), 1 cb_w (bf16 32x32 ring), 2 cb_bias (fp32 32x32 tile), 3 cb_score, 4 cb_rank, 5 KT, 6 SBK

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
    constexpr uint32_t cb_bias = get_compile_time_arg_val(2);
    constexpr uint32_t cb_score = get_compile_time_arg_val(3);
    constexpr uint32_t cb_rank = get_compile_time_arg_val(4);
    constexpr uint32_t KT = get_compile_time_arg_val(5);
    constexpr uint32_t SBK = get_compile_time_arg_val(6);

    compute_kernel_hw_startup(cb_x, cb_w, cb_score);
    CircularBuffer x(cb_x), w(cb_w), bias(cb_bias), score(cb_score), rank(cb_rank);

    x.wait_front(KT);
    bias.wait_front(1);
    reconfig_data_format(cb_x, cb_w);
    matmul_init(cb_x, cb_w);
    tile_regs_acquire();
    {
        DeviceZoneScopedN("T_RMM");
        for (uint32_t b = 0; b < KT / SBK; ++b) {
            w.wait_front(SBK);
            for (uint32_t j = 0; j < SBK; ++j) {
                matmul_tiles(cb_x, cb_w, b * SBK + j, j, 0);
            }
            w.pop_front(SBK);
        }
    }
    {
        DeviceZoneScopedN("T_REPI");
        sigmoid_tile_init();
        sigmoid_tile(0);
        reconfig_data_format_srca(cb_w, cb_bias);
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
    }
    x.pop_front(KT);
    bias.pop_front(1);
}
