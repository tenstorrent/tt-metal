// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// TRISCs: this core's slice of its bank's router logits tile, x[1 x kq] @ w[kq x 1] (fp32 Dest, the router's
// fidelity). The group's other cores pack their partial for the leader; the leader adds the partials (unpacked straight
// to Dest in fp32, in core order), then scores = sigmoid(logits) on the user rows and biased = scores + bias (bias also
// unpacked to Dest in fp32) -> cb_out pages 0 (scores) and 1 (biased).
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

using namespace ckernel;
#ifdef TRISC_MATH
using namespace ckernel::sfpu;
#endif

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(3);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t block_rows = get_compile_time_arg_val(5);
    constexpr uint32_t cb_part = get_compile_time_arg_val(6);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(7);
    constexpr uint32_t group = get_compile_time_arg_val(8);
    constexpr uint32_t rows = get_compile_time_arg_val(9);  // the user rows: 4, 8 or 16
    constexpr uint32_t kq = num_blocks * block_rows;
    const bool leader = get_arg_val<uint32_t>(0) == 1;

    // matmul unpacks in0 (x) to srcB and in1 (w) to srcA: configure the formats in that order.
    compute_kernel_hw_startup(cb_w, cb_x, cb_out);
    matmul_block_init(cb_x, cb_w, 0, 1, 1, kq);

    tile_regs_acquire();
    cb_wait_front(cb_x, kq);
    for (uint32_t b = 0; b < num_blocks; b++) {
        cb_wait_front(cb_w, block_rows);
        for (uint32_t r = 0; r < block_rows; r++) {
            matmul_block(cb_x, cb_w, b * block_rows + r, r, 0, 0, 1, 1, kq);
        }
        cb_pop_front(cb_w, block_rows);
    }
    cb_pop_front(cb_x, kq);

    if (!leader) {
        tile_regs_commit();
        cb_reserve_back(cb_part, 1);
        tile_regs_wait();
        pack_tile(0, cb_part);
        tile_regs_release();
        cb_push_back(cb_part, 1);
        return;
    }
    if constexpr (group > 1) {
        cb_wait_front(cb_gather, group - 1);
        reconfig_data_format_srca(cb_gather);
        for (uint32_t i = 0; i < group - 1; i++) {
            copy_tile_to_dst_init_short(cb_gather);
            copy_tile(cb_gather, i, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        cb_pop_front(cb_gather, group - 1);
    }
    // sigmoid on the user rows of faces 0 and 1 only (rows / 2 SFPU iterations per face)
    sigmoid_tile_init();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sigmoid, (false, DST_ACCUM_MODE, rows / 2), 0, VectorMode::R));
    cb_wait_front(cb_bias, 1);
    reconfig_data_format_srca(cb_bias);
    copy_tile_to_dst_init_short(cb_bias);
    copy_tile(cb_bias, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 1);
    tile_regs_commit();

    cb_reserve_back(cb_out, 2);
    tile_regs_wait();
    pack_tile(0, cb_out);
    pack_tile(1, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 2);
    cb_pop_front(cb_bias, 1);
}
