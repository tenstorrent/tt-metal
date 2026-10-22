// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batched decode dense MoE gate|up + bias + SwiGLU + routing weights, compute (dense_gu_reader.cpp). Per unit (column
// tile n): DST 0 / 1 = the gate / up bias tiles, the matmul over the K tiles accumulates the 32 rows' gate / up sums
// onto them (FP32); DST 2 = routing weights x expand tile (each column's expert weight, a K = 1 matmul); then the
// gpt-oss SwiGLU times DST 2 in one SFPU pass (gmoe_swiglu.hpp) into DST 0, packed as act tile n (BF16).
//
// compile-time args: [cb_x, cb_w, cb_b, cb_r, cb_e, cb_out, KT, NA, NC, GX]

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "gmoe_swiglu.hpp"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_b = get_compile_time_arg_val(2);
    constexpr uint32_t cb_r = get_compile_time_arg_val(3);
    constexpr uint32_t cb_e = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t KT = get_compile_time_arg_val(6);
    constexpr uint32_t NA = get_compile_time_arg_val(7);
    constexpr uint32_t NC = get_compile_time_arg_val(8);
    constexpr uint32_t GX = get_compile_time_arg_val(9);

    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t units = (c + 1) * NA / NC - c * NA / NC;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    cb_wait_front(cb_x, KT);
    cb_wait_front(cb_r, 1);
    for (uint32_t i = 0; i < units; ++i) {
        cb_wait_front(cb_w, 2 * KT);
        cb_wait_front(cb_b, 2);
        cb_wait_front(cb_e, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_b);
        copy_init(cb_b);
        copy_tile(cb_b, 0, 0);
        copy_tile(cb_b, 1, 1);
        reconfig_data_format(cb_w, cb_x);  // matmul: SrcA <- in1 (BFP4 weights), SrcB <- in0 (BFP8 rows)
        matmul_block_init(cb_x, cb_w, 0, 2, 1, KT);
        for (uint32_t k = 0; k < KT; ++k) {
            matmul_block(cb_x, cb_w, k, 2 * k, 0, false, 2, 1, KT);
        }
        reconfig_data_format(cb_e, cb_r);  // matmul: SrcA <- in1 (expand), SrcB <- in0 (routing weights)
        matmul_init(cb_r, cb_e, 0);
        matmul_tiles(cb_r, cb_e, 0, 0, 2);
        MATH((ckernel::gmoe_swiglu_init()));
        MATH((ckernel::gmoe_swiglu_weighted<DST_ACCUM_MODE>(0, 1, 2, 0)));
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        pack_reconfig_data_format(cb_out);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_w, 2 * KT);
        cb_pop_front(cb_b, 2);
        cb_pop_front(cb_e, 1);
    }
    cb_pop_front(cb_x, KT);
    cb_pop_front(cb_r, 1);
}
