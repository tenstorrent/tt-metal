// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused CG vector kernels, compute (SFPU, fp32 in DEST). DOT keeps the running partial in DEST; XR chains its
// partial r.r through CB_ACC (pack the running tile, unpack it with the next one). Each core's 1024-lane partial is
// then reduced on the SFPU to one value at element (0, 0), so the finish step adds only one number per core.
// The mode (0 dot, 1 axpy, 2 xr, 3 zero) is a runtime argument, checked once per launch or per tile: one binary for
// all four. Zero has no compute (the writer stores zeros).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    const uint32_t nt = get_arg_val<uint32_t>(0), mode = get_arg_val<uint32_t>(1);
    if (mode == 3) {
        return;
    }
    compute_kernel_hw_startup(CB_A, CB_OUT);
    if (mode != 0) {
        cb_wait_front(CB_S, 1);
    }
    if (mode == 0) {
        // the running partial sum stays in DEST tile 2 for the whole loop (one acquire; no pack / unpack per tile);
        // the multiply and the add stay separate fp32 operations
        tile_regs_acquire();
        copy_init(CB_A);
        mul_binary_tile_init();
        for (uint32_t t = 0; t < nt; ++t) {
            cb_wait_front(CB_A, 1);
            cb_wait_front(CB_B, 1);
            copy_tile(CB_A, 0, 0);
            copy_tile(CB_B, 0, 1);
            cb_pop_front(CB_A, 1);
            cb_pop_front(CB_B, 1);
            if (t == 0) {
                mul_binary_tile(0, 1, 2);
            } else {
                mul_binary_tile(0, 1, 0);
                add_binary_tile(2, 0, 2);
            }
        }
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(2);
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(2);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_OUT, 1);
        pack_tile(2, CB_OUT);
        cb_push_back(CB_OUT, 1);
        tile_regs_release();
        return;
    }
    const bool axpy = mode == 1;
    for (uint32_t t = 0; t < nt; ++t) {
        cb_wait_front(CB_A, 1);
        cb_wait_front(CB_B, 1);
        tile_regs_acquire();
        copy_init(CB_A);
        copy_tile(CB_A, 0, 0);
        copy_init(CB_B);
        copy_tile(CB_B, 0, 1);
        if (axpy) {
            copy_init(CB_S);
            copy_tile(CB_S, 0, 2);
            mul_binary_tile_init();
            mul_binary_tile(2, 1, 1);  // s*b
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);  // a + s*b
        } else {                       // xr
            cb_wait_front(CB_C, 1);
            cb_wait_front(CB_D, 1);
            copy_init(CB_S);
            copy_tile(CB_S, 0, 2);
            copy_init(CB_C);
            copy_tile(CB_C, 0, 3);
            copy_init(CB_D);
            copy_tile(CB_D, 0, 4);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);  // alpha*p
            mul_binary_tile(4, 2, 4);  // alpha*Ap
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);  // x + alpha*p
            sub_binary_tile_init();
            sub_binary_tile(3, 4, 3);  // r - alpha*Ap
            mul_binary_tile_init();
            mul_binary_tile(3, 3, 5);  // r_new^2
            if (t) {
                cb_wait_front(CB_ACC, 1);
                copy_init(CB_ACC);
                copy_tile(CB_ACC, 0, 6);
                cb_pop_front(CB_ACC, 1);
                add_binary_tile_init();
                add_binary_tile(6, 5, 5);
            }
            if (t + 1 == nt) {  // this core's r.r partial: reduced to one value at element (0, 0)
                sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
                sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(5);
                sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(5);
            }
            cb_pop_front(CB_C, 1);
            cb_pop_front(CB_D, 1);
        }
        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_B, 1);
        tile_regs_commit();
        tile_regs_wait();
        if (axpy) {
            cb_reserve_back(CB_OUT, 1);
            pack_tile(0, CB_OUT);
            cb_push_back(CB_OUT, 1);
        } else {
            cb_reserve_back(CB_OUT, 1);
            pack_tile(0, CB_OUT);
            cb_push_back(CB_OUT, 1);
            cb_reserve_back(CB_OUT2, 1);
            pack_tile(3, CB_OUT2);
            cb_push_back(CB_OUT2, 1);
            cb_reserve_back(t + 1 < nt ? CB_ACC : CB_OUT3, 1);
            pack_tile(5, t + 1 < nt ? CB_ACC : CB_OUT3);
            cb_push_back(t + 1 < nt ? CB_ACC : CB_OUT3, 1);
        }
        tile_regs_release();
    }
}
