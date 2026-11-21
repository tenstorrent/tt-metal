// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 LM-head argmax, compute (dflash/batched.py _Argmax custom path): this core's logits tiles (one 32-row tile
// row, columns col0 .. col0 + ncol - 1) streamed through DST: tile 0 the running per-element max (FP32), tile 1 the
// tile column each max came from (b32_argmax_sfpu.hpp); packed (FP32) to cb_part for the writer's in-tile reduce
// (cb_part holds 4 pages: a 2-tile push never ends on the CB's last page, which corrupted the grouped MoE's second
// tile).
//
// runtime args: [col0, ncol]; compile-time args: [cb_in, cb_part]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "b32_argmax_sfpu.hpp"

void kernel_main() {
    const uint32_t col0 = get_arg_val<uint32_t>(0);
    const uint32_t ncol = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_part = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_in, cb_part);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, -3.0e38f);
    fill_tile(1, 0.0f);
    for (uint32_t k = 0; k < ncol; ++k) {
        copy_init(cb_in);  // (after the SFPU init of the previous step)
        cb_wait_front(cb_in, 1);
        copy_tile(cb_in, 0, 2);
        cb_pop_front(cb_in, 1);
        const float c = static_cast<float>(col0 + k);
        uint32_t c_bits;
        __builtin_memcpy(&c_bits, &c, 4);
        MATH((ckernel::b32_argmax_step_init()));
        MATH((ckernel::b32_argmax_step(0, 2, 1, c_bits)));
    }
    tile_regs_commit();
    cb_reserve_back(cb_part, 2);
    tile_regs_wait();
    pack_tile(0, cb_part);
    pack_tile(1, cb_part);
    tile_regs_release();
    cb_push_back(cb_part, 2);
}
