// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-1 router projection, reducer compute (fused_decode/router.py: RouterB1): logits tile n = the sum of its `kb`
// K-slice partial tiles (partial n * kb + j), added on the SFPU in FP32 DST (the partials unpack straight to DST in
// FP32, so the sum keeps full FP32 precision; row 0 only), packed FP32.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

// faces 0 / 1 only (row 0 lives there): the partials' rows 1..31 are never written
ALWI void add_row0(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, 8, DST_ACCUM_MODE),
        a,
        b,
        out,
        VectorMode::R)));
}

void kernel_main() {
    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_logits = get_compile_time_arg_val(1);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t kb = get_compile_time_arg_val(3);
    compute_kernel_hw_startup(cb_part, cb_logits);
    cb_wait_front(cb_part, n_tiles * kb);
    cb_reserve_back(cb_logits, n_tiles);
    for (uint32_t n = 0; n < n_tiles; ++n) {
        tile_regs_acquire();
        copy_init(cb_part);
        copy_tile(cb_part, n * kb, 0);
        for (uint32_t j = 1; j < kb; ++j) {
            copy_init(cb_part);
            copy_tile(cb_part, n * kb + j, 1);
            add_binary_tile_init();
            add_row0(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_logits);
        tile_regs_release();
    }
    cb_push_back(cb_logits, n_tiles);
}
