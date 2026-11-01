// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill dequant, compute. Per unit: 32 code tiles x their group's scale row (row-broadcast
// straight out of the fp32 scale_rows tile via bcast_row_idx), packed to the output format.
// Code tile t of the unit is in group t/4, whose scale is row (G%4)*8 + t/4 of the scale tile.
// Runtime args: u0, nu (same split as the reader).

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"

void kernel_main() {
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nu = get_arg_val<uint32_t>(1);
    constexpr uint32_t ng = get_compile_time_arg_val(0);
    constexpr uint32_t ncols = get_compile_time_arg_val(2);
    constexpr uint32_t dt = get_compile_time_arg_val(1);  // tiles per DST half: 8 (bf16 DST) or 4 (fp32)
    constexpr uint32_t cb_w = 0, cb_s = 1, cb_out = 16;

    compute_kernel_hw_startup(cb_w, cb_s, cb_out);
    mul_bcast_rows_init(cb_w, cb_s);
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t G = u / ncols;
        cb_wait_front(cb_s, 1);
        cb_wait_front(cb_w, 32);
        for (uint32_t q = 0; q < 32 / dt; ++q) {
            tile_regs_acquire();
            for (uint32_t t = 0; t < dt; ++t) {
                const uint32_t i = q * dt + t;
                mul_tiles_bcast_rows(cb_w, cb_s, i, 0, t, (G % 4) * 8 + i / 4);
            }
            tile_regs_commit();
            cb_reserve_back(cb_out, dt);
            tile_regs_wait();
            for (uint32_t t = 0; t < dt; ++t) {
                pack_tile(t, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, dt);
        }
        cb_pop_front(cb_w, 32);
        cb_pop_front(cb_s, 1);
    }
}
