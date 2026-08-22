// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode conv compute: per tile-column, conv = silu(sum_j tap_j * x_j) with the
// four FIR inputs [old st1, old st2, old st3, current qkv] (the post-shift window).
// Taps are per-channel row vectors (tile row 0) broadcast down the batch rows; the
// four products land in dest tiles 0..3 and are summed dest-to-dest on SFPU so no
// partial ever round-trips through a CB.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_x = 0;
constexpr uint32_t cb_taps = 1;
constexpr uint32_t cb_out = 3;

void kernel_main() {
    const uint32_t wi_count = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_x, cb_taps, cb_out);
    pack_reconfig_data_format(cb_out);

    for (uint32_t wi = 0; wi < wi_count; wi++) {
        cb_wait_front(cb_x, 4);
        cb_wait_front(cb_taps, 4);
        cb_reserve_back(cb_out, 1);

        // Re-init per column: the SFPU inits below reprogram shared math state, so the
        // FPU bcast MOP cannot be assumed to survive across iterations.
        reconfig_data_format(cb_x, cb_taps);
        mul_bcast_rows_init(cb_x, cb_taps);

        tile_regs_acquire();
        for (uint32_t j = 0; j < 4; j++) {
            mul_tiles_bcast_rows(cb_x, cb_taps, j, j, j, 0);
        }
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        add_binary_tile(0, 2, 0);
        add_binary_tile(0, 3, 0);
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_x, 4);
        cb_pop_front(cb_taps, 4);
    }
}
