// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Sum of the active experts' down outputs: DST 0 = first tile, then DST 0 += each next tile (SFPU add from DST 1);
// no active expert -> a zero tile (0 * first-free tile is avoided: DST 0 is zeroed by a self-subtract of a copy).

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t cb_x = 0, cb_meta = 1, cb_out = 16;
    compute_kernel_hw_startup(cb_x, cb_out);
    cb_wait_front(cb_meta, 1);
    const uint32_t na = read_tile_value(cb_meta, 0, 0);
    cb_pop_front(cb_meta, 1);
    tile_regs_acquire();
    copy_tile_init(cb_x);
    for (uint32_t i = 0; i < na; ++i) {
        cb_wait_front(cb_x, 1);
        copy_tile(cb_x, 0, i == 0 ? 0 : 1);
        cb_pop_front(cb_x, 1);
        if (i > 0) {
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            copy_tile_init(cb_x);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
}
