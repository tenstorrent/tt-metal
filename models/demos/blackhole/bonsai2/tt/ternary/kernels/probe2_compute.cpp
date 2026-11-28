// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Probe: where in a 32x32 DEST tile do (MODE 0) a copied [1,32] tile and (MODE 1) a custom_mm output of a [1,32] in0
// tile land? Packs DEST tile 0 out as a full 32x32 tile.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = tt::CBIndex::c_0;     // [1,32] tiles (bf16)
    constexpr uint32_t cb_w = tt::CBIndex::c_1;     // 32x32 bf16 weight tiles
    constexpr uint32_t cb_out = tt::CBIndex::c_16;  // 32x32 fp32
    if constexpr (MODE == 0) {
        compute_kernel_hw_startup(cb_x, cb_out);
        copy_init(cb_x);
        cb_wait_front(cb_x, 1);
        tile_regs_acquire();
        copy_tile(cb_x, 0, 0);
    } else {
        custom_mm_block_init<false, CustomMmDestLayout::Default>(cb_x, cb_w, cb_out);
        cb_wait_front(cb_x, 1);
        cb_wait_front(cb_w, 1);
        tile_regs_acquire();
        custom_mm_block<CustomMmDestLayout::Default, false>(cb_x, cb_w, 0, 0, 0, 1);
        custom_mm_block_uninit<CustomMmDestLayout::Default>(cb_out);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_init(cb_out);
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
