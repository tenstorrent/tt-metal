// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#ifndef QB2_ENTRY
#define QB2_ENTRY kernel_main
#endif
// Fixed QB2 specialization of the native minimal-direct fold. Pack/reload the
// BF16 collective result before residual addition to preserve the traced path.
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

void reduce_add_phase() {
    compute_kernel_hw_startup(1, 1, 17);
    const auto* generation_read = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(2));
    invalidate_l1_cache();
    const uint32_t invocation = *generation_read;
    for (uint32_t chunk = 0; chunk < 2; ++chunk) {
        cb_wait_front(1, 32);
        cb_reserve_back(17, 8);
        add_tiles_init(1, 1, true);
        tile_regs_acquire();
        for (uint32_t block = 0; block < 4; block += 2) {
            for (uint32_t t = 0; t < 8; ++t) {
                add_tiles(1, 1, block * 8 + t, (block + 1) * 8 + t, t);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < 8; ++t) { pack_tile(t, 17, t); }
        tile_regs_release();
        cb_pop_front(1, 32);
        cb_push_back(17, 8);
        {
            DeviceZoneScopedN("MLP-RESIDUAL-ADD");
            cb_wait_front(17, 8);
            cb_wait_front(2, 8);
            cb_reserve_back(16, 8);
            add_tiles_init(17, 2, false);
            tile_regs_acquire();
            for (uint32_t t = 0; t < 8; ++t) { add_tiles(17, 2, t, t, t); }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < 8; ++t) { pack_tile(t, 16, t); }
            tile_regs_release();
            cb_pop_front(17, 8);
            cb_pop_front(2, 8);
            cb_push_back(16, 8);
        }
    }
    auto* generation = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(2));
    PACK((*generation = invocation + 1));
}

void QB2_ENTRY() {
    reduce_add_phase();
#ifdef FUSE_OUTPUT
    reduce_add_phase();
#endif
}
