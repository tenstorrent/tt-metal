// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    uint32_t x = get_arg_val<uint32_t>(0);

    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    const uint32_t Nt = get_compile_time_arg_val(2);

    constexpr auto cb_in_a = tt::CBIndex::c_0;
    constexpr auto cb_in_b = tt::CBIndex::c_1;
    constexpr auto cb_in_c = tt::CBIndex::c_2;

    constexpr auto cb_out = tt::CBIndex::c_16;

    uint32_t dbg_tile = 42;

    constexpr uint32_t dst_reg = 0;

    mm_init(cb_in_a, cb_in_b, cb_out);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t nt = 0; nt < Nt; nt++) {
            DPRINT << "COMPUTE - TILE " << mt * Nt + nt << " START" << ENDL();
            tile_regs_acquire();

            for (uint32_t kt = 0; kt < Kt; kt++) {
                cb_wait_front(cb_in_a, 1);
                cb_wait_front(cb_in_b, 1);
                // SliceRange sr = { .h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1 };
                // DPRINT << "A[0,0]=" << TileSlice(cb_in_a, 0, sr, true, false) << ENDL();
                // DPRINT << "B[0,0]=" << TileSlice(cb_in_b, 0, sr, true, false) << ENDL();
                matmul_tiles(cb_in_a, cb_in_b, 0, 0, 0);
                cb_pop_front(cb_in_a, 1);
                cb_pop_front(cb_in_b, 1);
            }

            cb_wait_front(cb_in_c, 1);
            binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in_c);
            binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in_c, 0, 0);
            cb_pop_front(cb_in_c, 1);

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);

            tile_regs_release();

            // Re-init matmul for next iteration (needed after binary op changed config)
            mm_init_short(cb_in_a, cb_in_b);

            DPRINT << "COMPUTE - TILE " << mt * Nt + nt << " END" << ENDL();
        }
    }
}
