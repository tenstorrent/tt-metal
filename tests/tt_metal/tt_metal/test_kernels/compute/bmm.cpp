// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "debug_print.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
namespace NAMESPACE {
void MAIN {

    constexpr int onetile = 1;

    int dst_tile_index = 0;
    int in0_block_tile_index = 0;

    uint32_t batch = get_compile_time_arg_val(0);
    uint32_t Mt = get_compile_time_arg_val(1);
    uint32_t Kt = get_compile_time_arg_val(2);
    uint32_t Nt = get_compile_time_arg_val(3);
    //DPRINT << "string" << 1 << SETW(4) << F32(2.0f) << ENDL();
    uint32_t debug_cb_id;

    mm_init();

    // the simplest possible version of outer product blocked matmul
    // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
    for (uint32_t nb = 0; nb < batch; nb++)
    for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) // output tile of C
    for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C) // output tile index of C
    {
        acquire_dst(tt::DstMode::Full);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            cb_wait_front(tt::CB::c_in0, onetile);
            cb_wait_front(tt::CB::c_in1, onetile);

            matmul_tiles(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0, false);

            debug_cb_id = tt::CB::c_in0;
            //for (int32_t r = 0; r < 32; ++ r) {
            //    SliceRange sr = SliceRange{.h0 = r, .h1 = r+1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
            //    UNPACK(( DPRINT << (uint)r << " --unpack-- " << TileSlice(debug_cb_id, 0, sr, true, false) << ENDL() ));
            //}

            cb_pop_front(tt::CB::c_in0, onetile);
            cb_pop_front(tt::CB::c_in1, onetile);
        }

        cb_reserve_back(tt::CB::c_out0, onetile);
        pack_tile(0, tt::CB::c_out0);
        debug_cb_id = tt::CB::c_in0;
        for (int32_t r = 0; r < 32; ++ r) {
            SliceRange sr = SliceRange{.h0 = r, .h1 = r+1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
            //PACK(( DPRINT << (uint)r << " --pack-- " << TileSlice(debug_cb_id, 0, sr, true, false) << ENDL() ));
            //UNPACK(( DPRINT << (uint)r << " --unpack-R- " << TileSlice(debug_cb_id, 0, sr, true, false) << ENDL() ));
        }
        cb_push_back(tt::CB::c_out0, onetile);

        release_dst(tt::DstMode::Full);
    }


}
} // NAMESPACE
