// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
namespace NAMESPACE {
void MAIN {
    uint32_t num_porduced_tiles = get_arg_val<uint32_t>(0);
    uint32_t Kt = get_arg_val<uint32_t>(1);

    mm_init(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    // the simplest possible version of outer product blocked matmul
    // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
    for (uint32_t i = 0; i < num_porduced_tiles; ++i) {
        acquire_dst();
        for (uint32_t kt = 0; kt < Kt; kt++) {
            cb_wait_front(tt::CBIndex::c_0, 1);
            cb_wait_front(tt::CBIndex::c_1, 1);

            matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, false);

            cb_pop_front(tt::CBIndex::c_0, 1);
            cb_pop_front(tt::CBIndex::c_1, 1);
        }

        cb_reserve_back(tt::CBIndex::c_16, 1);
        pack_tile(0, tt::CBIndex::c_16);
        cb_push_back(tt::CBIndex::c_16, 1);

        release_dst();
    }
}
}  // namespace NAMESPACE
