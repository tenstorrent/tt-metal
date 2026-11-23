// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/compute/eltwise_unary/dropout.h"
#include "api/dataflow/circular_buffer.h"

// Runtime args: num_tiles, start_tile, seed, param0, param1. Tile i of this core is tile start_tile + i of
// the stream. With KEYED_DROPOUT each input tile from c_0 goes through dropout_keyed_tile (param0 is the
// probability, param1 the scale); otherwise each output tile is rand_keyed_tile (param0 is from, param1 scale).
void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t seed = get_arg_val<uint32_t>(2);
    const uint32_t param0 = get_arg_val<uint32_t>(3);
    const uint32_t param1 = get_arg_val<uint32_t>(4);

    CircularBuffer cb_in(tt::CBIndex::c_0);
    CircularBuffer cb_out(tt::CBIndex::c_16);

#ifdef KEYED_DROPOUT
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_init(tt::CBIndex::c_0);
    dropout_keyed_tile_init();
#else
    compute_kernel_hw_startup(tt::CBIndex::c_16, tt::CBIndex::c_16);
    rand_keyed_tile_init();
#endif

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_out.reserve_back(1);
        tile_regs_acquire();
#ifdef KEYED_DROPOUT
        cb_in.wait_front(1);
        copy_tile(tt::CBIndex::c_0, 0, 0);
        dropout_keyed_tile(0, seed, start_tile + i, param0, param1);
#else
        rand_keyed_tile(0, seed, start_tile + i, param0, param1);
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);
        tile_regs_release();
#ifdef KEYED_DROPOUT
        cb_in.pop_front(1);
#endif
        cb_out.push_back(1);
    }
}
