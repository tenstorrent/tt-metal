// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// widen_partial compute (fp32 dest): each bf16 tile through the dest and packed fp32 (exact widening, as typecast).
// CBs: 0 in (bf16), 16 out (fp32).  Runtime arg 0: tiles.

#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const uint32_t tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = 0, cb_out = 16;
    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    reconfig_data_format(cb_in, cb_in);
    pack_reconfig_data_format(cb_out);
    copy_init(cb_in);
    CircularBuffer(cb_in).wait_front(tiles);
    CircularBuffer(cb_out).reserve_back(tiles);
    for (uint32_t t = 0; t < tiles; ++t) {
        tile_regs_acquire();
        copy_tile(cb_in, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    CircularBuffer(cb_out).push_back(tiles);
    CircularBuffer(cb_in).pop_front(tiles);
}
