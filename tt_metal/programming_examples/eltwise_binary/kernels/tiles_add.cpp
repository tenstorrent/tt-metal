// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

#include "universal_common.h"

KERNEL_MAIN {
    INIT_ARGUMENTS

    binary_op_init_common(in0_cb, in1_cb, out_cb);
    add_tiles_init(in0_cb, in1_cb);

    for (uint32_t i = 0; i < n_tiles; i++) {
        read_tile(in0_cb, i, in0, in0_page_size_bytes);
        read_tile(in1_cb, i, in1, in1_page_size_bytes);

        tile_regs_acquire();
        add_tiles(in0_cb, in1_cb, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();

        write_packed_tile(0, out_cb, i, out, out_page_size_bytes);

        release_write_tiles(out_cb, 1);
        release_read_tiles(in0_cb, 1);
        release_read_tiles(in1_cb, 1);
        tile_regs_release();
    }
}
