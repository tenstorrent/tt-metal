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

    binary_op_init_common(cb_in, cb_in, cb_out);
    add_tiles_init(cb_in, cb_in);

    for (uint32_t i = 0; i < n_tiles; i++) {
        read_tile(i, in0, in0_page_size_bytes);
        read_tile(i, in1, in1_page_size_bytes);

        tile_regs_acquire();
        add_tiles(cb_in, cb_in, 0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();

        write_packed_tile(0, i, out, out_page_size_bytes);

        release_write_tiles(1);
        release_read_tiles(2);
        tile_regs_release();
    }
}
