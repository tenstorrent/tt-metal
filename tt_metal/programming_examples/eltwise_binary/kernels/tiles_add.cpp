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
        auto tile0 = read_tile(in0, i);
        auto tile1 = read_tile(in1, i);

        tile_regs_acquire();
        constexpr uint32_t dst_idx = 0;
        add_tiles(tile0, tile1, dst_idx);
        tile_regs_commit();
        tile_regs_wait();

        write_tile(dst_idx, out, i);

        tile_regs_release();
    }
}
