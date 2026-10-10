// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

// Preserve the native op's four ordered multiplies, BF16 partial packs,
// DEST reuse/add and precise SiLU. Tile rows represent independent users.
void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 1, 4);
    silu_tile_init();
    for (uint32_t item = 0; item < count; ++item) {
        cb_wait_front(2, 4);
        for (uint32_t tap = 0; tap < 4; ++tap) {
            compute_kernel_lib::tilize<1, 0, 1>(1);
            cb_wait_front(1, 1);
            const uint32_t destination = tap == 3 ? 4 : 3;
            if (tap != 0) {
                cb_wait_front(3, 1);
            }
            cb_reserve_back(destination, 1);
            reconfig_data_format_srca(1);
            reconfig_data_format_srcb(2);
            mul_bcast_rows_init(1, 2);
            tile_regs_acquire();
            mul_tiles_bcast_rows(1, 2, 0, tap, 0);
            if (tap != 0) {
                reconfig_data_format_srca(3);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(3);
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(3, 0, 0);
                reconfig_data_format_srca(1);
            }
            if (tap == 3) {
                silu_tile(0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, destination);
            cb_push_back(destination, 1);
            if (tap != 0) {
                cb_pop_front(3, 1);
            }
            tile_regs_release();
            cb_pop_front(1, 1);
        }
        cb_pop_front(2, 4);
    }
}
