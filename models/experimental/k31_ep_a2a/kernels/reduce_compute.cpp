// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Combine reduction, compute: out = sum of N pseudo-tiles (fp32 DST accumulation), packed to bf16.
#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t in_cb = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb = get_compile_time_arg_val(2);
    const uint32_t units = get_arg_val<uint32_t>(0);
    if (units == 0) {
        return;
    }
    compute_kernel_hw_startup(in_cb, in_cb, out_cb);
    for (uint32_t u = 0; u < units; ++u) {
        cb_wait_front(in_cb, N);
        tile_regs_acquire();
        add_init(in_cb, in_cb);
        add_tiles(in_cb, in_cb, 0, 1, 0);
        if constexpr (N > 2) {
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(in_cb);
            for (uint32_t i = 2; i < N; ++i) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(in_cb, i, 0);
            }
        }
        tile_regs_commit();
        cb_pop_front(in_cb, N);
        cb_reserve_back(out_cb, 1);
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        cb_push_back(out_cb, 1);
    }
}
