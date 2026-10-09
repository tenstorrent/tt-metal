// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// out = a + b, adapted from binary_ng's kernels/compute/eltwise_binary_no_bcast.cpp. Works in the
// reader's batches so each DST acquire covers a whole batch.
//
// Compile-time args: batch
// Runtime args: num_tiles

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t batch = get_compile_time_arg_val(0);

    constexpr uint32_t cb_a_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_b_id = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_2;

    CircularBuffer cb_a(cb_a_id);
    CircularBuffer cb_b(cb_b_id);
    CircularBuffer cb_out(cb_out_id);

    compute_kernel_hw_startup(cb_a_id, cb_b_id, cb_out_id);
    add_init(cb_a_id, cb_b_id);

    for (uint32_t done = 0; done < num_tiles; done += batch) {
        const uint32_t n = (num_tiles - done) < batch ? (num_tiles - done) : batch;
        cb_a.wait_front(n);
        cb_b.wait_front(n);
        cb_out.reserve_back(n);

        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            add_tiles(cb_a_id, cb_b_id, i, i, i);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_out_id);
        }
        tile_regs_release();

        cb_out.push_back(n);
        cb_a.pop_front(n);
        cb_b.pop_front(n);
    }
}
