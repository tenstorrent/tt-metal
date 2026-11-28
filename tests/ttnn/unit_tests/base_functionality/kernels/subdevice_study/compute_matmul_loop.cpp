// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute-bound loop: iters x inner tile matmuls of the same L1-resident A and B tiles (no NoC or DRAM
// traffic). Packs the final A@B once so the host can check it.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_named_compile_time_arg_val("cb_a");
    constexpr uint32_t cb_b = get_named_compile_time_arg_val("cb_b");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t inner = get_named_compile_time_arg_val("inner");
    const uint32_t iters = get_arg_val<uint32_t>(0);

    CircularBuffer a(cb_a);
    CircularBuffer b(cb_b);
    CircularBuffer out(cb_out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_out);
    matmul_init(cb_a, cb_b);
    a.wait_front(1);
    b.wait_front(1);
    for (uint32_t it = 0; it < iters; ++it) {
        tile_regs_acquire();
        for (uint32_t d = 0; d < inner; ++d) {
            matmul_tiles(cb_a, cb_b, 0, 0, d);
        }
        tile_regs_commit();
        tile_regs_wait();
        if (it + 1 == iters) {
            out.reserve_back(1);
            pack_tile(0, cb_out);
            out.push_back(1);
        }
        tile_regs_release();
    }
    a.pop_front(1);
    b.pop_front(1);
}
