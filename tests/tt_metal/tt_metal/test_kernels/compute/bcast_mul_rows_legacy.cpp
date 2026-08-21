// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

// Legacy (CB-id) binary broadcast-MUL (ROW) kernel, classic circular buffers: c_0 = A (data), c_1 = B (bcast
// tile), c_16 = C. Per tile: C = A * broadcast_row(B), pack -> c_16. Regression baseline for the id-free
// variant bcast_mul_rows_2_0.cpp (which differs ONLY in the bcast init/op call; hw_startup / pack_tile are the
// legacy CB-id API in BOTH). Uses the generic bcast_init / any_tiles_bcast so both kernels share the same
// MATH_FIDELITY and produce bit-identical output.
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    CircularBuffer c0(cb_a);
    CircularBuffer c1(cb_b);
    CircularBuffer c16(cb_out);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::ROW>(cb_a, cb_b);

    for (std::uint32_t t = 0; t < per_core_tile_cnt; ++t) {
        c0.wait_front(1);
        c1.wait_front(1);
        c16.reserve_back(1);

        tile_regs_acquire();
        any_tiles_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::ROW>(cb_a, cb_b, 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        c0.pop_front(1);
        c1.pop_front(1);
        c16.push_back(1);
    }
}
