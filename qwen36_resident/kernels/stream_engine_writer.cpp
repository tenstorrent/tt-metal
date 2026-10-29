// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Weight-streaming engine writer (BRISC): drains each matmul's output and models the serial
// activation chain. After matmul e finishes it waits delay_cycles[e] (the stand-in for norm / CCL /
// recurrence work between projections) and only then releases compute into the next matmul. The
// weight reader is not gated, so it keeps streaming during these gaps. The last layer of the last
// iteration is copied to the output buffer for a correctness check.
//
// Runtime args: 0 out_l1_addr, then per entry e: 1 + 3e n_tiles, 2 + 3e out tile offset, 3 + 3e delay_cycles

#include "api/dataflow/dataflow_api.h"
#include "stream_engine_common.hpp"

using namespace stream_engine;

namespace {

FORCE_INLINE uint64_t wall_clock() {
    volatile uint tt_reg_ptr* lo = reinterpret_cast<volatile uint tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile uint tt_reg_ptr* hi = reinterpret_cast<volatile uint tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    const uint32_t l = lo[0];
    return l | (static_cast<uint64_t>(hi[0]) << 32);
}

}  // namespace

void kernel_main() {
    const uint32_t out_l1 = get_arg_val<uint32_t>(0);
    constexpr uint32_t out_tile_bytes = 64;  // 1x32 bf16
    const uint32_t total = iters * layers * kEntries;
    uint32_t step = 0;
    for (uint32_t it = 0; it < iters; it++) {
        for (uint32_t l = 0; l < layers; l++) {
            const bool keep = (it == iters - 1) && (l == layers - 1);
            for (uint32_t e = 0; e < kEntries; e++) {
                const uint32_t n_tiles = get_arg_val<uint32_t>(1 + 3 * e);
                const uint32_t out_off = get_arg_val<uint32_t>(2 + 3 * e);
                const uint32_t delay = get_arg_val<uint32_t>(3 + 3 * e);
                for (uint32_t t = 0; t < n_tiles; t++) {
                    cb_wait_front(cb_out, 1);
                    if (keep) {
                        volatile tt_l1_ptr uint32_t* s =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_out));
                        volatile tt_l1_ptr uint32_t* d =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1 + (out_off + t) * out_tile_bytes);
                        for (uint32_t w = 0; w < out_tile_bytes / 4; w++) {
                            d[w] = s[w];
                        }
                    }
                    cb_pop_front(cb_out, 1);
                }
                if (++step < total) {
                    if (delay > 0) {
                        const uint64_t until = wall_clock() + delay;
                        while (wall_clock() < until) {
                        }
                    }
                    cb_reserve_back(cb_gate, 1);
                    cb_push_back(cb_gate, 1);
                }
            }
        }
    }
}
