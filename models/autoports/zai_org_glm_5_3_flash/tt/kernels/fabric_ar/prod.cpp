// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 14 fabric all-reduce, producer core (NCRISC) of the standalone phase (tt/fabric_ar.py): the core that holds
// part of the partial [B, 4096] in L1 (8 tiles of a width-sharded tensor; in the layer megakernel: the o_proj / down
// DS-reader core). Per iteration: write the VALID ROWS of its tiles, packed (B <= 16: face-0 rows then face-1 rows,
// 64 B per row per tile; B > 16: the whole tile), into slot p1 of its lane's round-1 landing buffer, +1 OWN_READY on
// the lane; wait RESULT_READY, then read the reduced rows back into its output tiles (padding rows untouched).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "far_common.hpp"

constexpr uint32_t BROWS = get_compile_time_arg_val(0);  // valid rows (<= 16), or 32 = whole tiles
constexpr uint32_t TILES = get_compile_time_arg_val(1);  // tiles held by this producer
constexpr uint32_t TB = 2048;
constexpr uint32_t CH = BROWS * 32;                // bytes of one face's valid rows (B <= 16)
constexpr uint32_t V = BROWS >= 32 ? TB : 2 * CH;  // packed bytes per tile

void kernel_main() {
    size_t a = 0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t y_addr = get_arg_val<uint32_t>(a++);
    const uint32_t lane_x = get_arg_val<uint32_t>(a++);
    const uint32_t lane_y = get_arg_val<uint32_t>(a++);
    const uint32_t slot = get_arg_val<uint32_t>(a++);  // lane's land1[p1]
    const uint32_t res = get_arg_val<uint32_t>(a++);   // lane's result buffer
    const uint32_t t_base = get_arg_val<uint32_t>(a++);
    const uint32_t lane_sync = get_arg_val<uint32_t>(a++);
    const uint32_t iters = get_arg_val<uint32_t>(a++);
    const uint64_t own_ready = get_noc_addr(lane_x, lane_y, lane_sync + far::OWN_READY * far::WORD);
    volatile tt_l1_ptr uint32_t* gen = far::word(sync, far::GEN_PROD);
    uint32_t g = *gen;
    const uint64_t t_start = get_timestamp();
    for (uint32_t it = 0; it < iters; it++) {
        g++;
        for (uint32_t j = 0; j < TILES; j++) {
            const uint32_t src = x_addr + j * TB;
            const uint64_t dst = get_noc_addr(lane_x, lane_y, slot + (t_base + j) * V);
            if constexpr (BROWS >= 32) {
                noc_async_write(src, dst, TB);
            } else {
                noc_async_write(src, dst, CH);
                noc_async_write(src + 512, dst + CH, CH);
            }
        }
        noc_async_write_barrier();
        noc_semaphore_inc(own_ready, 1);
        noc_async_atomic_barrier();
        if (it == 0) {
            far::probe(sync, far::PROD_PROBE + 0, t_start);
            far::probe(sync, far::PROD_PROBE + 1, get_timestamp());
        }
        far::wait_ge(far::word(sync, far::RESULT_READY), g);
        if (it == 0) {
            far::probe(sync, far::PROD_PROBE + 2, get_timestamp());
        }
        for (uint32_t j = 0; j < TILES; j++) {
            const uint32_t dst = y_addr + j * TB;
            const uint64_t src = get_noc_addr(lane_x, lane_y, res + (t_base + j) * V);
            if constexpr (BROWS >= 32) {
                noc_async_read(src, dst, TB);
            } else {
                noc_async_read(src, dst, CH);
                noc_async_read(src + CH, dst + 512, CH);
            }
        }
        noc_async_read_barrier();
        if (it == 0) {
            far::probe(sync, far::PROD_PROBE + 3, get_timestamp());
        }
    }
    *gen = g;
    far::probe(sync, far::PROD_PROBE + 4, get_timestamp());
}
