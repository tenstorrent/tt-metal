// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Decode RMSNorm + rotation, reader (NCRISC). Core b owns 1024-block b = 32 rows, row jh = row 0 of activation tile
// 32 b + jh (64-B reads of face rows 0-1 into scratch). The norm's sum of squares: the compute makes this block's
// partial (a tile with it in every element); this RISC writes it into slot b of every core's c_19 and bumps their
// semaphore 0, then waits for the other B - 1 partials (gathering all B blocks on every core took ~9 us).
// Constants: H32, H32 / 32, OM (rows 0-15 of ones), W_b, WD_b, ONES (fp32 tiles).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "core_span.hpp"
#ifndef NR_CB_X  // CB indices (the fused form inside a ternary decode linear moves them off the linear's)
#define NR_CB_X 0
#define NR_CB_C 2
#define NR_CB_SC 3
#define NR_CB_Y 16
#endif
#ifndef NR_FIRST  // the fused form's cores start after the linear's
#define NR_FIRST 0
#endif

void kernel_main() {
    // common args: x, consts, grid_x, B, then the B cores' NOC x, y
    const uint32_t B = get_common_arg_val<uint32_t>(3);
    const uint32_t b = core_index(get_common_arg_val<uint32_t>(2)) - NR_FIRST;
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto c_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto cs = TensorAccessor(c_args, get_common_arg_val<uint32_t>(1), 4096);
    constexpr uint32_t cb_x = NR_CB_X, cb_c = NR_CB_C, cb_sc = NR_CB_SC;

    cb_reserve_back(cb_c, 6);
    const uint32_t C = get_write_ptr(cb_c);
    noc_async_read(cs.get_noc_addr(0), C, 4096);                     // H32
    noc_async_read(cs.get_noc_addr(1), C + 4096, 4096);              // H32 / 32
    noc_async_read(cs.get_noc_addr(2), C + 2 * 4096, 4096);          // OM: rows 0-15 ones (sums columns 0-15)
    noc_async_read(cs.get_noc_addr(3 + b), C + 3 * 4096, 4096);      // W_b
    noc_async_read(cs.get_noc_addr(3 + B + b), C + 4 * 4096, 4096);  // WD_b = W_b * D_b
    noc_async_read(cs.get_noc_addr(3 + 2 * B), C + 5 * 4096, 4096);  // ONES
    // This block's rows into scratch, as two 32 x 32 row-major tiles: tile 0 row jh = face 0 rows 0-1 of activation
    // tile 32 b + jh, tile 1 row jh = its face 1 rows 0-1 (64-B reads: Blackhole DRAM reads must be 64-B aligned).
    // Columns 0-15 hold face row 0 (the token); the compute's sum of squares counts only those.
    cb_reserve_back(cb_sc, 2);
    const uint32_t sc = get_write_ptr(cb_sc);
    for (uint32_t jh = 0; jh < 32; ++jh) {
        const uint64_t t = xs.get_noc_addr(32 * b + jh);
        noc_async_read(t, sc + jh * 64, 64);
        noc_async_read(t + 512, sc + 2048 + jh * 64, 64);
    }
    noc_async_read_barrier();
    cb_reserve_back(cb_x, 1);
    {
        // 32-B local NoC copies (a RISC word loop took ~5 us per 2 KB)
        const uint32_t dst = get_write_ptr(cb_x);
        for (uint32_t jh = 0; jh < 32; ++jh) {
            noc_async_read(get_noc_addr(sc + jh * 64), dst + jh * 64, 32);
            noc_async_read(get_noc_addr(sc + 2048 + jh * 64), dst + jh * 64 + 32, 32);
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb_x, 1);
    cb_push_back(cb_sc, 2);
    cb_push_back(cb_c, 6);

    // the partials: ours into slot b of every core's c_19 (the same L1 address on every core), then the others'
    constexpr uint32_t cb_p = 18, cb_peer = 19;
    cb_reserve_back(cb_peer, B);
    const uint32_t pb = get_write_ptr(cb_peer) + b * 4096;
    cb_wait_front(cb_p, 1);
    const uint32_t p = get_read_ptr(cb_p);
    const uint32_t sem = get_semaphore(0);
    for (uint32_t q = 0; q < B; ++q) {
        const uint32_t qx = get_common_arg_val<uint32_t>(4 + 2 * q), qy = get_common_arg_val<uint32_t>(5 + 2 * q);
        if (q == b) {
            noc_async_read(get_noc_addr(p), pb, 4096);
        } else {
            noc_async_write(p, get_noc_addr(qx, qy, pb), 4096);
        }
    }
    noc_async_read_barrier();
    noc_async_write_barrier();
    for (uint32_t q = 0; q < B; ++q) {
        if (q != b) {
            noc_semaphore_inc(
                get_noc_addr(get_common_arg_val<uint32_t>(4 + 2 * q), get_common_arg_val<uint32_t>(5 + 2 * q), sem), 1);
        }
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), B - 1);
    cb_push_back(cb_peer, B);
    cb_pop_front(cb_p, 1);
    noc_async_atomic_barrier();
}
