// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Decode RMSNorm + rotation, reader (NCRISC). Core b owns 1024-block b; for the norm's sum of squares every core
// gathers all B blocks: block blk = 32 rows, row jh = row 0 of activation tile 32 blk + jh (64-B reads of face rows
// 0-1 into scratch). Constants: H32, H32 / 32, OM (rows 0-15 of ones), W_b, WD_b, ONES (fp32 tiles).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "core_span.hpp"

void kernel_main() {
    // common args: x, consts, grid_x, B
    const uint32_t B = get_common_arg_val<uint32_t>(3);
    const uint32_t b = core_index(get_common_arg_val<uint32_t>(2));
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto c_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto cs = TensorAccessor(c_args, get_common_arg_val<uint32_t>(1), 4096);
    constexpr uint32_t cb_x = 0, cb_c = 2, cb_sc = 3;

    cb_reserve_back(cb_c, 6);
    const uint32_t C = get_write_ptr(cb_c);
    noc_async_read(cs.get_noc_addr(0), C, 4096);                     // H32
    noc_async_read(cs.get_noc_addr(1), C + 4096, 4096);              // H32 / 32
    noc_async_read(cs.get_noc_addr(2), C + 2 * 4096, 4096);          // OM: rows 0-15 ones (sums columns 0-15)
    noc_async_read(cs.get_noc_addr(3 + b), C + 3 * 4096, 4096);      // W_b
    noc_async_read(cs.get_noc_addr(3 + B + b), C + 4 * 4096, 4096);  // WD_b = W_b * D_b
    noc_async_read(cs.get_noc_addr(3 + 2 * B), C + 5 * 4096, 4096);  // ONES
    // All blocks' rows into scratch with one barrier, as two 32 x 32 row-major tiles per block: tile 2 blk row jh =
    // face 0 rows 0-1 of activation tile 32 blk + jh, tile 2 blk + 1 row jh = its face 1 rows 0-1 (64-B reads:
    // Blackhole DRAM reads must be 64-B aligned). Columns 0-15 hold face row 0 (the token); the compute's sum of
    // squares counts only those. This core's own block is also rearranged into X[jh, jl] = x[1024 b + 32 jh + jl].
    cb_reserve_back(cb_sc, 2 * B);
    const uint32_t sc = get_write_ptr(cb_sc);
    for (uint32_t blk = 0; blk < B; ++blk) {
        for (uint32_t jh = 0; jh < 32; ++jh) {
            const uint64_t t = xs.get_noc_addr(32 * blk + jh);
            noc_async_read(t, sc + (2 * blk) * 2048 + jh * 64, 64);
            noc_async_read(t + 512, sc + (2 * blk + 1) * 2048 + jh * 64, 64);
        }
    }
    noc_async_read_barrier();
    cb_reserve_back(cb_x, 1);
    {
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_x));
        const volatile tt_l1_ptr uint32_t* f0 =
            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(sc + (2 * b) * 2048);
        const volatile tt_l1_ptr uint32_t* f1 =
            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(sc + (2 * b + 1) * 2048);
        for (uint32_t jh = 0; jh < 32; ++jh) {
            for (uint32_t w = 0; w < 8; ++w) {
                dst[jh * 16 + w] = f0[jh * 16 + w];
                dst[jh * 16 + 8 + w] = f1[jh * 16 + w];
            }
        }
    }
    cb_push_back(cb_x, 1);
    cb_push_back(cb_sc, 2 * B);
    cb_push_back(cb_c, 6);
}
