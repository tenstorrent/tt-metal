// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// skinny_contract reader: small^T's Kt tiles once per core; per item (cols adjacent output columns n0..) the K
// rows big[t, n0 .. n0 + ncols), t < Kt, one row of `cols` tiles per push (a shorter last item still pushes cols).

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_small = 0;
constexpr uint32_t cb_big = 1;
constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t Nt = get_compile_time_arg_val(1);
constexpr uint32_t cols = get_compile_time_arg_val(2);
constexpr uint32_t prefetch = get_compile_time_arg_val(3);  // K rows per read batch
constexpr auto small_args = TensorAccessorArgs<4>();
constexpr auto big_args = TensorAccessorArgs<small_args.next_compile_time_args_offset()>();

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t small_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t big_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);
    const uint32_t tile_bytes = get_tile_size(cb_big);
    const auto small_acc = TensorAccessor(small_args, small_addr, tile_bytes);
    const auto big_acc = TensorAccessor(big_args, big_addr, tile_bytes);

    cb_reserve_back(cb_small, Kt);
    {
        uint32_t l1 = get_write_ptr(cb_small);
        for (uint32_t t = 0; t < Kt; ++t) {
            noc_async_read_page(t, small_acc, l1);
            l1 += tile_bytes;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_small, Kt);

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t n0 = (work_start + item) * cols;
        const uint32_t ncols = (n0 + cols <= Nt) ? cols : (Nt - n0);
        // `prefetch` K rows per reserve / barrier / push (the compute pops one row of `cols` tiles at a time; the
        // CB holds two batches, so the pops never straddle its end). A shorter last batch still pushes a full one.
        for (uint32_t t0 = 0; t0 < Kt; t0 += prefetch) {
            const uint32_t rows = (t0 + prefetch <= Kt) ? prefetch : (Kt - t0);
            cb_reserve_back(cb_big, prefetch * cols);
            uint32_t l1 = get_write_ptr(cb_big);
            for (uint32_t t = t0; t < t0 + rows; ++t) {
                for (uint32_t j = 0; j < ncols; ++j) {
#ifdef BIG_TRANSPOSED
                    noc_async_read_page((n0 + j) * Kt + t, big_acc, l1 + j * tile_bytes);
#else
                    noc_async_read_page(t * Nt + n0 + j, big_acc, l1 + j * tile_bytes);
#endif
                }
                l1 += cols * tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_big, prefetch * cols);
        }
    }
}
