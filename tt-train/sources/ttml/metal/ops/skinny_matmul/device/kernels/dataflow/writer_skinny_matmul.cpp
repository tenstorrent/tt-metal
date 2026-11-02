// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for both skinny_matmul phases.
//   phase A: per row of the item, Nt fp32 partial tiles to pages r*S*Nt + s*Nt + n
//   phase B (REDUCE): one bf16 output tile to page r*Nt + n, and with a tail target the same tile to tail page
//            r*tail_stride + tail_col + n plus zero tiles for the tail columns past Nt (iteration 24);
//            TRANSPOSE_OUT (iteration 40): the tile is (n, r) of [N, M]: page n*rows + r, the tail page
//            (tail_col + n)*tail_stride + r (tail_col a tile row of the target), the zero tiles the rows past Nt

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/skinny_matmul/device/kernels/skinny_matmul_cbs.hpp"

namespace cb = skinny_matmul_cb;

constexpr uint32_t rows = get_compile_time_arg_val(0);
constexpr uint32_t Kt = get_compile_time_arg_val(1);
constexpr uint32_t Nt = get_compile_time_arg_val(2);
constexpr uint32_t R = get_compile_time_arg_val(3);
constexpr uint32_t S = get_compile_time_arg_val(4);
constexpr uint32_t St = get_compile_time_arg_val(5);
constexpr uint32_t tail_on = get_compile_time_arg_val(7);
constexpr uint32_t tail_stride = get_compile_time_arg_val(8);
constexpr uint32_t tail_col = get_compile_time_arg_val(9);
constexpr uint32_t tail_tiles = get_compile_time_arg_val(10);

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tail_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    constexpr auto out_args = TensorAccessorArgs<11>();
    constexpr auto tail_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto out_acc = TensorAccessor(out_args, out_addr);
    [[maybe_unused]] const auto tail_acc = TensorAccessor(tail_args, tail_addr);

#ifdef REDUCE
    const uint32_t tile_bytes = get_tile_size(cb::out);
    uint32_t zero_addr = 0;
    if constexpr (tail_on) {
        zero_addr = get_write_ptr(cb::zero);
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_addr);
        for (uint32_t i = 0; i < tile_bytes / 4U; ++i) {
            z[i] = 0U;
        }
    }
    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;  // r*Nt + n
        const uint32_t r = work / Nt;
        const uint32_t n = work - r * Nt;
        cb_wait_front(cb::out, 1);
        const uint32_t l1 = get_read_ptr(cb::out);
#ifdef TRANSPOSE_OUT
        noc_async_write_page(n * rows + r, out_acc, l1);
#else
        noc_async_write_page(work, out_acc, l1);
#endif
        if constexpr (tail_on) {
#ifdef TRANSPOSE_OUT
            noc_async_write_page((tail_col + n) * tail_stride + r, tail_acc, l1);
            if (n + 1U == Nt) {
                for (uint32_t m = Nt; m < tail_tiles; ++m) {
                    noc_async_write_page((tail_col + m) * tail_stride + r, tail_acc, zero_addr);
                }
            }
#else
            noc_async_write_page(r * tail_stride + tail_col + n, tail_acc, l1);
            if (n + 1U == Nt) {
                for (uint32_t m = Nt; m < tail_tiles; ++m) {
                    noc_async_write_page(r * tail_stride + tail_col + m, tail_acc, zero_addr);
                }
            }
#endif
        }
        noc_async_write_barrier();
        cb_pop_front(cb::out, 1);
    }
#else
    const uint32_t tile_bytes = get_tile_size(cb::partial_out);
    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t g = work / S;
        const uint32_t s = work - g * S;
        const uint32_t r0 = g * R;
        const uint32_t nr = (r0 + R <= rows) ? R : (rows - r0);
        for (uint32_t r = r0; r < r0 + nr; ++r) {
            cb_wait_front(cb::partial_out, Nt);
            uint32_t l1 = get_read_ptr(cb::partial_out);
            for (uint32_t n = 0; n < Nt; ++n) {
                noc_async_write_page(r * S * Nt + s * Nt + n, out_acc, l1);
                l1 += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb::partial_out, Nt);
        }
    }
#endif
}
