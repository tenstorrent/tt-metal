// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// lowrank_update reader. Items are output tiles in column-major order per image: item -> (image, n, r).
// The a column of the image (Mt tiles, page image*Mt + r) is pushed once per image, the w tile of a column
// (page n in both layouts: the other dimension is one tile) once per column, the acc tile per item.

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/lowrank_update/device/kernels/lowrank_update_cbs.hpp"

namespace cb = lowrank_update_cb;

constexpr uint32_t Mt = get_compile_time_arg_val(0);
constexpr uint32_t Nt = get_compile_time_arg_val(1);
constexpr auto acc_args = TensorAccessorArgs<2>();
constexpr auto a_args = TensorAccessorArgs<acc_args.next_compile_time_args_offset()>();
constexpr auto w_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t acc_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t a_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    const uint32_t tile_bytes = get_tile_size(cb::acc);
    const auto acc_acc = TensorAccessor(acc_args, acc_addr, tile_bytes);
    const auto a_acc = TensorAccessor(a_args, a_addr, tile_bytes);
    const auto w_acc = TensorAccessor(w_args, w_addr, tile_bytes);

    constexpr uint32_t per_image = Mt * Nt;
    uint32_t cur_image = 0xFFFFFFFFU;
    uint32_t cur_col = 0xFFFFFFFFU;
    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t image = work / per_image;
        const uint32_t rem = work - image * per_image;
        const uint32_t n = rem / Mt;
        const uint32_t r = rem - n * Mt;
        if (image != cur_image) {
            cb_reserve_back(cb::a, Mt);
            uint32_t l1 = get_write_ptr(cb::a);
            for (uint32_t i = 0; i < Mt; ++i) {
                noc_async_read_page(image * Mt + i, a_acc, l1);
                l1 += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb::a, Mt);
            cur_image = image;
            cur_col = 0xFFFFFFFFU;
        }
        if (n != cur_col) {
            cb_reserve_back(cb::w, 1);
            noc_async_read_page(n, w_acc, get_write_ptr(cb::w));
            noc_async_read_barrier();
            cb_push_back(cb::w, 1);
            cur_col = n;
        }
        cb_reserve_back(cb::acc, 1);
        noc_async_read_page(image * per_image + r * Nt + n, acc_acc, get_write_ptr(cb::acc));
        noc_async_read_barrier();
        cb_push_back(cb::acc, 1);
    }
}
