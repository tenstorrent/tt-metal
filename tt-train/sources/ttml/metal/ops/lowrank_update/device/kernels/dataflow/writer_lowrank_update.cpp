// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// lowrank_update writer: one bf16 output tile per item to page image*Mt*Nt + r*Nt + n.

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/lowrank_update/device/kernels/lowrank_update_cbs.hpp"

namespace cb = lowrank_update_cb;

constexpr uint32_t Mt = get_compile_time_arg_val(0);
constexpr uint32_t Nt = get_compile_time_arg_val(1);  // output columns (w's N) in tiles: the item space
constexpr uint32_t Na = get_compile_time_arg_val(2);  // out row stride in tiles (>= Nt)
constexpr auto out_args = TensorAccessorArgs<3>();

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    const uint32_t tile_bytes = get_tile_size(cb::out);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);
    constexpr uint32_t per_image = Mt * Nt;
    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t image = work / per_image;
        const uint32_t rem = work - image * per_image;
        const uint32_t n = rem / Mt;
        const uint32_t r = rem - n * Mt;
        cb_wait_front(cb::out, 1);
        noc_async_write_page(image * Mt * Na + r * Na + n, out_acc, get_read_ptr(cb::out));
        noc_async_write_barrier();
        cb_pop_front(cb::out, 1);
    }
}
