// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KV-backward head reduce, writer: output tile p goes to grad_K page p (p < k_tiles) or grad_V page p - k_tiles.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = tt::CBIndex::c_1;

constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
constexpr auto gk_args = TensorAccessorArgs<1>();
constexpr auto gv_args = TensorAccessorArgs<gk_args.next_compile_time_args_offset()>();

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t gk_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gv_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t start = get_arg_val<uint32_t>(arg++);
    const uint32_t count = get_arg_val<uint32_t>(arg++);

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto gk_acc = TensorAccessor(gk_args, gk_addr, tile_bytes);
    const auto gv_acc = TensorAccessor(gv_args, gv_addr, tile_bytes);

    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t p = start + i;
        cb_wait_front(cb_out, 1);
        if (p < k_tiles) {
            noc_async_write_page(p, gk_acc, get_read_ptr(cb_out));
        } else {
            noc_async_write_page(p - k_tiles, gv_acc, get_read_ptr(cb_out));
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
