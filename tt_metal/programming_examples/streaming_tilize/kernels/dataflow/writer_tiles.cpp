// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Drains cb_out ONE TILE AT A TIME and writes each tile to its DRAM page.
// This is the consumer that proves the small double-buffered output CB: it
// pops a single tile per iteration, so the whole tile-row is never resident
// in cb_out. Works identically for the atomic producer (pushes W at once) and
// the streaming producer (pushes 1 at a time).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const InterleavedAddrGenFast<true> d = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = DataFormat::Float16_b,
    };

    for (uint32_t k = 0; k < W; ++k) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out);
        noc_async_write_page(k, d, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
